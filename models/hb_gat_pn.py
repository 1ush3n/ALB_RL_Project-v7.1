import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear, global_mean_pool, global_max_pool

# ---------------------------------------------------------------------------
# 特征嵌入模块 (Feature Embedder)
# 作用: 将原始异构节点特征投影到统一的隐藏层维度
# ---------------------------------------------------------------------------
class FeatureEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 为每种节点类型定义一个 MLP 
        self.task_emb = nn.Sequential(
            nn.Linear(config.task_feat_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        self.worker_emb = nn.Sequential(
            nn.Linear(config.worker_feat_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        self.station_emb = nn.Sequential(
            nn.Linear(config.station_feat_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )

    def forward(self, x_dict):
        """
        x_dict: PyG HeteroData.x_dict 字典
        返回: 嵌入后的字典 (Key -> [N, HiddenDim])
        """
        out = {}
        if 'task' in x_dict:
            out['task'] = self.task_emb(x_dict['task'])
        if 'worker' in x_dict:
            out['worker'] = self.worker_emb(x_dict['worker'])
        if 'station' in x_dict:
            out['station'] = self.station_emb(x_dict['station'])
        return out

# ---------------------------------------------------------------------------
# 异构图注意力编码器 (Hetero GAT Encoder)
# 作用: 通过消息传递捕获节点间的拓扑依赖和资源约束
# ---------------------------------------------------------------------------
class HeteroGATEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for _ in range(config.num_gat_layers):
            conv = HeteroConv({
                # 1. 拓扑流：任务间的优先关系 (Precedence Constraint)
                ('task', 'precedes', 'task'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                
                # 2. 归属流：任务与站位的动态绑定
                ('task', 'assigned_to', 'station'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                ('station', 'has_task', 'task'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                
                # 3. 资源流：工人与任务的能力匹配/执行关系
                ('worker', 'can_do', 'task'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                ('task', 'done_by', 'worker'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                
            }, aggr='sum')
            self.layers.append(conv)
            
    def forward(self, x_dict, edge_index_dict):
        for conv in self.layers:
            x_dict_out = conv(x_dict, edge_index_dict)
            
            # HeteroConv 只返回作为 Edge 终点的节点更新。
            # 必须手动保留未更新的节点（残差连接 + 身份映射）。
            x_dict_new = {k: v for k, v in x_dict.items()}
            
            for key, x in x_dict_out.items():
                x = F.relu(x)
                if key in x_dict:
                    # 残差连接 (Residual Connection)
                    x = x + x_dict[key] 
                x_dict_new[key] = x
            x_dict = x_dict_new
            
        return x_dict

# ---------------------------------------------------------------------------
# 决策一：工序选择 (Task Pointer)
# 机制: 指针网络 (Pointer Network) 从候选集中选择一个工序
# ---------------------------------------------------------------------------
class TaskPointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.context_proj = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.context_proj = nn.Linear(config.hidden_dim * 6, config.hidden_dim) # 扩展为 Mean+Max 拼接后的 6 倍维度
        self.task_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attn = nn.Linear(config.hidden_dim, 1)

    def forward(self, task_emb, global_context, mask=None):
        """
        task_emb: [N, H] 所有任务的 Embedding
        global_context: [B, H] 全局上下文（通常是 Station 的均值池化）
        mask: [B, N] True 表示 Invalid (不可选)
        """
        ctx = self.context_proj(global_context).unsqueeze(1) # [B, 1, H]
        
        if task_emb.dim() == 2:
             task_emb = task_emb.unsqueeze(0) # [1, N, H]
        
        tsk = self.task_proj(task_emb)      
        features = torch.tanh(ctx + tsk) 
        scores = self.attn(features).squeeze(-1) # [B, N]
        
        if mask is not None:
             if mask.dim() == 1: mask = mask.unsqueeze(0)
             # 将无效动作的 Logit 设为负无穷
             scores = scores.masked_fill(mask, -1e9)
            
        return scores 

# ---------------------------------------------------------------------------
# 决策二：站位选择 (Station Selector)
# 机制: 简单的 MLP 评分，输入 (SelectedTask, Station)
# ---------------------------------------------------------------------------
class StationSelector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, selected_task_emb, station_embs, mask=None):
        B, S, H = station_embs.size()
        # 复制 Task Embedding 以便与每个 Station 拼接
        task_repeat = selected_task_emb.unsqueeze(1).expand(-1, S, -1) 
        cat_feat = torch.cat([task_repeat, station_embs], dim=2)
        scores = self.scorer(cat_feat).squeeze(-1) 
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
            
        return scores

# ---------------------------------------------------------------------------
# 决策三：工人选择 (Worker Pointer)
# 机制: 自回归指针网络 (Autoregressive Pointer)
#       循环选择工人，直到选择 "Stop Action" 或无法继续
# ---------------------------------------------------------------------------
class WorkerPointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attn = nn.Linear(config.hidden_dim, 1)
        
        # Stop Head: 预测是否停止选人 [Logit_Continue, Logit_Stop]
        self.stop_head = nn.Linear(config.hidden_dim * 2, 2) 

    def forward_choice(self, task_emb, worker_embs, mask=None):
        """选择下一个工人"""
        query = self.query_proj(task_emb).unsqueeze(1) 
        keys = self.key_proj(worker_embs)
        features = torch.tanh(query + keys)
        scores = self.attn(features).squeeze(-1) 
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        return scores

    def forward_stop(self, task_emb, current_team_emb):
        """决定是否因为人够了/协同成本过高而停止"""
        cat_feat = torch.cat([task_emb, current_team_emb], dim=1)
        logits = self.stop_head(cat_feat) 
        return logits

# ---------------------------------------------------------------------------
# 完整模型: HB-GAT-PN (Heterogeneous Graph Attention Pointer Network)
# ---------------------------------------------------------------------------
class HBGATPN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 嵌入与编码 (Feature Extraction)
        self.embedder = FeatureEmbedder(config)
        self.encoder = HeteroGATEncoder(config)
        
        # 2. 解码器 (Policy Heads)
        self.task_head = TaskPointer(config)
        self.station_head = StationSelector(config)
        self.worker_head = WorkerPointer(config)
        
        # 3. 价值网络 (Critic) 
        # 用于 PPO 的 Advantage 计算
        # self.critic = nn.Sequential(
        #     nn.Linear(config.hidden_dim * 3, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim * 6, 64), # 扩展为 Mean+Max 拼接后的 6 倍维度
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch_data):
        """
        前向传播: 仅用于计算 Encoder 的输出和 Global Context。
        具体的 Action Logits 计算在 Agent 中分步调用各个 Head。
        """
        # --- Step 1: 编码 ---
        x_dict = self.embedder(batch_data.x_dict)
        x_dict_encoded = self.encoder(x_dict, batch_data.edge_index_dict)
        
        # Global Context: 对 Station, Task, Worker 节点进行 Mean + Max Pooling, 实现全视角及瓶颈感知
        if hasattr(batch_data['station'], 'batch') and batch_data['station'].batch is not None:
             # 原代码：
             # station_ctx = global_mean_pool(x_dict_encoded['station'], batch_data['station'].batch)
             # task_ctx = global_mean_pool(x_dict_encoded['task'], batch_data['task'].batch)
             # worker_ctx = global_mean_pool(x_dict_encoded['worker'], batch_data['worker'].batch)
             # global_context = torch.cat([station_ctx, task_ctx, worker_ctx], dim=1) # [B, H*3]
             
             station_mean = global_mean_pool(x_dict_encoded['station'], batch_data['station'].batch)
             task_mean = global_mean_pool(x_dict_encoded['task'], batch_data['task'].batch)
             worker_mean = global_mean_pool(x_dict_encoded['worker'], batch_data['worker'].batch)
             
             station_max = global_max_pool(x_dict_encoded['station'], batch_data['station'].batch)
             task_max = global_max_pool(x_dict_encoded['task'], batch_data['task'].batch)
             worker_max = global_max_pool(x_dict_encoded['worker'], batch_data['worker'].batch)
             
             global_context = torch.cat([station_mean, task_mean, worker_mean, station_max, task_max, worker_max], dim=1) # [B, H*6]
        else:
             # 原代码：
             # station_ctx = torch.mean(x_dict_encoded['station'], dim=0, keepdim=True)
             # task_ctx = torch.mean(x_dict_encoded['task'], dim=0, keepdim=True)
             # worker_ctx = torch.mean(x_dict_encoded['worker'], dim=0, keepdim=True)
             # global_context = torch.cat([station_ctx, task_ctx, worker_ctx], dim=1)
             
             station_mean = torch.mean(x_dict_encoded['station'], dim=0, keepdim=True)
             task_mean = torch.mean(x_dict_encoded['task'], dim=0, keepdim=True)
             worker_mean = torch.mean(x_dict_encoded['worker'], dim=0, keepdim=True)
             
             station_max = torch.max(x_dict_encoded['station'], dim=0, keepdim=True)[0]
             task_max = torch.max(x_dict_encoded['task'], dim=0, keepdim=True)[0]
             worker_max = torch.max(x_dict_encoded['worker'], dim=0, keepdim=True)[0]
             
             global_context = torch.cat([station_mean, task_mean, worker_mean, station_max, task_max, worker_max], dim=1)
             
        return x_dict_encoded, global_context

    def get_value(self, global_context):
        return self.critic(global_context)
