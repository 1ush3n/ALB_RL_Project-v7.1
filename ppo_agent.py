import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import math
from configs import configs
from utils.muon import Muon
class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 智能体。
    负责与 Environment 交互，收集轨迹，并更新 Strategy Network。
    """
    def __init__(self, model, lr, gamma, k_epochs, eps_clip, device, batch_size=4, 
                 lr_warmup_steps=0, min_lr=0, total_timesteps=0):
        self.policy = model.to(device)
        
        # [Optimizer Setup: Muon + AdamW]
        if hasattr(configs, 'use_muon') and configs.use_muon:
            muon_params = []
            adam_params = []
            for name, param in self.policy.named_parameters():
                if param.ndim >= 2:
                    muon_params.append(param)
                else:
                    adam_params.append(param)
                    
            self.optimizer = Muon(muon_params, lr=lr * 0.02, momentum=0.95)
            self.optimizer_adam = torch.optim.AdamW(adam_params, lr=lr)
            self.using_muon = True
        else:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
            self.using_muon = False
            
        self.lr = lr
        self.gamma = gamma          # 折扣因子
        self.k_epochs = k_epochs    # 每次 Update 的迭代轮数
        self.eps_clip = eps_clip    # PPO Clip参数 (e.g., 0.2)
        self.device = device
        self.batch_size = batch_size
        self.accumulation_steps = getattr(configs, 'accumulation_steps', 1)
        self.gae_lambda = getattr(configs, 'gae_lambda', 0.95)
        
        self.MseLoss = nn.MSELoss() # 回归标准 MSE，强制 Critic 网络具有针对大数值误差的抛物线追赶能力
        
        # [LR Scheduler Setup]
        # Linear Warmup + Cosine Annealing
        self.lr_warmup_steps = lr_warmup_steps
        self.min_lr = min_lr
        self.total_timesteps = max(1, total_timesteps) # 防止除零
        self.current_step = 0
        
        # 定义 LambdaLR
        def lr_lambda(current_step):
            # 1. Warmup Phase
            if current_step < self.lr_warmup_steps:
                return float(current_step) / float(max(1, self.lr_warmup_steps))
            
            # 2. SGDR (Cosine Annealing with Warm Restarts) Phase
            step_after_warmup = current_step - self.lr_warmup_steps
            T_0 = getattr(configs, 'sgdr_t0', 40)  # 第一周期步长 (由配置项控制)
            
            curr_cycle_step = step_after_warmup % T_0
            progress = float(curr_cycle_step) / float(T_0)
            
            # 余弦衰减，到底部直接重启
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # Scaling: range [min_lr/lr, 1.0]
            min_ratio = self.min_lr / self.lr
            return min_ratio + (1.0 - min_ratio) * cosine_decay
            
        if self.using_muon:
            self.scheduler_muon = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            self.scheduler_adam = torch.optim.lr_scheduler.LambdaLR(self.optimizer_adam, lr_lambda)
            self.scheduler = self.scheduler_adam # 对外暴露主特征器使用的LR
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def select_action(self, obs, mask_task=None, mask_station_matrix=None, mask_worker=None, deterministic=False, temperature=1.0):
        """
        选择动作 (Select Action)。
        
        Args:
            obs: 异构图观测数据 (HeteroData)
            mask_task: [N] Bool Tensor, True=Invalid
            mask_station_matrix: [N, S] Bool Tensor, True=Invalid
            mask_worker: [W] Bool Tensor, True=Invalid (Global)
            deterministic: 是否确定性选择 (ArgMax vs Sampling)
            temperature: 采样温度，T越小越贪婪，T越大越随机，忽略当 deterministic=True 时
            
        Returns:
            action_tuple: (task_id, station_id, team_indices_list)
            action_logprob: float
            state_value: float
            specific_station_mask: 用于 Memory 记录
        """
        with torch.no_grad():
            x_dict, global_context = self.policy(obs)
            
            # ------------------
            # 1. 选择工序 (Select Task)
            # ------------------
            task_logits = self.policy.task_head(x_dict['task'], global_context, mask=mask_task)
            
            # [Robustness] 检查并处理 NaN
            if torch.isnan(task_logits).any():
                task_logits = torch.nan_to_num(task_logits, nan=-1e9)
            
            if deterministic:
                if mask_task is not None:
                    task_logits = task_logits.masked_fill(mask_task, -1e9)
                task_action = torch.argmax(task_logits)
                task_logprob = torch.tensor(0.0).to(self.device)
            else:
                if mask_task is not None:
                     task_logits = task_logits.masked_fill(mask_task, -1e9)
                
                # Check for all -inf
                if (task_logits <= -1e8).all():
                     print("WARNING: All Task Logits -inf in select_action. Force picking 0.")
                     task_action = torch.tensor(0).to(self.device)
                     task_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if temperature != 1.0:
                        task_logits = task_logits / max(temperature, 1e-5)
                    task_dist = Categorical(logits=task_logits)
                    task_action = task_dist.sample()
                    task_logprob = task_dist.log_prob(task_action)
            
            t_idx = task_action.item()
            selected_task_emb = x_dict['task'][t_idx].unsqueeze(0) # [1, H]
            
            # 获取任务的人数需求
            raw_demand = obs['task'].x[t_idx, -1].item()
            demand = int(raw_demand)
            if demand < 1: demand = 1 # Safety clamp
            
            # ------------------
            # 2. 选择站位 (Select Station)
            # ------------------
            specific_station_mask = None
            if mask_station_matrix is not None:
                # [N, S] -> [1, S]
                specific_station_mask = mask_station_matrix[t_idx].unsqueeze(0)
            
            station_embs = x_dict['station'].unsqueeze(0)
            station_logits = self.policy.station_head(selected_task_emb, station_embs, mask=specific_station_mask)
            
            if torch.isnan(station_logits).any():
                station_logits = torch.nan_to_num(station_logits, nan=-1e9)
            
            if deterministic:
                if specific_station_mask is not None:
                     station_logits = station_logits.masked_fill(specific_station_mask, -1e9)
                station_action = torch.argmax(station_logits)
                station_logprob = torch.tensor(0.0).to(self.device)
            else:
                if specific_station_mask is not None:
                     station_logits = station_logits.masked_fill(specific_station_mask, -1e9)
                
                if (station_logits <= -1e8).all():
                     print("WARNING: All Station Logits -inf. Force picking 0.")
                     station_action = torch.tensor(0).to(self.device)
                     station_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if temperature != 1.0:
                        station_logits = station_logits / max(temperature, 1e-5)
                    station_dist = Categorical(logits=station_logits)
                    station_action = station_dist.sample()
                    station_logprob = station_dist.log_prob(station_action)
                
            # ------------------
            # 3. 选择工人 (Select Workers) - 自回归
            # ------------------
            team_indices = []
            worker_logprobs = []
            
            # 动态 Mask: 初始 Mask + 技能 Mask
            current_worker_mask = mask_worker.clone() if mask_worker is not None else torch.zeros(obs['worker'].num_nodes, dtype=torch.bool).to(self.device)
            
            worker_feats = obs['worker'].x
            worker_skills = worker_feats[:, 1:11] # 10 dim
            
            task_type_idx = torch.argmax(obs['task'].x[t_idx, 5:15]).item() 
            
            has_skill = worker_skills[:, task_type_idx] > 0.5
            skill_mask = ~has_skill 
            
            current_worker_mask = current_worker_mask | skill_mask.to(self.device)
            
            worker_embs = x_dict['worker'].unsqueeze(0)
            
            for _ in range(demand):
                # 还有可选工人吗?
                if current_worker_mask.all():
                    break
                
                worker_logits = self.policy.worker_head.forward_choice(selected_task_emb, worker_embs, mask=current_worker_mask)
                
                if torch.isnan(worker_logits).any():
                    worker_logits = torch.nan_to_num(worker_logits, nan=-1e9)
                
                if deterministic:
                     worker_logits = worker_logits.masked_fill(current_worker_mask, -1e9)
                     if (worker_logits <= -1e8).all(): break
                     
                     w_action = torch.argmax(worker_logits)
                     w_lp = torch.tensor(0.0).to(self.device)
                else:
                     worker_logits = worker_logits.masked_fill(current_worker_mask, -1e9)
                     
                     if (worker_logits <= -1e8).all():
                         break # 无法继续选人
                     
                     if temperature != 1.0:
                         worker_logits = worker_logits / max(temperature, 1e-5)
                         
                     w_dist = Categorical(logits=worker_logits)
                     w_action = w_dist.sample()
                     w_lp = w_dist.log_prob(w_action)
                
                w_idx = w_action.item()
                team_indices.append(w_idx)
                worker_logprobs.append(w_lp)
                
                # 更新 Mask (选过的人不能再选)
                current_worker_mask = current_worker_mask.clone() # 确保不 原地修改 影响下一轮
                current_worker_mask[w_idx] = True
            
            total_worker_logprob = sum(worker_logprobs) if worker_logprobs else torch.tensor(0.0).to(self.device)
            
            action_logprob = task_logprob + station_logprob + total_worker_logprob
            # [CRITICAL FIX] 物理隔离隔离 Critic 防止其巨大的 Value Error 梯度捣毁底层共享 GAT 拓扑特征 (灾难性干扰致盲)
            # state_value = self.policy.get_value(global_context)
            state_value = self.policy.get_value(global_context.detach())
            
            action_tuple = (t_idx, station_action.item(), team_indices)
            
        return action_tuple, action_logprob.item(), state_value.item(), specific_station_mask

    def update(self, memory, env=None):
        """
        PPO 更新逻辑。
        
        Args:
            memory: 存储轨迹的 Buffer
            
        Returns:
            metrics: dict, 用于 TensorBoard 记录
        """
        # 1. 计算广义优势估计 (GAE - Generalized Advantage Estimation)
        rewards = []
        advantages = []
        gae = 0
        
        # 将 rewards 与 values 张量化以进行 GAE 计算
        mem_rewards = memory.rewards
        mem_is_terminals = memory.is_terminals
        
        # 提取存储在 states 中的 state_values
        # (这需要在 select_action 之后被记录下来，如果没有记录，回退为普通的 MC 回报加基线)
        if hasattr(memory, 'values') and len(memory.values) == len(mem_rewards):
            mem_values = memory.values
            next_value = 0 # 终止状态后的 value 为 0
            
            for step in reversed(range(len(mem_rewards))):
                if mem_is_terminals[step]:
                    next_value = 0
                    gae = 0
                
                delta = mem_rewards[step] + self.gamma * next_value - mem_values[step]
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages.insert(0, gae)
                next_value = mem_values[step]
                
            advantages = torch.tensor(advantages, dtype=torch.float32)
            rewards = advantages + torch.tensor(mem_values, dtype=torch.float32)
        else:
            # Fallback 到 Monte-Carlo + Advantage (如果缺少 Value 记录)
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(mem_rewards), reversed(mem_is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
                
            rewards = torch.tensor(rewards, dtype=torch.float32)
            # 兼容处理
            advantages = rewards.clone()
            
        # 归一化 Advantages (有助于训练稳定性)
        if advantages.std() > 1e-7:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        else:
            advantages = advantages - advantages.mean()
        
        # 2. 准备 Batch 数据
        old_actions = memory.actions 
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
        
        # Pad Team List (变长 -> 定长 Tensor)
        max_team_size = max(len(a[2]) for a in old_actions) if old_actions else 1
        
        b_task = torch.tensor([a[0] for a in old_actions], dtype=torch.long)
        b_station = torch.tensor([a[1] for a in old_actions], dtype=torch.long)
        
        team_list = []
        for a in old_actions:
            t = a[2]
            pad = [-1] * (max_team_size - len(t))
            team_list.append(t + pad)
        b_team = torch.tensor(team_list, dtype=torch.long)
        
        # Attach targets to Data objects for Batching
        rebuilt_states = []
        if env is not None:
             for snap in memory.states:
                 rebuilt_states.append(env.rebuild_state_from_snapshot(snap))
        else:
             rebuilt_states = memory.states
             
        for i, state in enumerate(rebuilt_states):
            state.y_task = b_task[i].unsqueeze(0)
            state.y_station = b_station[i].unsqueeze(0)
            state.y_team = b_team[i].unsqueeze(0) 
            state.y_logprob = old_logprobs[i].unsqueeze(0)
            state.y_reward = rewards[i].unsqueeze(0)
            state.y_advantage = advantages[i].unsqueeze(0)
            
            # [Added] Load original state values for PPO Value Clipping
            if len(memory.values) > i:
                 state.y_value = torch.tensor([memory.values[i]], dtype=torch.float32)
            
            if i < len(memory.masks):
                t_mask, s_mask, w_mask = memory.masks[i]
                state.y_task_mask = t_mask
                state.y_station_mask = s_mask
                state.y_worker_mask = w_mask
        
        loader = DataLoader(rebuilt_states, batch_size=self.batch_size, shuffle=True)
        
        # 3. PPO Optimization Loop
        print(f"PPO Update: BatchSize={self.batch_size}, Total Batches={len(loader)}")
        
        avg_loss = 0
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy_loss = 0
        update_counts = 0
        
        self.optimizer.zero_grad()
        if self.using_muon:
            self.optimizer_adam.zero_grad()
            
        for i_epoch in range(self.k_epochs):
            for step_idx, batch in enumerate(loader):
                batch = batch.to(self.device)
                
                # 当前策略的前向传播
                x_dict, global_context = self.policy(batch)
                # [CRITICAL FIX: Critic Detachment]
                # state_values = self.policy.get_value(global_context).view(-1)
                state_values = self.policy.get_value(global_context.detach()).view(-1)
                
                # --- Re-evaluate LogProbs ---
                # A. Task LogProb
                from torch_geometric.utils import to_dense_batch
                task_x, p_mask = to_dense_batch(x_dict['task'], batch['task'].batch)
                
                # 恢复 Mask
                if hasattr(batch, 'y_task_mask'):
                    logical_task_mask, _ = to_dense_batch(batch.y_task_mask, batch['task'].batch)
                    combined_task_mask = logical_task_mask | (~p_mask)
                else:
                    combined_task_mask = ~p_mask
                    
                task_logits = self.policy.task_head(task_x, global_context, mask=combined_task_mask)
                if torch.isnan(task_logits).any(): task_logits = torch.nan_to_num(task_logits, nan=-1e9)
                
                task_dist = Categorical(logits=task_logits)
                task_lp = task_dist.log_prob(batch.y_task)
                entropy = task_dist.entropy()
                
                # B. Station LogProb
                batch_indices = torch.arange(batch.y_task.size(0)).to(self.device)
                sel_task_emb = task_x[batch_indices, batch.y_task] 
                
                station_x, s_p_mask = to_dense_batch(x_dict['station'], batch['station'].batch)
                
                if hasattr(batch, 'y_station_mask'):
                    dense_s_mask, _ = to_dense_batch(batch.y_station_mask, batch['task'].batch)
                    specific_station_mask = dense_s_mask[batch_indices, batch.y_task]
                    curr_s_mask = specific_station_mask | (~s_p_mask)
                else:
                    curr_s_mask = ~s_p_mask
                
                station_logits = self.policy.station_head(sel_task_emb, station_x, mask=curr_s_mask)
                if torch.isnan(station_logits).any(): station_logits = torch.nan_to_num(station_logits, nan=-1e9)
                
                station_dist = Categorical(logits=station_logits)
                station_lp = station_dist.log_prob(batch.y_station)
                entropy += station_dist.entropy()
                
                # C. Worker Team LogProb
                worker_x, w_p_mask = to_dense_batch(x_dict['worker'], batch['worker'].batch)
                team_lp = torch.zeros_like(task_lp)
                
                if hasattr(batch, 'y_worker_mask'):
                     d_w_mask, _ = to_dense_batch(batch.y_worker_mask.float(), batch['worker'].batch)
                     curr_mask = (d_w_mask > 0.5) | (~w_p_mask)
                else:
                     curr_mask = (~w_p_mask)
                
                # Add Skill Mask based on the selected task
                task_raw, _ = to_dense_batch(batch['task'].x, batch['task'].batch)
                sel_task_raw = task_raw[batch_indices, batch.y_task]
                task_type_idx = torch.argmax(sel_task_raw[:, 5:15], dim=1) # [B]
                
                worker_raw, _ = to_dense_batch(batch['worker'].x, batch['worker'].batch)
                worker_skills = worker_raw[:, :, 1:11] # [B, Max_W, 10]
                
                B_size, Max_W_size = worker_skills.shape[0], worker_skills.shape[1]
                b_indices_expanded = torch.arange(B_size).view(-1, 1).expand(-1, Max_W_size).reshape(-1)
                w_indices_expanded = torch.arange(Max_W_size).view(1, -1).expand(B_size, -1).reshape(-1)
                t_indices_expanded = task_type_idx.view(-1, 1).expand(-1, Max_W_size).reshape(-1)
                
                has_skill_flat = worker_skills[b_indices_expanded, w_indices_expanded, t_indices_expanded] > 0.5
                skill_mask = (~has_skill_flat).view(B_size, Max_W_size).to(self.device)
                
                curr_mask = curr_mask | skill_mask
                
                for k in range(batch.y_team.size(1)):
                    target = batch.y_team[:, k] 
                    valid_step = (target != -1)
                    if not valid_step.any(): continue
                    
                    logits = self.policy.worker_head.forward_choice(sel_task_emb, worker_x, mask=curr_mask)
                    if torch.isnan(logits).any(): logits = torch.nan_to_num(logits, nan=-1e9)
                    
                    dist = Categorical(logits=logits)
                    step_lp = dist.log_prob(torch.clamp(target, min=0)) 
                    team_lp[valid_step] += step_lp[valid_step]
                    entropy[valid_step] += dist.entropy()[valid_step]
                    
                    # Update mask for next worker in team
                    curr_mask = curr_mask.clone()
                    valid_b_indices = torch.nonzero(valid_step).squeeze(-1)
                    curr_mask[valid_b_indices, target[valid_step]] = True
                            
                total_lp = task_lp + station_lp + team_lp
                
                # --- PPO Loss Calculation ---
                ratios = torch.exp(total_lp - batch.y_logprob.view(-1))
                
                # Use GAE advantages if available, else batch.y_reward - state_values (MC fallback)
                b_adv = batch.y_advantage.view(-1) if hasattr(batch, 'y_advantage') else (batch.y_reward.view(-1) - state_values.detach())
                
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * b_adv
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value and Entropy Loss scaled by configs
                c_val = getattr(configs, 'c_value', 0.5)
                c_ent = getattr(configs, 'c_entropy', 0.01)
                c_pol = getattr(configs, 'c_policy', 1.0)
                
                # [CRITICAL FIX: Huber Loss] 使用 Huber Loss 替代 MSE，当预测误差过大（如数千万）时线性回传梯度，防止 Critic 巨型梯度经 clip 后将 Policy 梯度抹零致盲
                b_reward = batch.y_reward.view(-1)
                value_loss = c_val * torch.nn.functional.huber_loss(state_values, b_reward, delta=10.0)
                     
                entropy_loss = -c_ent * entropy.mean()
                
                loss = c_pol * policy_loss + value_loss + entropy_loss
                
                # Backprop
                loss = loss / self.accumulation_steps # 归一化 Gradient
                loss.backward()
                
                # [Gradient Accumulation]
                if ((step_idx + 1) % self.accumulation_steps == 0) or (step_idx + 1 == len(loader)):
                    # [Gradient Clipping]
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    
                    self.optimizer.step()
                    if self.using_muon:
                        self.optimizer_adam.step()
                        self.optimizer_adam.zero_grad()
                    self.optimizer.zero_grad()
                    
                    update_counts += 1
                
                # Log Stats (取消除以 accumulation_steps 来显示真实 loss 幅度)
                avg_loss += loss.item() * self.accumulation_steps
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy_loss += entropy_loss.item()
        
        # [Step Scheduler]
        if self.using_muon:
            self.scheduler_muon.step()
            self.scheduler_adam.step()
        else:
            self.scheduler.step()
            
        self.current_step += 1
                
        metrics = {
            'Loss/Total': avg_loss / update_counts if update_counts > 0 else 0,
            'Loss/Policy': avg_policy_loss / update_counts if update_counts > 0 else 0,
            'Loss/Value': avg_value_loss / update_counts if update_counts > 0 else 0,
            'Loss/Entropy': avg_entropy_loss / update_counts if update_counts > 0 else 0,
            'Train/LearningRate': self.scheduler_adam.get_last_lr()[0] if self.using_muon else self.scheduler.get_last_lr()[0]
        }
        return metrics
