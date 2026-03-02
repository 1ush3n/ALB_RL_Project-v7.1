import gymnasium as gym
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from gymnasium import spaces
import sys
import os
import pandas as pd
import heapq
import networkx as nx 

# (Removed hardcoded sys.path.append to comply with standard project struct)

from data_loader import load_data
from configs import configs

# Event Definition
# time: Event occur time
# type: 'TASK_FINISH'
# data: {'task_id': int, 'worker_ids': list, 'station_id': int}
# Event Definition
class Event:
    """
    仿真事件类
    Attributes:
        time (float): 事件发生的时间
        type (str): 事件类型 (目前主要使用 'TASK_FINISH')
        data (dict): 事件携带的数据 (如 task_id, worker_ids 等)
    """
    def __init__(self, time, type, data):
        self.time = time
        self.type = type
        self.data = data
        
    def __lt__(self, other):
        # 用于优先队列的排序，时间小的在前
        return self.time < other.time


# ---------------------------------------------------------------------------
# 航空装配线环境 (AirLineEnv_Graph)
# ---------------------------------------------------------------------------
class AirLineEnv_Graph(gym.Env):
    """
    基于图的航空装配线强化学习环境。
    
    核心特性:
    1. 异构图状态: 包含 Task, Worker, Station 三种节点及其相互关系。
    2. 离散事件仿真: 时间推进基于事件(Event-Driven)，而非固定步长。
    3. 复杂约束: 包含工艺优先关系、技能匹配、站位空间约束。
    """
    
    def __init__(self, data_path="工序约束_50.xlsx", seed=None):
        super().__init__()
        
        # 设置随机种子以保证环境复现性 (Determinism)
        # 这对于验证集评估至关重要
        if seed is not None:
            np.random.seed(seed)
            # torch.manual_seed(seed) # 如果涉及 torch 的随机生成，也应设置
        
        # 加载数据
        self.raw_data = load_data(data_path)
        self.num_tasks = self.raw_data['num_tasks']
        self.num_workers = configs.n_w
        self.num_stations = configs.n_m
        
        # 动作空间: Tuple(Task, Station, Worker_List_Leader, Num_Workers)
        # 注意: 标准 Gym 不支持变长动作，这里定义为多离散空间仅作示意。
        # 实际 Agent (PPOAgent) 会处理具体的动作解码。
        self.action_space = spaces.MultiDiscrete([self.num_tasks, self.num_stations, self.num_workers])
        
        # 状态变量初始化
        self.current_time = 0.0
        # 任务状态: 0=不可用(Not Ready), 1=就绪(Ready), 2=已调度(Scheduled)
        self.task_status = np.zeros(self.num_tasks, dtype=int) 
        self.worker_free_time = np.zeros(self.num_workers, dtype=float) 
        self.station_loads = np.zeros(self.num_stations, dtype=float)
        
        # 事件队列 (Priority Queue)
        self.event_queue = []
        
        # 解析固定站位约束 (Fixed Station Constraint)
        # 从原始数据中读取 'fixed_station' 列
        self.fixed_stations = -np.ones(self.num_tasks, dtype=int)
        if 'fixed_station' in self.raw_data['task_df'].columns:
            for idx, val in enumerate(self.raw_data['task_df']['fixed_station']):
                if pd.isna(val): continue
                # 解析逻辑: 支持 "Station 1", "S1", "1" 等格式
                s_idx = -1
                try:
                    val_str = str(val).lower().strip()
                    if val_str.startswith('station'):
                         s_idx = int(float(val_str.split()[-1])) - 1
                    elif val_str.startswith('s'): # S1, S2...
                         s_idx = int(float(val_str[1:])) - 1
                    else:
                         s_idx = int(float(val_str)) - 1 # 假设 Excel 中是 1-based index
                except:
                    pass
                
                if 0 <= s_idx < self.num_stations:
                    self.fixed_stations[idx] = s_idx
        
        # 初始化异构图数据结构
        self.init_hetero_data()
        
    def init_hetero_data(self):
        """
        初始化异构图的静态特征 (Task, Worker)。
        包含由 'seed' 控制的随机初始化逻辑。
        """
        data = HeteroData()
        
        # ------------------
        # 1. 任务节点 (Task Nodes)
        # ------------------
        task_df = self.raw_data['task_df']
        # 特征: [Duration, SkillType, DemandWorkers]
        dur = torch.tensor(task_df['duration'].values, dtype=torch.float).unsqueeze(1)
        skill = torch.tensor(task_df['skill_type'].values, dtype=torch.float).unsqueeze(1)
        demand = torch.tensor(task_df['demand_workers'].values, dtype=torch.float).unsqueeze(1)
        # 强制至少需要 1 人
        demand = torch.clamp(demand, min=1.0)
        
        self.task_static_feat = torch.cat([dur, skill, demand], dim=1)
        
        # ------------------
        # 2. 工人节点 (Worker Nodes)
        # ------------------
        # 效率系数: 在 [0.8, 1.2] 之间均匀分布
        self.worker_efficiency = np.random.uniform(0.8, 1.2, self.num_workers)
        self.worker_static_feat = torch.tensor(self.worker_efficiency, dtype=torch.float).unsqueeze(1)
        
        # 技能矩阵: [NumWorkers, 10] (假设最多10种技能类型)
        self.worker_skill_matrix = torch.zeros((self.num_workers, 10), dtype=torch.float)
        
        # [鲁棒性与可行性保证]
        # 计算每种技能的最大需求人数，确保每种技能至少有这么多工人拥有，
        # 防止出现 "任务需要5人，但全场只有3个合格工人" 的死锁情况。
        max_demand = demand.max().item()
        min_workers_per_skill = int(max(1, max_demand))
        
        # A. 保证覆盖率 (Guarantee Coverage)
        for s_idx in range(10):
            current_count = self.worker_skill_matrix[:, s_idx].sum()
            while current_count < min_workers_per_skill:
                # 随机挑选一个还没有该技能的工人
                choices = torch.where(self.worker_skill_matrix[:, s_idx] == 0)[0].numpy()
                if len(choices) == 0: break # 所有人都已经有了
                w_idx = np.random.choice(choices)
                self.worker_skill_matrix[w_idx, s_idx] = 1.0
                current_count += 1
            
        # B. 随机分配剩余技能 (Random Assignment)
        # 每个工人额外随机获得 1-2 个技能
        for w in range(self.num_workers):
            current_skills = self.worker_skill_matrix[w].sum()
            target_skills = np.random.randint(1, 4) # 目标总技能数
            
            if current_skills < target_skills:
                num_to_add = int(target_skills - current_skills)
                choices = torch.where(self.worker_skill_matrix[w] == 0)[0].numpy()
                if len(choices) > 0:
                    selected_skills = np.random.choice(choices, size=min(num_to_add, len(choices)), replace=False)
                    self.worker_skill_matrix[w, selected_skills] = 1.0
        
        # ------------------
        # 3. 静态边 (Static Edges)
        # ------------------
        # Task -> Task (工艺优先关系)
        data['task', 'precedes', 'task'].edge_index = self.raw_data['precedence_edges']
        
        # [显存优化] 将 Worker -> Task 的图连接从强依赖的全连接 O(W*T) 转换为极度稀疏的真实技能匹配连接
        # 此举能够把前向传播 GATv2Conv 的 660,000 级边缩减 80% 到 90%，从物理层面消除 9GB 级别的显存 OOM 和计算冗余！
        w_indices = torch.arange(self.num_workers).repeat_interleave(self.num_tasks)
        t_indices = torch.arange(self.num_tasks).repeat(self.num_workers)
        task_req_skills = skill.squeeze(1).long() # [num_tasks]
        
        # 批量检测对应工人是否真正拥有目前任务列举出的技能点
        has_skill_mask = self.worker_skill_matrix[w_indices, task_req_skills[t_indices]] == 1.0
        
        # 仅为有效节点保留连接
        data['worker', 'can_do', 'task'].edge_index = torch.stack([w_indices[has_skill_mask], t_indices[has_skill_mask]])
        
        # [再次鲁棒性检查] Check and Clamp Demand
        # 双重保险：如果初始化后发现某技能工人总数仍少于某任务需求，强制降低该任务需求。
        skill_capacity = self.worker_skill_matrix.sum(dim=0) # [10]
        
        clamped_count = 0
        for t in range(self.num_tasks):
            t_skill = int(skill[t].item())
            t_demand = int(demand[t].item())
            
            cap = int(skill_capacity[t_skill].item())
            if cap == 0:
                # 理论上不应发生，除非逻辑错误。兜底处理。
                print(f"CRITICAL: Skill {t_skill} has 0 workers! Force assigning Worker 0.")
                self.worker_skill_matrix[0, t_skill] = 1.0
                skill_capacity[t_skill] += 1
                cap = 1
                
            if t_demand > cap:
                demand[t] = cap
                clamped_count += 1
                
        if clamped_count > 0:
            print(f"[Robustness] Auto-clamped demand for {clamped_count} tasks to match worker availability.")
            
        # 更新被 Clamp 后的特征
        self.task_static_feat = torch.cat([dur, skill, demand], dim=1)
        
        # 预计算图拓扑 (前驱/后继)
        self.predecessors = {i: [] for i in range(self.num_tasks)}
        self.successors = {i: [] for i in range(self.num_tasks)}
        
        edge_index = self.raw_data['precedence_edges'].numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            self.successors[src].append(dst)
            self.predecessors[dst].append(src)
            
        self.num_preds = np.array([len(self.predecessors[i]) for i in range(self.num_tasks)])
        
        # 计算全局的关键路径和最晚允许站位 (持久化静态特征，只计算一次)
        self.is_critical = self._calculate_cpm()
        self.max_allowed_stations = self._calculate_max_allowed_stations()
        
        self.base_data = data
        self.obs_data = None # 将在 reset 中 clone
        
        # 预先分配静态底座张量，避免 step 过程中不断进行内存申请
        self.base_task_x = torch.zeros((self.num_tasks, 17))
        # [Domain Randomization] 备份只读的基础工时分布，用于后续加噪
        self.base_durations = dur.clone() / 100.0  
        self.base_task_x[:, 0:1] = self.base_durations
        
        type_onehot = torch.zeros((self.num_tasks, 10))
        type_indices = skill.long().clamp(0, 9)
        type_onehot.scatter_(1, type_indices, 1)
        self.base_task_x[:, 5:15] = type_onehot
        self.base_task_x[:, 16:17] = demand
        
        self.base_worker_x = torch.cat([self.worker_static_feat, self.worker_skill_matrix, torch.zeros((self.num_workers, 1))], dim=1)
        self.base_station_x = torch.zeros((self.num_stations, 8))
        
    def reset(self, randomize_duration=False):
        """
        重置环境状态以开始新的 Episode。
        如果在训练阶段开启 randomize_duration，则按 ±range 对静态工时进行伪装修改。
        """
        self.current_time = 0.0
        self.task_status.fill(0) 
        self.worker_free_time.fill(0.0)
        self.station_loads.fill(0.0)
        self.event_queue = []
        
        self.assigned_tasks = [] 
        self.task_station_map = {} 
        self.task_end_times = -np.ones(self.num_tasks) 
        
        # 预分配边的内存空间
        MAX_TS_EDGES = self.num_tasks
        MAX_TW_EDGES = self.num_tasks * self.num_workers
        self.edge_ts_mem = torch.zeros((2, MAX_TS_EDGES), dtype=torch.long)
        self.edge_tw_mem = torch.zeros((2, MAX_TW_EDGES), dtype=torch.long)
        self.edge_ts_cnt = 0
        self.edge_tw_cnt = 0
        
        # 预计算图拓扑 (前驱/后继)
        self.predecessors = {i: [] for i in range(self.num_tasks)}
        self.successors = {i: [] for i in range(self.num_tasks)}
        
        edge_index = self.raw_data['precedence_edges'].numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            self.successors[src].append(dst)
            self.predecessors[dst].append(src)
            
        self.num_preds = np.array([len(self.predecessors[i]) for i in range(self.num_tasks)])
        self.completed_preds = np.zeros(self.num_tasks, dtype=int)
        
        # 设定初始任务状态
        # 没有前驱的任务设为 Ready (1)
        for i in range(self.num_tasks):
            if self.num_preds[i] == 0:
                self.task_status[i] = 1 # Ready
            else:
                self.task_status[i] = 0 # Not Ready
                
        # 克隆 Observation 数据
        self.obs_data = self.base_data.clone()
        
        # [Domain Randomization] 动态篡改工时
        if randomize_duration:
            rnd_range = getattr(configs, 'dur_random_range', 0.2)
            noise = torch.ones_like(self.base_durations).uniform_(1.0 - rnd_range, 1.0 + rnd_range)
            perturbed_durations = self.base_durations * noise
            
            # 刷新模型底层观测到的图静态信息区 (Task_x[0])
            self.base_task_x[:, 0:1] = perturbed_durations
            # 刷新用于仿真计算真实验收时间 (Step duration calculation)
            self.task_static_feat[:, 0] = (perturbed_durations * 100.0).squeeze()
        else:
            # 安全还原成纯净考题卷子
            self.base_task_x[:, 0:1] = self.base_durations
            self.task_static_feat[:, 0] = (self.base_durations * 100.0).squeeze()
            
        # [关键路径计算 (CPM)]
        # 用于后续计算 Blocking Penalty
        self.is_critical = self._calculate_cpm()
        
        return self._get_observation()

    def _calculate_cpm(self):
        """
        关键路径法 (Critical Path Method, CPM)。
        逻辑:
        1. 正向递推 (Forward Pass) -> 计算最早开始时间 (ES)
        2. 反向递推 (Backward Pass) -> 计算最晚开始时间 (LS)
        3. 关键任务判定: 如果 ES == LS (Slack == 0)，则是关键任务。
        """
        durations = self.task_static_feat[:, 0].numpy()
        num_tasks = self.num_tasks
        
        # 1. 拓扑排序 (Kahn's Algorithm)
        in_degree = self.num_preds.copy()
        queue = [i for i in range(num_tasks) if in_degree[i] == 0]
        topo_order = []
        while queue:
            u = queue.pop(0)
            topo_order.append(u)
            for v in self.successors[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        # 2. 正向递推 (ES)
        es = np.zeros(num_tasks)
        for u in topo_order:
            my_es = 0
            for p in self.predecessors[u]:
                my_es = max(my_es, es[p] + durations[p])
            es[u] = my_es
            
        max_makespan = 0
        for u in range(num_tasks):
            max_makespan = max(max_makespan, es[u] + durations[u])
            
        # 3. 反向递推 (LS)
        ls = np.full(num_tasks, max_makespan)
        for u in reversed(topo_order):
            my_lf = max_makespan
            if self.successors[u]:
                children_ls = [ls[v] for v in self.successors[u]]
                my_lf = min(children_ls)
            
            ls[u] = my_lf - durations[u]
            
        # 4. 判定关键任务
        slack = ls - es
        is_critical = (slack < 1e-5)
        return is_critical

    def _calculate_max_allowed_stations(self):
        """
        通过反向拓扑遍历计算每个任务被允许部署的“最晚站位”。
        这是为了防止 RL 环境将一个无关任务扔到了非常靠后的工位，
        结果发现其【依赖子任务】在更早的站位是被限死 (Fixed Node) 的，导致永恒死锁。
        """
        num_tasks = self.num_tasks
        max_allowed = np.full(num_tasks, self.num_stations - 1)
        
        # 将 Fixed Stations 初始化进 max_allowed 
        for t in range(num_tasks):
            if self.fixed_stations[t] != -1:
                max_allowed[t] = self.fixed_stations[t]
                
        # 拓扑排序 (Kahn) - 用于获取线性处理顺序
        in_degree = self.num_preds.copy()
        queue = [i for i in range(num_tasks) if in_degree[i] == 0]
        topo_order = []
        while queue:
            u = queue.pop(0)
            topo_order.append(u)
            for v in self.successors[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
        # 沿着反向拓扑序更新最晚允许工位: 父节点的最晚工位不能晚于任何子节点的最晚工位
        for u in reversed(topo_order):
            for p in self.predecessors[u]:
                max_allowed[p] = min(max_allowed[p], max_allowed[u])
                
        return max_allowed

    def calculate_duration(self, task_id, team_indices):
        """
        非线性工时计算逻辑:
        T_real = (T_std * N_demand) / (Sum(Eff_i) * Synergy_Factor)
        
        Synergy Factor (协同系数): 
        人数越多，沟通成本越高，效率会有折扣。
        设定: 0.95 ^ (人数 - 1)
        """
        task_info = self.task_static_feat[task_id]
        t_std = task_info[0].item()
        n_demand = int(task_info[2].item())
        
        n_act = len(team_indices)
        if n_act == 0: return float('inf')
        
        # 效率求和
        sum_efficiency = sum(self.worker_efficiency[w] for w in team_indices)
        
        # 协同折扣
        syn_factor = 0.95 ** (n_act - 1)
        
        effective_capacity = sum_efficiency * syn_factor
        
        t_real = (t_std * n_demand) / effective_capacity
        return t_real

    def step(self, action):
        """
        执行一步动作。
        Action: (task_id, station_id, team_list)
        """
        task_id, station_id, team = action
        
        # 记录执行前的 makespan
        prev_makespan = np.max(self.station_loads)
        
        # 1. 执行逻辑
        duration = self.calculate_duration(task_id, team)
        
        start_time = self.current_time
        finish_time = start_time + duration
        
        # 更新工人状态
        for w in team:
            self.worker_free_time[w] = finish_time
        
        # 更新站位负载 (近似值，用于 Makespan 估算)
        self.station_loads[station_id] += duration 
        
        # 更新任务状态
        self.task_status[task_id] = 2 # 2=已调度
        self.task_end_times[task_id] = finish_time
        self.task_station_map[task_id] = station_id
        
        self.assigned_tasks.append((task_id, station_id, team, start_time, finish_time))
        
        if station_id != -1: # exclude virtual zero-duration
            ts_ptr = self.edge_ts_cnt
            self.edge_ts_mem[0, ts_ptr] = task_id
            self.edge_ts_mem[1, ts_ptr] = station_id
            self.edge_ts_cnt += 1
            
            for w in team:
                tw_ptr = self.edge_tw_cnt
                self.edge_tw_mem[0, tw_ptr] = task_id
                self.edge_tw_mem[1, tw_ptr] = w
                self.edge_tw_cnt += 1
        
        # 2. 添加事件到队列
        heapq.heappush(self.event_queue, Event(finish_time, 'TASK_FINISH', 
                                               {'task_id': task_id, 'worker_ids': team, 'station_id': station_id}))
        
        # 3. 推进仿真时间 (离散事件引擎)
        self._advance_time()
        
        # 4. 奖励函数计算 (Dense Reward)
        # ---------------------------
        # A. 完成奖励 (Completion Payout)
        reward = 1.0
        
        # B. 工时惩罚 (Efficiency Penalty)
        reward -= 0.1 * duration
        
        # C. 关键路径阻滞惩罚 (Blocking Penalty)
        # 如果关键任务已经 Ready，但因为刚才的分配导致现在没人手了，给予惩罚。
        ready_tasks = np.where(self.task_status == 1)[0]
        blocked_penalty = 0.0
        
        # 统计当前剩余的空闲工人技能
        worker_mask_np = (self.worker_free_time <= self.current_time)
        free_indices = np.where(worker_mask_np)[0]
        current_skill_counts = {}
        for w in free_indices:
            skills = np.where(self.worker_skill_matrix[w] == 1)[0]
            for s in skills:
                current_skill_counts[s] = current_skill_counts.get(s, 0) + 1
        
        for t in ready_tasks:
            if self.is_critical[t]:
                req_skill = int(self.task_static_feat[t, 1].item())
                req_demand = int(self.task_static_feat[t, 2].item())
                avail = current_skill_counts.get(req_skill, 0)
                
                if avail < req_demand:
                    # 关键任务被阻塞!
                    blocked_penalty += 0.5
                    
        reward -= blocked_penalty
        
        # D. Dense Makespan Reward (取代原来的稀疏惩罚)
        # 记录执行后的 makespan
        new_makespan = np.max(self.station_loads)
        delta_makespan = new_makespan - prev_makespan
        # 移除截断，防止发生突破上限后不受惩罚的漏洞
        reward -= getattr(configs, 'r_coef_makespan', 0.5) * delta_makespan
        
        # E. 终局奖励 (Final Reward)
        done = (len(self.assigned_tasks) == self.num_tasks)
        if done:
            # 站位平衡惩罚 (Balance Penalty)
            # [Reward Tuning] 权重从 1.0 降为 0.1，避免喧宾夺主 (当前系数0.5)
            st_std = np.std(self.station_loads)
            reward -= 0.5 * st_std 
        
        return self._get_observation(), reward, done, {}

    def _advance_time(self):
        """
        推进时间 current_time 到下一个事件点。
        处理逻辑:
        1. 处理所有 <= current_time 的事件 (Task Finish)，释放前驱。
        2. [Zero-Duration Logic]: 如果解锁了 0工时 任务，立即执行并完成，不推进时间。
        3. 检查是否有 Valid 任务可做。
           - 如果有 -> 返回控制权给 Agent。
           - 如果无 -> 跳跃到下一个事件发生的时间点。
        """
        while True:
            # 1. 处理所有已到期的事件
            while self.event_queue and self.event_queue[0].time <= self.current_time + 1e-5:
                ev = heapq.heappop(self.event_queue)
                if ev.type == 'TASK_FINISH':
                    tid = ev.data['task_id']
                    # 解锁后继
                    for succ in self.successors[tid]:
                        self.completed_preds[succ] += 1
                        if self.completed_preds[succ] == self.num_preds[succ]:
                            if self.task_status[succ] == 0:
                                self.task_status[succ] = 1 # Ready
            
            # 2. 0工时任务穿透逻辑 (Zero-Duration Penetration)
            # 必须立即处理掉所有 Ready 的 0工时任务
            ready_indices = np.where(self.task_status == 1)[0]
            zero_run_count = 0
            for t in ready_indices:
                dur = self.task_static_feat[t, 0].item()
                if dur < 1e-5: # Zero duration
                    # 立即完成
                    self.task_status[t] = 2 # Scheduled/Done
                    finish_time = self.current_time
                    self.task_end_times[t] = finish_time
                    self.task_station_map[t] = -1 # Virtual task
                    self.assigned_tasks.append((t, -1, [], finish_time, finish_time))
                    
                    # 加入事件队列 (为了统一触发 unlock 逻辑)
                    heapq.heappush(self.event_queue, Event(finish_time, 'TASK_FINISH', 
                                                           {'task_id': t, 'worker_ids': [], 'station_id': -1}))
                    zero_run_count += 1
                    
            if zero_run_count > 0:
                # 如果处理了 0工时任务，可能解锁了新任务，需要重新进入循环检查
                continue
            
            # 3. 检查是否需要 Agent 介入
            # 只有当存在 "可行 (Valid)" 任务时，才暂停并在 State 中返回。
            task_mask, _, _ = self.get_masks()
            
            if not task_mask.all():
                 # 至少有一个任务是 False (即 Valid)
                 break
            
            # 4. 如果没有 Valid 任务，则必须跳跃时间
            if not self.event_queue:
                if len(self.assigned_tasks) < self.num_tasks:
                     # 异常：队列空了且任务没做完 -> 死锁
                     break 
                else:
                     # 全部完成
                     break 
            
            # Jump to next event
            next_ev = self.event_queue[0]
            self.current_time = next_ev.time

    def get_masks(self):
        """
        生成动作掩码 (Action Masking)。
        
        Returns:
            task_mask: [N], True=Invalid (Masked), False=Valid
            station_mask: [N, M], True=Invalid
            worker_mask: [W], True=Invalid
            
        逻辑:
        1. 任务必须 Ready。
        2. 必须有足够的工人 (具备相应技能 & 当前空闲)。
        3. 站位必须符合拓扑约束 (<= 前驱的最大站位) - 暂未严格强制，目前主要靠 Fixed Station 约束。
        """
        # 1. Worker Mask (Global)
        worker_mask_np = (self.worker_free_time > self.current_time)
        worker_mask = torch.tensor(worker_mask_np, dtype=torch.bool)
        
        # 2. Task Mask
        task_mask = torch.ones(self.num_tasks, dtype=torch.bool) # Default Invalid
        station_mask = torch.ones((self.num_tasks, self.num_stations), dtype=torch.bool)
        
        ready_indices = np.where(self.task_status == 1)[0]
        
        # 使用向量化计算获取空闲技能可用量
        if (~worker_mask_np).any():
            avail_skills = self.worker_skill_matrix[~worker_mask_np].sum(dim=0).numpy()
        else:
            avail_skills = np.zeros(10)
                 
        for t in ready_indices:
            # A. 站位约束
            min_station = 0
            for p in self.predecessors[t]:
                p_s = self.task_station_map.get(p, -1)
                if p_s != -1:
                    min_station = max(min_station, p_s)
            
            fixed = self.fixed_stations[t]
            
            # B. 资源约束
            req_skill = int(self.task_static_feat[t, 1].item())
            req_demand = int(self.task_static_feat[t, 2].item())
            
            avail = avail_skills[req_skill]
            
            if avail >= req_demand:
                # 资源充足，检查站位
                valid_stations = False
                max_station = self.max_allowed_stations[t]
                if fixed != -1:
                    if min_station <= fixed <= max_station:
                        station_mask[t, fixed] = False
                        valid_stations = True
                else:
                    if min_station < self.num_stations and min_station <= max_station:
                        end_station = min(self.num_stations, max_station + 1)
                        station_mask[t, min_station:end_station] = False
                        valid_stations = True
                        
                if valid_stations:
                    task_mask[t] = False # Valid
                    
        return task_mask, station_mask, worker_mask

    def _get_observation(self):
        """
        [Phase 3.1: O(1) In-place Observation]
        构建异构图观测状态 (Observation)。
        彻底放弃在仿真步内的张量创建和拼接，转为 O(1) 预建内存片段的原地刷新。
        """
        data = self.base_data.clone()
        
        # 1. Task Features (In-place refresh)
        task_x = self.base_task_x.clone()
        task_x[:, 1:5] = 0.0 # reset status
        task_x[torch.arange(self.num_tasks), self.task_status + 1] = 1.0 # set status (offset by 1 to skip duration)
        data['task'].x = task_x
        
        # 2. Worker Features (In-place refresh)
        worker_x = self.base_worker_x.clone()
        is_free_bool = (self.worker_free_time <= self.current_time)
        worker_x[:, 11] = torch.tensor(is_free_bool, dtype=torch.float)
        data['worker'].x = worker_x
        
        # 3. Station Features (In-place refresh)
        station_x = self.base_station_x.clone()
        station_x[:, 0] = torch.tensor(self.station_loads, dtype=torch.float) / 1000.0
        data['station'].x = station_x
        
        # 4. Dynamic Edges (Assigned To)
        # 极速视图切片: O(1) 获取所有当前边索引，彻底剥离 Python 列表转换与动态构建张量的 O(N) 原罪!
        if self.edge_ts_cnt > 0:
            t_s_edge = self.edge_ts_mem[:, :self.edge_ts_cnt].clone()
            s_t_edge = torch.stack([t_s_edge[1], t_s_edge[0]], dim=0)
        else:
            t_s_edge = torch.empty((2, 0), dtype=torch.long)
            s_t_edge = torch.empty((2, 0), dtype=torch.long)
            
        data['task', 'assigned_to', 'station'].edge_index = t_s_edge
        data['station', 'has_task', 'task'].edge_index = s_t_edge
        
        if self.edge_tw_cnt > 0:
             t_w_edge = self.edge_tw_mem[:, :self.edge_tw_cnt].clone()
        else:
             t_w_edge = torch.empty((2, 0), dtype=torch.long)
             
        data['task', 'done_by', 'worker'].edge_index = t_w_edge
        
        return data

    def get_state_snapshot(self):
        """生成状态轻量级切片以存入 Buffer。"""
        return {
            'task_status': self.task_status.copy(),
            'worker_free_time': self.worker_free_time.copy(),
            'station_loads': self.station_loads.copy(),
            'current_time': self.current_time,
            'edge_ts_cnt': self.edge_ts_cnt,
            'edge_tw_cnt': self.edge_tw_cnt,
            'edge_ts_mem': self.edge_ts_mem[:, :self.edge_ts_cnt].clone() if self.edge_ts_cnt > 0 else torch.empty((2,0), dtype=torch.long),
            'edge_tw_mem': self.edge_tw_mem[:, :self.edge_tw_cnt].clone() if self.edge_tw_cnt > 0 else torch.empty((2,0), dtype=torch.long)
        }
        
    def rebuild_state_from_snapshot(self, snapshot):
        """
        基于快照恢复成 PyG 图结构，避免完整异构图深拷贝带来的极高缓存占用。
        """
        data = self.base_data.clone()
        
        task_x = self.base_task_x.clone()
        task_x[:, 1:5] = 0.0
        task_x[torch.arange(self.num_tasks), snapshot['task_status'] + 1] = 1.0
        data['task'].x = task_x
        
        worker_x = self.base_worker_x.clone()
        is_free_bool = (snapshot['worker_free_time'] <= snapshot['current_time'])
        worker_x[:, 11] = torch.tensor(is_free_bool, dtype=torch.float)
        data['worker'].x = worker_x
        
        station_x = self.base_station_x.clone()
        station_x[:, 0] = torch.tensor(snapshot['station_loads'], dtype=torch.float) / 1000.0
        data['station'].x = station_x
        
        if snapshot['edge_ts_cnt'] > 0:
            t_s_edge = snapshot['edge_ts_mem']
            s_t_edge = torch.stack([t_s_edge[1], t_s_edge[0]], dim=0)
        else:
            t_s_edge = torch.empty((2, 0), dtype=torch.long)
            s_t_edge = torch.empty((2, 0), dtype=torch.long)
            
        data['task', 'assigned_to', 'station'].edge_index = t_s_edge
        data['station', 'has_task', 'task'].edge_index = s_t_edge
        data['task', 'done_by', 'worker'].edge_index = snapshot['edge_tw_mem']
        
        return data
