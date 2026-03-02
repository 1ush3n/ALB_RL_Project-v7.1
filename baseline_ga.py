import numpy as np
import random
import time
import copy
import pandas as pd
import argparse
import sys
import os

# (Removed hardcoded sys.path.append to comply with standard project struct)

from environment import AirLineEnv_Graph
from configs import configs

class GeneticAlgorithmScheduler:
    """
    针对飞机装配线的遗传算法基线调度器 (GA)
    
    采用两段式实数/整数编码机制：
    1. 任务优先序列 (Task Priority/Sequence)
    2. 站位与工人的指派映射 (Assignment Map)
    
    使用基于拓扑排序的安全交叉与变异机制保障约束不被破坏。
    """
    def __init__(self, env, pop_size=50, max_gen=100, cx_pb=0.8, mut_pb=0.2):
        self.env = env
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.cx_pb = cx_pb
        self.mut_pb = mut_pb
        
        self.num_tasks = env.num_tasks
        self.num_stations = env.num_stations
        self.num_workers = env.num_workers
        
        # 预先重置环境以获得拓扑字典
        self.env.reset()
        
        # 预存所有任务的合法依赖关系以保障生成合法的拓扑序列
        self.predecessors = env.predecessors
        
    def _generate_valid_topological_sort(self):
        """生成一个随机的且满足前置拓扑约束的任务排单序列 (基于 Kahn 算法变体)"""
        in_degrees = {i: len(self.predecessors[i]) for i in range(self.num_tasks)}
        ready_queue = [i for i in range(self.num_tasks) if in_degrees[i] == 0]
        
        topo_seq = []
        while ready_queue:
            # 随机挑选一个没有任何前置依赖的任务
            idx = random.randrange(len(ready_queue))
            u = ready_queue.pop(idx)
            topo_seq.append(u)
            
            for v in self.env.successors[u]:
                in_degrees[v] -= 1
                if in_degrees[v] == 0:
                    ready_queue.append(v)
                    
        return topo_seq
        
    def _create_individual(self):
        """
        初始化单挑染色体
        Chrom1: 合法的工序序列 (形如 [7, 0, 12, ...])
        Chrom2: 对应工序指派的站位 (形如 [2, 0, 1, ...])
        Chrom3: 针对不同工序指派的“偏优”选人倾向种子序列
        """
        seq_chrom = self._generate_valid_topological_sort()
        
        # 随机分配站位 (并不考虑是否超载，交由适应度函数去惩罚)
        station_chrom = [random.randint(0, self.num_stations - 1) for _ in range(self.num_tasks)]
        
        # 队伍分配倾向随机映射 (0~1 浮点数，代表偏好顺序)
        # 每一次评估时，将按照这些偏好在满足技能束缚的池子里挑人
        team_preference_chrom = np.random.rand(self.num_tasks, self.num_workers).tolist()
        
        return {'seq': seq_chrom, 'station': station_chrom, 'team_pref': team_preference_chrom}

    def _init_population(self):
        return [self._create_individual() for _ in range(self.pop_size)]
        
    def _evaluate_fitness(self, individual):
        """
        使用沙盒仿真模拟这条染色体的调度逻辑，返回适应度 (makespan)
        越小越好。
        """
        # 利用 deepcopy 隔离环境状态，使得多条染色体独立模拟
        sim_env = copy.deepcopy(self.env)
        sim_env.reset() # Soft reset
        
        seq = individual['seq']
        stations = individual['station']
        team_prefs = np.array(individual['team_pref'])
        
        # 将工序排队转化为优先级字典 (排在越前面优先级数值越高)
        priority_map = {task_id: (self.num_tasks - idx) for idx, task_id in enumerate(seq)}
        
        done = False
        total_makespan = float('inf')
        total_balance_std = float('inf')
        
        # 强制步数容错
        max_limit = self.num_tasks * 3 
        step = 0
        
        while not done and step < max_limit:
            step += 1
            t_mask, s_mask, w_mask = sim_env.get_masks()
            
            # Deadlock 保护
            if t_mask.all():
                # 遭受锁死的染色体给予极其恶劣的惩罚
                return 999999.0, 9999.0
            
            # 1. 在当前可做任务中，挑剔出 priority_map 最高的那个
            available_tasks = [i for i in range(self.num_tasks) if not t_mask[i].item()]
            
            if not available_tasks: 
                sim_env._advance_time()
                continue
                
            best_task_id = max(available_tasks, key=lambda x: priority_map[x])
            
            # 2. 定站位
            # 若配置的站位非法(mask=True)，则强制分配给当前合法的、序号最小的站位
            desired_station = stations[best_task_id]
            if s_mask[best_task_id, desired_station]:
                # 找一个合法的站位 (容错)
                valid_stations = [s for s in range(self.num_stations) if not s_mask[best_task_id, s]]
                if not valid_stations:
                    sim_env._advance_time() # 没任何站位空闲，被逼跳过时间
                    continue
                desired_station = valid_stations[0] 
                
            # 3. 定人员
            task_type_idx = int(sim_env.task_static_feat[best_task_id, 1].item())
            req_demand = max(1, int(sim_env.task_static_feat[best_task_id, 2].item()))
            
            # 从全局掩码过滤可用工人
            available_workers = [w for w in range(self.num_workers) if not w_mask[w].item()]
            
            # 再过滤：有技能的可用工人
            skilled_available = []
            for w in available_workers:
                worker_skills = sim_env.worker_skill_matrix[w]
                if worker_skills[task_type_idx] > 0.5:
                    skilled_available.append(w)
            
            if len(skilled_available) < req_demand:
                 # 人数不够，无法开工，强行推进时间释放工人
                 sim_env._advance_time()
                 continue
                 
            # 根据当前工序的偏好染色体为这些合格工人打分并降序排序
            prefs = team_prefs[best_task_id]
            skilled_available.sort(key=lambda w: prefs[w], reverse=True)
            
            selected_team = skilled_available[:req_demand]
            
            # 执行环境演算
            action = (best_task_id, desired_station, selected_team)
            _, _, done, _ = sim_env.step(action)
            
        if done:
            total_makespan = np.max(sim_env.station_loads)
            total_balance_std = np.std(sim_env.station_loads)
            
        # 以 makespan 为第一适应度 (越小越好)，balance 为次要 (可转化为惩罚项)
        fitness = total_makespan + 0.1 * total_balance_std 
        return fitness, (total_makespan, total_balance_std, sim_env.assigned_tasks)

    def _crossover(self, p1, p2):
        """
        拓扑安全交叉算子。
        Seq: 由于拓扑序列存在强依赖，使用简单的单点交叉会破坏合法性，故采用类似 Order Crossover 的拓扑修复机制，或更简单地交替继承生成。这里采用重新基于父代倾向生成拓扑的方法。
        Station/Team: 采用平滑的均匀交叉 Uniform Crossover。
        """
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        
        # 1. 站位与团队倾向发生均匀交叉
        for i in range(self.num_tasks):
            if random.random() < 0.5:
                c1['station'][i], c2['station'][i] = p2['station'][i], p1['station'][i]
                c1['team_pref'][i], c2['team_pref'][i] = p2['team_pref'][i], p1['team_pref'][i]
                
        # 2. 序列发生融合交叉 (保留合规性)
        # 用启发式修复：按照 p1 的大体顺序构建，缺失的按 Kahn 补充
        # (在此简易实现中，我们直接用一定概率发生突变来代替复杂的序列交叉，因为拓扑合法性极其苛刻)
        if random.random() < 0.3:
            c1['seq'] = self._generate_valid_topological_sort()
        if random.random() < 0.3:
            c2['seq'] = self._generate_valid_topological_sort()
            
        return c1, c2

    def _mutate(self, ind):
        """变异算子"""
        # 序列部分：有概率重新洗牌 (生成新的合法序列)
        if random.random() < self.mut_pb:
             ind['seq'] = self._generate_valid_topological_sort()
             
        # 站位变异
        for i in range(self.num_tasks):
            if random.random() < (self.mut_pb / 10.0): # 小概率变异具体站位
                ind['station'][i] = random.randint(0, self.num_stations - 1)
                
            # 偏好变异：添加扰动
            if random.random() < (self.mut_pb / 5.0):
                ind['team_pref'][i] = (np.array(ind['team_pref'][i]) + np.random.normal(0, 0.2, self.num_workers)).tolist()
                
        return ind

    def run(self):
        print(f"--- 启动运筹学遗传算法 (GA) 基线 ---")
        print(f"配置: PopSize={self.pop_size}, MaxGen={self.max_gen}")
        
        pop = self._init_population()
        
        best_overall_individual = None
        best_overall_fitness = float('inf')
        best_overall_metrics = None
        
        start_t = time.time()
        
        for g in range(self.max_gen):
            fitnesses_and_metrics = []
            
            # 1. 评估种群
            for ind in pop:
                fit, metrics = self._evaluate_fitness(ind)
                fitnesses_and_metrics.append((fit, metrics, ind))
                
                if fit < best_overall_fitness:
                    best_overall_fitness = fit
                    best_overall_metrics = metrics
                    best_overall_individual = copy.deepcopy(ind)
                    
            # 2. 选择 (Tournament Selection)
            # 根据适应度排序 (从小到大，因越小越好)
            fitnesses_and_metrics.sort(key=lambda x: x[0])
            
            # Print stats
            best_g_fit = fitnesses_and_metrics[0][0]
            avg_g_fit = np.mean([x[0] for x in fitnesses_and_metrics if x[0] < 99999.0]) # 排除死锁异常的
            print(f"[Gen {g+1}/{self.max_gen}] Best Fit: {best_g_fit:.2f} (Makespan: {fitnesses_and_metrics[0][1][0]:.2f}) | Avg Fit: {avg_g_fit:.2f}")
            
            # 精英保留策略 (Elitism)
            next_pop = [x[2] for x in fitnesses_and_metrics[:int(self.pop_size * 0.1)]]
            
            # 3. 产生下一代
            while len(next_pop) < self.pop_size:
                # 锦标赛选择选出父亲母亲
                t_size = 3
                p1_candidates = random.sample(fitnesses_and_metrics, t_size)
                p2_candidates = random.sample(fitnesses_and_metrics, t_size)
                
                p1 = min(p1_candidates, key=lambda x: x[0])[2]
                p2 = min(p2_candidates, key=lambda x: x[0])[2]
                
                if random.random() < self.cx_pb:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                    
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                
                next_pop.extend([c1, c2])
                
            pop = next_pop[:self.pop_size] # 截断溢出
            
        time_elapsed = time.time() - start_t
        print(f"\n--- GA 基线运行结束 (耗时: {time_elapsed:.1f}s) ---")
        
        makespan, balance_std, assigned_tasks = best_overall_metrics
        print(f"最好成绩 -> Makespan: {makespan:.2f} h | Balance Std: {balance_std:.2f}")
        
        return makespan, balance_std, assigned_tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help='数据文件路径 (.csv)')
    parser.add_argument('--pop_size', type=int, default=30)
    parser.add_argument('--max_gen', type=int, default=20)
    args = parser.parse_args()
    
    data_path = args.data_path if args.data_path else configs.data_file_path
    if not os.path.exists(data_path) and os.path.exists(os.path.join(os.getcwd(), data_path)):
         data_path = os.path.join(os.getcwd(), data_path)
         
    env = AirLineEnv_Graph(data_path=data_path, seed=2026)
    
    ga_solver = GeneticAlgorithmScheduler(env, pop_size=args.pop_size, max_gen=args.max_gen)
    ga_solver.run()
