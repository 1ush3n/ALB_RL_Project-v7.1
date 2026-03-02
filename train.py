import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys
import traceback
import argparse

# 添加项目根目录到路径
# (Removed hardcoded sys.path.append to comply with standard project struct)

from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from configs import configs
import pandas as pd
from baseline_ga import GeneticAlgorithmScheduler
from utils.visualization import plot_gantt

# ---------------------------------------------------------------------------
# 经验回放缓冲区 (Memory Buffer)
# ---------------------------------------------------------------------------
class Memory:
    """
    存储 PPO 训练所需的轨迹数据。
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = [] # (task_mask, station_mask, worker_mask)
        self.values = [] # (state_value)
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.masks[:]
        del self.values[:]

# ---------------------------------------------------------------------------
# 评估函数
# ---------------------------------------------------------------------------
def evaluate_model(env, agent, num_runs=3, temperature=None):
    """
    使用包含温度平滑的定制定向策略评估当前模型性能，并执行多次以取均值。
    
    Returns:
        makespan (float): 多次运行均值最大完工时间 
        balance_std (float): 多次运行均值站位负载的标准差
        total_reward (float): 多次运行均值有效奖励总和
    """
    if temperature is None:
        temperature = getattr(configs, 'eval_temperature', 0.0)
        
    makespans = []
    balances = []
    rewards = []
    
    for _ in range(num_runs):
        # 验证场景绝对不可以使用任何数据扰乱！保证评估基线的绝对公平。
        state = env.reset(randomize_duration=False)
        done = False
        total_reward = 0
        device = agent.device
        
        while not done:
            task_mask, station_mask, worker_mask = env.get_masks()
            
            # 引入验证温度的动作选择
            action, _, _, _ = agent.select_action(
                state.to(device), 
                mask_task=task_mask.to(device), 
                mask_station_matrix=station_mask.to(device),
                mask_worker=worker_mask.to(device),
                deterministic=(temperature == 0.0),
                temperature=temperature
            )
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
        makespans.append(np.max(env.station_loads))
        balances.append(np.std(env.station_loads))
        rewards.append(total_reward)
        
    return np.mean(makespans), np.mean(balances), np.mean(rewards)

# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------
def train(args):
    try:
        print("--- 开始训练 (Starting Training) ---")
        
        # 1. 初始化环境
        data_path = str(configs.data_file_path) if configs.data_file_path else "3000.csv"
        # 转换为绝对路径以防万一
        if not os.path.exists(data_path) and os.path.exists(os.path.join(os.getcwd(), data_path)):
             data_path = os.path.join(os.getcwd(), data_path)
             
        print(f"数据路径: {data_path}")
        # 固定种子以保证训练环境的一致性 (Determinism)
        env = AirLineEnv_Graph(data_path=data_path, seed=42)
        print("环境初始化完成.")
        
        # 2. 初始化设备与模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        model = HBGATPN(configs).to(device)
        print("模型已加载至设备.")
        
        # Init Agent
        # Calculate Total Timesteps for Scheduler
        total_updates = int(configs.max_episodes / configs.update_every_episodes)

        agent = PPOAgent(
            model=model,
            lr=configs.lr,
            gamma=configs.gamma,
            k_epochs=configs.k_epochs,
            eps_clip=configs.eps_clip,
            device=device,
            batch_size=configs.batch_size,
            # [Scheduler Params]
            lr_warmup_steps=configs.lr_warmup_steps,
            min_lr=configs.min_lr,
            total_timesteps=total_updates
        )

        

        print(f"Agent Initialized. Total Scheduled Updates: {total_updates}")
        
        # 3. 断点续训 (Resume Training)
        start_episode = 1
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, "latest_checkpoint.pth")
        
        if args.resume and os.path.exists(checkpoint_path):
            print(f"正在从 {checkpoint_path} 恢复训练...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                agent.policy.load_state_dict(checkpoint['model_state_dict'])
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_episode = checkpoint['episode'] + 1
                print(f"恢复成功. 起始 Episode: {start_episode}")
            except RuntimeError as e:
                print(f"⚠️ 恢复失败: 模型结构不匹配 (可能是 configs 修改了层数/维度). 跳过恢复。\n报错信息截取: {str(e)[:100]}...")
        
        # 最佳模型记录
        best_makespan = float('inf')
        best_model_dir = os.path.join(model_dir, "bestmodel")
        os.makedirs(best_model_dir, exist_ok=True)
        best_model_path = os.path.join(best_model_dir, "best_model.pth")
        
        # 4. TensorBoard 设置
        run_name = f"ALB_PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = os.path.join(configs.log_dir, run_name)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard 日志目录: {log_dir}")
        
        memory = Memory()
        
        # 5. 训练循环参数
        max_episodes = configs.max_episodes 
        update_every_episodes = configs.update_every_episodes
        eval_freq = configs.eval_freq
        
        print(f"开始 Episode 循环 (Max: {max_episodes}, 开启 {configs.num_workers} 进程并发探索)...")
        
        import concurrent.futures
        import multiprocessing
        from utils.worker import single_episode_worker
        
        # 兼容 Linux 与 Windows 下多进程 CUDA 初始化防断连
        mp_ctx = multiprocessing.get_context('spawn')
        
        ep = start_episode
        while ep <= configs.max_episodes:
            
            # --- 1. 计算这一批次要发射多少个并行进程 ---
            batch_eps = min(configs.update_every_episodes, configs.max_episodes - ep + 1)
            
            # --- 2. 剥离 GPU 模型主参数至 CPU 内存以通过 Pipe 传输 ---
            shared_state_dict = {k: v.cpu() for k, v in agent.policy.state_dict().items()}
            
            worker_args = []
            for i in range(batch_eps):
                seed = 2026 + ep + i
                worker_args.append((i, shared_state_dict, configs, seed))
                
            print(f"\n--- 🚀 发射 {batch_eps} 个分布探险器前往平行宇宙 (Episode Batch {ep}~{ep+batch_eps-1}) ---")
            
            aggregated_memory = Memory() 
            total_make, total_bal, total_rew = 0, 0, 0
            success_workers = 0
            
            # --- 3. 阻塞式并行收集 ---
            with concurrent.futures.ProcessPoolExecutor(max_workers=configs.num_workers, mp_context=mp_ctx) as executor:
                results = executor.map(single_episode_worker, worker_args)
                
                for res in results:
                    if res.get('error'):
                         print(f"⚠️ Worker {res['worker_id']} Failed: {res['error']}\n{res.get('traceback', '')}")
                         continue
                        
                    mem = res['memory']
                    aggregated_memory.states.extend(mem.states)
                    aggregated_memory.actions.extend(mem.actions)
                    aggregated_memory.logprobs.extend(mem.logprobs)
                    aggregated_memory.rewards.extend(mem.rewards)
                    aggregated_memory.is_terminals.extend(mem.is_terminals)
                    aggregated_memory.masks.extend(mem.masks)
                    aggregated_memory.values.extend(mem.values)
                    
                    print(f"🌐 探测器返回 | Ep: {ep+res['worker_id']} | Reward: {res['ep_reward']:.2f} | 步数: {res['steps']} | Makespan: {res['makespan']:.1f}")
                    
                    total_make += res['makespan']
                    total_bal += res['balance']
                    total_rew += res['ep_reward']
                    success_workers += 1
            
            # --- 4. 统一 PPO 梯度反向传播 ---
            if success_workers > 0:
                avg_make = total_make / success_workers
                avg_bal = total_bal / success_workers
                avg_rew = total_rew / success_workers
                
                writer.add_scalar('Reward/Episode_Avg', avg_rew, ep)
                
                print(f"✅ 并行收集完成! 开始统合 {len(aggregated_memory.rewards)} 条超大规模经验执行 PPO 核心反向传播...")
                metrics = agent.update(aggregated_memory, env) 
                
                for k, v in metrics.items():
                    writer.add_scalar(k, v, ep)
                    
            # --- 5. 定期评估与快照留存 ---
            if ep % eval_freq == 0 or ep + batch_eps > configs.max_episodes:
                makespan, balance, eval_reward = evaluate_model(env, agent, num_runs=3, temperature=configs.eval_temperature)
                
                print(f"[Eval] Ep Batch {ep} | Avg Makespan: {makespan:.1f} | Avg Balance: {balance:.2f} | AvgReward: {eval_reward:.2f}")
                
                writer.add_scalar('Eval/Makespan', makespan, ep)
                writer.add_scalar('Eval/Balance_Std', balance, ep)
                
                # Save Latest
                torch.save({
                    'episode': ep + batch_eps - 1,
                    'model_state_dict': agent.policy.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                }, checkpoint_path)
                
                # Save Best
                if makespan < best_makespan:
                    best_makespan = makespan
                    torch.save(agent.policy.state_dict(), best_model_path)
                    print(f"🌟 New Best Model Saved! Makespan: {best_makespan}")
            
            # 步进 Episode
            ep += batch_eps
                    
        # =======================================================================
        # 6. 训练结束 - 终局性能测评与基线对比 (End of Training Evaluation)
        # =======================================================================
        print("\n" + "="*50)
        print("🎉 强化学习训练循环已结束！开始获取最强方案对比基线。")
        print("="*50)
        
        # 加载最好验证参数
        if os.path.exists(best_model_path):
             print(f"加载训练历史上最好的验证模型用于最终推演: {best_model_path}")
             try:
                 model.load_state_dict(torch.load(best_model_path, map_location=device))
             except RuntimeError as e:
                 print(f"⚠️ 警告: 历史最佳模型 ({best_model_path}) 的结构与当前配置不匹配，无法加载。将继续使用当前最新的训练结果进行推演！")
             
        # 配置 PPO 最终推演
        print("\n>>> [1/2] 开始执行 PPO Agent 的终局推演...")
        # 重新实例环境，避免脏数据
        eval_env = AirLineEnv_Graph(data_path=data_path, seed=2026)
        ppo_makespan, ppo_balance, _ = evaluate_model(eval_env, agent, num_runs=5, temperature=configs.eval_temperature)
        
        # 提取最后的分配单
        ppo_assigned = eval_env.assigned_tasks 
        
        # 配置 GA 基准对抗
        print("\n>>> [2/2] 开始执行 Genetic Algorithm (GA) 基线推演...")
        ga_env = AirLineEnv_Graph(data_path=data_path, seed=2026)
        ga_scheduler = GeneticAlgorithmScheduler(ga_env, pop_size=30, max_gen=20)
        ga_makespan, ga_balance, ga_assigned = ga_scheduler.run()
        
        # --- 报表总结生成 ---
        print("\n" + "#"*50)
        print("🚀 终局对比结果报告 (PPO vs GA) 🚀")
        print(f"指标说明：Makespan (越小越好), Balance (越小越好)")
        print("-"*50)
        print(f"| 模型算法类型          | Makespan (h) | Balance Std |")
        print(f"|-----------------------|--------------|-------------|")
        print(f"| 经典运筹学: (GA 基线) | {ga_makespan:12.2f} | {ga_balance:11.2f} |")
        print(f"| 强化学习: (HB-GAT-PN) | {ppo_makespan:12.2f} | {ppo_balance:11.2f} |")
        print("#"*50 + "\n")
        
        # 导出最佳 PPO 细节到 CSV 及画图
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        def save_schedule(tasks, prefix_name):
            if not tasks: return
            tasks_data = []
            for (tid, sid, team, start, end) in tasks:
                 tasks_data.append({
                     'TaskID': tid,
                     'StationID': sid + 1,
                     'Team': str(team),
                     'Start': start,
                     'End': end,
                     'Duration': end - start
                 })
            df = pd.DataFrame(tasks_data)
            df.to_csv(os.path.join(output_dir, f"{prefix_name}_schedule.csv"), index=False)
            plot_gantt(tasks, os.path.join(output_dir, f"{prefix_name}_gantt.png"))
            
        print(f"正在向目录 ./{output_dir} 保存排程细节与甘特图...")
        save_schedule(ppo_assigned, "PPO_Final")
        save_schedule(ga_assigned, "GA_Baseline")
        print("所有流程圆满结束！")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    args = parser.parse_args()
    
    train(args)
