import torch
import numpy as np
import argparse
import sys
import os
import pandas as pd

# (Removed hardcoded sys.path.append to comply with standard project struct)

from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from configs import configs
from utils.visualization import plot_gantt

def evaluate(args):
    """
    模型评估脚本。
    
    功能:
    1. 加载训练好的模型 (.pth)。
    2. 在环境中运行一轮确定性推理 (Deterministic Inference)。
    3. 输出评估指标 (Makespan, Balance)。
    4. 生成排程结果 CSV 和甘特图 PNG。
    """
    print("--- 开始评估 (Starting Evaluation) ---")
    
    # 1. 加载数据与环境
    data_path = args.data_path if args.data_path else configs.data_file_path
    if not os.path.exists(data_path) and os.path.exists(os.path.join(os.getcwd(), data_path)):
            data_path = os.path.join(os.getcwd(), data_path)
    
    print(f"数据路径: {data_path}")
    env = AirLineEnv_Graph(data_path=data_path, seed=42)
    print("环境初始化完成.")
    
    # 2. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = HBGATPN(configs).to(device)
    
    # 3. 加载 Checkpoint
    checkpoint_path = args.model_path
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型文件 {checkpoint_path}")
        return

    print(f"加载模型: {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # 支持加载完整 checkpoint 或仅 state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载 Checkpoint (Episode {checkpoint.get('episode', 'Unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("已加载 State Dict.")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # Agent 包装 (主要为了使用 select_action 方法)
    agent = PPOAgent(model, lr=0, gamma=0, k_epochs=0, eps_clip=0, device=device)
    
    # 4. 执行推理 (N Runs)
    num_runs = args.num_runs
    temperature = args.temperature
    
    print(f"正在执行推理 (Runs: {num_runs}, Temperature: {temperature})...")
    
    makespans = []
    balances = []
    rewards = []
    
    best_makespan = float('inf')
    best_assigned_tasks = None
    
    for run in range(num_runs):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            task_mask, station_mask, worker_mask = env.get_masks()
            
            # 引入温度采样的推理选择
            action, _, _, _ = agent.select_action(
                state.to(device),
                mask_task=task_mask.to(device),
                mask_station_matrix=station_mask.to(device),
                mask_worker=worker_mask.to(device),
                deterministic=(temperature == 0.0), # 如果温度是0，执行贪婪逻辑
                temperature=temperature
            )
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
        makespan = np.max(env.station_loads)
        balance_std = np.std(env.station_loads)
        
        makespans.append(makespan)
        balances.append(balance_std)
        rewards.append(total_reward)
        
        if makespan < best_makespan:
            best_makespan = makespan
            import copy
            best_assigned_tasks = copy.deepcopy(env.assigned_tasks)
            
        print(f"Run {run+1}/{num_runs} - Makespan: {makespan:.2f}, Balance: {balance_std:.2f}, Reward: {total_reward:.2f}")
        
    # 5. 计算指标
    avg_makespan = np.mean(makespans)
    avg_balance = np.mean(balances)
    avg_reward = np.mean(rewards)
    
    print("\n--- 评估结果汇总 ---")
    print(f"多次运行平均 Makespan: {avg_makespan:.2f} h (Best: {best_makespan:.2f} h)")
    print(f"多次运行平均 Balance Std: {avg_balance:.2f}")
    print(f"多次运行平均 Total Reward: {avg_reward:.4f}")
    
    # 6. 导出最佳排程结果
    if best_assigned_tasks is not None:
        tasks_data = []
        for (tid, sid, team, start, end) in best_assigned_tasks:
            tasks_data.append({
                'TaskID': tid,
                'StationID': sid + 1,
                'Team': str(team),
                'Start': start,
                'End': end,
                'Duration': end - start
            })
        
        df_res = pd.DataFrame(tasks_data)
        csv_path = "schedule_result_best.csv"
        df_res.to_csv(csv_path, index=False)
        print(f"最佳次详细排程表已保存至: {csv_path}")
        
        # 7. 生成甘特图
        png_path = "schedule_gantt_best.png"
        print("正在生成最佳轮次甘特图...")
        plot_gantt(best_assigned_tasks, png_path)
        print(f"最佳轮次甘特图已保存至: {png_path}")
    else:
        print("无法生成排程图: 没有可用的任务分配记录.")

    
    print("评估流程结束.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径 (.pth)')
    parser.add_argument('--data_path', type=str, default=None, help='数据文件路径 (.csv)')
    parser.add_argument('--num_runs', type=int, default=3, help='评估的重复运行次数 (用于计算平均值)')
    parser.add_argument('--temperature', type=float, default=getattr(configs, 'eval_temperature', 0.0), help='采样温度。0表示完全贪婪执行。')
    
    args = parser.parse_args()
    evaluate(args)
