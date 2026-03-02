import torch
import os
import sys
import traceback

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import AirLineEnv_Graph
from ppo_agent import PPOAgent
from models.hb_gat_pn import HBGATPN
from train import Memory

def single_episode_worker(worker_args):
    """
    单路探索矿工进程。
    接收来自主进程的 state_dict 最新大脑、参数和种子，
    独立收集一份长达数千步的节点序列经验，打包返回主进程更新网络。
    """
    worker_id, shared_policy_state_dict, configs, seed = worker_args
    try:
        # [1] Set Safe Environments and Seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        
        # [2] Init Env
        # Domain Randomization: True by default during training to enforce robust schedules
        apply_noise = getattr(configs, 'randomize_durations', True)
        env = AirLineEnv_Graph(data_path=configs.data_file_path, seed=seed)
        state = env.reset(randomize_duration=apply_noise)
        
        # [3] Inference Device: Enable GPU if available, else fallback
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # [4] Initialize Agent 
        model = HBGATPN(configs)
        agent = PPOAgent(
            model=model,
            lr=configs.lr,
            gamma=configs.gamma,
            k_epochs=configs.k_epochs,
            eps_clip=configs.eps_clip,
            device=device,
            batch_size=configs.batch_size
        )
        
        agent.policy.load_state_dict(shared_policy_state_dict)
        agent.policy.to(device)
        agent.policy.eval() # 开启 inference 模式免去 dropout 并在 GAE 选择上也更确定
        
        # [5] Traverse the episode
        memory = Memory()
        ep_reward = 0
        done = False
        max_steps = env.num_tasks * 2 # Safety mechanism
        
        for t in range(max_steps):
            task_mask, station_mask, worker_mask = env.get_masks()
            t_mask = task_mask.to(device)
            s_mask = station_mask.to(device)
            w_mask = worker_mask.to(device)
            
            # Deadlock Check
            if t_mask.all():
                # [Critical Fix] 如果遇到死锁，说明上一步的分配是一个致死动作。
                # 不可以插入全是 True 掩码的 dummy step (会使得 PPO 训练中 Categorical 计算 NaN)，
                # 应当把死锁的巨大惩罚回绝给上一次记录。
                if len(memory.rewards) > 0:
                    memory.rewards[-1] -= 1000.0
                    memory.is_terminals[-1] = True
                    ep_reward -= 1000.0
                else:
                    # 极其罕见: 初始化图本身便为死锁拓扑
                    pass 
                
                done = True
                break
                
            # inference (no grad required explicitly but let agent handle it)
            with torch.no_grad():
                action, logprob, val, _ = agent.select_action(
                    state.to(device), 
                    mask_task=t_mask, 
                    mask_station_matrix=s_mask,
                    mask_worker=w_mask,
                    deterministic=False
                )
            
            next_state, reward, done, info = env.step(action)
            
            memory.states.append(env.get_state_snapshot())
            memory.actions.append(action)
            
            # [CRITICAL] 剥离 GPU 张量到 Python Float以通过 Pipe 安全传输
            if isinstance(logprob, torch.Tensor):
                logprob = logprob.detach().cpu().item()
                
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            memory.masks.append((task_mask.cpu(), station_mask.cpu(), worker_mask.cpu()))
            memory.values.append(val)
            
            state = next_state
            ep_reward += reward
            
            if done:
                break
                
        # [6] Performance Stats
        makespan = env.current_time
        station_loads = env.station_loads
        balance = float(torch.tensor(station_loads, dtype=torch.float).std()) if len(station_loads) > 1 else 0.0
        
        # [7] Pack and send to Surface
        return {
            'worker_id': worker_id,
            'memory': memory,
            'ep_reward': ep_reward,
            'steps': t+1,
            'makespan': makespan,
            'balance': balance,
            'error': None
        }
        
    except Exception as e:
        return {
            'worker_id': worker_id,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
