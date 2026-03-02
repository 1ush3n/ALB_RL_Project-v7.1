import torch
import os
import sys
import traceback

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import AirLineEnv_Graph
from ppo_agent import PPOAgent
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
        
        # [2] Init Env
        # Domain Randomization: True by default during training to enforce robust schedules
        apply_noise = getattr(configs, 'randomize_durations', True)
        env = AirLineEnv_Graph(data_path=configs.data_file_path, seed=seed)
        state = env.reset(randomize_duration=apply_noise)
        
        # [3] Inference Device: Enable GPU if available, else fallback
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # [4] Initialize Agent without optimizers (save Memory) and Load weights
        agent = PPOAgent(configs, env)
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
                reward = -1000.0 
                done = True
                memory.states.append(env.get_state_snapshot())
                memory.actions.append((0,0,[]))
                memory.logprobs.append(0.0) # Float
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                memory.masks.append((task_mask.cpu(), station_mask.cpu(), worker_mask.cpu()))
                memory.values.append(0.0)
                ep_reward += reward
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
        station_loads = [s.workload for s in env.stations]
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
