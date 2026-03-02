
class configs:
    # ------------------
    # 路径配置 (Paths)
    # ------------------
    # 默认数据路径 (如果没有通过命令行参数指定)
    data_file_path = "3000.csv" 
    
    # ------------------
    # 环境与图相关 (Environment & Graph)
    # ------------------
    n_j = 3300                     # 任务(工序)数量估计 (Graph Nodes)
    n_m = 5                        # 站位数量 (Stations)
    n_w = 30                        # 工人数量 (Workers)
                                    # 注意：实际任务数由 data_loader 动态加载，此处仅作参考或 Embedding 初始化上界
    
    # ------------------
    # 模型超参数 (Model Hyperparameters)
    # ------------------
    hidden_dim = 128                # 隐藏层维度 (Embedding Size)
    num_gat_layers = 3              # GAT 层数 (Message Passing Depth)
    num_heads = 4                   # 多头注意力头数 (Attention Heads)
    dropout = 0.1                  # Dropout 比率 (防止过拟合)
    
    # 输入特征维度 (根据 environment.py 中的 _get_observation 定义)
    task_feat_dim = 17              # Task Node Input Features [Duration, Status(4), Type(10), Wait(1), Demand(1)]
    worker_feat_dim = 12            # Worker Node Input Features [Efficiency(1), Skills(10), IsFree(1)]
    station_feat_dim = 8            # Station Node Input Features [Load(1), NumTasks(1), Padding(6)]
    
    # ------------------
    # 泛化性与域随机化 (Domain Randomization)
    # ------------------
    randomize_durations = True      # 是否在训练期间开启工时随机扰动
    dur_random_range = 0.2          # 工时扰动幅度 (0.2 表示基础工时的 ±20% 波动)
    
    # ------------------
    # PPO 训练超参数 (PPO Training)
    # ------------------
    lr = 1e-4                       # 初始学习率 (3000节点序列极长，不可轻易放大以免陷入局部最优死坑)
    gamma = 0.9995                  # [治病良方：时间折现因子] 3000步超级长线，必须将远视能力拉满！(1 / (1-0.9995) = 2000步视野)
    k_epochs = 2                    # 每次更新循环次数 (每次少吸取一点教训，防止把错误的局部真理当做全局真理)
    eps_clip = 0.2                  # PPO Clip阈值 (e.g. 0.1 ~ 0.2)
    batch_size = 16                 # [防 OOM] 严重缩编 Batch Size，避免激活矩阵爆炸!
    
    # [Loss Balancing & Critic Isolation 2026-02-22]
    c_policy = 1.0                  # Policy Loss 权重
    c_value = 0.5                   # [已通过 Huber Loss 防爆] 安全调回标准的 0.5，Critic 不会再破坏全局梯度
    # [2026-02-27] Reduce Entropy to force network out of the random uniform policy (blindness)
    # [针对 3000 单的长序列防死锁补丁] 面对巨量状态，初期随机性非常关键。不可过低。
    c_entropy = 0.05                
    # [Advanced Training Features 2026-02-20]
    accumulation_steps = 128       # [防过拟合核心机制] 在内存中聚集高达 16*128=2048 步全局经验后才做 1 次 PPO Update！严防过快更新导致跌入“死磕前几个节点”的局部最优！
    gae_lambda = 0.98               # GAE 优势函数的衰减因子 (适配 3000 极长序列，将长期优势传导给前置任务)
    use_muon = True                 # 是否使用 Muon 优化器进行 2D 张量的更新
    # [SGDR Learning Rate Schedule]
    sgdr_t0 = 150                   # 针对多节点大图大幅延长重启周期 (150 ep 一个深空潜航)
    
    # [Training Control Parameters 2026-02-12]
    max_episodes = 3000             # 探索万亿级组合的三千大劫
    
    # [Distributed PPO Parallelism 2026-03-01]
    num_workers = 8                 # [Linux 并行参数] 建议 CPU 物理核心数的 50%~80%。您在 RTX5090 + 15+核 机器上可随时调至 8~14 榨除机器性能
    update_every_episodes = 8       # 多少个 Episode 收集一次数据进行 PPO 更新。(必须等于或为 num_workers 的倍数，以便无缝收集轨迹池)
    
    eval_freq = 8                  # 多少个 Episode 进行一次评估 (同步配合 update 频率)
    eval_temperature = 0.0         # 评估/推理时的采样温度 (0.0 表示确定的 Argmax 贪婪策略)
    
    # [Learning Rate Schedule]
    lr_warmup_steps = 3           # 学习率预热步数 (Linear Warmup)
    min_lr = 1e-5                   # 最小学习率 (Cosine Annealing 下界)

    # ------------------
    # 日志与监控 (Logging)
    # ------------------
    log_dir = "tf-logs"                # TensorBoard 日志目录
