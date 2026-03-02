# 基于异构图注意力网络与指针网络的飞机脉动装配线智能调度系统 (HB-GAT-PN)

本项目是一个融合了**深度强化学习 (PPO)**、**异构图注意力机制 (Heterogeneous GAT)** 与 **自回归指针网络 (Pointer Network)** 的工业级调度仿真与智能排产系统。专为解决“工序-站位-工人”三层高度耦合、包含严格拓扑前置约束的飞机装配线调度难题而设计。

本项目在 `v4.6` 基础上，进一步强化了对于极大规模数据的大批量更新稳定性（引入了广义优势估计 GAE、梯度累积以及 Muon 优化器）。

---

## 📂 核心代码目录与架构全景

系统遵循标准的强化学习环境闭环结构：**数据加载** -> **环境仿真 (离散事件引擎)** -> **神经网络前向推断** -> **反向传播与 PPO 策略更新**。

```text
ALB_RL_Project-v4.6/
│
├── configs.py             # 系统的全局控制中枢（超参数）
├── data_loader.py         # 任务工序拓扑与人员信息的统一解析器
├── environment.py         # 基于 OpenAI Gym/Gymnasium 规范构建的离散事件仿真环境
├── ppo_agent.py           # 强化学习 PPO 算法代理（执行采样、GAE计算与优化器更新）
├── train.py               # 模型的主训练循环
├── evaluate.py            # 模型的验证与评测入口
│
├── models/                # 神经网络的核心组件目录
│   └── hb_gat_pn.py       # 包含了从图卷积到自回归多头输出网络架构的设计
│
└── utils/
    └── muon.py            # 牛顿-舒尔茨正交化的高阶动量优化器 (Muon)
```

---

## 📜 脚本级别深度解析

以下是对项目中每个核心脚本的类、主要方法、核心参数的超详细阐述。

### 1. `configs.py` (全局配置文件)
此文件相当于系统的**中央控制台**，所有可调控的数值与超参数皆汇聚于此。由于使用了面向对象模式（定义静态属性的 class），其它脚本统一通过 `from configs import configs` 引用。

* **路径与环境配置** (`data_file_path`, `n_j`, `n_m`, `n_w`): 限定了加载的数据集（如 `3000.csv`）以及网络节点数的理论上限（工序数、站位数、工人数）。
* **模型超参** (`hidden_dim`, `num_gat_layers`, `dropout`): 用于定义图网络层数、多头注意力的头数以及隐藏层表示特征维度。并在下方严格定义了 `task_feat_dim` (17维) 等节点初始态维度。
* **PPO 强化学习参数**:
  * `batch_size`: 推断和更新的最小计算单元。
  * `accumulation_steps` (v4.6+): **梯度累积步长**，解决显存不足导致的小 Batch 剧烈震荡问题，实际更新时的逻辑 Batch 等效于 `batch_size * accumulation_steps`。
  * `gae_lambda`: **广义优势估计衰减因子**，调节长序列任务的偏差与方差。
  * `use_muon`: 是否针对多维全连接层使用前沿的正交化 Muon 优化器。

---

### 2. `environment.py` (环境与仿真核心)
这是整个项目的物理与时间引擎核心。通过继承 `gym.Env` 建立，它不仅要计算节点状态，还负责推进**离散事件仿真 (Discrete Event Simulation)**。

#### `class Event`
* **功能**：用来包裹仿真中即将发生的事件记录字典，支持优先级队列排序。
* **核心属性**：`time` (事件触发时间), `type` (当前只有 `TASK_FINISH`), `data` (包含受到影响的工序、人员和站位ID)。

#### `class AirLineEnv_Graph(gym.Env)`
这是系统内最庞大且最复杂的对象。
* **核心构造与初始化 (`__init__`, `init_hetero_data`)**:
  - `data_path` / `seed` 参数：指定读取文件及其随机种子保证可重复性。
  - 读取数据后，预构建邻接矩阵 (`predecessors`, `successors`)，并使用 `_calculate_cpm()` 通过**关键路径法 (CPM)** 给关键工序上烙印标志（影响阻滞惩罚因子）。
* **核心动作推进 (`step(self, action)`)**:
  - **入参 `action`**: 包含三个部分——`(task_id, station_id, team_list)`。
  - **逻辑**：给定动作后，通过模型 `calculate_duration` 算出这个“具体的人数团队”需要耗时多久。更新当前图网络的状态并将任务完工包装成事件压入 `event_queue`。
  - **奖励反馈 (Reward Design)**:
    - 步进惩罚：耗费工时基础惩罚 `-0.1 * duration`。
    - 关键路径死锁惩罚：如果发现关键工序因为人手短缺被阻塞，施加恶性惩罚（抑制模型滥用高级工人）。
    - 终局奖励：当总任务做完，施加关于 Makespan（最大完工节拍）的极其强烈的线性负向惩罚（首要目标），外加一小部分站位方差（Balance）惩罚。
* **时间引擎流转 (`_advance_time(self)`)**:
  - **原理**：Agent 下达指派后，当全场**无合规任务可分配** 或 **全员均在忙碌被占用** 时，将当前仿真时钟 `current_time` 直接跃迁到 `event_queue` 中时间最早弹出事件的时点。解放完成了任务的工人，刷新工序的 Ready 状态。
* **安全动作掩码发放 (`get_masks(self)`)**:
  - 核心合法性过滤器。保证模型输出的 Logits 被 Softmax 前，非法的动作（如拓扑前置任务没做完的、工人没当前工序技能的）被 Mask 为负无穷大 (`-1e9`)。
* **异构图状态生成 (`_get_observation(self)`)**:
  - 在每个离散时间帧，收集所有状态向量拼装为 `torch_geometric` 格式的 `HeteroData` 字典类型（包含点特征与实时的“任务隶属连边”即动态图建立）。

---

### 3. `ppo_agent.py` (图网络代理执行者)
将 `HBGATPN` 神经网络裹在具有采样和更新能力的智能体中壳层。

#### `class PPOAgent`
* **初始化 (`__init__`)**:
  - 最重要一环是将 Adam 和 **Muon** 优化器分离开：检索出张量大于二维的 Weight（权重矩阵）交给 Muon 处理；1D 或偏置 Bias 则保留高效的 AdamW。
  - 置入了随着 Step 线性 warmup + 余弦退火的混合调度器（`LambdaLR`）。
* **执行采样决策 (`select_action`)**:
  - 输入：前向图观测 `obs`；输出：联合动作和混合 Log Prob。
  - **级联自回归 (Pointer Network 核心体现)**：
    1. **选任务**：通过 Task-Head 在可选拓扑边界中采样一个 `task_id`。
    2. **定站位**：基于选定任务的表达向量，通过 Station-Head 锁定目标 `station_id`。
    3. **自选人员团队**：依照需求额度 (Demand)，不断通过 Worker-Head 一位位地做循环选择。每次选中一名工人后，此工人被当即打上 Mask 标签禁止重复筛选，直至人员配置足额。
    4. 汇总三次决策的 LogProb 用于反馈。
* **网络策略更新 (`update(self, memory)`)**:
  - 第一步：使用**广义优势估计 (GAE)** 对 `Memory` 保存的数据结合时序计算平滑和带有预见性的优势值 (`advantages`)，消除因蒙特卡洛回报带来的震荡。
  - 第二步：通过 `Batch` 张量扩充组建 `DataLoader`，以极小的批次分块推入模型，规避显存爆炸。
  - 第三步：实施 **PPO Surrogate Loss** 函数计算（融合了策略比例截断 Clip 和动作熵鼓励 Entropy）。
  - 第四步：**梯度累积 (Gradient Accumulation)**。将除以累积步长的梯度在本地累加，当收集规模等效于大批次时统一实施优化器 `step()`。大幅抵消在庞大复杂场景下带来的收敛抖动。

---

### 4. `train.py` & `evaluate.py` (执行闭环入口)
#### `class Memory` (位于 `train.py`)
主要属性涵盖：`states`, `actions`, `logprobs`, `rewards`, `is_terminals`, `masks`, `values`。负责按照交互时间轴忠实记录强化学习一条命（Trajectory）内发生的所有前序变量，用于一次性交接给 Agent 去求导学习。

#### 训练主函数 `train()`
1. 实例化环境和 Agent，检查存量断点 (Resume) 以应对意外停机。
2. 内部嵌套经典的**双重 Loop**（外部 `episodes`，内部 `env.steps`）。
3. 当发生 Deadlock（图网络锁死无法派工）时，自动触发负反馈中断机制。
4. 在满足预设论数 `update_every_episodes` 后执行 `agent.update(memory)`。
5. 集成了强健的 TensorBoard 日志流管线及最佳参数的自动快照覆盖。

#### 验证函数 `evaluate_model()` (位于 `evaluate.py`)
将模型从采样探索机制 (Sample) 切入为**确定性选择最高概率贪婪机制 (ArgMax Deterministic)**，以完全抛弃探索的形式测试模型的最强产出上限。

---

### 5. `utils/muon.py` (新型优化器组件)
#### 牛顿-舒尔茨正交计算 `zeropower_via_newtonschulz5`
通过五阶牛顿舒尔茨近似算法计算权重的近似正交矩阵。
#### `class Muon(torch.optim.Optimizer)`
前沿的优化器。相较于一阶梯度 Adam 甚至二阶调参算法，此优化器专注于大模型的**权重正交化处理**以加速神经网络表达能力的跃迁，不仅极速拉高了收敛速度，更能够防止图节点表达在经过多层 GAT 信息传播后的过度平滑崩溃。

---

### 6. `models/hb_gat_pn.py` (网络组件，概念说明)
这是代理使用的“大脑”脑图：
1. **异构图信息传播 (Message Passing)**：将设备负担、人员效率以及任务紧急程度通过点与点之间的边进行相互融合推演。
2. **掩码网络打分头 (Heads)**：将融合后的向量与可选项输入到不同维的 MLP 进行降维。被验证了非法的前馈输入经过 Softmax 前强制降权，在图节点中直接涌现出概率分布用于指导现实调度计划。
