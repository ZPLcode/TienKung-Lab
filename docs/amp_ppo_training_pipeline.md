# TienKung-Lab AMP-PPO 训练完整实现指南

> 本文档详细记录从 `train.py` 入口到机器人学会自然走路的完整训练流程，
> PPO 算法机制和 AMP 对抗运动先验机制并重，作为长期参考。

---

## 目录

1. [算法概览](#1-算法概览)
2. [文件结构与职责](#2-文件结构与职责)
3. [类层级关系](#3-类层级关系)
4. [初始化阶段](#4-初始化阶段)
5. [训练主循环总览](#5-训练主循环总览)
6. [Rollout 数据采集阶段](#6-rollout-数据采集阶段)
7. [GAE 优势计算阶段](#7-gae-优势计算阶段)
8. [网络更新阶段（update）](#8-网络更新阶段update)
9. [PPO 数学公式速查](#9-ppo-数学公式速查)
10. [AMP 数学公式速查](#10-amp-数学公式速查)
11. [超参数完整表](#11-超参数完整表)
12. [训练监控指标](#12-训练监控指标)
13. [完整调用栈](#13-完整调用栈)
14. [关键设计要点](#14-关键设计要点)

---

## 1. 算法概览

### 1.1 PPO（基础算法）

**Proximal Policy Optimization** 解决一个核心问题：

> 给定大量 `(状态, 动作, 奖励)` 经验数据，如何安全有效地更新策略 `π(a|s)`，让"未来累积奖励"最大化？

PPO 的三大核心机制：

1. **GAE 优势函数**：评估"这个动作比平均水平好多少"
2. **Surrogate Loss + Clip**：限制策略单步更新幅度
3. **Value Bootstrap + 自适应 KL**：稳定 Critic 估值与学习率

### 1.2 AMP（对抗运动先验扩展）

**Adversarial Motion Priors** 在 PPO 基础上增加：

> 让机器人在完成任务的同时，**步态像真人**（基于动捕数据训练判别器，提供风格奖励）。

```
普通 PPO：只追求"走得快、不摔倒"     → 步态机械、抽搐
AMP-PPO ：追求"走得快、不摔倒、像人" → 步态自然
```

### 1.3 PPO 和 AMP 的分工

```
┌──────────────────────────────────────────────────────┐
│                  机器人最终表现                       │
└────────────────┬─────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐   ┌──────────────┐
│  "做什么"    │   │  "怎么做"    │
│  PPO 负责    │   │  AMP 负责    │
├──────────────┤   ├──────────────┤
│ 速度跟踪      │   │ 步态自然     │
│ 不摔倒        │   │ 不抽搐       │
│ 22 项手设奖励 │   │ 模仿专家     │
│ Actor 优化   │   │ 判别器对抗   │
└──────────────┘   └──────────────┘
       │                  │
       └────────┬─────────┘
                ▼
         0.7 × task + 0.3 × amp
                │
                ▼
        Actor 学习方向（PPO 驱动）
```

**关键事实**：AMP 不直接训练 Actor，它只是修改 Actor 看到的 reward。Actor 始终由 PPO 的 `surrogate_loss` 通过 GAE 来更新。

---

## 2. 文件结构与职责

```
TienKung-Lab/
│
├── legged_lab/                                # 环境与训练脚本
│   ├── scripts/
│   │   ├── train.py                           # 入口：解析配置 → 启动 Runner.learn()
│   │   ├── play.py                            # 推理回放
│   │   └── sim2sim.py                         # MuJoCo Sim2Sim 验证
│   ├── envs/
│   │   ├── __init__.py                        # 任务注册表（walk→AMP, walk_ppo→纯PPO）
│   │   ├── base/base_env.py                   # 环境基类
│   │   └── tienkung/
│   │       ├── tienkung_env.py                # ★ 环境核心（含 amp_obs/terminal_amp_obs）
│   │       ├── walk_cfg.py                    # AMP 任务配置
│   │       ├── walk_ppo_cfg.py                # 纯 PPO 配置
│   │       └── datasets/motion_amp_expert/    # 专家动捕数据（walk.txt/run.txt）
│   └── mdp/rewards.py                         # 22 项任务奖励函数
│
└── rsl_rl/rsl_rl/                             # RL 算法库
    ├── runners/
    │   ├── on_policy_runner.py                # 纯 PPO 训练主循环
    │   └── amp_on_policy_runner.py            # ★ AMP-PPO 训练主循环
    ├── algorithms/
    │   ├── ppo.py                             # PPO 算法
    │   └── amp_ppo.py                         # ★ AMP-PPO 算法
    ├── modules/
    │   ├── actor_critic.py                    # Actor-Critic 网络（策略）
    │   └── discriminator.py                   # ★ 判别器网络 + AMP 奖励计算
    ├── storage/
    │   ├── rollout_storage.py                 # PPO 经验缓存（每轮清空）
    │   └── replay_buffer.py                   # ★ AMP 策略历史缓存（跨轮保留）
    └── utils/
        ├── motion_loader.py                   # ★ AMPLoader：加载专家动捕数据
        └── normalizer.py                      # AMP 状态在线归一化器（Welford）
```

★ = AMP 相比 PPO 新增/修改的核心文件

---

## 3. 类层级关系

```
AmpOnPolicyRunner
├── env: TienKungEnv                  # 环境
├── alg: AMPPPO                       # 算法主体
│   ├── policy: ActorCritic           # 策略网络
│   │   ├── actor: MLP(750→512→256→128→20)   # 输出动作均值
│   │   ├── std:  nn.Parameter(20)            # 可学习标准差
│   │   └── critic: MLP(800→512→256→128→1)   # 输出状态价值
│   ├── discriminator: Discriminator  # [AMP] 判别器
│   │   ├── trunk:      MLP(104→1024→512→256)
│   │   └── amp_linear: Linear(256→1)
│   ├── amp_data: AMPLoader           # [AMP] 专家数据加载器
│   ├── amp_normalizer: Normalizer    # [AMP] 状态归一化器
│   ├── amp_storage: ReplayBuffer     # [AMP] 策略历史缓存
│   ├── storage: RolloutStorage       # PPO 经验缓存
│   ├── transition: Transition        # 临时存当前步 PPO 数据
│   ├── amp_transition: Transition    # [AMP] 临时存当前步 AMP 数据
│   └── optimizer: Adam               # 3 个参数组
│       ├── policy.parameters()                  # 无 weight_decay
│       ├── discriminator.trunk    weight_decay=1e-3   # 轻正则
│       └── discriminator.amp_linear weight_decay=1e-1 # 重正则
├── obs_normalizer / privileged_obs_normalizer  # 观测归一化
└── writer: SummaryWriter             # TensorBoard 日志
```

---

## 4. 初始化阶段

`AmpOnPolicyRunner.__init__` 完成以下工作：

```python
1. 解析配置 train_cfg（来自 walk_cfg.py）

2. 创建 ActorCritic 策略网络
   ├── Actor:  MLP(750→512→256→128→20)  输出动作均值
   ├── std:    nn.Parameter([0.8]×20)    可学习标准差
   └── Critic: MLP(800→512→256→128→1)   输出状态价值

3. [AMP新增] 创建三个 AMP 组件
   ├── amp_data = AMPLoader(walk.txt, ...)
   │     └── 预加载 100万对 (s_t, s_{t+1}) 到 GPU
   ├── amp_normalizer = Normalizer(52)
   │     └── 在线均值-方差统计器（Welford 算法）
   └── discriminator = Discriminator(input_dim=104, ...)
         ├── trunk:      MLP(104→1024→512→256)，ReLU 激活
         └── amp_linear: Linear(256→1)，输出标量分数 d

4. 创建 AMPPPO 算法实例
   └── 在 AMPPPO.__init__ 中：
       ├── 创建 Adam 优化器（3 个参数组，不同 weight_decay）
       └── 创建 ReplayBuffer(obs_dim=52, buffer_size=100000)

5. init_storage：分配 RolloutStorage GPU 内存
   └── shape = (num_steps=24, num_envs=4096, ...)

6. 多 GPU 配置（如果有）
```

---

## 5. 训练主循环总览

```python
for it in range(num_learning_iterations):   # 主迭代循环（如 50000 轮）

    # ─────── A. Rollout 采样阶段 ───────
    with torch.inference_mode():
        for step in range(24):              # 每轮采 24 步
            采样一步并存入 storage

        compute_returns(last_critic_obs)    # GAE 计算优势

    # ─────── B. 网络更新阶段 ───────
    loss_dict = alg.update()                # 5 epochs × 4 mini-batches = 20 次

    # ─────── C. 日志与存盘 ───────
    log(...)
    if it % save_interval == 0: save(...)
```

每轮迭代采样 `24 × 4096 = 98304` 条经验，并用其更新网络 20 次。

---

## 6. Rollout 数据采集阶段

### 6.1 PPO 基础数据采集

每步采集的 `transition`：

| 字段 | 维度 | 含义 |
|--|--|--|
| `observations` | 750 | Actor 观测（含 10 步历史堆叠） |
| `privileged_obs` | 800 | Critic 特权观测（含基座线速度等） |
| `actions` | 20 | 当前步动作（从高斯分布采样） |
| `rewards` | 1 | 标量奖励 |
| `dones` | 1 | 0/1 是否重置 |
| `values` | 1 | Critic 当前估值 V(s) |
| `actions_log_prob` | 1 | 旧策略对数概率 log π(a\|s) |
| `action_mean` | 20 | 动作分布均值（用于 KL 计算） |
| `action_sigma` | 20 | 动作分布标准差 |

**为什么要存 `log π` 和 `V(s)`？**

PPO 是 **off-policy 修正的 on-policy 算法**——同一批数据要用 5 个 epoch 反复训练。在第 2、3 epoch 时策略已经变了，需要用"采样时的旧概率"做重要性采样修正。

### 6.2 AMP 额外采集

[AMP] 额外需要：

| 字段 | 维度 | 含义 |
|--|--|--|
| `amp_obs` | 52 | AMP 状态（关节角度+速度+末端位置） |
| `next_amp_obs` | 52 | 下一帧 AMP 状态 |

### 6.3 Rollout 一步的完整流程

```python
# === 步骤 1: 策略前向，决定动作 ===
actions = alg.act(obs, privileged_obs, amp_obs)
    │
    └── AMPPPO.act() 内部：
        ├── actions = policy.act(obs)                # Actor 采样动作
        ├── values  = policy.evaluate(critic_obs)    # Critic 估计价值
        ├── log_prob = policy.get_actions_log_prob(actions)
        ├── 存到 transition: obs, action, value, log_prob, mean, std
        └── [AMP] 存到 amp_transition: amp_obs (作为下一步的 s_t)

# === 步骤 2: 环境前进一步 ===
obs, rewards, dones, infos = env.step(actions)
    │
    └── env.step() 内部：
        ├── 4 次物理微步（5ms × 4 = 20ms）
        ├── 计算 task_rewards（22 项奖励之和）
        ├── 检查 done（摔倒/超时）
        ├── [AMP] 保存 terminal_amp_obs[reset_ids]   ← 在 reset 之前！
        ├── reset(reset_ids)                          ← 重置环境
        └── 返回新的 obs（reset 后的）

# === 步骤 3: [AMP] 获取下一步 AMP 状态 ===
next_amp_obs = env.get_amp_obs_for_expert_trans()  # 注意：是 reset 后的状态

# === 步骤 4: [AMP] 终态修复 ===
next_amp_obs_with_term = clone(next_amp_obs)
if len(reset_env_ids) > 0:
    next_amp_obs_with_term[reset_env_ids] = env.terminal_amp_obs[reset_env_ids]
    # ↑ 用 reset 之前保存的真实终态，替换 reset 后的初始姿态
    #   防止判别器学到 (走路中 → 跳变到初始姿态) 的假转移

# === 步骤 5: [AMP] 计算 AMP 奖励，替换原始奖励 ===
rewards = discriminator.predict_amp_reward(amp_obs, next_amp_obs_with_term, rewards)[0]
    │
    └── predict_amp_reward 内部（no_grad）：
        ├── 归一化 state, next_state
        ├── d = amp_linear(trunk(cat([s, s'])))    # 判别器前向
        ├── amp_reward = 0.3 × clamp(1 - 0.25×(d-1)², min=0)
        └── final_reward = 0.7 × task_reward + 0.3 × amp_reward

# === 步骤 6: 更新下一轮起始 s_t ===
amp_obs = clone(next_amp_obs)
    # 注意：用**未修复**的 next_amp_obs（reset 后的初始姿态）
    # 因为下一步的 s_t 应该是环境中真实存在的状态

# === 步骤 7: 数据入库 ===
alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
    │
    └── 内部：
        ├── transition.rewards = rewards
        ├── 超时自举: rewards += γ × V(s) × timeout
        ├── [AMP] amp_storage.insert(amp_transition.obs, amp_obs_with_term)
        │         └── 存入 ReplayBuffer 循环覆盖
        ├── storage.add_transitions(transition)
        │         └── 存入 RolloutStorage 顺序写入
        └── 清空 transition 和 amp_transition
```

### 6.4 终态修复的必要性

**问题**：`env.step()` 内部已完成 reset，`get_amp_obs_for_expert_trans()` 返回的是新回合的初始姿态。

**如果不修复**，判别器会学到一堆 `(走路中 → 跳变到初始姿态)` 的**假转移**，严重污染训练。

**解决**：env 在 reset 之前先保存终态到 `terminal_amp_obs`：

```python
# tienkung_env.py
self.reset_env_ids = self.reset_buf.nonzero(...).flatten()
if len(self.reset_env_ids) > 0:
    self.terminal_amp_obs[self.reset_env_ids] = self.get_amp_obs_for_expert_trans()[...]  # 先保存
self.reset(self.reset_env_ids)  # 后重置
```

Runner 端读取这个保存的终态：

```python
# amp_on_policy_runner.py
next_amp_obs_with_term[reset_env_ids] = env.terminal_amp_obs[reset_env_ids]
```

### 6.5 两个 amp_obs 变量的物理含义区分

| 变量 | 用途 | 应该用哪个 |
|--|--|--|
| `next_amp_obs_with_term` | 喂给判别器 / ReplayBuffer | **修复后的**（含终态） |
| `amp_obs`（下一轮 s_t） | 下一次 `act()` 时的起始状态 | **未修复的**（reset 后初始姿态） |

下一轮的 `s_t` 必须是**真实存在于环境中的状态**，否则 `act()` 会基于"已死亡机器人"做决策。

---

## 7. GAE 优势计算阶段

每轮采样结束后调用 `alg.compute_returns(last_critic_obs)`，从后往前倒序计算优势 `A_t` 和折扣回报 `R_t`：

```python
A_{T} = 0
for t in reversed(range(T)):
    δ_t = r_t + γ × V(s_{t+1}) × (1 - d_t) - V(s_t)   # TD 误差
    A_t = δ_t + γ × λ × (1 - d_t) × A_{t+1}            # GAE 累积
    R_t = A_t + V(s_t)                                   # 折扣回报
```

### 7.1 核心思想

- `V(s_t)`：Critic 拍脑袋估的"这个状态值多少"
- `r_t + γV(s_{t+1})`：实际拿到的奖励 + 下一状态的估值（更接近真相的估计）
- `δ_t`：两者差距，叫 **TD 误差**
- **优势 A_t**：多个 TD 误差按 λ 衰减加权累积，平衡偏差和方差

### 7.2 λ 的作用

| λ 值 | 等价形式 | 偏差 | 方差 |
|--|--|--|--|
| 0 | 纯 TD：`A_t = δ_t` | 大 | 小 |
| 1 | 纯 Monte Carlo：`A_t = ∑γᵏ rₜ₊ₖ - V(sₜ)` | 0 | 大 |
| 0.95 | GAE | 折中 | 折中 |

### 7.3 A_t 的直观含义

- `A_t > 0`：这个动作比 Critic 平均预期好 → 鼓励
- `A_t < 0`：这个动作比预期差 → 抑制

`R_t` 作为 Critic 的训练目标（让 V(s_t) → R_t）。

### 7.4 超时自举（Bootstrap on Timeout）

```python
# 在 process_env_step 中：
if "time_outs" in infos:
    transition.rewards += gamma × V(s_t) × timeout
```

**为什么需要？**

环境有两种 done：
- **失败 done**（摔倒）：未来真的没有奖励了，r = -200 penalty
- **超时 done**（达到 1000 步）：未来本来还有奖励，只是因为时间限制结束

如果对超时也设 done=1，Critic 会以为"未来价值=0"，**学到错误的悲观估值**。超时时把 `γ × V(s)` 加回 reward，等价于告诉 Critic："未来价值是 V(s)，别忘了估它"。

---

## 8. 网络更新阶段（update）

### 8.1 创建三路 mini-batch Generator

```python
# PPO 数据 generator
generator = storage.mini_batch_generator(num_mini_batches=4, num_learning_epochs=5)
# 总共生成 20 个 mini-batch

# [AMP] 策略历史 generator
amp_policy_gen = amp_storage.feed_forward_generator(
    20,                                     # 总 mini-batch 数
    98304 // 4                              # 每个 batch size
)

# [AMP] 专家数据 generator
amp_expert_gen = amp_data.feed_forward_generator(20, 98304 // 4)
```

三路 generator 用 `zip()` **同步迭代**，每次取出等量数据。

### 8.2 一个 mini-batch 内的完整流程

```python
for sample, sample_amp_policy, sample_amp_expert in zip(...):

    # ═══════════════════════════════════════════════════
    #                  PPO 部分（核心）
    # ═══════════════════════════════════════════════════

    # --- 8.2.1 解包 rollout 数据 ---
    obs_batch, critic_obs_batch, actions_batch,
    target_values_batch, advantages_batch, returns_batch,
    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
    hid_states_batch, masks_batch, rnd_state_batch = sample

    # --- 8.2.2 优势归一化（若启用 per_mini_batch） ---
    if normalize_advantage_per_mini_batch:
        advantages_batch = (advantages_batch - mean) / (std + 1e-8)

    # --- 8.2.3 对称数据增强（若启用） ---
    if symmetry["use_data_augmentation"]:
        obs_batch, actions_batch = mirror(obs_batch, actions_batch)
        critic_obs_batch = mirror(critic_obs_batch)
        # 复制相关张量保持维度对齐
        target_values_batch = target_values_batch.repeat(2, 1)
        advantages_batch = advantages_batch.repeat(2, 1)
        ...

    # --- 8.2.4 最新策略前向 ---
    policy.act(obs_batch)
    actions_log_prob_batch = policy.get_actions_log_prob(actions_batch)
    value_batch = policy.evaluate(critic_obs_batch)
    mu_batch = policy.action_mean[:original_batch_size]
    sigma_batch = policy.action_std[:original_batch_size]
    entropy_batch = policy.entropy[:original_batch_size]

    # --- 8.2.5 自适应 KL 调整学习率 ---
    kl = compute_kl(new_dist, old_dist)
    if kl > 2 × desired_kl: lr /= 1.5
    elif kl < 0.5 × desired_kl: lr *= 1.5
    update_optimizer_lr(lr)

    # --- 8.2.6 Surrogate Loss（Clip）---
    ratio = exp(actions_log_prob_batch - old_actions_log_prob_batch)
    surrogate         = -A × ratio
    surrogate_clipped = -A × clip(ratio, 1-ε, 1+ε)
    surrogate_loss    = max(surrogate, surrogate_clipped).mean()

    # --- 8.2.7 Value Loss（Clipped MSE）---
    if use_clipped_value_loss:
        V_clipped = V_old + clip(V_new - V_old, -ε, ε)
        value_loss = max((V_new - R)², (V_clipped - R)²).mean()
    else:
        value_loss = ((V_new - R)²).mean()

    # --- 8.2.8 合并 PPO 总损失 ---
    loss = surrogate_loss + c1 × value_loss - c2 × entropy

    # --- 8.2.9 对称镜像损失（若启用） ---
    if symmetry["use_mirror_loss"]:
        mirror_actions = policy.act_inference(mirror(obs))
        symmetry_loss = MSE(mirror_actions, mirror(actions_orig))
        loss += mirror_loss_coeff × symmetry_loss

    # --- 8.2.10 RND 好奇心损失（若启用） ---
    if rnd:
        rnd_loss = MSE(rnd.predictor(s), rnd.target(s).detach())

    # ═══════════════════════════════════════════════════
    #                  AMP 部分（新增）
    # ═══════════════════════════════════════════════════

    # --- 8.2.11 解包 AMP 数据 ---
    policy_state, policy_next_state = sample_amp_policy
    expert_state, expert_next_state = sample_amp_expert

    # --- 8.2.12 归一化 AMP 状态（no_grad）---
    if amp_normalizer is not None:
        with torch.no_grad():
            policy_state      = amp_normalizer.normalize(policy_state)
            policy_next_state = amp_normalizer.normalize(policy_next_state)
            expert_state      = amp_normalizer.normalize(expert_state)
            expert_next_state = amp_normalizer.normalize(expert_next_state)

    # --- 8.2.13 判别器前向 ---
    policy_d = discriminator(cat([policy_state, policy_next_state]))
    expert_d = discriminator(cat([expert_state, expert_next_state]))

    # --- 8.2.14 LSGAN 损失 ---
    expert_loss = MSE(expert_d, +1)    # 专家数据 → 目标分 +1
    policy_loss = MSE(policy_d, -1)    # 策略数据 → 目标分 -1
    amp_loss    = 0.5 × (expert_loss + policy_loss)

    # --- 8.2.15 梯度惩罚 ---
    grad_pen_loss = discriminator.compute_grad_pen(
        expert_state, expert_next_state, lambda_=10
    )

    # --- 8.2.16 加入总 Loss ---
    loss += amploss_coef × amp_loss + amploss_coef × grad_pen_loss

    # ═══════════════════════════════════════════════════
    #                反向传播与更新
    # ═══════════════════════════════════════════════════

    # --- 8.2.17 一次 backward 更新所有网络 ---
    optimizer.zero_grad()
    loss.backward()
    # 梯度流向：
    #   surrogate_loss → Actor
    #   value_loss     → Critic
    #   amp_loss       → Discriminator（不到 Actor，因为 policy_state 已 detach）
    #   grad_pen       → Discriminator

    # --- 8.2.18 梯度裁剪 ---
    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm=1.0)

    # --- 8.2.19 优化器更新 ---
    optimizer.step()
    # ↑ 同时更新 Actor + Critic + Discriminator trunk + Discriminator amp_linear

    # --- 8.2.20 [AMP] 更新归一化统计 ---
    amp_normalizer.update(policy_state)
    amp_normalizer.update(expert_state)

# ─────── 所有 mini-batch 处理完后 ───────
storage.clear()    # 清空 RolloutStorage（ReplayBuffer 不清！）

return loss_dict   # 含 PPO 和 AMP 所有损失指标
```

### 8.3 梯度流向澄清

**关键事实**：`amp_loss` 的梯度**不会**流回 Actor。

为什么？看 ReplayBuffer 里存的是什么：

```python
self.amp_storage.insert(self.amp_transition.observations, amp_obs)
```

存入的是从环境直接获取的物理观测，**从来没有经过 Actor 网络**——它们来自仿真器读出的关节角度、速度等。所以 ReplayBuffer 里的 `policy_state` 和 Actor 参数**没有任何计算图连接**。

```python
policy_state = ReplayBuffer 里取出的物理观测   # 和 Actor 参数无关
policy_d = discriminator(cat([policy_state, ...]))  # 只过判别器
amp_loss.backward()  # 梯度只回到 discriminator.parameters()
```

**Actor 怎么"知道"判别器的意见？** 通过 reward 间接传递：

```
Rollout: discriminator.predict_amp_reward() → amp_reward
         ↓
         混合 reward = 0.7×task + 0.3×amp
         ↓
         存入 RolloutStorage.rewards（已脱离计算图）
         ↓
Update:  compute_returns() → GAE → A_t
         ↓
         surrogate_loss = -A × ratio → Actor 梯度
```

**Actor 和 Discriminator 是松耦合的两个独立训练系统**，通过 reward 数值传递信号。

---

## 9. PPO 数学公式速查

### 9.1 重要性采样比

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} = \exp\big(\ln\pi_\theta(a_t|s_t) - \ln\pi_{\theta_{\text{old}}}(a_t|s_t)\big)$$

### 9.2 Surrogate Loss（PPO Clip）

$$\mathcal{L}_{\text{surr}}(\theta) = \hat{\mathbb{E}}_t \left[ \max\big( -r_t(\theta)\hat{A}_t,\ -\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \big) \right]$$

### 9.3 Value Loss（Clipped MSE）

$$V_{\phi}^{\text{clip}}(s_t) = V_{\phi_{\text{old}}}(s_t) + \text{clip}\big(V_\phi(s_t) - V_{\phi_{\text{old}}}(s_t), -\epsilon, \epsilon\big)$$

$$\mathcal{L}_{\text{value}}(\phi) = \hat{\mathbb{E}}_t \left[ \max\big( (V_\phi(s_t) - R_t)^2, (V_{\phi}^{\text{clip}}(s_t) - R_t)^2 \big) \right]$$

### 9.4 Entropy 正则

$$\mathcal{L}_{\text{entropy}}(\theta) = -\hat{\mathbb{E}}_t \big[ \mathcal{H}(\pi_\theta(\cdot|s_t)) \big]$$

$$\mathcal{H}(\pi) = \frac{1}{2}\sum_{j=1}^{D}\big(1 + \ln(2\pi\sigma_j^2)\big)$$

### 9.5 GAE 优势

$$\delta_t = r_t + \gamma V_{\phi_{\text{old}}}(s_{t+1})(1-d_t) - V_{\phi_{\text{old}}}(s_t)$$

$$\hat{A}_t = \delta_t + (\gamma\lambda)(1-d_t)\hat{A}_{t+1}$$

$$R_t = \hat{A}_t + V_{\phi_{\text{old}}}(s_t)$$

### 9.6 自适应 KL

$$\text{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) = \sum_j \left[ \ln\frac{\sigma_j}{\sigma_{j,\text{old}}} + \frac{\sigma_{j,\text{old}}^2 + (\mu_{j,\text{old}} - \mu_j)^2}{2\sigma_j^2} - \frac{1}{2} \right]$$

调整规则：
- $\text{KL} > 2 \cdot \text{KL}_{\text{desired}}$：$\eta \leftarrow \eta / 1.5$
- $\text{KL} < 0.5 \cdot \text{KL}_{\text{desired}}$：$\eta \leftarrow \eta \cdot 1.5$

### 9.7 PPO 总损失

$$\mathcal{L}_{\text{PPO}} = \mathcal{L}_{\text{surr}} + c_1 \mathcal{L}_{\text{value}} - c_2 \mathcal{H}(\pi)$$

---

## 10. AMP 数学公式速查

### 10.1 判别器损失（LSGAN）

$$\mathcal{L}_{\text{amp}} = \frac{1}{2}\hat{\mathbb{E}}_{(s,s')\sim\text{expert}}\big[(D(s,s') - 1)^2\big] + \frac{1}{2}\hat{\mathbb{E}}_{(s,s')\sim\pi}\big[(D(s,s') + 1)^2\big]$$

**为什么用 LSGAN 而非 BCE？**

| | BCE（传统 GAN） | LSGAN（AMP 使用） |
|--|--|--|
| 输出激活 | Sigmoid | 无（裸线性） |
| 损失 | Binary Cross-Entropy | MSE |
| 梯度饱和 | 判别器自信时梯度→0 | 梯度始终为 `2(d-target)` |
| 训练稳定性 | 易死锁 | 稳定 |

### 10.2 梯度惩罚

$$\mathcal{L}_{\text{gp}} = \lambda \cdot \hat{\mathbb{E}}_{(s,s')\sim\text{expert}}\big[\|\nabla_x D(x)\|_2^2\big], \quad x = [s, s']$$

**作用**：强迫判别器在专家数据附近的输出曲面平滑，防止判别器过强导致策略梯度消失。

### 10.3 AMP 奖励

$$d = D(s_t, s_{t+1})$$

$$r_{\text{amp}} = c_{\text{amp}} \cdot \max\left(1 - \frac{1}{4}(d - 1)^2, \ 0\right)$$

其中 $c_{\text{amp}} = 0.3$ 是 AMP 奖励系数。

**奖励映射表：**

| d 值 | amp_reward | 含义 |
|--|--|--|
| +1 | 0.30（满分） | 走路完全像专家 |
| 0 | 0.225 | 中间状态 |
| -1 | 0 | 策略初期乱动 |
| < -1 或 > +3 | 0（被 clamp） | 防止过拟合 |

### 10.4 最终奖励混合

$$r_{\text{total}} = (1 - \alpha) \cdot r_{\text{amp}} + \alpha \cdot r_{\text{task}}$$

其中 $\alpha = 0.7$ 是 `task_reward_lerp`：

$$r_{\text{total}} = 0.7 \cdot r_{\text{task}} + 0.3 \cdot r_{\text{amp}}$$

### 10.5 AMP-PPO 总损失

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{surr}} + c_1 \mathcal{L}_{\text{value}} - c_2 \mathcal{H}}_{\text{PPO}} + \underbrace{c_{\text{amploss}} \cdot \mathcal{L}_{\text{amp}} + c_{\text{amploss}} \cdot \mathcal{L}_{\text{gp}}}_{\text{AMP}}$$

其中 $c_{\text{amploss}} = 1.0$。

---

## 11. 超参数完整表

### 11.1 环境参数

| 参数 | 值 | 含义 |
|--|--|--|
| `num_envs` | 4096 | 并行环境数 |
| `physics_dt` | 0.005s | 物理仿真步长（200Hz） |
| `step_dt` | 0.02s | 控制步长（50Hz, decimation=4） |
| `episode_length_s` | 20s | 最大回合长度 |
| `action_scale` | 0.25 | 动作缩放（输出 × 0.25 + default_pos） |

### 11.2 PPO 超参数

| 参数 | 值 | 出处 | 含义 |
|--|--|--|--|
| `num_steps_per_env` | 24 | walk_cfg | 每轮采样步长 |
| `num_learning_epochs` | 5 | walk_cfg | 数据重复利用次数 |
| `num_mini_batches` | 4 | walk_cfg | mini-batch 切分数 |
| `clip_param` | 0.2 | walk_cfg | PPO 裁剪系数 ε |
| `gamma` | 0.99 | walk_cfg | 折扣率 |
| `lam` | 0.95 | walk_cfg | GAE λ |
| `value_loss_coef` | 1.0 | walk_cfg | c1 |
| `entropy_coef` | 0.005 | walk_cfg | c2 |
| `learning_rate` | 1e-3 | walk_cfg | 初始学习率 |
| `max_grad_norm` | 1.0 | walk_cfg | 梯度裁剪阈值 |
| `desired_kl` | 0.01 | walk_cfg | 目标 KL 散度 |
| `schedule` | "adaptive" | walk_cfg | 学习率调度方式 |
| `use_clipped_value_loss` | True | walk_cfg | 是否裁剪 Value Loss |

### 11.3 AMP 超参数

| 参数 | 值 | 出处 | 含义 |
|--|--|--|--|
| `amp_task_reward_lerp` | 0.7 | walk_cfg | 任务奖励占比 |
| `amp_reward_coef` | 0.3 | walk_cfg | AMP 奖励缩放系数 |
| `amp_discr_hidden_dims` | [1024, 512, 256] | walk_cfg | 判别器隐藏层维度 |
| `amp_replay_buffer_size` | 100000 | walk_cfg | ReplayBuffer 容量 |
| `amp_num_preload_transitions` | 2000000 | walk_cfg | 专家数据预采样数 |
| `amp_motion_files` | [walk.txt] | walk_cfg | 专家动捕文件 |
| `grad_pen λ` | 10 | amp_ppo.py | 梯度惩罚系数 |
| `amploss_coef` | 1.0 | amp_ppo.py | AMP loss 总系数 |
| `discr trunk weight_decay` | 1e-3 | amp_ppo.py | 判别器主干 L2 正则 |
| `discr amp_linear weight_decay` | 1e-1 | amp_ppo.py | 判别器输出层 L2 正则 |
| `min_normalized_std` | [0.05]×20 | walk_cfg | 动作标准差下界 |

### 11.4 观测维度

| 项 | 维度 |
|--|--|
| 单帧 Actor 观测 | 75 |
| Actor 总观测（10步堆叠） | 750 |
| 单帧 Critic 观测 | 80 |
| Critic 总观测（10步堆叠） | 800 |
| AMP 状态 | 52 |
| 判别器输入（拼接两帧） | 104 |
| 动作维度 | 20 |

---

## 12. 训练监控指标

通过 TensorBoard 监控以下指标：

### 12.1 PPO 指标

| 指标 | 健康表现 | 说明 |
|--|--|--|
| `Loss/surrogate` | 围绕 0 小幅震荡 | Actor 损失 |
| `Loss/value_function` | 持续下降然后稳定 | Critic 损失 |
| `Loss/entropy` | 缓慢下降 | 策略逐渐收敛 |
| `Loss/learning_rate` | 在 [1e-5, 1e-2] 自适应 | 自适应 KL 调节 |
| `Policy/mean_noise_std` | 从初始 0.8 缓慢减小 | 探索逐渐降低 |
| `Train/mean_reward` | 持续上升 | 平均回合奖励 |
| `Train/mean_episode_length` | 上升到接近 1000 | 平均生存步数 |

### 12.2 AMP 指标

| 指标 | 健康表现 | 说明 |
|--|--|--|
| `Loss/amp` | 先下降后稳定 | 判别器 LSGAN 损失 |
| `Loss/amp_grad_pen` | 较小且稳定 | 梯度惩罚损失 |
| `Loss/amp_expert_pred` | 稳定在 +1 附近 | 判别器始终能认出专家 |
| `Loss/amp_policy_pred` | 从 -1 上升到 +1 | **核心指标**：策略越来越像专家 |

### 12.3 Episode 指标

由 env 的 reward 函数自动写入 `Episode/` 命名空间：

- 各项奖励函数的均值（如 `Episode/track_lin_vel`, `Episode/feet_air_time`）
- 失败原因统计（摔倒次数、超时次数）

---

## 13. 完整调用栈

```
train.py
  └── runner.learn(num_learning_iterations=50000)
        └── for it in range(50000):
              │
              ├── ▼▼▼ A. Rollout 采样 ▼▼▼
              │   with torch.inference_mode():
              │     for _ in range(num_steps_per_env=24):
              │       │
              │       ├── alg.act(obs, critic_obs, amp_obs)
              │       │     ├── policy.act(obs)                # Actor 采样
              │       │     ├── policy.evaluate(critic_obs)    # Critic 估值
              │       │     ├── policy.get_actions_log_prob()
              │       │     └── 存 transition + amp_transition
              │       │
              │       ├── env.step(actions)
              │       │     ├── for _ in range(decimation=4):
              │       │     │     ├── 应用关节目标
              │       │     │     ├── sim.step() (5ms)
              │       │     │     └── 累加足部力/速度
              │       │     ├── 更新 episode_length
              │       │     ├── 更新步态相位 gait_phase
              │       │     ├── 计算 task rewards (22项)
              │       │     ├── 检查 done (摔倒/超时)
              │       │     ├── [AMP] 保存 terminal_amp_obs
              │       │     ├── reset 死亡环境
              │       │     └── 计算新 obs
              │       │
              │       ├── [AMP] env.get_amp_obs_for_expert_trans()
              │       │
              │       ├── [AMP] 终态修复:
              │       │     next_amp_obs_with_term[reset_ids] =
              │       │         env.terminal_amp_obs[reset_ids]
              │       │
              │       ├── [AMP] alg.discriminator.predict_amp_reward()
              │       │     ├── 归一化
              │       │     ├── d = trunk + amp_linear
              │       │     ├── amp_reward = 0.3 × clamp(...)
              │       │     └── mixed_reward = 0.7×task + 0.3×amp
              │       │
              │       └── alg.process_env_step(rewards, dones, infos, amp_obs)
              │             ├── transition.rewards = rewards
              │             ├── 超时自举: rewards += γ×V(s)×timeout
              │             ├── [AMP] amp_storage.insert(s_t, s_{t+1})
              │             ├── storage.add_transitions(transition)
              │             └── 清空 transition + amp_transition
              │
              │   ├── alg.compute_returns(last_critic_obs)
              │   │     └── storage.compute_returns(γ, λ)
              │   │           └── 倒序计算 A_t 和 R_t (GAE)
              │
              ├── ▼▼▼ B. 网络更新 ▼▼▼
              │   loss_dict = alg.update()
              │     │
              │     ├── 创建 3 路 generator (rollout / amp_policy / amp_expert)
              │     │
              │     └── for sample, amp_policy, amp_expert in zip(...): # 20 次
              │           │
              │           ├── 解包 sample (12 个张量)
              │           │
              │           ├── 优势归一化（若 per_mini_batch）
              │           │
              │           ├── 对称数据增强（若启用）
              │           │
              │           ├── policy.act(obs_batch)
              │           ├── policy.get_actions_log_prob()
              │           ├── policy.evaluate(critic_obs_batch)
              │           │
              │           ├── 自适应 KL 调 LR
              │           │     ├── 计算 KL 散度
              │           │     ├── if KL > 2×desired: lr /= 1.5
              │           │     ├── elif KL < 0.5×desired: lr *= 1.5
              │           │     └── 更新 optimizer.lr
              │           │
              │           ├── 算 surrogate_loss (PPO Clip)
              │           ├── 算 value_loss (Clipped MSE)
              │           ├── 算 entropy
              │           ├── loss = surrogate + c1×value - c2×entropy
              │           │
              │           ├── 对称镜像损失（若启用）
              │           ├── RND 好奇心损失（若启用）
              │           │
              │           ├── [AMP] 解包 amp_policy + amp_expert
              │           ├── [AMP] 归一化 4 个 amp 状态（no_grad）
              │           ├── [AMP] discriminator(policy) → policy_d
              │           ├── [AMP] discriminator(expert) → expert_d
              │           ├── [AMP] amp_loss = 0.5×(MSE(expert_d,+1) + MSE(policy_d,-1))
              │           ├── [AMP] grad_pen = discriminator.compute_grad_pen(...)
              │           ├── loss += amp_loss + grad_pen
              │           │
              │           ├── optimizer.zero_grad()
              │           ├── loss.backward()
              │           │     ├── surrogate → Actor 梯度
              │           │     ├── value     → Critic 梯度
              │           │     ├── amp_loss  → Discriminator 梯度（不到 Actor）
              │           │     └── grad_pen  → Discriminator 梯度
              │           │
              │           ├── clip_grad_norm_(policy.parameters(), 1.0)
              │           ├── optimizer.step()
              │           │     └── 同时更新 Actor + Critic + Discriminator
              │           │
              │           └── [AMP] amp_normalizer.update()
              │
              ├── storage.clear()    # 清 RolloutStorage，不清 ReplayBuffer
              │
              └── ▼▼▼ C. 日志 ▼▼▼
                    ├── log(locals())
                    │     ├── 写 TensorBoard
                    │     └── 控制台打印
                    └── if it % save_interval == 0:
                          save(model_{it}.pt)
                            ├── policy.state_dict()
                            ├── optimizer.state_dict()
                            ├── [AMP] discriminator.state_dict()
                            └── [AMP] amp_normalizer
```

---

## 14. 关键设计要点

### 14.1 为什么 PPO 可以做多个 epoch？

利用 **重要性采样** 修正新旧策略的分布偏差，并通过 **Clip** 和 **自适应 KL** 将新旧策略距离限制在邻域内，确保重要性采样的高效与低方差。

### 14.2 为什么 Critic 用 Clipped Value Loss？

与 Actor 的 Clip 同理，限制 Critic 单步更新幅度。如果 Critic 突变会破坏 GAE 的稳定性（GAE 依赖 V(s) 估值）。

### 14.3 为什么 AMP 用 LSGAN 而非 BCE？

传统 GAN 用 `sigmoid + BCE`，判别器很自信时梯度趋近 0，策略收不到学习信号。LSGAN 用裸线性输出 + MSE，梯度始终为 `2(d - target)`，永远不会消失。

### 14.4 为什么 AMP 需要 ReplayBuffer？

判别器的训练目标是区分"任意策略轨迹"和"专家轨迹"，而非"最近这轮的策略"。混入历史各阶段的策略数据，防止判别器因训练分布突变而震荡。这和 DQN 用 replay buffer 解决 non-stationary 问题是同一个思路。

### 14.5 为什么判别器要用梯度惩罚？

防止判别器过强 → policy_d 死死钉在 -1 → amp_reward 全为 0 → 策略学不到 AMP 信号。梯度惩罚强迫判别器输出曲面在专家数据附近平滑。

### 14.6 为什么判别器拆成 trunk 和 amp_linear？

为了对两部分施加不同强度的 L2 正则：
- **trunk**（特征提取）：`weight_decay=1e-3`（轻正则，保留特征能力）
- **amp_linear**（输出层）：`weight_decay=1e-1`（重正则，防止输出过拟合）

### 14.7 为什么 storage.clear() 但 amp_storage 不 clear？

- `RolloutStorage`：存 PPO on-policy 数据，下一轮必须重新采样
- `ReplayBuffer`：存策略历史轨迹，跨轮保留是 AMP 稳定性的关键

### 14.8 为什么 Rollout 中要用 `torch.inference_mode()`？

采样时不需要梯度，关掉计算图能节省内存并提速 ~20%。`compute_returns()` 也是纯数值运算，一起包在 `inference_mode` 里。

### 14.9 为什么终态修复用 `terminal_amp_obs` 而非再次调用 `get_amp_obs_for_expert_trans()`？

`env.step()` 内部已经 reset，再次调用获取的是新回合初始姿态。env 在 reset 之前先保存了真实终态到 `terminal_amp_obs`，runner 直接读这个保存值才能拿到"摔倒前最后一帧"。

### 14.10 为什么 `amp_obs`（下一轮 s_t）用未修复的版本？

下一轮的 `s_t` 必须是**真实存在于环境中的状态**，否则 `act()` 会基于"已死亡机器人"做决策。reset 后的初始姿态才是真实的环境状态。

---

## 附录 A：观测空间细分

### A.1 Actor 单帧观测（75维）

| 分量 | 维度 | 说明 |
|--|--|--|
| 角速度 | 3 | IMU 测量的机身角速度 |
| 投影重力 | 3 | 重力在机身坐标系的投影 |
| 速度指令 | 3 | 目标 vx, vy, ωz |
| 关节位置偏差 | 20 | 当前 - 默认姿态 |
| 关节速度 | 20 | 当前关节角速度 |
| 上步动作 | 20 | 历史动作记忆 |
| sin(gait_phase) | 2 | 步态正弦信号（左右脚） |
| cos(gait_phase) | 2 | 步态余弦信号 |
| phase_ratio | 2 | 步态腾空占比 |

历史 10 步堆叠 → Actor 输入 = 750 维

### A.2 Critic 额外特权信息（+5维）

| 分量 | 维度 | 说明 |
|--|--|--|
| 基座线速度 | 3 | 仿真器真实速度 |
| 足部接触状态 | 2 | 左右脚是否触地 |

总 Critic 输入 = 800 维（10 步堆叠 × 80 单帧）

### A.3 AMP 状态（52维）

| 分量 | 维度 | 说明 |
|--|--|--|
| 关节角度 | 20 | 各关节当前角度 |
| 关节速度 | 20 | 各关节当前角速度 |
| 末端位置/速度 | 12 | 脚、手等关键点 |

---

## 附录 B：步态时钟系统

| 参数 | Walk | Run |
|--|--|--|
| gait_cycle | 0.85s | 0.5s |
| air_ratio | 0.38 | 0.6 |
| phase_offset_l | 0.38 | 0.6 |
| phase_offset_r | 0.88 | 0.1 |
| 左右相位差 | 0.5（交替步行） | 0.5（交替跑步） |

```python
gait_phase = (t / gait_cycle + phase_offset) % 1.0
```

步态相位作为观测的 sin/cos 编码输入网络，同时驱动周期性步态奖励：
- `gait_feet_frc_perio`：摆动相（空中）受力越小得分越高
- `gait_feet_spd_perio`：支撑相（踩地）速度越小得分越高

---

## 附录 C：术语对照表

| 中文 | 英文 | 缩写 |
|--|--|--|
| 优势函数 | Advantage Function | A_t |
| 广义优势估计 | Generalized Advantage Estimation | GAE |
| 折扣回报 | Discounted Return | R_t |
| 时序差分 | Temporal Difference | TD |
| 重要性采样 | Importance Sampling | IS |
| 近端策略优化 | Proximal Policy Optimization | PPO |
| 对抗运动先验 | Adversarial Motion Priors | AMP |
| 最小二乘 GAN | Least-Squares GAN | LSGAN |
| 二元交叉熵 | Binary Cross-Entropy | BCE |
| 梯度惩罚 | Gradient Penalty | GP |
| 经验回放 | Replay Buffer | - |
| 域随机化 | Domain Randomization | DR |
| 随机网络蒸馏 | Random Network Distillation | RND |

---

**文档版本**：v1.0
**最后更新**：2026-05-28
**适用代码版本**：基于 TienKung-Lab 主分支
