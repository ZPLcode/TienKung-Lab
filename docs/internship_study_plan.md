# RL 运控实习准备计划

> 目标：基于 TienKung-Lab 项目，系统性地建立 RL 运控领域的技术深度，为实习面试做好准备。

---

## 当前进度

- [x] 理解项目整体架构和执行流程
- [x] 掌握 PPO 算法实现（clip loss、GAE、自适应 KL）
- [x] 理解 Actor-Critic 网络结构
- [x] 理解 RolloutStorage 数据管理
- [x] 理解 OnPolicyRunner 训练循环
- [x] 理解 TienKungEnv 环境（step/reset/观测/步态时钟）
- [x] 理解奖励系统设计（22 项奖励函数的物理含义）
- [x] 理解 Sim2Sim MuJoCo 迁移（关节映射、实时推理）
- [ ] **AMP 对抗运动先验**
- [ ] **动手实验**
- [ ] **面试准备**

---

## 阶段一：补齐 AMP 知识（2-3 天）

### 1.1 先读论文再读代码

| 顺序 | 内容 | 说明 |
|------|------|------|
| 1 | 论文 [AMP: Adversarial Motion Priors](https://arxiv.org/abs/2104.02180) | 重点理解 §3 方法部分：判别器如何从动捕数据中学习"风格" |
| 2 | 论文 [AMP for Hardware](https://arxiv.org/abs/2305.06291) | 真实机器人部署版 AMP，与本项目直接相关 |

**阅读论文时带着这些问题：**
- AMP 判别器的输入是什么？（答：连续两帧的状态转移 `(s_t, s_{t+1})`）
- AMP 奖励怎么算？（答：判别器给策略轨迹打分，越接近专家动作得分越高）
- AMP 和 GAIL 的区别是什么？（答：AMP 用 least-squares GAN loss，不需要显式的状态-动作匹配）
- 为什么 AMP 比纯 reward shaping 好？（答：不用手动设计每个关节的目标轨迹，只需提供参考动作片段）

### 1.2 阅读代码文件

按以下顺序逐个击破：

```
1. rsl_rl/rsl_rl/modules/discriminator.py        -- 判别器网络结构
2. rsl_rl/rsl_rl/storage/replay_buffer.py         -- AMP 经验回放缓存
3. rsl_rl/rsl_rl/algorithms/amp_ppo.py            -- AMP-PPO 算法（对比你已读的 ppo.py）
4. rsl_rl/rsl_rl/runners/amp_on_policy_runner.py  -- AMP 训练循环（对比 on_policy_runner.py）
5. legged_lab/envs/tienkung/walk_cfg.py L328-380  -- AMP 相关配置参数
```

**核心对比：AMP-PPO vs PPO**

| 模块 | PPO | AMP-PPO |
|------|-----|---------|
| 算法类 | `PPO` | `AMPPPO` |
| Runner | `OnPolicyRunner` | `AmpOnPolicyRunner` |
| 额外组件 | 无 | Discriminator + ReplayBuffer + AMPLoader |
| 观测 | actor_obs, critic_obs | + amp_obs（AMP 状态） |
| 奖励 | 纯任务奖励 | 0.7×任务奖励 + 0.3×判别器奖励 |
| Loss | surrogate + value + entropy | + amp_loss + grad_penalty |

### 1.3 关键理解检查点

读完后，确保你能回答：
- [ ] 判别器的 input_dim 是多少？它的输入由哪些物理量拼接而成？
- [ ] `amp_task_reward_lerp = 0.7` 这个参数控制什么？调大/调小会怎样？
- [ ] 梯度惩罚 `grad_pen_loss` 的作用是什么？（防止判别器过拟合）
- [ ] ReplayBuffer 在 AMP 中起什么作用？（存储策略产生的历史转移数据，稳定判别器训练）
- [ ] 专家数据从哪里加载？格式是什么？（`datasets/motion_amp_expert/walk.txt`）

---

## 阶段二：动手实验（3-5 天）

> [!IMPORTANT]
> 这一阶段是**最关键的**。面试时"我训过模型，调过参数，遇到过 XXX 问题并解决了"比"我读过代码"有说服力 10 倍。

### 2.1 基础训练实验

```bash
# 实验 1: 纯 PPO 训练 walk（无 AMP），观察收敛曲线
python legged_lab/scripts/train.py --task=walk_ppo --headless --logger=tensorboard --num_envs=4096

# 实验 2: AMP-PPO 训练 walk，对比收敛速度和步态自然度
python legged_lab/scripts/train.py --task=walk --headless --logger=tensorboard --num_envs=4096

# 实验 3: AMP-PPO 训练 run
python legged_lab/scripts/train.py --task=run --headless --logger=tensorboard --num_envs=4096
```

**观察指标（TensorBoard）：**
- `Train/mean_reward` — 平均回合奖励
- `Train/mean_episode_length` — 平均生存步数（越长越好）
- `Loss/surrogate` — Actor loss 是否稳定
- `Loss/value_function` — Critic loss 是否下降
- `Policy/mean_noise_std` — 探索噪声是否逐步减小
- `Loss/amp` — AMP 判别器 loss（仅 AMP 模式）

### 2.2 关键调参实验

选 2-3 个实验做，**记录结果和分析**：

| 实验 | 修改内容 | 预期观察 |
|------|----------|----------|
| A | 将 `action_scale` 从 0.25 改为 0.5 | 动作幅度变大，可能更激进但不稳定 |
| B | 将 `termination_penalty` 从 -200 改为 -50 | "生存压力"降低，可能学到更冒险的步态 |
| C | 关闭步态奖励（权重设为 0） | 观察步态是否退化成拖步/跳跃 |
| D | 修改 `gait_cycle` 从 0.85 改为 0.6 | 步频加快，观察是否能适应 |
| E | 修改 `amp_task_reward_lerp` 从 0.7 改为 0.3 | 更依赖 AMP 风格奖励 |

### 2.3 Sim2Sim 验证

```bash
# 训练完后回放
python legged_lab/scripts/play.py --task=walk --num_envs=1

# 导出并在 MuJoCo 中验证
python legged_lab/scripts/sim2sim.py --task walk --policy logs/walk/最新时间戳/exported/policy.pt
```

---

## 阶段三：做一个可展示的改进（5-7 天）

> [!TIP]
> 面试时能说"我在开源项目上做了改进，效果提升了"是非常加分的。

**以下选一个做：**

### 方案 A：添加新的奖励项并验证效果（推荐，最容易上手）
- 在 `rewards.py` 中添加新的奖励函数（如步长对称性奖励、能效优化奖励）
- 在 `walk_cfg.py` 中配置权重，对比训练曲线
- 写一份简短的实验报告

### 方案 B：实现一个新的运动任务
- 基于 `walk_cfg.py` 创建新任务配置（如侧走、转弯）
- 调整速度指令范围和奖励权重
- 在 `envs/__init__.py` 中注册新任务

### 方案 C：改进观测空间
- 添加足底压力分布到观测（而非简单 bool）
- 或引入速度估计器（用历史关节数据估计基座速度）
- 分析改进前后的训练效果差异

---

## 阶段四：面试知识准备（持续）

### 4.1 必须能清晰回答的核心问题

**PPO 相关：**
1. PPO 的 clip 机制为什么有效？与 TRPO 的区别？
2. GAE 的 λ 参数平衡了什么？λ=0 和 λ=1 分别退化成什么？
3. 为什么需要 Value Function Bootstrapping？超时和摔倒的处理有什么不同？
4. 自适应学习率是如何根据 KL 散度调整的？

**运控 specific：**
5. 观测空间为什么包含 sin/cos(gait_phase)？直接用 phase 角度行不行？
6. 为什么用历史观测堆叠？和 RNN 方案的优劣对比？
7. `action_scale = 0.25` 意味着什么？为什么不直接输出绝对角度？
8. 域随机化各项分别模拟了什么真实世界的不确定性？
9. Sim2Sim 的意义是什么？和直接 Sim2Real 相比有什么优势？
10. 奖励工程中权重如何调？

**AMP 相关：**
11. AMP 判别器为什么用 least-squares loss 而非 BCE？
12. AMP 参考动作数据的制作流程？
13. 纯 PPO 和 AMP-PPO 训练出的步态有什么质的区别？

### 4.2 推荐扩展阅读

| 优先级 | 论文 | 为什么重要 |
|--------|------|-----------|
| ★★★ | Learning Agile Locomotion (ETH, 2024) | 最先进的四足运控 |
| ★★★ | Sim-to-Real Transfer (Hwangbo 2019) | Sim2Real 经典 |
| ★★☆ | Walk These Ways (CMU, 2022) | 多步态切换 |
| ★★☆ | DreamWaQ (2023) | 无特权信息盲行走 |
| ★★☆ | Extreme Parkour (ETH, 2023) | 极限地形 |

### 4.3 简历项目描述建议

```
项目：基于 RL 的人形机器人运动控制（TienKung-Lab）
- 深入研究 PPO + AMP 算法在人形双足行走/跑步任务中的应用
- 掌握完整 pipeline：IsaacLab 训练 → 奖励设计 → Sim2Sim 验证
- 实现了 [你做的改进]，在 [指标] 上提升了 [X%]
- 理解域随机化、步态时钟、动作延迟等 Sim2Real 关键技术
技术栈：PyTorch, IsaacLab, MuJoCo, PPO, AMP, Domain Randomization
```

---

## 时间线总结

```
第 1-3 天:  阶段一 — 补齐 AMP（论文 + 代码）
第 4-8 天:  阶段二 — 动手训练 + 调参实验
第 9-15 天: 阶段三 — 做一个可展示的改进
持续进行:   阶段四 — 面试题准备 + 论文阅读
```

> [!TIP]
> **最重要的原则：代码要跑起来，手要动起来。** 面试官最看重的不是你能背多少概念，而是你能不能解决实际问题。能说出"我训了 20 个小时，发现 reward 在 3000 iteration 后突然崩了，排查发现是因为 XXX"这种经历，比任何理论都有说服力。
