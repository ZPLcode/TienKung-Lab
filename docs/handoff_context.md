# TienKung-Lab 学习进度交接文档

> 本文档用于将当前学习进度和上下文交接给下一个 AI 助手。请在新对话开头发送此文件。

---

## 用户目标

用户正在学习 TienKung-Lab（人形机器人 RL 运控）代码库，目标是**找 RL 运控方向实习**。

---

## 项目简介

TienKung-Lab 是一个基于 IsaacLab 的全尺寸人形机器人强化学习运动控制系统，支持 PPO 和 AMP-PPO 两种训练模式，包含 Sim2Sim（MuJoCo）验证。

代码路径：`/home/zhepeng/TienKung-Lab/`

---

## 已完成的学习

### 1. PPO 全链路（已完成 ✅）

用户已完全掌握以下内容，所有文件都已加了中文注释：

| 文件 | 内容 | 状态 |
|------|------|------|
| `legged_lab/scripts/train.py` | 训练入口，task_registry 加载 | ✅ 已读 |
| `legged_lab/envs/__init__.py` | 任务注册表（walk/walk_ppo/run 等） | ✅ 已读 |
| `legged_lab/envs/tienkung/walk_cfg.py` | Walk 任务配置，奖励权重，域随机化 | ✅ 已读 |
| `legged_lab/envs/tienkung/walk_ppo_cfg.py` | PPO 超参数配置 | ✅ 已读 |
| `legged_lab/envs/base/base_env.py` | 环境基类 | ✅ 已读 |
| `legged_lab/envs/tienkung/tienkung_env.py` | 核心环境（step/reset/obs/gait_phase） | ✅ 已读，已加注释 |
| `legged_lab/mdp/rewards.py` | 22 项奖励函数（步态/稳定/能耗/碰撞等） | ✅ 已读 |
| `rsl_rl/rsl_rl/modules/actor_critic.py` | Actor-Critic MLP 网络 | ✅ 已读 |
| `rsl_rl/rsl_rl/algorithms/ppo.py` | PPO 算法（clip loss/GAE/自适应 KL） | ✅ 已读，已加注释 |
| `rsl_rl/rsl_rl/runners/on_policy_runner.py` | 训练主循环（rollout→GAE→update） | ✅ 已读，已加注释 |
| `rsl_rl/rsl_rl/storage/rollout_storage.py` | 经验缓存（Transition/mini-batch） | ✅ 已读，已加注释 |
| `legged_lab/scripts/sim2sim.py` | Sim2Sim MuJoCo 验证 | ✅ 已读，已大幅重写注释，键盘从 pynput 改为 glfw |

### 2. AMP 学习（进行中 🔄）

| 文件 | 内容 | 状态 |
|------|------|------|
| `rsl_rl/rsl_rl/modules/discriminator.py` | AMP 判别器网络 | ✅ 刚加完中文注释，用户已理解 |
| `rsl_rl/rsl_rl/storage/replay_buffer.py` | AMP 经验回放缓存 | ❌ 下一个要看 |
| `rsl_rl/rsl_rl/algorithms/amp_ppo.py` | AMP-PPO 算法 | ❌ 待看（用户已读过，但未加注释） |
| `rsl_rl/rsl_rl/runners/amp_on_policy_runner.py` | AMP 训练循环 | ❌ 待看 |
| `legged_lab/envs/tienkung/walk_cfg.py` L328-380 | AMP 配置参数 | ❌ 待看 |

---

## 已生成的文档

1. **`/home/zhepeng/TienKung-Lab/docs/tienkung_lab_walkthrough.md`** — 完整的代码库学习指南（含架构图、流程图、奖励表、超参数表）
2. **`/home/zhepeng/TienKung-Lab/docs/internship_study_plan.md`** — RL 运控实习准备计划（4 阶段、2 周时间线）
3. **`/home/zhepeng/TienKung-Lab/docs/ppo_algorithm_and_update_guide.md`** — PPO 算法详解文档

---

## 下一步工作（按优先级）

### 立即要做：继续 AMP 代码阅读 + 加注释

阅读顺序：
1. `rsl_rl/rsl_rl/storage/replay_buffer.py` — 理解和 RolloutStorage 的区别
2. `rsl_rl/rsl_rl/algorithms/amp_ppo.py` — 对比 ppo.py，只看差异部分，加中文注释
3. `rsl_rl/rsl_rl/runners/amp_on_policy_runner.py` — 对比 on_policy_runner.py，加中文注释
4. `walk_cfg.py` 中 AMP 相关配置段

### 之后：动手实验

- 跑 PPO vs AMP-PPO 对比训练
- 调参实验（action_scale, termination_penalty, gait_cycle 等）
- Sim2Sim MuJoCo 验证

### 最终：做一个可展示的改进

用户对"倒地起身"功能很感兴趣。我们讨论了业界三种方案：
1. **HoST**（RSS 2025）：Multi-Critic + 辅助力课程学习，从零学起身
2. **FIRM**（2025）：统一的防摔+减伤+起身策略
3. **状态机切换**：独立训练起身策略 + 状态机调度（最实用）

---

## 关键技术要点速查

### PPO 核心公式
- GAE: `A_t = δ_t + γλ(1-done)A_{t+1}`，其中 `δ_t = r_t + γV(s_{t+1}) - V(s_t)`
- Clip Loss: `L = max(-A×ratio, -A×clip(ratio, 1-ε, 1+ε))`
- 自适应 KL: `KL > 2×desired → LR/1.5`; `KL < 0.5×desired → LR×1.5`

### AMP 核心公式
- 判别器训练: 专家→+1, 策略→-1 (Least-Squares GAN)
- AMP 奖励: `reward = 0.3 × clamp(1 - 0.25×(d-1)², min=0)`
- 最终奖励: `total = 0.7 × task_reward + 0.3 × amp_reward`
- 梯度惩罚: `λ × ||∇D(x)||²`，防止判别器过强

### 环境关键参数
- 物理步长 5ms (200Hz), 控制步长 20ms (50Hz, decimation=4)
- Actor 观测 750 维 (75×10 历史堆叠), Critic 观测 800 维
- 步态时钟: `gait_phase = (t/cycle + offset) % 1.0`
- Walk: cycle=0.85s, air_ratio=0.38; Run: cycle=0.5s, air_ratio=0.6

---

## 用户偏好

- 用中文交流
- 喜欢代码中加详细中文注释
- 目标是找 RL 运控实习，重视面试准备
- 喜欢对比式学习（PPO vs AMP-PPO 的差异对照）
- 已修改 sim2sim.py 的键盘控制从 pynput 改为 glfw（WASD 操控）
