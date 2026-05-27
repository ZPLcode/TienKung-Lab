# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND 随机网络蒸馏参数（用于好奇心内在奖励）
        rnd_cfg: dict | None = None,
        # 对称性正则化配置（用于规范双足机器人 gait 的对称性）
        symmetry_cfg: dict | None = None,
        # 多 GPU 分布式训练配置
        multi_gpu_cfg: dict | None = None,
    ):
        # 硬件设备分配 (如 "cuda:0")
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # ====== 1. 分布式多 GPU 训练配置 ======
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # ====== 2. RND 好奇心模块配置 ======
        # 作用：通过预测误差产生“内在好奇心奖励”，引导机器人在没有外在奖励时也能积极探索未知的状态空间
        if rnd_cfg is not None:
            # 实例化 RND 核心模块
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # 仅对 RND 中的 Predictor（预测网络）进行参数优化
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # ====== 3. 步态对称性约束配置 ======
        # 作用：强制让机器人左腿和右腿的动作具有对称美感，防止训出诸如“跛脚走”等畸形步态
        if symmetry_cfg is not None:
            # 检查是否启用了镜像数据增强或镜像损失
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # 将配置中的对称变换函数字符串解析为可执行函数
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # 参数校验
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # ====== 4. PPO 核心优化组件初始化 ======
        self.policy = policy
        self.policy.to(self.device)
        # 为策略网络创建 Adam 优化器，设定学习率
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # 经验回放缓冲区 RolloutStorage（将在之后的 init_storage 方法中真正实例化）
        self.storage: RolloutStorage = None  # type: ignore
        # 临时过渡容器，保存当前单步交互的动作、观测、奖励等，用于后续写入 storage
        self.transition = RolloutStorage.Transition()

        # ====== 5. PPO 超参数解析 ======
        self.clip_param = clip_param                                  # PPO 概率裁剪因子 ε，防止新旧策略更新幅度过大（通常为 0.2）
        self.num_learning_epochs = num_learning_epochs                # 每次收集完一批数据后，网络重复训练更新的轮数（Epochs）
        self.num_mini_batches = num_mini_batches                      # 一批大数据分成的 Mini-Batch 个数
        self.value_loss_coef = value_loss_coef                        # Critic 价值损失在总损失中的权重系数
        self.entropy_coef = entropy_coef                              # 熵损失系数（用于鼓励策略保持探索性）
        self.gamma = gamma                                            # 折扣因子 γ（对未来奖励的重视程度，双足通常设为 0.99 左右）
        self.lam = lam                                                # GAE（广义优势估计）的偏差 trade-off 因子 λ（通常为 0.95）
        self.max_grad_norm = max_grad_norm                            # 梯度裁剪最大模长，防止更新步长过大导致参数崩坏
        self.use_clipped_value_loss = use_clipped_value_loss          # 是否对 Value Loss 也进行裁剪，增加训练稳定性
        self.desired_kl = desired_kl                                  # 期望的 KL 散度上限（用于自适应学习率调整）
        self.schedule = schedule                                      # 学习率调整机制（固定或自适应）
        self.learning_rate = learning_rate                            # Adam 优化器初始学习率
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch  # 是否在每个 mini-batch 内对 Advantage 进行归一化


    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        # ====== 1. 如果策略是循环神经网络 (RNN/GRU/LSTM)，保存当前时刻的隐藏状态 ======
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        
        # ====== 2. 前向传播计算动作和状态估值 ======
        # 2.1 通过 Actor 策略网络生成动作（包含探索噪声），并用 .detach() 截断梯度以防止显存泄漏
        self.transition.actions = self.policy.act(obs).detach()
        # 2.2 通过 Critic 价值网络评估当前状态的 V(s) 价值（特权观测喂给 Critic），用 .detach() 截断梯度
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        # 2.3 计算所采样动作在当前策略高斯分布下的对数概率对数（用于重要性采样 PPO Ratio 计算）
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        # 2.4 保存动作分布的均值 (Mean)，用于后续 KL 散度计算与策略分析
        self.transition.action_mean = self.policy.action_mean.detach()
        # 2.5 保存动作分布的标准差 (Sigma/Std)，即当前的动作探索噪声大小
        self.transition.action_sigma = self.policy.action_std.detach()
        
        # ====== 3. 记录物理仿真 step() 之前的环境输入观测值 ======
        # 必须在执行仿真前记录当前状态，因为 env.step() 会更新为下一帧观测值，此时存入可以确保状态-动作匹配对齐
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        """
        在每个物理仿真步结束后调用，负责处理和记录该步的交互数据（如奖励、终止信号、超时自举等），并存入 Buffer。
        """
        # ====== 1. 记录外在（环境）奖励与终止信号 ======
        # 使用 .clone() 拷贝奖励张量，避免后续修改（如加入好奇心奖励、超时自举等）污染环境中的原始数据
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # ====== 2. 计算 RND（随机网络蒸馏）内在好奇心奖励 ======
        if self.rnd:
            # 从环境额外信息中获取用于计算好奇心的观测状态
            rnd_state = infos["observations"]["rnd_state"]
            # 计算内在奖励（根据预测网络对目标网络输出的拟合误差大小，误差越大代表越新颖，奖励越高）
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # 将好奇心奖励累加到总奖励中，鼓励机器人探索未知的动作/状态空间
            self.transition.rewards += self.intrinsic_rewards
            # 记录此时的好奇心状态
            self.transition.rnd_state = rnd_state.clone()

        # ====== 3. 处理超时截断的价值自举（Bootstrapping） ======
        # 这是强化学习处理有限时间步（Finite Horizon）的关键步骤：
        # - 死亡重置（如摔倒）：dones=True，未来的期望回报确实为 0，不需处理。
        # - 超时重置（如走满 1000 步）：dones=True，但机器人并未摔倒。若让它继续走，它还能拿更多分数。
        # 为了不破坏马尔可夫决策过程的完整性，必须将未来的期望回报（即当前状态价值 V(s)）折现累加到当前步奖励中。
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # ====== 4. 存入 Rollout 缓存并清理 ======
        # 4.1 将本步完整的 Transition 数据（包含 s_t, a_t, r_t, d_t, V(s_t), log_prob）存入 Rollout 存储器
        self.storage.add_transitions(self.transition)
        # 4.2 清空临时转移变量，准备记录下一步的数据
        self.transition.clear()
        # 4.3 若策略网络中包含循环神经网络（RNN/GRU/LSTM），将发生终止（dones=True）的环境的隐藏状态（Hidden States）强制清零
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        """
        计算折扣回报（Returns）和优势估计（Advantages）。
        这会在每轮数据收集（Rollout）结束、策略网络更新前调用。
        """
        # ====== 1. 评估最后一帧的边界自举状态价值 ======
        # 为了计算最后一步（例如第 23 步）的 TD 误差，必须知晓其执行动作后达到的最新状态（即第 24 步开头，last_critic_obs）的估值。
        # 用 .detach() 截断梯度以防止前向图保留造成显存泄露。
        last_values = self.policy.evaluate(last_critic_obs).detach()
        
        # ====== 2. 调用存储区计算 GAE 和 Returns ======
        # 将边界自举值、折扣因子 gamma、偏差平衡因子 lam 传入 RolloutStorage。
        # 如果没有配置在 mini-batch 内归一化，则会在整个 Batch 范围对优势 Advantage 进行均值方差归一化。
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        """
        PPO 算法的核心网络参数更新阶段。
        读取 Rollout 缓存区数据，切分成 Mini-Batches，前向传播算 Loss，反向传播更新 Actor & Critic。
        """
        # ====== 1. 初始化各项 Loss 和指标统计累加器 ======
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND 好奇心损失统计
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # 步态对称性损失统计
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # ====== 2. 实例化 Mini-Batch 生成器 ======
        # 根据是否是循环神经网络选择不同的生成器（Recurrent 生成器需要保持轨迹时序的连续性）
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # ====== 3. 循环遍历提取 Mini-Batch 数据包进行更新 ======
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # 记录对称数据增强的倍数（默认为 1，若开启对称增强则会翻倍）
            num_aug = 1
            # 记录当前 Mini-Batch 的原始样本大小
            original_batch_size = obs_batch.shape[0]

            # ====== 3.1 优势函数局部归一化 (若启用) ======
            # 在小批量 Mini-Batch 内部再次归一化 Advantage，能够进一步稳定梯度下降
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # ====== 3.2 执行对称数据增强 (Symmetry Data Augmentation) ======
            # 作用：人形机器人左右腿天然对称。把“左腿迈步，右腿支撑”的数据，镜像成“右腿迈步，左腿支撑”也是合理的。
            # 这能让我们零成本地将样本量翻倍，同时强迫网络不要学出跛脚走等畸形动作。
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # 镜像普通观测和动作
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                # 镜像特权观测
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # 计算翻倍倍数（通常是 2 倍）
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # 将对应的旧 log_prob、旧 V 估值、优势值、折扣回报也复制重复相同的倍数以保持张量对齐
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # ====== 3.3 最新策略前向传播评估 ======
            # 注意：因为策略网络参数已经发生了改变，我们必须重新评估当前状态以获得最新的输出
            # -- Actor 前向传播：计算在当前最新策略下的动作概率
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- Critic 前向传播：计算最新的状态价值评估
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- 动作分布熵：为了公平评估探索程度，只截取原始非增强部分的熵大小
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # ====== 3.4 自适应 KL 散度与学习率动态调节 (Adaptive KL Scheduler) ======
            # PPO 必须限制新策略和老策略不能差异过大。除了 Clip 操作，还可以计算 KL 散度动态改 LR
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    # 计算两个多维高斯分布之间的 KL 散度
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # 多 GPU 分布式训练下，将所有卡上的 KL 散度求和求平均，确保 LR 调整一致
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # 主进程根据 KL 散度大小自适应缩放学习率
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            # KL 太大（策略变化太快），强行降低学习率，防止更新崩坏
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            # KL 太小（策略变化保守），适度增大学习率，加快收敛速度
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # 广播更新后的学习率给所有 GPU
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # 将新学习率应用给 PPO 的优化器参数组
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # ====== 3.5 计算 PPO Actor 策略剪切损失 (Surrogate Loss) ======
            # 计算新旧策略概率比值 r(θ) = π_new / π_old
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            # 未剪切的优势损失
            surrogate = -torch.squeeze(advantages_batch) * ratio
            # 限制比值 r(θ) 在 [1 - ε, 1 + ε] 之间的剪切优势损失
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            # 取两者大值（相当于取最大上限，在负号作用下代表最小化该期望，防止过度策略更新）
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ====== 3.6 计算 Critic 价值评估损失 (Value Function Loss) ======
            if self.use_clipped_value_loss:
                # 类似 PPO，限制新 V 预测相比于旧 V 预测的更新幅度不能超出 clip 阈值，增加稳定性
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                # 常规 MSE（均方误差损失）
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # ====== 3.7 合并 PPO 总损失 (Total Loss) ======
            # 总 Loss = 策略 Clip 损失 + c1 * 价值 MSE 损失 - c2 * 熵增鼓励
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # ====== 3.8 计算对称镜像损失 (Symmetry / Mirror Loss, 若启用) ======
            # 作用：就算不用对称数据增强，也要在损失里直接加入惩罚。
            # 当你输入镜像后的观测，Actor 预测出的动作应当“等于”原始动作进行镜像变换后的结果。如果不等，就惩罚。
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # 计算镜像观测对应的动作均值
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                # 计算原始动作的理论镜像结果
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # 计算两者间的 MSE 差异
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # 若开启对称镜像损失约束，将其加入总 Loss 一起反向传播
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # ====== 3.9 计算 RND 好奇心蒸馏损失 (RND Loss, 若启用) ======
            if self.rnd:
                # Predictor 网络拟合 frozen Target 网络的输出
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # ====== 3.10 梯度计算与回传 ======
            # -- PPO 网络清空并回传梯度
            self.optimizer.zero_grad()
            loss.backward()
            # -- RND 网络清空并回传梯度
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # ====== 3.11 多 GPU 梯度同步汇总 ======
            if self.is_multi_gpu:
                self.reduce_parameters()

            # ====== 3.12 参数应用更新 (Optimizer Step) ======
            # -- 裁剪 PPO 策略梯度，防止动作突变、大参数爆炸
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- 更新 RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # ====== 3.13 累加当前步损失，用于后续求均值统计 ======
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # ====== 4. 计算整个迭代更新的平均损失数值 ======
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
            
        # ====== 5. 清空 Rollout 缓存，重置步指针 ======
        self.storage.clear()

        # ====== 6. 构造日志字典并返回 ======
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
