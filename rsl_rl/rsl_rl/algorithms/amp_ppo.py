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
from rsl_rl.storage import ReplayBuffer, RolloutStorage
from rsl_rl.utils import string_to_callable


class AMPPPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        discriminator,
        amp_data,
        amp_normalizer,
        amp_replay_buffer_size=100000,
        min_std=None,
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
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(
                params, lr=rnd_cfg.get("learning_rate", 1e-3)
            )
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = (
                symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            )
            # Print that we are not using symmetry
            if not use_symmetry:
                print(
                    "Symmetry not used for learning. We will use it for logging instead."
                )
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(
                    symmetry_cfg["data_augmentation_func"]
                )
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(
                symmetry_cfg["data_augmentation_func"]
            ):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # ====== [AMP新增] 判别器相关组件 ======
        # 以下几行是 AMP-PPO 相比普通 PPO 额外增加的：
        self.amploss_coef = 1.0  # 判别器损失在总 loss 中的权重系数
        self.min_std = min_std  # 动作标准差下界（防止策略方差塌缩到0）
        self.discriminator = discriminator  # 判别器网络（区分专家/策略轨迹）
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()  # 临时存储当前步的 AMP 状态
        # ReplayBuffer：存储策略历史状态转移 (s_t, s_{t+1})，判别器训练时从中抽样
        # 注意：这不同于 RolloutStorage（那个每轮清空），ReplayBuffer 是循环覆盖的大缓存
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device
        )
        self.amp_data = amp_data  # AMPLoader：专家动捕数据（walk.txt/run.txt）
        self.amp_normalizer = amp_normalizer  # 对 AMP 状态做在线均值-方差归一化

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # 创建优化器，[AMP新增] 判别器的 trunk 和 amp_linear 用不同的 weight_decay 正则化
        params = [
            {"params": self.policy.parameters(), "name": "policy"},
            # ↓ [AMP新增] 判别器特征提取层：轻正则，保留特征提取能力
            {
                "params": self.discriminator.trunk.parameters(),
                "weight_decay": 10e-4,
                "name": "amp_trunk",
            },
            # ↓ [AMP新增] 判别器输出层：重正则，防止输出层过拟合
            {
                "params": self.discriminator.amp_linear.parameters(),
                "weight_decay": 10e-2,
                "name": "amp_head",
            },
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
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

    def act(self, obs, critic_obs, amp_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        # [AMP新增] 记录这一步的 AMP 状态（52维），用于后续存入 ReplayBuffer
        # 普通 PPO 的 act() 没有这一行
        self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # [AMP新增] 将当前步的 (s_t, s_{t+1}) 存入 AMP ReplayBuffer
        # s_t = amp_transition.observations（act()时记录）
        # s_{t+1} = amp_obs（本步执行后的新 AMP 状态）
        # 普通 PPO 的 process_env_step() 没有这一行
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def update(self):  # noqa: C901
        """
        AMP-PPO 核心网络参数更新阶段。
        在 PPO 的 Actor/Critic 更新基础上，额外训练 AMP 判别器。
        三路数据（rollout/策略回放/专家动捕）同步迭代，一次 backward 更新所有网络。
        """
        # ====== 1. 初始化各项损失和统计累加器 ======
        # [AMP新增] 相比 ppo.py 多了 amp_loss / grad_pen / policy_pred / expert_pred
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_amp_loss = 0          # 判别器 LSGAN 损失（专家→+1，策略→-1）
        mean_grad_pen_loss = 0     # 判别器梯度惩罚损失（防止判别器过强）
        mean_policy_pred = 0       # 策略轨迹的平均判别分数（训练初期接近 -1，收敛后接近 +1）
        mean_expert_pred = 0       # 专家轨迹的平均判别分数（始终应接近 +1）
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # ====== 2. 实例化三个并行 Mini-Batch Generator ======
        # [AMP新增] 相比 ppo.py 多了 amp_policy_generator 和 amp_expert_generator
        # 三路 generator 通过 zip 严格同步迭代，每次取出同等大小的 batch
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            # rollout generator：从 RolloutStorage 取 PPO 训练数据（obs/action/reward 等）
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # 策略回放 generator：从 ReplayBuffer 随机采样策略历史状态转移 (s_t, s_{t+1})
        # 总迭代次数 = num_epochs × num_mini_batches，batch_size = num_envs × num_steps // num_mini_batches
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
        )
        # 专家数据 generator：从 AMPLoader 随机采样专家动捕状态转移 (s_t, s_{t+1})
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
        )

        # ====== 3. 循环遍历 Mini-Batch 进行更新 ======
        for sample, sample_amp_policy, sample_amp_expert in zip(
            generator, amp_policy_generator, amp_expert_generator
        ):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,      # 旧 Critic 对 s_t 的价值估计 V_old(s_t)
                advantages_batch,         # GAE 计算出的优势估计 A_t
                returns_batch,            # 折扣回报目标值 R_t = A_t + V_old(s_t)
                old_actions_log_prob_batch,  # 采样时旧策略的对数概率 log π_old(a|s)
                old_mu_batch,             # 旧策略动作分布均值（用于 KL 计算）
                old_sigma_batch,          # 旧策略动作分布标准差（用于 KL 计算）
                hid_states_batch,
                masks_batch,
                rnd_state_batch,
            ) = sample

            num_aug = 1
            original_batch_size = obs_batch.shape[0]

            # ====== 3.1 优势函数局部归一化（若启用） ======
            # 在 mini-batch 内再次归一化，进一步稳定梯度
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            # ====== 3.2 对称数据增强（若启用） ======
            # 人形机器人左右腿对称，镜像样本等价于免费获得额外训练数据
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                    obs_type="policy",
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch,
                    actions=None,
                    env=self.symmetry["_env"],
                    obs_type="critic",
                )
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # 其余张量也复制相同倍数以保持维度对齐
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # ====== 3.3 最新策略前向传播 ======
            # 因为策略参数已更新，必须重新计算当前策略下的动作概率和价值
            # -- Actor：计算新策略 log π_new(a|s)
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- Critic：计算新的状态价值估计 V_new(s)
            value_batch = self.policy.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            # -- 熵和分布参数：只取原始非增强部分，排除镜像副本的干扰
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # ====== 3.4 自适应 KL 学习率调节 ======
            # 通过监控新旧策略的 KL 散度，动态调整学习率防止更新步长过大
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    # 多维高斯分布的 KL 散度解析公式
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # 多 GPU：汇总所有卡的 KL 散度求均值
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(
                            kl_mean, op=torch.distributed.ReduceOp.SUM
                        )
                        kl_mean /= self.gpu_world_size

                    # 主进程根据 KL 自适应缩放学习率
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            # KL 过大（策略变化太快）→ 降低学习率
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            # KL 过小（策略变化太保守）→ 提高学习率
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # 广播更新后的学习率到所有 GPU
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # ====== 3.5 PPO Actor 策略裁剪损失（Surrogate Loss） ======
            # r(θ) = π_new(a|s) / π_old(a|s) = exp(log_new - log_old)
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            # 裁剪 r(θ) 到 [1-ε, 1+ε]，防止单步更新幅度过大
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            # 取两者中较大的（负号下等价于取最保守的更新），以 min 形式约束策略更新
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ====== 3.6 Critic 价值评估损失（Value Function Loss） ======
            if self.use_clipped_value_loss:
                # 类似 PPO，同样裁剪价值更新幅度，防止 Critic 突变
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                # 常规 MSE
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # ====== 3.7 合并 PPO 基础损失 ======
            # 总 Loss = Surrogate（Actor）+ c1×Value（Critic）- c2×Entropy（探索鼓励）
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # ====== 3.8 对称镜像损失（若启用） ======
            # 强制网络对镜像观测输出镜像动作，防止策略学出跛脚等不对称步态
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch,
                        actions=None,
                        env=self.symmetry["_env"],
                        obs_type="policy",
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # 对镜像观测做确定性推理（用均值，不采样）
                mean_actions_batch = self.policy.act_inference(
                    obs_batch.detach().clone()
                )
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None,
                    actions=action_mean_orig,
                    env=self.symmetry["_env"],
                    obs_type="policy",
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:],
                    # detach 让梯度只从左侧（镜像输出）反传，右侧作为固定目标
                    actions_mean_symm_batch.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # ====== 3.9 RND 好奇心蒸馏损失（若启用） ======
            # Predictor 网络拟合 frozen Target 网络，预测误差大的状态给额外内在奖励
            if self.rnd:
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # ====== 3.10 [AMP新增] 判别器损失计算 ======
            # 以下是 AMP-PPO 相比普通 PPO 最核心的额外逻辑
            # sample_amp_policy: 从 ReplayBuffer 取的策略历史状态转移 (s_t, s_{t+1})
            # sample_amp_expert: 从 AMPLoader 取的专家动捕状态转移 (s_t, s_{t+1})
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            # 3.10.1 对 AMP 状态做在线均值-方差归一化
            # 注意：用 no_grad 因为归一化参数不参与梯度计算
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

            # 3.10.2 判别器前向传播
            # 输入：将 s_t 和 s_{t+1} 拼接为 104 维向量（52 × 2）
            policy_d = self.discriminator(
                torch.cat([policy_state, policy_next_state], dim=-1)
            )  # 策略轨迹打分，理想训练收敛后应接近 +1
            expert_d = self.discriminator(
                torch.cat([expert_state, expert_next_state], dim=-1)
            )  # 专家轨迹打分，应始终稳定在 +1 附近

            # 3.10.3 Least-Squares GAN 损失
            # 专家数据目标 = +1（判别器认为是真实动作）
            # 策略数据目标 = -1（判别器认为是生成动作）
            # 注：用 MSE 而非 BCE，避免梯度饱和问题（BCE 在置信度高时梯度趋近于 0）
            expert_loss = torch.nn.MSELoss()(
                expert_d, torch.ones(expert_d.size(), device=self.device)
            )
            policy_loss = torch.nn.MSELoss()(
                policy_d, -1 * torch.ones(policy_d.size(), device=self.device)
            )
            amp_loss = 0.5 * (expert_loss + policy_loss)

            # 3.10.4 梯度惩罚（Gradient Penalty）
            # 惩罚判别器在专家数据处的输出梯度过大（参考 WGAN-GP）
            # 防止判别器过强 → 策略收不到有效学习信号 → 训练死锁
            grad_pen_loss = self.discriminator.compute_grad_pen(
                *sample_amp_expert, lambda_=10
            )

            # 3.10.5 将判别器损失并入总 Loss
            # PPO Loss 和 AMP Loss 共享一次 backward，同时更新 Actor/Critic/Discriminator
            loss += self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss

            # ====== 3.11 梯度回传与参数更新 ======
            self.optimizer.zero_grad()
            loss.backward()  # 一次反向传播同时计算 policy + discriminator 的梯度
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # 多 GPU：汇总所有卡的梯度
            if self.is_multi_gpu:
                self.reduce_parameters()

            # 裁剪 policy 梯度 L2 范数，防止梯度爆炸（只对 policy，不对 discriminator）
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()  # 同时更新 policy + discriminator（共享同一个 optimizer）
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # ====== 3.12 [AMP新增] 更新 AMP 状态归一化统计 ======
            # 用本 batch 的策略和专家数据更新在线均值-方差（Welford 算法）
            # 注意：此时 policy_state/expert_state 是归一化后的值，更新的是计数统计
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())

            # ====== 3.13 累加损失统计 ======
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # ====== 4. 计算所有 mini-batch 的平均损失 ======
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates

        # ====== 5. 清空 RolloutStorage，重置步指针 ======
        # 注意：ReplayBuffer 不清空（跨轮保留历史数据）
        self.storage.clear()

        # ====== 6. 构造日志字典并返回 ======
        # amp_policy_pred 趋近 +1 说明策略动作越来越像专家，是 AMP 训练效果的核心指标
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
            "amp_policy_pred": mean_policy_pred,   # 策略轨迹判别分数均值（收敛后应接近 +1）
            "amp_expert_pred": mean_expert_pred,   # 专家轨迹判别分数均值（应稳定在 +1 附近）
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
        grads = [
            param.grad.view(-1)
            for param in self.policy.parameters()
            if param.grad is not None
        ]
        if self.rnd:
            grads += [
                param.grad.view(-1)
                for param in self.rnd.parameters()
                if param.grad is not None
            ]
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
                param.grad.data.copy_(
                    all_grads[offset : offset + numel].view_as(param.grad.data)
                )
                # update the offset for the next parameter
                offset += numel
