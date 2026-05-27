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

import torch
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
    """
    AMP（对抗运动先验）判别器神经网络。

    核心思想：训练一个"裁判"网络，学会区分"专家动捕数据产生的状态转移"和"策略网络产生的状态转移"。
    然后用判别器的输出作为额外奖励信号，引导策略网络生成更像专家动作的运动风格。

    类比 GAN：
    - Generator（生成器）= Actor 策略网络，产生运动轨迹
    - Discriminator（判别器）= 本类，区分真假轨迹
    - 训练目标：判别器尽量区分真假，策略尽量骗过判别器

    Args:
        input_dim (int): 输入特征维度 = 2 × AMP状态维度（因为输入是 s_t 和 s_{t+1} 拼接）
        amp_reward_coef (float): AMP 奖励缩放系数（walk 任务中 = 0.3）
        hidden_layer_sizes (list[int]): MLP 隐藏层维度列表（walk 任务中 = [1024, 512, 256]）
        device (torch.device): 计算设备
        task_reward_lerp (float): 任务奖励与 AMP 奖励的插值因子（walk 任务中 = 0.7，即 70% 任务 + 30% AMP）
    """

    def __init__(
        self,
        input_dim,
        amp_reward_coef,
        hidden_layer_sizes,
        device,
        task_reward_lerp=0.0,
    ):
        super().__init__()

        self.device = device
        self.input_dim = input_dim  # 输入维度 = 2 × AMP 状态维度

        # ====== 1. 构建判别器 MLP 网络 ======
        # 网络被拆成 trunk（特征提取层）和 amp_linear（输出层）两部分
        # 原因：在 AMPPPO 的优化器中，这两部分使用不同的 weight_decay 正则化强度
        #       trunk: weight_decay=1e-3（轻正则，保留特征提取能力）
        #       amp_linear: weight_decay=1e-1（重正则，防止输出层过拟合）
        self.amp_reward_coef = amp_reward_coef  # AMP 奖励缩放系数
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())  # 注意：判别器用 ReLU，而 ActorCritic 用 ELU
            curr_in_dim = hidden_dim
        # trunk: Input → Linear(1024) → ReLU → Linear(512) → ReLU → Linear(256) → ReLU
        self.trunk = nn.Sequential(*amp_layers).to(device)
        # amp_linear: Linear(256) → 1（输出一个标量判别分数 d）
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        # task_reward_lerp = 0.7 表示最终奖励 = 0.7 × 任务奖励 + 0.3 × AMP奖励
        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        """
        判别器前向传播：输入状态转移，输出判别分数 d。

        训练目标（Least-Squares GAN）：
        - 对专家数据（动捕）: 让 d → +1
        - 对策略数据（RL生成）: 让 d → -1

        Args:
            x (torch.Tensor): 拼接后的状态转移 [s_t, s_{t+1}]，形状 (batch_size, input_dim)
        Returns:
            torch.Tensor: 判别分数 d，形状 (batch_size, 1)
        """
        h = self.trunk(x)  # 特征提取
        d = self.amp_linear(h)  # 输出标量判别分数
        return d

    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
        """
        计算梯度惩罚（Gradient Penalty），防止判别器过拟合。

        为什么需要梯度惩罚？
        - 判别器太强 → 对所有策略数据都输出 -1 → 策略梯度信号消失 → 策略学不到东西
        - 梯度惩罚强迫判别器在专家数据附近的输出变化要平缓（梯度接近 0）
        - 这让判别器成为一个有用的"教练"而非冷酷的"拒绝者"
        - 灵感来自 WGAN-GP，但这里只对专家数据施加惩罚

        Args:
            expert_state: 专家动捕数据的当前帧状态
            expert_next_state: 专家动捕数据的下一帧状态
            lambda_: 梯度惩罚系数，默认 10（越大判别器越"温和"）
        Returns:
            grad_pen: 标量梯度惩罚损失
        """
        # 1. 拼接专家的连续两帧状态转移
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True  # 必须开启，因为要对输入（而非参数）求导

        # 2. 前向传播得到判别分数
        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)

        # 3. 计算判别器输出对输入数据的梯度（注意：不是对网络参数的梯度！）
        # 这告诉我们：输入数据微小变化时，判别器输出会变化多少
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # 4. 惩罚梯度的 L2 范数：强迫 ||∇D(x)|| → 0
        # 即在专家数据附近，判别器输出要尽可能"平坦"，不能剧烈波动
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
        """
        计算 AMP 奖励：判别器给策略轨迹打分，越像专家动作得分越高。

        背景概念：
            专家（Expert）= 动捕数据（walk.txt，真人穿动捕服录的自然走路轨迹）
            策略（Policy）= 机器人当前 Actor 网络输出的动作在仿真中产生的轨迹
            判别器训练目标（LSGAN）：专家样本 → d=+1，策略样本 → d=-1

        为什么用 LSGAN 而非传统 GAN：
            传统 GAN 用 Sigmoid+BCE，当判别器已经很自信时（输出接近 0 或 1），
            Sigmoid 两端梯度≈0 → 策略收不到有用的学习信号 → 训练死锁。
            LSGAN 用裸线性输出 + MSE Loss，d ∈ (-∞,+∞)，梯度始终有效，训练更稳定。

        奖励公式：
            d = Discriminator(s_t, s_{t+1})         # 判别分数，d ∈ (-∞, +∞)
            amp_reward = coef × clamp(1 - 0.25 × (d - 1)², min=0)

        为什么系数是 0.25 = 1/4：
            训练目标：专家 → d=+1，策略 → d=-1，两者距离 = 2
            0.25 = 1/(距离²) = 1/2² = 1/4
            作用：让 d 从 -1 到 +1 刚好映射到奖励从 0 到 1
                d = +1（专家级） → 1 - 0.25×0   = 1.0（满分）
                d =  0（中间态） → 1 - 0.25×1   = 0.75
                d = -1（策略级） → 1 - 0.25×4   = 0.0（零分）
                d < -1 或 d > +3 → clamp 截断为 0（防止过拟合）

        最终奖励（当 task_reward_lerp=0.7 时）：
            total = 0.3 × amp_reward + 0.7 × task_reward
            即 30% 风格奖励（像不像人类走路）+ 70% 任务奖励（走得快不快、稳不稳）

        Args:
            state: 当前帧 AMP 状态 s_t，52维（关节角度、角速度、末端位置）
            next_state: 下一帧 AMP 状态 s_{t+1}
            task_reward: 环境任务奖励（速度跟踪、步态等）
            normalizer: 可选的状态归一化器
        Returns:
            reward: 混合后的最终奖励 (batch_size,)
            d: 原始判别分数 (batch_size, 1)，用于日志记录
        """
        with torch.no_grad():
            self.eval()  # 切换到评估模式（关闭 dropout 等）

            # 1. 可选：对输入状态进行归一化
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            # 2. 判别器前向传播：
            #    输入 [s_t, s_{t+1}] 拼接成 104 维 → trunk 特征提取 → amp_linear 输出标量 d
            #    d 无 Sigmoid 约束（LSGAN），d ∈ (-∞, +∞)
            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))

            # 3. 将判别分数 d 转换为奖励值
            #    公式：reward = coef × clamp(1 - (1/4) × (d - 1)², min=0)
            #    d=+1 满分（像专家），d=-1 零分（像策略初期的乱动），d 偏离更远则 clamp 到 0
            #
            #    为什么 clamp(min=0)，不允许负数：
            #      如果允许负奖励，不像专家的动作会被"惩罚"，策略会变得保守不敢探索。
            #      设计意图：不像专家 → 奖励=0（不鼓励但不惩罚），像专家 → 奖励>0（鼓励）。
            #
            #    为什么满分是 1.0：
            #      1.0 只是归一化后的"单位满分"，实际奖励幅度由 amp_reward_coef 控制。
            #      实际奖励范围 = [0, amp_reward_coef] = [0, 0.3]，方便只调一个系数就能控制幅度。
            reward = self.amp_reward_coef * torch.clamp(
                1 - (1 / 4) * torch.square(d - 1), min=0
            )

            # 4. 与任务奖励进行线性插值混合
            #    lerp=0.7 时：最终奖励 = 30% AMP风格奖励 + 70% 任务奖励
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))

            self.train()  # 切回训练模式
        return reward.squeeze(), d

    def _lerp_reward(self, disc_r, task_r):
        """
        线性插值混合判别器奖励与任务奖励。

        公式：r = (1 - lerp) × disc_r + lerp × task_r
        当 lerp=0.7 时：r = 0.3 × 判别器奖励（风格像不像） + 0.7 × 任务奖励（走得好不好）
        """
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r

