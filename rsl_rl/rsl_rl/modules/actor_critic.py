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

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        # 解析并实例化激活函数（通常是 nn.ELU）
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        
        # ====== 1. 构建 Actor 神经网络（策略网络，计算动作均值 Mean） ======
        actor_layers = []
        # 输入层：从单步观测特征维度（例如 750）映射到第一个隐藏层维度
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        # 中间隐藏层循环搭建
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                # 输出层：从最后一个隐藏层映射到动作数量（例如 20维，代表各关节的目标偏置）
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        # 用 nn.Sequential 包装成一个顺序执行的模型
        self.actor = nn.Sequential(*actor_layers)

        # ====== 2. 构建 Critic 神经网络（价值评估网络，计算当前状态评分 Value） ======
        critic_layers = []
        # 输入层：从特权观测维度（例如 800）映射到第一个隐藏层维度
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        # 中间隐藏层循环搭建
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                # 输出层：输出一个标量数值（1维），即当前状态的价值期望 V(s)
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        # 同样包装成顺序执行模型
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # ====== 3. 动作探索噪声（Action Noise，控制策略的随机探索程度） ======
        # 在 PPO 中，Actor 输出动作的均值 Mean，同时需要标准差 Std 来构建正态分布进行随机采样
        self.noise_std_type = noise_std_type
        # 将标准差（或对数标准差）作为 PyTorch 可学习参数 (nn.Parameter)，随着训练进行会自动减小（收敛）
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # 动作采样分布（将在 update_distribution 步骤中实例化）
        self.distribution = None
        # 禁用 PyTorch 对正态分布参数的范围校验，以获得更快的运行速度
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # ====== 1. 前向传播计算动作均值 (Mean) ======
        # 将当前环境观测输入 Actor 多层感知机 (MLP)，预测当前状态下的期望最优动作
        mean = self.actor(observations)
        
        # ====== 2. 解析探索噪声标准差 (Standard Deviation) ======
        if self.noise_std_type == "scalar":
            # 如果是标量模式，直接把预设的固定标准差 std 扩展到与 mean 相同的形状 (num_envs, num_actions)
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            # 如果是对数模式（防止梯度更新使得标准差变为负值），通过指数变换 torch.exp 得到实际标准差，再进行形状扩展
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            
        # ====== 3. 实例化多维独立正态分布对象 ======
        # 以预测的 mean 为中心，以 std 为偏差大小，构造高斯分布，供后续动作采样和对数概率计算使用
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        # ====== 4. 执行动作采样 ======
        # 4.1 根据当前状态观测，更新动作的高斯分布
        self.update_distribution(observations)
        # 4.2 从分布中随机采样出一个动作返回。这种带有噪声的采样就是强化学习的“探索 (Exploration)”机制
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
