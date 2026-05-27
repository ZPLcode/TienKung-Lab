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

import numpy as np
import torch


class ReplayBuffer:
    """AMP 专用经验回放缓存，存储策略历史状态转移 (s_t, s_{t+1})，用于训练判别器。

    与 RolloutStorage 的核心区别：
    - RolloutStorage 每轮清空；ReplayBuffer 跨轮保留，循环覆盖旧数据
    - ReplayBuffer 随机有放回采样，不依赖时序连续性
    - 混入历史各阶段策略数据，防止判别器因训练分布突变而震荡
    """

    def __init__(self, obs_dim, buffer_size, device):
        # obs_dim：单帧 AMP 状态维度（52维），buffer 存 s_t 和 s_{t+1} 各一份
        # buffer_size：最大容量，默认 100000 条（约 40MB GPU 显存）
        self.states = torch.zeros(buffer_size, obs_dim).to(device)       # s_t
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)  # s_{t+1}
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0         # 下一次写入的起始位置指针（循环覆盖）
        self.num_samples = 0  # 当前已写入的有效样本数（满之前 < buffer_size）

    def insert(self, states, next_states):
        """将新的状态转移批量写入缓存，满后从头循环覆盖最旧数据。

        每次 process_env_step 调用一次，写入量 = num_envs（4096条）。
        """
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states

        if end_idx > self.buffer_size:
            # 写入会超出末尾，分两段写：先写到 buffer 末尾，剩余部分从头覆盖
            self.states[self.step : self.buffer_size] = states[: self.buffer_size - self.step]
            self.next_states[self.step : self.buffer_size] = next_states[: self.buffer_size - self.step]
            self.states[: end_idx - self.buffer_size] = states[self.buffer_size - self.step :]
            self.next_states[: end_idx - self.buffer_size] = next_states[self.buffer_size - self.step :]
        else:
            # 未溢出，直接顺序写入
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        # 有效样本数：满了封顶在 buffer_size，否则取 end_idx 和历史最大值中的较大者
        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        # 指针循环推进
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """随机有放回采样，生成 num_mini_batch 个 mini-batch，供判别器训练使用。"""
        for _ in range(num_mini_batch):
            # 从已有的有效样本中随机抽取（有放回），不保证时序连续
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (self.states[sample_idxs].to(self.device), self.next_states[sample_idxs].to(self.device))
