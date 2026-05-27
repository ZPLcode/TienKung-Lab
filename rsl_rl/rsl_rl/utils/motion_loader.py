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

import glob
import json

import numpy as np
import torch


class AMPLoader:
    """
    AMP 专家动捕数据加载器与采样器。

    负责将 walk.txt / run.txt 中的原始动捕帧数据解析、插值，
    并在训练时向判别器提供专家状态转移对 (s_t, s_{t+1})。

    AMP 状态维度组成（每帧 52 维）：
        [0  :20] 关节角度  (JOINT_POS,  20个关节)
        [20 :40] 关节速度  (JOINT_VEL,  20个关节)
        [40 :52] 末端位置  (END_POS, 手/脚3D坐标 × 4 = 12)

    判别器 input_dim = 52 × 2 = 104（两帧拼接: s_t + s_{t+1}）
    """

    # ====== AMP 状态的各段维度定义 ======
    JOINT_POS_SIZE = 20  # 20 个关节的角度
    JOINT_VEL_SIZE = 20  # 20 个关节的角速度
    END_EFFECTOR_POS_SIZE = 12  # 4 个末端执行器（手/脚）× 3D坐标

    # ====== 各段在状态向量中的起止索引 ======
    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE  # = 20

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX  # = 20
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE  # = 40

    END_POS_START_IDX = JOINT_VEL_END_IDX  # = 40
    END_POS_END_IDX = END_POS_START_IDX + END_EFFECTOR_POS_SIZE  # = 52

    def __init__(
        self,
        device,
        time_between_frames,  # 两帧之间的时间间隔，对应控制步长 dt = 0.02s
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,  # 预采样 100万 个状态转移对
        motion_files=glob.glob("datasets/motion_amp_expert/*"),  # 默认加载所有动捕文件
    ):
        """
        加载 AMP 专家动捕数据集，解析帧序列并可选地预采样大批量状态转移对。

        time_between_frames: 相邻两个控制步之间的时间差（秒），用于确定 s_t 和 s_{t+1} 的时间间隔。
        """
        self.device = device
        self.time_between_frames = (
            time_between_frames  # 控制步长，即 s_t → s_{t+1} 的时间差
        )

        # ====== 存储每条轨迹的元数据 ======
        self.trajectories = []  # 每条轨迹的帧数据 Tensor（仅 AMP 维度）
        self.trajectories_full = []  # 同上，完整维度版本（用于插值）
        self.trajectory_names = []  # 轨迹名称（文件路径）
        self.trajectory_idxs = []  # 轨迹索引列表
        self.trajectory_lens = []  # 每条轨迹的总时长（秒）
        self.trajectory_weights = (
            []
        )  # 采样权重（MotionWeight 字段，控制各动作被采样的频率）
        self.trajectory_frame_durations = (
            []
        )  # 每帧的持续时间（秒），walk.txt 中为 0.033s ≈ 30fps
        self.trajectory_num_frames = []  # 每条轨迹的总帧数

        # ====== 逐文件加载动捕数据 ======
        # 遍历所有动捕文件，例如 ["walk.txt", "run.txt"]
        # enumerate 同时给出序号 i 和文件路径 motion_file
        for i, motion_file in enumerate(motion_files):
            # 去掉扩展名，存文件名用于日志，例如 "walk.txt" → "walk"
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)  # 解析 JSON，拿到完整字典

                # 取 "Frames" 字段，转为 numpy 数组
                # walk.txt 有 746 帧，每帧原始数据超过 52 维（含根节点位置/旋转等）
                motion_data = np.array(motion_json["Frames"])  # shape: (746, 原始维度)

                # motion_data[:, :52]：切片，取所有帧的前 52 维
                #   ":"  → 所有帧（所有行）
                #   ":52" → 只要前 52 列（关节角度 + 速度 + 末端位置）
                # 转成 GPU Tensor 存入列表，供后续按帧索引采样
                self.trajectories.append(
                    torch.tensor(
                        motion_data[:, : AMPLoader.END_POS_END_IDX],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectories_full.append(
                    torch.tensor(
                        motion_data[:, : AMPLoader.END_POS_END_IDX],
                        dtype=torch.float32,
                        device=device,
                    )
                )

                self.trajectory_idxs.append(i)  # 记录该轨迹的序号，用于加权采样

                # MotionWeight：控制该动作被采样的频率，walk/run 都设为 1.0 时概率各 50%
                # 若 walk=2.0, run=1.0，归一化后 walk 被采样 67%，run 33%
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))

                # FrameDuration：每帧持续多少秒，walk.txt 中为 0.033s（≈ 30fps）
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)

                # 总时长 = (帧数 - 1) × 每帧时长
                # -1 是因为 N 帧只有 N-1 个帧间隔
                # walk.txt: (746 - 1) × 0.033 ≈ 24.6 秒
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                print(f"traj_len:{traj_len}")
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(
                    float(motion_data.shape[0])
                )  # 存总帧数（746.0）

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # 把 Python list 转成 numpy array，方便后续批量索引和向量化运算
        # 同时做权重归一化：[1.0, 1.0] → [0.5, 0.5]，使权重总和为 1（用于 np.random.choice 的 p 参数）
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(
            self.trajectory_weights
        )
        self.trajectory_frame_durations = np.array(
            self.trajectory_frame_durations
        )  # shape: (2,)
        self.trajectory_lens = np.array(self.trajectory_lens)  # shape: (2,)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)  # shape: (2,)

        # ====== 关键优化：训练开始前预采样大批量 (s_t, s_{t+1}) 对 ======
        # 原因：训练中频繁随机采样很慢，提前采好放在 GPU 内存里，每次直接取索引，速度极快
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")

            # 第一步：决定每个样本去哪个动捕文件（walk 还是 run）
            # → np.random.choice([0,1], size=100万, p=[0.5,0.5])
            # → 结果如 [0,1,0,0,1,...], shape: (1000000,)，0=walk, 1=run
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)

            # 第二步：决定去该文件的哪个时刻（在 [0, 总时长-0.02s] 内随机）
            # → 结果如 [3.2s, 11.7s, 0.8s, ...], shape: (1000000,)
            # 末尾留出 0.02s 余量，保证 s_{t+1} 不越界
            times = self.traj_time_sample_batch(traj_idxs)

            # 第三步：在采样时刻处做帧插值（因为动捕 30fps 和控制 50Hz 时间不对齐）
            # preloaded_s[i]      = traj_idxs[i] 文件在 times[i] 处的插值状态，shape: (1000000, 52)
            # preloaded_s_next[i] = traj_idxs[i] 文件在 times[i]+0.02s 处的插值状态
            # 两个矩阵常驻 GPU，训练时直接按索引取，无需重复计算
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames
            )
            print("Finished preloading")

        # 将所有轨迹帧拼接为一个大矩阵，便于全局随机访问
        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """按 MotionWeight 权重随机抽取一条轨迹的索引（单次采样）。"""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """按 MotionWeight 权重批量随机抽取轨迹索引（批量采样）。"""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True
        )

    def traj_time_sample(self, traj_idx):
        """
        在轨迹时间轴上随机采样一个时间点 t，确保 t + dt 也在轨迹范围内
        （即 s_t 和 s_{t+1} 都有效）。
        """
        # subst 保证采样的时间点留出足够余量，使得 t + time_between_frames 不越界
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """批量版本的时间点采样，同时处理多条轨迹。"""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = (
            self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs))
            - subst
        )
        return np.maximum(np.zeros_like(time_samples), time_samples)  # 保证 >= 0

    def slerp(self, frame1, frame2, blend):
        """
        两帧之间的线性插值（Spherical Linear intERPolation 的简化版）。
        blend=0 → frame1, blend=1 → frame2
        解决动捕帧率（30fps）和控制频率（50Hz）不对齐的问题。
        """
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low

        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(
            np.int
        )
        all_frame_starts = torch.zeros(
            len(traj_idxs), self.observation_dim, device=self.device
        )
        all_frame_ends = torch.zeros(
            len(traj_idxs), self.observation_dim, device=self.device
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(
            p * n - idx_low, device=self.device, dtype=torch.float32
        ).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(
            np.int64
        )
        all_frame_amp_starts = torch.zeros(
            len(traj_idxs),
            AMPLoader.END_POS_END_IDX - AMPLoader.JOINT_POSE_START_IDX,
            device=self.device,
        )
        all_frame_amp_ends = torch.zeros(
            len(traj_idxs),
            AMPLoader.END_POS_END_IDX - AMPLoader.JOINT_POSE_START_IDX,
            device=self.device,
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.END_POS_END_IDX
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.END_POS_END_IDX
            ]
        blend = torch.tensor(
            p * n - idx_low, device=self.device, dtype=torch.float32
        ).unsqueeze(-1)

        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        joints0, joints1 = AMPLoader.get_joint_pose(frame0), AMPLoader.get_joint_pose(
            frame1
        )
        joint_vel_0, joint_vel_1 = AMPLoader.get_joint_vel(
            frame0
        ), AMPLoader.get_joint_vel(frame1)

        blend_joint_q = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([blend_joint_q, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """
        ★ 核心方法：为判别器训练生成专家状态转移对 (s_t, s_{t+1})。

        在 amp_ppo.py 的 update() 中被调用：
            amp_expert_generator = amp_data.feed_forward_generator(...)
            for sample_amp_expert in amp_expert_generator:
                expert_state, expert_next_state = sample_amp_expert
                expert_d = discriminator(cat[expert_state, expert_next_state])
                expert_loss = MSE(expert_d, +1)  # 专家数据目标分数 = +1

        Args:
            num_mini_batch: 总共产出多少批（对应 epochs × num_mini_batches）
            mini_batch_size: 每批的样本数
        Yields:
            (s, s_next): 专家当前帧和下一帧，各 shape (mini_batch_size, 52)
        """
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                # 快速路径：直接从预采样池中随机取索引，避免实时插值开销
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[
                    idxs, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.END_POS_END_IDX
                ]
                s_next = self.preloaded_s_next[
                    idxs, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.END_POS_END_IDX
                ]
            else:
                # 慢速路径：实时采样时间点并插值，仅在未预加载时使用
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(
                        self.get_frame_at_time(
                            traj_idx, frame_time + self.time_between_frames
                        )
                    )
                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next  # 产出一批专家 (当前帧, 下一帧) 对

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_joint_pose(pose):
        return pose[AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    def get_end_pos(pose):
        return pose[AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]

    def get_end_pos_batch(poses):
        return poses[:, AMPLoader.END_POS_START_IDX : AMPLoader.END_POS_END_IDX]
