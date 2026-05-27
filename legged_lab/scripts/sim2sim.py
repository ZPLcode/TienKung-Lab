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

import argparse
import os
import sys

import mujoco
import mujoco_viewer
import numpy as np
import torch
import glfw
import time


class SimToSimCfg:
    """Sim2Sim 仿真验证超参数配置类。

    必须与在 Isaac Lab 中训练该策略时使用的超参数保持 100% 一致。
    """

    class sim:
        sim_duration = 100.0  # 仿真总时长 (秒)
        num_action = 20  # 机器人关节执行器的个数 (20 个自由度)
        num_obs_per_step = 75  # 单个时间步常规观测的维度
        actor_obs_history_length = 10  # 输入网络滑窗历史的步数长度
        dt = 0.005  # 物理引擎步进的时间精度 (5ms, 对应 200Hz)
        decimation = 4  # 控制器降频比，每 4 步物理步更新一次网络动作 (控制频率 50Hz)
        clip_observations = 100.0  # 观测值限幅幅值
        clip_actions = 100.0  # 动作输出限幅幅值
        action_scale = 0.25  # 动作的缩放系数 (等同于训练时的 action_scale)

    class robot:
        # 双腿的步态时序参数 (由 main() 底部根据任务 walk/run 分别动态覆写)
        gait_air_ratio_l: float = 0.38
        gait_air_ratio_r: float = 0.38
        gait_phase_offset_l: float = 0.38
        gait_phase_offset_r: float = 0.88
        gait_cycle: float = 0.85


class MujocoRunner:
    """
    MuJoCo 步态仿真推理执行器。
    负责载入 TorchScript policy 并在 MuJoCo 引擎中同步运行单体人形机器人的位置闭环控制。
    """

    def __init__(self, cfg: SimToSimCfg, policy_path, model_path):
        self.cfg = cfg
        network_path = policy_path

        # ====== 1. 加载 MuJoCo 物理模型并设置步长 ======
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)

        # ====== 2. 载入打包了 obs_normalizer 的 TorchScript 策略模型 ======
        self.policy = torch.jit.load(network_path)

        # ====== 3. 初始化 MuJoCo 图形渲染窗口 ======
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False

        # ====== 4. 初始化运行变量与关节重映射映射表 ======
        self.init_variables()

    def init_variables(self) -> None:
        """初始化运动控制变量，以及 Isaac 与 MuJoCo 的关节索引映射表。"""
        # 计算控制时间步长：0.005s * 4 = 0.02s
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)

        # 悟空机器人在仿真中初始站立的关节默认弧度位置 (保证稍微弯膝稳定站立)
        self.default_dof_pos = np.array(
            [
                0,
                -0.5,
                0,
                1.0,
                -0.5,
                0,
                0,
                -0.5,
                0,
                1.0,
                -0.5,
                0,
                0,
                0.1,
                0.0,
                -0.3,
                0,
                -0.1,
                0.0,
                -0.3,
            ]
        )
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array(
            [self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r]
        )
        self.phase_offset = np.array(
            [self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r]
        )

        # ====== 骨骼关节索引映射 (重中之重) ======
        # 原因：MuJoCo 的 XML 排序树和 Isaac Gym 的内定 Lexicographical 排序完全不一致。
        # 必须显式对齐，否则动作输入和控制信号输出错位会导致瞬间瘫倒。

        # 从 MuJoCo 的传感器序列转换为 Isaac 的观测状态序列的索引重排映射表
        self.mujoco_to_isaac_idx = [
            0,  # hip_roll_l_joint
            6,  # hip_roll_r_joint
            12,  # shoulder_pitch_l_joint
            16,  # shoulder_pitch_r_joint
            1,  # hip_pitch_l_joint
            7,  # hip_pitch_r_joint
            13,  # shoulder_roll_l_joint
            17,  # shoulder_roll_r_joint
            2,  # hip_yaw_l_joint
            8,  # hip_yaw_r_joint
            14,  # shoulder_yaw_l_joint
            18,  # shoulder_yaw_r_joint
            3,  # knee_pitch_l_joint
            9,  # knee_pitch_r_joint
            15,  # elbow_pitch_l_joint
            19,  # elbow_pitch_r_joint
            4,  # ankle_pitch_l_joint
            10,  # ankle_pitch_r_joint
            5,  # ankle_roll_l_joint
            11,  # ankle_roll_r_joint
        ]
        # 从 Isaac 网络动作输出序列转换为 MuJoCo 执行器驱动序列的索引重排映射表
        self.isaac_to_mujoco_idx = [
            0,  # hip_roll_l_joint
            4,  # hip_pitch_l_joint
            8,  # hip_yaw_l_joint
            12,  # knee_pitch_l_joint
            16,  # ankle_pitch_l_joint
            18,  # ankle_roll_l_joint
            1,  # hip_roll_r_joint
            5,  # hip_pitch_r_joint
            9,  # hip_yaw_r_joint
            13,  # knee_pitch_r_joint
            17,  # ankle_pitch_r_joint
            19,  # ankle_roll_r_joint
            2,  # shoulder_pitch_l_joint
            6,  # shoulder_roll_l_joint
            10,  # shoulder_yaw_l_joint
            14,  # elbow_pitch_l_joint
            3,  # shoulder_pitch_r_joint
            7,  # shoulder_roll_r_joint
            11,  # shoulder_yaw_r_joint
            15,  # elbow_pitch_r_joint
        ]

        # 键盘实时微调指令初始速度：[x线速度, y线速度, yaw旋转角速度]
        self.command_vel = np.array([0.0, 0.0, 0.0])
        # 预分配 750 维的滑动历史状态缓冲区大数组
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,),
            dtype=np.float32,
        )

    def get_obs(self) -> np.ndarray:
        """从 MuJoCo 虚拟传感器提取、重排、并滑窗堆叠 750 维的状态观测张量。"""
        # 读取 MuJoCo 传感器缓存中的关节当前角度和角速度 (前 20 维是位置，后 20 维是速度)
        self.dof_pos = self.data.sensordata[0:20]
        self.dof_vel = self.data.sensordata[20:40]

        # 拼接当前时间步的 75 维状态观测向量
        obs = np.concatenate(
            [
                # 1. IMU 测量的基座角速度 (3维)
                self.data.sensor("angular-velocity").data.astype(np.double),
                # 2. 投影重力向量 (3维，通过基座姿态四元数对重力方向 [0, 0, -1] 执行逆向旋转)
                # 注意：MuJoCo 传感器的四元数排列为 (w, x, y, z)，此处 [[1, 2, 3, 0]] 重排为标准 (x, y, z, w) 格式以配合 quat_rotate_inverse
                self.quat_rotate_inverse(
                    self.data.sensor("orientation")
                    .data[[1, 2, 3, 0]]
                    .astype(np.double),
                    np.array([0, 0, -1]),
                ),
                # 3. 控制速度指令 (3维，实时键盘操控)
                self.command_vel,
                # 4. 当前关节角度偏置值 (20维，用 mujoco_to_isaac_idx 重排列对齐)
                (self.dof_pos - self.default_dof_pos)[self.mujoco_to_isaac_idx],
                # 5. 当前关节角速度 (20维，用 mujoco_to_isaac_idx 重排列对齐)
                self.dof_vel[self.mujoco_to_isaac_idx],
                # 6. 上一次下发动作缓存 (20维)
                np.clip(
                    self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions
                ),
                # 7. 双腿步态相位的正弦和余弦量 (4维)
                np.sin(2 * np.pi * self.gait_phase),
                np.cos(2 * np.pi * self.gait_phase),
                # 8. 步态占空比 Air Ratio (2维)
                self.phase_ratio,
            ],
            axis=0,
        ).astype(np.float32)

        # ====== 历史滑动滑窗实现 ======
        # 将历史数组向左滚动 75 维，自动挤掉最老的一步，并在最右侧写入当前最新步的状态
        self.obs_history = np.roll(
            self.obs_history, shift=-self.cfg.sim.num_obs_per_step
        )
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs.copy()

        # 对观测执行防爆幅值截断并回传
        return np.clip(
            self.obs_history,
            -self.cfg.sim.clip_observations,
            self.cfg.sim.clip_observations,
        )

    def position_control(self) -> np.ndarray:
        """将网络输出的动作转换为绝对角度，并映射为 MuJoCo 作动器所需排列顺序。"""
        # 乘以 action_scale (0.25)
        actions_scaled = self.action * self.cfg.sim.action_scale
        # 通过 isaac_to_mujoco_idx 重排动作指令，并加上站立初始姿态，还原绝对关节目标角度
        return actions_scaled[self.isaac_to_mujoco_idx] + self.default_dof_pos

    def run(self) -> None:
        """主控制仿真运行循环。键盘监听、网络前向传播与物理更新。"""
        # 注册 GLFW 窗口键盘事件监听回调
        self.setup_keyboard_listener()

        # 仿真时间控制主循环
        while self.data.time < self.cfg.sim.sim_duration:
            # 1. 获取当前步的 750 维状态特征向量
            self.obs_history = self.get_obs()
            # 2. 神经网络前向推理，得到 20 个关节的目标控制输出 (无梯度，使用 Numpy 转换)
            self.action[:] = (
                self.policy(torch.tensor(self.obs_history, dtype=torch.float32))
                .detach()
                .numpy()[:20]
            )
            self.action = np.clip(
                self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions
            )

            # ====== Decimation 物理循环更新 (50Hz 控制器频率对齐 200Hz 物理频率) ======
            for sim_update in range(self.cfg.sim.decimation):
                step_start_time = time.time()

                # 执行重排位置控制，注入 MuJoCo 动作控制寄存器 self.data.ctrl
                self.data.ctrl = self.position_control()
                # 物理积分器步进一步
                mujoco.mj_step(self.model, self.data)
                # 刷新 MuJoCo Viewer 界面画面
                self.viewer.render()

                # 睡眠补偿，对齐物理真实时间的 5ms 流逝，防止仿真画面运行速度暴走
                elapsed = time.time() - step_start_time
                sleep_time = self.cfg.sim.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # 单局时序计步加 1，并更新当前各腿的步态相位 gait_phase
            self.episode_length_buf += 1
            self.calculate_gait_para()

        # 结束时优雅释放窗口
        self.viewer.close()

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """使用四元数 q 的逆（共轭）旋转向量 v，常用于将全局重力方向转换至机器人局部传感器坐标系中。"""
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0

        return a - b + c

    def calculate_gait_para(self) -> None:
        """根据当前累积的时间帧、相角偏置与步态周期，更新左右脚当前的 gait_phase 相位周期。"""
        t = self.episode_length_buf * self.dt / self.gait_cycle
        self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0  # 左腿相位归一化 [0, 1]
        self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0  # 右腿相位归一化 [0, 1]

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """改变指定维度的键盘控制命令速度值并执行幅值限幅。"""
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], -1.0, 1.0)

    def key_callback(self, window, key, scancode, action, mods):
        """GLFW 窗口级别的键盘回调处理函数，支持 WASD/方向键，Q/E，以及 Space 刹车。"""
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_W or key == glfw.KEY_UP:  # 前进
                self.adjust_command_vel(0, 0.05)
            elif key == glfw.KEY_S or key == glfw.KEY_DOWN:  # 后退
                self.adjust_command_vel(0, -0.05)
            elif key == glfw.KEY_A or key == glfw.KEY_LEFT:  # 向左侧移
                self.adjust_command_vel(1, 0.05)
            elif key == glfw.KEY_D or key == glfw.KEY_RIGHT:  # 向右侧移
                self.adjust_command_vel(1, -0.05)
            elif key == glfw.KEY_Q:  # 左转（逆时针）
                self.adjust_command_vel(2, 0.05)
            elif key == glfw.KEY_E:  # 右转（顺时针）
                self.adjust_command_vel(2, -0.05)
            elif key == glfw.KEY_SPACE:  # 空格键急停刹车
                self.command_vel = np.array([0.0, 0.0, 0.0])

            # 在终端打印当前的目标速度指令，方便调试
            print(
                f"[CMD] vx: {self.command_vel[0]:.2f} | vy: {self.command_vel[1]:.2f} | wy: {self.command_vel[2]:.2f}"
            )

    def setup_keyboard_listener(self) -> None:
        """注册 GLFW 窗口键盘事件回调，无需额外的 pynput 库和全局按键权限。"""
        glfw.set_key_callback(self.viewer.window, self.key_callback)


if __name__ == "__main__":
    LEGGED_LAB_ROOT_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    parser = argparse.ArgumentParser(description="Run sim2sim Mujoco controller.")
    parser.add_argument(
        "--task",
        type=str,
        default="walk",
        choices=["walk", "run"],
        help="Task type: 'walk' or 'run' to set gait parameters",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to policy.pt. If not specified, it will be set automatically based on --task",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(
            LEGGED_LAB_ROOT_DIR, "legged_lab/assets/tienkung2_lite/mjcf/tienkung.xml"
        ),
        help="Path to model.xml",
    )
    parser.add_argument(
        "--duration", type=float, default=100.0, help="Simulation duration in seconds"
    )
    args = parser.parse_args()

    # 默认策略权重映射规则
    if args.policy is None:
        args.policy = os.path.join(
            LEGGED_LAB_ROOT_DIR, "Exported_policy", f"{args.task}.pt"
        )

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    print(f"[INFO] Loaded task preset: {args.task.upper()}")
    print(f"[INFO] Loaded policy: {args.policy}")
    print(f"[INFO] Loaded model: {args.model}")

    sim_cfg = SimToSimCfg()
    sim_cfg.sim.sim_duration = args.duration

    # 根据选择的任务类型 walk (行走) / run (跑步) 分别激活不同的步态周期与相位偏置参数
    if args.task == "walk":
        sim_cfg.robot.gait_air_ratio_l = 0.38
        sim_cfg.robot.gait_air_ratio_r = 0.38
        sim_cfg.robot.gait_phase_offset_l = 0.38
        sim_cfg.robot.gait_phase_offset_r = (
            0.88  # 左右脚相差 0.5，形成完美的 180 度交替对齐步态
        )
        sim_cfg.robot.gait_cycle = 0.85
    elif args.task == "run":
        sim_cfg.robot.gait_air_ratio_l = 0.6
        sim_cfg.robot.gait_air_ratio_r = 0.6
        sim_cfg.robot.gait_phase_offset_l = 0.6
        sim_cfg.robot.gait_phase_offset_r = 0.1  # 双脚悬空比例大幅提升，适应快速奔跑
        sim_cfg.robot.gait_cycle = 0.5

    # 实例化 MuJoCo 运行器并执行
    runner = MujocoRunner(
        cfg=sim_cfg,
        policy_path=args.policy,
        model_path=args.model,
    )
    runner.run()
