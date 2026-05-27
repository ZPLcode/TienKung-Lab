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

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate
from scipy.spatial.transform import Rotation

from legged_lab.envs.tienkung.run_cfg import TienKungRunFlatEnvCfg
from legged_lab.envs.tienkung.run_with_sensor_cfg import TienKungRunWithSensorFlatEnvCfg
from legged_lab.envs.tienkung.walk_cfg import TienKungWalkFlatEnvCfg
from legged_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorFlatEnvCfg,
)
from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay


class TienKungEnv(VecEnv):
    def __init__(
        self,
        cfg: (
            TienKungRunFlatEnvCfg
            | TienKungWalkFlatEnvCfg
            | TienKungWalkWithSensorFlatEnvCfg
            | TienKungRunWithSensorFlatEnvCfg
        ),
        headless,
    ):
        # ====== 类型注解：告诉 IDE self.cfg 的可能类型，方便自动补全 ======
        self.cfg: (
            TienKungRunFlatEnvCfg
            | TienKungWalkFlatEnvCfg
            | TienKungWalkWithSensorFlatEnvCfg
            | TienKungRunWithSensorFlatEnvCfg
        )

        # ====== 1. 存储基础参数 ======
        self.cfg = cfg  # 环境配置对象（来自 walk_cfg.py / walk_ppo_cfg.py）
        self.headless = headless  # 是否无头模式（不渲染画面）
        self.device = self.cfg.device  # 计算设备，如 "cuda:0"
        self.physics_dt = self.cfg.sim.dt  # 物理引擎步长 = 0.005s（200Hz）
        self.step_dt = (
            self.cfg.sim.decimation * self.cfg.sim.dt
        )  # RL 控制步长 = 4 × 0.005 = 0.02s（50Hz）
        self.num_envs = self.cfg.scene.num_envs  # 并行环境数量，默认 4096
        self.seed(cfg.scene.seed)  # 设置随机种子，确保可复现

        # ====== 2. 配置并创建物理引擎（PhysX） ======
        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,  # GPU 加速物理仿真
            dt=cfg.sim.dt,  # 物理步长 0.005s
            render_interval=cfg.sim.decimation,  # 每 4 步物理才渲染一帧（节省 GPU）
            physx=PhysxCfg(
                gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count
            ),  # GPU 刚体接触 patch 上限
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",  # 两物体接触时摩擦系数 = 两者相乘
                restitution_combine_mode="multiply",  # 弹性恢复系数同理
                static_friction=1.0,  # 全局默认静摩擦系数
                dynamic_friction=1.0,  # 全局默认动摩擦系数
            ),
        )
        self.sim = SimulationContext(sim_cfg)  # 创建仿真上下文（物理引擎实例）

        # ====== 3. 创建场景（机器人 + 地形 + 传感器 + 灯光） ======
        scene_cfg = SceneCfg(
            config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt
        )
        self.scene = InteractiveScene(scene_cfg)  # 根据配置在仿真世界中生成所有资产
        self.sim.reset()  # 重置物理引擎，使生成的资产生效

        # ====== 4. 从场景中取出运行时对象（可读写实际数据的句柄） ======
        self.robot: Articulation = self.scene["robot"]  # 机器人关节控制句柄
        self.contact_sensor: ContactSensor = self.scene.sensors[
            "contact_sensor"
        ]  # 足部接触力传感器

        # 可选传感器：高度扫描、LiDAR、深度相机（walk_ppo 任务中均未启用）
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        # Instantiate LiDAR and Depth Camera Sensors if enabled
        if self.cfg.scene.lidar.enable_lidar:
            self.lidar: RayCaster = self.scene.sensors["lidar"]
        if self.cfg.scene.depth_camera.enable_depth_camera:
            self.depth_camera: TiledCamera = self.scene.sensors["depth_camera"]

        # ====== 5. 创建速度指令生成器：随机采样目标速度（vx, vy, ωz）给机器人追踪 ======
        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,  # 每隔多久换一次新指令（默认 10s）
            rel_standing_envs=self.cfg.commands.rel_standing_envs,  # 20% 环境指令为零（练站立）
            rel_heading_envs=self.cfg.commands.rel_heading_envs,  # 100% 使用朝向控制模式
            heading_command=self.cfg.commands.heading_command,  # True = 用目标朝向角代替直接角速度
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,  # 朝向误差→角速度的 P 增益
            debug_vis=self.cfg.commands.debug_vis,  # 是否在仿真画面中可视化指令箭头
            ranges=self.cfg.commands.ranges,  # 速度采样范围（vx, vy, ωz）
        )
        self.command_generator = UniformVelocityCommand(
            cfg=command_cfg, env=self
        )  # 实例化指令生成器

        # ====== 6. 创建奖励管理器：扫描 LiteRewardCfg 中所有 RewTerm，统一管理计算 ======
        self.reward_manager = RewardManager(self.cfg.reward, self)

        # ====== 7. 初始化所有训练用 GPU Tensor（关节索引、步态参数、观测缓存等） ======
        self.init_buffers()

        # ====== 8. 创建域随机化事件管理器并执行一次性初始化事件 ======
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(
            self.cfg.domain_rand.events, self
        )  # 管理摩擦随机化、质量扰动、推力等
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="startup"
            )  # 执行 startup 事件：随机化摩擦系数、添加质量扰动
        self.reset(env_ids)  # 初始重置所有环境

        # ====== 9. 加载 AMP 参考动作数据（用于动捕动画可视化，非训练必需） ======
        self.amp_loader_display = AMPLoaderDisplay(
            motion_files=self.cfg.amp_motion_files_display,
            device=self.device,
            time_between_frames=self.physics_dt,
        )
        self.motion_len = self.amp_loader_display.trajectory_num_frames[
            0
        ]  # 参考动画的总帧数

    def init_buffers(self):
        """初始化所有训练过程中需要的 GPU Tensor 和索引映射。"""
        self.extras = (
            {}
        )  # 用于存放日志信息（各奖励分量、episode 统计等），传给 TensorBoard

        # ====== A. 基础训练参数 ======
        self.max_episode_length_s = (
            self.cfg.scene.max_episode_length_s
        )  # 每个 episode 最长时间 = 20s
        self.max_episode_length = np.ceil(
            self.max_episode_length_s / self.step_dt
        )  # 最大步数 = ceil(20/0.02) = 1000
        self.num_actions = self.robot.data.default_joint_pos.shape[
            1
        ]  # 动作维度 = 20（20 个关节）
        self.clip_actions = self.cfg.normalization.clip_actions  # 动作裁剪范围 = 100.0
        self.clip_obs = self.cfg.normalization.clip_observations  # 观测裁剪范围 = 100.0

        # ====== B. 动作延迟缓冲区（模拟真实机器人的通信延迟） ======
        self.action_scale = (
            self.cfg.robot.action_scale
        )  # 动作缩放系数 = 0.25，q_target = action * 0.25 + q_default
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"],
            self.num_envs,
            device=self.device,
        )  # 创建延迟缓冲区，最大延迟 5 步
        self.action_buffer.compute(
            torch.zeros(
                self.num_envs,
                self.num_actions,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
        )  # 用全零动作初始化缓冲区
        if (
            self.cfg.domain_rand.action_delay.enable
        ):  # 当前配置中 enable=False，不启用延迟
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )  # 为每个环境随机采样一个延迟步数（0~5）
            self.action_buffer.set_time_lag(
                time_lags, torch.arange(self.num_envs, device=self.device)
            )

        # ====== C. 场景实体配置解析（将名字字符串解析为数字索引，供奖励函数等使用） ======
        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(
            self.scene
        )  # 解析后可通过 self.robot_cfg.body_ids 等访问
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor",
            body_names=self.cfg.robot.terminate_contacts_body_names,
        )  # 终止条件用到的身体部位：膝盖、肩膀、肘部、骨盆
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.feet_body_names
        )
        self.feet_cfg.resolve(self.scene)  # 足部传感器索引：ankle_roll_l, ankle_roll_r

        # ====== D. 身体部位索引查找（名字 → GPU Tensor 列索引） ======
        # 用途：后续通过数字索引从 (num_envs, num_bodies) 的 Tensor 中快速切片
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["ankle_roll_l_link", "ankle_roll_r_link"], preserve_order=True
        )  # 左右脚踝的 body 索引，用于读取脚部位置
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["elbow_pitch_l_link", "elbow_pitch_r_link"], preserve_order=True
        )  # 左右肘部的 body 索引，用于 AMP 手部位置计算

        # ====== E. 关节索引查找（名字 → joint_pos/joint_vel 的列索引） ======
        # 用途：从 robot.data.joint_pos (num_envs, 20) 中取出特定肢体的关节角度
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "hip_roll_l_joint",  # 左髋外展
                "hip_pitch_l_joint",  # 左髋屈伸
                "hip_yaw_l_joint",  # 左髋旋转
                "knee_pitch_l_joint",  # 左膝
                "ankle_pitch_l_joint",  # 左踝俯仰
                "ankle_roll_l_joint",  # 左踝横滚
            ],
            preserve_order=True,
        )  # 左腿 6 个关节的索引
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "hip_roll_r_joint",
                "hip_pitch_r_joint",
                "hip_yaw_r_joint",
                "knee_pitch_r_joint",
                "ankle_pitch_r_joint",
                "ankle_roll_r_joint",
            ],
            preserve_order=True,
        )  # 右腿 6 个关节的索引
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "shoulder_pitch_l_joint",  # 左肩俯仰
                "shoulder_roll_l_joint",  # 左肩横滚
                "shoulder_yaw_l_joint",  # 左肩旋转
                "elbow_pitch_l_joint",  # 左肘
            ],
            preserve_order=True,
        )  # 左臂 4 个关节的索引
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "shoulder_pitch_r_joint",
                "shoulder_roll_r_joint",
                "shoulder_yaw_r_joint",
                "elbow_pitch_r_joint",
            ],
            preserve_order=True,
        )  # 右臂 4 个关节的索引
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=[
                "ankle_pitch_l_joint",
                "ankle_pitch_r_joint",
                "ankle_roll_l_joint",
                "ankle_roll_r_joint",
            ],
            preserve_order=True,
        )  # 4 个踝关节的索引（用于踝关节扭矩/动作惩罚奖励）

        # ====== F. 观测缩放与噪声配置 ======
        self.obs_scales = self.cfg.normalization.obs_scales  # 各观测分量的缩放系数
        self.add_noise = (
            self.cfg.noise.add_noise
        )  # 是否给观测添加噪声（模拟真实传感器）

        # ====== G. 运行时计数器 ======
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )  # 每个环境当前走了多少步
        self.sim_step_counter = 0  # 全局仿真总步数
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )  # 是否因超时而终止

        # ====== H. 手臂末端位置计算用的局部向量（用于 AMP 状态表示） ======
        self.left_arm_local_vec = torch.tensor(
            [0.0, 0.0, -0.3], device=self.device
        ).repeat((self.num_envs, 1))
        self.right_arm_local_vec = torch.tensor(
            [0.0, 0.0, -0.3], device=self.device
        ).repeat((self.num_envs, 1))

        # ====== I. 步态时钟参数（核心特色：驱动左右脚交替步态的周期信号） ======
        # gait_phase: 左右脚的相位 φ ∈ [0, 1)，每步递增 step_dt / gait_cycle
        self.gait_phase = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        # gait_cycle: 一个完整步态周期的时长 = 0.85s（约 1.18 步/秒）
        self.gait_cycle = torch.full(
            (self.num_envs,),
            self.cfg.gait.gait_cycle,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # ====== 终态 AMP 观测缓存（用于修复 Reset 边界问题）======
        # 在每次环境重置之前，把该环境的最后一帧 AMP 状态保存在这里
        # Runner 可以用这个来替换 reset 后错误的初始姿态，保证判别器看到正确的 (s_t, s_terminal) 对
        # AMP obs 维度 = 2×(所有关节角度+速度) + 4个末端×3D位置
        self._amp_obs_dim = (
            2 * (len(self.left_arm_ids) + len(self.right_arm_ids)
                 + len(self.left_leg_ids) + len(self.right_leg_ids))
            + 12  # 左手(3) + 右手(3) + 左脚(3) + 右脚(3)
        )
        self.terminal_amp_obs = torch.zeros(
            self.num_envs, self._amp_obs_dim, dtype=torch.float, device=self.device
        )



        # phase_ratio: 摆动相占比 [0.38, 0.38]，即每只脚 38% 时间在空中
        self.phase_ratio = torch.tensor(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        # phase_offset: 相位偏移 [0.38, 0.88]，左右脚差 0.5 周期 → 交替步行
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)

        # ====== J. 动作与足部统计缓存 ======
        self.action = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # 当前步的动作缓存（20 维），用于 action_rate 奖励计算
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs,
            len(self.feet_cfg.body_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # 每步平均足部接触力（用于步态周期奖励）
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs,
            len(self.feet_cfg.body_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # 每步平均足部速度（用于步态周期奖励）

        # ====== K. 初始化观测历史缓存（CircularBuffer，保存最近 10 步观测） ======
        self.init_obs_buffer()

    def visualize_motion(self, time):
        """
        Update the robot simulation state based on the AMP motion capture data at a given time.

        This function sets the joint positions and velocities, root position and orientation,
        and linear/angular velocities according to the AMP motion frame at the specified time,
        then steps the simulation and updates the scene.

        Args:
            time (float): The time (in seconds) at which to fetch the AMP motion frame.

        Returns:
            None
        """
        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        device = self.device

        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)

        dof_pos[:, self.left_leg_ids] = visual_motion_frame[6:12]
        dof_pos[:, self.right_leg_ids] = visual_motion_frame[12:18]
        dof_pos[:, self.left_arm_ids] = visual_motion_frame[18:22]
        dof_pos[:, self.right_arm_ids] = visual_motion_frame[22:26]

        dof_vel[:, self.left_leg_ids] = visual_motion_frame[32:38]
        dof_vel[:, self.right_leg_ids] = visual_motion_frame[38:44]
        dof_vel[:, self.left_arm_ids] = visual_motion_frame[44:48]
        dof_vel[:, self.right_arm_ids] = visual_motion_frame[48:52]

        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)

        env_ids = torch.arange(self.num_envs, device=device)

        root_pos = visual_motion_frame[:3].clone()
        root_pos[2] += 0.3

        euler = visual_motion_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler(
            "XYZ", euler, degrees=False
        ).as_quat()  # [x, y, z, w]
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=torch.float32,
            device=device,
        )

        lin_vel = visual_motion_frame[26:29].clone()
        ang_vel = torch.zeros_like(lin_vel)

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)

        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(
                self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7],
                self.left_arm_local_vec,
            )
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(
                self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7],
                self.right_arm_local_vec,
            )
        )
        left_hand_pos = quat_apply(
            quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos
        )
        right_hand_pos = quat_apply(
            quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos
        )
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(
            quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos
        )
        right_foot_pos = quat_apply(
            quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos
        )

        self.left_leg_dof_pos = dof_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = dof_pos[:, self.right_leg_ids]
        self.left_leg_dof_vel = dof_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = dof_vel[:, self.right_leg_ids]
        self.left_arm_dof_pos = dof_pos[:, self.left_arm_ids]
        self.right_arm_dof_pos = dof_pos[:, self.right_arm_ids]
        self.left_arm_dof_vel = dof_vel[:, self.left_arm_ids]
        self.right_arm_dof_vel = dof_vel[:, self.right_arm_ids]
        return torch.cat(
            (
                self.right_arm_dof_pos,
                self.left_arm_dof_pos,
                self.right_leg_dof_pos,
                self.left_leg_dof_pos,
                self.right_arm_dof_vel,
                self.left_arm_dof_vel,
                self.right_leg_dof_vel,
                self.left_leg_dof_vel,
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )

    def compute_current_observations(self):
        """计算当前单帧时刻的 Actor 观测（75维）和 Critic 特权观测（80维）。"""
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # ====== 1. 提取机器人基本运动学/动力学状态 ======
        ang_vel = robot.data.root_ang_vel_b  # 基座（骨盆）角速度 (3维)
        projected_gravity = robot.data.projected_gravity_b  # 重力投影向量 (3维)
        command = (
            self.command_generator.command
        )  # 当前追踪的目标线速度/角速度指令 (3维)

        # 关节位置与速度的偏差值（当前值 - 默认姿态值）
        joint_pos = (
            robot.data.joint_pos - robot.data.default_joint_pos
        )  # 关节位置偏差 (20维)
        joint_vel = (
            robot.data.joint_vel - robot.data.default_joint_vel
        )  # 关节速度偏差 (20维)

        # 提取上一步发送给电机的历史动作目标（取环形队列的最后一帧 -1）
        # 作用：1. 提供动作连续性记忆，帮助策略降低关节抖动并计算动作变化率惩罚（Action Rate Penalty）
        #       2. 维持马尔可夫链完整性，让网络能区分“姿态发生变化”是由于自己之前的动作还是外部推力导致的
        action = self.action_buffer._circular_buffer.buffer[
            :, -1, :
        ]  # 上步动作历史 (20维)

        # ====== 2. 提取特权状态（仅 Critic 使用，实机不可直接测得） ======
        root_lin_vel = robot.data.root_lin_vel_b  # 基座世界坐标系绝对线速度 (3维)
        # 判断左右脚是否着地（足部法向力绝对值大于 0.5N 视为着地）
        feet_contact = (
            torch.max(
                torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1),
                dim=1,
            )[0]
            > 0.5
        )  # 2维

        # ====== 3. 拼装 Actor 的单帧观测 (3 + 3 + 3 + 20 + 20 + 20 + 2 + 2 + 2 = 75维) ======
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,  # 角速度 * 缩放 (3)
                projected_gravity
                * self.obs_scales.projected_gravity,  # 重力投影 * 缩放 (3)
                command * self.obs_scales.commands,  # 速度指令 * 缩放 (3)
                joint_pos * self.obs_scales.joint_pos,  # 关节角度差 * 缩放 (20)
                joint_vel * self.obs_scales.joint_vel,  # 关节速度 * 缩放 (20)
                action * self.obs_scales.actions,  # 上步动作 * 缩放 (20)
                torch.sin(2 * torch.pi * self.gait_phase),  # 步态相位正弦值 (2)
                torch.cos(2 * torch.pi * self.gait_phase),  # 步态相位余弦值 (2)
                self.phase_ratio,  # 步态腾空占比 (2)
            ],
            dim=-1,
        )

        # ====== 4. 拼装 Critic 的单帧特权观测 (Actor观测 + 真实线速度 + 脚着地状态 = 80维) ======
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact],
            dim=-1,
        )

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        """计算最终包含历史缓存、传感器噪声、高度扫描、相机图象等的网络输入向量。"""
        # 1. 计算单步基础观测
        current_actor_obs, current_critic_obs = self.compute_current_observations()

        # 2. 为 Actor 观测注入噪声以进行 Sim2Real 域随机化
        if self.add_noise:
            current_actor_obs += (
                2 * torch.rand_like(current_actor_obs) - 1
            ) * self.noise_scale_vec

        # 3. 将新的一帧推入历史环形缓冲区
        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        # 4. 展平最近 10 步的历史数据得到最终的 1D 输入向量 (Actor: 750维, Critic: 800维)
        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        # ====== 5. 高度扫描仪（雷达）起伏高度数据拼接 ======
        if self.cfg.scene.height_scanner.enable_height_scan:
            # 计算雷达发射点相对于脚底击中点的相对高度差
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan

            # 特权观测直接拼接雷达高度
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            # Actor 雷达添加高度感应噪声后拼接
            if self.add_noise:
                height_scan += (
                    2 * torch.rand_like(height_scan) - 1
                ) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        # ====== 6. 深度相机图像展平拼接 ======
        if self.cfg.scene.depth_camera.enable_depth_camera:
            depth_image = self.depth_camera.data.output["distance_to_image_plane"]
            flattened_depth = depth_image.view(self.num_envs, -1)  # 展平深度图像为一维

            # 将深度像素数据加到观测末尾
            actor_obs = torch.cat([actor_obs, flattened_depth], dim=-1)
            critic_obs = torch.cat([critic_obs, flattened_depth], dim=-1)

        # ====== 7. 限制观测最大幅度，防梯度爆炸 ======
        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        # ====== 1. 安全检查 ======
        # 如果当前步没有任何环境需要重置，直接返回以节省 CPU/GPU 计算资源
        if len(env_ids) == 0:
            return

        # ====== 2. 清空单步足部物理量缓存 ======
        # 将被重置环境对应的足底平均接触力和平均速度清零
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        # ====== 3. 更新课程学习（Curriculum Learning）地形等级 ======
        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            # 如果启用了地形课程（即走得好就去更难的地形，摔得多就退回到简单地形）
            if self.cfg.scene.terrain_generator.curriculum:
                # 动态评估机器人表现并更新其所处地形的关卡等级
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        # ====== 4. 重置物理场景与执行域随机化事件 ======
        # 4.1 在仿真场景中将机器人和物体传送回对应的起点位置
        self.scene.reset(env_ids)
        # 4.2 触发配置在 "reset" 模式下的事件（如随机化关节初始角度、随机微调机身初始朝向等）
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        # ====== 5. 奖励管理器账目结算与超时标记 ======
        # 5.1 结算被重置环境在上一个回合累积的各项奖励分数，并输出日志字典
        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        # 5.2 记录超时截断标志，用于 RSL-RL 进行值函数自举（Bootstrapping）
        self.extras["time_outs"] = self.time_out_buf

        # ====== 6. 清空历史缓存与重采指令 ======
        # 6.1 为刚复活的机器人重新随机采样一组运动目标速度指令
        self.command_generator.reset(env_ids)
        # 6.2 将 Actor 观测历史缓存、Critic 观测历史缓存以及动作延迟缓存区全部归零（切断历史干扰）
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        # 6.3 将本回合生存步数计数器重置为 0
        self.episode_length_buf[env_ids] = 0

        # ====== 7. 写入物理引擎并同步状态 ======
        # 7.1 将 Python 中更新好（随机化后）的机器人位置和关节数据写入 PhysX 物理仿真引擎
        self.scene.write_data_to_sim()
        # 7.2 物理仿真器执行一次正向同步，确保物理世界状态立即刷新生效
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        # ====== 1. 动作处理（模拟控制延迟与动作裁剪） ======
        # 1.1 从动作延迟缓存区计算当前步应该应用动作，模拟真实机器人通信/硬件执行延迟
        delayed_actions = self.action_buffer.compute(actions)
        # 1.2 动作硬裁剪，防止网络输出异常大范围动作，并搬运到环境设备上
        self.action = torch.clip(
            delayed_actions, -self.clip_actions, self.clip_actions
        ).to(self.device)

        # ====== 2. 动作空间物理映射（将偏差转化为关节目标位置） ======
        processed_actions = (
            self.action * self.action_scale + self.robot.data.default_joint_pos
        )

        # ====== 3. 初始化子步（Decimation）内的传感器平均值缓存 ======
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs,
            len(self.feet_cfg.body_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs,
            len(self.feet_cfg.body_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # ====== 4. 执行微步物理仿真循环 ======
        # PPO 的一个控制步长（比如 20ms）通常包含多个物理引擎微步（比如 4个 5ms 物理步，即 decimation=4）
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            # 4.1 设置关节控制目标角度
            self.robot.set_joint_position_target(processed_actions)
            # 4.2 将最新的控制和物体属性数据写入物理仿真引擎
            self.scene.write_data_to_sim()
            # 4.3 物理引擎仿真推进一步（不在此步渲染以提高速度）
            self.sim.step(render=False)
            # 4.4 从物理引擎读取新状态，更新虚拟场景状态数据
            self.scene.update(dt=self.physics_dt)

            # 4.5 累加当前子步各环境机器人足部的受力大小（L2 范数）
            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3],
                dim=-1,
            )
            # 4.6 累加当前子步足部的线速度大小，用于步态评估与惩罚
            self.avg_feet_speed_per_step += torch.norm(
                self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1
            )

        # ====== 5. 对子步累加的足部力与速度求均值 ======
        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        # ====== 6. 渲染图形界面（如果在非 headless 模式下运行） ======
        if not self.headless:
            self.sim.render()

        # ====== 7. 环境状态与控制命令更新 ======
        self.episode_length_buf += 1  # 增加本回合生存步数计数器
        self._calculate_gait_para()  # 计算周期性步态参数（如支撑相/摆动相比例）

        # ====== 8. 指令发生器与事件管理器步进 ======
        # 更新运动指令（如摇杆给定的前向、横向和自转目标速度）
        self.command_generator.compute(self.step_dt)
        # 定期给环境施加扰动事件（如突然的推力，以增加机器人鲁棒性）
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # ====== 9. 终止判定与奖励计算 ======
        # 9.1 检测各环境是否发生重置（非脚掌部位触地，或者达到最大回合步数超时）
        self.reset_buf, self.time_out_buf = self.check_reset()
        # 9.2 计算所有未发生重置环境在当前控制步长下的累加奖励（包含各项任务奖励与惩罚项）
        reward_buf = self.reward_manager.compute(self.step_dt)
        # 9.3 找到刚刚摔倒或超时需要被重置的环境索引并执行 Reset 重置
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # [修复] 在 reset() 之前保存终态 AMP 观测，确保判别器能看到"摔倒瞬间"的真实状态
        # reset() 执行后关节位置/速度会归为初始值，终态就永久丢失了
        if len(self.reset_env_ids) > 0:
            self.terminal_amp_obs[self.reset_env_ids] = self.get_amp_obs_for_expert_trans()[self.reset_env_ids]
        self.reset(self.reset_env_ids)


        # ====== 10. 计算下一时刻的观测，返回给 PPO 算法更新策略 ======
        actor_obs, critic_obs = self.compute_observations()
        # 将特权观测（Critic 输入）打包记录在 extras 辅助字典中
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        # ====== 1. 获取所有部件的接触力历史缓存 ======
        # 形状为 [num_envs, history_length, num_bodies, 3]
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # ====== 2. 摔倒检测（非脚掌的关键部位与地面产生撞击） ======
        reset_buf = torch.any(
            torch.max(
                # 2.1 过滤出禁忌碰撞部位（如膝盖、躯干、肩膀等），并求三维力的 L2 范数（合外力大小）
                # 形状降维为 [num_envs, history_length, num_termination_bodies]
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                # 2.2 在历史时间步维度寻找最大碰撞力，形状降为 [num_envs, num_termination_bodies]
                dim=1,
            )[0]
            # 2.3 如果任意禁忌部位的碰撞力大于 1.0 牛顿，则判定该部位触地（摔倒）
            > 1.0,
            # 2.4 在禁忌部位轴上求 union，只要有一个部位触地即判定机器人摔倒
            # 最终 reset_buf 形状为 [num_envs]
            dim=1,
        )

        # ====== 3. 超时检测（达到单回合最大生存步数限制） ======
        time_out_buf = self.episode_length_buf >= self.max_episode_length

        # ====== 4. 合并重置信号（摔倒或超时均触发 Reset） ======
        reset_buf |= time_out_buf

        # 同时返回总重置标志与超时标志，用于算法端的未来奖励自举（Value Bootstrapping）
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.lin_vel * self.obs_scales.lin_vel
            noise_vec[3:6] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[6:9] = (
                noise_scales.projected_gravity * self.obs_scales.projected_gravity
            )
            noise_vec[9:12] = 0
            noise_vec[12 : 12 + self.num_actions] = (
                noise_scales.joint_pos * self.obs_scales.joint_pos
            )
            noise_vec[12 + self.num_actions : 12 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[12 + self.num_actions * 2 : 12 + self.num_actions * 3] = 0.0
            noise_vec[12 + self.num_actions * 3 : 18 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = (
                    noise_scales.height_scan * self.obs_scales.height_scan
                )
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length,
            batch_size=self.num_envs,
            device=self.device,
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length,
            batch_size=self.num_envs,
            device=self.device,
        )

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(
            self.robot.data.root_pos_w[env_ids, :2]
            - self.scene.env_origins[env_ids, :2],
            dim=1,
        )
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance
            < torch.norm(self.command_generator.command[env_ids, :2], dim=1)
            * self.max_episode_length_s
            * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(
            self.scene.terrain.terrain_levels.float()
        )
        return extras

    def get_observations(self):
        # 1. 计算当前时刻的 Actor 观测（750维）和 Critic 特权观测
        actor_obs, critic_obs = self.compute_observations()
        # 2. 将 Critic 观测打包放入 extras 字典，以便 Runner 能够提取并喂给 Critic 网络
        self.extras["observations"] = {"critic": critic_obs}
        # 3. 返回 Actor 观测和包含特权观测的 extras 字典
        return actor_obs, self.extras

    def get_amp_obs_for_expert_trans(self):
        """
        计算当前时刻的 52 维 AMP 状态观测，供判别器与专家动捕数据对比。

        输出格式（按 cat 顺序）：
          [0  :10] 右臂+左臂关节角度（各5维）
          [10 :22] 右腿+左腿关节角度（各6维）
          [22 :42] 右臂+左臂+右腿+左腿关节速度（同维度）
          [40 :46] 左手/右手位置（各3D，相对根节点，根坐标系下）
          [46 :52] 左脚/右脚位置（各3D，相对根节点，根坐标系下）
        注：具体维度以实际关节数为准，合计 52 维。

        坐标变换逻辑：
          1. body_pos - root_pos → 相对位置（去掉机器人整体位移）
          2. quat_apply(quat_conjugate(root_quat), rel_pos) → 转到根坐标系（去掉机器人整体朝向）
          这样无论机器人在世界中走到哪个位置/朝哪个方向，AMP 状态只反映"局部姿态"，
          与动捕数据的坐标系一致，判别器才能做有效对比。
        """
        # ====== 手部位置计算（分两步，模型里没有"手"body，用肘+前臂偏移近似）======
        #
        # 第一步（L871-878）：计算"手相对腰的位置向量"，方向仍是世界坐标系
        #   A = 左肘世界坐标 - 腰世界坐标       → 肘相对腰的向量（世界方向）
        #   B = quat_rotate(肘姿态, 前臂本地向量) → 把"肘→手"的本地方向旋转到世界方向
        #   A + B = 手相对腰的位置（世界方向）   ← 此时坐标原点=腰，但坐标轴=世界坐标系
        #
        # 第二步（L889-891）：把坐标轴也转成机器人自身坐标系
        #   quat_apply(quat_conjugate(腰姿态), 上一步结果)
        #   → 手相对腰的位置（根坐标系方向）    ← 坐标原点=腰，坐标轴=机器人自身坐标系 ✓
        #
        # 为什么要两步：仿真器只提供世界坐标系数据，第一步去位移，第二步去朝向，缺一不可。

        # 第一步：世界系下"手相对腰"的位置（坐标轴仍是世界系）
        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]   # 左肘世界坐标 xyz
            - self.robot.data.root_state_w[:, 0:3]                         # 减去腰部世界坐标（去位移）
            + quat_rotate(
                self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7],  # 左肘四元数（世界朝向）
                self.left_arm_local_vec,                                         # 前臂方向（肘本地坐标系，固定值）
            )   # quat_rotate 把本地前臂方向转到世界方向，加上后得到肘→手的世界偏移
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(
                self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7],
                self.right_arm_local_vec,
            )
        )

        # 第二步：把世界系朝向旋转成根坐标系朝向（去机器人整体转向）
        # quat_conjugate(root_quat) = 腰部姿态的逆旋转，等价于"把机器人面朝方向转回正前方"
        left_hand_pos = quat_apply(
            quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos
        )   # 结果：手在根坐标系下的位置（原点=腰，轴=机器人自身坐标系）
        right_hand_pos = quat_apply(
            quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos
        )

        # ====== 脚部位置计算（脚踝关节世界坐标 - 根节点，再转到根坐标系）======
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3]    # 左脚世界坐标
            - self.robot.data.root_state_w[:, 0:3]                         # 减去根节点，得相对位置
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(
            quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos  # 转到根坐标系
        )
        right_foot_pos = quat_apply(
            quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos
        )

        # ====== 读取各关节角度和速度 ======
        self.left_leg_dof_pos  = self.robot.data.joint_pos[:, self.left_leg_ids]   # 左腿关节角度
        self.right_leg_dof_pos = self.robot.data.joint_pos[:, self.right_leg_ids]  # 右腿关节角度
        self.left_leg_dof_vel  = self.robot.data.joint_vel[:, self.left_leg_ids]   # 左腿关节速度
        self.right_leg_dof_vel = self.robot.data.joint_vel[:, self.right_leg_ids]  # 右腿关节速度
        self.left_arm_dof_pos  = self.robot.data.joint_pos[:, self.left_arm_ids]   # 左臂关节角度
        self.right_arm_dof_pos = self.robot.data.joint_pos[:, self.right_arm_ids]  # 右臂关节角度
        self.left_arm_dof_vel  = self.robot.data.joint_vel[:, self.left_arm_ids]   # 左臂关节速度
        self.right_arm_dof_vel = self.robot.data.joint_vel[:, self.right_arm_ids]  # 右臂关节速度

        # ====== 拼接成 52 维 AMP 状态向量并返回 ======
        # 顺序须与 walk.txt 动捕数据的列顺序完全一致，判别器才能做有效对比
        return torch.cat(
            (
                self.right_arm_dof_pos,   # 右臂角度
                self.left_arm_dof_pos,    # 左臂角度
                self.right_leg_dof_pos,   # 右腿角度
                self.left_leg_dof_pos,    # 左腿角度
                self.right_arm_dof_vel,   # 右臂速度
                self.left_arm_dof_vel,    # 左臂速度
                self.right_leg_dof_vel,   # 右腿速度
                self.left_leg_dof_vel,    # 左腿速度
                left_hand_pos,            # 左手位置（根坐标系，3D）
                right_hand_pos,           # 右手位置（根坐标系，3D）
                left_foot_pos,            # 左脚位置（根坐标系，3D）
                right_foot_pos,           # 右脚位置（根坐标系，3D）
            ),
            dim=-1,
        )


    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

    def _calculate_gait_para(self) -> None:
        """
        根据当前仿真时间和各环境的相位偏移，更新左右足部的周期性步态相位。
        """
        # ====== 1. 计算当前时刻在整个步态周期中所占的循环数 ======
        # (生存步数 * 控制周期) 得到当前已生存的真实秒数
        # 除以单个步态周期时间 (gait_cycle, 如 0.5s)，得到当前已经运行了多少个完整的步态周期 (如 3.4 个周期)
        t = self.episode_length_buf * self.step_dt / self.gait_cycle

        # ====== 2. 计算左脚和右脚当前的相位位置 ======
        # 加上配置的初始相位偏移 (phase_offset)，然后对 1.0 进行取余操作（% 1.0），从而将相位约束在 [0.0, 1.0) 之间
        # 0.0 表示周期刚开始，0.5 表示进行到一半，0.99 表示周期即将结束
        # 左脚相位 (通道 0)
        self.gait_phase[:, 0] = (t + self.phase_offset[:, 0]) % 1.0
        # 右脚相位 (通道 1)，左右脚通常相差 0.5 的相位，以实现交替迈步
        self.gait_phase[:, 1] = (t + self.phase_offset[:, 1]) % 1.0
