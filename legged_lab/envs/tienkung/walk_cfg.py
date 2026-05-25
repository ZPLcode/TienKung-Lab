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

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.tienkung2_lite import TIENKUNG2LITE_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401


@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.38
    gait_air_ratio_r: float = 0.38
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88
    gait_cycle: float = 0.85


@configclass
class LiteRewardCfg:
    # ====== 1. 任务指标跟踪奖励（正权重） ======
    # 1.1 底盘线速度跟踪奖励：利用指数函数，鼓励机器人的横向和纵向速度完美跟上目标摇杆指令
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    # 1.2 旋转角速度跟踪奖励：鼓励机器人偏航角速度（绕 z 轴自转）完美跟上自转指令
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})

    # ====== 2. 机身姿态稳定惩罚（负权重） ======
    # 2.1 垂直线速度惩罚：惩罚机身在垂直 z 方向上的上下多余窜动，防止蹦跳，使行走更平稳
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # 2.2 横滚/俯仰角速度惩罚：惩罚身体绕 x 和 y 轴的多余晃动，防止机器人左右扭捏和前后摇晃
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 2.3 机身朝向（姿态）惩罚：约束骨盆（Pelvis）滚转角和俯仰角，强迫上半身必须保持直立状态
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}, weight=-2.0
    )
    # 2.4 水平姿态稳定惩罚：进一步迫使机身维持在水平参考面上
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # ====== 3. 硬件安全性与控制平滑度惩罚（负权重） ======
    # 3.1 能量消耗惩罚：计算关节输出机械功耗（力矩*速度），降低电池功耗，鼓励省力的经济步态
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    # 3.2 关节加速度惩罚：惩罚关节速度的剧烈改变，防止电机过度磨损，使驱动顺滑
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # 3.3 动作变化率惩罚：惩罚相邻两个控制周期输出动作的差值，抑制控制高频抖动（实机部署非常关键）
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # 3.4 关节角度极限惩罚：当关节角度逼近限位保护挡板时惩罚，引导关节在舒适角度中段活动
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    # ====== 4. 触地碰撞与摔倒惩罚（负权重） ======
    # 4.1 非预期接触惩罚：惩罚除了足部以外的部位（膝盖、躯干、肩膀等）碰地，起防摔效果
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"]
            ),
            "threshold": 1.0,
        },
    )
    # 4.2 死亡/摔倒惩罚：机器人摔倒重置时扣除巨额分数（-200），让神经网络产生强烈的生存欲望
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # ====== 5. 足部行为规范惩罚（负权重） ======
    # 5.1 脚掌滑动（打滑）惩罚：在脚底接触地面时，惩罚脚掌与地面产生的相对水平滑动速度（防止溜冰）
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
        },
    )
    # 5.2 足底踩地力惩罚：当脚踩地产生的垂直撞击力过大时予以惩罚，防止机器人重重“跺脚”震碎减速器
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    # 5.3 双脚间距过近惩罚：若两脚水平间距小于 20 厘米进行惩罚，防止交叉腿、打架或者绊倒自己
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ankle_roll.*"]), "threshold": 0.2},
    )
    # 5.4 足部绊倒惩罚：惩罚在摆动迈腿时突然撞击前方障碍物的动作
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"])},
    )
    # 5.5 双脚 y 轴间距惩罚：惩罚两脚在宽度方向上不合适（过窄或过宽）的间距，维持标准步宽
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0)

    # ====== 6. 关节姿态偏差惩罚（负权重，塑形人形体态） ======
    # 6.1 髋关节与上半身偏离默认姿态惩罚：让髋关节旋转、肩肘关节不要过度外展，保持手臂微屈贴在身体旁
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_yaw_.*_joint",
                    "hip_roll_.*_joint",
                    "shoulder_pitch_.*_joint",
                    "elbow_pitch_.*_joint",
                ],
            )
        },
    )
    # 6.2 手臂内翻外展偏差惩罚：控制手臂的摆动幅度处于优雅自然的范围
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_roll_.*_joint", "shoulder_yaw_.*_joint"])},
    )
    # 6.3 腿部关节偏离默认姿态惩罚：保持膝关节微屈、脚踝角度中立等
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_pitch_.*_joint",
                    "knee_pitch_.*_joint",
                    "ankle_pitch_.*_joint",
                    "ankle_roll_.*_joint",
                ],
            )
        },
    )

    # ====== 7. 周期性步态相位奖励与对齐（由时间时钟 gait_phase 驱动，核心项） ======
    # 7.1 足底接触力周期对齐奖励：惩罚在摆动相（悬空期）受力，以及支撑相（踩地期）悬空的违规行为
    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02})
    # 7.2 足底线速度周期对齐奖励：惩罚在踩地支撑期间足部还有很高的速度（踩下地不能再前后滑动）
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02})
    # 7.3 足部支撑相力充实奖励：鼓励在需要踩地的周期里踩出合理、饱满的支撑力
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=0.6, params={"delta_t": 0.02})

    # ====== 8. 特定关节专项控制惩罚（负权重，防野蛮动作） ======
    # 8.1 踝关节扭矩惩罚：约束踝关节的出力大小，防止脚踝拼命发力导致舵机发热损坏
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    # 8.2 踝关节动作幅度惩罚：避免脚踝快速扇动导致步态滑稽
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)
    # 8.3 髋关节横滚控制惩罚：防止机器人走路时屁股过度往两边倾斜扭动
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-1.0)
    # 8.4 髋关节偏航控制惩罚：约束大腿自转的动作幅度，防止内八字和外八字
    hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-1.0)


@configclass
class TienKungWalkFlatEnvCfg:
    amp_motion_files_display = ["legged_lab/envs/tienkung/datasets/motion_visualization/walk.txt"]
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=TIENKUNG2LITE_CFG,
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        # terrain_type="plane",
        # terrain_generator= None,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  # (0.3, 0.3)
        ),
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
        feet_body_names=["ankle_roll.*"],
    )
    reward = LiteRewardCfg()
    gait = GaitCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class TienKungWalkAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "walk"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk"
    wandb_project = "walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.3
    amp_motion_files = ["legged_lab/envs/tienkung/datasets/motion_amp_expert/walk.txt"]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 20
