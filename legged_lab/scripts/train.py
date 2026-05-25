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

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
from datetime import datetime

import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

# ====== PyTorch GPU 计算与加速设置 ======
torch.backends.cuda.matmul.allow_tf32 = True  # 允许矩阵乘法使用 TF32 精度（Ampere 及以上架构显卡可加速 8-10 倍）
torch.backends.cudnn.allow_tf32 = True       # 允许 cuDNN 卷积使用 TF32 精度
torch.backends.cudnn.deterministic = False   # 禁用确定性算法以提升运行速度（强化学习有随机探索，不需要严格的确定性）
torch.backends.cudnn.benchmark = False       # 禁用 cuDNN 自动基准寻找，因为强化学习输入尺寸固定，开启它反而可能因初始化开销导致变慢


def train():
    runner: OnPolicyRunner | AmpOnPolicyRunner

    # ====== 1. 从任务注册表中获取配置和环境类 ======
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)   # 获取该任务的 EnvCfg 和 AgentCfg 配置实例
    env_class = task_registry.get_task_class(env_class_name)     # 获取对应的环境类（例如 TienKungEnv）

    # ====== 2. 使用命令行参数覆盖默认配置 ======
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs               # 覆盖并行环境数量

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)           # 根据 CLI 参数更新 RSL-RL 训练器配置
    env_cfg.scene.seed = agent_cfg.seed                          # 保持环境种子与智能体种子一致

    # ====== 3. 分布式多卡训练设置 ======
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"   # 将仿真器绑定到特定的 GPU 核心
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"     # 将策略网络绑定到相同的 GPU 核心

        # 为不同 GPU 进程设置不同的种子，以确保数据采样的多样性
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    # ====== 4. 实例化环境 ======
    env = env_class(env_cfg, args_cli.headless)                  # 这会触发 TienKungEnv.__init__()，完成仿真世界搭建

    # ====== 5. 设置日志保存路径 ======
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # 用当前时间命名运行文件夹，防止覆盖历史实验
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    
    # ====== 6. 实例化训练器 Runner (OnPolicyRunner 或 AmpOnPolicyRunner) ======
    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # ====== 7. 加载历史 Checkpoint (如果开启了恢复训练) ======
    if agent_cfg.resume:
        # 获取最新的或指定的模型权重路径
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # 加载历史模型参数以继续训练
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train()
    simulation_app.close()
