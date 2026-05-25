# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

import legged_lab.mdp as mdp
from legged_lab.envs.tienkung.walk_cfg import GaitCfg, LiteRewardCfg, TienKungWalkFlatEnvCfg

@configclass
class TienKungWalkPPOFlatEnvCfg(TienKungWalkFlatEnvCfg):
    pass

@configclass
class TienKungWalkPPOAgentCfg(RslRlOnPolicyRunnerCfg):
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
        class_name="PPO",  # Using standard PPO instead of AMPPPO
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
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "OnPolicyRunner"  # Using standard OnPolicyRunner instead of AmpOnPolicyRunner
    experiment_name = "walk_ppo"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk_ppo"
    wandb_project = "walk_ppo"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
