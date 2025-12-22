#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联合强化学习训练脚本：
- 环境：WalkEnvV4Multi（人体肌肉激活 + 外骨骼 CPG 助力）
- 观测：Dict(human, exo)
- 策略：SB3-PPO + MultiInputPolicy + 自定义特征提取器 HumanExoExtractor
- 可选 WandB 日志
"""

import os
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch as th

import gym
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 如果你使用 gymnasium，可以改为 from gymnasium import spaces
# 并确保 WalkEnvV4Multi 里用同一套 gym/gymnasium API

# 可选 WandB
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WandbCallback = None
    WANDB_AVAILABLE = False

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 这里按你之前的文件名导入
from envs.walk_gait_exo import WalkEnvV4Multi


# ============================================================
# 1. 特征提取器：Dict(human, exo) → 统一特征向量
# ============================================================

class HumanExoExtractor(BaseFeaturesExtractor):
    """
    观测空间: Dict("human", "exo")
      - "human": (dim_h,)
      - "exo":   (dim_e,)

    思路：
      - human 分支：MLP 编码人体观测（与旧人体策略 obs_keys 对齐）
      - exo 分支：MLP 编码外骨骼观测（qpos/qvel + CPG/相对位置等）
      - 最终将两个 latent 拼接为一个特征向量，供 PPO 的 actor / critic 使用
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # features_dim 会被我们内部重写
        super().__init__(observation_space, features_dim)

        assert isinstance(observation_space, spaces.Dict), \
            "HumanExoExtractor 只支持 Dict 观测空间"

        human_space = observation_space.spaces["human"]
        exo_space = observation_space.spaces["exo"]

        assert isinstance(human_space, spaces.Box)
        assert isinstance(exo_space, spaces.Box)

        human_dim = human_space.shape[0]
        exo_dim = exo_space.shape[0]

        hidden_h = 128
        hidden_e = 128

        self.human_net = th.nn.Sequential(
            th.nn.Linear(human_dim, hidden_h),
            th.nn.ReLU(),
            th.nn.Linear(hidden_h, hidden_h),
            th.nn.ReLU(),
        )

        self.exo_net = th.nn.Sequential(
            th.nn.Linear(exo_dim, hidden_e),
            th.nn.ReLU(),
            th.nn.Linear(hidden_e, hidden_e),
            th.nn.ReLU(),
        )

        # 最终特征维度 = human 分支 + exo 分支
        self._features_dim = hidden_h + hidden_e

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        # SB3 的 MultiInputPolicy 会传入一个 dict-like 的 tensor 容器
        human_obs = observations["human"]
        exo_obs = observations["exo"]

        h_latent = self.human_net(human_obs)
        e_latent = self.exo_net(exo_obs)

        return th.cat([h_latent, e_latent], dim=-1)


# ============================================================
# 2. 环境工厂：方案 A，直接实例化 WalkEnvV4Multi
# ============================================================

def make_env(rank: int,
             base_seed: int,
             env_kwargs: Dict[str, Any]):
    """
    用于 DummyVecEnv / SubprocVecEnv 的工厂函数。
    """
    def _init():
        env = WalkEnvV4Multi(
            seed=base_seed + rank,
            **env_kwargs,
        )
        return env
    return _init


# ============================================================
# 3. 训练主逻辑
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # ---- 基本路径与环境参数 ----
    parser.add_argument("--model_xml", type=str, required=True,
                        help="Mujoco XML 路径（myoleg+exo 穿戴后的模型）")
    parser.add_argument("--obsd_xml", type=str, default=None,
                        help="可选：观测用 XML，不指定则与 model_xml 相同")

    parser.add_argument("--reset_type", type=str, default="init",
                        choices=["init", "random", "key0"],
                        help="环境初始姿态类型，对应 WalkEnvV4Multi.reset_type")
    parser.add_argument("--target_y_vel", type=float, default=1.2,
                        help="期望前进速度 (m/s)")
    parser.add_argument("--hip_period", type=int, default=100,
                        help="hip CPG 周期（步频），与 WalkEnvV4Multi 内部一致")

    # ---- RL 超参数 ----
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--clip_range", type=float, default=0.2)

    # ---- 日志与保存 ----
    parser.add_argument("--logdir", type=str, default="./logs_exo_joint",
                        help="训练日志与模型保存根目录")
    parser.add_argument("--run_name", type=str, default="exo_joint_walk",
                        help="本次训练标识（用于目录/W&B run 名称）")
    parser.add_argument("--eval_freq", type=int, default=50_000,
                        help="评估频率（以 env step 计）")
    parser.add_argument("--save_freq", type=int, default=200_000,
                        help="模型 checkpoint 保存频率（env step）")

    parser.add_argument("--device", type=str, default="auto",
                        help='"auto", "cpu", "cuda" 等')

    # ---- W&B 配置（可选）----
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否启用 Weights & Biases 日志")
    parser.add_argument("--wandb_project", type=str, default="exo_joint_rl",
                        help="WandB project 名称")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="WandB entity（team/user）")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="WandB group 名称")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run 名，不指定则与 run_name 相同")

    # ---- 续训 ----
    parser.add_argument("--resume_path", type=str, default=None,
                        help="可选：从已有 PPO 模型 .zip 继续训练")

    return parser.parse_args()


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    set_global_seeds(args.seed)

    model_xml = Path(args.model_xml).expanduser().resolve()
    obsd_xml = Path(args.obsd_xml).expanduser().resolve() if args.obsd_xml is not None else model_xml

    log_root = Path(args.logdir).expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    run_dir = log_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------- 构造环境参数 --------
    env_kwargs = dict(
        model_path=str(model_xml),
        obsd_model_path=str(obsd_xml),
        reset_type=args.reset_type,
        target_y_vel=args.target_y_vel,
        hip_period=args.hip_period,
    )

    base_seed = args.seed
    num_envs = args.num_envs

    # -------- 训练环境：SubprocVecEnv / DummyVecEnv + VecMonitor --------
    if num_envs > 1:
        env_fns = [make_env(rank=i, base_seed=base_seed, env_kwargs=env_kwargs)
                   for i in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv([make_env(rank=0, base_seed=base_seed, env_kwargs=env_kwargs)])

    vec_env = VecMonitor(vec_env, filename=str(run_dir / "monitor_train.csv"))

    # -------- 评估环境 --------
    eval_env = DummyVecEnv([make_env(rank=1000, base_seed=base_seed, env_kwargs=env_kwargs)])
    eval_env = VecMonitor(eval_env, filename=None)

    # -------- PPO 策略设置：MultiInputPolicy + HumanExoExtractor --------
    policy_kwargs = dict(
        features_extractor_class=HumanExoExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(
            pi=[256, 128],
            vf=[256, 128],
        ),
    )

    # -------- WandB 初始化（可选） --------
    wandb_run = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("[WARN] 未安装 wandb 或 wandb.integration.sb3，忽略 --use_wandb 选项")
        else:
            wandb_run_name = args.wandb_run_name or args.run_name
            wandb_config = vars(args).copy()
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                name=wandb_run_name,
                config=wandb_config,
                sync_tensorboard=True,
            )

    # -------- 构造 / 续训 PPO 模型 --------
    device = args.device

    if args.resume_path is not None and Path(args.resume_path).is_file():
        print(f"[INFO] 从 {args.resume_path} 加载 PPO 模型并续训")
        model = PPO.load(
            path=args.resume_path,
            env=vec_env,
            device=device,
        )
    else:
        print("[INFO] 新建 PPO 模型（MultiInputPolicy + HumanExoExtractor）")
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            device=device,
            verbose=2,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            clip_range=args.clip_range,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(run_dir / "tb"),
        )

    # -------- 回调：评估 + checkpoint (+ 可选 WandB) --------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval"),
        eval_freq=max(args.eval_freq // num_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // num_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_exo_joint",
    )

    callbacks = [eval_callback, checkpoint_callback]

    if args.use_wandb and WANDB_AVAILABLE:
        wandb_callback = WandbCallback(
            gradient_save_freq=0,      # 一般不用保存梯度
            model_save_freq=0,         # 模型保存通过 CheckpointCallback 控制
            verbose=1,
        )
        callbacks.append(wandb_callback)

    # -------- 开始训练 --------
    print(f"[INFO] 开始训练，总步数: {args.total_timesteps}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
    )

    # -------- 保存最终模型 --------
    final_model_path = run_dir / "ppo_exo_joint_final.zip"
    model.save(str(final_model_path))
    print(f"[INFO] 训练完成，最终模型已保存至: {final_model_path}")

    # -------- 结束 WandB --------
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
