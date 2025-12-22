# scripts/train_walkenv.py
"""
单/并行 PPO 训练脚本（myosuite / myoLegWalk-v0）
- 从 body_walk_config.yaml 读取训练/环境/评估参数
- 支持：CSV 日志（progress.csv）、学习率与奖励分解项记录（画“学习率/奖励分解”曲线）
- 环境直接通过 gym.make('myoLegWalk-v0') 创建，不依赖 envs.walk_v0
"""

import os
import sys
import skvideo.io

from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import defaultdict, deque

from datetime import datetime
import pandas as pd

import yaml
import myosuite
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.logger import configure
import wandb
from wandb.integration.sb3 import WandbCallback
from sb3_lattice_policy import LatticeActorCriticPolicy
# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # 假设结构: project_root/{envs, scripts, configs}
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# YAML 读取工具
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML 文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


# ---------------------------------------------------------------------------
# 回调：记录 GUI 预览
# ---------------------------------------------------------------------------

def get_frame(env, env_cfg, camera_id=None):
    """
    从 myosuite 环境里拿一帧 RGB 图像 (H, W, 3, uint8)
    - 自动 unwrap 到最底层 env
    - 使用 sim.renderer.render_offscreen 离屏渲染
    """
    base = env
    # 向下剥 wrapper：Monitor / ObservationWrapper / TimeLimit / VecEnv 子环境等
    for _ in range(10):
        if hasattr(base, "env"):
            new_base = base.env
            if new_base is base:
                break
            base = new_base
        else:
            break

    # gymnasium 通用接口
    try:
        sim = base.sim
    except AttributeError:
        sim = getattr(base, "sim", None)
    if sim is None:
        raise RuntimeError("无法在环境中找到 sim 对象用于渲染。")
    # 渲染一帧
    width, height = env_cfg.get("render_size", [480, 480])
    camera_id = camera_id if camera_id is not None else env_cfg.get("render_camera", 0)

    try:
        frame = sim.renderer.render_offscreen(width, height, camera_id=camera_id)
    except Exception as e:
        print(f"[get_frame] 渲染失败: {e}")
        raise
    return frame


class VideoRecorderCallback(BaseCallback):
    """
    每隔 save_freq 训练步，用当前 model 录制一段视频：
    - 单独创建一个 myoLegWalk-v0 环境，不影响训练环境
    - 使用 get_frame 离屏渲染
    - 保存到 out_dir（一般是 logs/walkenv/videos）
    """

    def __init__(
        self,
        ecfg: Dict[str, Any],
        save_freq: int,
        out_dir: Path,
        rollout_length: int = 1000,
        fps: int = 30,
        fmt: str = "mp4",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.ecfg = ecfg
        self.env_id = ecfg.get("env_id", "myoLegWalk-v0")
        self.save_freq = int(save_freq)
        self.out_dir = Path(out_dir)
        self.rollout_length = int(rollout_length)
        self.fps = int(fps)
        self.fmt = fmt
        self.last_record_step = 0
        self.record_idx = 0
        self.eval_env = None  # 单独用来录视频的 env

    def _init_eval_env(self):
        if self.eval_env is not None:
            return
        print("[VideoRecorder] 创建视频录制环境:", self.env_id)
        env_kwargs: Dict[str, Any] = {}
        self.eval_env = gym.make(self.env_id, **env_kwargs)

    def _record_episode(self):
        self._init_eval_env()
        env = self.eval_env

        obs, _ = env.reset()
        frames = []
        steps = 0

        while steps < self.rollout_length:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            frame = get_frame(env, self.ecfg)
            frames.append(frame)
            steps += 1

            # 录到一个 episode 结束就停，不再 reset 录第二个
            if terminated or truncated:
                break

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.record_idx += 1
        video_path = self.out_dir / f"walk_step{self.num_timesteps}_idx{self.record_idx}.{self.fmt}"
        print(f"[VideoRecorder] 保存视频: {video_path}")

        if self.fmt in ["mp4", "gif"]:
            skvideo.io.vwrite(str(video_path), np.array(frames))
        else:
            print(f"[VideoRecorder] 不支持的视频格式: {self.fmt}")


    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True
        # num_timesteps 是 SB3 内部维护的总步数（包含所有 env）
        if (self.num_timesteps - self.last_record_step) >= self.save_freq:
            self.last_record_step = self.num_timesteps
            self._record_episode()
        return True


# ---------------------------------------------------------------------------
# 回调：奖励分解记录（把 info 里的各项 reward/cost 写到 CSV / TensorBoard）
# ---------------------------------------------------------------------------

class RewardDecomposeCallback(BaseCallback):
    """
    将 info 中的 reward 成分（或指定 keys）记录下来，并输出 CSV。
    """

    def __init__(
        self,
        log_root: Path,
        window_size: int = 100,
        keys_whitelist: Optional[List[str]] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_root = Path(log_root)
        self.window_size = int(window_size)
        self.keys_whitelist = set(keys_whitelist or [])
        self.buf = defaultdict(lambda: deque(maxlen=self.window_size))

    def _to_float(self, v):
        if v is None:
            return None
        if isinstance(v, (float, int, np.floating, np.integer)):
            return float(v)
        if isinstance(v, (np.ndarray, list, tuple)):
            arr = np.asarray(v, dtype=float)
            if arr.size == 0:
                return None
            return float(arr.mean())
        if isinstance(v, dict):
            # 如果是 dict，比如 {"reward": x}
            if "reward" in v:
                return self._to_float(v["reward"])
        return None

    def _maybe_record(self, k, v):
        fv = self._to_float(v)
        if fv is not None:
            self.buf[k].append(fv)

    def _on_rollout_start(self):
        self.buf.clear()

    # -------- 主逻辑：收集 info --------
    def _on_step(self):
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        for info in infos:
            if info is None:
                continue

            # 顶层 info
            for k, v in info.items():
                lname = k.lower()
                if (
                    ("reward" in lname)
                    or ("cost" in lname)
                    or ("penalty" in lname)
                    or (k in self.keys_whitelist)
                ):
                    self._maybe_record(k, v)

            # rwd_dict
            rwd_dict = info.get("rwd_dict", None)
            if isinstance(rwd_dict, dict):
                for k, v in rwd_dict.items():
                    self._maybe_record(k, v)

        return True

    def _on_rollout_end(self):
        if not self.buf:
            return

        # 计算窗口内平均值，并写入 logger（tensorboard/csv）
        for k, dq in self.buf.items():
            if len(dq) == 0:
                continue
            mean_v = float(np.mean(dq))
            # 统一放在 "reward_decompose/xxx"
            self.logger.record(f"reward_decompose/{k}", mean_v)

        # 写入 logger 后清空
        self.buf.clear()


# ---------------------------------------------------------------------------
# 环境构建
# ---------------------------------------------------------------------------

def make_env_factory(env_id: str, rank: int, base_seed: int, env_cfg: Dict[str, Any]):
    """
    返回一个函数 _init()，用于创建单个 gym 环境实例。
    """
    def _init():
        env_kwargs: Dict[str, Any] = {}
        env = gym.make(env_id, **env_kwargs)
        # 设定随机种子
        try:
            env.reset(seed=base_seed + rank)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(base_seed + rank)
        return env
    return _init


def build_model_and_env(
    cfg_path: Path,
    log_root: Path,
    resume_from_path: Optional[Path] = None,
    tb_log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    根据 YAML 构建训练 VecEnv + PPO 模型；logger 在 main 里用 log_root 配好后 set_logger。
    环境直接使用 gym.make('myoLegWalk-v0')。
    """
    cfg = load_yaml(cfg_path)
    ecfg = cfg.get("env", {})
    pcfg = cfg.get("ppo", {})

    # 环境 ID，可以在 YAML 中 env.env_id 覆盖，默认 myoLegWalk-v0
    env_id = ecfg.get("env_id", "myoLegWalk-v0")

    base_seed = int(ecfg.get("seed", 42))
    num_envs = int(ecfg.get("num_envs", 1))
    print(f"[build_model_and_env] base_seed={base_seed}, num_envs={num_envs}")

    # 启用并行环境
    if num_envs > 1:
        def make_train_env(rank: int = 0):
            def _init():
                env_kwargs: Dict[str, Any] = {}
                # —— 核心：直接用 gym.make 创建 myoLegWalk-v0 环境 ——
                env = gym.make(env_id, **env_kwargs)

                # 设定随机种子
                try:
                    env.reset(seed=base_seed + rank)
                except TypeError:
                    if hasattr(env, "seed"):
                        env.seed(base_seed + rank)
                return env
            return _init

        env_fns = [make_train_env(rank=i) for i in range(num_envs)]
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv([make_env_factory(env_id, 0, base_seed, ecfg)])
    # VecMonitor，保证 episode 统计写入日志（尤其多环境）
    env = VecMonitor(env, filename=str(log_root))

    # policy net 定义网络的隐藏层
    # policy_kwargs: Dict[str, Any] = {}
    # hidden_cfg = pcfg.get("policy_hidden_sizes", None)
    # if hidden_cfg is not None:
    #     policy_kwargs["net_arch"] = dict(
    #         pi=list(hidden_cfg.get("pi", [])),
    #         vf=list(hidden_cfg.get("vf", [])),
    #     )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        lattice_init_log_std=-0.5,
        lattice_fix_std=False,     # True=固定探索强度；False=可学习探索强度
        lattice_eps=1e-6,
    )

    # CPU/GPU 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[build_model_and_env] 使用设备: {device}")
    print(f"[build_model_and_env] 训练环境 ID: {env_id}, num_envs={num_envs}")

    # 继承/从头训练判断
    if resume_from_path is not None and resume_from_path.is_file():
        print(f"[build_model_and_env] 从已有模型继续训练: {resume_from_path}")
        model = PPO.load(str(resume_from_path), env=env, device=device)
    else:
        print("[build_model_and_env] 未指定已有模型，创建新模型从头训练。")
        model = PPO(
            policy=LatticeActorCriticPolicy,# 如果 obs 为 Dict 可改为 MultiInputPolicy
            env=env,
            device=device,
            verbose=2,
            n_steps=int(pcfg.get("n_steps", 2048)),
            batch_size=int(pcfg.get("batch_size", 256)),
            n_epochs=int(pcfg.get("n_epochs", 10)),
            gamma=float(pcfg.get("gamma", 0.99)),
            gae_lambda=float(pcfg.get("gae_lambda", 0.95)),
            learning_rate=float(pcfg.get("learning_rate", 3e-4)),
            ent_coef=float(pcfg.get("ent_coef", 0.0)),
            vf_coef=float(pcfg.get("vf_coef", 0.5)),
            use_sde=bool(pcfg.get("use_sde", False)),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tb_log_dir) if tb_log_dir is not None else None,
        )

    # 单独评估环境（一般用 DummyVecEnv 包一次）
    eval_env_vec = DummyVecEnv([make_env_factory(env_id, 1000, base_seed, ecfg)])
    eval_env_vec = VecMonitor(eval_env_vec, filename=None)
    return {
        "env": env,
        "eval_env_vec": eval_env_vec,
        "model": model,
        "env_id": env_id,
    }


# ---------------------------------------------------------------------------
# 主训练入口
# ---------------------------------------------------------------------------

def main():
    # 配置路径
    if len(sys.argv) > 1:
        cfg_path = Path(sys.argv[1])
    else:
        cfg_path = PROJECT_ROOT / "configs" / "body_walk_config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"未找到配置文件: {cfg_path}")
    # 本次训练的时间戳（启动时间）
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # 先读一次配置，确定日志与续训
    raw_cfg = load_yaml(cfg_path)
    tcfg = raw_cfg.get("train", {})
    log_root = Path(tcfg.get("log_dir", "logs/walkenv")).resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    # ---- 初始化 Weights & Biases (W&B) 实验记录 ----
    wandb_cfg = raw_cfg.get("wandb", {})
    use_wandb = bool(wandb_cfg.get("wandb_enabled", True))
    wandb_run = None
    if use_wandb:
        # 将当前 YAML 中的关键配置同步到 W&B
        wandb_config = {
            "env": raw_cfg.get("env", {}),
            "train": tcfg,
            "ppo": raw_cfg.get("ppo", {}),
            "reward_weights": raw_cfg.get("reward_weights", {}),
        }
        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "myoLegWalk"),
            name=wandb_cfg.get("name", run_id),
            config=wandb_config,
            dir=str(log_root),
            sync_tensorboard=True,
        )
        print(f"[train_walkenv] 已启动 W&B 实验: project={wandb_cfg.get('project', 'myoLegWalk')}, name={wandb_cfg.get('name', run_id)}")
    else:
        print("[train_walkenv] W&B 未启用（wandb.enabled = False）")
    # === 每个 run 单独的 TensorBoard 目录 ===
    if use_wandb and wandb_run is not None:
        tb_log_dir = log_root / "runs" / wandb_run.id
    else:
        tb_log_dir = log_root / "runs" / run_id
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    # 续训：默认从 best_model 续
    best_model_path = (log_root / "models" / "best" / "best_model.zip").resolve()
    resume_flag = bool(tcfg.get("resume_from_best", False))
    if resume_flag and best_model_path.is_file():
        print(f"[train_walkenv] 检测到 best_model，准备从其继续训练: {best_model_path}")
        resume_from_path = best_model_path
        reset_num_timesteps = False
    else:
        print("[train_walkenv] 不从 best_model 续训，重新开始计步。")
        resume_from_path = None
        reset_num_timesteps = True

    # 构建环境+模型
    build_result = build_model_and_env(
    cfg_path,
    log_root,
    resume_from_path=resume_from_path,
    tb_log_dir=tb_log_dir,
    )   
    env = build_result["env"]
    eval_env_vec = build_result["eval_env_vec"]
    model: PPO = build_result["model"]

    ecfg = raw_cfg.get("env", {})
    eval_cfg = raw_cfg.get("eval", {})

    total_timesteps = int(tcfg.get("total_timesteps", 100000))

    # 准备目录
    models_root = (log_root / "models")          # 模型总目录
    models_root.mkdir(parents=True, exist_ok=True)

    # 本次训练的模型子目录（用于 checkpoint 和最终模型）
    run_models_dir = models_root / run_id
    run_models_dir.mkdir(parents=True, exist_ok=True)

    # eval 日志根目录 + 本次训练子目录（时间戳）
    eval_root = (log_root / "eval")
    eval_log_path = (eval_root / run_id)
    eval_log_path.mkdir(parents=True, exist_ok=True)

    # ====== 输出 CSV + TensorBoard ======
    new_logger = configure(str(tb_log_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # 评估最佳模型 / checkpoint 回调
    eval_cb = EvalCallback(
        eval_env_vec,
        best_model_save_path=str(models_root / "best"),
        log_path=str(eval_log_path),
        eval_freq=int(eval_cfg.get("eval_freq", 10000)),
        n_eval_episodes=int(eval_cfg.get("n_episodes", 5)),
        deterministic=bool(eval_cfg.get("deterministic", True)),
        render=bool(eval_cfg.get("eval_render", False)),
    )

    ckpt_cb = CheckpointCallback(
        save_freq=int(tcfg.get("save_frequency", 50000)),
        save_path=str(run_models_dir),
        name_prefix="ppo_myoLegWalk",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    # 奖励分解记录回调
    reward_window = int(tcfg.get("reward_decompose_window", 0))
    reward_keys = tcfg.get("reward_keys", [])
    if reward_window > 0:
        rwd_cb = RewardDecomposeCallback(
            log_root=log_root,
            window_size=reward_window,
            keys_whitelist=reward_keys,
            verbose=1,
        )
    else:
        rwd_cb = RewardDecomposeCallback(
            log_root=log_root,
            window_size=100,
            keys_whitelist=reward_keys,
            verbose=0,
        )

    callbacks: List[BaseCallback] = [eval_cb, ckpt_cb, rwd_cb]

    # 如果启用了 W&B，则加入 WandbCallback，将训练过程同步到 W&B
    if "wandb_run" in locals() and wandb_run is not None:
        wandb_cb = WandbCallback(
            gradient_save_freq=0,      # 不保存梯度
            model_save_path=None,      # 模型保存仍然由 CheckpointCallback 管理
            verbose=1,
        )
        callbacks.append(wandb_cb)

    # 定期 GUI 预览
    vcfg = raw_cfg.get("video", {})
    if bool(vcfg.get("video_enabled", False)):
        video_root = (PROJECT_ROOT / vcfg.get("out_dir", "logs/walkenv/videos")).resolve()
        video_out_dir = (video_root / run_id).resolve()
        video_cb = VideoRecorderCallback(
            ecfg=ecfg,
            save_freq=int(tcfg.get("save_frequency", 50000)),  # 用同一个 save_frequency
            out_dir=video_out_dir,
            rollout_length=int(vcfg.get("rollout_length", 1000)),
            fps=int(vcfg.get("fps", 30)),
            fmt=str(vcfg.get("format", "mp4")),
            verbose=1,
        )
        callbacks.append(video_cb)

    # 开始训练
    print(f"[train_walkenv] 本次训练步数 = {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        tb_log_name="ppo_myoLegWalk",
        reset_num_timesteps=reset_num_timesteps,
    )

    # 保存最终模型
    final_model_path = run_models_dir / "ppo_myoLegWalk_final"
    model.save(final_model_path)
    print(f"[train_walkenv] 训练结束，最终模型已保存到: {final_model_path}")
    
    #    ========= 上传 best_model.zip 到 W&B =========
    # best_model 由 EvalCallback 保存在 logs/walkenv/models/best/best_model.zip
    best_model_path = (models_root / "best" / "best_model.zip").resolve()
    if best_model_path.is_file() and "wandb_run" in locals() and wandb_run is not None:
        try:
            artifact = wandb.Artifact(
                name=f"best_model_{run_id}",  # 每次 run 一个独立名字
                type="model",
                metadata={
                    "env_id": raw_cfg.get("env", {}).get("env_id", "myoLegWalk-v0"),
                    "total_timesteps": total_timesteps,
                },
            )
            artifact.add_file(str(best_model_path))
            wandb_run.log_artifact(artifact)
            # 如果希望脚本退出前确保上传完成，可以等待：
            artifact.wait()
            print(f"[train_walkenv] 已将 best_model 上传到 W&B Artifact: {best_model_path}")
        except Exception as e:
            print(f"[train_walkenv] 上传 best_model 到 W&B 失败: {e}")
    else:
        print("[train_walkenv] 未找到 best_model.zip 或 W&B 未启用，跳过上传。")

    # 重命名 progress.csv，加上时间戳
    try:
        progress_path = log_root / "progress.csv"
        if progress_path.is_file():
            timed_progress_path = log_root / f"progress_{run_id}.csv"
            progress_path.rename(timed_progress_path)
            print(f"[train_walkenv] progress.csv 已重命名为: {timed_progress_path}")
        else:
            print("[train_walkenv] 未找到 progress.csv，跳过重命名。")
    except Exception as e:
        print(f"[train_walkenv] 重命名 progress.csv 失败: {e}")

    env.close()
    eval_env_vec.close()

    # 结束 W&B 运行
    if "wandb_run" in locals() and wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
