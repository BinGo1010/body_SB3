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
import wandb
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
# 回调：记录实时 GUI 预览（可选）
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
        base = base.unwrapped
    except Exception:
        pass

    sim = getattr(base, "sim", None)
    renderer = getattr(sim, "renderer", None) if sim is not None else None
    if renderer is None or not hasattr(renderer, "render_offscreen"):
        print("[get_frame] 找不到 sim.renderer.render_offscreen，返回 None")
        return None

    size = env_cfg.get("render_size", [480, 480])
    width, height = int(size[0]), int(size[1])

    if camera_id is None:
        camera_id = int(env_cfg.get("render_camera", 0))

    try:
        frame = renderer.render_offscreen(
            width=width,
            height=height,
            camera_id=camera_id,
        )
    except Exception as e:
        print(f"[get_frame] render_offscreen 失败: {type(e).__name__}: {e}")
        return None

    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)

    # 有些版本返回 (3, H, W)，统一转成 (H, W, 3)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))

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
        import gymnasium as gym
        from gymnasium.wrappers import FlattenObservation

        env = gym.make(self.env_id)
        obs, _ = env.reset()
        if isinstance(obs, dict):
            env = FlattenObservation(env)
            obs, _ = env.reset()
        self.eval_env = env
        if self.verbose > 0:
            print(f"[VideoRecorder] 创建视频录制环境: {self.env_id}")

    def _record_video(self):
        self._init_eval_env()
        env = self.eval_env
        ecfg = self.ecfg

        obs, _ = env.reset()
        frames = []

        for t in range(self.rollout_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = get_frame(env, ecfg)
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                break

        if len(frames) == 0:
            print("[VideoRecorder] 未采集到任何帧，跳过此次录制。")
            return

        frames = np.asarray(frames)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.record_idx += 1
        filename = f"walk_step{self.num_timesteps}_idx{self.record_idx}.{self.fmt}"
        out_path = self.out_dir / filename

        # 写视频
        skvideo.io.vwrite(
            str(out_path),
            frames,
            outputdict={
                "-pix_fmt": "yuv420p",
                "-r": str(self.fps),
            },
        )
        print(f"[VideoRecorder] 保存视频: {out_path}")

    def _on_step(self) -> bool:
        # 每隔 save_freq 训练步录一次
        if (self.num_timesteps - self.last_record_step) >= self.save_freq:
            self.last_record_step = self.num_timesteps
            self._record_video()
        return True

    def _on_training_end(self) -> None:
        # 如果想在训练结束额外录一次，也可以在这里再调 self._record_video()
        pass

# ---------------------------------------------------------------------------
# 回调：学习率 + 奖励分解写入 CSV（progress.csv）
# ---------------------------------------------------------------------------

class RewardDecomposeLogger(BaseCallback):

    def __init__(self, window=100, keys_whitelist=None, verbose=0, save_root="reward_logs"):
        super().__init__(verbose)
        self.window = window
        self.buf = defaultdict(lambda: deque(maxlen=window))
        self.keys_whitelist = set(keys_whitelist or [])

        # 创建根目录
        self.save_root = save_root
        os.makedirs(self.save_root, exist_ok=True)

        # 时间戳文件夹（根据训练启动时间）
        self.session_dir = os.path.join(
            self.save_root,
            datetime.now().strftime("%Y-%m-%d_%H-%M")
        )
        os.makedirs(self.session_dir, exist_ok=True)

        # 数据缓存
        self.history = []

    # -------- 工具函数：统一 float 化 --------
    def _to_float(self, v):
        if isinstance(v, (int, float, np.float32, np.float64)):
            return float(v)
        if isinstance(v, np.ndarray) and v.size == 1:
            return float(v.item())
        if isinstance(v, (bool, np.bool_)):
            return float(v)
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
                    lname = k.lower()
                    if (
                        ("reward" in lname)
                        or ("cost" in lname)
                        or ("penalty" in lname)
                        or (k in self.keys_whitelist)
                    ):
                        self._maybe_record(k, v)

        return True

    # -------- rollout 结束：写入 SB3 logger 并缓存 --------
    def _on_rollout_end(self):
        record = {}
        for k, dq in self.buf.items():
            if len(dq) > 0:
                mean_val = float(np.mean(dq))
                self.logger.record(k, mean_val)
                record[k] = mean_val

        # 把这一个 rollout 的记录保存到 history
        if len(record) > 0:
            self.history.append(record)

    # -------- 训练结束：保存 CSV --------
    def _on_training_end(self):
        if len(self.history) == 0:
            print("[RewardLogger] no data to save.")
            return

        df = pd.DataFrame(self.history)
        save_path = os.path.join(self.session_dir, "reward_summary.csv")

        df.to_csv(save_path, index=False)
        print(f"[RewardLogger] reward summary saved to:\n   {save_path}")

        


# ---------------------------------------------------------------------------
# 构建 env + 模型（核心：改为 gym.make('myoLegWalk-v0')）
# ---------------------------------------------------------------------------

def build_model_and_env(
    cfg_path: Path,
    log_root: Path,
    resume_from_path: Optional[Path] = None
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
    train_render = bool(ecfg.get("enable_render", False))  # 训练时是否 render

    # 单环境工厂
    def make_train_env(rank: int = 0):
        def _init():
            env_kwargs: Dict[str, Any] = {}
            if train_render:
                env_kwargs["render_mode"] = "rgb_array"
            # —— 核心：直接用 gym.make 创建 myoLegWalk-v0 环境 ——
            env = gym.make(env_id, **env_kwargs)

            # 设定随机种子
            try:
                env.reset(seed=base_seed + rank)
            except TypeError:
                if hasattr(env, "seed"):
                    env.seed(base_seed + rank)
            # 若 obs 是 dict，则使用 FlattenObservation 转换为 Box
            try:
                obs, _ = env.reset()
                if isinstance(obs, dict):
                    env = FlattenObservation(env)
            except Exception:
                pass
            return env
        return _init

    # VecEnv，向量化环境并行化接口
    if num_envs > 1:
        env = SubprocVecEnv([make_train_env(rank=i) for i in range(num_envs)])
    else:
        env = DummyVecEnv([make_train_env(rank=0)])

    # VecMonitor，保证 episode 统计写入日志（尤其多环境）
    env = VecMonitor(env, filename=str(log_root))

    # policy net 定义网络的隐藏层
    policy_kwargs: Dict[str, Any] = {}
    hidden_cfg = pcfg.get("policy_hidden_sizes", None)
    if hidden_cfg is not None:
        policy_kwargs["net_arch"] = dict(
            pi=list(hidden_cfg.get("pi", [])),
            vf=list(hidden_cfg.get("vf", [])),
        )

    # CPU/GPU 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[build_model_and_env] 使用设备: {device}")
    print(f"[build_model_and_env] 训练环境 ID: {env_id}, num_envs={num_envs}, render={train_render}")

    # 继承/从头训练判断
    if resume_from_path is not None and resume_from_path.is_file():
        print(f"[build_model_and_env] 从已有模型继续训练: {resume_from_path}")
        model = PPO.load(str(resume_from_path), env=env, device=device)
    else:
        print("[build_model_and_env] 未指定已有模型，创建新模型从头训练。")
        model = PPO(
            policy="MlpPolicy",      # 如果 obs 为 Dict 可改为 MultiInputPolicy
            env=env,
            device=device,
            verbose=2,
            n_steps=int(pcfg.get("n_steps", 2048)),
            batch_size=int(pcfg.get("batch_size", 64)),
            n_epochs=int(pcfg.get("n_epochs", 10)),
            learning_rate=pcfg.get("learning_rate", 3e-4),
            gamma=float(pcfg.get("gamma", 0.99)),
            gae_lambda=float(pcfg.get("gae_lambda", 0.95)),
            ent_coef=float(pcfg.get("ent_coef", 0.0)),
            vf_coef=float(pcfg.get("vf_coef", 0.5)),
            use_sde=bool(pcfg.get("use_sde", False)),
            tensorboard_log=str(log_root / "tb"),
            policy_kwargs=policy_kwargs or None,
        )

    # 创建评估环境（单环境，EvalCallback 需要 VecEnv）
    def make_eval_env():
        enable_render = bool(ecfg.get("enable_render", False))
        eval_kwargs: Dict[str, Any] = {}
        if enable_render:
            eval_kwargs["render_mode"] = "rgb_array"
        e = gym.make(env_id, **eval_kwargs)
        try:
            e.reset(seed=base_seed + 10000)
        except TypeError:
            if hasattr(e, "seed"):
                e.seed(base_seed + 10000)
        try:
            o, _ = e.reset()
            if isinstance(o, dict):
                e = FlattenObservation(e)
        except Exception:
            pass
        return e

    eval_env_vec = DummyVecEnv([make_eval_env])
    eval_env_vec = VecMonitor(eval_env_vec)  # 新增这一行，让类型和训练的一致
    
    return {"cfg": cfg, "env": env, "eval_env_vec": eval_env_vec, "model": model}


# ---------------------------------------------------------------------------
# 主流程
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

    # 续训：默认从 best_model 续
    best_model_path = (log_root / "models" / "best" / "best_model.zip").resolve()
    resume_flag = bool(tcfg.get("resume_from_best", False))
    if resume_flag and best_model_path.is_file():
        print(f"[train_walkenv] 检测到 best_model，开启断点续训: {best_model_path}")
        resume_from_path = best_model_path
        reset_num_timesteps = False
    else:
        if resume_flag:
            print(f"[train_walkenv] 配置允许续训，但未找到 best_model: {best_model_path}，从头训练。")
        else:
            print("[train_walkenv] 配置关闭续训，从头训练。")
        resume_from_path = None
        reset_num_timesteps = True

    # 构建 env + model
    build_result = build_model_and_env(cfg_path, log_root=log_root, resume_from_path=resume_from_path)
    cfg = build_result["cfg"]
    env = build_result["env"]
    vcfg = cfg.get("video", {})
    eval_env_vec = build_result["eval_env_vec"]
    model: PPO = build_result["model"]

    ecfg = cfg.get("env", {})
    eval_cfg = cfg.get("eval", {})

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

    # ====== 如果想输出 CSV + TensorBoard，可以启用下面这段 ======
    new_logger = configure(str(log_root), ["stdout", "csv", "tensorboard"])
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
    # 定期保存模型进度
    ckpt_cb = CheckpointCallback(
        save_freq=int(tcfg.get("save_frequency", 50000)),
        save_path=str(run_models_dir),   # 每次训练单独一个时间戳子目录
        name_prefix="ppo_myoLegWalk",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )

    # 学习率 & 奖励分解记录回调
    rwd_cb = RewardDecomposeLogger(
        window=int(tcfg.get("reward_decompose_window", 100)),
        keys_whitelist=tcfg.get("reward_keys", ["dense"]),
        save_root=str(log_root / "reward_logs") 
    )

    callbacks: List[BaseCallback] = [eval_cb, ckpt_cb, rwd_cb]

    # 定期 GUI 预览
    if bool(vcfg.get("enabled", False)):
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

    
    # 重命名 progress.csv，避免覆盖
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



if __name__ == "__main__":
    main()
