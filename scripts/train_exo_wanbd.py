import os
import sys
import argparse
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List
from collections import defaultdict, deque
from datetime import datetime

import yaml
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
import mujoco
import skvideo.io

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ---------------------------------------------------------------------------
# 路径与工程导入
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.walk_gait_exo import WalkEnvV4Multi


# ---------------------------------------------------------------------------
# YAML 工具
# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML 文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def pick_config_path(cli_config: Optional[str]) -> Path:
    if cli_config:
        return Path(cli_config).expanduser().resolve()
    env_cfg = os.environ.get("BODY_EXO_WALK_CONFIG", "").strip()
    if env_cfg:
        return Path(env_cfg).expanduser().resolve()
    p1 = Path("/home/lvchen/body_SB3/configs/body_exo_walk_config.yaml")
    if p1.is_file():
        return p1.resolve()
    return (PROJECT_ROOT / "configs" / "body_exo_walk_config.yaml").resolve()


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)



# ---------------------------------------------------------------------------
# W&B（wandb）日志上传（可选）
# ---------------------------------------------------------------------------
def init_wandb(cfg: Dict[str, Any], run_dir: Path, run_name: str, mode_tag: str, ts_dir: str):
    """按 YAML 的 wandb 配置初始化 W&B，并返回 (wandb_run, wandb_callback, enabled)。
    兼容字段：
      wandb:
        wandb_enabled: true   # 或 enabled: true
        project: "xxx"
        name: "yyy"
    """
    wcfg = cfg.get("wandb", {}) or {}
    enabled = bool(wcfg.get("wandb_enabled", wcfg.get("enabled", False)))
    if not enabled:
        return None, None, False

    try:
        import wandb  # type: ignore
        from wandb.integration.sb3 import WandbCallback  # type: ignore
    except Exception as e:
        print(f"[train_exo] WARN: wandb is enabled in YAML but import failed: {repr(e)}")
        return None, None, False

    project = str(wcfg.get("project", "body_exo_SB3"))
    name = wcfg.get("name", None)
    if name is None or str(name).strip() == "":
        name = f"{run_name}_{ts_dir}"

    entity = wcfg.get("entity", None)
    group = wcfg.get("group", None)
    tags = wcfg.get("tags", None)
    notes = wcfg.get("notes", None)
    mode = str(wcfg.get("mode", "online"))  # online/offline/disabled
    sync_tb = bool(wcfg.get("sync_tensorboard", True))
    save_code = bool(wcfg.get("save_code", True))

    # 将 wandb 本地缓存统一放在 run_dir/wandb 下，方便归档
    wandb_root = (run_dir / "wandb").resolve()
    (wandb_root / "cache").mkdir(parents=True, exist_ok=True)
    (wandb_root / "config").mkdir(parents=True, exist_ok=True)
    (wandb_root / "artifacts").mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_DIR"] = str(wandb_root)
    os.environ["WANDB_CACHE_DIR"] = str(wandb_root / "cache")
    os.environ["WANDB_CONFIG_DIR"] = str(wandb_root / "config")
    os.environ["WANDB_ARTIFACT_DIR"] = str(wandb_root / "artifacts")

    # 尽量把配置写进 wandb.config（需 JSON-serializable）
    wb_config = {
        "mode_tag": mode_tag,
        "run_dir": str(run_dir),
        "ts_dir": ts_dir,
        "run_name": run_name,
        "env": cfg.get("env", {}) or {},
        "train": cfg.get("train", {}) or {},
        "ppo": cfg.get("ppo", {}) or {},
        "eval": cfg.get("eval", {}) or {},
    }

    try:
        wandb_run = wandb.init(
            project=project,
            name=str(name),
            entity=entity,
            group=group,
            tags=tags,
            notes=notes,
            dir=str(wandb_root),
            config=wb_config,
            mode=mode,
            sync_tensorboard=sync_tb,
            save_code=save_code,
        )
    except Exception as e:
        print(f"[train_exo] WARN: wandb.init failed: {repr(e)}")
        return None, None, False

    # WandbCallback：同步 SB3 logger（尤其是 TB）到 wandb
    model_save_freq = int(wcfg.get("model_save_freq", 0))
    gradient_save_freq = int(wcfg.get("gradient_save_freq", 0))
    model_save_path = str((run_dir / "wandb_models").resolve())

    wandb_cb = WandbCallback(
        gradient_save_freq=gradient_save_freq,
        model_save_path=model_save_path,
        model_save_freq=model_save_freq,
        verbose=0,
    )

    print(f"[train_exo] wandb initialized: project={project} | name={name} | mode={mode} | sync_tensorboard={sync_tb}")
    return wandb_run, wandb_cb, True


def _to_jsonable(x: Any) -> Any:
    """将对象递归转换为 wandb.config 可接受的 JSON-serializable 结构。"""
    # 基础类型
    if x is None or isinstance(x, (str, bool, int, float)):
        return x

    # numpy / torch
    try:
        import numpy as _np
        if isinstance(x, (_np.floating, _np.integer)):
            return x.item()
        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass

    try:
        import torch as _th
        if isinstance(x, _th.Tensor):
            return x.detach().cpu().numpy().tolist()
    except Exception:
        pass

    # 容器
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # 兜底：转字符串，避免 wandb.config.update 报错
    return str(x)


def _save_yaml(obj: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def log_wandb_artifact_best_model(
    wandb_run,
    best_model_path: Path,
    mode_tag: str,
    ts_dir: str,
    reward_weights_used: Dict[str, Any],
    extra_meta: Optional[Dict[str, Any]] = None,
):
    """将 best_model.zip 上传到 W&B Artifacts（type='model'）。"""
    try:
        import wandb  # type: ignore
    except Exception:
        return

    best_model_path = Path(best_model_path)
    if not best_model_path.is_file():
        return

    meta = {
        "mode": mode_tag,
        "timestamp_dir": ts_dir,
        "reward_weights": _to_jsonable(reward_weights_used),
    }
    if isinstance(extra_meta, dict):
        meta.update(_to_jsonable(extra_meta))

    # artifact 名在同一 project 内应稳定，W&B 会自动递增版本（v0, v1, ...）
    art_name = f"best_{mode_tag}_model"
    art = wandb.Artifact(name=art_name, type="model", metadata=meta)
    art.add_file(str(best_model_path), name=best_model_path.name)

    # aliases: latest + 时间戳（便于回溯）
    aliases = ["latest", ts_dir]
    try:
        wandb_run.log_artifact(art, aliases=aliases)
    except Exception:
        # 部分 wandb 版本不支持 aliases 参数：退化为 log_artifact(art)
        wandb_run.log_artifact(art)



# ---------------------------------------------------------------------------
# 训练时可以在 TensorBoard 中实时查看
# ---------------------------------------------------------------------------
# class DictToBoxObs(gym.ObservationWrapper):
#     """将 Dict 观测转换为单个 Box 向量（默认取 key='exo'），不在 __init__ 里 reset。"""

#     def __init__(self, env, key: str = "exo"):
#         super().__init__(env)
#         self.key = key

#         src_space = env.observation_space
#         if isinstance(src_space, spaces.Dict):
#             if self.key not in src_space.spaces:
#                 raise KeyError(
#                     f"DictToBoxObs: observation_space keys={list(src_space.spaces.keys())}, missing '{self.key}'"
#                 )
#             src_space = src_space.spaces[self.key]

#         if not isinstance(src_space, spaces.Box):
#             raise TypeError(f"DictToBoxObs: expected Box space for '{self.key}', got {type(src_space)}")

#         flat_dim = int(np.prod(src_space.shape))
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
#         )

#     def observation(self, obs):
#         if isinstance(obs, dict):
#             obs = obs[self.key]
#         return np.asarray(obs, dtype=np.float32).ravel()
class DictToBoxObs(gym.ObservationWrapper):
    """将 Dict 观测转换为单个 Box 向量，并可选裁剪到 [-clip_obs, clip_obs] 以匹配历史模型空间。"""

    def __init__(self, env, key: str = "exo", clip_obs: float = 10.0, force_clip_bounds: bool = True):
        super().__init__(env)
        self.key = key
        self.clip_obs = float(clip_obs)
        self.force_clip_bounds = bool(force_clip_bounds)

        src_space = env.observation_space
        if isinstance(src_space, spaces.Dict):
            if self.key not in src_space.spaces:
                raise KeyError(
                    f"DictToBoxObs: observation_space keys={list(src_space.spaces.keys())}, missing '{self.key}'"
                )
            src_space = src_space.spaces[self.key]

        if not isinstance(src_space, spaces.Box):
            raise TypeError(f"DictToBoxObs: expected Box space for '{self.key}', got {type(src_space)}")

        flat_dim = int(np.prod(src_space.shape))

        # 尽量继承原始 low/high；若为 ±inf 或强制要求，则用 [-clip_obs, clip_obs]
        low = np.asarray(src_space.low, dtype=np.float32).ravel()
        high = np.asarray(src_space.high, dtype=np.float32).ravel()
        if self.force_clip_bounds or np.isinf(low).any() or np.isinf(high).any():
            low = np.full((flat_dim,), -self.clip_obs, dtype=np.float32)
            high = np.full((flat_dim,), self.clip_obs, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        if isinstance(obs, dict):
            obs = obs[self.key]
        x = np.asarray(obs, dtype=np.float32).ravel()
        # 与 observation_space 一致：裁剪
        if self.clip_obs is not None:
            x = np.clip(x, -self.clip_obs, self.clip_obs)
        return x


# ---------------------------------------------------------------------------
# 特征提取（Dict obs）
# ---------------------------------------------------------------------------
class HumanExoExtractor(BaseFeaturesExtractor):
    """
    obs: Dict(human, exo)
    两条 MLP 编码后 concat。
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        assert isinstance(observation_space, spaces.Dict)
        human_dim = observation_space.spaces["human"].shape[0]
        exo_dim = observation_space.spaces["exo"].shape[0]

        h = 128
        e = 128
        self.human_net = th.nn.Sequential(
            th.nn.Linear(human_dim, h),
            th.nn.ReLU(),
            th.nn.Linear(h, h),
            th.nn.ReLU(),
        )
        self.exo_net = th.nn.Sequential(
            th.nn.Linear(exo_dim, e),
            th.nn.ReLU(),
            th.nn.Linear(e, e),
            th.nn.ReLU(),
        )
        self._features_dim = h + e

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        hh = self.human_net(observations["human"])
        ee = self.exo_net(observations["exo"])
        return th.cat([hh, ee], dim=-1)


# ---------------------------------------------------------------------------
# 环境工厂
# ---------------------------------------------------------------------------
def make_env(rank: int, base_seed: int, env_kwargs: Dict[str, Any]):
    def _init():
        env = WalkEnvV4Multi(seed=base_seed + rank, **env_kwargs)

        # 关键修复：避免“联合训练时 dict obs 被强制改成 Box(exo-only)”
        freeze_human = bool(env_kwargs.get("freeze_human", False))
        freeze_exo = bool(env_kwargs.get("freeze_exo", False))
        if freeze_human and freeze_exo:
            raise ValueError("[train_exo] freeze_human 与 freeze_exo 不能同时为 True")

        # 单侧训练才压 Box：
        # - freeze_human=True => key='exo'
        # - freeze_exo=True   => key='human'
        if (freeze_human or freeze_exo) and isinstance(env.observation_space, spaces.Dict):
            key = "exo" if freeze_human else "human"
            try:
                clip_obs = float(env_kwargs.get("clip_obs", 10.0))
                env = DictToBoxObs(env, key=key, clip_obs=clip_obs, force_clip_bounds=False)
            except Exception as _e_obs:
                print(f"[train_exo] WARN: DictToBoxObs failed (key={key}): {repr(_e_obs)}")

        return env
    return _init


# ---------------------------------------------------------------------------
# Env sanity check (fail fast)
# ---------------------------------------------------------------------------
def sanity_check_env(vec_env, tag: str = "env"):
    """
    注意：SubprocVecEnv 下，vec_env.get_attr() 返回值必须可 pickle。
    因此这里**禁止**拉取 PPO 模型对象（human_policy/exo_policy），否则会触发
    "Can't pickle local object ... <lambda>"。
    我们只检查：freeze 标志、肌肉执行器数量、以及 policy_loaded 布尔标志。
    """

    def _safe_get0(attr: str, default=None):
        try:
            v = vec_env.get_attr(attr)[0]
            return v
        except Exception:
            return default

    n_m = _safe_get0("n_muscle_act", None)
    freeze_human = _safe_get0("freeze_human", None)
    freeze_exo = _safe_get0("freeze_exo", None)
    human_loaded = _safe_get0("human_policy_loaded", None)
    exo_loaded = _safe_get0("exo_policy_loaded", None)
    exo_ids = _safe_get0("exo_hip_ids", None)

    print(
        f"[{tag}] freeze_human={freeze_human} | freeze_exo={freeze_exo} | "
        f"human_policy_loaded={human_loaded} | exo_policy_loaded={exo_loaded} | "
        f"n_muscle_act={n_m} | exo_hip_ids={exo_ids}"
    )

    # 肌肉执行器数量检查（与之前一致）
    if n_m is not None and int(n_m) <= 0:
        try:
            base = vec_env.envs[0]
            while hasattr(base, "env"):
                base = base.env
            sim = getattr(base, "sim", None)
            if sim is not None:
                nu = int(sim.model.nu)
                names = []
                for i in range(min(nu, 30)):
                    try:
                        nm = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    except Exception:
                        nm = None
                    names.append(nm)
                print(f"[{tag}] mujoco.nu={nu} | first_actuator_names={names}")
        except Exception:
            pass

        raise RuntimeError(
            f"[{tag}] n_muscle_act=0：当前加载的 MJCF 里没有 muscle actuator。"
            f" 请确认你加载的是“包含 myoLegs 肌肉 + exo_hip_l/r 执行器”的组合 XML。"
        )

    # 冻结侧必须成功加载其预训练策略：只在明确为 False 时 fail-fast；
    # Subproc 下若拿不到（None），打印 WARN 但不误报中断。
    if freeze_human is True:
        if human_loaded is False:
            raise RuntimeError(f"[{tag}] freeze_human=True 但 human_policy_loaded=False：预训练人体策略未加载成功。")
        if human_loaded is None:
            print(f"[{tag}] WARN: freeze_human=True 但无法读取 human_policy_loaded（SubprocVecEnv 下禁止传 PPO 对象）。")

    if freeze_exo is True:
        if exo_loaded is False:
            raise RuntimeError(f"[{tag}] freeze_exo=True 但 exo_policy_loaded=False：预训练外骨骼策略未加载成功。")
        if exo_loaded is None:
            print(f"[{tag}] WARN: freeze_exo=True 但无法读取 exo_policy_loaded（SubprocVecEnv 下禁止传 PPO 对象）。")


# ---------------------------------------------------------------------------
# 视频 get_frame()
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# VecNormalize helpers (obs normalization + clip to ±clip_obs)
# ---------------------------------------------------------------------------

def find_vecnormalize_stats_path(resume_path: Optional[str], tcfg: Dict[str, Any], mode_tag: str) -> Optional[str]:
    """在续训时定位 VecNormalize 统计文件（如果存在）。
    约定优先级：
      1) YAML/CLI 显式指定的 vecnorm_path / vecnormalize_path
      2) 与 resume_path 同目录下的常见文件名
    """
    # 1) explicit path from config
    for k in ("vecnorm_path", "vecnormalize_path", "vecnormalize_stats_path"):
        p = tcfg.get(k, None)
        if p:
            pp = Path(str(p)).expanduser().resolve()
            if pp.is_file():
                return str(pp)

    # 2) co-located with resume zip
    if resume_path:
        rp = Path(str(resume_path)).expanduser().resolve()
        cand = [
            rp.parent / "vecnormalize.pkl",
            rp.parent / "vecnorm.pkl",
            rp.parent / f"vecnormalize_{mode_tag}.pkl",
            rp.parent / "vecnormalize_stats.pkl",
        ]
        for c in cand:
            if c.is_file():
                return str(c)

    return None


def maybe_wrap_vecnormalize(vec_env, clip_obs: float, norm_reward: bool, stats_path: Optional[str], training: bool):
    """对 VecEnv 进行 VecNormalize 包装；如果提供 stats_path 则从文件加载。"""
    if stats_path and Path(stats_path).is_file():
        venv = VecNormalize.load(stats_path, vec_env)
    else:
        venv = VecNormalize(vec_env, norm_obs=True, norm_reward=norm_reward, clip_obs=clip_obs)

    # 显式设置状态，避免续训/评估混淆
    venv.training = bool(training)
    venv.norm_reward = bool(norm_reward)

    # 某些版本 load 后 clip_obs 可能不可写；做一次 best-effort 同步
    try:
        venv.clip_obs = float(clip_obs)
    except Exception:
        pass
    return venv


class SaveVecNormalizeCallback(BaseCallback):
    """周期性保存 VecNormalize 统计（只在 env 是 VecNormalize 时生效）。"""

    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.save_freq = int(save_freq)
        self.save_path = str(save_path)

    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True
        if (self.n_calls % self.save_freq) != 0:
            return True

        env = self.training_env
        # training_env 是 VecEnvWrapper；VecNormalize 本身也继承 VecEnvWrapper
        if isinstance(env, VecNormalize):
            try:
                env.save(self.save_path)
                if self.verbose > 0:
                    print(f"[train_exo] saved VecNormalize stats => {self.save_path}")
            except Exception as e:
                print(f"[train_exo] WARN: VecNormalize.save failed: {repr(e)}")
        return True

def get_frame(env, video_cfg: Dict[str, Any], camera_id=None):
    base = env
    for _ in range(10):
        if hasattr(base, "env"):
            new_base = base.env
            if new_base is base:
                break
            base = new_base
        else:
            break

    sim = getattr(base, "sim", None)
    if sim is None:
        raise RuntimeError("无法在环境中找到 sim 对象用于渲染。")

    if "render_size" in video_cfg:
        width, height = video_cfg.get("render_size", [480, 480])
    else:
        width = int(video_cfg.get("width", 480))
        height = int(video_cfg.get("height", 480))

    if camera_id is None:
        if "render_camera" in video_cfg:
            camera_id = int(video_cfg.get("render_camera", 0))
        else:
            camera_id = int(video_cfg.get("camera", 0))

    renderer = getattr(sim, "renderer", None)
    if renderer is None or (not hasattr(renderer, "render_offscreen")):
        raise RuntimeError("sim.renderer.render_offscreen 不存在：请确认 myosuite 渲染器已初始化。")

    frame = renderer.render_offscreen(int(width), int(height), camera_id=int(camera_id))
    return np.asarray(frame, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 回调：视频录制（录制环境单独创建，不影响训练）
# ---------------------------------------------------------------------------
class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        env_kwargs: Dict[str, Any],
        video_cfg: Dict[str, Any],
        save_freq: int,
        out_dir: Path,
        rollout_length: int = 1000,
        fps: int = 30,
        fmt: str = "mp4",
        box_key: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_kwargs = dict(env_kwargs)
        self.video_cfg = dict(video_cfg or {})
        self.save_freq = int(save_freq)
        self.out_dir = Path(out_dir)
        self.rollout_length = int(rollout_length)
        self.fps = int(fps)
        self.fmt = fmt
        self.box_key = box_key

        self.last_record_step = 0
        self.record_idx = 0
        self.eval_env = None

    def _init_eval_env(self):
        if self.eval_env is not None:
            return
        print("[VideoRecorder] 创建视频录制环境: WalkEnvV4Multi")
        self.eval_env = WalkEnvV4Multi(seed=12345, **self.env_kwargs)

    def _record_episode(self):
        self._init_eval_env()
        env = self.eval_env

        obs, _ = env.reset()
        model_expects_dict = isinstance(getattr(self.model, "observation_space", None), spaces.Dict)
        if (not model_expects_dict) and isinstance(obs, dict):
            key = self.box_key or "exo"
            obs = obs[key]

        frames = []
        steps = 0

        while steps < self.rollout_length:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _info = env.step(action)
            if (not model_expects_dict) and isinstance(obs, dict):
                key = self.box_key or "exo"
                obs = obs[key]
            frame = get_frame(env, self.video_cfg)
            frames.append(frame)
            steps += 1
            if terminated or truncated:
                break

        if len(frames) == 0:
            return

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.record_idx += 1
        video_path = self.out_dir / f"walk_step{self.num_timesteps}_idx{self.record_idx}.{self.fmt}"
        print(f"[VideoRecorder] 保存视频: {video_path}")

        if self.fmt.lower() in ["mp4", "gif"]:
            try:
                skvideo.io.vwrite(
                    str(video_path),
                    np.asarray(frames, dtype=np.uint8),
                    outputdict={"-r": str(self.fps)},
                )
            except Exception as e:
                print(f"[VideoRecorder] vwrite failed: {repr(e)}")
        else:
            print(f"[VideoRecorder] 不支持的视频格式: {self.fmt}")

    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True
        if (self.num_timesteps - self.last_record_step) >= self.save_freq:
            self.last_record_step = self.num_timesteps
            print(f"[VideoRecorderCallback] 触发视频录制，num_timesteps={self.num_timesteps}")
            try:
                self._record_episode()
            except Exception as e:
                print(f"[VideoRecorderCallback] 视频录制失败: {repr(e)}")
                import traceback
                traceback.print_exc()
        return True


# ---------------------------------------------------------------------------
# 回调：奖励分解（写到 SB3 logger：CSV/TensorBoard）
# ---------------------------------------------------------------------------
class RewardDecomposeCallback(BaseCallback):
    def __init__(self, window_size: int = 100, keys_whitelist: Optional[List[str]] = None, verbose: int = 0):
        super().__init__(verbose)
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
        if isinstance(v, dict) and "reward" in v:
            return self._to_float(v["reward"])
        return None

    def _maybe_record(self, k, v):
        fv = self._to_float(v)
        if fv is not None:
            self.buf[k].append(fv)

    def _on_rollout_start(self):
        self.buf.clear()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        for info in infos:
            if info is None:
                continue
            for k, v in info.items():
                lname = str(k).lower()
                if (("reward" in lname) or ("cost" in lname) or ("penalty" in lname) or (k in self.keys_whitelist)):
                    self._maybe_record(k, v)

            rwd_dict = info.get("rwd_dict", None)
            if isinstance(rwd_dict, dict):
                for k, v in rwd_dict.items():
                    self._maybe_record(k, v)

        return True

    def _on_rollout_end(self):
        if not self.buf:
            return
        for k, dq in self.buf.items():
            if len(dq) == 0:
                continue
            self.logger.record(f"reward_decompose/{k}", float(np.mean(dq)))
        self.buf.clear()


class InfoDebugCallback(BaseCallback):
    DEFAULT_KEYS = [
        "debug/human_policy_used",
        "debug/muscle_mean",
        "debug/muscle_std",
        "debug/tau_L",
        "debug/tau_R",
        "debug/q_des_L", "debug/q_des_R",
        "debug/q_L",     "debug/q_R",
        "debug/dq_des_L","debug/dq_des_R",
        "debug/dq_L",    "debug/dq_R",
    ]

    def __init__(self, log_freq: int = 1, window_size: int = 256, keys: Optional[List[str]] = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_freq = int(max(1, log_freq))
        self.window_size = int(max(1, window_size))
        self.keys = list(keys) if keys is not None else list(self.DEFAULT_KEYS)
        self.buf = defaultdict(lambda: deque(maxlen=self.window_size))
        self._printed_once = False

    @staticmethod
    def _to_float(v):
        try:
            if v is None:
                return None
            if isinstance(v, (float, int, np.floating, np.integer)):
                return float(v)
            if isinstance(v, (np.ndarray, list, tuple)):
                arr = np.asarray(v, dtype=float).ravel()
                if arr.size == 0:
                    return None
                return float(np.mean(arr))
            return float(v)
        except Exception:
            return None

    def _on_rollout_start(self) -> None:
        self.buf.clear()

    def _pull_debug_dict(self) -> Optional[Dict[str, Any]]:
        d: Optional[Dict[str, Any]] = None
        infos = self.locals.get("infos", None)
        if infos and isinstance(infos, (list, tuple)) and len(infos) > 0:
            cand = infos[0] or {}
            if isinstance(cand, dict):
                d = cand

        if d is None or not any(k.startswith("debug/") for k in d.keys()):
            try:
                caches = self.training_env.get_attr("_debug_cache")
                if caches and isinstance(caches, list) and len(caches) > 0:
                    dc = caches[0]
                    if isinstance(dc, dict):
                        d = dict(d or {})
                        d.update(dc)
            except Exception:
                pass

        if d is None or not isinstance(d, dict):
            return None

        if not self._printed_once:
            try:
                ks = sorted([k for k in d.keys() if k.startswith("debug/")])
                print("[InfoDebugCallback] first seen debug keys:", ks)
            except Exception:
                print("[InfoDebugCallback] first seen debug keys: <unavailable>")
            self._printed_once = True
        return d

    def _collect_once(self) -> None:
        d = self._pull_debug_dict()
        if d is None:
            for k in self.keys:
                self.buf[k].append(float("nan"))
            return
        for k in self.keys:
            fv = self._to_float(d.get(k, None))
            self.buf[k].append(float("nan") if fv is None else float(fv))

    def _record_buf_means(self) -> None:
        for k in self.keys:
            dq = self.buf.get(k, None)
            if dq is None or len(dq) == 0:
                self.logger.record(k, float("nan"))
            else:
                self.logger.record(k, float(np.mean(dq)))

    def _on_step(self) -> bool:
        self._collect_once()
        if (self.n_calls % self.log_freq) == 0:
            self._record_buf_means()
        return True

    def _on_rollout_end(self) -> None:
        self._record_buf_means()
        self.buf.clear()


# ---------------------------------------------------------------------------
# Resume / 继续训练工具：更鲁棒的 PPO.load（兼容不可反序列化的 schedule）
# ---------------------------------------------------------------------------
def load_resume_ppo(resume_path: str, env, device: str, pcfg: Dict[str, Any]):
    """从已保存的 zip 继续训练。
    说明：
    - 某些情况下（例如使用 schedule 或自定义对象）PPO.load 可能反序列化失败；
      这里提供一个兜底：将 learning_rate / clip_range 视作常数 schedule。
    """
    rp = str(Path(resume_path).expanduser().resolve())

    try:
        return PPO.load(rp, env=env, device=device)
    except Exception as e:
        print(f"[train_exo] WARN: PPO.load failed: {repr(e)}")
        lr = float(pcfg.get("learning_rate", 3e-4))
        clip = float(pcfg.get("clip_range", 0.2))
        custom_objects = {
            "lr_schedule": (lambda _progress: lr),
            "clip_range":  (lambda _progress: clip),
        }
        print("[train_exo] WARN: retry PPO.load with custom_objects (constant schedules).")
        return PPO.load(rp, env=env, device=device, custom_objects=custom_objects)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume_path", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = pick_config_path(args.config)
    cfg = load_yaml(cfg_path)

    ecfg = cfg.get("env", {}) or {}
    tcfg = cfg.get("train", {}) or {}
    pcfg = cfg.get("ppo", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}

    seed = int(args.seed if args.seed is not None else ecfg.get("seed", 4))
    num_envs = int(args.num_envs if args.num_envs is not None else ecfg.get("num_envs", 1))
    total_timesteps = int(args.total_timesteps if args.total_timesteps is not None else tcfg.get("total_timesteps", 2_000_000))
    device = str(args.device if args.device is not None else "auto")
    set_global_seeds(seed)

    model_xml = ecfg.get("model_xml") or ecfg.get("model_path") or ecfg.get("xml_path")
    obsd_xml = ecfg.get("obsd_xml") or ecfg.get("obsd_model_path") or model_xml
    if model_xml is None:
        raise RuntimeError("未在 YAML env 中找到 model_xml/model_path/xml_path。")
    model_xml = str(Path(model_xml).expanduser().resolve())
    obsd_xml = str(Path(obsd_xml).expanduser().resolve())

    reset_type = ecfg.get("reset_type", "init")
    target_y_vel = float(ecfg.get("target_y_vel", 1.2))
    hip_period = int(ecfg.get("hip_period", 100))

    human_pretrained_path = ecfg.get("human_pretrained_path", None)
    freeze_human = bool(ecfg.get("freeze_human", False))
    human_policy_device = ecfg.get("human_policy_device", "cpu")
    human_deterministic = bool(ecfg.get("human_deterministic", True))

    exo_pretrained_path = ecfg.get("exo_pretrained_path", None)
    freeze_exo = bool(ecfg.get("freeze_exo", False))
    exo_policy_device = ecfg.get("exo_policy_device", "cpu")
    exo_deterministic = bool(ecfg.get("exo_deterministic", True))

    if freeze_human and freeze_exo:
        raise RuntimeError("freeze_human 与 freeze_exo 不能同时为 True")
    if freeze_human and not human_pretrained_path:
        raise RuntimeError("freeze_human=True 但未提供 human_pretrained_path")
    if freeze_exo and not exo_pretrained_path:
        raise RuntimeError("freeze_exo=True 但未提供 exo_pretrained_path")

    # 训练模式标签
    if freeze_human:
        mode_tag = "exo"
    elif freeze_exo:
        mode_tag = "human"
    else:
        # 两侧都不冻结：联合训练（可选）
        mode_tag = "joint"

    # =========================
    # 奖励权重：按训练模式选择，并确保是可序列化的 float dict
    # 兼容 YAML：reward_weights / reward_weights_exo / reward_weights_human / reward_weights_joint
    # =========================
    rw_all = cfg.get("reward_weights", {}) or {}
    rw_exo = cfg.get("reward_weights_exo", None)
    rw_human = cfg.get("reward_weights_human", None)
    rw_joint = cfg.get("reward_weights_joint", None)

    if mode_tag == "exo":
        rw_src = rw_exo if isinstance(rw_exo, dict) and len(rw_exo) > 0 else rw_all
    elif mode_tag == "human":
        rw_src = rw_human if isinstance(rw_human, dict) and len(rw_human) > 0 else rw_all
    else:
        rw_src = rw_joint if isinstance(rw_joint, dict) and len(rw_joint) > 0 else rw_all

    reward_weights_used: Dict[str, float] = {}
    if isinstance(rw_src, dict):
        for k, v in rw_src.items():
            try:
                reward_weights_used[str(k)] = float(v)
            except Exception:
                # 兜底：不影响训练，但不把非数值写入 wandb/config
                pass

    if len(reward_weights_used) == 0:
        print("[train_exo] WARN: reward_weights 为空或不可解析，将使用环境内部 DEFAULT_RWD_KEYS_AND_WEIGHTS。")

    # =========================
    # 固定根目录 + 时间戳子目录
    # =========================
    ts_dir = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # 你指定的两条根目录（强制使用，不再依赖 YAML 的 train.log_dir）
    if mode_tag == "exo":
        base_root = Path("/home/lvchen/body_SB3/logs/joint_exo")
    elif mode_tag == "human":
        base_root = Path("/home/lvchen/body_SB3/logs/joint_human")


    run_dir = (base_root / ts_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # run_name 仍可用于打印/区分，但不再参与目录结构（避免重复嵌套）
    run_name = str(tcfg.get("run_name", "exo_joint_run"))
    if not run_name.endswith(f"_{mode_tag}"):
        run_name = f"{run_name}_{mode_tag}"
    # -------------------------
    # W&B init（可选）
    # -------------------------
    wandb_run, wandb_cb, wandb_enabled = init_wandb(
        cfg=cfg,
        run_dir=run_dir,
        run_name=run_name,
        mode_tag=mode_tag,
        ts_dir=ts_dir,
    )

    # -------------------------
    # 将“实际使用的奖励权重”写入 W&B config，并把本次运行的配置 YAML 作为 artifact 上传
    # -------------------------
    cfg_used_path = (run_dir / "config_used.yaml").resolve()
    try:
        _save_yaml(cfg, cfg_used_path)
    except Exception as e:
        print(f"[train_exo] WARN: save config_used.yaml failed: {repr(e)}")

    if wandb_enabled and (wandb_run is not None):
        try:
            import wandb as _wandb  # type: ignore
            # 1) 在 Run 的 config 面板里可直接看到
            _wandb.config.update(
                {
                    "mode_tag": mode_tag,
                    "timestamp_dir": ts_dir,
                    "reward_weights": _to_jsonable(reward_weights_used),
                },
                allow_val_change=True,
            )
        except Exception as e:
            print(f"[train_exo] WARN: wandb.config.update failed: {repr(e)}")

        # 2) 额外把 YAML 作为 artifact（便于复现实验）
        try:
            import wandb as _wandb  # type: ignore
            # artifact 名保持稳定，W&B 自动递增版本；用 aliases 记录本次时间戳
            art_cfg = _wandb.Artifact(
                name=f"run_config_{mode_tag}",
                type="config",
                metadata={"mode": mode_tag, "timestamp_dir": ts_dir},
            )
            if cfg_used_path.is_file():
                art_cfg.add_file(str(cfg_used_path), name="config_used.yaml")
            try:
                wandb_run.log_artifact(art_cfg, aliases=["latest", ts_dir])
            except Exception:
                wandb_run.log_artifact(art_cfg)
        except Exception as e:
            print(f"[train_exo] WARN: log config artifact failed: {repr(e)}")

    # 观测归一化/裁切（续训空间一致性关键）
    # - clip_obs: VecNormalize 输出裁切范围，且会体现在 env.observation_space 的 low/high 上
    # - use_vecnormalize: 建议在单侧训练（Box obs）时开启；联合训练（Dict obs）默认不启用
    clip_obs = float(tcfg.get("clip_obs", 10.0))
    use_vecnormalize = bool(tcfg.get("use_vecnormalize", True))
    norm_reward = bool(tcfg.get("norm_reward", False))

    env_kwargs = dict(
        clip_obs=clip_obs,
        model_path=model_xml,
        obsd_model_path=obsd_xml,
        reset_type=reset_type,
        target_y_vel=target_y_vel,
        hip_period=hip_period,
        human_pretrained_path=human_pretrained_path,
        freeze_human=freeze_human,
        human_policy_device=human_policy_device,
        human_deterministic=human_deterministic,
        exo_pretrained_path=exo_pretrained_path,
        freeze_exo=freeze_exo,
        exo_policy_device=exo_policy_device,
        exo_deterministic=exo_deterministic,
    )

    # 将奖励权重注入环境（WalkEnvV4Multi._setup(weighted_reward_keys=...)）
    if len(reward_weights_used) > 0:
        env_kwargs["weighted_reward_keys"] = reward_weights_used

    resume_path = args.resume_path or tcfg.get("resume_path", None)
    is_resume = (resume_path is not None and Path(resume_path).is_file())

    base_seed = seed
    if num_envs > 1:
        env_fns = [make_env(i, base_seed, env_kwargs) for i in range(num_envs)]
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv([make_env(0, base_seed, env_kwargs)])

    env = VecMonitor(env, filename=str(run_dir / "monitor_train.csv"))

    eval_env = DummyVecEnv([make_env(1000, base_seed, env_kwargs)])
    eval_env = VecMonitor(eval_env, filename=None)

    # Box(obs) 单侧训练：使用 VecNormalize 做归一化，并在归一化后裁切到 ±clip_obs
    # 说明：VecNormalize 会把 observation_space 的 low/high 也改成 [-clip_obs, clip_obs]，
    #      这正是 SB3 续训时对齐 observation_space 的关键。
    is_dict_obs = isinstance(env.observation_space, spaces.Dict)
    vecnorm_save_path = str(run_dir / "vecnormalize.pkl")
    vecnorm_stats_path = find_vecnormalize_stats_path(resume_path if is_resume else None, tcfg, mode_tag)

    if use_vecnormalize and (not is_dict_obs):
        env = maybe_wrap_vecnormalize(
            env, clip_obs=clip_obs, norm_reward=norm_reward,
            stats_path=vecnorm_stats_path, training=True
        )
        eval_env = maybe_wrap_vecnormalize(
            eval_env, clip_obs=clip_obs, norm_reward=norm_reward,
            stats_path=vecnorm_stats_path, training=False
        )

    sanity_check_env(env, tag="train_env")
    sanity_check_env(eval_env, tag="eval_env")

    is_dict_obs = isinstance(env.observation_space, spaces.Dict)
    if is_dict_obs:
        policy_name = "MultiInputPolicy"
        policy_kwargs = dict(
            features_extractor_class=HumanExoExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        )
    else:
        if freeze_exo:
            from sb3_lattice_policy import LatticeActorCriticPolicy
            policy_name = LatticeActorCriticPolicy
            policy_kwargs = dict()
            print(f"[train_exo] INFO: mode={mode_tag} | obs_space=Box{env.observation_space.shape} | use LatticeActorCriticPolicy")
        else:
            policy_name = "MlpPolicy"
            policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))
            print(f"[train_exo] INFO: mode={mode_tag} | obs_space=Box{env.observation_space.shape} | use {policy_name}.")

    eval_freq = int(eval_cfg.get("eval_freq", 50_000))
    n_eval_episodes = int(eval_cfg.get("n_episodes", 5))
    save_freq = int(tcfg.get("save_frequency", 200_000))
    eval_freq = max(eval_freq // max(num_envs, 1), 1)
    save_freq = max(save_freq // max(num_envs, 1), 1)

    if is_resume:
        # 继续训练：从已保存 zip 恢复
        # 注意：恢复的模型必须与当前训练模式一致（动作维度/观测形状一致）
        if freeze_exo:
            # freeze_exo=True 时可能使用自定义 policy（例如 LatticeActorCriticPolicy）
            try:
                from sb3_lattice_policy import LatticeActorCriticPolicy  # noqa: F401
            except Exception as e:
                print(f"[train_exo] WARN: import LatticeActorCriticPolicy failed: {repr(e)}")

        model = load_resume_ppo(resume_path, env=env, device=device, pcfg=pcfg)

        # 确保本次运行写入当前 run_dir（TB/CSV）
        model.tensorboard_log = str(run_dir / "tb")
    else:
        model = PPO(
            policy=policy_name,
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
            clip_range=float(pcfg.get("clip_range", 0.2)),
            use_sde=bool(pcfg.get("use_sde", False)),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(run_dir / "tb"),
        )

    callbacks: List[BaseCallback] = []
    # W&B callback：若启用，则将 SB3 日志（含 TensorBoard）同步到 W&B
    if wandb_enabled and (wandb_cb is not None):
        callbacks.append(wandb_cb)


    callbacks.append(
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / f"best_model_{mode_tag}"),
            log_path=str(run_dir / f"eval_{mode_tag}"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=bool(eval_cfg.get("deterministic", True)),
        )
    )

    callbacks.append(
        CheckpointCallback(
            save_freq=save_freq,
            save_path=str(run_dir / "checkpoints"),
            name_prefix=f"ppo_{mode_tag}",
        )
    )

    # VecNormalize 统计保存（仅 Box(obs) 单侧训练时启用）
    if use_vecnormalize and (not is_dict_obs):
        callbacks.append(
            SaveVecNormalizeCallback(
                save_freq=save_freq,
                save_path=vecnorm_save_path,
                verbose=0,
            )
        )

    rdcfg = (tcfg.get("reward_decompose", {}) or {})
    if bool(rdcfg.get("enable", True)):
        callbacks.append(
            RewardDecomposeCallback(
                window_size=int(rdcfg.get("window_size", 100)),
                keys_whitelist=rdcfg.get("keys_whitelist", None),
                verbose=0,
            )
        )

    idcfg = (tcfg.get("info_debug", {}) or {})
    if bool(idcfg.get("enable", True)):
        callbacks.append(
            InfoDebugCallback(
                log_freq=int(idcfg.get("log_freq", 1)),
                window_size=int(idcfg.get("window_size", 256)),
                keys=idcfg.get("keys", None),
                verbose=0,
            )
        )

    # video：默认放在 <run_dir>/videos/
    vtop = cfg.get("video", {}) or {}
    if bool(vtop.get("video_enabled", True)):
        video_out_dir = (run_dir / "videos").resolve()
        video_out_dir.mkdir(parents=True, exist_ok=True)

        v_save_freq = int(vtop.get("save_freq", int(tcfg.get("save_frequency", 200_000))))
        v_save_freq = max(v_save_freq // max(num_envs, 1), 1)

        box_key = ("exo" if freeze_human else ("human" if freeze_exo else None))

        callbacks.append(
            VideoRecorderCallback(
                env_kwargs=env_kwargs,
                video_cfg=vtop,
                save_freq=v_save_freq,
                out_dir=video_out_dir,
                rollout_length=int(vtop.get("rollout_length", 1000)),
                fps=int(vtop.get("fps", 30)),
                fmt=str(vtop.get("format", "mp4")),
                box_key=box_key,
                verbose=1,
            )
        )

    print(
        f"[train_exo] mode={mode_tag} | run_dir={run_dir} | run_name={run_name} | ts={ts_dir}\n"
        f"           total_timesteps={total_timesteps} | num_envs={num_envs} | freeze_human={freeze_human} | freeze_exo={freeze_exo} | is_resume={is_resume}"
    )
    print("[train_exo] callbacks:", [type(c).__name__ for c in callbacks])

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        reset_num_timesteps=(not is_resume),
        tb_log_name=run_name,
    )

    final_path = run_dir / f"final_{mode_tag}_model.zip"
    model.save(str(final_path))
    # 最终保存 VecNormalize 统计（若启用）
    if isinstance(env, VecNormalize):
        try:
            env.save(vecnorm_save_path)
            print(f"[OK] VecNormalize stats saved to: {vecnorm_save_path}")
        except Exception as e:
            print(f"[train_exo] WARN: final VecNormalize.save failed: {repr(e)}")

    print(f"[OK] Training done. Final model saved to: {final_path}")

    best_zip = (run_dir / f"best_model_{mode_tag}" / "best_model.zip")
    if best_zip.is_file():
        best_alias = run_dir / f"best_{mode_tag}_model.zip"
        try:
            shutil.copy2(str(best_zip), str(best_alias))
            print(f"[OK] Best model copied to: {best_alias}")
        except Exception as e:
            print(f"[train_exo] WARN: copy best model failed: {repr(e)}")

    # -------------------------
    # W&B Artifacts：上传 best model（优先 best_alias，其次 best_zip，最后退化为 final）
    # -------------------------
    if wandb_enabled and (wandb_run is not None):
        try:
            cand = None
            best_alias = (run_dir / f"best_{mode_tag}_model.zip")
            if best_alias.is_file():
                cand = best_alias
            elif best_zip.is_file():
                cand = best_zip
            elif final_path.is_file():
                cand = final_path

            if cand is not None:
                log_wandb_artifact_best_model(
                    wandb_run=wandb_run,
                    best_model_path=cand,
                    mode_tag=mode_tag,
                    ts_dir=ts_dir,
                    reward_weights_used=reward_weights_used,
                    extra_meta={"run_dir": str(run_dir)},
                )
                print(f"[train_exo] wandb artifact uploaded: {cand}")
            else:
                print("[train_exo] WARN: no model file found for wandb artifact.")
        except Exception as e:
            print(f"[train_exo] WARN: upload best model artifact failed: {repr(e)}")

    # -------------------------
    # W&B finish（可选）
    # -------------------------
    try:
        if wandb_enabled and (wandb_run is not None):
            import wandb as _wandb  # type: ignore
            _wandb.finish()
            print("[train_exo] wandb finished.")
    except Exception as e:
        print(f"[train_exo] WARN: wandb.finish failed: {repr(e)}")




if __name__ == "__main__":
    main()