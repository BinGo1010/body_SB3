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

# ---- headless rendering defaults (allow user override) ----
# 必须在 import mujoco 之前设置，否则离屏渲染可能无法初始化
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl") or "egl")

import mujoco

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

from envs.walk_gait_exo_joint_tarque_setpoint import WalkEnvV4Multi


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
    p1 = Path("/home/lvchen/body_SB3/configs/body_exo_walk_config_joint.yaml")
    if p1.is_file():
        return p1.resolve()
    return (PROJECT_ROOT / "configs" / "body_exo_walk_config_joint.yaml").resolve()


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

def _sanitize_frame(frame: np.ndarray) -> np.ndarray:
    """将渲染输出统一成 uint8 RGB(H,W,3)。"""
    arr = np.asarray(frame)
    # RGBA -> RGB
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    # float -> uint8
    if arr.dtype != np.uint8:
        mx = float(np.max(arr)) if arr.size else 0.0
        if mx <= 1.0:
            arr = (arr * 255.0).clip(0.0, 255.0).astype(np.uint8)
        else:
            arr = arr.clip(0.0, 255.0).astype(np.uint8)
    return arr


def get_frame(env, video_cfg: Dict[str, Any], camera_id=None) -> np.ndarray:
    """
    从 env 取一帧 RGB 图像（用于视频录制）。
    优先走 myosuite 的 sim.renderer.render_offscreen；若不可用，则回退到 mujoco.Renderer。
    """
    # unwrap（兼容多层 gym wrapper）
    base = env
    for _ in range(10):
        if hasattr(base, "env"):
            base = base.env
        else:
            break

    sim = getattr(base, "sim", None)
    if sim is None:
        raise RuntimeError("无法从 env 获取 sim（用于渲染）。")

    render_size = video_cfg.get("render_size", [640, 480])
    if isinstance(render_size, (list, tuple)) and len(render_size) >= 2:
        width, height = int(render_size[0]), int(render_size[1])
    else:
        width = height = int(render_size) if render_size else 480

    if camera_id is None:
        camera_id = video_cfg.get("render_camera", 0)
    camera_id = int(camera_id)

    # 1) myosuite renderer（如果存在）
    renderer = getattr(sim, "renderer", None)
    if renderer is not None and hasattr(renderer, "render_offscreen"):
        frame = renderer.render_offscreen(int(width), int(height), camera_id=int(camera_id))
        return _sanitize_frame(frame)

    # 2) fallback：mujoco.Renderer（只依赖 sim.model/sim.data）
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        raise RuntimeError("sim.model/sim.data 不存在，无法使用 mujoco.Renderer 离屏渲染。")

    try:
        mjr = mujoco.Renderer(model, height=int(height), width=int(width))
        mjr.update_scene(data, camera=camera_id)
        frame = mjr.render()
        mjr.close()
        return _sanitize_frame(frame)
    except Exception as e:
        raise RuntimeError(
            f"离屏渲染失败: {repr(e)}。建议在命令行设置 MUJOCO_GL=egl（或 osmesa），并确保系统支持离屏渲染。"
        )


def write_video(frames: List[np.ndarray], out_path: Path, fps: int = 30) -> Path:
    """
    写视频文件：
      - mp4 优先；若 mp4 写失败自动回退保存 gif（同名 .gif）
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = [_sanitize_frame(f) for f in (frames or [])]
    if len(frames) == 0:
        raise RuntimeError("没有帧可写入视频。")

    try:
        import imageio.v2 as imageio
    except Exception as e:
        raise RuntimeError(f"缺少 imageio 依赖，无法写视频: {repr(e)}")

    ext = out_path.suffix.lower()
    fps = int(fps)

    # gif 直接写
    if ext == ".gif":
        imageio.mimsave(str(out_path), frames, fps=fps)
        return out_path

    # mp4：优先尝试
    try:
        writer = imageio.get_writer(str(out_path), fps=fps)
        try:
            for f in frames:
                writer.append_data(f)
        finally:
            writer.close()
        return out_path
    except Exception as e:
        # 回退 gif（通常不依赖 ffmpeg）
        gif_path = out_path.with_suffix(".gif")
        try:
            imageio.mimsave(str(gif_path), frames, fps=fps)
            print(f"[video] mp4 写入失败，已回退保存 gif: {gif_path} | err={repr(e)}")
            return gif_path
        except Exception as e2:
            raise RuntimeError(
                f"写视频失败（mp4 与 gif 都失败）。mp4 err={repr(e)} ; gif err={repr(e2)}。"
                f"请确认系统 ffmpeg 或安装 imageio-ffmpeg，并检查 MUJOCO_GL 设置。"
            )


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

        try:
            write_video(frames, video_path, fps=self.fps)
        except Exception as e:
            print(f"[VideoRecorder] 写视频失败: {repr(e)}")

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
        "debug/muscle_sat_ratio",  # 肌肉饱和率
        "debug/tau_L",
        "debug/tau_R",
        "debug/q_des_L", "debug/q_des_R",
        "debug/q_L",     "debug/q_R",
        "debug/dq_des_L","debug/dq_des_R",
        "debug/dq_L",    "debug/dq_R",
        # "debug/cpg_omega",
        # "debug/cpg_amp_L", "debug/cpg_amp_R",
        # "debug/cpg_off_L", "debug/cpg_off_R",
        # "debug/cpg_phase_diff_mod", "debug/cpg_phase_err",
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
    vcfg = cfg.get("video", {}) or {}

    # -------------------------
    # 基本参数
    # -------------------------
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

    # “双 True”作为开关：进入间断式联合训练（Interleave）
    freeze_human_cfg = bool(ecfg.get("freeze_human", False))
    freeze_exo_cfg = bool(ecfg.get("freeze_exo", False))
    interleave_enabled = bool(freeze_human_cfg and freeze_exo_cfg)

    human_pretrained_path = ecfg.get("human_pretrained_path", None)
    exo_pretrained_path = ecfg.get("exo_pretrained_path", None)
    human_policy_device = ecfg.get("human_policy_device", "cpu")
    exo_policy_device = ecfg.get("exo_policy_device", "cpu")
    human_deterministic = bool(ecfg.get("human_deterministic", True))
    exo_deterministic = bool(ecfg.get("exo_deterministic", True))

    if interleave_enabled:
        if not human_pretrained_path:
            raise RuntimeError("interleave 模式下必须提供 human_pretrained_path（Human 初始/预训练模型）。")
        if not exo_pretrained_path:
            raise RuntimeError("interleave 模式下必须提供 exo_pretrained_path（EXO 初始/预训练模型）。")
        mode_tag = "interleave"
    else:
        # 训练模式标签
        if freeze_human_cfg:
            mode_tag = "exo"
        elif freeze_exo_cfg:
            mode_tag = "human"
        else:
            mode_tag = "joint"

        if freeze_human_cfg and freeze_exo_cfg:
            raise RuntimeError("freeze_human 与 freeze_exo 不能同时为 True（除非进入 interleave 模式）。")
        if freeze_human_cfg and not human_pretrained_path:
            raise RuntimeError("freeze_human=True 但未提供 human_pretrained_path")
        if freeze_exo_cfg and not exo_pretrained_path:
            raise RuntimeError("freeze_exo=True 但未提供 exo_pretrained_path")

    # =========================
    # 奖励权重解析（支持：reward_weights / reward_weights_exo / reward_weights_human / reward_weights_joint）
    # =========================
    rw_all = cfg.get("reward_weights", {}) or {}
    rw_exo = cfg.get("reward_weights_exo", None)
    rw_human = cfg.get("reward_weights_human", None)
    rw_joint = cfg.get("reward_weights_joint", None)

    def pick_reward_weights(tag: str) -> Dict[str, float]:
        if tag == "exo":
            rw_src = rw_exo if isinstance(rw_exo, dict) and len(rw_exo) > 0 else rw_all
        elif tag == "human":
            rw_src = rw_human if isinstance(rw_human, dict) and len(rw_human) > 0 else rw_all
        else:
            rw_src = rw_joint if isinstance(rw_joint, dict) and len(rw_joint) > 0 else rw_all

        out: Dict[str, float] = {}
        if isinstance(rw_src, dict):
            for k, v in rw_src.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    pass
        return out

    # =========================
    # 固定根目录 + 时间戳子目录
    # =========================
    ts_dir = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if mode_tag == "exo":
        base_root = Path("/home/lvchen/body_SB3/logs/joint_exo")
    elif mode_tag == "human":
        base_root = Path("/home/lvchen/body_SB3/logs/joint_human")
    elif mode_tag == "interleave":
        base_root = Path("/home/lvchen/body_SB3/logs/joint_interleave")
    else:
        base_root = Path("/home/lvchen/body_SB3/logs/joint_both")

    run_dir = (base_root / ts_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    run_name = str(tcfg.get("run_name", "exo_joint_run"))
    if not run_name.endswith(f"_{mode_tag}"):
        run_name = f"{run_name}_{mode_tag}"

    # -------------------------
    # W&B init（只做一次）
    # -------------------------
    wandb_run, _wandb_cb_unused, wandb_enabled = init_wandb(
        cfg=cfg,
        run_dir=run_dir,
        run_name=run_name,
        mode_tag=mode_tag,
        ts_dir=ts_dir,
    )

    def build_wandb_callback(phase_tag: str) -> Optional[BaseCallback]:
        """每个 phase 重新创建一个 WandbCallback，避免跨 model 复用同一 callback 的潜在状态残留。"""
        if not wandb_enabled:
            return None
        wcfg = (cfg.get("wandb", {}) or {})
        try:
            from wandb.integration.sb3 import WandbCallback  # type: ignore
            model_save_path = (run_dir / "wandb_models" / phase_tag).resolve()
            model_save_path.mkdir(parents=True, exist_ok=True)
            return WandbCallback(
                model_save_path=str(model_save_path),
                model_save_freq=int(wcfg.get("model_save_freq", 0)),
                verbose=int(wcfg.get("verbose", 0)),
            )
        except Exception as e:
            print(f"[train_exo] WARN: build WandbCallback failed: {repr(e)}")
            return None

    # -------------------------
    # 保存 config_used.yaml + 上传配置 artifact（可选）
    # -------------------------
    cfg_used_path = (run_dir / "config_used.yaml").resolve()
    try:
        _save_yaml(cfg, cfg_used_path)
    except Exception as e:
        print(f"[train_exo] WARN: save config_used.yaml failed: {repr(e)}")

    if wandb_enabled and (wandb_run is not None):
        try:
            import wandb as _wandb  # type: ignore
            _wandb.config.update(
                {
                    "mode_tag": mode_tag,
                    "timestamp_dir": ts_dir,
                    "reward_weights_all": _to_jsonable(pick_reward_weights("joint")),
                    "reward_weights_human": _to_jsonable(pick_reward_weights("human")),
                    "reward_weights_exo": _to_jsonable(pick_reward_weights("exo")),
                },
                allow_val_change=True,
            )
        except Exception as e:
            print(f"[train_exo] WARN: wandb.config.update failed: {repr(e)}")

        try:
            import wandb as _wandb  # type: ignore
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

    # -------------------------
    # 观测归一化/裁切
    # -------------------------
    clip_obs = float(tcfg.get("clip_obs", 10.0))
    use_vecnormalize = bool(tcfg.get("use_vecnormalize", False))
    norm_reward = bool(tcfg.get("norm_reward", False))

    # ---------------------------------------------------------------------
    # Interleave: 交替训练 Human / Exo（同一个 run，重复 n 轮）
    # ---------------------------------------------------------------------
    if interleave_enabled:
        icfg = (tcfg.get("interleave", {}) or {})
        n_cycles = int(icfg.get("n_cycles", 1))
        T_h = int(icfg.get("human_steps", 200_000))
        T_e = int(icfg.get("exo_steps", 200_000))

        video_every = int(vcfg.get("interleave_every_n_cycles", 0))
        video_do = bool(vcfg.get("video_enabled", True)) and (video_every > 0)

        # 目录：best / checkpoints / vecnormalize
        best_human_dir = (run_dir / "best_human").resolve(); best_human_dir.mkdir(parents=True, exist_ok=True)
        best_exo_dir = (run_dir / "best_exo").resolve(); best_exo_dir.mkdir(parents=True, exist_ok=True)
        ckpt_human_dir = (run_dir / "checkpoints_human").resolve(); ckpt_human_dir.mkdir(parents=True, exist_ok=True)
        ckpt_exo_dir = (run_dir / "checkpoints_exo").resolve(); ckpt_exo_dir.mkdir(parents=True, exist_ok=True)
        vecnorm_human_path = str((run_dir / "vecnormalize_human.pkl").resolve())
        vecnorm_exo_path = str((run_dir / "vecnormalize_exo.pkl").resolve())

        # 初始化 best_model.zip（若不存在，则拷贝预训练模型作为起点）
        def ensure_best(best_dir: Path, init_zip: str):
            best_zip = best_dir / "best_model.zip"
            if best_zip.is_file():
                return
            src = Path(init_zip).expanduser().resolve()
            if not src.is_file():
                raise RuntimeError(f"初始化 best_model.zip 失败：找不到预训练模型: {src}")
            shutil.copy2(str(src), str(best_zip))
            print(f"[interleave] init best_model.zip -> {best_zip}")

        ensure_best(best_human_dir, human_pretrained_path)
        ensure_best(best_exo_dir, exo_pretrained_path)

        # phase 训练函数
        def train_phase(
            phase_tag: str,
            phase_steps: int,
            freeze_human: bool,
            freeze_exo: bool,
            human_frozen_path: str,
            exo_frozen_path: str,
            best_dir: Path,
            ckpt_dir: Path,
            vecnorm_save_path: str,
            reward_tag: str,
        ) -> None:
            reward_weights = pick_reward_weights(reward_tag)
            if len(reward_weights) == 0:
                print(f"[interleave] WARN: reward_weights_{reward_tag} 为空，将使用环境默认权重。")

            env_kwargs = dict(
                clip_obs=clip_obs,
                model_path=model_xml,
                obsd_model_path=obsd_xml,
                reset_type=reset_type,
                target_y_vel=target_y_vel,
                hip_period=hip_period,
                human_pretrained_path=str(Path(human_frozen_path).expanduser().resolve()),
                freeze_human=bool(freeze_human),
                human_policy_device=human_policy_device,
                human_deterministic=human_deterministic,
                exo_pretrained_path=str(Path(exo_frozen_path).expanduser().resolve()),
                freeze_exo=bool(freeze_exo),
                exo_policy_device=exo_policy_device,
                exo_deterministic=exo_deterministic,
                # 环境侧仅需要区分 human / exo（用于 debug/日志），不携带 cycle 编号
                interleave_phase=("human" if bool(freeze_exo) else ("exo" if bool(freeze_human) else phase_tag)),
            )
            if len(reward_weights) > 0:
                env_kwargs["weighted_reward_keys"] = reward_weights

            # 本 phase 可重建 VecEnv（满足你的“只创建 1 个环境允许每个 phase 重建 VecEnv”）
            base_seed = seed
            if num_envs > 1:
                env_fns = [make_env(i, base_seed, env_kwargs) for i in range(num_envs)]
                env = SubprocVecEnv(env_fns)
            else:
                env = DummyVecEnv([make_env(0, base_seed, env_kwargs)])
            env = VecMonitor(env, filename=str(run_dir / f"monitor_train_{phase_tag}.csv"))

            eval_env = DummyVecEnv([make_env(1000, base_seed, env_kwargs)])
            eval_env = VecMonitor(eval_env, filename=None)

            is_dict_obs = isinstance(env.observation_space, spaces.Dict)
            if use_vecnormalize and (not is_dict_obs):
                # interleave：每个 phase 维护各自的 VecNormalize 统计文件
                stats_path = vecnorm_save_path if Path(vecnorm_save_path).is_file() else None
                env = maybe_wrap_vecnormalize(env, clip_obs=clip_obs, norm_reward=norm_reward, stats_path=stats_path, training=True)
                eval_env = maybe_wrap_vecnormalize(eval_env, clip_obs=clip_obs, norm_reward=norm_reward, stats_path=stats_path, training=False)
            elif use_vecnormalize and is_dict_obs:
                print(f"[interleave] WARN: phase={phase_tag} 是 Dict obs，已跳过 VecNormalize（建议单侧训练用 Box obs）。")

            sanity_check_env(env, tag=f"train_env_{phase_tag}")
            sanity_check_env(eval_env, tag=f"eval_env_{phase_tag}")

            # policy 选择
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
                    try:
                        from sb3_lattice_policy import LatticeActorCriticPolicy
                        policy_name = LatticeActorCriticPolicy
                        policy_kwargs = dict()
                        print(f"[interleave] phase={phase_tag} | use LatticeActorCriticPolicy")
                    except Exception as e:
                        print(f"[interleave] WARN: import LatticeActorCriticPolicy failed: {repr(e)}; fallback to MlpPolicy")
                        policy_name = "MlpPolicy"
                        policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))
                else:
                    policy_name = "MlpPolicy"
                    policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))

            eval_freq = int(eval_cfg.get("eval_freq", 50_000))
            n_eval_episodes = int(eval_cfg.get("n_episodes", 5))
            save_freq = int(tcfg.get("save_frequency", 200_000))
            eval_freq = max(eval_freq // max(num_envs, 1), 1)
            save_freq = max(save_freq // max(num_envs, 1), 1)

            # 从上一轮自己的 best_model 继续训练
            resume_zip = str((best_dir / "best_model.zip").resolve())
            model = load_resume_ppo(resume_zip, env=env, device=device, pcfg=pcfg)
            model.tensorboard_log = str(run_dir / "tb")

            callbacks: List[BaseCallback] = []
            wcb = build_wandb_callback(phase_tag)
            if wcb is not None:
                callbacks.append(wcb)

            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(best_dir),
                    log_path=str(run_dir / f"eval_{phase_tag}"),
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=bool(eval_cfg.get("deterministic", True)),
                )
            )
            callbacks.append(
                CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(ckpt_dir),
                    name_prefix=f"ppo_{phase_tag}",
                )
            )

            if use_vecnormalize and (not is_dict_obs):
                callbacks.append(SaveVecNormalizeCallback(save_freq=save_freq, save_path=vecnorm_save_path, verbose=0))

            rdcfg = (tcfg.get("reward_decompose", {}) or {})
            if bool(rdcfg.get("enable", True)):
                callbacks.append(
                    RewardDecomposeCallback(
                        window_size=int(rdcfg.get("window_size", 100)),
                        keys_whitelist=rdcfg.get("keys_whitelist", None),
                        verbose=0,
                    )
                )

            print(
                f"\n[interleave] >>> phase={phase_tag} | steps={phase_steps} | num_envs={num_envs} | reward_tag={reward_tag}\n"
                f"            freeze_human={freeze_human} | freeze_exo={freeze_exo}\n"
                f"            resume={resume_zip}\n"
                f"            frozen_human={human_frozen_path}\n"
                f"            frozen_exo  ={exo_frozen_path}\n"
            )

            model.learn(
                total_timesteps=int(phase_steps),
                callback=CallbackList(callbacks),
                reset_num_timesteps=False,
                tb_log_name=f"{run_name}_{phase_tag}",
            )

            # 保存 latest + best alias
            latest_path = (run_dir / f"latest_{phase_tag}_model.zip").resolve()
            model.save(str(latest_path))

            best_zip = (best_dir / "best_model.zip").resolve()
            best_alias = (run_dir / f"best_{phase_tag}_model.zip").resolve()
            if best_zip.is_file():
                try:
                    shutil.copy2(str(best_zip), str(best_alias))
                except Exception as e:
                    print(f"[interleave] WARN: copy best model failed: {repr(e)}")

            # 最终保存 VecNormalize 统计（若启用）
            if isinstance(env, VecNormalize):
                try:
                    env.save(vecnorm_save_path)
                except Exception as e:
                    print(f"[interleave] WARN: VecNormalize.save failed: {repr(e)}")

            try:
                env.close(); eval_env.close()
            except Exception:
                pass

        # -------------------------
        # 交替训练 n 轮
        # -------------------------
        for cyc in range(1, n_cycles + 1):
            print(f"\n[interleave] ====== cycle {cyc}/{n_cycles} ======")

            # 1) 训练 Human：冻结 EXO（使用上一轮 best_exo）
            train_phase(
                phase_tag=f"human_c{cyc}",
                phase_steps=T_h,
                freeze_human=False,
                freeze_exo=True,
                human_frozen_path=str((best_human_dir / "best_model.zip").resolve()),
                exo_frozen_path=str((best_exo_dir / "best_model.zip").resolve()),
                best_dir=best_human_dir,
                ckpt_dir=ckpt_human_dir,
                vecnorm_save_path=vecnorm_human_path,
                reward_tag="human",
            )

            # 2) 训练 EXO：冻结 Human（使用上一轮 best_human）
            train_phase(
                phase_tag=f"exo_c{cyc}",
                phase_steps=T_e,
                freeze_human=True,
                freeze_exo=False,
                human_frozen_path=str((best_human_dir / "best_model.zip").resolve()),
                exo_frozen_path=str((best_exo_dir / "best_model.zip").resolve()),
                best_dir=best_exo_dir,
                ckpt_dir=ckpt_exo_dir,
                vecnorm_save_path=vecnorm_exo_path,
                reward_tag="exo",
            )

            # 3) 每 N_video 轮录一次视频（可选）
            if video_do and (cyc % video_every == 0):
                try:

                    video_out_dir = (run_dir / "videos" / f"cycle_{cyc:03d}").resolve()
                    video_out_dir.mkdir(parents=True, exist_ok=True)

                    def _unwrap_base_env(venv):
                        # VecNormalize -> VecMonitor -> DummyVecEnv/SubprocVecEnv
                        e = venv
                        while hasattr(e, "venv"):
                            e = e.venv
                        if hasattr(e, "envs") and len(e.envs) > 0:
                            base = e.envs[0]
                        else:
                            base = e
                        while hasattr(base, "env"):
                            base = base.env
                        return base

                    def record_one(model_zip: str, phase_name: str, freeze_human: bool, freeze_exo: bool, box_key: str, vecnorm_stats: Optional[str]):
                        # 用 VecEnv 以便可选 VecNormalize；渲染时取 base env
                        env_kwargs = dict(
                            clip_obs=clip_obs,
                            model_path=model_xml,
                            obsd_model_path=obsd_xml,
                            reset_type=reset_type,
                            target_y_vel=target_y_vel,
                            hip_period=hip_period,
                            human_pretrained_path=str((best_human_dir / "best_model.zip").resolve()),
                            freeze_human=bool(freeze_human),
                            human_policy_device=human_policy_device,
                            human_deterministic=True,
                            exo_pretrained_path=str((best_exo_dir / "best_model.zip").resolve()),
                            freeze_exo=bool(freeze_exo),
                            exo_policy_device=exo_policy_device,
                            exo_deterministic=True,
                            interleave_phase=phase_name,
                        )

                        env = DummyVecEnv([make_env(999, seed, env_kwargs)])
                        env = VecMonitor(env, filename=None)
                        if use_vecnormalize and (not isinstance(env.observation_space, spaces.Dict)):
                            env = maybe_wrap_vecnormalize(env, clip_obs=clip_obs, norm_reward=False, stats_path=vecnorm_stats, training=False)

                        model = load_resume_ppo(model_zip, env=env, device=device, pcfg=pcfg)
                        obs = env.reset()

                        frames = []
                        base_env = _unwrap_base_env(env)
                        rollout_len = int(vcfg.get("rollout_length", 1000))

                        for _ in range(rollout_len):
                            frames.append(get_frame(base_env, vcfg))
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, dones, _ = env.step(action)
                            if bool(dones[0]):
                                break

                        out_path = (video_out_dir / f"{phase_name}.{str(vcfg.get('format', 'mp4'))}").resolve()
                        write_video(frames, out_path, fps=int(vcfg.get('fps', 30)))

                        try:
                            env.close()
                        except Exception:
                            pass

                    # 录 human / exo 两段
                    record_one(
                        model_zip=str((best_human_dir / "best_model.zip").resolve()),
                        phase_name="human",
                        freeze_human=False,
                        freeze_exo=True,
                        box_key="human",
                        vecnorm_stats=vecnorm_human_path if (use_vecnormalize and Path(vecnorm_human_path).is_file()) else None,
                    )
                    record_one(
                        model_zip=str((best_exo_dir / "best_model.zip").resolve()),
                        phase_name="exo",
                        freeze_human=True,
                        freeze_exo=False,
                        box_key="exo",
                        vecnorm_stats=vecnorm_exo_path if (use_vecnormalize and Path(vecnorm_exo_path).is_file()) else None,
                    )
                    print(f"[interleave] video saved under: {video_out_dir}")
                except Exception as e:
                    print(f"[interleave] WARN: record video failed: {repr(e)}")

        # cycle 结束：上传 best artifacts（可选）
        if wandb_enabled and (wandb_run is not None):
            try:
                bh = (run_dir / "best_human_model.zip")
                be = (run_dir / "best_exo_model.zip")
                if (best_human_dir / "best_model.zip").is_file():
                    shutil.copy2(str(best_human_dir / "best_model.zip"), str(bh))
                if (best_exo_dir / "best_model.zip").is_file():
                    shutil.copy2(str(best_exo_dir / "best_model.zip"), str(be))
                if bh.is_file():
                    log_wandb_artifact_best_model(wandb_run, bh, mode_tag="best_human", ts_dir=ts_dir, reward_weights_used=pick_reward_weights("human"), extra_meta={"run_dir": str(run_dir)})
                if be.is_file():
                    log_wandb_artifact_best_model(wandb_run, be, mode_tag="best_exo", ts_dir=ts_dir, reward_weights_used=pick_reward_weights("exo"), extra_meta={"run_dir": str(run_dir)})
            except Exception as e:
                print(f"[interleave] WARN: upload best artifacts failed: {repr(e)}")

        # wandb finish
        try:
            if wandb_enabled and (wandb_run is not None):
                import wandb as _wandb  # type: ignore
                _wandb.finish()
        except Exception as e:
            print(f"[interleave] WARN: wandb.finish failed: {repr(e)}")

        print(f"[interleave] DONE. run_dir={run_dir}")
        return

    # ---------------------------------------------------------------------
    # Non-interleave: 原单模式训练流程
    # ---------------------------------------------------------------------
    freeze_human = freeze_human_cfg
    freeze_exo = freeze_exo_cfg
    reward_weights_used = pick_reward_weights(mode_tag)
    if len(reward_weights_used) == 0:
        print("[train_exo] WARN: reward_weights 为空或不可解析，将使用环境内部 DEFAULT_RWD_KEYS_AND_WEIGHTS。")

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
        interleave_phase="none",
    )
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

    is_dict_obs = isinstance(env.observation_space, spaces.Dict)
    vecnorm_save_path = str(run_dir / "vecnormalize.pkl")
    vecnorm_stats_path = find_vecnormalize_stats_path(resume_path if is_resume else None, tcfg, mode_tag)

    if use_vecnormalize and (not is_dict_obs):
        env = maybe_wrap_vecnormalize(env, clip_obs=clip_obs, norm_reward=norm_reward, stats_path=vecnorm_stats_path, training=True)
        eval_env = maybe_wrap_vecnormalize(eval_env, clip_obs=clip_obs, norm_reward=norm_reward, stats_path=vecnorm_stats_path, training=False)

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
        if freeze_exo:
            try:
                from sb3_lattice_policy import LatticeActorCriticPolicy  # noqa: F401
            except Exception as e:
                print(f"[train_exo] WARN: import LatticeActorCriticPolicy failed: {repr(e)}")
        model = load_resume_ppo(resume_path, env=env, device=device, pcfg=pcfg)
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
    wcb = build_wandb_callback(mode_tag)
    if wcb is not None:
        callbacks.append(wcb)

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

    if use_vecnormalize and (not is_dict_obs):
        callbacks.append(SaveVecNormalizeCallback(save_freq=save_freq, save_path=vecnorm_save_path, verbose=0))

    rdcfg = (tcfg.get("reward_decompose", {}) or {})
    if bool(rdcfg.get("enable", True)):
        callbacks.append(
            RewardDecomposeCallback(
                window_size=int(rdcfg.get("window_size", 100)),
                keys_whitelist=rdcfg.get("keys_whitelist", None),
                verbose=0,
            )
        )

    # video：默认放在 <run_dir>/videos/
    if bool(vcfg.get("video_enabled", True)):
        video_out_dir = (run_dir / "videos").resolve(); video_out_dir.mkdir(parents=True, exist_ok=True)
        v_save_freq = int(vcfg.get("save_freq", int(tcfg.get("save_frequency", 200_000))))
        v_save_freq = max(v_save_freq // max(num_envs, 1), 1)
        box_key = ("exo" if freeze_human else ("human" if freeze_exo else None))
        callbacks.append(
            VideoRecorderCallback(
                env_kwargs=env_kwargs,
                video_cfg=vcfg,
                save_freq=v_save_freq,
                out_dir=video_out_dir,
                rollout_length=int(vcfg.get("rollout_length", 1000)),
                fps=int(vcfg.get("fps", 30)),
                fmt=str(vcfg.get("format", "mp4")),
                box_key=box_key,
                verbose=1,
            )
        )

    print(
        f"[train_exo] mode={mode_tag} | run_dir={run_dir} | run_name={run_name} | ts={ts_dir}\n"
        f"           total_timesteps={total_timesteps} | num_envs={num_envs} | freeze_human={freeze_human} | freeze_exo={freeze_exo} | is_resume={is_resume}"
    )
    print("[train_exo] callbacks:", [type(c).__name__ for c in callbacks])

    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks), reset_num_timesteps=(not is_resume), tb_log_name=run_name)

    final_path = (run_dir / f"final_{mode_tag}_model.zip").resolve()
    model.save(str(final_path))
    if isinstance(env, VecNormalize):
        try:
            env.save(vecnorm_save_path)
            print(f"[OK] VecNormalize stats saved to: {vecnorm_save_path}")
        except Exception as e:
            print(f"[train_exo] WARN: final VecNormalize.save failed: {repr(e)}")

    print(f"[OK] Training done. Final model saved to: {final_path}")

    best_zip = (run_dir / f"best_model_{mode_tag}" / "best_model.zip")
    if best_zip.is_file():
        best_alias = (run_dir / f"best_{mode_tag}_model.zip").resolve()
        try:
            shutil.copy2(str(best_zip), str(best_alias))
            print(f"[OK] Best model copied to: {best_alias}")
        except Exception as e:
            print(f"[train_exo] WARN: copy best model failed: {repr(e)}")

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
                log_wandb_artifact_best_model(wandb_run, cand, mode_tag=mode_tag, ts_dir=ts_dir, reward_weights_used=reward_weights_used, extra_meta={"run_dir": str(run_dir)})
                print(f"[train_exo] wandb artifact uploaded: {cand}")
        except Exception as e:
            print(f"[train_exo] WARN: upload best model artifact failed: {repr(e)}")

    try:
        if wandb_enabled and (wandb_run is not None):
            import wandb as _wandb  # type: ignore
            _wandb.finish()
            print("[train_exo] wandb finished.")
    except Exception as e:
        print(f"[train_exo] WARN: wandb.finish failed: {repr(e)}")


if __name__ == "__main__":
    main()