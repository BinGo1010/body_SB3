# -*- coding: utf-8 -*-
"""
evaluate_model_final.py (MyoSuite / Gymnasium + SB3)

输出文件：
- metrics.csv：逐 episode 评估指标（包含 y 方向“前进”(forward) 与 x 方向“横向漂移”(drift)）
- summary.json：精简汇总（适合快速查看/写论文）
- summary_full.json：全量汇总（便于追溯/复现）
- speed_body_distribution.json：逐 episode 的速度分布（可选含直方图）
- 可选：评估视频（离屏抓帧写 mp4）

关键兼容点：
1) MyoSuite 常见 “recenter / treadmill” 机制：root 位置可能被回中，导致 raw 位移恒为 0
   -> 推荐使用 dist_integrated_* = ∑ v * dt 作为“真实净位移/净进展”
2) 渲染：部分 MyoSuite 环境未实现 gymnasium 的 env.render()，或内部用 dm_control physics.render()
   -> get_frame() 自动寻找 physics / sim 内部可用的 render()，做离屏抓帧
3) SB3 模型反序列化：若训练时使用自定义 policy（例如 sb3_lattice_policy），load 时需要 module 可导入
   -> 提供 import alias：scripts.sb3_lattice_policy -> sb3_lattice_policy
"""

import os
import sys
import json
import time
import csv
import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

try:
    import yaml
except Exception as e:
    raise RuntimeError(f"Missing dependency: pyyaml. Install it. Error: {e}")

import gymnasium as gym
import myosuite  # noqa: F401 触发 MyoSuite env 注册

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ============================================================
# 0) 解决 cloudpickle 反序列化：sb3_lattice_policy 找不到的问题
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    _m = importlib.import_module("scripts.sb3_lattice_policy")
    # 关键：把它注册成顶层模块名 sb3_lattice_policy，供 cloudpickle import
    sys.modules["sb3_lattice_policy"] = _m
except Exception as e:
    print(f"[WARN] alias scripts.sb3_lattice_policy -> sb3_lattice_policy failed: {e}", flush=True)


# ============================================================
# 1) Env/Sim helpers
# ============================================================
def _get_sim(env):
    """尽量从 MyoSuite/Gymnasium wrapper 栈中拿到 sim（MyoSuite 通常有 sim）"""
    unwrapped = getattr(env, "unwrapped", env)
    sim = getattr(unwrapped, "sim", None)
    if sim is not None:
        return sim
    try:
        return env.get_wrapper_attr("sim")
    except Exception:
        return None


def get_dt(env) -> float:
    """优先 env.dt；否则用 sim.model.opt.timestep * frame_skip 估计"""
    unwrapped = getattr(env, "unwrapped", env)
    dt = getattr(unwrapped, "dt", None)
    if dt is not None:
        try:
            return float(dt)
        except Exception:
            pass

    sim = _get_sim(env)
    if sim is not None and hasattr(sim, "model"):
        try:
            base_ts = float(sim.model.opt.timestep)
            frame_skip = getattr(unwrapped, "frame_skip", 1)
            return base_ts * float(frame_skip)
        except Exception:
            pass

    return 1.0


def make_env(env_id: str, seed: int):
    """给 DummyVecEnv 用的 env thunk"""
    def _thunk():
        env = gym.make(env_id)
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()
        return env
    return _thunk


# ============================================================
# 2) root body / 位移 / 速度
# ============================================================
def _pick_root_body_id(sim) -> int:
    """
    选一个更像“root”的 body：
    先按名字匹配 pelvis/torso/root/...；否则选最大质量（跳过 world body=0）
    """
    preferred = ["pelvis", "torso", "root", "trunk", "abdomen", "spine", "hip", "head"]
    names: List[str] = []
    nbody = int(getattr(sim.model, "nbody", 0) or 0)
    for i in range(nbody):
        try:
            names.append(sim.model.body(i).name)
        except Exception:
            names.append("")

    for key in preferred:
        for i, nm in enumerate(names):
            if nm and key in nm.lower():
                return i

    try:
        mass = np.asarray(sim.model.body_mass).reshape(-1)
        if mass.shape[0] == nbody and nbody > 1:
            return int(np.argmax(mass[1:]) + 1)
    except Exception:
        pass

    return 1 if nbody > 1 else 0


def get_root_body_info(env) -> Tuple[Optional[int], Optional[str]]:
    sim = _get_sim(env)
    if sim is None or not hasattr(sim, "model"):
        return None, None
    bid = _pick_root_body_id(sim)
    try:
        name = sim.model.body(bid).name
    except Exception:
        name = None
    return bid, name


def get_body_xyz(env, bid: Optional[int]) -> Optional[np.ndarray]:
    sim = _get_sim(env)
    if sim is None or bid is None or not hasattr(sim, "data"):
        return None
    try:
        pos = np.asarray(sim.data.xpos[bid]).reshape(-1)
        return pos[:3].copy() if pos.size >= 3 else None
    except Exception:
        return None


def get_body_xy(env, bid: Optional[int]) -> Optional[np.ndarray]:
    p = get_body_xyz(env, bid)
    if p is None:
        return None
    return p[:2].copy()


def get_body_vxyz(env, bid: Optional[int]) -> Optional[np.ndarray]:
    """
    获取 body 线速度（世界系）：
    - 首选 sim.data.cvel[bid][3:6]（MuJoCo 线速度, COM）
    - 兜底：xvelp / body_xvelp（不同版本可能存在）
    """
    sim = _get_sim(env)
    if sim is None or bid is None or not hasattr(sim, "data"):
        return None

    data = sim.data

    # 首选：cvel[bid][3:6]（COM linear velocity）
    if hasattr(data, "cvel"):
        try:
            cv = np.asarray(data.cvel[bid]).reshape(-1)
            if cv.shape[0] >= 6:
                lin = cv[3:6]
                return np.asarray(lin, dtype=float)
        except Exception:
            pass

    # 兜底：老版本属性
    for attr in ["xvelp", "body_xvelp"]:
        if hasattr(data, attr):
            try:
                v = np.asarray(getattr(data, attr)[bid]).reshape(-1)
                if v.size >= 3:
                    return np.asarray(v[:3], dtype=float)
            except Exception:
                pass

    return None


def get_body_vxy(env, bid: Optional[int]) -> Optional[np.ndarray]:
    v = get_body_vxyz(env, bid)
    if v is None:
        return None
    return np.asarray(v[:2], dtype=float)


# ============================================================
# 3) forward axis / sign
# ============================================================
def parse_forward(forward: str) -> Tuple[int, float]:
    """
    定义“前进方向”：
    - x:  +x 为前进
    - y:  +y 为前进
    - -y: -y 为前进（很多 MyoSuite Walk 默认是朝 -y 走）
    """
    f = str(forward).strip().lower()
    if f == "x":
        return 0, +1.0
    if f in ["y", "+y"]:
        return 1, +1.0
    if f == "-y":
        return 1, -1.0
    raise ValueError(f"Unsupported --forward={forward!r}. Use x, y, +y, -y. (推荐 --forward=-y)")


def forward_component(vxy: Optional[np.ndarray], axis: int, sign: float) -> Optional[float]:
    """把 vxy 投影到“前进轴”，并按 sign 处理（例如 -y 前进时，vf=-vy）"""
    if vxy is None:
        return None
    vxy = np.asarray(vxy).reshape(-1)
    if vxy.shape[0] < 2:
        return None
    return float(vxy[0]) if axis == 0 else float(sign * vxy[1])


# ============================================================
# 4) 分布统计
# ============================================================
def summarize_distribution(samples: np.ndarray) -> Optional[Dict[str, Any]]:
    if samples is None:
        return None
    s = np.asarray(samples, dtype=float).reshape(-1)
    if s.size == 0:
        return None
    return {
        "mean": float(np.mean(s)),
        "std": float(np.std(s, ddof=0)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "p05": float(np.percentile(s, 5)),
        "p25": float(np.percentile(s, 25)),
        "p50": float(np.percentile(s, 50)),
        "p75": float(np.percentile(s, 75)),
        "p95": float(np.percentile(s, 95)),
        "n": int(s.size),
    }


def hist_distribution(samples: np.ndarray, bins: int = 50) -> Optional[Dict[str, Any]]:
    if samples is None:
        return None
    s = np.asarray(samples, dtype=float).reshape(-1)
    if s.size == 0:
        return None
    hist, edges = np.histogram(s, bins=int(bins))
    return {"bins": int(bins), "edges": edges.tolist(), "counts": hist.tolist()}


def _compact_speed_summary(all_speed_summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    summary.json 精简：只保留写论文最常用的统计量（避免 JSON 过大）
    """
    if not all_speed_summary:
        return None
    keep = ["mean", "std", "min", "max", "p05", "p50", "p95", "n"]
    return {k: all_speed_summary.get(k, None) for k in keep}


# ============================================================
# 5) 渲染（离屏抓帧，兼容 dm_control physics.render）
# ============================================================
def _is_gym_env_like(obj) -> bool:
    if obj is None:
        return False
    if hasattr(obj, "action_space") and hasattr(obj, "observation_space"):
        return True
    mod = getattr(type(obj), "__module__", "") or ""
    if mod.startswith("gym") or mod.startswith("gymnasium"):
        return True
    return False


def _find_render_candidates(env, sim=None):
    """在 env/sim 结构里找所有可 render() 的对象（优先 physics）"""
    cand = []

    def add(x):
        if x is None:
            return
        if _is_gym_env_like(x):
            return
        if hasattr(x, "render") and callable(getattr(x, "render")):
            cand.append(x)

    if sim is not None:
        add(getattr(sim, "_physics", None))
        add(getattr(sim, "physics", None))

    unwrapped = getattr(env, "unwrapped", None)
    for obj in [env, unwrapped]:
        if obj is None:
            continue
        for nm in ["_physics", "physics", "_mjcf_physics", "_dm_physics", "_sim_physics"]:
            add(getattr(obj, nm, None))
        for inner_nm in ["_env", "env", "_inner_env", "task", "_task"]:
            inner = getattr(obj, inner_nm, None)
            if inner is not None:
                add(getattr(inner, "_physics", None))
                add(getattr(inner, "physics", None))

    def scan_dict(obj):
        if obj is None or not hasattr(obj, "__dict__"):
            return
        for v in obj.__dict__.values():
            add(v)

    scan_dict(sim)
    scan_dict(unwrapped)
    scan_dict(env)

    uniq, seen = [], set()
    for x in cand:
        i = id(x)
        if i not in seen:
            uniq.append(x)
            seen.add(i)
    return uniq


def _as_uint8_rgb(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3:
        return img
    if img.dtype == np.uint8:
        return img
    mx = float(np.max(img)) if img.size else 1.0
    if mx <= 1.5:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def get_frame(env, width=640, height=480, camera=None, camera_id=None, camera_name=None) -> np.ndarray:
    """
    返回 RGB frame (H,W,3) uint8。
    支持 camera 指定：
    - camera=int -> camera_id
    - camera=str -> camera_name（可能是 dm_control 的 camera_id 也接受 str）
    """
    if camera is not None and camera_id is None and camera_name is None:
        if isinstance(camera, (int, np.integer)):
            camera_id = int(camera)
        elif isinstance(camera, str):
            camera_name = camera

    sim = _get_sim(env)
    candidates = _find_render_candidates(env, sim=sim)

    # 优先尝试 physics.render(height=, width=, camera_id=)
    for obj in candidates:
        # dm_control 风格：render(height=, width=, camera_id=)
        try:
            if camera_name is not None:
                img = obj.render(height=height, width=width, camera_id=camera_name)
            elif camera_id is not None:
                img = obj.render(height=height, width=width, camera_id=int(camera_id))
            else:
                img = obj.render(height=height, width=width)
            img = _as_uint8_rgb(np.asarray(img))
            if img.ndim == 3:
                return img
        except TypeError:
            pass
        except Exception:
            pass

        # 位置参数形式：render(h, w, camera_id=?)
        try:
            if camera_name is not None:
                img = obj.render(height, width, camera_id=camera_name)
            elif camera_id is not None:
                img = obj.render(height, width, camera_id=int(camera_id))
            else:
                img = obj.render(height, width)
            img = _as_uint8_rgb(np.asarray(img))
            if img.ndim == 3:
                return img
        except Exception:
            pass

        # 兜底：render()
        try:
            img = obj.render()
            if img is not None:
                img = _as_uint8_rgb(np.asarray(img))
                if img.ndim == 3:
                    return img
        except Exception:
            pass

        # 兜底：render(mode="rgb_array")
        try:
            img = obj.render(mode="rgb_array")
            if img is not None:
                img = _as_uint8_rgb(np.asarray(img))
                if img.ndim == 3:
                    return img
        except Exception:
            pass

    # 最后兜底：env.render（很多 MyoSuite gymnasium 环境不实现，这里只是兜底）
    for base in [env, getattr(env, "unwrapped", None)]:
        if base is None:
            continue
        try:
            img = base.render()
            if img is not None:
                img = _as_uint8_rgb(np.asarray(img))
                if img.ndim == 3:
                    return img
        except Exception:
            pass
        try:
            img = base.render(mode="rgb_array")
            if img is not None:
                img = _as_uint8_rgb(np.asarray(img))
                if img.ndim == 3:
                    return img
        except Exception:
            pass

    mtype = None
    if sim is not None and hasattr(sim, "model"):
        mtype = str(type(sim.model))
    raise RuntimeError(
        "Offscreen rendering failed. No candidate returned an RGB image.\n"
        f"sim.model type={mtype}\n"
    )


def record_episode_video(
    env_id: str,
    model,
    out_path: str,
    seed: int = 0,
    rollout_length: int = 1000,
    deterministic: bool = True,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    camera=None,
    camera_id=None,
    camera_name=None,
) -> Tuple[int, int]:
    """
    单独创建一个 env 录视频（不影响 DummyVecEnv 评估）
    """
    try:
        import skvideo.io  # noqa
    except Exception as e:
        raise RuntimeError(f"skvideo not available: {e}")

    # 尽力把 render_mode 设成 rgb_array（不一定被 env 支持，get_frame 会兜底）
    try:
        env = gym.make(env_id, disable_env_checker=True, render_mode="rgb_array")
    except TypeError:
        env = gym.make(env_id, disable_env_checker=True)

    obs, info = env.reset(seed=seed)
    frames = []
    steps = 0

    while steps < rollout_length:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        frame = get_frame(
            env,
            width=width,
            height=height,
            camera=camera,
            camera_id=camera_id,
            camera_name=camera_name,
        )
        frames.append(frame)

        steps += 1
        if terminated or truncated:
            break

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    import skvideo.io
    skvideo.io.vwrite(out_path, np.asarray(frames), outputdict={"-r": str(int(fps))})

    try:
        env.close()
    except Exception:
        pass

    return steps, len(frames)


# ============================================================
# 6) YAML + CLI merge（CLI 显式覆盖 > YAML > default）
# ============================================================
def _load_yaml(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_cfg(cfg: Dict[str, Any], args: argparse.Namespace, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并优先级：CLI（显式覆盖） > YAML > defaults
    约定：
    - bool 参数：CLI 只要置 True，就覆盖为 True；否则沿用 YAML/默认
    - 非 bool 参数：如果 CLI 与 default 不同，视为显式覆盖；否则用 YAML/默认
    """
    merged = dict(cfg) if cfg else {}
    for k, dval in defaults.items():
        if not hasattr(args, k):
            continue
        aval = getattr(args, k)

        if isinstance(dval, bool):
            if aval is True:
                merged[k] = True
            else:
                if k not in merged:
                    merged[k] = dval
            continue

        if aval != dval:
            merged[k] = aval
        else:
            if k not in merged:
                merged[k] = aval
    return merged


# ============================================================
# 7) 统计工具
# ============================================================
def _mean_std(arr: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    x = [v for v in arr if (v is not None and np.isfinite(v))]
    if len(x) == 0:
        return None, None
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=0))


def _safe_float(x):
    try:
        if x is None:
            return None
        y = float(x)
        if not np.isfinite(y):
            return None
        return y
    except Exception:
        return None


# ============================================================
# 8) 评估主逻辑
# ============================================================
def evaluate(
    env_id: str,
    model_path: str,
    vecnorm_path: Optional[str],
    n_eval: int,
    seed: int,
    deterministic: bool,
    max_steps: int,
    out_dir: str,

    forward: str,
    dist_thresh: Optional[float],
    speed_thresh: Optional[float],
    use_body_velocity_mean: bool,

    auto_mode: bool,
    in_place_dist_eps: float,
    in_place_speed_min: float,
    recenter_dist_min: float,

    record_video: bool,
    video_dir: str,
    video_episodes: int,
    video_length: int,
    video_fps: int,
    video_width: int,
    video_height: int,
    video_camera: Optional[str],
    video_camera_id: Optional[int],
    video_camera_name: Optional[str],

    save_speed_hist: bool,
    speed_hist_bins: int,
):
    os.makedirs(out_dir, exist_ok=True)
    set_random_seed(seed)

    axis, sign = parse_forward(forward)

    venv = DummyVecEnv([make_env(env_id, seed)])

    if vecnorm_path and os.path.isfile(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(model_path, device="auto")

    # episode horizon（优先 max_steps；否则用 env.spec.max_episode_steps；再兜底 1000）
    base_env0 = venv.envs[0]
    spec_steps = None
    try:
        spec_steps = getattr(getattr(base_env0, "spec", None), "max_episode_steps", None)
    except Exception:
        spec_steps = None
    horizon = int(max_steps) if (max_steps and max_steps > 0) else (int(spec_steps) if spec_steps else 1000)

    # ===== 录视频（可选）=====
    if record_video and video_episodes and video_episodes > 0:
        os.makedirs(video_dir, exist_ok=True)
        for ep in range(min(video_episodes, n_eval)):
            ep_seed = seed + 1000 * ep
            out_path = os.path.join(video_dir, f"eval_ep{ep:03d}_seed{ep_seed}.mp4")
            rec_len = int(video_length) if (video_length and video_length > 0) else horizon
            rec_len = min(rec_len, horizon)
            steps_rec, nframes = record_episode_video(
                env_id=env_id,
                model=model,
                out_path=out_path,
                seed=ep_seed,
                rollout_length=rec_len,
                deterministic=deterministic,
                fps=video_fps,
                width=video_width,
                height=video_height,
                camera=video_camera,
                camera_id=video_camera_id,
                camera_name=video_camera_name,
            )
            print(f"[OK] Video saved: {out_path} | steps={steps_rec} frames={nframes}", flush=True)

    # ===== 逐 episode 记录 =====
    rows: List[Dict[str, Any]] = []
    speed_dist_rows: List[Dict[str, Any]] = []
    all_forward_speed_samples: List[float] = []  # 汇总全部 step 的 forward 速度样本（用于整体分布）

    t0 = time.time()

    for ep in range(int(n_eval)):
        ep_seed = seed + 1000 * ep

        # reset
        try:
            obs = venv.reset(seed=ep_seed)
        except TypeError:
            obs = venv.reset()

        base_env = venv.envs[0]
        dt = get_dt(base_env)
        bid, bname = get_root_body_info(base_env)

        p0 = get_body_xy(base_env, bid)

        ep_return = 0.0
        ep_len = 0
        done = False
        term_reason = None

        # 速度样本（forward）
        v_forward_samples: List[float] = []

        # 关键：速度积分得到等效位移（forward 与 x漂移）
        dist_integrated_fwd = 0.0   # 方向由 forward_def 决定（例如 -y 前进）
        dist_integrated_x = 0.0     # world x 方向漂移

        n_v_used = 0

        while not done and ep_len < horizon:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = venv.step(action)

            ep_return += float(np.asarray(reward).reshape(-1)[0])
            ep_len += 1
            done = bool(np.asarray(dones).reshape(-1)[0])

            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            if isinstance(info, dict) and term_reason is None:
                for k in ["termination_reason", "done_reason", "fail_reason"]:
                    if k in info:
                        term_reason = f"{k}={info[k]}"

            # 速度 -> 积分位移
            if bid is not None:
                vxy = get_body_vxy(base_env, bid)
                if vxy is not None and np.all(np.isfinite(vxy)):
                    vx = float(vxy[0])
                    dist_integrated_x += vx * float(dt)

                    vf = forward_component(vxy, axis, sign)
                    if vf is not None and np.isfinite(vf):
                        v_forward_samples.append(float(vf))
                        dist_integrated_fwd += float(vf) * float(dt)
                        n_v_used += 1

        p1 = get_body_xy(base_env, bid)

        # ===== raw displacement（可能被 recenter 回中而变 0）=====
        dist_forward_raw = None
        dx_raw = dy_raw = None
        if (p0 is not None) and (p1 is not None):
            dx_raw = float(p1[0] - p0[0])
            dy_raw = float(p1[1] - p0[1])
            dist_forward_raw = dx_raw if axis == 0 else (sign * dy_raw)

        ep_time = float(ep_len * dt)

        mean_speed_disp_raw = None
        if dist_forward_raw is not None and ep_time > 1e-9:
            mean_speed_disp_raw = float(dist_forward_raw / ep_time)

        # ===== integrated mean speeds（推荐口径）=====
        mean_speed_int_fwd = None
        if ep_time > 1e-9 and n_v_used > 0:
            mean_speed_int_fwd = float(dist_integrated_fwd / ep_time)

        mean_speed_int_x = None
        if ep_time > 1e-9:
            mean_speed_int_x = float(dist_integrated_x / ep_time)

        # ===== speed distribution（forward）=====
        v_arr = np.asarray(v_forward_samples, dtype=float)
        if v_arr.size > 0:
            all_forward_speed_samples.extend([float(x) for x in v_arr.tolist()])
        v_dist = summarize_distribution(v_arr)
        v_hist = hist_distribution(v_arr, bins=speed_hist_bins) if (save_speed_hist and v_dist is not None) else None
        mean_speed_body_fwd = v_dist["mean"] if v_dist else None  # forward 方向“身体速度”均值

        # survive：未提前终止（启发式：跑满 95% horizon 视为存活）
        survive = (ep_len >= int(0.95 * horizon))

        # ============================================================
        # 自动模式：判别 recenter / 原地踏步 / 正常世界位移
        # ============================================================
        mode = "manual"
        dist_gate = dist_thresh
        speed_gate = speed_thresh

        dist_used = dist_forward_raw
        speed_used = mean_speed_disp_raw

        # 若用户要求使用“身体速度均值”，则把速度门限的观测速度切到 mean_speed_body_fwd
        if use_body_velocity_mean:
            if mean_speed_body_fwd is not None:
                speed_used = mean_speed_body_fwd

        if auto_mode:
            # recentered_move：raw 位移≈0 但 integrated 位移很大 -> 判定为“回中行走”
            recentered = False
            if dist_forward_raw is not None:
                if abs(dist_forward_raw) <= float(in_place_dist_eps) and abs(dist_integrated_fwd) >= float(recenter_dist_min):
                    recentered = True

            if recentered:
                mode = "recentered_move"
                dist_used = float(dist_integrated_fwd)
                if mean_speed_int_fwd is not None:
                    speed_used = mean_speed_int_fwd
                elif mean_speed_body_fwd is not None:
                    speed_used = mean_speed_body_fwd
            else:
                # in_place：raw 位移≈0 且 integrated 也很小，但身体速度均值较大 -> 原地踏步/跑步机
                in_place = False
                if dist_forward_raw is not None:
                    if abs(dist_forward_raw) <= float(in_place_dist_eps) and abs(dist_integrated_fwd) <= float(5.0 * in_place_dist_eps):
                        if mean_speed_body_fwd is not None and mean_speed_body_fwd >= float(in_place_speed_min):
                            in_place = True

                if in_place:
                    mode = "in_place"
                    # 原地模式通常不使用位移阈值
                    dist_gate = None
                    if speed_gate is None:
                        speed_gate = 0.2
                    if mean_speed_body_fwd is not None:
                        speed_used = mean_speed_body_fwd
                    elif mean_speed_int_fwd is not None:
                        speed_used = mean_speed_int_fwd
                else:
                    mode = "world_move"
                    # 若 raw 拿不到但可以积分，就切到 integrated
                    if dist_forward_raw is None and n_v_used > 0:
                        dist_used = float(dist_integrated_fwd)
                        speed_used = mean_speed_int_fwd if mean_speed_int_fwd is not None else speed_used

        # ============================================================
        # Gate 判定：progress_ok
        # ============================================================
        progress_ok = True
        if dist_gate is not None and dist_used is not None:
            progress_ok = progress_ok and (float(dist_used) >= float(dist_gate))
        if speed_gate is not None and speed_used is not None:
            progress_ok = progress_ok and (float(speed_used) >= float(speed_gate))

        success = bool(survive and progress_ok)

        rows.append({
            "episode": ep,
            "seed": ep_seed,

            # 回报与时长
            "return": ep_return,
            "return_per_step": ep_return / max(ep_len, 1),
            "length": ep_len,
            "dt": dt,
            "time_sec": ep_time,

            # root 选择信息（便于 debug）
            "root_body_id": bid,
            "root_body_name": bname,

            # raw displacement（可能被回中）
            "root_dx_raw": dx_raw,
            "root_dy_raw": dy_raw,
            "forward_def": forward,
            "dist_forward_raw": dist_forward_raw,
            "mean_speed_disp_raw": mean_speed_disp_raw,

            # integrated displacement/speed（推荐：真实净位移/净速度）
            "dist_integrated_fwd": float(dist_integrated_fwd),   # y 方向“前进”（由 forward_def 决定）
            "mean_speed_int_fwd": mean_speed_int_fwd,
            "dist_integrated_x": float(dist_integrated_x),       # x 方向漂移
            "mean_speed_int_x": mean_speed_int_x,
            "n_v_used": int(n_v_used),

            # forward 速度分布（身体速度）
            "mean_speed_body_fwd": mean_speed_body_fwd,
            "v_forward_std": (v_dist["std"] if v_dist else None),
            "v_forward_p05": (v_dist["p05"] if v_dist else None),
            "v_forward_p50": (v_dist["p50"] if v_dist else None),
            "v_forward_p95": (v_dist["p95"] if v_dist else None),
            "v_forward_n": (v_dist["n"] if v_dist else 0),

            # gating / mode
            "mode": mode,
            "dist_gate_used": dist_gate,
            "speed_gate_used": speed_gate,
            "dist_used": dist_used,
            "speed_used": speed_used,

            "survive": int(survive),
            "progress_ok": int(progress_ok),
            "success": int(success),

            "termination": term_reason,
        })

        speed_dist_rows.append({
            "episode": ep,
            "seed": ep_seed,
            "root_body_name": bname,
            "mode": mode,

            "forward_def": forward,
            "dist_forward_raw": dist_forward_raw,
            "dist_integrated_fwd": float(dist_integrated_fwd),
            "dist_integrated_x": float(dist_integrated_x),

            "mean_speed_disp_raw": mean_speed_disp_raw,
            "mean_speed_int_fwd": mean_speed_int_fwd,
            "mean_speed_int_x": mean_speed_int_x,
            "mean_speed_body_fwd": mean_speed_body_fwd,

            "dist_summary": v_dist,
            "hist": v_hist,
        })

    # ============================================================
    # Save metrics.csv
    # ============================================================
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if len(rows) == 0:
            f.write("")
        else:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Save per-episode distributions
    speed_json_path = os.path.join(out_dir, "speed_body_distribution.json")
    with open(speed_json_path, "w", encoding="utf-8") as f:
        json.dump(speed_dist_rows, f, ensure_ascii=False, indent=2)

    # ============================================================
    # Summary stats
    # ============================================================
    success_rate = float(np.mean([r["success"] for r in rows])) if rows else None
    survive_rate = float(np.mean([r["survive"] for r in rows])) if rows else None
    progress_ok_rate = float(np.mean([r["progress_ok"] for r in rows])) if rows else None

    ret_mean, ret_std = _mean_std([_safe_float(r["return"]) for r in rows])
    len_mean, len_std = _mean_std([_safe_float(r["length"]) for r in rows])

    # y-forward：raw/integrated（注意 raw 可能因 recenter 变 0）
    dist_raw_mean, dist_raw_std = _mean_std([_safe_float(r["dist_forward_raw"]) for r in rows])
    spd_raw_mean, spd_raw_std = _mean_std([_safe_float(r["mean_speed_disp_raw"]) for r in rows])

    dist_y_int_mean, dist_y_int_std = _mean_std([_safe_float(r["dist_integrated_fwd"]) for r in rows])
    spd_y_int_mean, spd_y_int_std = _mean_std([_safe_float(r["mean_speed_int_fwd"]) for r in rows])

    # x-drift：raw/integrated（新增）
    dx_raw_mean, dx_raw_std = _mean_std([_safe_float(r["root_dx_raw"]) for r in rows])
    dx_int_mean, dx_int_std = _mean_std([_safe_float(r["dist_integrated_x"]) for r in rows])

    dx_raw_abs_mean, _ = _mean_std([abs(_safe_float(r["root_dx_raw"])) if _safe_float(r["root_dx_raw"]) is not None else None for r in rows])
    dx_int_abs_mean, _ = _mean_std([abs(_safe_float(r["dist_integrated_x"])) if _safe_float(r["dist_integrated_x"]) is not None else None for r in rows])

    spd_x_int_mean, spd_x_int_std = _mean_std([_safe_float(r["mean_speed_int_x"]) for r in rows])

    # forward 身体速度均值（每 episode 的均值的再均值）
    spd_body_mean, spd_body_std = _mean_std([_safe_float(r["mean_speed_body_fwd"]) for r in rows])

    # mode counts
    mode_counts = {}
    for r in rows:
        m = r.get("mode", None)
        if m is None:
            continue
        mode_counts[m] = mode_counts.get(m, 0) + 1

    # 全部 step 的 forward 速度分布（合并所有 episode 的样本）
    all_speed_arr = np.asarray([float(x) for x in all_forward_speed_samples], dtype=float)
    all_speed_summary = summarize_distribution(all_speed_arr)
    all_speed_hist = hist_distribution(all_speed_arr, bins=speed_hist_bins) if (save_speed_hist and all_speed_summary) else None

    # ============================================================
    # summary_full.json（全量）
    # ============================================================
    full_summary = {
        # ---------- 复现信息 ----------
        "env_id": env_id,
        "model_path": model_path,
        "vecnorm_path": vecnorm_path if vecnorm_path else None,
        "n_eval": int(n_eval),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "elapsed_sec": float(time.time() - t0),

        # ---------- 评估设置 ----------
        "horizon_used": int(horizon),
        "forward_def": forward,
        "dist_thresh": dist_thresh,
        "speed_thresh": speed_thresh,
        "use_body_velocity_mean": bool(use_body_velocity_mean),
        "auto_mode": bool(auto_mode),
        "in_place_dist_eps": float(in_place_dist_eps),
        "in_place_speed_min": float(in_place_speed_min),
        "recenter_dist_min": float(recenter_dist_min),
        "mode_counts": mode_counts,

        # ---------- 通过率/稳定性 ----------
        "success_rate": success_rate,
        "survive_rate": survive_rate,
        "progress_ok_rate": progress_ok_rate,

        # ---------- 回报（对齐训练曲线） ----------
        "return_mean": ret_mean,
        "return_std": ret_std,

        # ---------- 时长 ----------
        "len_mean": len_mean,
        "len_std": len_std,

        # ---------- y 方向（前进/forward） ----------
        # raw 可能被 recenter 影响；integrated 推荐作为“真实净位移/净速度”
        "dist_forward_raw_mean": dist_raw_mean,
        "dist_forward_raw_std": dist_raw_std,
        "speed_disp_raw_mean": spd_raw_mean,
        "speed_disp_raw_std": spd_raw_std,
        "dist_y_integrated_mean": dist_y_int_mean,
        "dist_y_integrated_std": dist_y_int_std,
        "speed_y_int_mean": spd_y_int_mean,
        "speed_y_int_std": spd_y_int_std,

        # ---------- x 方向（横向漂移） ----------
        "dx_raw_mean": dx_raw_mean,
        "dx_raw_std": dx_raw_std,
        "dx_raw_abs_mean": dx_raw_abs_mean,
        "dx_integrated_mean": dx_int_mean,
        "dx_integrated_std": dx_int_std,
        "dx_integrated_abs_mean": dx_int_abs_mean,
        "speed_x_int_mean": spd_x_int_mean,
        "speed_x_int_std": spd_x_int_std,

        # ---------- forward 速度（身体速度） ----------
        "speed_body_fwd_mean": spd_body_mean,
        "speed_body_fwd_std": spd_body_std,
        "speed_body_all_summary": all_speed_summary,
        "speed_body_all_hist": all_speed_hist,

        # ---------- 输出文件 ----------
        "saved_files": {
            "metrics_csv": csv_path,
            "summary_json": os.path.join(out_dir, "summary.json"),
            "summary_full_json": os.path.join(out_dir, "summary_full.json"),
            "speed_body_distribution_json": speed_json_path,
            "video_dir": video_dir if record_video else None,
        }
    }

    # ============================================================
    # summary.json（精简）
    # 目标：保留“写论文/快速对比”最关键的口径
    # ============================================================
    slim_summary = {
        # ---------- 复现信息 ----------
        "env_id": env_id,
        "model_path": model_path,
        "vecnorm_path": vecnorm_path if vecnorm_path else None,
        "n_eval": int(n_eval),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "elapsed_sec": float(time.time() - t0),

        # ---------- 评估设置（关键）----------
        "horizon_used": int(horizon),
        "forward_def": forward,
        "dist_thresh": dist_thresh,
        "speed_thresh": speed_thresh,
        "auto_mode": bool(auto_mode),
        "mode_counts": mode_counts,

        # ---------- 核心通过率 ----------
        "success_rate": success_rate,
        "survive_rate": survive_rate,
        "progress_ok_rate": progress_ok_rate,

        # ---------- 回报与时长（稳定性） ----------
        "return_mean": ret_mean,
        "return_std": ret_std,
        "len_mean": len_mean,
        "len_std": len_std,

        # ---------- y 方向进展（推荐：integrated） ----------
        "dist_y_integrated_mean": dist_y_int_mean,
        "dist_y_integrated_std": dist_y_int_std,
        "speed_y_int_mean": spd_y_int_mean,
        "speed_y_int_std": spd_y_int_std,

        # ---------- x 方向漂移（推荐：integrated） ----------
        "dx_integrated_mean": dx_int_mean,
        "dx_integrated_std": dx_int_std,
        "dx_integrated_abs_mean": dx_int_abs_mean,
        "speed_x_int_mean": spd_x_int_mean,
        "speed_x_int_std": spd_x_int_std,

        # ---------- forward 速度分布（精简版） ----------
        "speed_body_all_summary": _compact_speed_summary(all_speed_summary),

        # ---------- 文件索引 ----------
        "saved_files": {
            "metrics_csv": csv_path,
            "summary_json": os.path.join(out_dir, "summary.json"),
            "summary_full_json": os.path.join(out_dir, "summary_full.json"),
            "speed_body_distribution_json": speed_json_path,
            "video_dir": video_dir if record_video else None,
        }
    }

    # 写文件
    json_path = os.path.join(out_dir, "summary.json")
    json_full_path = os.path.join(out_dir, "summary_full.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(slim_summary, f, ensure_ascii=False, indent=2)

    with open(json_full_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {csv_path}")
    print(f"[OK] Saved: {json_path}")
    print(f"[OK] Saved: {json_full_path}")
    print(f"[OK] Saved: {speed_json_path}")
    if record_video:
        print(f"[OK] Video dir: {video_dir}")
    print(json.dumps(slim_summary, ensure_ascii=False, indent=2))


# ============================================================
# 9) CLI
# ============================================================
def _patch_argv_forward(argv: List[str]) -> List[str]:
    """
    兼容用户常见写法：--forward -y
    argparse 默认会把 "-y" 当作选项而不是值，导致:
      evaluate_model.py: error: argument --forward: expected one argument
    这里把它重写为：--forward=-y
    """
    out = list(argv)
    if "--forward" in out:
        i = out.index("--forward")
        if i + 1 < len(out) and out[i + 1] in ["-y", "y", "+y", "x"]:
            out[i] = f"--forward={out[i+1]}"
            del out[i + 1]
    return out


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, default="/home/lvchen/body_SB3/configs/body_walk_evaluate_config.yaml")
    p.add_argument("--env_id", type=str, default=None)
    p.add_argument("--model", type=str, default="/home/lvchen/body_SB3/logs/walkenv/models/best/best_model.zip")
    p.add_argument("--vecnorm", type=str, default="")

    p.add_argument("--n_eval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--max_steps", type=int, default=0)

    # 注意：如果你想传 -y，请用 --forward=-y 或 --forward "-y"
    p.add_argument("--forward", type=str, default="-y", help='x, y, +y, -y（推荐：--forward=-y）')

    p.add_argument("--dist_thresh", type=float, default=0.5, help="<=0 disables")
    p.add_argument("--speed_thresh", type=float, default=0.0, help="<=0 disables")
    p.add_argument("--use_body_velocity_mean", action="store_true")

    p.add_argument("--auto_mode", action="store_true")
    p.add_argument("--in_place_dist_eps", type=float, default=1e-3)
    p.add_argument("--in_place_speed_min", type=float, default=0.1)
    p.add_argument("--recenter_dist_min", type=float, default=0.5)

    p.add_argument("--out_dir", type=str, default="./eval_out")

    p.add_argument("--record_video", action="store_true")
    p.add_argument("--video_dir", type=str, default="./eval_videos")
    p.add_argument("--video_episodes", type=int, default=1)
    p.add_argument("--video_length", type=int, default=2000)
    p.add_argument("--video_fps", type=int, default=30)
    p.add_argument("--video_width", type=int, default=640)
    p.add_argument("--video_height", type=int, default=480)

    p.add_argument("--video_camera", type=str, default="")
    p.add_argument("--video_camera_id", type=int, default=-1)
    p.add_argument("--video_camera_name", type=str, default="")

    p.add_argument("--save_speed_hist", action="store_true")
    p.add_argument("--speed_hist_bins", type=int, default=50)

    argv = _patch_argv_forward(sys.argv[1:])
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    cfg = _load_yaml(args.config)

    defaults = {
        "env_id": None,
        "model": "/home/lvchen/body_SB3/logs/walkenv/models/best/best_model.zip",
        "vecnorm": "",
        "n_eval": 10,
        "seed": 0,
        "deterministic": False,
        "max_steps": 0,
        "forward": "-y",
        "dist_thresh": 0.5,
        "speed_thresh": 0.0,
        "use_body_velocity_mean": False,
        "auto_mode": False,
        "in_place_dist_eps": 1e-3,
        "in_place_speed_min": 0.1,
        "recenter_dist_min": 0.5,
        "out_dir": "./eval_out",
        "record_video": False,
        "video_dir": "./eval_videos",
        "video_episodes": 1,
        "video_length": 2000,
        "video_fps": 30,
        "video_width": 640,
        "video_height": 480,
        "video_camera": "",
        "video_camera_id": -1,
        "video_camera_name": "",
        "save_speed_hist": False,
        "speed_hist_bins": 50,
    }

    merged = _merge_cfg(cfg, args, defaults)

    if not merged.get("env_id"):
        raise ValueError("env_id must be provided either in YAML config or via --env_id")

    # <=0 视为关闭
    dist_thresh = merged.get("dist_thresh", None)
    dist_thresh = None if (dist_thresh is None or float(dist_thresh) <= 0) else float(dist_thresh)

    speed_thresh = merged.get("speed_thresh", None)
    speed_thresh = None if (speed_thresh is None or float(speed_thresh) <= 0) else float(speed_thresh)

    # camera 参数清洗
    video_camera = merged.get("video_camera", "")
    video_camera = None if (video_camera is None or str(video_camera).strip() == "") else str(video_camera)

    video_camera_id = merged.get("video_camera_id", -1)
    video_camera_id = None if (video_camera_id is None or int(video_camera_id) < 0) else int(video_camera_id)

    video_camera_name = merged.get("video_camera_name", "")
    video_camera_name = None if (video_camera_name is None or str(video_camera_name).strip() == "") else str(video_camera_name)

    evaluate(
        env_id=str(merged["env_id"]),
        model_path=str(merged["model"]),
        vecnorm_path=str(merged.get("vecnorm", "")),
        n_eval=int(merged.get("n_eval", 10)),
        seed=int(merged.get("seed", 0)),
        deterministic=bool(merged.get("deterministic", False)),
        max_steps=int(merged.get("max_steps", 0)),
        out_dir=str(merged.get("out_dir", "./eval_out")),

        forward=str(merged.get("forward", "-y")),
        dist_thresh=dist_thresh,
        speed_thresh=speed_thresh,
        use_body_velocity_mean=bool(merged.get("use_body_velocity_mean", False)),

        auto_mode=bool(merged.get("auto_mode", False)),
        in_place_dist_eps=float(merged.get("in_place_dist_eps", 1e-3)),
        in_place_speed_min=float(merged.get("in_place_speed_min", 0.1)),
        recenter_dist_min=float(merged.get("recenter_dist_min", 0.5)),

        record_video=bool(merged.get("record_video", False)),
        video_dir=str(merged.get("video_dir", "./eval_videos")),
        video_episodes=int(merged.get("video_episodes", 1)),
        video_length=int(merged.get("video_length", 2000)),
        video_fps=int(merged.get("video_fps", 30)),
        video_width=int(merged.get("video_width", 640)),
        video_height=int(merged.get("video_height", 480)),
        video_camera=video_camera,
        video_camera_id=video_camera_id,
        video_camera_name=video_camera_name,

        save_speed_hist=bool(merged.get("save_speed_hist", False)),
        speed_hist_bins=int(merged.get("speed_hist_bins", 50)),
    )