# -*- coding: utf-8 -*-
"""evaluate_model_xy_simple.py

目的：
  按你的要求“删掉庞杂评价体系”，仅输出 **世界坐标系 x/y 方向** 的：
  - 位移（raw 与 速度积分 integrated 两种口径）
  - 速度（速度积分得到的平均速度 + body 速度采样均值/方差）

为什么同时保留 raw 与 integrated：
  MyoSuite/跑步机/回中(recenter)机制下，root 的 world 位置可能被回中，导致 raw 位移接近 0；
  这时 integrated（∑ v*dt）更能反映“真实净位移”。

输出：
  - metrics_xy.csv   : 每个 episode 一行
  - summary_xy.json  : 跨 episode 的均值/方差汇总

依赖：gymnasium, myosuite, stable-baselines3, pyyaml, numpy
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
#    （保留：不影响核心指标，但能避免加载失败）
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    _m = importlib.import_module("scripts.sb3_lattice_policy")
    sys.modules["sb3_lattice_policy"] = _m
except Exception:
    pass

from pathlib import Path

def resolve_model_path(p: str) -> str:
    """
    Robustly resolve SB3 model path:
    - If p is a directory: try common filenames inside.
    - If p is a file: fix double .zip.zip and ensure .zip suffix.
    """
    path = Path(p).expanduser()

    # If directory -> choose best_model.zip first, then model.zip, then any *.zip
    if path.exists() and path.is_dir():
        candidates = [
            path / "best_model.zip",
            path / "model.zip",
            path / "checkpoint.zip",
        ]
        for c in candidates:
            if c.exists():
                return str(c)

        zips = sorted(path.glob("*.zip"))
        if zips:
            return str(zips[0])

        raise FileNotFoundError(f"No .zip model found under directory: {path}")

    # If looks like file path but doesn't exist: repair common mistakes
    s = str(path)

    # fix ...zip.zip -> ...zip
    if s.endswith(".zip.zip"):
        s = s[:-4]  # remove the last ".zip"
        path = Path(s)

    # ensure .zip
    if not s.endswith(".zip"):
        s = s + ".zip"
        path = Path(s)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return str(path)


# ============================================================
# 1) Env/Sim helpers
# ============================================================
def _get_sim(env):
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
    """DummyVecEnv 的 env thunk"""

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

    # fallback: 最大质量（跳过 world=0）
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


def get_body_xy(env, bid: Optional[int]) -> Optional[np.ndarray]:
    sim = _get_sim(env)
    if sim is None or bid is None or not hasattr(sim, "data"):
        return None
    try:
        pos = np.asarray(sim.data.xpos[bid]).reshape(-1)
        return pos[:2].copy() if pos.size >= 2 else None
    except Exception:
        return None


def get_body_vxy(env, bid: Optional[int]) -> Optional[np.ndarray]:
    """body 线速度（世界系）"""
    sim = _get_sim(env)
    if sim is None or bid is None or not hasattr(sim, "data"):
        return None

    data = sim.data

    # 首选：cvel[bid][3:6]（COM 线速度）
    if hasattr(data, "cvel"):
        try:
            cv = np.asarray(data.cvel[bid]).reshape(-1)
            if cv.size >= 6:
                return np.asarray(cv[3:5], dtype=float)
        except Exception:
            pass

    # 兜底：不同版本属性
    for attr in ["xvelp", "body_xvelp"]:
        if hasattr(data, attr):
            try:
                v = np.asarray(getattr(data, attr)[bid]).reshape(-1)
                if v.size >= 2:
                    return np.asarray(v[:2], dtype=float)
            except Exception:
                pass

    return None


# ============================================================
# 3) YAML + CLI merge（CLI 显式覆盖 > YAML > default）
# ============================================================
def _load_yaml(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_cfg(cfg: Dict[str, Any], args: argparse.Namespace, defaults: Dict[str, Any]) -> Dict[str, Any]:
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


def _mean_std(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    x = [v for v in vals if (v is not None and np.isfinite(v))]
    if len(x) == 0:
        return None, None
    a = np.asarray(x, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=0))


# ============================================================
# 4) 评估主逻辑：只输出 x/y 速度与位移
# ============================================================
def evaluate_xy(
    env_id: str,
    model_path: str,
    vecnorm_path: Optional[str],
    n_eval: int,
    seed: int,
    deterministic: bool,
    max_steps: int,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    set_random_seed(seed)

    venv = DummyVecEnv([make_env(env_id, seed)])
    if vecnorm_path and os.path.isfile(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    model_path = resolve_model_path(model_path)
    model = PPO.load(model_path, device="auto")


    base_env0 = venv.envs[0]
    spec_steps = None
    try:
        spec_steps = getattr(getattr(base_env0, "spec", None), "max_episode_steps", None)
    except Exception:
        spec_steps = None
    horizon = int(max_steps) if (max_steps and max_steps > 0) else (int(spec_steps) if spec_steps else 1000)

    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep in range(int(n_eval)):
        ep_seed = seed + 1000 * ep
        try:
            _ = venv.reset(seed=ep_seed)
        except TypeError:
            _ = venv.reset()

        base_env = venv.envs[0]
        dt = float(get_dt(base_env))
        bid, bname = get_root_body_info(base_env)
        p0 = get_body_xy(base_env, bid)

        # 速度积分位移
        dx_int = 0.0
        dy_int = 0.0
        vx_samples: List[float] = []
        vy_samples: List[float] = []

        ep_len = 0
        done = False

        obs = None
        # DummyVecEnv.reset 已返回 obs，但我们不依赖它；这里重新取一次更清晰
        try:
            obs = venv.reset(seed=ep_seed)
        except TypeError:
            obs = venv.reset()

        while (not done) and (ep_len < horizon):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, dones, _infos = venv.step(action)
            ep_len += 1
            done = bool(np.asarray(dones).reshape(-1)[0])

            vxy = get_body_vxy(base_env, bid)
            if vxy is not None and np.all(np.isfinite(vxy)):
                vx = float(vxy[0])
                vy = float(vxy[1])
                dx_int += vx * dt
                dy_int += vy * dt
                vx_samples.append(vx)
                vy_samples.append(vy)

        p1 = get_body_xy(base_env, bid)
        dx_raw = dy_raw = None
        if (p0 is not None) and (p1 is not None):
            dx_raw = float(p1[0] - p0[0])
            dy_raw = float(p1[1] - p0[1])

        time_sec = float(ep_len * dt)
        mean_vx_int = (float(dx_int / time_sec) if time_sec > 1e-9 else None)
        mean_vy_int = (float(dy_int / time_sec) if time_sec > 1e-9 else None)

        vx_mean, vx_std = _mean_std(vx_samples)
        vy_mean, vy_std = _mean_std(vy_samples)

        rows.append({
            "episode": ep,
            "seed": ep_seed,
            "length": int(ep_len),
            "dt": dt,
            "time_sec": time_sec,
            "root_body_id": bid,
            "root_body_name": bname,

            # 位移：raw（可能受 recenter 影响）
            "dx_raw": dx_raw,
            "dy_raw": dy_raw,

            # 位移：integrated（推荐）
            "dx_int": float(dx_int),
            "dy_int": float(dy_int),

            # 速度：integrated mean（推荐）
            "mean_vx_int": mean_vx_int,
            "mean_vy_int": mean_vy_int,

            # 速度：body 采样统计（用于看波动）
            "vx_mean": vx_mean,
            "vx_std": vx_std,
            "vy_mean": vy_mean,
            "vy_std": vy_std,
            "n_v_samples": int(len(vx_samples)),
        })

    # =====================
    # 保存 metrics_xy.csv
    # =====================
    csv_path = os.path.join(out_dir, "metrics_xy.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("")

    # =====================
    # 保存 summary_xy.json
    # =====================
    summary = {
        "env_id": env_id,
        "model_path": model_path,
        "vecnorm_path": vecnorm_path if vecnorm_path else None,
        "n_eval": int(n_eval),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "horizon_used": int(horizon),
        "elapsed_sec": float(time.time() - t0),
        "saved_files": {
            "metrics_xy_csv": csv_path,
            "summary_xy_json": os.path.join(out_dir, "summary_xy.json"),
        },
        "stats": {
            # raw
            "dx_raw_mean": _mean_std([r["dx_raw"] for r in rows])[0],
            "dx_raw_std": _mean_std([r["dx_raw"] for r in rows])[1],
            "dy_raw_mean": _mean_std([r["dy_raw"] for r in rows])[0],
            "dy_raw_std": _mean_std([r["dy_raw"] for r in rows])[1],
            # integrated
            "dx_int_mean": _mean_std([r["dx_int"] for r in rows])[0],
            "dx_int_std": _mean_std([r["dx_int"] for r in rows])[1],
            "dy_int_mean": _mean_std([r["dy_int"] for r in rows])[0],
            "dy_int_std": _mean_std([r["dy_int"] for r in rows])[1],
            # mean speeds
            "mean_vx_int_mean": _mean_std([r["mean_vx_int"] for r in rows])[0],
            "mean_vx_int_std": _mean_std([r["mean_vx_int"] for r in rows])[1],
            "mean_vy_int_mean": _mean_std([r["mean_vy_int"] for r in rows])[0],
            "mean_vy_int_std": _mean_std([r["mean_vy_int"] for r in rows])[1],
        },
        "notes": [
            "dx/dy 为世界坐标系位移；raw=终点-起点（可能被 recenter 回中影响）。",
            "dx_int/dy_int=∑ v*dt（推荐用于 MyoSuite treadmill/recenter 口径）。",
        ],
    }

    json_path = os.path.join(out_dir, "summary_xy.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {csv_path}")
    print(f"[OK] Saved: {json_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ============================================================
# 5) CLI
# ============================================================
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
    p.add_argument("--out_dir", type=str, default="./eval_out")

    return p.parse_args()


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
        "out_dir": "./eval_out",
    }

    merged = _merge_cfg(cfg, args, defaults)
    if not merged.get("env_id"):
        raise ValueError("env_id must be provided either in YAML config or via --env_id")

    evaluate_xy(
        env_id=str(merged["env_id"]),
        model_path=str(merged.get("model", defaults["model"])),
        vecnorm_path=str(merged.get("vecnorm", "")) if merged.get("vecnorm") else "",
        n_eval=int(merged.get("n_eval", defaults["n_eval"])),
        seed=int(merged.get("seed", defaults["seed"])),
        deterministic=bool(merged.get("deterministic", defaults["deterministic"])),
        max_steps=int(merged.get("max_steps", defaults["max_steps"])),
        out_dir=str(merged.get("out_dir", defaults["out_dir"])),
    )