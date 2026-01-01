# -*- coding: utf-8 -*-
"""
evaluate_muscle_saturation.py (MyoSuite / Gymnasium + SB3)

目的：
- 单纯评估“肌肉激活饱和比例”（muscle saturation ratio）
- 复用你现有 evaluate_model 脚本的整体结构（DummyVecEnv/VecNormalize/PPO.load + episode rollouts）

定义（每个 step）：
- 给定肌肉激活向量 m \in R^N（通常取 sim.data.act 或 sim.data.ctrl，范围近似 [0,1]）
- sat_ratio_step = #{i | m_i > sat_threshold} / N
- sat_any_step   = 1{ max(m) > sat_threshold }

输出：
- metrics_muscle.csv：逐 episode 指标
- summary_muscle.json：整体汇总（跨 episode 均值/方差等）
"""

import os
import sys
import json
import csv
import time
import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

import gymnasium as gym
import myosuite  # noqa: F401 触发 MyoSuite env 注册

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ============================================================
# 0) 解决 cloudpickle 反序列化：sb3_lattice_policy 找不到的问题（按需）
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    _m = importlib.import_module("scripts.sb3_lattice_policy")
    sys.modules["sb3_lattice_policy"] = _m
except Exception as e:
    # 不强制要求存在；只是兼容你之前的训练脚本
    print(f"[WARN] alias scripts.sb3_lattice_policy -> sb3_lattice_policy failed: {e}", flush=True)


# ============================================================
# 1) Env helpers
# ============================================================
def _get_sim(env):
    """尽量从 wrapper 栈中拿到 sim（MyoSuite 通常有 sim）"""
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
# 2) 肌肉激活提取（鲁棒）
# ============================================================
def _score_01(arr: np.ndarray) -> float:
    """
    评分：越像 [0,1] 的激活越高分；并轻微偏好更长的向量
    """
    if arr is None:
        return -1e9
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.size == 0:
        return -1e9
    fin = np.isfinite(a)
    if not np.any(fin):
        return -1e9
    a = a[fin]
    in01 = np.mean((a >= -1e-6) & (a <= 1.0 + 1e-6))
    # log1p(size) 防止极端偏好大向量
    return float(10.0 * in01 + np.log1p(a.size))


def extract_muscle_activation(
    env,
    info: Optional[Dict[str, Any]] = None,
    prefer: str = "auto",
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    返回 (m, source)
    - prefer:
        auto  : 自动在多个候选中选最像 [0,1] 的
        act   : 强制用 sim.data.act
        ctrl  : 强制用 sim.data.ctrl
        info  : 强制从 info 字段里找
    """
    prefer = str(prefer).strip().lower()
    sim = _get_sim(env)

    # 1) info 里找（如果你的 env.step 里已经把肌肉激活写进 info）
    def _from_info(d: Optional[Dict[str, Any]]):
        if not isinstance(d, dict):
            return None, None
        # 常见命名尝试
        keys = [
            "muscle_activation", "muscle_activations", "muscle_act",
            "act", "activation", "activations",
            "muscle", "muscles",
            "muscle_excitation", "muscle_excitations", "excitation", "excitations",
        ]
        for k in keys:
            if k in d:
                try:
                    arr = np.asarray(d[k], dtype=float).reshape(-1)
                    if arr.size > 0:
                        return arr, f"info[{k}]"
                except Exception:
                    pass
        return None, None

    # 2) sim.data.act / sim.data.ctrl
    def _from_sim_act():
        if sim is None or not hasattr(sim, "data"):
            return None, None
        if hasattr(sim.data, "act"):
            try:
                arr = np.asarray(sim.data.act, dtype=float).reshape(-1).copy()
                if arr.size > 0:
                    return arr, "sim.data.act"
            except Exception:
                pass
        return None, None

    def _from_sim_ctrl():
        if sim is None or not hasattr(sim, "data"):
            return None, None
        if hasattr(sim.data, "ctrl"):
            try:
                arr = np.asarray(sim.data.ctrl, dtype=float).reshape(-1).copy()
                if arr.size > 0:
                    return arr, "sim.data.ctrl"
            except Exception:
                pass
        return None, None

    # prefer 分支
    if prefer == "info":
        return _from_info(info)
    if prefer == "act":
        return _from_sim_act()
    if prefer == "ctrl":
        return _from_sim_ctrl()

    # auto：候选里择优
    cands: List[Tuple[np.ndarray, str]] = []
    a, s = _from_sim_act()
    if a is not None:
        cands.append((a, s))
    a, s = _from_sim_ctrl()
    if a is not None:
        cands.append((a, s))
    a, s = _from_info(info)
    if a is not None:
        cands.append((a, s))

    if len(cands) == 0:
        return None, None

    best = None
    best_score = -1e18
    for arr, src in cands:
        sc = _score_01(arr)
        if sc > best_score:
            best_score = sc
            best = (arr, src)
    return best


# ============================================================
# 3) 统计工具
# ============================================================
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


def _mean_std(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    x = [v for v in vals if (v is not None and np.isfinite(v))]
    if len(x) == 0:
        return None, None
    a = np.asarray(x, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=0))


# ============================================================
# 4) 评估主逻辑（只做肌肉饱和）
# ============================================================
def evaluate_muscle_saturation(
    env_id: str,
    model_path: str,
    vecnorm_path: Optional[str],
    out_dir: str,
    n_eval: int = 10,
    seed: int = 0,
    deterministic: bool = True,
    max_steps: int = 0,
    sat_threshold: float = 0.95,
    activation_source: str = "auto",
):
    os.makedirs(out_dir, exist_ok=True)
    set_random_seed(seed)

    venv = DummyVecEnv([make_env(env_id, seed)])

    if vecnorm_path and os.path.isfile(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(model_path, device="auto")

    # episode horizon：优先 max_steps；否则用 env.spec.max_episode_steps；再兜底 1000
    base_env0 = venv.envs[0]
    spec_steps = None
    try:
        spec_steps = getattr(getattr(base_env0, "spec", None), "max_episode_steps", None)
    except Exception:
        spec_steps = None
    horizon = int(max_steps) if (max_steps and max_steps > 0) else (int(spec_steps) if spec_steps else 1000)

    rows: List[Dict[str, Any]] = []
    all_step_sat_ratio: List[float] = []   # 全局（跨 episode）step 级 sat_ratio
    all_step_m_u_max: List[float] = []     # 全局 step 级 max(m)
    all_step_m_mean: List[float] = []      # 全局 step 级 mean(m)

    t0 = time.time()
    n_no_muscle = 0

    for ep in range(int(n_eval)):
        ep_seed = seed + 1000 * ep
        try:
            obs = venv.reset(seed=ep_seed)
        except TypeError:
            obs = venv.reset()

        base_env = venv.envs[0]
        dt = get_dt(base_env)

        ep_len = 0
        done = False

        # per-step cache
        step_sat_ratio: List[float] = []
        step_sat_any: List[int] = []
        step_m_u_max: List[float] = []
        step_m_mean: List[float] = []
        step_m_std: List[float] = []
        src_used: Optional[str] = None
        n_steps_valid = 0
        n_m_size = None

        while not done and ep_len < horizon:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = venv.step(action)
            ep_len += 1
            done = bool(np.asarray(dones).reshape(-1)[0])

            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            m, src = extract_muscle_activation(base_env, info=info if isinstance(info, dict) else None, prefer=activation_source)
            if m is None or m.size == 0:
                continue

            # 记录一次 source（以 episode 第一次成功为准）
            if src_used is None:
                src_used = src
                n_m_size = int(np.asarray(m).size)

            m = np.asarray(m, dtype=float).reshape(-1)
            m = m[np.isfinite(m)]
            if m.size == 0:
                continue

            n_steps_valid += 1
            m_mean = float(np.mean(m))
            m_std = float(np.std(m, ddof=0))
            m_u_max = float(np.max(m))

            n_saturated = int(np.sum(m > float(sat_threshold)))
            sat_ratio = float(n_saturated / float(m.size))
            sat_any = int(m_u_max > float(sat_threshold))

            step_m_mean.append(m_mean)
            step_m_std.append(m_std)
            step_m_u_max.append(m_u_max)
            step_sat_ratio.append(sat_ratio)
            step_sat_any.append(sat_any)

            # global
            all_step_sat_ratio.append(sat_ratio)
            all_step_m_u_max.append(m_u_max)
            all_step_m_mean.append(m_mean)

        if n_steps_valid == 0:
            n_no_muscle += 1

        # episode stats
        sat_ratio_dist = summarize_distribution(np.asarray(step_sat_ratio, dtype=float))
        u_max_dist = summarize_distribution(np.asarray(step_m_u_max, dtype=float))
        m_mean_dist = summarize_distribution(np.asarray(step_m_mean, dtype=float))

        sat_any_frac = float(np.mean(step_sat_any)) if len(step_sat_any) > 0 else None

        rows.append({
            "episode": ep,
            "seed": ep_seed,
            "length": int(ep_len),
            "dt": float(dt),
            "time_sec": float(ep_len * dt),

            "activation_source": src_used,
            "muscle_dim": n_m_size,

            "sat_threshold": float(sat_threshold),
            "sat_ratio_mean": (sat_ratio_dist["mean"] if sat_ratio_dist else None),
            "sat_ratio_std": (sat_ratio_dist["std"] if sat_ratio_dist else None),
            "sat_ratio_p95": (sat_ratio_dist["p95"] if sat_ratio_dist else None),
            "sat_any_frac": sat_any_frac,   # 有任一肌肉 > threshold 的时间比例

            "muscle_u_max_mean": (u_max_dist["mean"] if u_max_dist else None),
            "muscle_u_max_p95": (u_max_dist["p95"] if u_max_dist else None),
            "muscle_u_max_max": (u_max_dist["max"] if u_max_dist else None),

            "muscle_mean_mean": (m_mean_dist["mean"] if m_mean_dist else None),
            "muscle_mean_p95": (m_mean_dist["p95"] if m_mean_dist else None),

            "n_steps_valid": int(n_steps_valid),
        })

    # ============================================================
    # Save metrics CSV
    # ============================================================
    csv_path = os.path.join(out_dir, "metrics_muscle.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if len(rows) == 0:
            f.write("")
        else:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # ============================================================
    # Summary JSON
    # ============================================================
    sat_mean, sat_std = _mean_std([_safe_float(r["sat_ratio_mean"]) for r in rows])
    sat_p95_mean, sat_p95_std = _mean_std([_safe_float(r["sat_ratio_p95"]) for r in rows])
    sat_any_mean, sat_any_std = _mean_std([_safe_float(r["sat_any_frac"]) for r in rows])
    umax_mean, umax_std = _mean_std([_safe_float(r["muscle_u_max_max"]) for r in rows])
    valid_steps_mean, valid_steps_std = _mean_std([_safe_float(r["n_steps_valid"]) for r in rows])

    global_sat = summarize_distribution(np.asarray(all_step_sat_ratio, dtype=float)) if len(all_step_sat_ratio) else None
    global_umax = summarize_distribution(np.asarray(all_step_m_u_max, dtype=float)) if len(all_step_m_u_max) else None
    global_mmean = summarize_distribution(np.asarray(all_step_m_mean, dtype=float)) if len(all_step_m_mean) else None

    summary = {
        "env_id": env_id,
        "model_path": model_path,
        "vecnorm_path": vecnorm_path if vecnorm_path else None,
        "n_eval": int(n_eval),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "horizon_used": int(horizon),
        "elapsed_sec": float(time.time() - t0),

        "sat_threshold": float(sat_threshold),
        "activation_source_prefer": str(activation_source),

        # per-episode aggregate
        "sat_ratio_mean_mean": sat_mean,
        "sat_ratio_mean_std": sat_std,
        "sat_ratio_p95_mean": sat_p95_mean,
        "sat_ratio_p95_std": sat_p95_std,
        "sat_any_frac_mean": sat_any_mean,
        "sat_any_frac_std": sat_any_std,

        "muscle_u_max_max_mean": umax_mean,
        "muscle_u_max_max_std": umax_std,

        "n_steps_valid_mean": valid_steps_mean,
        "n_steps_valid_std": valid_steps_std,
        "episodes_no_muscle_data": int(n_no_muscle),

        # global step-level distributions（便于你后续画直方图/箱线图）
        "global_step_sat_ratio": global_sat,
        "global_step_muscle_u_max": global_umax,
        "global_step_muscle_mean": global_mmean,

        "saved_files": {
            "metrics_csv": csv_path,
            "summary_json": os.path.join(out_dir, "summary_muscle.json"),
        }
    }

    json_path = os.path.join(out_dir, "summary_muscle.json")
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

    p.add_argument("--env_id", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--vecnorm", type=str, default="")

    p.add_argument("--n_eval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="./eval_out_muscle")

    p.add_argument("--sat_threshold", type=float, default=0.95, help="例如 0.95 或 0.98")
    p.add_argument(
        "--activation_source",
        type=str,
        default="auto",
        choices=["auto", "act", "ctrl", "info"],
        help="auto: 自动选择最像[0,1]的候选；act: sim.data.act；ctrl: sim.data.ctrl；info: 从 info 字段取",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_muscle_saturation(
        env_id=str(args.env_id),
        model_path=str(args.model),
        vecnorm_path=str(args.vecnorm) if str(args.vecnorm).strip() else None,
        out_dir=str(args.out_dir),
        n_eval=int(args.n_eval),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        max_steps=int(args.max_steps),
        sat_threshold=float(args.sat_threshold),
        activation_source=str(args.activation_source),
    )