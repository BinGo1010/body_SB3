# -*- coding: utf-8 -*-
"""
diagnose_recenter.py

一次性验证环境是否存在：
A) 全局回中/搬运世界(recenter): 多个body的xpos在同一步出现几乎一致的平移
B) 跑步机式(treadmill): pelvis世界位移很小，但速度显著；且地面/跑带body在动而其他body不整体平移
C) 正常世界系: pelvis位移与速度积分一致，不出现整体平移事件

Usage (推荐用你的训练模型触发行走):
python diagnose_recenter.py --env_id myoLegWalk-v0 \
  --model /home/lvchen/body_SB3/logs/walkenv/models/best/best_model.zip \
  --steps 1200 --seed 4 --forward -y --out /home/lvchen/body_SB3/logs/walkenv/eval/diag

不带模型也可跑（但可能不走起来）:
python diagnose_recenter.py --env_id myoLegWalk-v0 --steps 1200 --seed 4
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np

import gymnasium as gym
import myosuite  # noqa: F401 触发注册

# 可选 SB3
try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


# ============== 兼容 cloudpickle 里 sb3_lattice_policy 的 import ==============
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import importlib
    _m = importlib.import_module("scripts.sb3_lattice_policy")
    sys.modules["sb3_lattice_policy"] = _m
except Exception:
    pass


def get_sim(env):
    unwrapped = getattr(env, "unwrapped", env)
    sim = getattr(unwrapped, "sim", None)
    if sim is not None:
        return sim
    try:
        return env.get_wrapper_attr("sim")
    except Exception:
        return None


def parse_forward(fwd: str) -> Tuple[int, float]:
    f = str(fwd).strip().lower()
    if f == "x":
        return 0, +1.0
    if f in ["y", "+y"]:
        return 1, +1.0
    if f == "-y":
        return 1, -1.0
    raise ValueError("forward must be one of: x, y, +y, -y")


def pick_body_id_by_name(sim, name_key: str) -> Optional[int]:
    # 优先用 body_name2id（你环境里通常有）
    try:
        return int(sim.model.body_name2id(name_key))
    except Exception:
        pass
    # 退化：扫名字
    try:
        nbody = int(sim.model.nbody)
        for i in range(nbody):
            nm = sim.model.body(i).name
            if nm and name_key.lower() == nm.lower():
                return i
        for i in range(nbody):
            nm = sim.model.body(i).name
            if nm and name_key.lower() in nm.lower():
                return i
    except Exception:
        pass
    return None


def find_ground_like_body(sim) -> Optional[int]:
    keys = ["floor", "ground", "terrain", "plane", "track", "tread", "belt"]
    try:
        nbody = int(sim.model.nbody)
        for i in range(nbody):
            nm = sim.model.body(i).name
            if not nm:
                continue
            low = nm.lower()
            if any(k in low for k in keys):
                return i
    except Exception:
        pass
    return None


def top_mass_body_ids(sim, k: int = 20, exclude: Optional[List[int]] = None) -> List[int]:
    exclude = set(exclude or [])
    ids = []
    try:
        mass = np.asarray(sim.model.body_mass).reshape(-1)
        nbody = mass.shape[0]
        cand = [(float(mass[i]), i) for i in range(nbody) if i not in exclude and i != 0]
        cand.sort(reverse=True)
        ids = [i for _, i in cand[:k]]
    except Exception:
        # 兜底：从 1..nbody-1 取前 k 个
        try:
            nbody = int(sim.model.nbody)
            ids = [i for i in range(1, min(nbody, k + 1)) if i not in exclude]
        except Exception:
            ids = []
    return ids


def body_xy(sim, bid: int) -> np.ndarray:
    # 你环境里常见为 sim.data.body_xpos
    if hasattr(sim.data, "body_xpos"):
        return np.asarray(sim.data.body_xpos[bid][:2], dtype=float)
    # 兜底：xpos
    if hasattr(sim.data, "xpos"):
        return np.asarray(sim.data.xpos[bid][:2], dtype=float)
    raise RuntimeError("No body_xpos/xpos available in sim.data")


def body_vxyz_from_cvel(sim, bid: int) -> Optional[np.ndarray]:
    # cvel: spatial velocity (rot(3), lin(3))
    if not hasattr(sim.data, "cvel"):
        return None
    cv = np.asarray(sim.data.cvel[bid]).reshape(-1)
    if cv.shape[0] < 6:
        return None
    # 注意：你原代码用了负号；这里不强行取负，直接输出原始，最后由 forward sign 处理
    return cv[3:6].astype(float)


def get_dt(env, sim) -> float:
    # MyoSuite 常用：sim.model.opt.timestep * frame_skip
    try:
        base_ts = float(sim.model.opt.timestep)
        fs = getattr(getattr(env, "unwrapped", env), "frame_skip", 1)
        return base_ts * float(fs)
    except Exception:
        return 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", required=True)
    ap.add_argument("--model", default="", help="optional SB3 PPO .zip")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forward", type=str, default="-y")
    ap.add_argument("--out", type=str, default="./diag_out")

    # 判据阈值（一般不需要改）
    ap.add_argument("--k_bodies", type=int, default=25, help="how many massive bodies to test global shift")
    ap.add_argument("--shift_min", type=float, default=1e-4, help="mean global shift magnitude threshold (m/step)")
    ap.add_argument("--shift_std_eps", type=float, default=2e-5, help="std of per-body shift around mean (m/step)")
    ap.add_argument("--tread_ground_min", type=float, default=1e-4, help="ground step movement threshold")
    ap.add_argument("--tread_others_max", type=float, default=5e-5, help="others mean movement threshold (treadmill)")
    ap.add_argument("--pelvis_origin_eps", type=float, default=0.05, help="pelvis near-origin radius for treadmill/in-place")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    axis, sign = parse_forward(args.forward)

    env = gym.make(args.env_id, disable_env_checker=True)
    obs, info = env.reset(seed=args.seed)

    sim = get_sim(env)
    if sim is None:
        raise RuntimeError("Cannot access sim from env. Need env.unwrapped.sim or wrapper attr 'sim'.")

    dt = get_dt(env, sim)
    pelvis_id = pick_body_id_by_name(sim, "pelvis")
    if pelvis_id is None:
        # 兜底：按质量最大body近似 root
        pelvis_id = top_mass_body_ids(sim, k=1)[0] if top_mass_body_ids(sim, k=1) else 1

    ground_id = find_ground_like_body(sim)

    # 用若干个“重体”body检测是否发生全局平移
    test_ids = top_mass_body_ids(sim, k=args.k_bodies, exclude=[pelvis_id])
    # 也把 pelvis 自己加进去观察一致性（可选）
    test_ids = [pelvis_id] + test_ids

    # 可选加载 SB3 模型
    model = None
    if args.model:
        if PPO is None:
            raise RuntimeError("stable-baselines3 not available in this env.")
        model = PPO.load(args.model, device="auto")

    # 初值
    prev_xy = {bid: body_xy(sim, bid) for bid in test_ids}
    prev_ground_xy = body_xy(sim, ground_id) if ground_id is not None else None

    # 累积指标
    global_shift_events = 0
    treadmill_like_events = 0
    pelvis_near_origin_steps = 0

    pelvis_xy0 = body_xy(sim, pelvis_id).copy()
    pelvis_xy_min = pelvis_xy0.copy()
    pelvis_xy_max = pelvis_xy0.copy()

    dist_int = 0.0  # 速度积分等效位移（forward）
    v_fwd_list = []

    # CSV
    csv_path = os.path.join(args.out, "diagnosis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "k", "time", "pelvis_x", "pelvis_y", "pelvis_dx", "pelvis_dy",
                "v_fwd", "dist_int",
                "global_shift_flag", "treadmill_flag",
                "mean_shift_x", "mean_shift_y", "std_shift",
                "ground_dx", "ground_dy"
            ],
        )
        writer.writeheader()

        for k in range(args.steps):
            # action
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # 尽量不引入随机：零动作
                if hasattr(env.action_space, "shape"):
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                else:
                    action = env.action_space.sample()

            obs, rew, terminated, truncated, info = env.step(action)

            # 当前 pelvis
            p_xy = body_xy(sim, pelvis_id)
            pelvis_xy_min = np.minimum(pelvis_xy_min, p_xy)
            pelvis_xy_max = np.maximum(pelvis_xy_max, p_xy)

            # 速度（用 pelvis cvel 的线速度）
            vxyz = body_vxyz_from_cvel(sim, pelvis_id)
            v_fwd = None
            if vxyz is not None:
                if axis == 0:
                    v_fwd = float(vxyz[0])
                else:
                    v_fwd = float(sign * vxyz[1])
                dist_int += v_fwd * dt
                v_fwd_list.append(v_fwd)

            # pelvis near origin?
            if float(np.linalg.norm(p_xy - pelvis_xy0)) <= float(args.pelvis_origin_eps):
                pelvis_near_origin_steps += 1

            # 全局平移检测：看多个 body 的 dxy 是否几乎一致
            dxy = []
            for bid in test_ids:
                cur = body_xy(sim, bid)
                dxy.append(cur - prev_xy[bid])
                prev_xy[bid] = cur
            dxy = np.asarray(dxy, dtype=float)  # (nb,2)
            mean_d = dxy.mean(axis=0)
            std_d = float(np.sqrt(np.mean(np.sum((dxy - mean_d[None, :]) ** 2, axis=1))))

            global_shift_flag = 0
            if float(np.linalg.norm(mean_d)) >= float(args.shift_min) and std_d <= float(args.shift_std_eps):
                global_shift_flag = 1
                global_shift_events += 1

            # 跑步机式检测（启发式）：
            # ground 在动，但“其他body的整体平移”很小（不是全局搬运），且 pelvis 长期在原点附近
            treadmill_flag = 0
            gdx = gdy = None
            if ground_id is not None and prev_ground_xy is not None:
                gxy = body_xy(sim, ground_id)
                gd = gxy - prev_ground_xy
                prev_ground_xy = gxy
                gdx, gdy = float(gd[0]), float(gd[1])

                # 其他 body 的平均平移大小（去掉 ground，只看 test_ids 的 mean_d 也可）
                others_mean_norm = float(np.linalg.norm(mean_d))
                ground_norm = float(np.linalg.norm(gd))

                if (ground_norm >= float(args.tread_ground_min)) and (others_mean_norm <= float(args.tread_others_max)) and (global_shift_flag == 0):
                    treadmill_flag = 1
                    treadmill_like_events += 1

            # 写 CSV
            if k == 0:
                pdx, pdy = 0.0, 0.0
            else:
                # pelvis 自身步进位移
                # 用 mean_d 不行；用 p_xy 与上一步 pelvis 保存更好，这里用 dxy 第一项就是 pelvis
                pdx, pdy = float(dxy[0, 0]), float(dxy[0, 1])

            writer.writerow({
                "k": k,
                "time": float(sim.data.time) if hasattr(sim.data, "time") else k * dt,
                "pelvis_x": float(p_xy[0]),
                "pelvis_y": float(p_xy[1]),
                "pelvis_dx": pdx,
                "pelvis_dy": pdy,
                "v_fwd": (None if v_fwd is None else float(v_fwd)),
                "dist_int": float(dist_int),
                "global_shift_flag": global_shift_flag,
                "treadmill_flag": treadmill_flag,
                "mean_shift_x": float(mean_d[0]),
                "mean_shift_y": float(mean_d[1]),
                "std_shift": float(std_d),
                "ground_dx": gdx,
                "ground_dy": gdy,
            })

            if terminated or truncated:
                break

    # 汇总判定
    steps_run = k + 1
    shift_ratio = global_shift_events / max(1, steps_run)
    tread_ratio = treadmill_like_events / max(1, steps_run)
    origin_ratio = pelvis_near_origin_steps / max(1, steps_run)

    pelvis_disp = body_xy(sim, pelvis_id) - pelvis_xy0
    pelvis_disp_norm = float(np.linalg.norm(pelvis_disp))

    v_fwd_arr = np.asarray([x for x in v_fwd_list if x is not None], dtype=float)
    v_mean = float(v_fwd_arr.mean()) if v_fwd_arr.size else None

    # 结论：优先判断“全局回中”
    if shift_ratio >= 0.02:
        verdict = "检测到【全局回中/搬运世界(recenter)】（多body同一步几乎一致平移）"
    elif tread_ratio >= 0.02 and origin_ratio >= 0.5:
        verdict = "更像【跑步机式(treadmill)】（地面在动/参考系相对运动，pelvis常在原点附近）"
    else:
        verdict = "更像【正常世界系】（未见显著全局平移事件；地面也未呈现跑步机特征）"

    print("==================================================")
    print(f"[diag] env_id: {args.env_id}")
    print(f"[diag] steps_run: {steps_run}, dt: {dt}")
    print(f"[diag] pelvis_id: {pelvis_id}, ground_id: {ground_id}")
    print(f"[diag] global_shift_events: {global_shift_events}, ratio={shift_ratio:.4f}")
    print(f"[diag] treadmill_like_events: {treadmill_like_events}, ratio={tread_ratio:.4f}")
    print(f"[diag] pelvis_near_origin_ratio: {origin_ratio:.4f} (eps={args.pelvis_origin_eps} m)")
    print(f"[diag] pelvis_disp_norm(end-start): {pelvis_disp_norm:.6f} m")
    print(f"[diag] dist_integrated(from v_fwd): {dist_int:.6f} m, v_fwd_mean={v_mean}")
    print(f"[diag] pelvis_xy_range: x[{pelvis_xy_min[0]:.4f},{pelvis_xy_max[0]:.4f}] "
          f"y[{pelvis_xy_min[1]:.4f},{pelvis_xy_max[1]:.4f}]")
    print(f"[diag] verdict: {verdict}")
    print(f"[diag] csv: {csv_path}")
    print("==================================================")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
