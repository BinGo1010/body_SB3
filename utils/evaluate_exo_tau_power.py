# -*- coding: utf-8 -*-
"""
eval_human_exo_sensors_video.py

在同一环境中同时加载 human 与 exo 的 SB3 策略进行推理，
并读取 MuJoCo sensor：EXO 左右髋关节角度/角速度/执行器力矩，输出时域曲线 + CSV + 可选视频。

默认路径已按你提供的信息填好，尽量做到开箱即用。
你只需要保证：
1) 你的 myolegs.xml 里确实包含这些 sensor 与 actuator 名称；
2) 环境 step() 支持拼接动作（human 在前、exo 在末尾）或 Dict 动作。

依赖：
  pip install stable-baselines3 matplotlib pyyaml imageio
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, Optional, Tuple

# --- SB3 ---
from stable_baselines3 import PPO
try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except Exception:
    DummyVecEnv, VecNormalize = None, None
import sys
from pathlib import Path
def _ensure_module_importable(module_name: str, search_roots=None):
    try:
        __import__(module_name)
        print(f"[OK] import {module_name}")
        return True
    except ModuleNotFoundError:
        pass

    if search_roots is None:
        search_roots = [
            "/home/lvchen/body_SB3",
            "/home/lvchen/body_exo_SB3",
            "/home/lvchen/body_SB3-main",
            "/home/lvchen",
        ]

    candidates = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob(f"{module_name}.py"):
            candidates.append(p)
            break
        if candidates:
            break
        for p in root.rglob(str(Path(module_name) / "__init__.py")):
            candidates.append(p)
            break
        if candidates:
            break

    if not candidates:
        # 如果你确定模型不需要这个模块，可以把这行注释掉
        raise ModuleNotFoundError(
            f"找不到 {module_name}.py 或 {module_name}/__init__.py。\n"
            f"若你的 PPO.load 报错提示缺它，请把该文件所在目录加入 search_roots。"
        )

    found = candidates[0]
    add_path = found.parent if found.name.endswith(".py") else found.parent.parent
    sys.path.insert(0, str(add_path))
    __import__(module_name)
    print(f"[Path] add {add_path} -> import {module_name} success")
    return True


# 如果你之前就是这个报错，则保留；若不需要可注释
_ensure_module_importable("sb3_lattice_policy")

# =========================================================
# 通用：动态导入自定义环境（可选）
# =========================================================
def import_from_path(py_path: str, class_name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("custom_env_module", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise AttributeError(f"模块 {py_path} 中找不到类 {class_name}")
    return getattr(module, class_name)


# =========================================================
# MuJoCo sensor 读取：兼容 mujoco / mujoco_py / myosuite 包装
# =========================================================
def _get_sim(env) -> Any:
    # 常见：env.sim / env.unwrapped.sim
    if hasattr(env, "sim"):
        return env.sim
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "sim"):
        return env.unwrapped.sim
    return None


def get_sensor_by_name(env, sensor_name: str) -> float:
    """
    尽可能用“按名字读取”的方式拿到 sensor 标量。
    支持：
      - mujoco (python bindings): data.sensor(name).data[0]
      - mujoco_py: sim.data.get_sensor(name)
      - fallback: sensordata + sensor_name2id + adr/dim
    """
    sim = _get_sim(env)
    if sim is None:
        raise RuntimeError("env 中找不到 sim（无法读取 MuJoCo 传感器）")

    # 1) mujoco (官方) 常见接口
    try:
        # 有些包装：sim.data.sensor("name").data
        val = sim.data.sensor(sensor_name).data
        return float(np.array(val).ravel()[0])
    except Exception:
        pass

    # 2) mujoco_py
    try:
        val = sim.data.get_sensor(sensor_name)
        return float(np.array(val).ravel()[0])
    except Exception:
        pass

    # 3) fallback：通过 sensordata 地址映射
    try:
        model = sim.model
        data = sim.data
        # mujoco_py: model.sensor_name2id
        if hasattr(model, "sensor_name2id"):
            sid = model.sensor_name2id(sensor_name)
            adr = model.sensor_adr[sid]
            dim = model.sensor_dim[sid]
            return float(np.array(data.sensordata[adr:adr+dim]).ravel()[0])
    except Exception:
        pass

    raise KeyError(f"读取 sensor 失败: {sensor_name}（请确认 XML 中已定义且名称一致）")


def _as_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """将图像转换为 uint8 RGB 格式"""
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


def _find_render_candidates(env, sim=None):
    """在 env/sim 结构里找所有可 render() 的对象"""
    cand = []
    def add(x):
        if x is None or hasattr(x, "action_space"):
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

    uniq, seen = [], set()
    for x in cand:
        i = id(x)
        if i not in seen:
            uniq.append(x)
            seen.add(i)
    return uniq


def get_frame(env, width=640, height=480, camera_id: Optional[int]=None) -> np.ndarray:
    """
    返回 RGB frame (H,W,3) uint8。
    参考 evaluate_model _full.py 的健壮实现
    """
    sim = _get_sim(env)
    candidates = _find_render_candidates(env, sim=sim)

    # 优先尝试 physics.render(height=, width=, camera_id=)
    for obj in candidates:
        # dm_control 风格：render(height=, width=, camera_id=)
        try:
            if camera_id is not None:
                img = obj.render(height=height, width=width, camera_id=int(camera_id))
            else:
                img = obj.render(height=height, width=width)
            img = _as_uint8_rgb(np.asarray(img))
            if img.ndim == 3:
                return img
        except (TypeError, AttributeError):
            pass
        except Exception:
            pass

        # 位置参数形式
        try:
            if camera_id is not None:
                img = obj.render(height, width, camera_id=int(camera_id))
            else:
                img = obj.render(height, width)
            img = _as_uint8_rgb(np.asarray(img))
            if img.ndim == 3:
                return img
        except (TypeError, AttributeError):
            pass
        except Exception:
            pass

        # 兜底
        try:
            img = obj.render()
            if img is not None:
                img = _as_uint8_rgb(np.asarray(img))
                if img.ndim == 3:
                    return img
        except Exception:
            pass

    # 最后兜底：env.render
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

    raise RuntimeError("Unable to render frame from environment or sim")


def get_dt(env) -> float:
    # 常见：env.dt / env.model.opt.timestep*frame_skip / sim.model.opt.timestep*frame_skip
    if hasattr(env, "dt"):
        try:
            return float(env.dt)
        except Exception:
            pass
    sim = _get_sim(env)
    if sim is not None and hasattr(sim, "model") and hasattr(sim.model, "opt"):
        try:
            base = float(sim.model.opt.timestep)
            fs = int(getattr(env, "frame_skip", 1))
            return base * fs
        except Exception:
            pass
    return 0.01  # 保底


# =========================================================
# 动作合并：兼容 Box 拼接 或 Dict
# =========================================================
def merge_actions(env, a_h, a_e, exo_at_end: bool = True):
    """
    - 若 env.action_space 是 Dict：优先 {"human": a_h, "exo": a_e}
    - 否则默认 Box：拼接向量。exo_at_end=True => [a_h, a_e]
    """
    space = getattr(env, "action_space", None)

    # Dict 动作空间
    try:
        import gymnasium as gymn
        DictSpace = gymn.spaces.Dict
    except Exception:
        DictSpace = None

    try:
        import gym
        DictSpace2 = gym.spaces.Dict
    except Exception:
        DictSpace2 = None

    if space is not None and ((DictSpace is not None and isinstance(space, DictSpace)) or
                              (DictSpace2 is not None and isinstance(space, DictSpace2))):
        return {"human": a_h, "exo": a_e}

    # 默认 Box：拼接
    a_h = np.array(a_h).ravel()
    a_e = np.array(a_e).ravel()
    if exo_at_end:
        return np.concatenate([a_h, a_e], axis=0)
    else:
        return np.concatenate([a_e, a_h], axis=0)


def split_obs_for_policies(obs):
    """
    - 若 obs 是 dict：尝试取 obs["human"], obs["exo"]
    - 否则：两个策略都用同一 obs
    """
    if isinstance(obs, dict):
        oh = obs.get("human", obs)
        oe = obs.get("exo", obs)
        return oh, oe
    return obs, obs


# =========================================================
# 环境创建（两种路径）：
#  A) 用你项目里的 custom env py + class
#  B) 直接 gym.make（如果你已注册 entrypoint）
# =========================================================
def make_env(args):
    if args.custom_module and args.custom_class:
        EnvCls = import_from_path(args.custom_module, args.custom_class)
        # 这里按你的配置参数最小化传入；若你的 Env 构造函数参数名不同，请改这里
        env = EnvCls(
            model_path=args.model_xml,
            reset_type=args.reset_type,
            target_y_vel=args.target_y_vel,
            hip_period=args.hip_period,
        )
        return env

    # fallback: gym.make
    try:
        import gym
    except Exception as e:
        raise ImportError("未提供 custom env 且无法 import gym。") from e

    if args.env_id is None:
        raise ValueError("未提供 custom env 时，需要 --env_id（gym 注册的环境ID）")

    env = gym.make(
        args.env_id,
        model_xml=args.model_xml,
        reset_type=args.reset_type,
        target_y_vel=args.target_y_vel,
        hip_period=args.hip_period,
    )
    return env


# =========================================================
# 主流程
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    # --- 模型/环境默认路径（按你提供的） ---
    parser.add_argument("--human_model", type=str, default="/home/lvchen/body_SB3/logs/eval/best_human_c4_model")
    parser.add_argument("--exo_model",   type=str, default="/home/lvchen/body_SB3/logs/eval/best_exo_c4_model")
    parser.add_argument("--model_xml",   type=str, default="/home/lvchen/miniconda3/envs/myosuite/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml")

    # 环境参数
    parser.add_argument("--reset_type",  type=str, default="lyly")
    parser.add_argument("--target_y_vel", type=float, default=1.6)
    parser.add_argument("--hip_period",   type=int, default=62)

    # 环境创建方式（优先 custom）
    parser.add_argument("--custom_module", type=str, default="/home/lvchen/body_SB3/envs/walk_gait_exo_joint.py", help="例如 ")
    parser.add_argument("--custom_class",  type=str, default="WalkEnvV4Multi", help="例如 ")
    parser.add_argument("--env_id",        type=str, default="", help="若使用 gym.make，填写已注册 env id")

    # VecNormalize（可选：若训练时用了，需要传入；否则留空）
    parser.add_argument("--vecnorm", type=str, default="", help="VecNormalize.pkl 的路径（只能加载一份）")

    # 输出
    parser.add_argument("--out_dir", type=str, default="/home/lvchen/body_SB3/utils/exo_human")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--deterministic", action="store_true", default=True)

    # 动作拼接规则
    parser.add_argument("--exo_at_end", action="store_true", default=True,
                        help="Box 动作空间时，是否按 [human, exo] 拼接（默认 True）")

    # 视频
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------
    # 1) 创建 env（注意：SB3 模型通常期望 VecEnv）
    # -------------------------
    env_raw = make_env(args)

    # 包一层 DummyVecEnv 以兼容 SB3（predict 不强制，但 load 时传 env 更稳）
    if DummyVecEnv is not None:
        env = DummyVecEnv([lambda: env_raw])
    else:
        env = env_raw  # 退化

    # VecNormalize（只能加载一份）
    if args.vecnorm:
        if VecNormalize is None:
            raise RuntimeError("当前 stable-baselines3 环境缺少 VecNormalize，无法加载 --vecnorm")
        env = VecNormalize.load(args.vecnorm, env)
        env.training = False
        env.norm_reward = False

    # -------------------------
    # 2) 加载两套策略
    # -------------------------
    human_policy = PPO.load(args.human_model, device="cpu")
    exo_policy   = PPO.load(args.exo_model,   device="cpu")

    # 打印策略的动作空间维度
    # print(f"[DEBUG] human_policy.action_space: {human_policy.action_space}")
    # print(f"[DEBUG] exo_policy.action_space: {exo_policy.action_space}")
    # if hasattr(env_raw, "action_space"):
    #     print(f"[DEBUG] env.action_space: {env_raw.action_space}")

    # -------------------------
    # 3) rollout + 记录
    # -------------------------
    # reset
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    dt = get_dt(env_raw)

    # 尝试第一次渲染以诊断是否支持video生成
    if args.save_video:
        try:
            test_frame = get_frame(env_raw, width=args.width, height=args.height, camera_id=args.camera_id)
            print(f"[OK] 可以成功渲染，frame shape: {test_frame.shape}")
        except Exception as e:
            print(f"[WARN] 无法渲染第一帧: {e}")
            print(f"       视频功能可能不可用")

    # 需要记录的传感器名（按你的 XML）
    SENSOR_NAMES = {
        "rr_pos": "rr_joint_pos",
        "rr_vel": "rr_joint_vel",
        "lr_pos": "lr_joint_pos",
        "lr_vel": "lr_joint_vel",
        "rr_tau": "rr_act_torque",
        "lr_tau": "lr_act_torque",
    }

    t_list = []
    logs = {k: [] for k in SENSOR_NAMES.keys()}
    # CPG 目标（若环境在 info 中暴露 debug/q_des_* 等键，则会记录；否则为 NaN）
    logs.update({
        "rr_q_des": [], "lr_q_des": [],
        "rr_dq_des": [], "lr_dq_des": [],
        # CPG 参数（来自 env.info 的 debug/cpg_*，需要你在 walk_gait_exo_fixed.py 中写入）
        "cpg_Omega_target": [],
        "cpg_amp_target_L": [], "cpg_amp_target_R": [],
        "cpg_off_target_L": [], "cpg_off_target_R": [],
    })
    rew_list = []
    done_list = []

    frames = []

    for step in range(args.max_steps):
        # obs 可能是 VecEnv 的 batch（shape: [n_env, ...]）
        # 取第 0 个 env 的 obs 用于策略推理（同时兼容 dict）
            # ==========================
            # (A) 从 Dict 观测中拆分给两套策略
            # ==========================
        if isinstance(obs, dict):
            oh = obs["human"]  # shape: (n_env, 476) 或 (476,)
            oe = obs["exo"]    # shape: (n_env, 16)  或 (16,)
        else:
            oh = obs
            oe = obs

        # VecEnv 情况下取第 0 个环境
        oh0 = oh[0] if isinstance(oh, np.ndarray) and oh.ndim >= 2 else oh
        oe0 = oe[0] if isinstance(oe, np.ndarray) and oe.ndim >= 2 else oe

        # ==========================
        # (B) 两个策略分别推理
        # ==========================
        a_h, _ = human_policy.predict(oh0, deterministic=args.deterministic)
        a_e, _ = exo_policy.predict(oe0, deterministic=args.deterministic)

        # 调试：打印第一步的动作维度
        if step == 0:
            print(f"[DEBUG Step 0] a_h shape: {np.array(a_h).shape}, a_e shape: {np.array(a_e).shape}")
            print(f"[DEBUG Step 0] a_h: {a_h}, a_e: {a_e}")

        # ==========================
        # (C) 动作合并并 step
        #     这里假设 env.action_space 是 Box 拼接：[human, exo]
        # ==========================
        action = np.concatenate([np.array(a_h).ravel(), np.array(a_e).ravel()], axis=0)
        
        # 调试：打印拼接后的动作维度
        if step == 0:
            print(f"[DEBUG Step 0] Concatenated action shape: {action.shape}, action: {action}")
        
        # DummyVecEnv 需要 (num_envs, action_dim) 的形状，需要添加批次维度
        if isinstance(env, DummyVecEnv):
            action = action.reshape(1, -1)
        
        obs, reward, done, info = env.step(action)

        # 读取 env info 里的 CPG 目标（若你的 WalkEnvV4Multi 在 info 中写入了 debug/*）
        info0 = None
        try:
            if isinstance(info, (list, tuple)) and len(info) > 0:
                info0 = info[0]
            elif isinstance(info, dict):
                info0 = info
        except Exception:
            info0 = None

        def _pick_first(keys, default=float('nan')):
            if not isinstance(info0, dict):
                return default
            for kk in keys:
                if kk in info0:
                    try:
                        return float(np.asarray(info0[kk]).ravel()[0])
                    except Exception:
                        return default
            return default

        rr_q_des = _pick_first(["debug/q_des_R", "debug/q_des_rr", "q_des_R", "qdes_R"])
        lr_q_des = _pick_first(["debug/q_des_L", "debug/q_des_lr", "q_des_L", "qdes_L"])
        rr_dq_des = _pick_first(["debug/dq_des_R", "debug/dq_des_rr", "dq_des_R", "dqdes_R"])
        lr_dq_des = _pick_first(["debug/dq_des_L", "debug/dq_des_lr", "dq_des_L", "dqdes_L"])
        logs["rr_q_des"].append(rr_q_des)
        logs["lr_q_des"].append(lr_q_des)
        logs["rr_dq_des"].append(rr_dq_des)
        logs["lr_dq_des"].append(lr_dq_des)

        # 读取 CPG 物理目标参数（5 个量）
        # - Omega_target: rad/s（你也可以在画图时除以 2π 转为 Hz）
        logs["cpg_Omega_target"].append(_pick_first(["debug/cpg_Omega_target", "debug/cpg_omega", "cpg_Omega_target"]))
        # - amp_target/off_target: rad
        logs["cpg_amp_target_L"].append(_pick_first(["debug/cpg_amp_target_L", "debug/cpg_amp_L", "cpg_amp_target_L"]))
        logs["cpg_amp_target_R"].append(_pick_first(["debug/cpg_amp_target_R", "debug/cpg_amp_R", "cpg_amp_target_R"]))
        logs["cpg_off_target_L"].append(_pick_first(["debug/cpg_off_target_L", "debug/cpg_off_L", "cpg_off_target_L"]))
        logs["cpg_off_target_R"].append(_pick_first(["debug/cpg_off_target_R", "debug/cpg_off_R", "cpg_off_target_R"]))

        # VecEnv 返回 reward/done 往往是 array
        r0 = float(np.array(reward).ravel()[0])
        d0 = bool(np.array(done).ravel()[0])

        # ==========================
        # (D) 这里开始做你的传感器记录、渲染、写csv等
        # ==========================
        # 记录传感器
        t = step * dt
        t_list.append(t)
        for k, sname in SENSOR_NAMES.items():
            try:
                val = get_sensor_by_name(env_raw, sname)
            except Exception as e:
                raise RuntimeError(f"step={step} 读取 sensor={sname} 失败：{e}")
            logs[k].append(val)

        rew_list.append(r0)
        done_list.append(d0)

        # 记录视频帧
        if args.save_video:
            try:
                frame = get_frame(env_raw, width=args.width, height=args.height, camera_id=args.camera_id)
                frames.append(frame)
            except Exception as e:
                if step == 0:
                    print(f"[WARN] 第一帧渲染失败: {e}，可能无法生成视频")
                pass

        if d0:
            break


    # -------------------------
    # 3.5) 计算机械功率与正/负功（评估“助力/刹车”）
    #   机械功率：P = tau * qdot  (W)
    #   正功：P>0 ；负功：P<0（这里用其幅值 -P 记为正数，便于统计）
    # -------------------------
    rr_tau_arr = np.asarray(logs["rr_tau"], dtype=np.float64)
    lr_tau_arr = np.asarray(logs["lr_tau"], dtype=np.float64)
    rr_vel_arr = np.asarray(logs["rr_vel"], dtype=np.float64)
    lr_vel_arr = np.asarray(logs["lr_vel"], dtype=np.float64)

    rr_power_arr = rr_tau_arr * rr_vel_arr
    lr_power_arr = lr_tau_arr * lr_vel_arr

    rr_power_pos = np.maximum(rr_power_arr, 0.0)
    rr_power_neg = np.maximum(-rr_power_arr, 0.0)  # 负功幅值
    lr_power_pos = np.maximum(lr_power_arr, 0.0)
    lr_power_neg = np.maximum(-lr_power_arr, 0.0)

    rr_work_pos_cum = np.cumsum(rr_power_pos) * dt
    rr_work_neg_cum = np.cumsum(rr_power_neg) * dt
    lr_work_pos_cum = np.cumsum(lr_power_pos) * dt
    lr_work_neg_cum = np.cumsum(lr_power_neg) * dt

    # 写回 logs 方便 CSV/画图
    logs["rr_power"] = rr_power_arr.tolist()
    logs["lr_power"] = lr_power_arr.tolist()
    logs["rr_work_pos_cum"] = rr_work_pos_cum.tolist()
    logs["rr_work_neg_cum"] = rr_work_neg_cum.tolist()
    logs["lr_work_pos_cum"] = lr_work_pos_cum.tolist()
    logs["lr_work_neg_cum"] = lr_work_neg_cum.tolist()

    def _safe_last(arr):
        return float(arr[-1]) if arr is not None and len(arr) > 0 else 0.0

    rr_wpos = _safe_last(rr_work_pos_cum)
    rr_wneg = _safe_last(rr_work_neg_cum)
    lr_wpos = _safe_last(lr_work_pos_cum)
    lr_wneg = _safe_last(lr_work_neg_cum)

    total_pos = rr_wpos + lr_wpos
    total_neg = rr_wneg + lr_wneg
    frac_pos = total_pos / (total_pos + total_neg + 1e-9)

    print("[Power/Work] Summary (J):")
    print(f"  Right: +{rr_wpos:.3f}  -{rr_wneg:.3f} (abs)")
    print(f"  Left : +{lr_wpos:.3f}  -{lr_wneg:.3f} (abs)")
    print(f"  Total: +{total_pos:.3f}  -{total_neg:.3f} (abs)  frac_pos={frac_pos:.3f}")

    # -------------------------
    # 4) 保存 CSV / 元数据
    # -------------------------
    csv_path = os.path.join(args.out_dir, "exo_sensors_timeseries.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = [
            "t",
            "rr_pos", "rr_vel", "lr_pos", "lr_vel",
            "rr_tau", "lr_tau",
            "rr_q_des", "lr_q_des", "rr_dq_des", "lr_dq_des",
            # CPG 参数（5 个量）
            "cpg_Omega_target", "cpg_amp_target_L", "cpg_amp_target_R", "cpg_off_target_L", "cpg_off_target_R",
            "rr_power", "lr_power",
            "rr_work_pos_cum", "rr_work_neg_cum", "lr_work_pos_cum", "lr_work_neg_cum",
            "reward", "done",
        ]
        f.write(",".join(header) + "\n")
        for i in range(len(t_list)):
            row = [
                t_list[i],
                logs["rr_pos"][i], logs["rr_vel"][i],
                logs["lr_pos"][i], logs["lr_vel"][i],
                logs["rr_tau"][i], logs["lr_tau"][i],
                logs["rr_q_des"][i], logs["lr_q_des"][i],
                logs["rr_dq_des"][i], logs["lr_dq_des"][i],
                logs["cpg_Omega_target"][i],
                logs["cpg_amp_target_L"][i], logs["cpg_amp_target_R"][i],
                logs["cpg_off_target_L"][i], logs["cpg_off_target_R"][i],
                logs["rr_power"][i], logs["lr_power"][i],
                logs["rr_work_pos_cum"][i], logs["rr_work_neg_cum"][i],
                logs["lr_work_pos_cum"][i], logs["lr_work_neg_cum"][i],
                rew_list[i], int(done_list[i]),
            ]
            f.write(",".join([f"{x:.8f}" if isinstance(x, float) else str(x) for x in row]) + "\n")

    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "human_model": args.human_model,
            "exo_model": args.exo_model,
            "model_xml": args.model_xml,
            "reset_type": args.reset_type,
            "target_y_vel": args.target_y_vel,
            "hip_period": args.hip_period,
            "dt": dt,
            "steps": len(t_list),
            "vecnorm": args.vecnorm,
        }, f, ensure_ascii=False, indent=2)

    # -------------------------
    # 5) 画图：角度/角速度/力矩（左右对比）
    # -------------------------
    t_arr = np.array(t_list)

    def _plot_two(yL, yR, title, ylab, out_png):
        plt.figure()
        plt.plot(t_arr, yR, label="Right (rr)")
        plt.plot(t_arr, yL, label="Left  (lr)")
        plt.xlabel("Time [s]")
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    _plot_two(
        yL=np.array(logs["lr_pos"]), yR=np.array(logs["rr_pos"]),
        title="EXO Hip Joint Position vs Time",
        ylab="Joint position [rad]",
        out_png=os.path.join(args.out_dir, "exo_jointpos_timeseries.png"),
    )

    # -------------------------
    # 5.0) CPG 目标角度/角速度 vs 实际（用于判断相位是否对齐，是否在“刹车”）
    # -------------------------
    rr_q_des_arr = np.asarray(logs.get("rr_q_des", []), dtype=np.float64)
    lr_q_des_arr = np.asarray(logs.get("lr_q_des", []), dtype=np.float64)
    rr_dq_des_arr = np.asarray(logs.get("rr_dq_des", []), dtype=np.float64)
    lr_dq_des_arr = np.asarray(logs.get("lr_dq_des", []), dtype=np.float64)

    # 若 info 中没有提供目标（全 NaN），仍会输出图，但会打印提示
    if rr_q_des_arr.size == 0 or np.all(~np.isfinite(rr_q_des_arr)) and np.all(~np.isfinite(lr_q_des_arr)):
        print("[WARN] 没有在 info 中读到 CPG 目标角度（rr_q_des/lr_q_des 均为 NaN）。"
              "请检查环境 step() 是否把 self._debug_cache 写入 info。")

    # 角度：实际 vs 目标
    plt.figure(figsize=(10, 6))
    plt.plot(t_arr, np.asarray(logs["rr_pos"], dtype=np.float64), label="Right q")
    plt.plot(t_arr, rr_q_des_arr, "--", label="Right q_des (CPG)")
    plt.plot(t_arr, np.asarray(logs["lr_pos"], dtype=np.float64), label="Left q")
    plt.plot(t_arr, lr_q_des_arr, "--", label="Left q_des (CPG)")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint position [rad]")
    plt.title("EXO Joint Position: Actual vs CPG Target")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "cpg_target.png"), dpi=200)
    plt.close()

    # 角速度：实际 vs 目标（可选，额外输出）
    plt.figure(figsize=(10, 6))
    plt.plot(t_arr, np.asarray(logs["rr_vel"], dtype=np.float64), label="Right dq")
    plt.plot(t_arr, rr_dq_des_arr, "--", label="Right dq_des (CPG)")
    plt.plot(t_arr, np.asarray(logs["lr_vel"], dtype=np.float64), label="Left dq")
    plt.plot(t_arr, lr_dq_des_arr, "--", label="Left dq_des (CPG)")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint velocity [rad/s]")
    plt.title("EXO Joint Velocity: Actual vs CPG Target")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "cpg_target_vel.png"), dpi=200)
    plt.close()
    _plot_two(
        yL=np.array(logs["lr_vel"]), yR=np.array(logs["rr_vel"]),
        title="EXO Hip Joint Velocity vs Time",
        ylab="Joint velocity [rad/s]",
        out_png=os.path.join(args.out_dir, "exo_jointvel_timeseries.png"),
    )
    _plot_two(
        yL=np.array(logs["lr_tau"]), yR=np.array(logs["rr_tau"]),
        title="EXO Actuator Torque vs Time",
        ylab="Actuator torque [N·m]",
        out_png=os.path.join(args.out_dir, "exo_acttorque_timeseries.png"),
    )

    # -------------------------
    # 5.1) 机械功率/正负功评估图
    # -------------------------
    _plot_two(
        yL=np.array(logs["lr_power"]), yR=np.array(logs["rr_power"]),
        title="EXO Mechanical Power (tau*qdot) vs Time",
        ylab="Power [W]",
        out_png=os.path.join(args.out_dir, "exo_power_timeseries.png"),
    )

    # 累积正功/负功（负功取绝对值）
    plt.figure()
    plt.plot(t_arr, np.array(logs["rr_work_pos_cum"]), label="Right +Work")
    plt.plot(t_arr, np.array(logs["rr_work_neg_cum"]), label="Right -Work(abs)")
    plt.plot(t_arr, np.array(logs["lr_work_pos_cum"]), label="Left  +Work")
    plt.plot(t_arr, np.array(logs["lr_work_neg_cum"]), label="Left  -Work(abs)")
    plt.xlabel("Time [s]")
    plt.ylabel("Cumulative work [J]")
    plt.title("EXO Cumulative Positive/Negative Work")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "exo_work_cumulative.png"), dpi=200)
    plt.close()

    # 总正功/负功条形图（负功取绝对值）
    rr_wpos = float(np.array(logs["rr_work_pos_cum"])[-1]) if len(logs["rr_work_pos_cum"]) > 0 else 0.0
    rr_wneg = float(np.array(logs["rr_work_neg_cum"])[-1]) if len(logs["rr_work_neg_cum"]) > 0 else 0.0
    lr_wpos = float(np.array(logs["lr_work_pos_cum"])[-1]) if len(logs["lr_work_pos_cum"]) > 0 else 0.0
    lr_wneg = float(np.array(logs["lr_work_neg_cum"])[-1]) if len(logs["lr_work_neg_cum"]) > 0 else 0.0

    labels = ["Right", "Left"]
    pos = [rr_wpos, lr_wpos]
    neg = [rr_wneg, lr_wneg]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, pos, width, label="+Work")
    plt.bar(x + width/2, neg, width, label="-Work(abs)")
    plt.xticks(x, labels)
    plt.ylabel("Work [J]")
    plt.title("EXO Total Positive/Negative Work")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "exo_work_bar.png"), dpi=200)
    plt.close()

    # -------------------------
    # 5.2) CPG 参数（Omega/amp/offset）随时间变化
    #      便于排查“为什么策略总往负 offset 偏”、以及频率/幅值是否乱跳
    # -------------------------
    cpg_Omega = np.asarray(logs.get("cpg_Omega_target", []), dtype=np.float64)
    cpg_amp_L = np.asarray(logs.get("cpg_amp_target_L", []), dtype=np.float64)
    cpg_amp_R = np.asarray(logs.get("cpg_amp_target_R", []), dtype=np.float64)
    cpg_off_L = np.asarray(logs.get("cpg_off_target_L", []), dtype=np.float64)
    cpg_off_R = np.asarray(logs.get("cpg_off_target_R", []), dtype=np.float64)

    if cpg_Omega.size == 0 or np.all(~np.isfinite(cpg_Omega)):
        print("[WARN] 没有在 info 中读到 CPG 参数 debug/cpg_*（全 NaN）。"
              "请确认 walk_gait_exo_fixed.py 已写入 self._debug_cache['debug/cpg_Omega_target' 等]。")

    # Omega (rad/s) -> f (Hz)
    cpg_f_hz = cpg_Omega / (2.0 * np.pi)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t_arr, cpg_f_hz, label="f_target (Hz)")
    axes[0].set_ylabel("Frequency [Hz]")
    axes[0].set_title("CPG Targets vs Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t_arr, cpg_amp_R, label="amp_R")
    axes[1].plot(t_arr, cpg_amp_L, label="amp_L")
    axes[1].set_ylabel("Amplitude [rad]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t_arr, cpg_off_R, label="offset_R")
    axes[2].plot(t_arr, cpg_off_L, label="offset_L")
    axes[2].set_ylabel("Offset [rad]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "cpg_params.png"), dpi=200)
    plt.close()

    # -------------------------
    # 6) 保存视频（可选，使用 skvideo.io）
    # -------------------------
    if args.save_video:
        print(f"[INFO] 收集到 {len(frames)} 帧用于视频保存")
        if len(frames) > 0:
            video_path = os.path.join(args.out_dir, "rollout.mp4")
            try:
                import skvideo.io
                frames_array = np.asarray(frames, dtype=np.uint8)
                skvideo.io.vwrite(
                    video_path, 
                    frames_array, 
                    outputdict={"-r": str(int(args.video_fps))}
                )
                print(f"[OK] 视频已保存: {video_path}")
            except ImportError:
                print("[WARN] skvideo 未安装，尝试使用 imageio 作为备选...")
                try:
                    import imageio
                    imageio.mimwrite(video_path, frames, fps=args.video_fps)
                    print(f"[OK] 视频已保存（使用 imageio）: {video_path}")
                except Exception as e:
                    print(f"[WARN] imageio 保存也失败: {e}")
            except Exception as e:
                print(f"[WARN] 保存视频失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("[WARN] 没有收集到任何视频帧，可能是环境不支持渲染")

    print("Done.")
    print(f"  CSV : {csv_path}")
    print(f"  Png : {os.path.join(args.out_dir, 'exo_jointpos_timeseries.png')}")
    print(f"        {os.path.join(args.out_dir, 'exo_jointvel_timeseries.png')}")
    print(f"        {os.path.join(args.out_dir, 'exo_acttorque_timeseries.png')}")
    print(f"        {os.path.join(args.out_dir, 'cpg_target.png')}")
    print(f"        {os.path.join(args.out_dir, 'cpg_target_vel.png')}")
    print(f"        {os.path.join(args.out_dir, 'cpg_params.png')}")
    if args.save_video:
        print(f"  MP4 : {os.path.join(args.out_dir, 'rollout.mp4')}")
    print(f"  Meta: {meta_path}")


if __name__ == "__main__":
    main()