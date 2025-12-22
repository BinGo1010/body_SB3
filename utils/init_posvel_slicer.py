#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用滑块调节相位 phase ∈ [0,1]，实时在 MuJoCo viewer 中显示对应姿态（运动学回放：写 qpos + mj_forward）。
- 不需要重力（脚本会将 gravity 置 0）
- 不做动力学积分（不 mj_step），所以非常稳定、可控
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 添加项目根路径，确保可导入 cpg.params_12
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cpg.params_12_bias import CMUParams


JOINT_NAMES = [
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_l", "knee_angle_r",
    "ankle_angle_l", "subtalar_angle_l",
    "ankle_angle_r", "subtalar_angle_r",
]


def fourier_eval_time(t: float, a0: float, a: np.ndarray, b: np.ndarray, omega: float) -> float:
    k = np.arange(1, len(a) + 1, dtype=float)
    theta = omega * t
    return a0 + np.sum(a * np.cos(k * theta) + b * np.sin(k * theta))


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=str, default="/home/lvchen/miniconda3/envs/myosuite_raw/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml")
    p.add_argument("--coeff_csv", type=str, default=os.path.join(PROJECT_ROOT, "cpg", "fourier_coefficients_order6_132.csv"))
    p.add_argument("--gait_T", type=float, default=3.0, help="步态周期(s)")
    p.add_argument("--use_keyframe0", action="store_true", help="若模型有 keyframe，使用 keyframe0 作为初始")
    return p.parse_args()


def maybe_reset_to_keyframe0(model, data, use_keyframe0: bool):
    if use_keyframe0 and getattr(model, "nkey", 0) > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)


def main():
    args = build_args()

    if not os.path.isfile(args.xml):
        raise FileNotFoundError(f"找不到 XML: {args.xml}")
    if not os.path.isfile(args.coeff_csv):
        raise FileNotFoundError(f"找不到 Fourier CSV: {args.coeff_csv}")

    # 读取傅里叶参数
    params = CMUParams(csv_path=args.coeff_csv, T=args.gait_T)

    # 加载模型
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    maybe_reset_to_keyframe0(model, data, args.use_keyframe0)

    # 不需要重力：直接置零
    model.opt.gravity[:] = 0.0

    # joint -> qpos index
    qpos_adr = {}
    for name in JOINT_NAMES:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id < 0:
            raise RuntimeError(f"关节名 {name} 在 XML 里找不到，请检查 JOINT_NAMES 与模型一致。")
        qpos_adr[name] = int(model.jnt_qposadr[j_id])

    # ====== 定义“应用相位 -> 写 qpos -> forward”的函数 ======
    def apply_phase(phase: float):
        phase = float(np.clip(phase, 0.0, 1.0))
        t = phase * float(args.gait_T)

        for name in JOINT_NAMES:
            a0 = getattr(params, f"{name}_a0")
            a = getattr(params, f"{name}_a")
            b = getattr(params, f"{name}_b")
            q = fourier_eval_time(t, a0, a, b, params.omega)

            data.qpos[qpos_adr[name]] = q

        data.qvel[:] = 0.0
        if model.nu > 0:
            data.ctrl[:] = 0.0  # 避免执行器残余控制影响显示
        mujoco.mj_forward(model, data)

    # ====== 启动 viewer（被动模式，循环里 sync）======
    viewer = mujoco.viewer.launch_passive(model, data)

    # ====== Matplotlib UI：相位滑块 + 播放按钮（可选）======
    plt.close("all")
    fig = plt.figure("Phase Slider (0~1)")
    fig.set_size_inches(6.5, 2.4)

    ax_phase = fig.add_axes([0.12, 0.55, 0.78, 0.18])
    s_phase = Slider(ax_phase, "phase", 0.0, 1.0, valinit=0.0, valstep=0.001)

    ax_speed = fig.add_axes([0.12, 0.25, 0.78, 0.18])
    s_speed = Slider(ax_speed, "play Hz", 0.0, 2.5, valinit=1.0, valstep=0.01)  # 自动播放速度（Hz）

    ax_btn = fig.add_axes([0.12, 0.05, 0.18, 0.14])
    btn = Button(ax_btn, "Play/Pause")

    playing = {"flag": False}

    def on_phase_change(val):
        apply_phase(val)

    def on_click(_):
        playing["flag"] = not playing["flag"]

    s_phase.on_changed(on_phase_change)
    btn.on_clicked(on_click)

    # 初始应用一次
    apply_phase(0.0)
    plt.show(block=False)

    # ====== 主循环：同步 viewer，同时让滑块窗口保持响应 ======
    last = time.time()
    try:
        while viewer.is_running():
            now = time.time()
            dt = now - last
            last = now

            if playing["flag"]:
                hz = float(s_speed.val)
                # phase += hz * dt (周期：1.0 对应一整周期)
                new_phase = (float(s_phase.val) + hz * dt) % 1.0
                # set_val 会触发 on_phase_change -> apply_phase
                s_phase.set_val(new_phase)

            viewer.sync()
            plt.pause(0.001)

    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
