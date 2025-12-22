#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用滑块实时调节“相位 phase(0~1)”并在 MuJoCo viewer 中刷新姿态。

用法示例：
  (myosuite) python utils/view_phase_slider.py --xml /path/to/myolegs.xml --coeff_csv cpg/fourier_coefficients_order6_132.csv

说明：
- 这里的 phase 是归一化相位 ∈ [0,1)；
- 实际写入 qpos 时调用 CMUParams.eval_joint_phase(name, phase)，因此你在 params_12_bias 里实现的全局 phase_bias 会自动生效；
- 本脚本只做 mj_forward（几何前向），不做动力学积分，不需要重力。
"""
import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer as mjviewer
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 添加项目根路径，确保可导入 cpg.params_12_bias
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
            "--coeff_csv",
            type=str,
            default=os.path.join(PROJECT_ROOT, "cpg", "fourier_coefficients_order6_132.csv"),
            help="12DOF 傅里叶系数 CSV"
        )
    p.add_argument(
        "--xml",
        type=str,
        default="/home/lvchen/miniconda3/envs/myosuite_raw/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml",
        help="MuJoCo XML 路径"
    )
    p.add_argument("--gait_T", type=float, default=3.0, help="步态周期(s)，用于初始化 CMUParams")
    p.add_argument("--phase0", type=float, default=0.0, help="滑块初值相位(0~1)")
    p.add_argument("--hz", type=float, default=60.0, help="viewer 刷新频率(Hz)")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.coeff_csv).is_file():
        raise FileNotFoundError(f"找不到 coeff_csv: {args.coeff_csv}")
    if not Path(args.xml).is_file():
        raise FileNotFoundError(f"找不到 xml: {args.xml}")

    params = CMUParams(csv_path=args.coeff_csv, T=float(args.gait_T))

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # joint name -> qpos adr
    qpos_adr = {}
    for name in JOINT_NAMES:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id < 0:
            raise RuntimeError(f"关节名 {name} 在模型中找不到，请检查 XML 与 JOINT_NAMES。")
        qpos_adr[name] = int(model.jnt_qposadr[j_id])

    # ====== Matplotlib Slider UI ======
    plt.ion()
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    plt.subplots_adjust(left=0.10, right=0.98, bottom=0.35, top=0.90)
    ax.set_title("Phase Slider (0~1)")
    ax.set_axis_off()

    ax_phase = plt.axes([0.10, 0.18, 0.78, 0.08])
    s_phase = Slider(ax_phase, "phase", 0.0, 1.0, valinit=float(args.phase0), valstep=0.001)

    ax_btn = plt.axes([0.90, 0.18, 0.08, 0.08])
    btn_zero = Button(ax_btn, "0.0")

    state = {"phase": float(args.phase0), "running": True}

    def on_slider(val):
        state["phase"] = float(val) % 1.0

    def on_zero(_):
        s_phase.set_val(0.0)

    def on_close(_):
        state["running"] = False

    s_phase.on_changed(on_slider)
    btn_zero.on_clicked(on_zero)
    fig.canvas.mpl_connect("close_event", on_close)

    # ====== MuJoCo Viewer ======
    viewer = mjviewer.launch_passive(model, data)

    dt = 1.0 / max(1e-6, float(args.hz))

    try:
        while state["running"] and viewer.is_running():
            phase = state["phase"]

            for name in JOINT_NAMES:
                q_des = float(params.eval_joint_phase(name, phase))  # rad
                data.qpos[qpos_adr[name]] = q_des

            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            viewer.sync()
            plt.pause(0.001)  # 让 slider UI 有机会响应
            time.sleep(dt)

    finally:
        # viewer 由 launch_passive 管理，无需显式 close
        try:
            plt.close(fig)
        except Exception:
            pass


if __name__ == "__main__":
    main()
