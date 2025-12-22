# -*- coding: utf-8 -*-
"""
view_gait_12.py

从 cmu_12dofs_fourier.csv 中读取 12 个下肢关节的傅立叶系数，
使用统一的周期 T=1.0833 s、基频 Omega=5.80 rad/s，
在 MuJoCo 中按时间播放 12 条关节角参考轨迹，并在退出后画图。

依赖：
    - cpg/params_12.py 中的 CMUParams 类
    - cmu_12dofs_fourier.csv（12 个关节的 a0,a,b 系数）
"""

import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. 路径与参数导入
# -------------------------

# 加入项目根路径，确保可导入 cpg.params_12
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cpg.params_12 import CMUParams  # noqa: E402

# CSV 路径（相对项目根目录，你可以按需要修改）
# 注意：不要在第二个参数前加 '/'，否则 os.path.join 会忽略 project_root
CSV_PATH = os.path.join(project_root, "cpg", "fourier_coefficients_order6_132.csv")

# 步态周期与基频（和 CMUParams 保持一致）
GAIT_T = 3           # s
GAIT_OMEGA = 30         # rad/s

# 仿真 XML 路径：请改成你自己的人体模型 / myoLeg 模型路径
XML_PATH = "/home/lvchen/miniconda3/envs/myosuite/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml"
# XML_PATH = "/home/lvchen/miniconda3/envs/myosuite_raw/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml"
# 12 个关节的 MuJoCo 名称（与 params_12 中 alias_map 一致）
JOINT_NAMES = [
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_l", "knee_angle_r",
    "ankle_angle_l", "subtalar_angle_l",
    "ankle_angle_r", "subtalar_angle_r",
]


# -------------------------
# 2. 傅立叶评估函数
# -------------------------

def fourier_eval_time(t: float, a0: float, a: np.ndarray, b: np.ndarray, omega: float) -> float:
    """
    按时间 t 计算傅立叶序列：
        q(t) = a0 + Σ_{k=1..K} [ a_k cos(k * omega * t) + b_k sin(k * omega * t) ]
    所有 12 条曲线共用同一 omega = 2π/T ≈ 5.80 rad/s
    """
    k = np.arange(1, len(a) + 1, dtype=float)
    theta = omega * t
    return a0 + np.sum(a * np.cos(k * theta) + b * np.sin(k * theta))


# -------------------------
# 3. 主函数：驱动 12 关节并记录
# -------------------------

def main():
    # 3.1 读取 12 关节的傅立叶参数
    params = CMUParams(csv_path=CSV_PATH, T=GAIT_T)

    # 3.2 载入 MuJoCo 模型
    if not Path(XML_PATH).is_file():
        raise FileNotFoundError(f"请先将 XML_PATH 改成有效路径，目前为: {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 3.3 建立关节名 -> qpos index 映射（假设每个关节 1 DOF）
    qpos_adr = {}
    for name in JOINT_NAMES:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = model.jnt_qposadr[j_id]
        qpos_adr[name] = adr

    # 3.4 仿真时间设置（这里仅做“运动播放”，不做动力学积分）
    dt = 1.0 / 120.0       # 对齐 CMU 采样率
    total_time = 50 * GAIT_T  # 播放 3 个周期
    n_steps = int(total_time / dt)

    # 记录 12 关节的时间与角度（弧度）
    time_log = np.zeros(n_steps)
    q_log = {name: np.zeros(n_steps) for name in JOINT_NAMES}

    # 3.5 启动 MuJoCo Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        for i in range(n_steps):
            if not viewer.is_running():
                break

            step_start = time.time()
            time_log[i] = t

            # 计算 12 个 DOF 在时间 t 的傅立叶角度，并直接写入 qpos
            for name in JOINT_NAMES:
                a0 = getattr(params, f"{name}_a0")
                a = getattr(params, f"{name}_a")
                b = getattr(params, f"{name}_b")
                q_des = fourier_eval_time(t, a0, a, b, params.omega)

                idx = qpos_adr[name]
                data.qpos[idx] = q_des
                q_log[name][i] = q_des

            # 更新从 qpos 推导的所有量（不做动力学积分，只做几何前向）
            mujoco.mj_forward(model, data)

            # 同步到可视化
            viewer.sync()

            # 控制播放节奏
            t += dt
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # 3.6 退出 viewer 后，画 12 个关节的角度轨迹
    plot_12_joint_gait(time_log, q_log)


# -------------------------
# 4. 画图：12 个关节的轨迹
# -------------------------

def plot_12_joint_gait(time_arr: np.ndarray, q_log: dict):
    """
    将 12 个关节的弧度轨迹画成 12 个子图（单位：deg）
    """
    RAD2DEG = 180.0 / np.pi

    n_joints = len(JOINT_NAMES)
    n_rows = 4
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for idx, name in enumerate(JOINT_NAMES):
        ax = axes[idx]
        q_deg = q_log[name] * RAD2DEG
        ax.plot(time_arr, q_deg, label="ref (Fourier)")
        ax.set_title(name)
        ax.set_ylabel("angle (deg)")
        ax.grid(True)
        ax.legend(fontsize=8)

    for ax in axes[-n_cols:]:
        ax.set_xlabel("time (s)")

    plt.tight_layout()
    plt.show()


# -------------------------
# 5. 入口
# -------------------------

if __name__ == "__main__":
    main()
