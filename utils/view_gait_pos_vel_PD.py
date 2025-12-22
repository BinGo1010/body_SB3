# -*- coding: utf-8 -*-
"""
view_gait_12.py (PD 力控版)

从 CSV 读取 12 个下肢关节的傅立叶系数，生成参考角度 q_ref(t) 与参考角速度 dq_ref(t)，
用 PD 力矩控制（tau = Kp*(q_ref-q) + Kd*(dq_ref-dq)）在 MuJoCo 中进行动力学积分 mj_step，
实现“力控位置+速度”的轨迹加载，并记录 ref vs actual，退出后画图。

依赖：
    - cpg/params_12.py 中的 CMUParams 类
    - fourier_coefficients_order6_132.csv（12 个关节的 a0,a,b 系数）
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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cpg.params_12 import CMUParams  # noqa: E402

CSV_PATH = os.path.join(project_root, "cpg", "fourier_coefficients_order6_132.csv")

# 步态周期（用于 phase = t/T）
GAIT_T = 2.0  # s

# 仿真 XML
# XML_PATH = "/home/lvchen/miniconda3/envs/myosuite/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml"
XML_PATH = "/home/lvchen/miniconda3/envs/myosuite_raw/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml"

# 12 个关节的 MuJoCo 名称（与 params_12 中 alias_map 一致）
JOINT_NAMES = [
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_l", "knee_angle_r",
    "ankle_angle_l", "subtalar_angle_l",
    "ankle_angle_r", "subtalar_angle_r",
]

# -------------------------
# 2. 控制参数（你主要调这几个）
# -------------------------

CTRL_HZ = 120.0                 # 控制频率（外层 PD 刷新频率）
KP = 200.0                      # 位置增益
KD = 3.0                       # 速度增益
TAU_LIMIT = None                # 例如 150.0（Nm）；不限制就 None

# 播放多少个周期
N_CYCLES = 100


# -------------------------
# 3. 主函数
# -------------------------

def main():
    # 3.1 读取 12 关节傅立叶参数
    params = CMUParams(csv_path=CSV_PATH, T=GAIT_T)

    # 3.2 载入 MuJoCo 模型
    if not Path(XML_PATH).is_file():
        raise FileNotFoundError(f"请先将 XML_PATH 改成有效路径，目前为: {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 3.3 关节名 -> qpos/qvel(=dof) index 映射
    qpos_adr = {}
    dof_adr = {}
    for name in JOINT_NAMES:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr[name] = int(model.jnt_qposadr[j_id])
        dof_adr[name] = int(model.jnt_dofadr[j_id])  # 对 hinge/slide 关节为 1 DOF

    # 3.4 时间设置：外层控制 dt 与 MuJoCo 内部积分 dt
    sim_dt = float(model.opt.timestep)
    ctrl_dt = 1.0 / float(CTRL_HZ)

    # 每个控制周期内做多少次 mujoco.mj_step（子步）
    n_substeps = max(1, int(round(ctrl_dt / sim_dt)))
    # 实际控制周期（为了和子步对齐）
    ctrl_dt_eff = n_substeps * sim_dt

    total_time = float(N_CYCLES) * float(GAIT_T)
    n_ctrl_steps = int(np.floor(total_time / ctrl_dt_eff))

    # 3.5 预置初始状态到参考第 0 帧，避免一上来误差过大
    t0 = 0.0
    phase0 = (t0 / GAIT_T) % 1.0
    for name in JOINT_NAMES:
        q_ref0 = float(params.eval_joint_phase(name, phase0))
        dq_ref0 = float(params.eval_joint_phase_vel(name, phase0, phase_dot=1.0 / GAIT_T))
        data.qpos[qpos_adr[name]] = q_ref0
        data.qvel[dof_adr[name]] = dq_ref0
    mujoco.mj_forward(model, data)

    # 3.6 日志：ref vs actual
    time_log = np.zeros(n_ctrl_steps)
    q_ref_log = {name: np.zeros(n_ctrl_steps) for name in JOINT_NAMES}
    q_act_log = {name: np.zeros(n_ctrl_steps) for name in JOINT_NAMES}
    dq_ref_log = {name: np.zeros(n_ctrl_steps) for name in JOINT_NAMES}
    dq_act_log = {name: np.zeros(n_ctrl_steps) for name in JOINT_NAMES}
    tau_log = {name: np.zeros(n_ctrl_steps) for name in JOINT_NAMES}

    # 3.7 启动 MuJoCo Viewer（被动 viewer，我们自己推进 mj_step）
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        for i in range(n_ctrl_steps):
            if not viewer.is_running():
                break

            wall_start = time.time()

            time_log[i] = t
            phase = (t / GAIT_T) % 1.0

            # 清空本控制周期的外加广义力
            data.qfrc_applied[:] = 0.0

            # 计算并施加 PD 力矩（基于 ref 位置 + ref 速度）
            for name in JOINT_NAMES:
                q_ref = float(params.eval_joint_phase(name, phase))
                dq_ref = float(params.eval_joint_phase_vel(name, phase, phase_dot=1.0 / GAIT_T))

                q = float(data.qpos[qpos_adr[name]])
                dq = float(data.qvel[dof_adr[name]])

                tau = KP * (q_ref - q) + KD * (dq_ref - dq)
                if TAU_LIMIT is not None:
                    tau = float(np.clip(tau, -TAU_LIMIT, TAU_LIMIT))

                data.qfrc_applied[dof_adr[name]] = tau

                # 记录
                q_ref_log[name][i] = q_ref
                dq_ref_log[name][i] = dq_ref
                q_act_log[name][i] = q
                dq_act_log[name][i] = dq
                tau_log[name][i] = tau

            # 子步积分
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)

            viewer.sync()

            # 实时播放节奏（不想限速就把下面这段注释掉）
            t += ctrl_dt_eff
            elapsed = time.time() - wall_start
            if elapsed < ctrl_dt_eff:
                time.sleep(ctrl_dt_eff - elapsed)

    # 3.8 画图：角度 ref vs actual（deg），可选再画 tau
    plot_12_joint_tracking(time_log[:i+1], q_ref_log, q_act_log, title="PD Tracking (Angle)")
    plot_12_joint_vel_tracking(time_log[:i+1], dq_ref_log, dq_act_log, title="PD Tracking (Velocity)")
    # 如需看力矩曲线，取消下一行注释
    plot_12_joint_tau(time_log[:i+1], tau_log, title="Applied Joint Torques (qfrc_applied)")


# -------------------------
# 4. 画图：ref vs actual
# -------------------------

def plot_12_joint_tracking(time_arr: np.ndarray, q_ref_log: dict, q_act_log: dict, title: str = ""):
    RAD2DEG = 180.0 / np.pi

    n_rows, n_cols = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for idx, name in enumerate(JOINT_NAMES):
        ax = axes[idx]
        ref_deg = q_ref_log[name][:len(time_arr)] * RAD2DEG
        act_deg = q_act_log[name][:len(time_arr)] * RAD2DEG
        ax.plot(time_arr, ref_deg, label="ref")
        ax.plot(time_arr, act_deg, label="actual", linewidth=1.2)
        ax.set_title(name)
        ax.set_ylabel("angle (deg)")
        ax.grid(True)
        ax.legend(fontsize=8)

    for ax in axes[-n_cols:]:
        ax.set_xlabel("time (s)")

    if title:
        fig.suptitle(title)
        plt.subplots_adjust(top=0.92)

    plt.tight_layout()
    plt.show()


def plot_12_joint_tau(time_arr: np.ndarray, tau_log: dict, title: str = ""):
    n_rows, n_cols = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for idx, name in enumerate(JOINT_NAMES):
        ax = axes[idx]
        tau = tau_log[name][:len(time_arr)]
        ax.plot(time_arr, tau, label="tau")
        ax.set_title(name)
        ax.set_ylabel("torque (Nm)")
        ax.grid(True)
        ax.legend(fontsize=8)

    for ax in axes[-n_cols:]:
        ax.set_xlabel("time (s)")

    if title:
        fig.suptitle(title)
        plt.subplots_adjust(top=0.92)

    plt.tight_layout()
    plt.show()

def plot_12_joint_vel_tracking(time_arr: np.ndarray, dq_ref_log: dict, dq_act_log: dict, title: str = ""):
    RAD2DEG = 180.0 / np.pi

    n_rows, n_cols = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for idx, name in enumerate(JOINT_NAMES):
        ax = axes[idx]
        ref_deg_s = dq_ref_log[name][:len(time_arr)] * RAD2DEG
        act_deg_s = dq_act_log[name][:len(time_arr)] * RAD2DEG
        ax.plot(time_arr, ref_deg_s, label="ref")
        ax.plot(time_arr, act_deg_s, label="actual", linewidth=1.2)
        ax.set_title(name)
        ax.set_ylabel("vel (deg/s)")
        ax.grid(True)
        ax.legend(fontsize=8)

    for ax in axes[-n_cols:]:
        ax.set_xlabel("time (s)")

    if title:
        fig.suptitle(title)
        plt.subplots_adjust(top=0.92)

    plt.tight_layout()
    plt.show()

# -------------------------
# 5. 入口
# -------------------------

if __name__ == "__main__":
    main()
