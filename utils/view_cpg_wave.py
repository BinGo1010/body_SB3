# -*- coding: utf-8 -*-
"""
view_cpg_wave.py  （增强版：记录 EXO 实际角度/角速度/力矩 + CPG 理论输出角度(及角速度)做对比，
                   并在关闭窗口/关闭viewer后弹出曲线图）

功能：
- 复用同目录下的 model.py 里的 HipAngleCPG，以及 params.py 里的 PaperParams
- GUI（Tkinter）用滑块实时调 CPG 参数：
    Omega_raw（rad/s）,
    amp_raw_L, amp_raw_R（比例系数，缩放 Fourier 角度模板）,
    off_raw_L, off_raw_R（rad）
- HipAngleCPG 输出左右髋关节期望角度 q_des（rad）和期望角速度 dq_des（rad/s）
- 在 MuJoCo 中对 exo_hip_l/exo_hip_r（motor）施加 PD 力矩，使 lr_joint/rr_joint 跟踪 q_des
- 记录：
    1) lr_joint/rr_joint 实际角度 q、角速度 dq
    2) CPG 理论输出角度 q_des（以及 dq_des，可选）
    3) exo_hip_l/exo_hip_r 的力矩命令（写入 data.ctrl，已按 ctrlrange 饱和）
- 当关闭 GUI 窗口或关闭 MuJoCo viewer 窗口后，自动弹出曲线图对比显示

运行：
  python view_cpg_wave.py --xml /path/to/your_model.xml

XML 必须包含以下名字：
  joints : lr_joint, rr_joint
  actuators: exo_hip_l, exo_hip_r   （motor, gear=1, ctrlrange 例如 [-200,200]）
"""

import argparse
import time
import types
import sys
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk

import importlib.util

import mujoco
import mujoco.viewer

# 绘图：在关闭窗口后弹出显示
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# ----------------------------
# 动态加载同目录下的 model.py / params.py
# 解决 model.py 里可能存在的相对导入（from .params import PaperParams 等）问题
# ----------------------------
def load_local_modules(cpg_dir: Path):
    pkg_name = "cpg_pkg_runtime"
    if pkg_name not in sys.modules:
        sys.modules[pkg_name] = types.ModuleType(pkg_name)

    def _load(mod_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载模块: {mod_name} <- {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)  # type: ignore
        return mod

    params_mod = _load(f"{pkg_name}.params", cpg_dir / "params.py")
    model_mod = _load(f"{pkg_name}.model", cpg_dir / "model.py")
    return params_mod, model_mod


# ----------------------------
# GUI
# ----------------------------
class SliderRow(ttk.Frame):
    def __init__(self, parent, text, var, frm, to):
        super().__init__(parent)
        self.var = var
        self.label = ttk.Label(self, text=text, width=22)
        self.label.grid(row=0, column=0, sticky="w")
        self.scale = ttk.Scale(self, from_=frm, to=to, variable=var)
        self.scale.grid(row=0, column=1, sticky="ew", padx=8)
        self.value = ttk.Label(self, textvariable=var, width=12)
        self.value.grid(row=0, column=2, sticky="e")
        self.columnconfigure(1, weight=1)


def build_gui():
    root = tk.Tk()
    root.title("HipAngleCPG 参数调节 + MuJoCo PD 跟踪（记录对比）")

    # 5 个 CPG 参数
    omega = tk.DoubleVar(value=6.0)  # rad/s （约 1 Hz -> 2*pi ≈ 6.283）
    ampL = tk.DoubleVar(value=1.0)   # scale
    ampR = tk.DoubleVar(value=1.0)   # scale
    offL = tk.DoubleVar(value=0.0)   # rad
    offR = tk.DoubleVar(value=0.0)   # rad

    # PD 参数（可选）
    kp = tk.DoubleVar(value=20.0)
    kd = tk.DoubleVar(value=2.0)

    paused = tk.BooleanVar(value=False)

    main = ttk.Frame(root, padding=10)
    main.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    frm = ttk.LabelFrame(main, text="CPG 参数（滑块实时生效）", padding=10)
    frm.grid(row=0, column=0, sticky="nsew")
    main.columnconfigure(0, weight=1)
    frm.columnconfigure(0, weight=1)

    SliderRow(frm, "Omega_raw (rad/s)", omega, 0.0, 16.0).grid(row=0, column=0, sticky="ew", pady=2)
    SliderRow(frm, "amp_raw_L (scale)", ampL, 0.0, 2.5).grid(row=1, column=0, sticky="ew", pady=2)
    SliderRow(frm, "amp_raw_R (scale)", ampR, 0.0, 2.5).grid(row=2, column=0, sticky="ew", pady=2)
    SliderRow(frm, "off_raw_L (rad)", offL, -0.8, 0.8).grid(row=3, column=0, sticky="ew", pady=2)
    SliderRow(frm, "off_raw_R (rad)", offR, -0.8, 0.8).grid(row=4, column=0, sticky="ew", pady=2)

    frm_pd = ttk.LabelFrame(main, text="PD 参数（可选）", padding=10)
    frm_pd.grid(row=1, column=0, sticky="ew", pady=(10, 0))
    frm_pd.columnconfigure(0, weight=1)
    SliderRow(frm_pd, "Kp (Nm/rad)", kp, 0.0, 300.0).grid(row=0, column=0, sticky="ew", pady=2)
    SliderRow(frm_pd, "Kd (Nm/(rad/s))", kd, 0.0, 50.0).grid(row=1, column=0, sticky="ew", pady=2)

    status = ttk.LabelFrame(main, text="状态", padding=10)
    status.grid(row=2, column=0, sticky="ew", pady=(10, 0))
    status.columnconfigure(0, weight=1)

    s_text = tk.StringVar(value="等待仿真启动...")
    lbl = ttk.Label(status, textvariable=s_text, justify="left")
    lbl.grid(row=0, column=0, sticky="w")

    btns = ttk.Frame(main)
    btns.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    ttk.Checkbutton(btns, text="Pause(输出零力矩)", variable=paused).grid(row=0, column=0, sticky="w")

    return root, (omega, ampL, ampR, offL, offR, kp, kd, paused, s_text)


def _setup_matplotlib_cn():
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]


def _plot_logs(log: dict):
    if len(log["t"]) <= 1:
        print("[log] 数据点太少，未绘图。")
        return

    _setup_matplotlib_cn()

    t = np.asarray(log["t"])
    qL = np.asarray(log["qL"]); qR = np.asarray(log["qR"])
    dqL = np.asarray(log["dqL"]); dqR = np.asarray(log["dqR"])
    qdL = np.asarray(log["qdesL"]); qdR = np.asarray(log["qdesR"])
    dqdL = np.asarray(log["dqdesL"]); dqdR = np.asarray(log["dqdesR"])
    tauL = np.asarray(log["tauL"]); tauR = np.asarray(log["tauR"])

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(11, 7.5))

    # 角度：实际 vs CPG理论
    ax[0].plot(t, qL, label="左髋 实际角度 q(L)")
    ax[0].plot(t, qdL, label="左髋 CPG理论角度 q_des(L)", linestyle="--")
    ax[0].plot(t, qR, label="右髋 实际角度 q(R)")
    ax[0].plot(t, qdR, label="右髋 CPG理论角度 q_des(R)", linestyle="--")
    ax[0].set_ylabel("角度 (rad)")
    ax[0].legend(ncol=2)
    ax[0].grid(True)

    # 角速度：实际 vs CPG理论（如果你只想画角度对比，可以把这一段删掉）
    ax[1].plot(t, dqL, label="左髋 实际角速度 dq(L)")
    ax[1].plot(t, dqdL, label="左髋 CPG理论角速度 dq_des(L)", linestyle="--")
    ax[1].plot(t, dqR, label="右髋 实际角速度 dq(R)")
    ax[1].plot(t, dqdR, label="右髋 CPG理论角速度 dq_des(R)", linestyle="--")
    ax[1].set_ylabel("角速度 (rad/s)")
    ax[1].legend(ncol=2)
    ax[1].grid(True)

    # 力矩命令
    ax[2].plot(t, tauL, label="exo_hip_l 力矩命令(L)")
    ax[2].plot(t, tauR, label="exo_hip_r 力矩命令(R)")
    ax[2].set_ylabel("力矩 (Nm)")
    ax[2].set_xlabel("时间 (s)")
    ax[2].legend()
    ax[2].grid(True)

    fig.suptitle("EXO 关节：实际状态 vs CPG理论输出（关闭窗口后显示）")
    plt.tight_layout()
    plt.show()


# ----------------------------
# 主程序
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--xml",
        type=str,
        default="/home/lvchen/miniconda3/envs/myosuite/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml",
        help="MuJoCo XML 路径（需含 lr_joint/rr_joint + exo_hip_l/exo_hip_r）",
    )
    ap.add_argument("--ctrl_dt", type=float, default=0.02,
                    help="CPG/PD 目标刷新周期（s），默认 0.02=50Hz")
    ap.add_argument("--realtime", action="store_true",
                    help="尽量按真实时间运行（会 sleep）")
    args = ap.parse_args()

    xml_path = Path(args.xml).expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML 不存在: {xml_path}")

    # 加载本地 HipAngleCPG / PaperParams（从 cpg 目录）
    # 若你的目录结构不同，请把下面这一行改成你的实际路径
    cpg_dir = Path(__file__).resolve().parent.parent / "cpg"
    params_mod, model_mod = load_local_modules(cpg_dir)
    PaperParams = params_mod.PaperParams
    HipAngleCPG = model_mod.HipAngleCPG

    # 构造参数并把 hip Fourier 系数从“度” -> “弧度”
    p = PaperParams()
    deg2rad = np.pi / 180.0
    p.hip_a0 = float(p.hip_a0) * deg2rad
    p.hip_a = np.asarray(p.hip_a, dtype=float) * deg2rad
    p.hip_b = np.asarray(p.hip_b, dtype=float) * deg2rad

    # CPG：dt 取 ctrl_dt，保证输出与控制刷新一致
    cpg = HipAngleCPG(p, dt=float(args.ctrl_dt), phase0=0.0, S_hip=6)

    # GUI
    root, (omega, ampL, ampR, offL, offR, kp, kd, paused, s_text) = build_gui()

    # 日志：按控制周期记录
    log = {
        "t": [],
        "qL": [], "qR": [],
        "dqL": [], "dqR": [],
        "qdesL": [], "qdesR": [],        # CPG理论角度
        "dqdesL": [], "dqdesR": [],      # CPG理论角速度（可选）
        "tauL": [], "tauR": [],
    }

    closing = {"flag": False}

    def _on_close():
        closing["flag"] = True
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", _on_close)

    # MuJoCo
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    sim_dt = float(model.opt.timestep)
    ctrl_dt = float(args.ctrl_dt)
    substeps = max(1, int(round(ctrl_dt / sim_dt)))
    ctrl_dt_eff = substeps * sim_dt  # 实际控制周期

    # 预检查名称
    for jn in ["lr_joint", "rr_joint"]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) < 0:
            raise RuntimeError(f"XML 中不存在 joint '{jn}'，请确认命名一致。")
    for an in ["exo_hip_l", "exo_hip_r"]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, an) < 0:
            raise RuntimeError(f"XML 中不存在 actuator '{an}'，请确认 <actuator> motor 命名一致。")

    # 预取索引
    jidL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lr_joint")
    jidR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rr_joint")
    qadrL = int(model.jnt_qposadr[jidL]); dadrL = int(model.jnt_dofadr[jidL])
    qadrR = int(model.jnt_qposadr[jidR]); dadrR = int(model.jnt_dofadr[jidR])

    aidL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "exo_hip_l")
    aidR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "exo_hip_r")
    loL, hiL = model.actuator_ctrlrange[aidL]
    loR, hiR = model.actuator_ctrlrange[aidR]

    # 初始化 CPG 输出缓存（保证第一次记录时也有值）
    q_des = np.zeros(2, dtype=float)
    dq_des = np.zeros(2, dtype=float)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            step_count = 0
            t_wall0 = time.time()
            t_sim0 = float(data.time)

            while viewer.is_running():
                # 如果 GUI 发起关闭
                if closing["flag"]:
                    break

                # Tk 刷新（若窗口已销毁会抛 TclError）
                try:
                    root.update_idletasks()
                    root.update()
                except tk.TclError:
                    break

                # 每 ctrl_dt_eff 更新一次 CPG 输出（期望角/角速度）
                if step_count % substeps == 0:
                    Omega_target = float(omega.get())
                    amp_target = np.array([float(ampL.get()), float(ampR.get())], dtype=float)
                    off_target = np.array([float(offL.get()), float(offR.get())], dtype=float)

                    out = cpg.step(Omega_target, amp_target, off_target)
                    q_des = np.asarray(out["q_des"], dtype=float).ravel()    # [L, R]
                    dq_des = np.asarray(out["dq_des"], dtype=float).ravel()  # [L, R]

                # 读取当前关节状态
                qL = float(data.qpos[qadrL]); dqL = float(data.qvel[dadrL])
                qR = float(data.qpos[qadrR]); dqR = float(data.qvel[dadrR])

                # PD
                if bool(paused.get()):
                    tauL = 0.0
                    tauR = 0.0
                else:
                    Kp = float(kp.get())
                    Kd = float(kd.get())
                    tauL = Kp * (q_des[0] - qL) + Kd * (dq_des[0] - dqL)
                    tauR = Kp * (q_des[1] - qR) + Kd * (dq_des[1] - dqR)

                # 写入 actuator（按 ctrlrange 饱和）
                tauL_clip = float(np.clip(tauL, loL, hiL))
                tauR_clip = float(np.clip(tauR, loR, hiR))
                data.ctrl[aidL] = tauL_clip
                data.ctrl[aidR] = tauR_clip

                # 记录（按控制周期）
                if step_count % substeps == 0:
                    log["t"].append(float(data.time))

                    log["qL"].append(qL)
                    log["qR"].append(qR)
                    log["dqL"].append(dqL)
                    log["dqR"].append(dqR)

                    log["qdesL"].append(float(q_des[0]))
                    log["qdesR"].append(float(q_des[1]))
                    log["dqdesL"].append(float(dq_des[0]))
                    log["dqdesR"].append(float(dq_des[1]))

                    log["tauL"].append(tauL_clip)
                    log["tauR"].append(tauR_clip)

                mujoco.mj_step(model, data)
                step_count += 1

                # 状态显示（约 10Hz）
                if step_count % max(1, int(round(0.1 / sim_dt))) == 0:
                    s_text.set(
                        "控制周期: %.4fs (substeps=%d)\n"
                        "CPG理论 q_des [L,R] = [%.3f, %.3f] rad\n"
                        "实际    q    [L,R] = [%.3f, %.3f] rad\n"
                        "力矩命令 tau  [L,R] = [%.1f, %.1f] Nm (clipped)\n"
                        "sim_time = %.2fs"
                        % (ctrl_dt_eff, substeps, q_des[0], q_des[1], qL, qR, tauL_clip, tauR_clip, float(data.time))
                    )

                viewer.sync()

                if args.realtime:
                    t_sim = float(data.time) - t_sim0
                    t_wall = time.time() - t_wall0
                    sleep_s = t_sim - t_wall
                    if sleep_s > 0:
                        time.sleep(min(sleep_s, 0.01))

    finally:
        # 无论如何都尝试销毁 GUI
        try:
            root.destroy()
        except Exception:
            pass

        # 关闭后显示曲线（含：实际 vs CPG理论）
        _plot_logs(log)


if __name__ == "__main__":
    main()
