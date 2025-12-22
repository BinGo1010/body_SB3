# hip_torque_cpg_demo.py
# 最小 HipTorqueCPG 演示：固定步频 + 扭矩幅值，画出 τ_L / τ_R 曲线

import numpy as np
import matplotlib.pyplot as plt

# Support importing both when run as package (python -m cpg.cpg) and as a
# standalone script (python cpg/cpg.py). Prefer relative imports for package
# usage and fall back to absolute imports / sys.path adjustment when needed.
# try:
#     # Package-style import (preferred when run via `python -m cpg.cpg`)
#     from .params import PaperParams
#     from .model import HipTorqueCPG
# except Exception:
    # Script-style fallback: ensure repository root is on sys.path, then import
import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# Absolute imports relative to repo root
from params import PaperParams
from model import HipTorqueCPG


def main():
    # 1) 创建参数和髋关节扭矩 CPG
    p = PaperParams()
    dt = 0.002       # 控制周期 / 仿真步长，和 HipTorqueCPG 里的一致
    T  = 50.0        # 模拟 10 秒
    steps = int(T / dt)

    cpg_tau = HipTorqueCPG(p, dt=dt)

    # 2) 设定固定的目标参数
    Omega_target  = 0.5               # 目标步频 rad/s
    amp_target    = np.array([0.08, 0.08])  # 左右髋扭矩“幅值系数”
    offset_target = np.array([0.0, 0.0])    # 左右髋扭矩偏置（Nm）

    # 提醒：base(φ) - a0 是一个“零均值”的 hip 角度（deg 级），
    #       tau = offset + amp * (base - a0)
    #       amp 的量纲等价于 Nm/deg，数值可以按需要调大/调小

    # 3) 存储时间和扭矩轨迹
    t_arr   = np.linspace(0.0, T, steps)
    tau_hist = np.zeros((steps, 2))   # [:, 0] -> τ_L, [:, 1] -> τ_R

    for k in range(steps):
        out = cpg_tau.step(Omega_target, amp_target, offset_target)
        tau_hist[k, :] = out["tau"]

    # 4) 画图：τ_L / τ_R 随时间变化
    plt.figure(figsize=(10, 5))
    plt.plot(t_arr, tau_hist[:, 0], label=r"$\tau_L$ (hip left)")
    plt.plot(t_arr, tau_hist[:, 1], label=r"$\tau_R$ (hip right)", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Torque (arb. units / Nm)")
    plt.title("HipTorqueCPG outputs: left / right hip torque")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
