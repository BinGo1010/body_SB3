# -*- coding: utf-8 -*-
"""
show_fourier_derivatives.py

功能：
- 从 cmu_12dofs_fourier.csv 读取 12 个关节的傅立叶系数
- 使用周期 T=1.0833 s, 基频 Omega=5.80 rad/s
- 对每个关节计算 q(t)、q''(t)、q'''(t)
- 以图像形式展示（每个关节一个 figure，包含 3 条曲线）

CSV 格式示例：
joint_name,a0,a1,a2,a3,a4,a5,a6,b1,b2,b3,b4,b5,b6
lfemur_0_pos,-0.2153,0.3614,...,b6
...

使用方法：
    python show_fourier_derivatives.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- 参数设置 ----------------

# 基频 & 周期（与你拟合时一致）
OMEGA = 5.80          # rad/s
T = 1.0833            # s，1 个步态周期
N_HARM = 6            # K=6 阶傅立叶

# 播放时间：这里展示 2 个周期
N_PERIOD = 2
DT = 1.0 / 120.0      # 对齐 120 Hz 采样率
t = np.arange(0.0, N_PERIOD * T, DT)   # 时间轴


# ---------------- 傅立叶评估函数 ----------------

def fourier_eval(t, a0, a, b, omega):
    """
    q(t) = a0 + Σ_k [ a_k cos(k ω t) + b_k sin(k ω t) ]
    t: (N,) ndarray
    a0: float
    a, b: (K,) ndarray
    omega: float
    返回: q(t) (N,)
    """
    t = np.asarray(t)
    k = np.arange(1, len(a) + 1, dtype=float)  # (K,)
    theta = np.outer(t, k * omega)            # (N, K)

    q = a0 + np.sum(
        a[np.newaxis, :] * np.cos(theta) +
        b[np.newaxis, :] * np.sin(theta),
        axis=1
    )
    return q


def fourier_eval_ddot(t, a0, a, b, omega):
    """
    二阶导：
    q''(t) = Σ_k [ -(kω)^2 a_k cos(kωt) - (kω)^2 b_k sin(kωt) ]
    """
    t = np.asarray(t)
    k = np.arange(1, len(a) + 1, dtype=float)
    theta = np.outer(t, k * omega)            # (N, K)
    factor = - (k * omega) ** 2               # (K,)

    q_dd = np.sum(
        factor[np.newaxis, :] * a[np.newaxis, :] * np.cos(theta) +
        factor[np.newaxis, :] * b[np.newaxis, :] * np.sin(theta),
        axis=1
    )
    return q_dd


def fourier_eval_dddot(t, a0, a, b, omega):
    """
    三阶导：
    q'''(t) = Σ_k [ (kω)^3 a_k sin(kωt) - (kω)^3 b_k cos(kωt) ]
    """
    t = np.asarray(t)
    k = np.arange(1, len(a) + 1, dtype=float)
    theta = np.outer(t, k * omega)            # (N, K)
    factor3 = (k * omega) ** 3                # (K,)

    q_ddd = np.sum(
        factor3[np.newaxis, :] * a[np.newaxis, :] * np.sin(theta) -
        factor3[np.newaxis, :] * b[np.newaxis, :] * np.cos(theta),
        axis=1
    )
    return q_ddd


# ---------------- 主程序：读取 CSV + 计算 + 展示 ----------------

def main():
    # 1. 读取 CSV（修改成你的文件名/路径）
    csv_path = "fourier_coefficients_order6_02.csv"
    df = pd.read_csv(csv_path)

    a_cols = [f"a{i}" for i in range(1, N_HARM + 1)]
    b_cols = [f"b{i}" for i in range(1, N_HARM + 1)]

    # 2. 遍历每一个关节
    for idx, row in df.iterrows():
        joint_name = row["joint_name"]

        a0 = float(row["a0"])
        a = row[a_cols].to_numpy(dtype=float)   # (K,)
        b = row[b_cols].to_numpy(dtype=float)   # (K,)

        # 3. 计算 q, q_ddot, q_dddot
        q = fourier_eval(t, a0, a, b, OMEGA)
        q_dd = fourier_eval_ddot(t, a0, a, b, OMEGA)
        q_ddd = fourier_eval_dddot(t, a0, a, b, OMEGA)

        # 4. 画图展示
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Joint: {joint_name} (T={T:.4f}s, Ω={OMEGA:.2f} rad/s)")

        # 位置
        axes[0].plot(t, q, label="q(t)")
        axes[0].set_ylabel("angle (rad)")
        axes[0].grid(True)
        axes[0].legend(loc="upper right")

        # 二阶导
        axes[1].plot(t, q_dd, label="q''(t)")
        axes[1].set_ylabel("acc (rad/s²)")
        axes[1].grid(True)
        axes[1].legend(loc="upper right")

        # 三阶导
        axes[2].plot(t, q_ddd, label="q'''(t)")
        axes[2].set_ylabel("jerk (rad/s³)")
        axes[2].set_xlabel("time (s)")
        axes[2].grid(True)
        axes[2].legend(loc="upper right")

        plt.tight_layout()
        plt.show()

        # 如果你不想逐个关节手动关图，可以在这里加个小延时或直接保存图片：
        # fig.savefig(f"{joint_name}_q_dd_dddot.png", dpi=150)
        # plt.close(fig)


if __name__ == "__main__":
    main()
