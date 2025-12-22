import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === 配置参数 ===
csv_path = "mocap_lower_body_filtered_132.csv"
t_start = 6  # 截取开始时间
t_end = 8.4    # 截取结束时间
order = 6         # 傅里叶阶数
fps = 120         # 采样率

# === 1. 傅里叶拟合函数 ===
def fit_fourier_series(t, y, order, omega):
    """
    使用最小二乘法拟合傅里叶级数:
    f(t) = a0 + sum(an * cos(n*w*t) + bn * sin(n*w*t))
    """
    # 构建设计矩阵 A
    # A * x = y, 其中 x 是系数向量 [a0, a1, b1, ..., a6, b6]
    n_samples = len(t)
    A_mat = np.ones((n_samples, 1)) # 第一列是 a0 (常数项)
    
    for n in range(1, order + 1):
        cos_term = np.cos(n * omega * t).reshape(-1, 1)
        sin_term = np.sin(n * omega * t).reshape(-1, 1)
        A_mat = np.hstack((A_mat, cos_term, sin_term))
    
    # 最小二乘解
    # x, residuals, rank, s = np.linalg.lstsq(A_mat, y, rcond=None)
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    
    # 提取系数
    a0 = coeffs[0]
    a = coeffs[1::2] # 奇数索引对应 cos 系数
    b = coeffs[2::2] # 偶数索引对应 sin 系数
    
    return a0, a, b

def eval_fourier_series(t, a0, a, b, omega):
    """根据系数计算傅里叶级数值"""
    y_pred = np.full_like(t, a0)
    for n in range(len(a)):
        y_pred += a[n] * np.cos((n + 1) * omega * t)
        y_pred += b[n] * np.sin((n + 1) * omega * t)
    return y_pred

# === 2. 加载与预处理数据 ===
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"找不到文件: {csv_path}")

df = pd.read_csv(csv_path)

# 创建全长的时间轴
total_frames = len(df)
time_all = np.arange(total_frames) / fps

# 截取指定窗口的数据索引
mask_window = (time_all >= t_start) & (time_all <= t_end)
t_window = time_all[mask_window]

# 计算窗口持续时间作为基波周期 T
T_period = t_end - t_start
omega = 2 * np.pi / T_period  # 基频 rad/s

print(f"=== 拟合配置 ===")
print(f"窗口范围: {t_start}s - {t_end}s (持续 {T_period:.4f}s)")
print(f"基频 Omega: {omega:.4f} rad/s")
print(f"拟合阶数: {order}")

# === 3. 循环拟合前12条曲线并绘图 ===
target_cols = df.columns[:12] # 选取前12列 (通常是 joint angles)

fig, axes = plt.subplots(4, 3, figsize=(18, 12), constrained_layout=True)
fig.suptitle(f'6th-Order Fourier Fit & Extension ({t_start}-{t_end}s)', fontsize=16)
axes_flat = axes.flatten()

# 用于拓延的时间轴 (比原始数据更长，或者覆盖整个原始范围以便对比)
# 这里我们选择覆盖 0s 到 数据结尾
t_ext = time_all 

for i, col_name in enumerate(target_cols):
    ax = axes_flat[i]
    
    # 获取当前列的窗口数据
    y_window = df.loc[mask_window, col_name].values
    
    # --- 核心：执行拟合 ---
    a0, a_coeffs, b_coeffs = fit_fourier_series(t_window, y_window, order, omega)
    
    # --- 计算拟合值 (在拓延时间轴上) ---
    y_fitted_ext = eval_fourier_series(t_ext, a0, a_coeffs, b_coeffs, omega)
    
    # --- 计算误差 (仅在窗口内) ---
    y_fit_window = eval_fourier_series(t_window, a0, a_coeffs, b_coeffs, omega)
    rmse = np.sqrt(np.mean((y_window - y_fit_window)**2))
    
    # --- 绘图 ---
    # 1. 原始全长数据 (灰色背景)
    ax.plot(time_all, df[col_name], color='gray', alpha=0.5, linewidth=1, label='Original Filtered')
    
    # 2. 傅里叶拟合/拓延曲线 (红色虚线)
    ax.plot(t_ext, y_fitted_ext, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Fourier Ext (RMSE={rmse:.4f})')
    
    # 3. 拟合使用的窗口数据 (蓝色实线，加粗)
    ax.plot(t_window, y_window, color='blue', linewidth=2, label='Training Window')
    
    # 4. 标注窗口范围
    ax.axvline(t_start, color='k', linestyle=':', alpha=0.3)
    ax.axvline(t_end, color='k', linestyle=':', alpha=0.3)
    
    # 样式设置
    ax.set_title(col_name, fontsize=10, fontweight='bold')
    if i >= 9: ax.set_xlabel('Time (s)')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 仅在第一个图显示图例，避免遮挡
    if i == 0:
        ax.legend(loc='best', fontsize=8)

print("绘图完成，正在显示...")
plt.show()

# === 4. (可选) 打印某一列的系数用于验证 ===
print("\n=== 示例系数 (第1列: " + target_cols[0] + ") ===")
y_example = df.loc[mask_window, target_cols[0]].values
a0_ex, a_ex, b_ex = fit_fourier_series(t_window, y_example, order, omega)
print(f"A0 (直流分量): {a0_ex:.6f}")
print("n\tAn (Cos)\tBn (Sin)")
for n in range(order):
    print(f"{n+1}\t{a_ex[n]:.6f}\t{b_ex[n]:.6f}")