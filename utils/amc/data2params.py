import pandas as pd
import numpy as np
import os

# === 1. 配置参数 ===
input_csv_path = "mocap_lower_body_filtered_132.csv"
output_csv_path = "fourier_coefficients_order6_132.csv"
t_start = 6
t_end = 8.4
order = 6
fps = 120

# === 2. 定义拟合函数 ===
def fit_fourier_series(t, y, order, omega):
    """
    最小二乘法求解: f(t) = a0 + Σ(an*cos(nwt) + bn*sin(nwt))
    """
    n_samples = len(t)
    # 构建设计矩阵 A
    A_mat = np.ones((n_samples, 1)) # column for a0
    
    for n in range(1, order + 1):
        cos_term = np.cos(n * omega * t).reshape(-1, 1)
        sin_term = np.sin(n * omega * t).reshape(-1, 1)
        A_mat = np.hstack((A_mat, cos_term, sin_term))
    
    # 求解 A * x = y
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    
    a0 = coeffs[0]
    a = coeffs[1::2] # 奇数位置是 a (cos)
    b = coeffs[2::2] # 偶数位置是 b (sin)
    return a0, a, b

# === 3. 数据处理 ===
if not os.path.exists(input_csv_path):
    raise FileNotFoundError(f"找不到文件: {input_csv_path}")

df = pd.read_csv(input_csv_path)

# 准备时间轴
time_all = np.arange(len(df)) / fps
mask_window = (time_all >= t_start) & (time_all <= t_end)
t_window = time_all[mask_window]

# 计算基频 (Base Frequency)
# 注意：在仿真中重建曲线时，必须使用完全相同的 omega
T_period = t_end - t_start
omega = 2 * np.pi / T_period

print(f"=== 拟合参数 ===")
print(f"时间窗口: {t_start:.4f}s ~ {t_end:.4f}s")
print(f"周期 T: {T_period:.4f}s")
print(f"基频 Omega: {omega:.6f} rad/s (请记下此数值用于仿真)")

# 准备存储列表
data_rows = []
target_cols = df.columns[:12] # 前12个关节角度数据

for col_name in target_cols:
    y_window = df.loc[mask_window, col_name].values
    
    # 拟合
    a0, a_coeffs, b_coeffs = fit_fourier_series(t_window, y_window, order, omega)
    
    # 构建一行数据: [Name, a0, a1..a6, b1..b6]
    row = {
        'joint_name': col_name,
        'a0': a0
    }
    
    # 填充 an
    for i in range(order):
        row[f'a{i+1}'] = a_coeffs[i]
        
    # 填充 bn
    for i in range(order):
        row[f'b{i+1}'] = b_coeffs[i]
        
    data_rows.append(row)

# === 4. 保存为 CSV ===
df_coeffs = pd.DataFrame(data_rows)

# 调整列顺序，确保直观 (Name, a0, a1...a6, b1...b6)
cols_order = ['joint_name', 'a0'] + [f'a{i+1}' for i in range(order)] + [f'b{i+1}' for i in range(order)]
df_coeffs = df_coeffs[cols_order]

df_coeffs.to_csv(output_csv_path, index=False, float_format='%.8f')

print(f"\n✅ 成功保存系数文件: {output_csv_path}")
print("文件内容预览:")
print(df_coeffs.head(3))