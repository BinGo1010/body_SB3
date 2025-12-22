import pandas as pd
import matplotlib.pyplot as plt
import os

# === 1. 读取 CSV 文件 ===
csv_path = "mocap_lower_body_filtered_132.csv"

if not os.path.exists(csv_path):
    print(f"错误: 找不到文件 {csv_path}，请确保先运行了上面的生成脚本。")
else:
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 获取前 12 个列名
    # 根据之前的逻辑，这通常包含 lfemur(3), rfemur(3), ltibia(1/3), rtibia(1/3) 等的位置信息
    target_cols = df.columns[:12]
    
    print("=== 文件读取成功 ===")
    print(f"总行数: {len(df)}")
    print("=== 前 12 列数据预览 (前 5 行) ===")
    print(df[target_cols].head())

    # === 2. 绘制前 12 条曲线 ===
    # 设置画布：4行3列，共12个子图
    fig, axes = plt.subplots(4, 3, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(f'First 12 Filtered Curves from {csv_path}', fontsize=16)
    
    # 展平 axes 数组方便遍历
    axes_flat = axes.flatten()
    
    # 生成时间轴 (假设 120fps)
    fps = 120
    time_axis = [i/fps for i in range(len(df))]

    for i, col_name in enumerate(target_cols):
        ax = axes_flat[i]
        
        # 绘制曲线
        ax.plot(time_axis, df[col_name], label=col_name, color='#1f77b4', linewidth=1.5)
        
        # 样式调整
        ax.set_title(col_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Rad', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 如果你想看具体的点，可以把下面这行取消注释
        # ax.scatter(time_axis, df[col_name], s=1, c='r')

    print("\n正在显示绘图窗口...")
    plt.show()