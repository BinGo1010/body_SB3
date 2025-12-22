import os
from amc_parser import parse_asf, parse_amc
import numpy as np
from scipy.signal import butter, filtfilt  # 引入信号处理库

# 当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

asf_path = os.path.join(BASE_DIR, "132/132.asf")
amc_path = os.path.join(BASE_DIR, "132/132_46.amc")
output_path = "mocap_lower_body_filtered_132.csv" # 修改输出文件名以区分

# === 滤波参数设置 ===
fps = 120
cutoff_freq = 6.0  # 截止频率 (Hz)。对于人体运动，通常取 6Hz - 12Hz
filter_order = 4   # 滤波器阶数 (通常 2阶 或 4阶)

# === 定义巴特沃斯低通滤波器函数 ===
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyquist
    # 设计滤波器系数
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # 使用 filtfilt 进行零相位滤波 (前向和后向各滤一次，消除相位延迟)
    y = filtfilt(b, a, data, axis=0)
    return y

# === 加载数据 ===
print("正在加载 AMC/ASF 文件...")
joints = parse_asf(asf_path)
motions = parse_amc(amc_path)
print(f"总帧数: {len(motions)}")

# === 关键关节列表 ===
joint_order = ['lfemur', 'rfemur', 'ltibia', 'rtibia', 'lfoot', 'rfoot']
dt = 1.0 / fps

# === 提取原始角度序列（rad） ===
qpos_raw = []
header_names = []

# 生成表头
first_frame = motions[0]
qpos_headers = []
qvel_headers = []

for jname in joint_order:
    angles = first_frame.get(jname, [0])
    if isinstance(angles, (int, float)):
        angles = [angles]
    dim = len(angles)
    for d in range(dim):
        qpos_headers.append(f"{jname}_{d}_pos")
        qvel_headers.append(f"{jname}_{d}_vel")

# 提取数据
for frame in motions:
    qpos_frame = []
    for jname in joint_order:
        angles = frame.get(jname, [0])
        if isinstance(angles, (int, float)):
            angles = [angles]
        qpos_frame.extend(angles)
    qpos_raw.append(qpos_frame)

# 转换为 numpy 数组并转弧度
qpos_seq = np.deg2rad(np.array(qpos_raw))  # shape = (T, DOF)

# === 核心修改：应用低通滤波 ===
print(f"正在应用低通滤波 (截止频率: {cutoff_freq}Hz, 阶数: {filter_order})...")

# 1. 对位置数据进行滤波
qpos_filtered = butter_lowpass_filter(qpos_seq, cutoff_freq, fps, order=filter_order)

# 2. 基于滤波后的位置数据计算速度 (中心差分)
# 这样得出的速度曲线会比直接从原始数据计算更加平滑
qvel_filtered = np.gradient(qpos_filtered, dt, axis=0)

# === 合并数据并保存为 CSV ===
# 拼接 [滤波后的位置, 滤波后的速度]
data_to_save = np.hstack((qpos_filtered, qvel_filtered))
full_header = ",".join(qpos_headers + qvel_headers)

np.savetxt(output_path, data_to_save, delimiter=",", header=full_header, comments='', fmt='%.6f')

print(f"✅ 保存成功: {output_path}")
print(f"数据形状: {data_to_save.shape}")
print("处理说明: 已对 qpos 应用零相位巴特沃斯低通滤波，并由此计算 qvel。")