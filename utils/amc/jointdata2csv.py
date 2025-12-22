import os
from amc_parser import parse_asf, parse_amc
import numpy as np

# 当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

asf_path = os.path.join(BASE_DIR, "02/02.asf")
amc_path = os.path.join(BASE_DIR, "02/02_02.amc")

# === 修改 1: 将输出文件名后缀改为 .csv ===
output_path = "mocap_lower_body_qpos_qvel_02_02.csv"

# === 加载数据 ===
print("正在加载 AMC/ASF 文件...")
joints = parse_asf(asf_path)
motions = parse_amc(amc_path)
print(f"总帧数: {len(motions)}")

# === 关键关节列表，需与你的 myosuite 模型关节顺序一致 ===
joint_order = ['lfemur', 'rfemur', 'ltibia', 'rtibia', 'lfoot', 'rfoot']
fps = 120
dt = 1.0 / fps

# === 提取角度序列（rad） ===
qpos_seq = []
header_names = [] # 用于存储 CSV 表头

# 为了确定表头，我们先处理第一帧来获取每个关节的维度
first_frame = motions[0]
qpos_headers = []
qvel_headers = []

for jname in joint_order:
    angles = first_frame.get(jname, [0])
    if isinstance(angles, (int, float)):
        angles = [angles]
    
    # 根据该关节的数据长度生成表头 (例如: lfemur_0, lfemur_1, lfemur_2)
    dim = len(angles)
    for d in range(dim):
        qpos_headers.append(f"{jname}_{d}_pos")
        qvel_headers.append(f"{jname}_{d}_vel")

# 正式处理所有帧
for frame in motions:
    qpos_frame = []
    for jname in joint_order:
        angles = frame.get(jname, [0])
        if isinstance(angles, (int, float)):
            angles = [angles]
        qpos_frame.extend(angles)
    qpos_seq.append(qpos_frame)

qpos_seq = np.deg2rad(np.array(qpos_seq))  # 转为弧度，shape = (T, DOF)

# === 计算速度 ===
qvel_seq = np.gradient(qpos_seq, dt, axis=0)  # 中心差分

# === 修改 2: 合并数据并保存为 CSV ===
# 将位置和速度在列方向上拼接 [Position Columns, Velocity Columns]
data_to_save = np.hstack((qpos_seq, qvel_seq))

# 合并表头
full_header = ",".join(qpos_headers + qvel_headers)

# 使用 numpy 保存文本文件
# fmt='%.6f' 保留6位小数，delimiter=',' 使用逗号分隔
np.savetxt(output_path, data_to_save, delimiter=",", header=full_header, comments='', fmt='%.6f')

print(f"✅ 保存成功: {output_path}")
print(f"数据形状: {data_to_save.shape} (行=帧数, 列=自由度x2)")
print(f"包含内容: 前 {qpos_seq.shape[1]} 列为位置(qpos), 后 {qvel_seq.shape[1]} 列为速度(qvel)")