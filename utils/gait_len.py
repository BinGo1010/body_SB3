import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
from collections import deque

# 加入项目路径，确保可导入 cpg.params
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cpg.params import PaperParams

# 修改成你自己的 myolegs.xml 路径
xml_path = "/home/lvchen/miniconda3/envs/myosuite/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 足底 site ID（用于计算几何步幅）
# 确保 XML 中存在名为 'l_foot' 和 'r_foot' 的 site
l_foot_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'l_foot')
r_foot_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'r_foot')

# 骨骼关节名（必须匹配 XML 中名称）
joint_names = ["hip_flexion_l", "hip_flexion_r", "knee_angle_l", "knee_angle_r"]
joint_ids = [model.joint(name).qposadr.item() for name in joint_names]

# 傅里叶轨迹生成函数
def fourier_eval(phi, a0, a, b):
    l = np.arange(1, len(a) + 1, dtype=float)
    return a0 + np.sum(a * np.cos(l * phi) + b * np.sin(l * phi))

params = PaperParams()
hip_a0, hip_a, hip_b = params.hip_a0, params.hip_a, params.hip_b
knee_a0, knee_a, knee_b = params.knee_a0, params.knee_a, params.knee_b

# 仿真参数
dt = 0.002
omega = 2 * np.pi * params.Omega_base  # 也可以直接指定，比如 omega = 2*np.pi*1.0

# 设置角度限制（rad）
hip_min, hip_max = np.deg2rad([-30, 120])
knee_min, knee_max = np.deg2rad([0, 120])

# 缓存数据用于绘图
time_buffer = []
actual_hip_l = []
desired_hip_l = []
actual_knee_l = []
desired_knee_l = []
actual_hip_r = []
desired_hip_r = []
actual_knee_r = []
desired_knee_r = []

# 足端轨迹缓存用于几何步幅估算
footL_hist = []  # 左脚在前进方向上的位置
footR_hist = []  # 右脚在前进方向上的位置
time_foot_hist = []  # 记录对应的时间（可选）

# 前进方向轴索引：0=x, 1=y, 2=z；根据模型设置修改
stride_axis = 1

# 启动 Viewer
t0 = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = time.time() - t0
        phase = (omega * t) % (2 * np.pi)

        phi_L = phase + np.pi
        phi_R = phase

        # 用 PaperParams 的多谐波系数生成期望关节角（先以 deg 计算，再转 rad）
        des_angles = np.deg2rad([
            fourier_eval(phi_L, hip_a0, hip_a, hip_b),
            fourier_eval(phi_R, hip_a0, hip_a, hip_b),
            fourier_eval(phi_L, knee_a0, knee_a, knee_b),
            fourier_eval(phi_R, knee_a0, knee_a, knee_b),
        ])

        # 限幅
        des_angles = np.clip(
            des_angles,
            [hip_min, hip_min, knee_min, knee_min],
            [hip_max, hip_max, knee_max, knee_max],
        )

        # 简单 PD 控制
        for i, (jid, desired) in enumerate(zip(joint_ids, des_angles)):
            name = joint_names[i]
            qpos = data.qpos[jid]
            qvel = data.qvel[jid]

            if name in ["hip_flexion_l", "hip_flexion_r"]:
                kp_i = 10000
                kd_i = 50
            else:
                kp_i = 3000
                kd_i = 15

            torque = kp_i * (desired - qpos) - kd_i * qvel
            data.ctrl[i] = float(np.clip(torque, -15000, 15000))

        # 记录关节角轨迹（用于绘图）
        time_buffer.append(t)
        actual_hip_l.append(data.qpos[joint_ids[0]])
        desired_hip_l.append(des_angles[0])
        actual_knee_l.append(data.qpos[joint_ids[2]])
        desired_knee_l.append(des_angles[2])
        actual_hip_r.append(data.qpos[joint_ids[1]])
        desired_hip_r.append(des_angles[1])
        actual_knee_r.append(data.qpos[joint_ids[3]])
        desired_knee_r.append(des_angles[3])

        # 推进仿真一步
        mujoco.mj_step(model, data)

        # 记录足端在前进方向上的轨迹（世界坐标系）
        l_foot_pos = data.site_xpos[l_foot_sid]  # [x, y, z]
        r_foot_pos = data.site_xpos[r_foot_sid]
        footL_hist.append(l_foot_pos[stride_axis])
        footR_hist.append(r_foot_pos[stride_axis])
        time_foot_hist.append(t)

        viewer.sync()
        time.sleep(dt)

# 计算几何步幅（基于足端在前进方向上的最大/最小位移）
if len(footL_hist) > 0 and len(footR_hist) > 0:
    footL_arr = np.array(footL_hist)
    footR_arr = np.array(footR_hist)

    stride_L = float(footL_arr.max() - footL_arr.min())
    stride_R = float(footR_arr.max() - footR_arr.min())

    print(f"[Gait] 左脚几何步幅 ≈ {stride_L:.4f} m")
    print(f"[Gait] 右脚几何步幅 ≈ {stride_R:.4f} m")
else:
    print("[Gait] 未记录到足端轨迹，请检查 site 名称和仿真是否运行。")

# 绘图：左右髋/膝实际与目标轨迹
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axs[0].plot(time_buffer, actual_hip_l, label="hip_flexion_l (actual)")
axs[0].plot(time_buffer, desired_hip_l, label="hip_flexion_l (target)", linestyle="--")
axs[0].plot(time_buffer, actual_knee_l, label="knee_angle_l (actual)")
axs[0].plot(time_buffer, desired_knee_l, label="knee_angle_l (target)", linestyle="--")
axs[0].set_ylabel("Angle (rad)")
axs[0].set_title("Left Hip and Knee")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(time_buffer, actual_hip_r, label="hip_flexion_r (actual)")
axs[1].plot(time_buffer, desired_hip_r, label="hip_flexion_r (target)", linestyle="--")
axs[1].plot(time_buffer, actual_knee_r, label="knee_angle_r (actual)")
axs[1].plot(time_buffer, desired_knee_r, label="knee_angle_r (target)", linestyle="--")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Angle (rad)")
axs[1].set_title("Right Hip and Knee")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
