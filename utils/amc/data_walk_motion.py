
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from amc_parser import parse_asf, parse_amc
import numpy as np

# 当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

asf_path = os.path.join(BASE_DIR, "02/02.asf")
amc_path = os.path.join(BASE_DIR, "02/02_02.amc")

joints = parse_asf(asf_path)
motions = parse_amc(amc_path)

# === 获取所有骨架对 (child-parent) ===
def get_bone_pairs(joint):
    pairs = []
    for child in joint.children:
        pairs.append((child, child.parent))
        pairs += get_bone_pairs(child)
    return pairs

bone_pairs = get_bone_pairs(joints["root"])

# === Matplotlib 设置 ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])
ax.set_zlim([0, 120])
ax.view_init(elev=90, azim=-90)

# === 每一帧更新函数 ===
def update(frame_idx):
    ax.cla()
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([0, 120])
    ax.view_init(elev=90, azim=-90)
    ax.set_title(f"Frame {frame_idx}")

    joints["root"].set_motion(motions[frame_idx])
    for child, parent in bone_pairs:
        if child.coordinate is not None and parent.coordinate is not None:
            x = [parent.coordinate[0, 0], child.coordinate[0, 0]]
            y = [parent.coordinate[1, 0], child.coordinate[1, 0]]
            z = [parent.coordinate[2, 0], child.coordinate[2, 0]]
            ax.plot(x, y, z, color='black')

# === 创建动画 ===
ani = animation.FuncAnimation(fig, update, frames=len(motions), interval=1000/120, repeat=False)

# === 保存为视频 ===
output_path = "walk_motion_02.mp4"
ani.save(output_path, writer="ffmpeg", fps=120)
print(f"✅ 视频已保存到：{output_path}")