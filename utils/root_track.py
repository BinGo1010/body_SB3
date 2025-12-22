# -*- coding: utf-8 -*-
"""
实时绘制 MuJoCo 仿真中的 root 轨迹 vs CMU mocap root 参考轨迹

使用说明：
1. 修改 AMC_PATH / MODEL_XML / OBSD_XML / ENV_CLS 为你本地的路径和类名；
2. 确认你的 mocap 序列采样率（CMU 默认 120Hz），修改 FPS；
3. 直接 python 运行本脚本，即可打开图像窗口实时看到 root 轨迹曲线。
"""

import numpy as np
import matplotlib.pyplot as plt
import collections
import mujoco
from myosuite.utils import gym
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2mat
from cpg.params_12_bias import CMUParams
from pathlib import Path
import yaml
import os
import sys
# ====== 按你的工程结构修改这里 ======
# 例如：from envs.walk_v1 import WalkEnvV5
from envs.walk_gait_kinesis import WalkEnvV5  # 按实际类名修改 /home/lvchen/body_SB3/utils/amc/132/132_46.amc
# 当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

asf_path = os.path.join(BASE_DIR, "amc/132/132.asf")
AMC_PATH = os.path.join(BASE_DIR, "amc/132/132_46.amc")
MODEL_XML = "/home/lvchen/miniconda3/envs/myosuite_raw/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml"  # 你的 myosuite xml
OBSD_XML = "/home/lvchen/miniconda3/envs/myosuite_raw/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml"
FPS = 120.0  # CMU mocap 一般 120Hz，如有不同请改
MAX_STEPS = 4000  # 仿真步数


# =========================================================
# 1. 从 AMC 读取 root 平移轨迹（只要 TX TY TZ）
# =========================================================
def load_cmu_root_from_amc(amc_path: str,
                           asf_length_scale: float = 0.45) -> np.ndarray:
    """
    从 CMU AMC 文件中读取 root 的平移轨迹 (TX, TY, TZ)，并转成“米”。

    AMC / ASF 约定：
      - ASF header 中 :units length 0.45
      - 解释：文件中的坐标需要先除以 0.45 才是英寸，再乘 2.54/100 变成米
      - 整体缩放因子 = (1 / 0.45) * 2.54 / 100 ≈ 0.056444

    返回：
        root_xyz_cmu_m: (N, 3) 的 numpy 数组，单位: 米，
                         坐标仍在「CMU 世界系 X右/Y上/Z前」下。
    """
    scale_to_m = (1.0 / asf_length_scale) * 2.54 / 100.0  # ≈ 0.056444

    root_list = []
    with open(amc_path, "r") as f:
        in_header = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            # AMC 前几行是 header，第一行 frame number 是纯数字
            if in_header:
                if line.isdigit():
                    in_header = False
                else:
                    continue

            # 到这里说明已经在 frame 数据区域，line 要么是 frame号，要么是 "bone data"
            if line.isdigit():
                # 新的一帧，从下一行找 root
                continue

            tokens = line.split()
            name = tokens[0].lower()
            if name == "root":
                # order: TX TY TZ RX RY RZ
                if len(tokens) < 4:
                    raise ValueError(f"AMC root 行格式异常: {line}")
                tx, ty, tz = map(float, tokens[1:4])
                pos = np.array([tx, ty, tz], dtype=float) * scale_to_m
                root_list.append(pos)

    if len(root_list) == 0:
        raise RuntimeError("在 AMC 中没有找到任何 root 数据")

    root_xyz_cmu_m = np.vstack(root_list)  # (N, 3)
    return root_xyz_cmu_m


# =========================================================
# 2. CMU → MuJoCo 世界系映射 (Y上/Z前 → Z上/Y前) + 平移对齐
# =========================================================
def map_cmu_root_to_mujoco(root_xyz_cmu_m: np.ndarray,
                           p_sim0_mj: np.ndarray,
                           t0_index: int = 0) -> np.ndarray:
    """
    将 CMU mocap 的 root 轨迹 (X右/Y上/Z前, 单位m)
    映射到 MuJoCo 世界系 (X右/Y前/Z上)，并与当前仿真 pelvis 初始位置对齐。

    Args:
        root_xyz_cmu_m: (N,3) CMU 世界中的 root 位置
        p_sim0_mj     : (3,) MuJoCo 仿真中 reset 后 pelvis 的世界位置
        t0_index      : 用 mocap 中第几帧对齐仿真初始位置

    Returns:
        root_xyz_mj_aligned: (N,3) 在 MuJoCo 世界下的参考 root 轨迹
    """
    root_xyz_cmu_m = np.asarray(root_xyz_cmu_m, dtype=float)
    assert root_xyz_cmu_m.shape[1] == 3
    p_sim0_mj = np.asarray(p_sim0_mj, dtype=float)

    # 坐标轴重排矩阵：
    # CMU: X右, Y上, Z前 -> MuJoCo: X右, Y前, Z上
    R_cmu2mj = np.array([
        [1.0, 0.0, 0.0],  # X_mj =  X_cmu
        [0.0, 0.0, 1.0],  # Y_mj =  Z_cmu
        [0.0, 1.0, 0.0],  # Z_mj =  Y_cmu
    ], dtype=float)

    # 线性变换到 MuJoCo
    root_xyz_mj_raw = root_xyz_cmu_m @ R_cmu2mj.T  # (N,3)

    # 以第 t0_index 帧对齐仿真初始姿态
    p0_mj = root_xyz_mj_raw[t0_index]
    offset = p_sim0_mj - p0_mj

    root_xyz_mj_aligned = root_xyz_mj_raw + offset[None, :]
    return root_xyz_mj_aligned


# =========================================================
# 3. 简单 root 参考轨迹采样器：按时间 t 取 mocap 的一帧
# =========================================================
class RootReference:
    def __init__(self, root_xyz_mj: np.ndarray, fps: float):
        self.traj = np.asarray(root_xyz_mj, dtype=float)
        assert self.traj.ndim == 2 and self.traj.shape[1] == 3
        self.fps = float(fps)
        self.N = self.traj.shape[0]
        self.T = self.N / self.fps  # 一个序列总时长 (s)

    def sample(self, t: float, loop: bool = True) -> np.ndarray:
        """
        给定时间 t（秒），返回对应的参考 root 位置 (3,)。

        loop=True 时按周期循环使用 mocap 序列。
        """
        frame = int(round(t * self.fps))
        if loop:
            frame = frame % self.N
        else:
            frame = min(frame, self.N - 1)
        return self.traj[frame]


# =========================================================
# 4. 主程序：仿真 + 实时绘制
# =========================================================
def main():
    # 1) 创建环境
    env = WalkEnvV5(model_path=MODEL_XML, obsd_model_path=OBSD_XML, seed=0)
    obs = env.reset()

    # 2) 获取仿真中 pelvis 初始位置（对齐用）
    pelvis_id = env.sim.model.body_name2id("pelvis")
    p_sim0 = env.sim.data.body_xpos[pelvis_id].copy()  # (3,)

    # 3) 从 AMC 读 root 轨迹并映射到 MuJoCo 世界坐标
    root_xyz_cmu_m = load_cmu_root_from_amc(AMC_PATH, asf_length_scale=0.45)
    root_xyz_mj = map_cmu_root_to_mujoco(root_xyz_cmu_m, p_sim0, t0_index=0)

    # 4) 建立 root 参考轨迹采样器
    root_ref = RootReference(root_xyz_mj, FPS)

    # 5) Matplotlib 实时绘图设置
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    # 预先根据 mocap 参考轨迹设置一下坐标范围
    x_ref = root_xyz_mj[:, 0]
    y_ref = root_xyz_mj[:, 1]
    margin = 0.2
    ax.set_xlim(x_ref.min() - margin, x_ref.max() + margin)
    ax.set_ylim(y_ref.min() - margin, y_ref.max() + margin)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Root 轨迹: 仿真 vs Mocap 参考")

    # 参考轨迹（整条曲线） + 当前点
    line_ref, = ax.plot(x_ref, y_ref, linestyle="--", linewidth=1.0, label="mocap root (ref)")
    point_ref, = ax.plot([], [], marker="x", markersize=6, linestyle="None", label="ref current")

    # 仿真 root 轨迹（实时累积） + 当前点
    line_sim, = ax.plot([], [], linewidth=2.0, label="sim root")
    point_sim, = ax.plot([], [], marker="o", markersize=6, linestyle="None", label="sim current")

    ax.legend(loc="best")

    sim_x, sim_y = [], []

    # 6) 仿真主循环
    for step in range(MAX_STEPS):
        t = float(env.sim.data.time)

        # 当前仿真 root 位置
        p_now = env.sim.data.body_xpos[pelvis_id].copy()  # (3,)
        sim_x.append(p_now[0])
        sim_y.append(p_now[1])

        # 当前 mocap 参考点
        p_ref = root_ref.sample(t, loop=True)  # (3,)

        # 更新曲线数据
        line_sim.set_data(sim_x, sim_y)
        point_sim.set_data([p_now[0]], [p_now[1]])

        point_ref.set_data([p_ref[0]], [p_ref[1]])

        plt.pause(0.001)  # 刷新图像

        # 这里先用零控制 / 随机控制占位；你可以替换成 policy.predict(obs) 的操作
        # action = env.action_space.sample()
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        obs, reward, done, info = env.step(action)

        # 如果你用 gymnasium，可以改成：
        # obs, reward, terminated, truncated, info = env.step(action)
        # done = terminated or truncated

        if done:
            obs = env.reset()
            sim_x.clear()
            sim_y.clear()

    plt.ioff()
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
