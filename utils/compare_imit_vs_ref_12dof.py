# compare_imit_vs_ref_12dof.py
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 加入项目根路径，确保可导入 cpg.params_12
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.walk_v1 import WalkEnvV5
from cpg.params_12 import CMUParams


# ===== 用户需要修改：MuJoCo 模型路径 =====
# 换成你实际用的 myoLeg / myoLegWalk 模型 XML 路径
MODEL_XML = "/home/lvchen/miniconda3/envs/myosuite_raw/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml"  # TODO: 修改为你自己的 xml
OBSD_XML = None  # 如果有单独的 obsd_model_path，就填进去；否则保持 None


# 12 个需要对比的关节名字（MuJoCo 里的关节名）
JOINT_NAMES = [
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_l", "knee_angle_r",
    "ankle_angle_l", "subtalar_angle_l",
    "ankle_angle_r", "subtalar_angle_r",
]


def eval_joint_phase_from_cmu(cmu_params: CMUParams, joint_name: str, phase: float) -> float:
    """
    兼容两种情况：
    1）CMUParams 已经实现 eval_joint_phase(name, phase)
    2）CMUParams 只有 get_coeffs(name)，这里手动用傅立叶公式计算 q(phase)
    """
    # 情况 1：你已经按之前建议在 CMUParams 里写了 eval_joint_phase
    if hasattr(cmu_params, "eval_joint_phase"):
        return float(cmu_params.eval_joint_phase(joint_name, phase))

    # 情况 2：只有 get_coeffs，按 φ = 2π phase 手动计算
    a0, a, b = cmu_params.get_coeffs(joint_name)  # 支持 MuJoCo 关节名（走 alias_map）
    phase_wrapped = phase % 1.0
    k = np.arange(1, len(a) + 1, dtype=float)
    phi = 2.0 * np.pi * phase_wrapped * k
    return float(a0 + np.sum(a * np.cos(phi) + b * np.sin(phi)))


def main():
    # 1. 创建环境 & CMUParams
    env = WalkEnvV5(model_path=MODEL_XML, obsd_model_path=OBSD_XML, seed=0, min_height=0.0, max_rot=999.0, reset_type='init')
    cmu_params = CMUParams()  # 默认读取 cmu_12dofs_fourier.csv

    # 跑 N 个步态周期，用于对比
    n_cycles = 2
    n_steps = int(env.hip_period * n_cycles)

    # 时间轴（物理时间）
    dt = env.dt
    t = np.arange(n_steps) * dt

    # 2. 轨迹缓存
    ref_traj = {name: [] for name in JOINT_NAMES}  # mocap 参考（傅立叶）
    act_traj = {name: [] for name in JOINT_NAMES}  # 实际关节角
    phases = []

    # 3. reset 环境
    obs = env.reset()

    # 这里为了专注对比轨迹形状，先用“零动作”：
    # 如有训练好的策略，可以替换为 policy(obs)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    for step in range(n_steps):
        obs, reward, done, info = env.step(zero_action)

        # 当前相位：和奖励函数里保持一致
        phase_var = (env.steps / env.hip_period) % 1.0
        phases.append(phase_var)

        # 参考角度（傅立叶 + 相位）
        for jn in JOINT_NAMES:
            q_ref = eval_joint_phase_from_cmu(cmu_params, jn, phase_var)
            ref_traj[jn].append(q_ref)

        # 实际角度（MuJoCo 当前 qpos）
        q_cur = env._get_angle(JOINT_NAMES)  # 返回与 JOINT_NAMES 对应的当前角度
        for jn, q_val in zip(JOINT_NAMES, q_cur):
            act_traj[jn].append(float(q_val))

        # 如果你担心摔倒之后数据没意义，可以在 done 时 break
        # if done:
        #     print(f"Episode done at step {step}")
        #     break

    # 转成 numpy，方便绘图
    phases = np.array(phases)
    for jn in JOINT_NAMES:
        ref_traj[jn] = np.array(ref_traj[jn])
        act_traj[jn] = np.array(act_traj[jn])

    # 4. 画图：12 行，每行一个关节，实线 ref，虚线 actual
    n_joints = len(JOINT_NAMES)
    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2.5 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    for idx, jn in enumerate(JOINT_NAMES):
        ax = axes[idx]
        ax.plot(t, ref_traj[jn], label="ref (mocap Fourier)", linewidth=2)
        ax.plot(t, act_traj[jn], "--", label="actual (env)", linewidth=1)
        ax.set_ylabel(jn)
        ax.grid(True)
        if idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
