#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 python "utils/init_posvel copy.py" --viz viewer --realtime --speed 0.2

init_posvel.py  (基于 view_gait_12 的步态)

新增功能：
- --viz none|viewer|video
  * viewer: 打开 MuJoCo 窗口，实时播放“写入qpos并记录初始状态”的过程
  * video : 离屏渲染输出 MP4，便于回看/发给别人
- --show_samples: 在生成完毕后，把最终抽样出的 N_RESET_STATES 帧逐帧展示（可选）
- qvel 计算优先使用 mj_differentiatePos（支持 freejoint/quaternion 的正确差分）

CSV 每一行格式:
    time, qpos_0, ..., qpos_{nq-1}, qvel_0, ..., qvel_{nv-1}
"""

import os
import csv
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import mujoco

# 添加项目根路径，确保可导入 cpg.params_12
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cpg.params_12_bias import CMUParams  # 与 view_gait_12.py 保持一致

# 12 个 DOF 名称（建议和 view_gait_12.py 中完全一致）
JOINT_NAMES = [
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_l", "knee_angle_r",
    "ankle_angle_l", "subtalar_angle_l",
    "ankle_angle_r", "subtalar_angle_r",
]


def fourier_eval_time(t: float, a0: float, a: np.ndarray, b: np.ndarray, omega: float) -> float:
    """
    与 view_gait_12.py 同形的傅里叶评估：
        q(t) = a0 + Σ_{k=1..K} [ a_k cos(k * omega * t) + b_k sin(k * omega * t) ]
    """
    k = np.arange(1, len(a) + 1, dtype=float)
    theta = omega * t
    return a0 + np.sum(a * np.cos(k * theta) + b * np.sin(k * theta))


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--coeff_csv", type=str, default=None,
                   help="Fourier 系数 CSV 路径（默认：PROJECT_ROOT/cpg/fourier_coefficients_order6_132.csv）")
    p.add_argument("--xml", type=str, default=None,
                   help="MuJoCo 模型 XML 路径（默认：你脚本原来的 XML_PATH）")
    p.add_argument("--out_csv", type=str, default=None,
                   help="输出 reset_states_from_mujoco.csv 路径（默认：当前 utils 目录）")

    p.add_argument("--gait_T", type=float, default=3.0, help="步态周期(s)")
    p.add_argument("--n_cycles", type=int, default=2, help="采样周期数")
    p.add_argument("--dt", type=float, default=1.0/120.0, help="采样 dt")
    p.add_argument("--n_reset_states", type=int, default=32, help="最终抽样帧数")

    # 可视化
    p.add_argument("--viz", type=str, default="viewer", choices=["none", "viewer", "video"],
                   help="可视化方式：none|viewer|video")
    p.add_argument("--realtime", action="store_true",
                   help="viewer 模式下按真实时间播放（默认：尽快跑完）")
    p.add_argument("--speed", type=float, default=1.0,
                   help="viewer 实时播放倍速（realtime打开时生效，>1更快）")

    # video 参数
    p.add_argument("--video_path", type=str, default="init_posvel_playback.mp4", help="输出 MP4 文件名/路径")
    p.add_argument("--fps", type=int, default=30, help="视频 FPS")
    p.add_argument("--width", type=int, default=1280, help="视频宽")
    p.add_argument("--height", type=int, default=720, help="视频高")
    p.add_argument("--camera", type=str, default=None, help="相机名（可选，不填用默认自由相机）")

    # 初始姿态
    p.add_argument("--use_keyframe0", action="store_true",
                   help="若模型有 keyframe，优先用 keyframe0 作为初始姿态（推荐）")

    # 抽样帧展示
    p.add_argument("--show_samples", action="store_true",
                   help="生成结束后，把抽样出来的 reset 帧逐帧展示（仅 viewer 模式有效）")
    p.add_argument("--sample_hold", type=float, default=0.25, help="show_samples 每帧停留秒数")

    return p


def maybe_reset_to_keyframe0(model, data, use_keyframe0: bool):
    if use_keyframe0 and getattr(model, "nkey", 0) > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)


def main():
    args = build_parser().parse_args()

    # 工程根目录（假设 init_posvel.py 在 body_SB3/utils 下）
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 默认路径（保持你原脚本习惯）
    default_coeff_csv = os.path.join(PROJECT_ROOT, "cpg", "fourier_coefficients_order6_132.csv")
    default_xml = "/home/lvchen/miniconda3/envs/myosuite_raw/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml"
    default_out_csv = os.path.join(os.path.dirname(__file__), "reset_states_from_mujoco.csv")

    coeff_csv = args.coeff_csv or default_coeff_csv
    xml_path = args.xml or default_xml
    out_csv = args.out_csv or default_out_csv

    # 1) 读取 12 关节的傅里叶参数
    if not Path(coeff_csv).is_file():
        raise FileNotFoundError(f"找不到 Fourier 系数文件: {coeff_csv}")
    params = CMUParams(csv_path=coeff_csv, T=args.gait_T)

    # 2) 载入 MuJoCo 模型
    if not Path(xml_path).is_file():
        raise FileNotFoundError(f"找不到 XML: {xml_path}\n请用 --xml 指定实际路径。")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    maybe_reset_to_keyframe0(model, data, args.use_keyframe0)

    nq = model.nq
    nv = model.nv
    print(f"[Info] nq={nq}, nv={nv}")

    # 3) 关节名 -> qpos index
    qpos_adr = {}
    for name in JOINT_NAMES:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id < 0:
            raise RuntimeError(f"关节名 {name} 在模型中找不到，请检查 XML 和 JOINT_NAMES 是否一致。")
        qpos_adr[name] = int(model.jnt_qposadr[j_id])

    # 4) 时间轴 & 预分配
    sim_duration = float(args.n_cycles) * float(args.gait_T)
    n_steps = int(np.round(sim_duration / float(args.dt)))
    dt = float(args.dt)

    time_arr = np.zeros(n_steps, dtype=float)
    qpos_arr = np.zeros((n_steps, nq), dtype=float)

    # ========== 可视化准备 ==========
    viewer = None
    renderer = None
    writer = None
    stride = 1

    # if args.viz == "viewer":
    #     import mujoco.viewer
    #     viewer = mujoco.viewer.launch_passive(model, data)
    if args.viz == "viewer":
        import mujoco.viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)

    elif args.viz == "video":
        try:
            from mujoco import Renderer
        except Exception as e:
            raise RuntimeError("你的 mujoco python 版本缺少 Renderer，建议升级 mujoco>=2.3。") from e

        try:
            import imageio.v2 as imageio
        except Exception as e:
            raise RuntimeError("缺少 imageio：请先 `pip install imageio imageio-ffmpeg`") from e

        renderer = Renderer(model, width=args.width, height=args.height)
        writer = imageio.get_writer(args.video_path, fps=int(args.fps), codec="libx264")

        # 按 fps 抽帧，保证视频时长接近 sim_duration
        stride = max(1, int(round(1.0 / (dt * float(args.fps)))))

    # 5) 遍历时间：写入 12 关节角度 -> mj_forward -> 记录
    t = 0.0
    try:
        for i in range(n_steps):
            time_arr[i] = t

            # 写入 12 个关节的傅里叶轨迹
            for name in JOINT_NAMES:
                a0 = getattr(params, f"{name}_a0")
                a = getattr(params, f"{name}_a")
                b = getattr(params, f"{name}_b")
                q_des = fourier_eval_time(t, a0, a, b, params.omega)

                idx = qpos_adr[name]
                data.qpos[idx] = q_des

            # 只做几何前向（本脚本的“初始状态记录”就是这种 kinematic replay）
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            # 记录整条 qpos
            qpos_arr[i, :] = data.qpos[:]

            # --- 展示 ---
            if args.viz == "viewer":
                viewer.sync()
                if args.realtime:
                    time.sleep(max(0.0, dt / max(1e-6, args.speed)))

            elif args.viz == "video":
                if (i % stride) == 0:
                    if args.camera is None:
                        renderer.update_scene(data)
                    else:
                        renderer.update_scene(data, camera=args.camera)
                    frame = renderer.render()
                    writer.append_data(frame)

            t += dt

    finally:
        if args.viz == "viewer" and viewer is not None:
            # 不强制关闭：给你机会在结束画面停留查看
            pass
        if args.viz == "video" and writer is not None:
            writer.close()
            print(f"[Save] 已输出视频: {args.video_path}")

    # 6) 用更稳健的方法计算 qvel（支持 freejoint/quaternion）
    qvel_arr = np.zeros((n_steps, nv), dtype=float)
    if nv > 0:
        has_diff = hasattr(mujoco, "mj_differentiatePos")
        for i in range(1, n_steps):
            if has_diff:
                mujoco.mj_differentiatePos(model, qvel_arr[i], dt, qpos_arr[i], qpos_arr[i - 1])
            else:
                # 兼容极旧版本：退化为简单差分（freejoint时不严格）
                qvel_arr[i, :] = (qpos_arr[i, :nv] - qpos_arr[i - 1, :nv]) / dt
        qvel_arr[0, :] = qvel_arr[1, :]

    # 7) 抽样 N_RESET_STATES 帧
    if n_steps <= args.n_reset_states:
        indices = np.arange(n_steps, dtype=int)
    else:
        indices = np.linspace(0, n_steps - 1, int(args.n_reset_states), dtype=int)

    reset_qpos = qpos_arr[indices]
    reset_qvel = qvel_arr[indices]
    reset_time = time_arr[indices]
    K = reset_qpos.shape[0]

    # 8) 写 CSV
    header = (["time"] + [f"qpos_{i}" for i in range(nq)] + [f"qvel_{i}" for i in range(nv)])
    with open(out_csv, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(header)
        for k in range(K):
            row = [reset_time[k]] + reset_qpos[k].tolist() + reset_qvel[k].tolist()
            writer_csv.writerow(row)

    print(f"[Save] 已保存 {K} 帧步态状态至 {out_csv}")
    print(f"        每行: 1(time) + {nq}(qpos) + {nv}(qvel) 列")

    # 9) 可选：把抽样出来的 reset 帧逐帧展示
    if args.viz == "viewer" and args.show_samples and viewer is not None:
        print("[Show] 逐帧展示抽样 reset states ...")
        for k, idx in enumerate(indices):
            data.qpos[:] = qpos_arr[idx]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(max(0.0, float(args.sample_hold)))

        print("[Show] 完成。你可手动关闭窗口。")

    # 结束后保持窗口（避免瞬间退出）
    if args.viz == "viewer" and viewer is not None:
        print("[Info] viewer 播放结束，窗口保持打开（Ctrl+C 退出脚本）。")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
