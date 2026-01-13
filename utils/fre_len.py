#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将目标前向速度 v_target (m/s) 换算为 CPG 中使用的 hip_period (steps)

计算逻辑（假设一个 gait cycle / stride 含两步）:
------------------------------------------------
记:
    L_step    : 一步(step)长度, m
    f_ctrl    : 控制频率, Hz
    hip_period: 一个 stride 周期内所用的 step 数
    Δt        : 每个控制步的时间 = 1 / f_ctrl
    v_target  : 目标平均前向速度, m/s

一个 stride 的时间:
    T_stride = hip_period * Δt

stride 频率:
    f_stride = 1 / T_stride = f_ctrl / hip_period

由于一个 stride 包含两步:
    f_step = 2 * f_stride = 2 * f_ctrl / hip_period

平均速度:
    v = L_step * f_step = L_step * (2 * f_ctrl / hip_period)

解出 hip_period:
    hip_period = 2 * L_step * f_ctrl / v_target
"""

import argparse


def compute_hip_period_from_step_length(
    v_target: float,
    L_step: float,
    f_ctrl: float,
) -> dict:
    """
    根据目标速度 v_target、一步长度 L_step 和控制频率 f_ctrl
    计算建议的 hip_period 及相关量。

    返回:
        {
            "hip_period": int,          # 建议取的整数步数
            "hip_period_float": float,  # 未取整前的理论值
            "f_stride": float,          # stride 频率 (Hz)
            "f_step": float,            # step 频率   (Hz)
            "v_est": float,             # 用整数 hip_period 反算得到的速度 (m/s)
        }
    """
    if v_target <= 0.0:
        raise ValueError("v_target 必须为正数 (m/s).")
    if L_step <= 0.0:
        raise ValueError("L_step 必须为正数 (m).")
    if f_ctrl <= 0.0:
        raise ValueError("f_ctrl 必须为正数 (Hz).")

    # 理论连续值
    hip_period_float = 2.0 * L_step * f_ctrl / v_target

    # 实际使用中必须是整数 step 数
    hip_period_int = int(round(hip_period_float))

    # 用整数 hip_period 反算 stride / step 频率与速度，方便检查量级
    f_stride = f_ctrl / hip_period_int          # 一个 gait cycle / stride 的频率
    f_step = 2.0 * f_stride                     # 两步一周期
    v_est = L_step * f_step                     # 用一步长度 + step频率反算出的速度

    return {
        "hip_period": hip_period_int,
        "hip_period_float": hip_period_float,
        "f_stride": f_stride,
        "f_step": f_step,
        "v_est": v_est,
    }


def main():
    parser = argparse.ArgumentParser(
        description="根据目标速度 v_target (m/s) 和一步长度 L_step (m) 计算 hip_period (steps)."
    )
    parser.add_argument(
        "--vel",
        type=float,
        default=1.0,
        help="目标前向速度 v_target (m/s)，默认 1.2",
    )
    parser.add_argument(
        "--step-len",
        type=float,
        default=0.493,
        help="几何上一‘步’的长度 L_step (m)，默认 0.8367（你从 2D 几何模型估出来的值）",
    )
    parser.add_argument(
        "--ctrl-freq",
        type=float,
        default=100.0,
        help="控制频率 f_ctrl (Hz)，默认 50.0",
    )

    args = parser.parse_args()

    v_target = args.vel
    L_step = args.step_len
    f_ctrl = args.ctrl_freq

    res = compute_hip_period_from_step_length(
        v_target=v_target,
        L_step=L_step,
        f_ctrl=f_ctrl,
    )

    print("==== vel → hip_period 换算结果 ====")
    print(f"  目标速度 v_target      : {v_target:.4f} m/s")
    print(f"  一步长度 L_step       : {L_step:.4f} m")
    print(f"  控制频率 f_ctrl       : {f_ctrl:.2f} Hz")
    print("")
    print(f"  理论 hip_period(浮点): {res['hip_period_float']:.4f} steps")
    print(f"  建议 hip_period(整数): {res['hip_period']} steps")
    print("")
    print("  采用该 hip_period 时的反算量级：")
    print(f"    stride 频率 f_stride: {res['f_stride']:.4f} Hz")
    print(f"    step   频率 f_step  : {res['f_step']:.4f} Hz")
    print(f"    估计速度 v_est      : {res['v_est']:.4f} m/s")
    print("==================================")


if __name__ == "__main__":
    main()
