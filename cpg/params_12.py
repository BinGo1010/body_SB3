# -*- coding: utf-8 -*-
"""
params_12.py

功能：
- 从 CSV 读取 12 个下肢关节的傅立叶系数 (a0, a1..a6, b1..b6)
- 为每个关节提供：
    self.<csv_name>_a0, self.<csv_name>_a, self.<csv_name>_b
    self.<mujoco_joint_name>_a0/_a/_b  (带别名与符号修正)
- 提供相位形式的评估函数：
    q = CMUParams.eval_joint_phase(joint_name, phase)
    其中 phase ∈ [0,1)，φ = 2π phase

使用方式（在环境里）：
    phase_var = (self.steps / self.hip_period) % 1.0
    hip_l = params.eval_joint_phase("hip_flexion_l", phase_var)
"""

import csv
from pathlib import Path
import numpy as np


class CMUParams:
    """
    从 CSV 读取 12 个关节的傅立叶系数：
    CSV 列格式：
        joint_name,a0,a1,a2,a3,a4,a5,a6,b1,b2,b3,b4,b5,b6

    读取后存为：
        self.<csv_name>_a0
        self.<csv_name>_a   (np.array[6])
        self.<csv_name>_b   (np.array[6])

    同时按 MuJoCo 关节名建立别名：
        hip_flexion_l -> lfemur_0_pos
        ...

    并提供相位评估函数：
        q = eval_joint_phase("hip_flexion_l", phase)
    """

    def __init__(self,
                 csv_path: str = "cmu_12dofs_fourier.csv",
                 T: float = 1.0833):
        """
        csv_path : 傅立叶系数 CSV 路径
        T        : 步态周期（秒），这里只做记录，eval_joint_phase 用的是相位
        """
        self.T = float(T)
        # 这里保留一个“推荐基频”供其他地方需要按时间使用
        self.omega = 2.0 * np.pi / self.T

        # 原始 CSV 名对应的系数字典
        self.joint_names = []   # CSV 中的 joint_name 列
        self.a0_dict = {}
        self.a_dict = {}
        self.b_dict = {}

        # 别名映射（MuJoCo 关节名 -> CSV 名）与符号表
        self.alias_map = {}
        self.alias_sign_map = {}

        # 读 CSV + 构造别名
        self._load_csv(csv_path)
        self._build_mujoco_alias()

    # --------- CSV 读取 ---------
    def _load_csv(self, csv_path: str):
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Fourier csv not found: {path}")

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["joint_name"].strip()
                self.joint_names.append(name)

                a0 = float(row["a0"])
                a_list = [float(row[f"a{k}"]) for k in range(1, 7)]
                b_list = [float(row[f"b{k}"]) for k in range(1, 7)]

                a = np.array(a_list, dtype=float)
                b = np.array(b_list, dtype=float)

                self.a0_dict[name] = a0
                self.a_dict[name] = a
                self.b_dict[name] = b

                # 按 CSV 名挂一份属性，保持“原始系数”
                setattr(self, f"{name}_a0", a0)
                setattr(self, f"{name}_a", a)
                setattr(self, f"{name}_b", b)

    # --------- MuJoCo 关节名别名映射 + 符号修正 ---------
    def _build_mujoco_alias(self):
        """
        将 CSV 里的关节名映射到 MuJoCo 里的 12 个关节：

        JOINT_NAMES = [
            "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
            "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
            "knee_angle_l", "knee_angle_r",
            "ankle_angle_l", "subtalar_angle_l",
            "ankle_angle_r", "subtalar_angle_r",
        ]
        """
        alias_map = {
            # 左髋 3 自由度
            "hip_flexion_l":   "lfemur_0_pos",
            "hip_adduction_l": "lfemur_1_pos",
            "hip_rotation_l":  "lfemur_2_pos",

            # 右髋 3 自由度
            "hip_flexion_r":   "rfemur_0_pos",
            "hip_adduction_r": "rfemur_1_pos",
            "hip_rotation_r":  "rfemur_2_pos",

            # 膝：左膝 -> ltibia_0_pos, 右膝 -> rtibia_0_pos
            "knee_angle_l": "ltibia_0_pos",
            "knee_angle_r": "rtibia_0_pos",

            # 踝 + 跖下
            "ankle_angle_l":    "lfoot_0_pos",
            "subtalar_angle_l": "lfoot_1_pos",
            "ankle_angle_r":    "rfoot_0_pos",
            "subtalar_angle_r": "rfoot_1_pos",
        }
        self.alias_map = alias_map

        # 需要符号翻转的关节（MuJoCo 名）
        sign_map = {
            # 01
            # "hip_flexion_l": -1.0,
            # "hip_adduction_l": 0.8,
            # "hip_rotation_l": 1.0,
            # "hip_flexion_r": -1.0,
            # "hip_adduction_r": -0.8,
            # "hip_rotation_r": -1.0,

            # 22
            # "hip_flexion_l": -1.0,
            # "hip_adduction_l": 0.8,
            # "hip_rotation_l": 1.0,
            # "hip_flexion_r": -1.0,
            # "hip_adduction_r": 0.8,
            # "hip_rotation_r": -1.0,
            
            # 132
            "hip_flexion_l": -1.0,
            "hip_adduction_l": -1.0,
            "hip_rotation_l": 1.0,
            "ankle_angle_l": -1.0,

            "hip_flexion_r": -1.0,
            "hip_adduction_r": 1.0,
            "hip_rotation_r": -1.0,
            "ankle_angle_r": -1.0,
            "subtalar_angle_r": -1.0,
        }
        self.alias_sign_map = sign_map

        # 为每个别名挂上“符号修正后的”属性
        for mj_name, csv_name in alias_map.items():
            if csv_name not in self.a0_dict:
                raise KeyError(f"Alias '{mj_name}' -> '{csv_name}' not in CSV")

            sign = sign_map.get(mj_name, 1.0)

            a0_raw = self.a0_dict[csv_name]
            a_raw = self.a_dict[csv_name]
            b_raw = self.b_dict[csv_name]

            a0 = sign * a_raw[0] * 0 + sign * a0_raw  # 清晰一点: 只对 a0 乘 sign
            a = sign * a_raw
            b = sign * b_raw

            setattr(self, f"{mj_name}_a0", a0)
            setattr(self, f"{mj_name}_a",  a)
            setattr(self, f"{mj_name}_b",  b)

    # --------- 通用接口：拿系数 ---------
    def get_coeffs(self, name: str):
        """
        返回 (a0, a, b)

        name 可以是：
        - CSV 里的原始 joint_name，例如 "lfemur_0_pos"
          -> 返回原始 a0,a,b（不带符号修正）
        - MuJoCo 关节名，例如 "hip_flexion_l"
          -> 返回带 alias + 符号修正后的 a0,a,b
        """
        # 先按 CSV 原名
        if name in self.a0_dict:
            return (self.a0_dict[name],
                    self.a_dict[name],
                    self.b_dict[name])

        # 再按别名（MuJoCo 名）
        if name in self.alias_map:
            csv_name = self.alias_map[name]
            sign = self.alias_sign_map.get(name, 1.0)
            return (sign * self.a0_dict[csv_name],
                    sign * self.a_dict[csv_name],
                    sign * self.b_dict[csv_name])

        raise KeyError(f"Unknown joint name: {name}")

    # --------- 相位形式评估函数（重点）---------
    def eval_joint_phase(self, joint_name: str, phase):
        """
        以“归一化相位” phase 计算关节角度。

        参数：
            joint_name : str
                可以是 MuJoCo 关节名（如 "hip_flexion_l"）
                或 CSV 原名（如 "lfemur_0_pos"）
            phase : float 或 np.ndarray
                归一化步态相位，通常定义为
                    phase_var = (steps / hip_period) % 1.0
                取值在 [0,1) 内表示一个周期。

        计算：
            φ = 2π * phase
            q(φ) = a0 + Σ_k [ a_k cos(kφ) + b_k sin(kφ) ]
        """
        a0, a, b = self.get_coeffs(joint_name)

        phase_arr = np.asarray(phase, dtype=float)
        # 保证在 [0,1) 内
        phase_wrapped = np.mod(phase_arr, 1.0)

        k = np.arange(1, len(a) + 1, dtype=float)  # (K,)
        phi = 2.0 * np.pi * phase_wrapped[..., None]  # (..., 1) -> (..., K)

        # 广播计算，返回形状与 phase 相同
        q = a0 + np.sum(
            a[None, :] * np.cos(k * phi) +
            b[None, :] * np.sin(k * phi),
            axis=-1
        )
        return q

    def eval_joint_phase_vel(self, joint_name: str, phase, phase_dot: float = None):
        """
        返回参考角速度 qdot（rad/s）
        phase     : 归一化相位 [0,1)
        phase_dot : d(phase)/dt (cycle/s). 若不传，默认 1/T
        """
        a0, a, b = self.get_coeffs(joint_name)

        phase_arr = np.asarray(phase, dtype=float)
        phase_wrapped = np.mod(phase_arr, 1.0)

        # dphi/dt
        if phase_dot is None:
            dphi_dt = self.omega          # 2π/T  (rad/s)，params_12.py 里已有 self.omega
        else:
            dphi_dt = 2.0 * np.pi * float(phase_dot)

        k = np.arange(1, len(a) + 1, dtype=float)              # (K,)
        phi = 2.0 * np.pi * phase_wrapped[..., None]           # (...,K)

        qdot = np.sum(
            (-a[None, :] * (k * dphi_dt) * np.sin(k * phi)) +
            ( b[None, :] * (k * dphi_dt) * np.cos(k * phi)),
            axis=-1
        )
        return qdot
