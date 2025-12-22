# -*- coding: utf-8 -*-
"""
params_12_bias.py

功能：
- 从 CSV 读取 12 个下肢关节的傅立叶系数 (a0, a1..a6, b1..b6)
- 为每个关节提供：
    self.<csv_name>_a0, self.<csv_name>_a, self.<csv_name>_b
    self.<mujoco_joint_name>_a0/_a/_b  (带别名与符号修正)
- 提供相位形式的评估函数（定义层面全局相位偏置）：
    q    = CMUParams.eval_joint_phase(joint_name, phase)
    qdot = CMUParams.eval_joint_phase_vel(joint_name, phase, phase_dot=...)

核心改动（相位定义层面全局偏置）：
- self.phase_bias 默认 0.75（即所有自由度相位向前偏置 0.75 个周期）
- 内部统一使用：
    phase_eff = (phase + phase_bias) % 1.0

使用方式（在环境里）：
    from cpg.params_12_bios import CMUParams
    cmu_params = CMUParams(csv_path=..., T=..., phase_bias=0.75)  # 默认已是 0.75

    phase_var = (self.steps / self.hip_period) % 1.0
    des = np.array([cmu_params.eval_joint_phase(jn, phase_var) for jn in imit_joint_names], dtype=np.float32)

    phase_dot = 1.0 / (hip_period * dt)  # cycle/s
    des_vel = np.array([cmu_params.eval_joint_phase_vel(jn, phase_var, phase_dot=phase_dot) for jn in imit_joint_names], dtype=np.float32)
"""

import csv
import numpy as np
from pathlib import Path


class CMUParams:
    def __init__(self,
                 csv_path: str = "cmu_12dofs_fourier.csv",
                 T: float = 1.0833,
                 phase_bias: float = 0.75):
        """
        csv_path  : 含 Fourier 系数的 CSV 文件路径
        T         : 步态周期（秒），这里只做记录；eval_joint_phase 用的是归一化相位
        phase_bias: 全局相位偏置（cycle，默认 0.75 表示整体向前偏置 3/4 周期）
        """
        self.T = float(T)
        # 推荐基频（rad/s），供按时间/按相位速度推导使用
        self.omega = 2.0 * np.pi / self.T

        # 全局相位偏置：统一作用于所有自由度
        self.phase_bias = float(phase_bias) % 1.0

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

    # ---------------- 公共接口：随时改偏置（可选） ----------------
    def set_phase_bias(self, phase_bias: float):
        """运行时更新全局相位偏置（cycle）。"""
        self.phase_bias = float(phase_bias) % 1.0

    # ---------------- 相位包装：统一应用偏置 ----------------
    def _wrap_phase(self, phase):
        """
        输入 phase（标量或数组），输出应用全局偏置后的 phase_eff，范围 [0,1)。
        """
        phase_arr = np.asarray(phase, dtype=float)
        return np.mod(phase_arr + self.phase_bias, 1.0)

    # --------- CSV 读取 ---------
    def _load_csv(self, csv_path: str):
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Fourier csv not found: {path}")

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["joint_name"].strip()
                a0 = float(row["a0"])

                # a1..a6, b1..b6
                a = []
                b = []
                # 兼容不同列名风格：a1/a2... 或 a_1/a_2...
                for k in range(1, 7):
                    if f"a{k}" in row:
                        a.append(float(row[f"a{k}"]))
                    else:
                        a.append(float(row.get(f"a_{k}", 0.0)))

                    if f"b{k}" in row:
                        b.append(float(row[f"b{k}"]))
                    else:
                        b.append(float(row.get(f"b_{k}", 0.0)))

                a = np.asarray(a, dtype=float)
                b = np.asarray(b, dtype=float)

                self.joint_names.append(name)
                self.a0_dict[name] = a0
                self.a_dict[name] = a
                self.b_dict[name] = b

                # 同时把它们挂到对象属性上，便于外部直接访问
                setattr(self, f"{name}_a0", a0)
                setattr(self, f"{name}_a", a)
                setattr(self, f"{name}_b", b)

    # --------- 构造 MuJoCo 关节名 alias（与你原 params_12.py 保持一致）---------
    def _build_mujoco_alias(self):
        """
        MuJoCo 关节名 -> CSV 关节名的映射。
        你可以在这里按需要扩展/修改映射关系。
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
        # 如果你发现某个自由度方向相反，可以在这里给 -1.0
        sign_map = {
            
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

        # 同步生成 “MuJoCo 名” 的 _a0/_a/_b 属性，便于外部直接用 mujoco 名访问
        for mj_name, csv_name in self.alias_map.items():
            if csv_name not in self.a0_dict:
                # CSV 里缺这个关节就跳过
                continue
            sign = float(self.alias_sign_map.get(mj_name, 1.0))
            setattr(self, f"{mj_name}_a0", sign * self.a0_dict[csv_name])
            setattr(self, f"{mj_name}_a",  sign * self.a_dict[csv_name])
            setattr(self, f"{mj_name}_b",  sign * self.b_dict[csv_name])

    # --------- 系数获取（支持 CSV 原名 / MuJoCo alias）---------
    def get_coeffs(self, joint_name: str):
        """
        返回 (a0, a, b)，其中 a/b 为 shape(K,) 的 numpy 数组。
        joint_name 支持：
        - CSV 原名（如 lfemur_0_pos）
        - MuJoCo 名（如 hip_flexion_l），会走 alias_map + sign_map
        """
        name = joint_name.strip()

        # 1) 直接是 CSV 名
        if name in self.a0_dict:
            return self.a0_dict[name], self.a_dict[name], self.b_dict[name]

        # 2) MuJoCo alias
        if name in self.alias_map:
            csv_name = self.alias_map[name]
            if csv_name not in self.a0_dict:
                raise KeyError(f"CSV joint not found for alias: {name} -> {csv_name}")
            sign = float(self.alias_sign_map.get(name, 1.0))
            return (sign * self.a0_dict[csv_name],
                    sign * self.a_dict[csv_name],
                    sign * self.b_dict[csv_name])

        raise KeyError(f"Unknown joint name: {name}")

    # --------- 相位形式评估函数（角度）---------
    def eval_joint_phase(self, joint_name: str, phase):
        """
        以“归一化相位” phase 计算关节角度（内部自动全局偏置）。

        φ = 2π * phase_eff
        q(φ) = a0 + Σ_k [ a_k cos(kφ) + b_k sin(kφ) ]

        返回：
            - 若 phase 为标量：返回 float
            - 若 phase 为数组：返回与 phase 同形状的 ndarray
        """
        a0, a, b = self.get_coeffs(joint_name)

        phase_arr = np.asarray(phase, dtype=float)
        phase_eff = self._wrap_phase(phase_arr)

        k = np.arange(1, len(a) + 1, dtype=float)               # (K,)
        phi = 2.0 * np.pi * phase_eff[..., None]                # (...,1)

        q = a0 + np.sum(
            a[None, :] * np.cos(k * phi) +
            b[None, :] * np.sin(k * phi),
            axis=-1
        )

        # 让标量输入返回标量（避免出现 shape=(1,) 的意外）
        if phase_arr.ndim == 0:
            return float(np.asarray(q).reshape(-1)[0])
        return q

    # --------- 相位形式评估函数（角速度）---------
    def eval_joint_phase_vel(self, joint_name: str, phase, phase_dot=None):
        """
        计算关节角速度 qdot（内部自动全局偏置）。

        若提供 phase_dot（cycle/s），则：
            dphi/dt = 2π * phase_dot
        否则默认使用 self.omega = 2π/T（rad/s）。

        qdot = Σ_k [ -a_k * k * dphi/dt * sin(kφ) + b_k * k * dphi/dt * cos(kφ) ]

        返回：
            - 若 phase 为标量：返回 float
            - 若 phase 为数组：返回与 phase 同形状的 ndarray
        """
        _, a, b = self.get_coeffs(joint_name)

        phase_arr = np.asarray(phase, dtype=float)
        phase_eff = self._wrap_phase(phase_arr)

        if phase_dot is None:
            dphi_dt = self.omega  # rad/s
        else:
            dphi_dt = 2.0 * np.pi * float(phase_dot)  # rad/s

        k = np.arange(1, len(a) + 1, dtype=float)              # (K,)
        phi = 2.0 * np.pi * phase_eff[..., None]               # (...,1)

        qdot = np.sum(
            (-a[None, :] * (k * dphi_dt) * np.sin(k * phi)) +
            ( b[None, :] * (k * dphi_dt) * np.cos(k * phi)),
            axis=-1
        )

        if phase_arr.ndim == 0:
            return float(np.asarray(qdot).reshape(-1)[0])
        return qdot
