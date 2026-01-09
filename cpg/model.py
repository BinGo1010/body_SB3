"""CPG-based hip torque reference generator (detailed comments)

本模块实现一个基于傅里叶基函数的髋关节扭矩 CPG（Central Pattern Generator），
用于生成左右髋关节的参考扭矩（tau_L, tau_R）。

设计要点（概要）：
- 使用论文/参数文件中给定的傅里叶系数 a0, a_k, b_k 构造基函数 base(phi)
- 基函数减去中心值 a0 得到零均值形状 (base - a0)，再由 amp（幅值，单位 Nm）缩放，
  并加上 offset（偏置，单位 Nm）得到最终扭矩参考：
    tau = offset + amp * (base(phi) - a0)
- amp、offset、步频 omega 使用二阶跟踪器（二阶线性系统）平滑地追踪外部目标值
- 双侧相位通过一个 π 锁相项耦合，以维持左右步态相位关系

注意：本文件仅添加注释，算法与原实现等价，所有数值更新方式（半隐式积分）保持不变。
"""

import numpy as np
from .params import PaperParams
# from params import PaperParams

class HipTorqueCPG:
    """
    髋关节扭矩参考生成器（CPG）

    主要输入/输出说明：
    - 输入（step 方法）：
        * Omega_target: 目标步频，单位 rad/s
        * amp_target: 目标幅值，标量或长度为2的数组（[amp_L, amp_R]），单位 Nm
        * offset_target: 目标偏置，标量或长度为2的数组（[off_L, off_R]），单位 Nm
    - 输出（step 返回字典）：
        * 'tau'   : numpy array, shape=(2,), [tau_L, tau_R]，单位 Nm
        * 'omega' : 当前步频，单位 rad/s
        * 'amp'   : 当前幅值数组（2,）
        * 'offset': 当前偏置数组（2,）
        * 'phi'   : 当前左右相位 [phi_L, phi_R]，单位 rad，范围 [0, 2π)

    实现细节与变量含义：
    - self.S_hip: 使用多少阶的傅里叶级数作为基函数（截断阶数）
    - self.gamma_*: 二阶跟踪器的阻尼/带宽参数（paper params 中给定）
    - self.v_pi: 相位耦合强度，用于实现左右 π 锁相
    - 半隐式（半隐式/半显式）积分：对位置态量（omega, amp, offset）采用
      x += dt * xdot + 0.5 * dt^2 * xddot; xdot += dt * xddot，这种形式在数值上
      比简单的显式欧拉更稳定一些
    """

    def __init__(self, params: PaperParams, dt: float = 0.002):
        """初始化 CPG

        参数:
        - params: PaperParams 包含傅里叶系数 a0,a,b 以及控制增益等超参数
        - dt: 仿真时间步长 (s)，用于积分更新
        """
        self.p = params
        self.dt = float(dt)

        # 使用的傅里叶阶数（可调，越高能表示越复杂的形状，但计算成本更高）
        self.S_hip = 8

        # 从参数中读取二阶跟踪器的增益（用于 omega / amp / offset 的加速度项）
        self.gamma_w = params.gamma_w    # 步频追踪增益
        self.gamma_amp = params.gamma_r  # 幅值追踪增益
        self.gamma_off = params.gamma_x  # 偏置追踪增益

        # 双侧相位耦合增益，用于维持左右 π 相位锁定
        self.v_pi = params.v_pi

        # 初始化状态变量
        self.reset()

    def reset(self):
        """重置内部状态到初始值。

        选择初始 omega（至少有一个下限以避免静止）和相位初值。
        amp/offset 默认为零（无扭矩）并且速度为零。
        """
        p = self.p

        # 初始步频（rad/s），使用参数文件中的基值但保证不低于 0.1
        self.omega = max(0.1, float(p.Omega_base))
        self.omegadot = 0.0

        # 左右相位的初值（rad）。这里选一个偏移使左右不同步开始
        self.phi_L = 2.0 + np.pi   # 左髋相位
        self.phi_R = 2.0           # 右髋相位

        # 幅值与偏置（及其一阶导数），单位为 Nm（扭矩）
        self.amp = np.zeros(2, float)      # [amp_L, amp_R]
        self.ampdot = np.zeros(2, float)
        self.offset = np.zeros(2, float)   # [off_L, off_R]
        self.offsetdot = np.zeros(2, float)

    def hip_fourier(self, phi: float) -> float:
        """计算给定相位 phi 的傅里叶基函数值。

        使用参数文件中的系数：
            base(phi) = a0 + sum_{k=1..S} [ a_k cos(k phi) + b_k sin(k phi) ]

        返回标量（角度或弧度相关的基形，具体取决于参数定义），
        注意：该函数返回的是原始基函数值，用于后续与 a0 相减得到零均值形状。
        """
        p = self.p
        S = int(np.clip(self.S_hip, 1, len(p.hip_a)))
        a0 = p.hip_a0
        a = p.hip_a[:S]
        b = p.hip_b[:S]
        l = np.arange(1, S + 1, dtype=float)
        return float(a0 + np.sum(a * np.cos(l * phi) + b * np.sin(l * phi)))

    def step(self, Omega_target: float, amp_target, offset_target):
        """按一个仿真步长更新 CPG 的内部状态并输出扭矩参考。

        算法步骤（按顺序）：
        1. 将目标 amp/offset 广播为长度为2的数组（左右）
        2. 使用局部定义的二阶跟踪器 `track2` 计算 omega/amp/offset 的加速度（xddot）
           track2(x, xdot, x_star, gamma) = gamma * (0.25 * gamma * (x_star - x) - xdot)
           该公式来源于二阶线性系统的状态空间写法，作用是快速且平滑地将状态追踪到目标值
        3. 数值防御：将非有限（inf/nan）加速度归零
        4. 半隐式积分更新（位置态量使用 x += dt * xdot + 0.5 * dt^2 * xddot，速度使用显式欧拉）
        5. 计算相位导数 phidot_L/ phidot_R：基础角速度 + π 锁相耦合项
        6. 更新相位并取模到 [0, 2π)
        7. 用当前相位调用傅里叶基函数生成 base_L/base_R，并以 a0 为中心生成零均值波形
        8. 由 offset + amp * shape 得到左右扭矩输出

        返回值：包含 tau、omega、amp、offset、phi 的字典
        """
        dt = self.dt

        # ----------------- 目标参数（广播） -----------------
        amp_target = np.asarray(amp_target, float)
        offset_target = np.asarray(offset_target, float)

        if amp_target.size == 1:
            amp_target = np.repeat(amp_target, 2)
        if offset_target.size == 1:
            offset_target = np.repeat(offset_target, 2)

        # ----------------- 二阶跟踪器（计算加速度） -----------------
        def track2(x, xdot, x_star, gamma):
            """二阶跟踪器产生加速度 xddot。

            形式： xddot = gamma * (0.25 * gamma * (x_star - x) - xdot)
            该表达式等价于将期望极点设置在 -0.5*gamma（带有一定阻尼），用于快速但稳定的追踪。
            对向量 x（如 amp）能进行逐元素运算。
            """
            return gamma * (0.25 * gamma * (x_star - x) - xdot)

        # 步频（标量）加速度
        omegaddot = track2(self.omega, self.omegadot, float(Omega_target), self.gamma_w)
        # 幅值 / 偏置（向量）加速度
        ampddot = track2(self.amp, self.ampdot, amp_target, self.gamma_amp)
        offddot = track2(self.offset, self.offsetdot, offset_target, self.gamma_off)

        # ----------------- 数值防御 -----------------
        # 防止 NaN / Inf 传播
        if not np.isfinite(omegaddot):
            omegaddot = 0.0
        ampddot[~np.isfinite(ampddot)] = 0.0
        offddot[~np.isfinite(offddot)] = 0.0

        # ----------------- 半隐式积分（提高数值稳定性） -----------------
        # 位置类量（omega, amp, offset）采用半隐式更新：x += dt * xdot + 0.5 * dt^2 * xddot
        # 速度类量（xdot）采用显式欧拉更新：xdot += dt * xddot
        self.omega += dt * self.omegadot + 0.5 * (dt ** 2) * omegaddot
        self.omegadot += dt * omegaddot

        self.amp += dt * self.ampdot + 0.5 * (dt ** 2) * ampddot
        self.ampdot += dt * ampddot

        self.offset += dt * self.offsetdot + 0.5 * (dt ** 2) * offddot
        self.offsetdot += dt * offddot

        # ----------------- 相位耦合（π 锁相） -----------------
        # 基本相速度为 self.omega，v_pi 项实现左右相位互锁，使左右相位接近相差 π
        vpi = self.v_pi
        phidot_L = self.omega + vpi * np.sin(self.phi_R - self.phi_L - np.pi)
        phidot_R = self.omega + vpi * np.sin(self.phi_L - self.phi_R - np.pi)

        # 对相位积分并将范围限制在 [0, 2π)
        self.phi_L = float(np.mod(self.phi_L + dt * phidot_L, 2 * np.pi))
        self.phi_R = float(np.mod(self.phi_R + dt * phidot_R, 2 * np.pi))

        # ----------------- 生成扭矩波形 -----------------
        base_L = self.hip_fourier(self.phi_L)
        base_R = self.hip_fourier(self.phi_R)
        # 将 a0 作为中心值，使用 (base - a0) 得到零均值形状
        center = self.p.hip_a0

        shape_L = base_L - center
        shape_R = base_R - center

        # 最终扭矩：offset + amp * shape
        tau_L = self.offset[0] + self.amp[0] * shape_L
        tau_R = self.offset[1] + self.amp[1] * shape_R

        return {
            "tau": np.array([tau_L, tau_R], dtype=float),  # [τ_L, τ_R] (Nm)
            "omega": float(self.omega),
            "amp": self.amp.copy(),
            "offset": self.offset.copy(),
            "phi": np.array([self.phi_L, self.phi_R], dtype=float),
        }
    

    
# ============================
# Angle-trajectory CPG (Fourier) + time-derivative
# ============================

class HipAngleCPG:
    """与 HipTorqueCPG 相同的状态更新/相位耦合，但输出为髋关节**角度轨迹**(q_des, dq_des)。

    输出：
      q_des  : [q_L, q_R]  (度)
      dq_des : [dq_L, dq_R] (度/s)

    说明：
    - 仍使用参数文件中的傅里叶基函数 base(phi)，并以 a0 为中心值得到零均值形状：
        shape(phi) = base(phi) - a0
    - 角度轨迹定义为：
        q_des = offset + amp * shape(phi)
      其中 amp、offset 的物理单位应按“角度尺度”配置（rad），不再是 Nm。
    - dq_des 通过解析求导得到：
        dbase/dphi = sum_k [ -k a_k sin(k phi) + k b_k cos(k phi) ]
        dshape/dphi = dbase/dphi
        dq/dt = amp * dshape/dphi * (dphi/dt)
      这里 dphi/dt 取相位更新时的 phidot（rad/s）。
    """

    def __init__(self, params: PaperParams, dt: float = 0.02, phase0: float = 0.0, S_hip: int = 6):
        self.p = params
        self.dt = float(dt)
        self.S_hip = int(S_hip)
        

        # 相位
        self.phi_L = float(phase0)
        self.phi_R = float(phase0 + np.pi)

        # 频率/尺度（单位由外部约定：omega 为 rad/s；amp、offset 为 rad）
        self.omega = 6.0  # rad/s, 约 1Hz
        self.amp = np.array([1.0, 1.0], dtype=float)      # rad
        self.offset = np.array([0, 0], dtype=float)   # rad

        # 二阶跟踪器状态（同 HipTorqueCPG）
        self.omega_x = 0.0
        self.omega_dx = 0.0
        self.amp_x = np.zeros(2, dtype=float)
        self.amp_dx = np.zeros(2, dtype=float)
        self.offset_x = np.zeros(2, dtype=float)
        self.offset_dx = np.zeros(2, dtype=float)

        # 跟踪器参数（可按需调整）
        self.omega_wn = 8.0
        self.omega_zeta = 1.0

        self.amp_wn = 8.0
        self.amp_zeta = 1.0

        self.offset_wn = 8.0
        self.offset_zeta = 1.0

        self.phase_bias_wn = 8.0
        self.phase_bias_zeta = 1.0

        self.phase_bias = 0.425  # cycle 0.1525+0.25(rad)
        self.phase_bias_dx = 0.0       # 若用二阶跟踪

        # 相位耦合（左右锁相）
        self.k_couple = 10.0

    def reset(self, phase0: float = 0.0):
        self.phi_L = float(phase0)
        self.phi_R = float(phase0 + np.pi)

    def _track2(self, x, dx, x_des, wn, zeta):
        # 二阶系统：ddx = wn^2 (x_des - x) - 2 zeta wn dx
        ddx = (wn * wn) * (x_des - x) - 2.0 * zeta * wn * dx
        dx = dx + ddx * self.dt
        x = x + dx * self.dt
        return x, dx

    def hip_fourier(self, phi: float) -> float:
        p = self.p
        S = int(np.clip(self.S_hip, 1, len(p.hip_a)))
        a0 = p.hip_a0
        a = p.hip_a[:S]
        b = p.hip_b[:S]
        l = np.arange(1, S + 1, dtype=float)
        return float(a0 + np.sum(a * np.cos(l * phi) + b * np.sin(l * phi)))

    def hip_fourier_dphi(self, phi: float) -> float:
        """dbase/dphi"""
        p = self.p
        S = int(np.clip(self.S_hip, 1, len(p.hip_a)))
        a = p.hip_a[:S]
        b = p.hip_b[:S]
        l = np.arange(1, S + 1, dtype=float)
        return float(np.sum((-l) * a * np.sin(l * phi) + (l) * b * np.cos(l * phi)))

    def step(self, Omega_target, amp_target, offset_target, phase_bias_target):
        # 1) 二阶跟踪：omega/amp/offset
        self.omega, self.omega_dx = self._track2(
            self.omega, self.omega_dx, float(Omega_target), self.omega_wn, self.omega_zeta
        )

        amp_target = np.asarray(amp_target, dtype=float).reshape(2)
        offset_target = np.asarray(offset_target, dtype=float).reshape(2)

        for i in range(2):
            self.amp[i], self.amp_dx[i] = self._track2(
                self.amp[i], self.amp_dx[i], float(amp_target[i]), self.amp_wn, self.amp_zeta
            )
            self.offset[i], self.offset_dx[i] = self._track2(
                self.offset[i], self.offset_dx[i], float(offset_target[i]), self.offset_wn, self.offset_zeta
            )
        # 相位偏置二阶跟踪
        self.phase_bias, self.phase_bias_dx = self._track2(
            self.phase_bias, self.phase_bias_dx, float(phase_bias_target), self.phase_bias_wn, self.phase_bias_zeta
        )
        # 2) 相位耦合（锁相到 pi）
        delta = (self.phi_R - self.phi_L) - np.pi
        delta = (delta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        couple_term = self.k_couple * delta

        # 3) 相位积分（phidot 为 rad/s）
        phidot_L = self.omega + couple_term
        phidot_R = self.omega - couple_term
        self.phi_L = (self.phi_L + phidot_L * self.dt) % (2 * np.pi)
        self.phi_R = (self.phi_R + phidot_R * self.dt) % (2 * np.pi)

        # 4) 角度轨迹 q_des / dq_des
        center = self.p.hip_a0

        # 用 cycle 表示的相位偏置（推荐外部设置 self.phase_bias = 0.152）

        phi_bias = 2.0 * np.pi * float(self.phase_bias)

        # 若你的 Δ* 定义为 “phase -> phase + Δ”，用 +；否则用 -
        phiL_eff = (self.phi_L + phi_bias) % (2.0 * np.pi)
        phiR_eff = (self.phi_R + phi_bias) % (2.0 * np.pi)

        base_L = self.hip_fourier(phiL_eff)
        base_R = self.hip_fourier(phiR_eff)
        dbase_L = self.hip_fourier_dphi(phiL_eff)
        dbase_R = self.hip_fourier_dphi(phiR_eff)


        shape_L = base_L - center
        shape_R = base_R - center

        q_L = self.offset[0] + self.amp[0] * shape_L
        q_R = self.offset[1] + self.amp[1] * shape_R

        dq_L = self.amp[0] * dbase_L * phidot_L
        dq_R = self.amp[1] * dbase_R * phidot_R

        return {
            "q_des": np.array([q_L, q_R], dtype=float),
            "dq_des": np.array([dq_L, dq_R], dtype=float),
            "omega": float(self.omega),
            "amp": self.amp.copy(),
            "offset": self.offset.copy(),
            "phi": np.array([self.phi_L, self.phi_R], dtype=float),
            "phidot": np.array([phidot_L, phidot_R], dtype=float),
            "delta": float(delta),
            "k_couple": float(self.k_couple),
        }