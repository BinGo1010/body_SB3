# -*- coding: utf-8 -*-
import numpy as np

class PaperParams:
    """参数与边界（含多谐波系数）"""
    def __init__(self, use_small_margin=True):
        # 二阶跟踪增益
        self.gamma_w = 22.0
        self.gamma_r = 22.0
        self.gamma_x = 22.0

        # 障碍/约束增益
        self.kp   = 1.6
        self.kxi1 = 5.0
        self.kxi2 = 5.0

        # ρ、ξ 基线（deg）
        self.A_rho_H = 1.0
        self.A_xi_H  = 10.13

        self.A_rho_K = 1.0
        self.A_xi_K  = 23.44

        # 可调基频
        self.Omega_base = 0.5  # rad/s

        # ρ 阈值与上界
        self.rho_H_th  = 1.1
        self.rho_H_max = 1.2

        self.rho_K_th  = 1.15
        self.rho_K_max = 1.2

        # ξ 阈值/边界（deg）
        # 髋关节
        self.xi_H_th_plus  = 10.13 + 5.0
        self.xi_H_th_minus = 10.13 - 5.0
        self.xi_H_max = 10.13 + 8.0
        self.xi_H_min = 10.13 - 8.0
        # 膝关节
        self.xi_K_th_plus  = 23.44 + 9.0
        self.xi_K_th_minus = 23.44 - 9.0
        self.xi_K_max = 23.44 + 9.0
        self.xi_K_min = 23.44 - 9.0

        # 左右 π 锁相
        self.v_pi = 1.5

        # 多谐波（deg）
        self.hip_a0 = 10.13
        self.hip_a  = np.array([21.80,-5.07,-0.49,-0.52, 0.20,-0.07,-0.09,-0.09], float)
        self.hip_b  = np.array([-10.77,-2.21, 1.86, 0.41, 0.20,-0.06,-0.05,-0.05], float)
  
        self.knee_a0= 23.44
        self.knee_a = np.array([-2.93,-14.32,0.05,-0.38,0.36,0.20,-0.01,0.03], float)
        self.knee_b = np.array([-26.48, 9.81,4.44, 1.87,0.59,-0.15,-0.08,-0.07], float)

        # 安全夹紧
        if use_small_margin:
            eps = 1e-3
            self.xi_H_th_plus  = min(self.xi_H_th_plus,  self.xi_H_max  - eps)
            self.xi_H_th_minus = max(self.xi_H_th_minus, self.xi_H_min  + eps)
            
            self.xi_K_th_plus  = min(self.xi_K_th_plus,  self.xi_K_max  - eps)
            self.xi_K_th_minus = max(self.xi_K_th_minus, self.xi_K_min  + eps)



