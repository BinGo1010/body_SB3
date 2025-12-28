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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

CSV_PATH = os.path.join(project_root, "cpg", "fourier_coefficients_order6_132.csv")

# # ---------------------------------------------------------------------------
# # YAML 读取工具
# # ---------------------------------------------------------------------------

def load_reward_weights_from_yaml(yaml_path: str = None):
    """
    从 body_walk_config.yaml 中读取 reward_weights 字段，
    用来覆盖默认的奖励权重。

    优先级：
    1) 环境变量 BODY_WALK_CONFIG 指定的路径
    2) 调用时传入的 yaml_path
    3) 工程根目录下 configs/body_walk_config.yaml
       （假设当前文件在 envs/walk_v1.py）
    """

    # 1. 默认值（和 DEFAULT_RWD_KEYS_AND_WEIGHTS 保持一致）
    default_weights = {
        # "vel_reward": 5.0,
        # "done": -100.0,
        # "cyclic_hip": -10.0,
        # "ref_rot": 10.0,
        # "joint_angle_rew": 5.0,
    }

    # 2. 优先使用环境变量
    env_path = os.getenv("BODY_WALK_CONFIG", None)
    if env_path is not None:
        yaml_path = env_path
        print(f"[WalkEnvV1] 使用环境变量 BODY_WALK_CONFIG 指定的配置文件: {yaml_path}")
    else:
        # 如果外部调用没显式传 yaml_path，则默认使用工程根目录下 configs/body_walk_config.yaml
        if yaml_path is None:
            this_file = Path(__file__).resolve()
            project_root = this_file.parent.parent       # /home/lvchen/body_SB3
            yaml_path = project_root / "configs" / "body_walk_config.yaml"
        print(f"[WalkEnvV1] 使用配置文件路径: {yaml_path}")

    yaml_path = Path(yaml_path)

    # 3. 检查文件是否存在
    if not yaml_path.exists():
        print(f"[WalkEnvV1] 未找到配置文件 {yaml_path}，使用默认 reward_weights: {default_weights}")
        return default_weights

    # 4. 尝试读取并解析 YAML
    try:
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WalkEnvV1] 解析 {yaml_path} 失败（{e}），使用默认 reward_weights: {default_weights}")
        return default_weights

    # 5. 读取 reward_weights 字段
    rw = cfg.get("reward_weights", {})
    if not isinstance(rw, dict):
        print(f"[WalkEnvV1] {yaml_path} 中未找到有效的 reward_weights 字段，使用默认 reward_weights: {default_weights}")
        return default_weights

    # 6. 覆盖已存在的 key
    for k, v in rw.items():
        try:
            default_weights[k] = float(v)
        except (TypeError, ValueError):
            print(f"[WalkEnvV1] reward_weights['{k}']={v} 无法转换为 float，已忽略该项")

    print(f"[WalkEnvV1] 成功从 {yaml_path} 加载 reward_weights: {default_weights}")
    return default_weights

class WalkEnvV5(BaseV0):
    # 默认观测空间
    DEFAULT_OBS_KEYS = [
        # self
        'qpos_without_xy',
        'qvel',
        'com_vel',
        'torso_angle',
        'feet_heights',
        'height',
        'feet_rel_positions',
        'muscle_length',
        'muscle_velocity',
        'muscle_force', 

        # imitation 
        'phase_var',

        'imit_cur_pos',
        'imit_ref_pos',
        'imit_err_pos',

        'imit_cur_vel',
        'imit_ref_vel',
        'imit_err_vel',
         
    ]
    # 默认奖励及其权重
    DEFAULT_RWD_KEYS_AND_WEIGHTS = load_reward_weights_from_yaml()

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path,
                         seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    # 环境的具体配置和初始化
    def _setup(self,
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               # 核心超参数
               min_height=0.8,
               max_rot=0.8,
               hip_period=100,
               reset_type='init',
               target_x_vel=0.0,
               target_y_vel=1.2,
               target_rot=None,
               _root_ref_xy=np.array([0.0, 0.0], dtype=float),
               **kwargs,
               ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period   # 每个步态周期对应的环境 step 数
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.steps = 0  # 步数计数器
        # --- 12 DOF CMU mocap 傅立叶参数，用相位驱动 ---
        self.cmu_params = CMUParams(csv_path=CSV_PATH)
        # 12 个需要模仿的关节名称（MuJoCo 关节名，需与 params_12 中 alias_map 对齐）
        self.imit_joint_names = [
            "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
            "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
            "knee_angle_l", "knee_angle_r",
            "ankle_angle_l", "subtalar_angle_l",
            "ankle_angle_r", "subtalar_angle_r",
        ]
        self.imit_scale = np.array([
            0.55709089,  # hip_flexion_l
            0.11318287,  # hip_adduction_l
            0.48977719,  # hip_rotation_l

            0.61566119,  # hip_flexion_r
            0.21976970,  # hip_adduction_r
            0.58476293,  # hip_rotation_r

            1.06549257,  # knee_angle_l
            1.23463128,  # knee_angle_r
            
            0.23938650,  # ankle_angle_l
            0.30004443,  # subtalar_angle_l
            0.27570828,  # ankle_angle_r
            0.40677454,  # subtalar_angle_r
        ], dtype=np.float32)
        # precompute a reasonable joint-velocity scale from the reference gait (rad/s)
        self.imit_vel_scale = np.array([
            # 左髋：屈伸 / 内收 / 旋转
            1.33083484,  # hip_flexion_l     max|dq| (rad/s)
            0.34742492,  # hip_adduction_l
            0.67381677,  # hip_rotation_l

            # 右髋：屈伸 / 内收 / 旋转
            1.47632789,  # hip_flexion_r
            0.77815466,  # hip_adduction_r
            0.75692577,  # hip_rotation_r

            # 膝：左右屈伸
            2.33031931,  # knee_angle_l
            2.85285944,  # knee_angle_r

            # 踝 + 跟距：左右
            1.48036847,  # ankle_angle_l
            0.95385693,  # subtalar_angle_l
            1.82795103,  # ankle_angle_r
            0.90227851,  # subtalar_angle_r
        ], dtype=np.float32)

        # 初始时的根位置参考
        self._root_ref_xy = np.array([0.0, 0.0], dtype=float)
        # 通用的环境设置
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs)

        # 初始化状态
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0

        # 调整地形（如果未使用）
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])

    # 计算观测字典
    def get_obs_dict(self, sim):
        obs_dict = {}
        # Base
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['phase_var'] = np.array([(self.steps / self.hip_period) % 1.0]).copy()  # 相位变量
        # Pelvis State
        obs_dict['height'] = np.array([self._get_height()]).copy() # 质心高度1d
        obs_dict['com_vel'] = np.array([self._get_com_velocity().copy()])  # 质心速度3D ！升至3D
        obs_dict['torso_angle'] = np.array([self._get_torso_angle().copy()]) # 姿态四元数4D
        # Feet Contact Forces
        obs_dict["feet_contacts"] = np.array([self._get_feet_contacts().copy()]) # 足部接触力4D
        obs_dict['feet_heights'] = self._get_feet_heights().copy()  # 双脚高度2D
        obs_dict['feet_rel_positions'] = self._get_feet_relative_position().copy()  # 脚部相对骨盆位置
        # Joint Kinematics
        obs_dict['qpos_without_xy'] = sim.data.qpos[2:].copy()  # 关节位置
        obs_dict['qvel'] = sim.data.qvel[:].copy() * self.dt    # 关节速度不需要* self.dt
        
        
        # Muscle States
        obs_dict['muscle_length'] = self.muscle_lengths()
        obs_dict['muscle_velocity'] = self.muscle_velocities()
        obs_dict['muscle_force'] = self.muscle_forces()
        if sim.model.na > 0:  
            obs_dict['act'] = sim.data.act[:].copy()
        
        # ===== 相位 =====
        phase_var = (self.steps / self.hip_period) % 1.0
        phase_next = (phase_var + 1.0 / self.hip_period) % 1.0

        # ===== 角度参考与误差（t -> t+1）=====
        cur = self._get_angle(self.imit_joint_names).astype(np.float32)

        des = np.array(
            [self.cmu_params.eval_joint_phase(jn, phase_next)
            for jn in self.imit_joint_names],
            dtype=np.float32
        )
        err = des - cur
        obs_dict['imit_cur_pos'] = (cur / self.imit_scale).copy()
        obs_dict['imit_ref_pos'] = (des / self.imit_scale).copy()
        obs_dict['imit_err_pos'] = (err / self.imit_scale).copy()

        # ===== 速度参考与误差（t -> t+1）=====
        phase_dot = 1.0 / (float(self.hip_period) * float(self.dt))
        cur_vel = self._get_vel(self.imit_joint_names).astype(np.float32)
        des_vel = np.array(
            [self.cmu_params.eval_joint_phase_vel(
                jn, phase_next, phase_dot=phase_dot
            ) for jn in self.imit_joint_names],
            dtype=np.float32
        )
        err_vel = des_vel - cur_vel
        obs_dict['imit_cur_vel'] = (cur_vel / self.imit_vel_scale).copy()
        obs_dict['imit_ref_vel'] = (des_vel / self.imit_vel_scale).copy()
        obs_dict['imit_err_vel'] = (err_vel / self.imit_vel_scale).copy()

        return obs_dict

    # 计算奖励字典
    def get_reward_dict(self, obs_dict):
        # --- 主模仿奖励：12个 DOF 关节轨迹及速度 ---
        imit_pose = self._get_imit_pose_rew()
        imit_vel = self._get_imit_vel_rew()

        # --- “约束型”奖励 ---
        upright = self._get_upright_rew()
        vel_reward = self._get_vel_reward()
        act_mag = self.get_act_energy()
        center = self._get_center_reward() 
        # 终止
        done = self._get_done()

        #  构建奖励字典
        rwd_dict = collections.OrderedDict((
            # 主模仿奖励
            ('imit_pose', imit_pose),
            ('imit_vel',  imit_vel),
            # 约束与正则项
            ('upright', upright), 
            ('vel_reward', vel_reward),
            ('act_mag', act_mag),
            ('center', center), 
            # 必须字段
            ('sparse', vel_reward),            # 可以根据需要改成 imit_pose
            ('solved', (vel_reward > 0.8) and (imit_pose > 0.8)),      # 简单判定：模仿奖励足够高
            ('done', done),
        ))

        # 计算综合奖励（由 YAML 中的权重加权）
        rwd_dict['dense'] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()],
            axis=0
        )
        return rwd_dict

    # # 随机初始状态-2种姿势
    # def get_randomized_initial_state(self):
    #     # randomly start with flexed left or right knee
    #     if self.np_random.uniform() < 0.5:
    #         qpos = self.sim.model.key_qpos[2].copy()
    #         qvel = self.sim.model.key_qvel[2].copy()
    #     else:
    #         qpos = self.sim.model.key_qpos[3].copy()
    #         qvel = self.sim.model.key_qvel[3].copy()

    #     # randomize qpos coordinates, but dont change height or rot state
    #     rot_state = qpos[3:7]
    #     height = qpos[2]
    #     qpos[:] = qpos[:] + self.np_random.normal(0, 0.02, size=qpos.shape)
    #     qpos[3:7] = rot_state
    #     qpos[2] = height
    #     return qpos, qvel

    def step(self, *args, **kwargs):
        results = super().step(*args, **kwargs)
        self.steps += 1
        return results
    
    def reset(self, **kwargs):
        self.steps = 0

        if self.reset_type == 'random':
            qpos, qvel = self.get_randomized_initial_state()

        elif self.reset_type == 'init':
            # 尝试使用 CSV 的第一帧作为“固定初始状态”
            self._load_reset_states_from_csv()
            if getattr(self, "_reset_qpos_list", None) is not None:
                qpos = self._reset_qpos_list[0].copy()
                qvel = self._reset_qvel_list[0].copy()
            else:
                qpos, qvel = self.sim.model.key_qpos[2], self.sim.model.key_qvel[2]
        else:
            qpos, qvel = self.sim.model.key_qpos[2], self.sim.model.key_qvel[2]

        # pelvis_id = self.sim.model.body_name2id("pelvis")
        # p0 = self.sim.data.body_xpos[pelvis_id].copy()   # (3,)
        # self._root_ref_xy = p0[:2].copy()  
        self._root_ref_xy = np.array([0.0, 0.0], dtype=float)

        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return obs

    def _load_reset_states_from_csv(self):
        # 如果已经加载过，就直接返回
        if hasattr(self, "_reset_qpos_list") and self._reset_qpos_list is not None:
            return

        # 根据你的工程结构，这里假设 CSV 在 ../utils/reset_states_from_mujoco.csv
        csv_path = os.path.join(
            os.path.dirname(__file__),  # envs 目录
            "reset_states_from_mujoco.csv",
        )
        csv_path = os.path.abspath(csv_path)

        if not os.path.exists(csv_path):
            print(f"[ResetCSV] 未找到 {csv_path}，将回退到 keyframe 初始化。")
            self._reset_qpos_list = None
            self._reset_qvel_list = None
            return

        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        nq = self.sim.model.nq
        nv = self.sim.model.nv

        if data.shape[1] != 1 + nq + nv:
            print(
                f"[ResetCSV] CSV 列数不匹配: {data.shape[1]} vs 期望 {1+nq+nv}，"
                "将回退到 keyframe 初始化。"
            )
            self._reset_qpos_list = None
            self._reset_qvel_list = None
            return
        # 拆分 time / qpos / qvel
        self._reset_time_list = data[:, 0]
        self._reset_qpos_list = data[:, 1 : 1 + nq]
        self._reset_qvel_list = data[:, 1 + nq : 1 + nq + nv]
        print(
            f"[ResetCSV] 成功加载 reset 状态: {self._reset_qpos_list.shape[0]} 帧, "
            f"nq={nq}, nv={nv}"
        )

    def get_randomized_initial_state(self):
        """
        优先从 reset_states_from_mujoco.csv 中随机抽一帧 (qpos, qvel) 作为初始状态；
        若 CSV 不存在或维度不匹配，则回退到原来的 keyframe 方式。
        """
        # 1) 尝试加载 CSV（只会加载一次）
        self._load_reset_states_from_csv()

        if getattr(self, "_reset_qpos_list", None) is not None:
            # --- 使用 CSV 中的多帧状态 ---
            n = self._reset_qpos_list.shape[0]
            idx = self.np_random.integers(n)

            qpos = self._reset_qpos_list[idx].copy()
            qvel = self._reset_qvel_list[idx].copy()

            # 适当加一点轻微噪声，避免过拟合到某一帧
            qpos += self.np_random.normal(0, 0.005, size=qpos.shape)
            qvel += self.np_random.normal(0, 0.02, size=qvel.shape)
        return qpos, qvel

    # ----------------- 关节轨迹模仿奖励 -----------------
    def _get_imit_pose_rew(self, sigma=20.0):
        phase_var = (self.steps / self.hip_period) % 1.0
        phase_next = (phase_var + 1.0 / self.hip_period) % 1.0

        des = np.array(
            [self.cmu_params.eval_joint_phase(jn, phase_next) for jn in self.imit_joint_names],
            dtype=np.float32
        )  # rad
        cur = self._get_angle(self.imit_joint_names).astype(np.float32)  # rad

        # 关键：先做按关节归一化
        diff = (des - cur) / self.imit_scale  # 无量纲，12 维

        # 可以加权（髋膝权重大一些）
        w = np.array([
            3.0, 1.0, 1.0,   # 左髋：屈伸最重要
            3.0, 1.0, 1.0,   # 右髋
            3.0, 3.0,        # 左右膝
            1.5, 0.5,        # 左踝/跟距
            1.5, 0.5,        # 右踝/跟距
        ], dtype=np.float32)

        mse = float(np.mean(w * diff**2))   # 这里就是“各关节相对误差”的加权平均
        rew = float(np.exp(-sigma * mse))
        return rew
    # ----------------- 速度模仿奖励 -----------------
    def _get_imit_vel_rew(self, sigma=5.0):
        """DeepMimic-style joint velocity imitation reward."""
        phase_var = (self.steps / self.hip_period) % 1.0
        phase_next = (phase_var + 1.0 / self.hip_period) % 1.0
        phase_dot = 50.0 / float(self.hip_period)   # cycle/s

        dq_ref = np.array([self.cmu_params.eval_joint_phase_vel(jn, phase_next, phase_dot=phase_dot) for jn in self.imit_joint_names])
        dq_cur = self._get_vel(self.imit_joint_names).astype(np.float32)

        diff = (dq_ref - dq_cur) / self.imit_vel_scale  # dimensionless

        # reuse pose weights (hip/knee more important)
        w = np.array([
            3.0, 1.0, 1.0,
            3.0, 1.0, 1.0,
            3.0, 3.0,
            1.5, 0.5,
            1.5, 0.5,
        ], dtype=np.float32)

        mse = float(np.mean(w * diff**2))
        rew = float(np.exp(-sigma * mse))
        return rew
    # ----------------- 你原有的各种辅助函数 -----------------

    def muscle_lengths(self):
        return self.sim.data.actuator_length

    def muscle_forces(self):
        return np.clip(self.sim.data.actuator_force / 1000, -100, 100)

    def muscle_velocities(self):
        return np.clip(self.sim.data.actuator_velocity, -100, 100)

    def _get_done(self):
        height = self._get_height()
        if height < self.min_height:
            return 1
        if self._get_rot_condition():
            return 1
        return 0

    def _get_joint_angle_rew(self, joint_names):
        """
        惩罚指定关节角的“过大摆动”：
        mag = mean(|q_joint|)
        r = exp(-5 * mag)
        """
        joint_angles = self._get_angle(joint_names)
        mag = np.mean(np.abs(joint_angles))
        return np.exp(-5 * mag)

    def _get_feet_heights(self):
        """
        Get the height of both feet.
        """
        foot_id_l = self.sim.model.body_name2id('talus_l')
        foot_id_r = self.sim.model.body_name2id('talus_r')
        return np.array([
            self.sim.data.body_xpos[foot_id_l][2],
            self.sim.data.body_xpos[foot_id_r][2]
        ])

    def _get_feet_relative_position(self):
        """
        Get the feet positions relative to the pelvis.
        """
        foot_id_l = self.sim.model.body_name2id('talus_l')
        foot_id_r = self.sim.model.body_name2id('talus_r')
        pelvis = self.sim.model.body_name2id('pelvis')
        return np.array([
            self.sim.data.body_xpos[foot_id_l] - self.sim.data.body_xpos[pelvis],
            self.sim.data.body_xpos[foot_id_r] - self.sim.data.body_xpos[pelvis]
        ])

    def _get_vel_reward(self):
        """
        Gaussian that incentivizes a walking velocity.
        """
        vel = self._get_com_velocity()
        return np.exp(-np.square(self.target_y_vel - vel[1])) + \
               np.exp(-np.square(self.target_x_vel - vel[0]))

    def _get_center_reward(self, y_tol=0.10, vy_tol=0.20):
        pelvis_id = self.sim.model.body_name2id('pelvis')
        p_xy = self.sim.data.body_xpos[pelvis_id][:2]
        v_xy = self.sim.data.body_xvelp[pelvis_id][:2]  # MuJoCo: world linear vel

        y0 = float(self._root_ref_xy[1])
        y_err  = float(p_xy[1] - y0)
        vy_err = float(v_xy[1] - 0.0)  # 期望横向速度为0

        # 归一化二次罚：在容忍带内仍有梯度，不会很快饱和到0
        ry  = 1.0 - min((y_err / y_tol)**2, 1.0)
        rvy = 1.0 - min((vy_err / vy_tol)**2, 1.0)

        return 0.5 * ry + 0.5 * rvy
    
    # def _get_vel_reward(self, k_xy: float = 5.0) -> float:
    #     """
    #     Root 轨迹跟踪奖励：

    #     - 参考轨迹：从 reset 时的 pelvis 初始位置出发，
    #     以 (target_x_vel, target_y_vel) 在世界 x-y 平面上匀速前进：
    #         p_ref(t) = p0_xy + [vx, vy] * t
    #     - 当前 root 位置：pelvis 的 world 坐标 (x, y)
    #     - 奖励：误差越小，奖励越接近 1；偏离越大，奖励趋近于 0：
    #         r = exp( -k_xy * || p_xy - p_ref(t) ||^2 )
    #     """
    #     # 当前时间
    #     t = float(self.sim.data.time)

    #     # 当前 pelvis 的平面位置 (x, y)
    #     pelvis_id = self.sim.model.body_name2id('pelvis')
    #     p_xy = self.sim.data.body_xpos[pelvis_id][:2]

    #     # 参考平面位置：初始点 + 期望速度 * 时间
    #     vx = float(self.target_x_vel)
    #     vy = -float(self.target_y_vel)
    #     p_ref_xy = self._root_ref_xy + np.array([vx, vy]) * t

    #     # 位置误差
    #     err_xy = p_xy - p_ref_xy
    #     err_sq = float(np.dot(err_xy, err_xy))

    #     # 高斯型奖励
    #     reward = np.exp(-k_xy * err_sq)
    #     return reward

    def _get_cyclic_rew(self):
        """
        旧版：仅用 cos 对髋做简单双边周期参考（现在可以视为次要/关闭）。
        """
        phase_var = (self.steps / self.hip_period) % 1.0
        des_angles = np.array([
            0.8 * np.cos(phase_var * 2 * np.pi + np.pi),
            0.8 * np.cos(phase_var * 2 * np.pi)
        ], dtype=np.float32)
        angles = self._get_angle(['hip_flexion_l', 'hip_flexion_r'])
        return np.linalg.norm(des_angles - angles)

    def _get_ref_rotation_rew(self):
        """
        Incentivize staying close to the initial reference orientation up to a certain threshold.
        """
        target_rot = [self.target_rot if self.target_rot is not None else self.init_qpos[3:7]][0]
        return np.exp(-np.linalg.norm(5.0 * (self.sim.data.qpos[3:7] - target_rot)))

    def _get_torso_angle(self):
        body_id = self.sim.model.body_name2id('torso')
        return self.sim.data.body_xquat[body_id]

    def _get_com_velocity(self): # 或许存在问题输出查看
        """
        Compute the center of mass velocity of the model.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = - self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:6]

    def _get_height(self):
        """
        Get center-of-mass height.
        """
        return self._get_com()[2]

    def _get_rot_condition(self):
        """
        检查根部朝向是否“转太偏”。
        """
        quat = self.sim.data.qpos[3:7].copy()
        return [1 if np.abs((quat2mat(quat) @ [1, 0, 0])[0]) > self.max_rot else 0][0]

    def _get_com(self):
        """
        Compute the center of mass of the robot.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com = self.sim.data.xipos
        return (np.sum(mass * com, 0) / np.sum(mass))

    def _get_angle(self, names):
        """
        Get the angles of a list of named joints.
        """
        return np.array([
            self.sim.data.qpos[
                self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(name)]
            ]
            for name in names
        ])
    
    def _mj_model(self):
        m = self.sim.model
        return m.ptr if hasattr(m, "ptr") else m

    def _joint_id(self, name: str) -> int:
        """用原生 mujoco 的 name2id 获取 joint id"""
        jid = mujoco.mj_name2id(self._mj_model(), mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"[mimic] joint name not found: {name}")
        return int(jid)
    
    def _get_vel(self, names):
        """qvel 用 jnt_dofadr 索引；hinge/slide 的 DOF 就是 1"""
        m = self._mj_model()
        return np.array([
            float(self.sim.data.qvel[int(m.jnt_dofadr[self._joint_id(name)])])
            for name in names
        ], dtype=np.float32)

    def _get_track_centerline_rew(self):
        pelvis_x = self.sim.data.body_xpos[self.sim.model.body_name2id('pelvis')][0]
        return np.exp(-10.0 * pelvis_x ** 2).item()

    def _get_head_stability_rew(self, v_scale=0.3, r_scale=0.2):
        """
        头部稳定性惩罚（负奖励）：
        - 惩罚头部线速度变化率 v_err
        - 惩罚头部姿态偏离 angle_diff
        形式：
            cost_v = (v_err / v_scale)^2
            cost_r = (angle_diff / r_scale)^2
            penalty = cost_v + cost_r
            reward = - penalty <= 0
        其中：
            v_scale   : 速度尺度（m/s），决定多大变化算“严重”
            r_scale   : 姿态尺度（rad），比如 0.2 rad ≈ 11 度
        """
        head_id = self.sim.model.body_name2id("head")

        # Δv: 线速度变化率
        if not hasattr(self, "_prev_head_vel"):
            self._prev_head_vel = np.zeros(3)

        curr_vel = self.sim.data.cvel[head_id][3:]
        delta_v = curr_vel - self._prev_head_vel
        self._prev_head_vel = curr_vel.copy()
        v_err = np.linalg.norm(delta_v)   # [m/s]

        # θ: 姿态误差（弧度）
        current_quat = self.sim.data.body_xquat[head_id]
        target_quat = self.init_qpos[3:7]

        current_quat = current_quat / np.linalg.norm(current_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)

        cos_theta = np.abs(np.dot(current_quat, target_quat))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_diff = 2 * np.arccos(cos_theta)  # [rad]

        # 归一化到无量纲，并平方
        cost_v = (v_err / v_scale) ** 2
        cost_r = (angle_diff / r_scale) ** 2

        penalty = cost_v + cost_r

        # 返回负数：惩罚
        reward = -penalty
        return float(reward)
    
    def _get_upright_rew(self, k=3.0):
        """
        直立奖励（基于 pelvis 的整体倾斜角）：
        - 利用 pelvis 的姿态四元数，计算其“身体 z 轴”相对世界竖直方向的夹角
        - 倾斜越小（roll、pitch 越接近 0），奖励越接近 1
        - 完全倒地时，夹角接近 π/2（甚至更大），奖励接近 0
        """
        # 根节点 free joint 的 quaternion：qpos[3:7]，MuJoCo 顺序 [w, x, y, z]
        quat = self.sim.data.qpos[3:7].copy()

        # 转成旋转矩阵（world_from_body）
        R = quat2mat(quat)   # 3x3

        # 身体自身 z 轴，在世界坐标下的方向
        body_z_world = R @ np.array([0.0, 0.0, 1.0])

        # 与世界 z 轴 [0,0,1] 的夹角：cos(theta) = dot(body_z, world_z)
        cz = np.clip(body_z_world[2], -1.0, 1.0)
        tilt_angle = np.arccos(cz)   # rad, 0=完全直立

        # 高斯型奖励：直立 -> 1，倾斜 -> 0
        upright_reward = np.exp(-k * (tilt_angle ** 2))
        return float(upright_reward)

    def _get_feet_contacts(self):
        """
        读取足底/脚趾 touch 传感器并返回：
        - feet_contacts: np.ndarray shape (4,) = [r_foot, r_toes, l_foot, l_toes]
        - planted_feet: int, 取值：
            -1: 双脚都未接触
            0: 右脚接触（r_foot 或 r_toes）
            1: 左脚接触（l_foot 或 l_toes）
            2: 双脚都接触
        """
        # 传感器名字顺序与你的 XML 对齐
        sensor_names = ["r_foot", "r_toes", "l_foot", "l_toes"]

        vals = np.zeros(4, dtype=float)
        for i, nm in enumerate(sensor_names):
            sid = self.sim.model.sensor_name2id(nm)     # 传感器 id
            adr = self.sim.model.sensor_adr[sid]        # sensordata 起始索引
            dim = self.sim.model.sensor_dim[sid]        # touch 一般 dim=1
            vals[i] = float(self.sim.data.sensordata[adr:adr + dim][0])

        # 判定接触（阈值可调；touch 通常为正即接触，但建议给个小阈值抗噪）
        th = 1e-6
        r_contact = (vals[0] > th) or (vals[1] > th)
        l_contact = (vals[2] > th) or (vals[3] > th)

        planted_feet = -1
        if r_contact:
            planted_feet = 0
        if l_contact:
            planted_feet = 1
        if r_contact and l_contact:
            planted_feet = 2
        onehot = np.zeros(4, dtype=np.float32)   # [none, right, left, both]
        onehot[{ -1: 0, 0: 1, 1: 2, 2: 3 }[planted_feet]] = 1.0

        return onehot
    
    def get_act_energy(self, k=0.05):
        """
        肌肉激活能量效率奖励（越省力越接近 1）。

        思路：
        - 用激活向量的 L1 + L2 范数刻画“用力大小”；
        - 先取负号得到 energy_term（越费力越负）；
        - 再通过 exp(k_act * energy_term) 映射到 (0, 1]：
            * 激活越小 -> energy_term 越接近 0 -> 奖励接近 1
            * 激活越大 -> energy_term 越负 -> 奖励接近 0

        其中 k_act > 0 控制惩罚强度，建议在 reward_specs 里配置。
        """
        # 没有肌肉或没有 act 观测，直接返回 0
        if self.sim.model.na == 0 or "act" not in self.obs_dict:
            return 0.0
        act = np.asarray(self.obs_dict["act"], dtype=float)
        # L1 / L2 “能量”度量（与 compute_energy_reward 同风格）
        l1_act = np.abs(act).sum()
        l2_act = np.linalg.norm(act)
        # 能量越大 -> energy_term 越负
        energy_term = -l1_act - l2_act
        act_energy_reward = np.exp(k * energy_term)

        return float(act_energy_reward)