import collections
import mujoco
from myosuite.utils import gym
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2mat
import sys
from pathlib import Path
# When running this file as a script, Python sets sys.path[0] to the
# script directory (envs/), so top-level packages (like `cpg`) in the
# project root may be missing. Ensure project root is on sys.path.
_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from cpg.params_12_bias import CMUParams
from pathlib import Path
from collections import deque
import yaml
import os
import sys
import csv



# ---------------------------------------------------------------------------
# Pickle-safe constant schedule/callable
# ---------------------------------------------------------------------------
class ConstantFn:
    """A pickle-safe callable that always returns a constant."""

    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, _progress: float = 0.0) -> float:
        return self.value

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

CSV_PATH = os.path.join(project_root, "cpg", "fourier_coefficients_order6_132.csv")

# # ---------------------------------------------------------------------------
# # YAML 读取工具
# # ---------------------------------------------------------------------------

def load_reward_weights_from_yaml(yaml_path: str = None):
    """
    从 body_exo_walk_config.yaml 中读取 reward_weights 字段，
    用来覆盖默认的奖励权重。

    优先级：
    1) 环境变量 BODY_EXO_WALK_CONFIG 指定的路径
    2) 调用时传入的 yaml_path
    3) 工程根目录下 configs/body_exo_walk_config.yaml
       （假设当前文件在 envs/walk_gait_exo.py）
    """

    # 1. 默认值（和 DEFAULT_RWD_KEYS_AND_WEIGHTS 保持一致）
    default_weights = {
        # "vel_reward": 5.0,
        # "done": -100.0,
        # "cyclic_hip": -10.0,
        # "ref_rot": 10.0,
        # "joint_angle_rew": 5.0,
        # ===== added (exo / muscle auxiliary rewards) =====
        # "exo_track": 0.0,
        # "muscle_small": 0.0,

    }

    # 2. 优先使用环境变量
    env_path = os.getenv("BODY_WALK_CONFIG", None)
    if env_path is not None:
        yaml_path = env_path
        print(f"[WalkEnvV1] 使用环境变量 BODY_WALK_CONFIG 指定的配置文件: {yaml_path}")
    else:
        # 如果外部调用没显式传 yaml_path，则默认使用工程根目录下 configs/body_exo_walk_config.yaml
        if yaml_path is None:
            this_file = Path(__file__).resolve()
            project_root = this_file.parent.parent       # /home/lvchen/body_SB3
            yaml_path = project_root / "configs" / "body_exo_walk_config_joint.yaml"
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

class WalkEnvV4Multi(BaseV0):
    """
    联合训练用环境：
    - 观测：Dict(human, exo)
        human: 与原 WalkEnvV3 / 旧人体行走策略完全一致的观测拼接（便于对齐 model.zip）
        exo:   只包含 exo 关心的状态（qpos, qvel, com_vel, feet_rel_positions, phase_var），维度不同
    - 动作：连续 Box 向量
        a = [ a_human (n_muscle_act), a_exo (2维髋关节角度setpoint: L,R) ]
    """
    # 人体观测
    DEFAULT_OBS_KEYS_HUMAN = [
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

        'act',

    ]
    # 外骨骼观测(需要修改为两个电机相关)
    DEFAULT_OBS_KEYS_EXO = [
        "exo_q_hist",
        "exo_dq_hist",
        "exo_phase_est",
    ]

    # 奖励：沿用 V3 的默认设置（你原来那套 vel_reward + cyclic_hip + ref_rot + joint_angle_rew）
    DEFAULT_RWD_KEYS_AND_WEIGHTS = load_reward_weights_from_yaml()

    def __init__(self,
                 model_path,
                 obsd_model_path=None,
                 seed=None,
                 obs_keys_human=None,
                 obs_keys_exo=None,
                 **kwargs):

        # 记录人/机各自使用的 obs_keys（如果外部不传，就用默认）
        self.obs_keys_human = obs_keys_human or self.DEFAULT_OBS_KEYS_HUMAN
        self.obs_keys_exo = obs_keys_exo or self.DEFAULT_OBS_KEYS_EXO

        gym.utils.EzPickle.__init__(self,
                                    model_path,
                                    obsd_model_path,
                                    seed,
                                    obs_keys_human,
                                    obs_keys_exo,
                                    **kwargs)

        super().__init__(model_path=model_path,
                         obsd_model_path=obsd_model_path,
                         seed=seed,
                         env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    def _setup(self,
               # 注意：这里 super()._setup() 仍然用 “人体 obs_keys”，保证 get_obs_dict 构造的含义不变
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               min_height=0.8,
               max_rot=0.8,
               hip_period=100,
               reset_type='init',
               target_x_vel=0.0,
               target_y_vel=1.2,
               target_rot=None,
               frame_skip=10, 
                # ====== Human pretrained policy (optional) ======
                human_pretrained_path: str = None,
                freeze_human: bool = False,
                human_policy_device: str = "cpu",
                human_deterministic: bool = True,
                # ====== EXO pretrained policy (optional) ======
                exo_pretrained_path: str = None,
                freeze_exo: bool = False,
                exo_policy_device: str = "cpu",
                exo_deterministic: bool = True,
                # ====== Interleave training phase tag (optional, for logging/debug) ======
                interleave_phase: str = "none",
               **kwargs):
        
        self.exo_state_nq = int(kwargs.pop("exo_state_nq", 4))   # exo 在 qpos 上占多少维（默认4）
        self.exo_state_nv = int(kwargs.pop("exo_state_nv", 4))   # exo 在 qvel 上占多少维（默认4）

        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.steps = 0

        # 仅用于“间断式联合训练”中区分当前 phase（不影响动力学/控制）
        self.interleave_phase = str(interleave_phase or "none")
        _ph = self.interleave_phase.lower()
        if _ph in ("human", "h"):
            self.interleave_phase_id = 1
        elif _ph in ("exo", "e"):
            self.interleave_phase_id = 2
        else:
            self.interleave_phase_id = 0

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

        # 这里把 obs_keys 设置为 “人体那一套”，保证与旧模型完全一致
        super()._setup(
            obs_keys=self.obs_keys_human,
            weighted_reward_keys=weighted_reward_keys,
            frame_skip=frame_skip, 
            **kwargs
        )

        # 初始状态
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0

        # 2) 肌肉 actuator 索引 & 数量
        self.muscle_act_mask = (
            self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        )
        self.muscle_act_ids = np.where(self.muscle_act_mask)[0]
        self.n_muscle_act = int(self.muscle_act_ids.size)

        # 3) 外骨骼髋扭矩 actuator 索引（与你 XML 中的名字一致）
        self.exo_hip_ids = np.array(
            [
                self.sim.model.actuator_name2id("exo_hip_l"),
                self.sim.model.actuator_name2id("exo_hip_r"),
            ],
            dtype=int,
        )
        assert not np.any(self.muscle_act_mask[self.exo_hip_ids])

        # 4) exo 动作维度：2维左右髋关节角度 setpoint（PPO 输出，[-1,1] 归一化后映射到关节角范围）
        self.n_exo_act = 2
        self.exo_q_min = -1.571
        self.exo_q_max =  1.571 
        # ====== 预训练人体策略（可选） ======
        self.freeze_human = bool(freeze_human)
        self.human_pretrained_path = human_pretrained_path
        self.human_policy_device = str(human_policy_device or "cpu")
        self.human_deterministic = bool(human_deterministic)
        self.human_policy = None
        self.human_policy_loaded = False
        self.human_policy_obs_dim = None  # 预训练人体策略期望的 obs 维度（来自 policy）
        if self.freeze_human:
            self._load_human_policy_if_needed()
        # ====== 预训练 EXO 策略（可选） ======
        self.freeze_exo = bool(freeze_exo)
        self.exo_pretrained_path = exo_pretrained_path
        self.exo_policy_device = str(exo_policy_device or "cpu")
        self.exo_deterministic = bool(exo_deterministic)
        self.exo_policy = None
        self.exo_policy_loaded = False
        self.exo_policy_obs_dim = None
        
        self.imit_pose_log_enabled = kwargs.get("imit_pose_log_enabled", False)
        self.imit_pose_log_dir = kwargs.get("imit_pose_log_dir", "./logs/imit_pose")
        self.imit_pose_log_every = int(kwargs.get("imit_pose_log_every", 1))          # 每 N 步记录一次
        self.imit_pose_log_flush_every = int(kwargs.get("imit_pose_log_flush_every", 200))

        if self.freeze_exo:
            self._load_exo_policy_if_needed()

        if self.freeze_human and self.freeze_exo:
            raise ValueError("[WalkEnvV4Multi] freeze_human 与 freeze_exo 不能同时为 True")

        # 5) 动作空间
        n_m = int(self.n_muscle_act)
        n_e = int(self.n_exo_act)

        if self.freeze_human:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(n_e,), dtype=np.float32
            )
        elif self.freeze_exo:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(n_m,), dtype=np.float32
            )
        else:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(n_m + n_e,), dtype=np.float32
            )
        # 6) 先用当前状态构造一次 obs，自动推断人/机观测维度 → Dict space
        dummy_h = self._build_dummy_human_obs()
        dummy_e = self._build_dummy_exo_obs()
        self.human_obs_dim = dummy_h.shape[0]
        self.exo_obs_dim = dummy_e.shape[0]

        self.observation_space = gym.spaces.Dict({
            "human": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.human_obs_dim,),
                dtype=np.float32
            ),
            "exo": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.exo_obs_dim,),
                dtype=np.float32
            ),
        })

        # 7) 地形设为不可见
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])

        # 缓存一下上一次肌肉动作，后面如果你想加 imitation/BC 奖励会用到
        self.last_muscle_cmd = np.zeros(self.n_muscle_act, dtype=float)
        self._finalize_observation_space()

        self._exo_tau_hist = deque(maxlen=3)   # 存 [tau_L, tau_R]
        self.exo_sigma_as = float(kwargs.pop("exo_sigma_as", 10))     # σ_as，可放 YAML
        self.exo_tau_smooth_scale = float(kwargs.pop("exo_tau_smooth_scale", 0.5))  # 归一化尺度100(Nm)

    # ========== 观测相关 ==========

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
        # 确保 exo 相关索引缓存已建立（否则下面 exo_hist 可能用到 exo_hip_qadr 会报错）
        try:
            self._ensure_exo_pd_cache()
        except Exception:
            pass
        nq = int(sim.model.nq)
        nv = int(sim.model.nv)
        # 假设 exo 的 qpos/qvel 维度在末尾（与你 “最后4个DOF是EXO” 的描述一致）
        human_qpos = sim.data.qpos[: max(0, nq - self.exo_state_nq)]
        human_qvel = sim.data.qvel[: max(0, nv - self.exo_state_nv)]
        obs_dict['qpos_without_xy'] = human_qpos[2:].copy()
        obs_dict['qvel'] = human_qvel.copy() * self.dt # 是否需要dt

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

        if not hasattr(self, "_exo_hist_q") or not hasattr(self, "_exo_hist_dq"):
            self._exo_hist_reset()

        # 展平成 8 维（4 帧 * 2 电机）
        exo_q_hist  = np.concatenate([np.asarray(self._exo_hist_q[i],  dtype=np.float32).ravel()  for i in range(4)], axis=0)
        exo_dq_hist = np.concatenate([np.asarray(self._exo_hist_dq[i], dtype=np.float32).ravel() for i in range(4)], axis=0)

        obs_dict["exo_q_hist"]  = exo_q_hist   # shape (8,)
        obs_dict["exo_dq_hist"] = exo_dq_hist  # shape (8,)
        phase_est = self._get_state_based_phase()
        obs_dict["exo_phase_est"] = phase_est
        return obs_dict

    def _get_human_obs_vec(self, obs_dict=None):
        """
        按 obs_keys_human 的顺序展开，作为“人体策略”的输入。
        与旧的人体行走策略保持完全一致。
        """
        if obs_dict is None:
            obs_dict = self.get_obs_dict(self.sim)
        obs_list = []
        for k in self.obs_keys_human:
            v = obs_dict[k]
            obs_list.append(np.asarray(v).ravel())
        return np.concatenate(obs_list, axis=0).astype(np.float32)

    def _get_exo_obs_vec(self, obs_dict=None):
        """
        只选取 DEFAULT_OBS_KEYS_EXO（或你自定义的 obs_keys_exo），
        作为“外骨骼策略”的输入。
        """
        if obs_dict is None:
            obs_dict = self.get_obs_dict(self.sim)
        feats = []
        for k in self.obs_keys_exo:
            v = obs_dict[k]
            feats.append(np.asarray(v).ravel())
        return np.concatenate(feats, axis=0).astype(np.float32)

    def _build_dummy_human_obs(self):
        obs_dict = self.get_obs_dict(self.sim)
        return self._get_human_obs_vec(obs_dict)

    def _build_dummy_exo_obs(self):
        obs_dict = self.get_obs_dict(self.sim)
        return self._get_exo_obs_vec(obs_dict)

    def _get_obs(self, sim):
        """
        覆盖 BaseV0._get_obs，返回 Dict(human, exo)。
        BaseV0.forward() 会调用这个函数，所以 reset/step 最终输出就是 Dict。
        """
        obs_dict = self.get_obs_dict(sim)
        obs_h = self._get_human_obs_vec(obs_dict)
        obs_e = self._get_exo_obs_vec(obs_dict)
        return {"human": obs_h, "exo": obs_e}

# ========== 奖励、终止等 ==========
    def get_reward_dict(self, obs_dict):
        # --- 主模仿奖励：12个 DOF 关节轨迹及速度 ---
        imit_pose = self._get_imit_pose_rew()
        imit_vel = self._get_imit_vel_rew()
        # --- “约束型”奖励 ---
        upright = self._get_upright_rew()
        vel_reward = self._get_vel_reward()
        act_mag = self._get_act_energy()
        # --- 外骨骼辅助奖励 ---
        exo_track = self._get_exo_track_reward()
        exo_load = self._get_exo_load_reward()
        exo_power = self._get_exo_power_reward_soft()
        exo_smooth = self._get_exo_smooth_reward()

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

            # 外骨骼奖励
            ('exo_track', exo_track),
            ('exo_load', exo_load),
            ('exo_smooth', exo_smooth),
            ('exo_power', exo_power),
            # 必须字段
            ('sparse', vel_reward),            # 可以根据需要改成 imit_pose
            ('solved', (vel_reward > 0.8) and (imit_pose > 0.8)),      # 简单判定：模仿奖励足够高
            ('done', done),
        ))

        # 若在训练 Human（freeze_exo=True），显式屏蔽 exo_* 奖励，避免误加入 dense
        if bool(getattr(self, "freeze_exo", False)):
            for k in list(rwd_dict.keys()):
                if str(k).startswith("exo_"):
                    rwd_dict[k] = 0.0

        # 计算综合奖励（由 YAML 中的权重加权；仅对存在的 key 求和，避免 KeyError）
        dense = 0.0
        for key, wt in getattr(self, "rwd_keys_wt", {}).items():
            if key in rwd_dict and float(wt) != 0.0:
                dense += float(wt) * float(rwd_dict[key])
        rwd_dict["dense"] = dense
        return rwd_dict
    
    def get_randomized_initial_state(self):
        if self.np_random.uniform() < 0.5:
            qpos = self.sim.model.key_qpos[2].copy()
            qvel = self.sim.model.key_qvel[2].copy()
        else:
            qpos = self.sim.model.key_qpos[3].copy()
            qvel = self.sim.model.key_qvel[3].copy()

        rot_state = qpos[3:7]
        height = qpos[2]
        qpos[:] = qpos[:] + self.np_random.normal(0, 0.02, size=qpos.shape)
        qpos[3:7] = rot_state
        qpos[2] = height
        return qpos, qvel

    def reset(self, **kwargs):
        self.steps = 0
        self._exo_soft_t = 0.0  # seconds
        nkey = int(getattr(self.sim.model, "nkey", 0))

        if self.reset_type == 'random':
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        elif self.reset_type == 'init':
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]

        self.robot.sync_sims(self.sim, self.sim_obsd)

        # Call parent reset. Different gym(gymnasium) versions and BaseV0
        # implementations may return either `obs` or `(obs, info)`.
        parent_ret = super().reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)

        # Normalize to (obs, info) tuple form
        info = {}
        if isinstance(parent_ret, tuple) and len(parent_ret) == 2:
            obs_ret, info = parent_ret
        else:
            obs_ret = parent_ret

        # If parent returned a flattened numpy observation (not a dict),
        # convert to the Dict(human, exo) observation that this env exposes.
        if not isinstance(obs_ret, dict):
            # Use current sim state to build dict obs
            obs_ret = self._get_obs(self.sim)

        # Ensure we return the raw dict (not a flattened array) so VecEnv
        # code that expects mapping obs[key] works correctly.
        self.last_muscle_cmd = np.zeros(self.n_muscle_act, dtype=float)
        # Return (obs, info) to follow Gymnasium API. VecEnv wrappers
        # (stable-baselines3 DummyVecEnv) expect a 2-tuple so they can
                # 更新 reset 时的 root 平面参考点（用于 _get_vel_reward 的轨迹跟踪）
        try:
            pelvis_id = self.sim.model.body_name2id('pelvis')
            self._root_ref_xy = self.sim.data.body_xpos[pelvis_id][:2].copy()
        except Exception:
            # 兜底：不更新也不影响主流程
            pass
        # 初始化外骨骼速度历史缓存（4 步窗口：t, t-1, t-2, t-3）
        try:
            self._ensure_exo_pd_cache()
            self._exo_dq_hist = collections.deque(maxlen=4)
            dq_pair = self._get_exo_motor_dq()
            for _ in range(4):
                self._exo_dq_hist.append(dq_pair.copy())
        except Exception:
            # 不影响 reset 主流程
            self._exo_dq_hist = collections.deque(maxlen=4)
        self._ensure_exo_pd_cache()

        return obs_ret, info

# === 一些工具函数（直接沿用 V3） ===

    def muscle_lengths(self):
        """Return muscle actuator lengths only (exclude non-muscle actuators such as exo motors)."""
        mask = (self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
        return self.sim.data.actuator_length[mask].copy().astype(np.float32)


    def muscle_forces(self):
        """Return muscle actuator forces only (exclude non-muscle actuators)."""
        mask = (self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
        f = self.sim.data.actuator_force[mask].copy() / 1000.0
        return np.clip(f, -100.0, 100.0).astype(np.float32)


    def muscle_velocities(self):
        """Return muscle actuator velocities only (exclude non-muscle actuators)."""
        mask = (self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
        v = self.sim.data.actuator_velocity[mask].copy()
        return np.clip(v, -100.0, 100.0).astype(np.float32)

    def _get_done(self):
        if not hasattr(self, "n_muscle_act"):
            return 0
        if self.steps <= 0:
            return 0
        height = self._get_height()
        if height < self.min_height:
            return 1
        if self._get_rot_condition():
            return 1
        return 0

    def _get_joint_angle_rew(self, joint_names):
        joint_angles = self._get_angle(joint_names)
        mag = np.mean(np.abs(joint_angles))
        return np.exp(-5 * mag)

    def _get_feet_heights(self):
        foot_id_l = self.sim.model.body_name2id('talus_l')
        foot_id_r = self.sim.model.body_name2id('talus_r')
        return np.array([
            self.sim.data.body_xpos[foot_id_l][2],
            self.sim.data.body_xpos[foot_id_r][2]
        ])

    def _get_feet_relative_position(self):
        foot_id_l = self.sim.model.body_name2id('talus_l')
        foot_id_r = self.sim.model.body_name2id('talus_r')
        pelvis = self.sim.model.body_name2id('pelvis')
        return np.array([
            self.sim.data.body_xpos[foot_id_l] - self.sim.data.body_xpos[pelvis],
            self.sim.data.body_xpos[foot_id_r] - self.sim.data.body_xpos[pelvis]
        ])
    def _get_state_based_phase(self):
            """
            利用双腿髋关节差分计算鲁棒的相位变量 (0~1)。
            抗躯干俯仰干扰，且轨迹更接近正圆。
            """
            # 1. 确保缓存与索引存在
            self._ensure_exo_pd_cache()
            
            # 2. 读取数据
            # 假设 exo_hip_qadr = [左髋索引, 右髋索引]
            q_L  = self.sim.data.qpos[self.exo_hip_qadr[0]]
            q_R  = self.sim.data.qpos[self.exo_hip_qadr[1]]
            dq_L = self.sim.data.qvel[self.exo_hip_dadr[0]]
            dq_R = self.sim.data.qvel[self.exo_hip_dadr[1]]

            # 3. 计算差分 (左 - 右)
            # 这消除了躯干整体倾斜的共模噪声
            q_diff  = q_L - q_R
            dq_diff = dq_L - dq_R

            # 4. 相平面参数调节
            # omega: 归一化比例，将角速度映射到与角度同量级
            # 5.0 对应约 0.8Hz 的步频，适合中速/慢速行走
            omega = 6.0  

            # 5. 计算极角 (Phase Angle)
            # 坐标系定义：
            # x = q_diff (正值代表左腿在前，右腿在后)
            # y = -dq_diff / omega 
            # 使用 arctan2(y, x)
            phi = np.arctan2(-dq_diff / omega, q_diff) # 结果范围 (-pi, pi]

            # 6. 映射到 [0, 1)
            # phi = 0 (即 q_diff最大，左腿最前) -> phase = 0.5 还是 0.0 取决于你的定义
            # 通常我们希望 phase 随时间单调递增。
            # 此处 (phi + pi) / 2pi 将范围映射为 0 -> 1
            phase = (phi + np.pi) / (2.0 * np.pi)

            # (可选) 相位偏移校准
            # 如果你希望 phase=0 对应左腿着地瞬间，可以在这里加 offset
            # phase = (phase + 0.25) % 1.0 

            return np.array([phase], dtype=np.float32)

    # def _get_vel_reward(self):
    #     vel = self._get_com_velocity()
    #     return (
    #         np.exp(-np.square(self.target_y_vel - vel[1])) +
    #         np.exp(-np.square(self.target_x_vel - vel[0]))
    #     )
    def _get_vel_reward(self, k_xy: float = 5.0) -> float:
        """
        Root 轨迹跟踪奖励：

        - 参考轨迹：从 reset 时的 pelvis 初始位置出发，
        以 (target_x_vel, target_y_vel) 在世界 x-y 平面上匀速前进：
            p_ref(t) = p0_xy + [vx, vy] * t
        - 当前 root 位置：pelvis 的 world 坐标 (x, y)
        - 奖励：误差越小，奖励越接近 1；偏离越大，奖励趋近于 0：
            r = exp( -k_xy * || p_xy - p_ref(t) ||^2 )
        """
        # 当前时间
        t = float(self.sim.data.time)

        # 当前 pelvis 的平面位置 (x, y)
        pelvis_id = self.sim.model.body_name2id('pelvis')
        p_xy = self.sim.data.body_xpos[pelvis_id][:2]

        # 参考平面位置：初始点 + 期望速度 * 时间
        vx = float(self.target_x_vel)
        vy = -float(self.target_y_vel)
        p_ref_xy = self._root_ref_xy + np.array([vx, vy]) * t

        # 位置误差
        err_xy = p_xy - p_ref_xy
        err_sq = float(np.dot(err_xy, err_xy))

        # 高斯型奖励
        reward = np.exp(-k_xy * err_sq)
        return reward

    def _get_cyclic_rew(self):
        phase_var = (self.steps / self.hip_period) % 1
        des_angles = np.array([
            0.8 * np.cos(phase_var * 2 * np.pi + np.pi),
            0.8 * np.cos(phase_var * 2 * np.pi)
        ], dtype=np.float32)
        angles = self._get_angle(['hip_flexion_l', 'hip_flexion_r'])
        return np.linalg.norm(des_angles - angles)

    def _get_ref_rotation_rew(self):
        target_rot = self.target_rot if self.target_rot is not None else self.init_qpos[3:7]
        return np.exp(-np.linalg.norm(5.0 * (self.sim.data.qpos[3:7] - target_rot)))

    def _get_torso_angle(self):
        body_id = self.sim.model.body_name2id('torso')
        return self.sim.data.body_xquat[body_id]

    def _get_com_velocity(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = - self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:6]

    def _get_height(self):
        return self._get_com()[2]

    def _get_rot_condition(self):
        quat = self.sim.data.qpos[3:7].copy()
        return 1 if np.abs((quat2mat(quat) @ [1, 0, 0])[0]) > self.max_rot else 0

    def _get_com(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com = self.sim.data.xipos
        return (np.sum(mass * com, 0) / np.sum(mass))

    def _get_angle(self, names):
        return np.array([
            self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(name)]]
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

    # ----------------- 关节轨迹模仿奖励 -----------------
    def _get_imit_pose_rew(self, sigma=20.0):
        phase_var = (self.steps / self.hip_period) % 1.0
        phase_next = (phase_var + 1.0 / self.hip_period) % 1.0

        des = np.array(
            [self.cmu_params.eval_joint_phase(jn, phase_next) for jn in self.imit_joint_names],
            dtype=np.float32
        )  # rad
        cur = self._get_angle(self.imit_joint_names).astype(np.float32)  # rad
        diff = (des - cur) / self.imit_scale  # 无量纲
        w = np.array([
            3.0, 1.0, 1.0,
            3.0, 1.0, 1.0,
            3.0, 3.0,
            1.5, 0.5,
            1.5, 0.5,
        ], dtype=np.float32)
        mse = float(np.mean(w * diff**2))
        rew = float(np.exp(-sigma * mse))
        # ====== 记录到 CSV（可开关）======
        if bool(getattr(self, "imit_pose_log_enabled", False)):
            self._log_imit_pose(phase_var, phase_next, des, cur, diff, mse, rew)
        return rew
    # ----------------- 速度模仿奖励 -----------------
    def _get_imit_vel_rew(self, sigma=5.0):
        """DeepMimic-style joint velocity imitation reward."""
        phase_var = (self.steps / self.hip_period) % 1.0
        phase_next = (phase_var + 1.0 / self.hip_period) % 1.0
        phase_dot = 1.0 / (float(self.hip_period) * float(self.dt))  # cycles/s

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
    
    def _get_act_energy(self, k=0.05):
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
    

# ==================== 外骨骼奖励相关 ======================
    def _get_exo_track_reward(self) -> float:
        """EXO setpoint 跟踪奖励（与“PPO输出角度setpoint + PD”语义对齐）。

        使用上一 step 的 setpoint（debug/qsp_L,R）与当前关节角 q_L,R 的误差：
            e_q = q_sp - q
        目标角速度默认取 0，因此速度误差：
            e_dq = 0 - dq

        奖励采用指数型（0~1]：
            r = exp( - (e_q^2/sigma_q^2) - (e_dq^2/sigma_dq^2) )

        超参数：
            exo_sigma_q  : rad，默认 0.25
            exo_sigma_dq : rad/s，默认 2.0（若不希望速度项，可设很大或置 0）
        """
        dc = getattr(self, "_debug_cache", {}) or {}

        # setpoint（若缺失，返回 0 避免误导训练）
        qsp_L = dc.get("debug/qsp_L", None)
        qsp_R = dc.get("debug/qsp_R", None)
        if qsp_L is None or qsp_R is None or (isinstance(qsp_L, float) and np.isnan(qsp_L)):
            return 0.0

        # 当前状态（优先用 debug_cache；缺失则直接从 sim 读）
        q_L = dc.get("debug/q_L", None)
        q_R = dc.get("debug/q_R", None)
        dq_L = dc.get("debug/dq_L", None)
        dq_R = dc.get("debug/dq_R", None)

        if q_L is None or q_R is None or dq_L is None or dq_R is None or (
            isinstance(q_L, float) and np.isnan(q_L)
        ):
            try:
                q_L  = float(self.sim.data.qpos[self.exo_hip_qadr[0]])
                q_R  = float(self.sim.data.qpos[self.exo_hip_qadr[1]])
                dq_L = float(self.sim.data.qvel[self.exo_hip_dadr[0]])
                dq_R = float(self.sim.data.qvel[self.exo_hip_dadr[1]])
            except Exception:
                return 0.0

        e_q_L = float(qsp_L) - float(q_L)
        e_q_R = float(qsp_R) - float(q_R)
        e_dq_L = -float(dq_L)  # dq_sp = 0
        e_dq_R = -float(dq_R)

        sigma_q = float(getattr(self, "exo_sigma_q", 0.25))
        sigma_dq = float(getattr(self, "exo_sigma_dq", 2.0))

        # 防止除零；sigma_dq<=0 视为不使用速度项
        sigma_q = max(sigma_q, 1e-6)

        if sigma_dq is None or float(sigma_dq) <= 0.0:
            rL = np.exp(-(e_q_L * e_q_L) / (sigma_q * sigma_q))
            rR = np.exp(-(e_q_R * e_q_R) / (sigma_q * sigma_q))
        else:
            sigma_dq = max(float(sigma_dq), 1e-6)
            rL = np.exp(-(e_q_L * e_q_L) / (sigma_q * sigma_q) - (e_dq_L * e_dq_L) / (sigma_dq * sigma_dq))
            rR = np.exp(-(e_q_R * e_q_R) / (sigma_q * sigma_q) - (e_dq_R * e_dq_R) / (sigma_dq * sigma_dq))

        return float(0.5 * (rL + rR))

    def _get_exo_smooth_reward(self) -> float:
        """
        EXO 平滑奖励（扭矩二阶差分）:
        r = exp(-sigma_as * || (tau_t -2 tau_{t-1} + tau_{t-2}) / tau_scale ||^2)
        """
        hist = getattr(self, "_exo_tau_hist", None)
        if hist is None or len(hist) < 3:
            return 1.0  # 前两步没有二阶差分，给满分不惩罚

        tau_t  = np.asarray(hist[-1], dtype=np.float32).ravel()
        tau_t1 = np.asarray(hist[-2], dtype=np.float32).ravel()
        tau_t2 = np.asarray(hist[-3], dtype=np.float32).ravel()

        tau_scale = float(getattr(self, "exo_tau_smooth_scale", 30.0))
        tau_scale = max(tau_scale, 1e-6)

        dd = (tau_t - 2.0 * tau_t1 + tau_t2) / tau_scale   # 归一化二阶差分
        J = float(np.sum(dd ** 2))

        sigma_as = float(getattr(self, "exo_sigma_as", 0.2))
        return float(np.exp(-sigma_as * J))

    def _get_exo_load_reward(self) -> float:
        """
        EXO 能耗/负载代理奖励（tau^2 版本）:
        r = exp(-sigma_tau * ((tau_L/tau_scale)^2 + (tau_R/tau_scale)^2))
        取值范围 (0, 1]，越省力越接近 1。
        """
        # 取“上一 step 最终写入 actuator 的扭矩”（你 step() 里已经缓存了）
        dc = getattr(self, "_debug_cache", {}) or {}
        tau_L = float(dc.get("debug/tau_L", 0.0))
        tau_R = float(dc.get("debug/tau_R", 0.0))

        # 建议做归一化（避免 sigma 难调）
        tau_scale = float(getattr(self, "exo_tau_scale", 50.0))  # 典型力矩尺度(N·m)，可放到 YAML
        sigma_tau = float(getattr(self, "exo_sigma_tau", 0.5))   # 衰减系数，可放到 YAML

        # 防止除零
        tau_scale = max(tau_scale, 1e-6)

        J = (tau_L / tau_scale) ** 2 + (tau_R / tau_scale) ** 2
        return float(np.exp(-sigma_tau * J))
    
    # def _get_exo_power_reward_soft(
    #     self,
    #     P_scale: float = 100.0,      # W，功率归一化尺度
    #     beta: float = 6.0,           # sigmoid 陡峭度
    #     tau_scale: float = 25.0,     # Nm，扭矩归一化尺度（决定“多大算大”）
    #     sigma_tau: float = 1.0,      # 扭矩门控强度
    # ) -> float:
    #     self._ensure_exo_pd_cache()

    #     # 实际扭矩（更建议 actuator_force）
    #     tau = np.asarray(self.sim.data.actuator_force[self.exo_hip_ids], dtype=np.float32).ravel()  # [tau_L, tau_R]
    #     dq  = np.asarray(self.sim.data.qvel[self.exo_hip_dadr], dtype=np.float32).ravel()           # [dq_L, dq_R]
    #     P   = tau * dq


    #     # 正功映射到 (0,1)
    #     Pn = np.clip(P / max(P_scale, 1e-6), -10.0, 10.0)
    #     rP = 1.0 / (1.0 + np.exp(-beta * Pn))  # (0,1)

    #     # 扭矩软约束门控 (0,1]
    #     tn = tau / max(tau_scale, 1e-6)
    #     g = np.exp(-sigma_tau * (tn * tn))     # (0,1]

    #     r = rP * g
    #     return float(np.mean(r))
    def _get_exo_power_reward_soft(
        self,
        P_scale: float = 80.0,       # W：正功归一化尺度（建议按你日志的典型正功量级调）
        beta: float = 4.0,           # 正功映射曲线陡峭度
        tau_scale: float = 25.0,     # Nm：允许范围阈值
        sigma_tau: float = 2.0,      # 超阈值后的衰减强度
        eps: float = 1e-6,
    ) -> float:
        self._ensure_exo_pd_cache()

        # 扭矩与角速度
        tau = np.asarray(self.sim.data.actuator_force[self.exo_hip_ids], dtype=np.float32).ravel()
        dq  = np.asarray(self.sim.data.qvel[self.exo_hip_dadr], dtype=np.float32).ravel()

        # 机械功率
        P = tau * dq  # [W]，正=输出能量，负=吸收能量

        # -------- 1) 仅奖励正功：负功直接 0 --------
        P_pos = np.maximum(P, 0.0)

        # 归一化（避免极端值）
        Pn = np.clip(P_pos / max(P_scale, eps), 0.0, 10.0)

        # 平滑映射：Pn=0 -> 0；Pn增大 -> 1
        # 形式：1 - exp(-beta*Pn)，比 sigmoid 更符合“0功率不给分”的语义
        rP = 1.0 - np.exp(-beta * Pn)   # [0,1)

        # -------- 2) 扭矩软约束：仅当 |tau| 超过 tau_scale 才衰减 --------
        # 低于阈值不惩罚，高于阈值按超出比例衰减
        over = np.maximum(np.abs(tau) - tau_scale, 0.0) / max(tau_scale, eps)
        g = np.exp(-sigma_tau * (over * over))     # (0,1]

        r = rP * g
        return float(np.mean(r))

# ========== Human policy helpers ==========

    def _load_human_policy_if_needed(self):
        if self.human_policy is not None:
            return
        if not self.human_pretrained_path:
            raise RuntimeError("[WalkEnvV4Multi] freeze_human=True 但未提供 human_pretrained_path")

        p = Path(self.human_pretrained_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"[WalkEnvV4Multi] human_pretrained_path 不存在: {p}")

        from stable_baselines3 import PPO

        # 关键：用 custom_objects 替换掉无法反序列化的 schedule/clip 对象（仅推理用，安全）
        custom_objects = {
            "lr_schedule": ConstantFn(0.0),
            "clip_range":  ConstantFn(0.2),
        }

        self.human_policy = PPO.load(
            str(p),
            device=self.human_policy_device,
            custom_objects=custom_objects,
        )
        self.human_policy_loaded = True

        # eval mode
        if hasattr(self.human_policy, "policy") and hasattr(self.human_policy.policy, "set_training_mode"):
            self.human_policy.policy.set_training_mode(False)

        # 动作维度校验（可选但建议保留）
        try:
            act_dim = int(self.human_policy.action_space.shape[0])
            if act_dim != int(self.n_muscle_act):
                raise RuntimeError(
                    f"[WalkEnvV4Multi] human_policy action_dim={act_dim} 与当前环境 n_muscle_act={self.n_muscle_act} 不一致"
                )
        except Exception:
            pass
        self.human_policy_obs_dim = int(self.human_policy.observation_space.shape[0])  # 例如 403



    def _predict_human_muscle_cmd(self) -> np.ndarray:
        """
        用预训练人体策略输出肌肉动作（shape: [n_muscle_act]）。
        强约束版本：obs 维度必须与预训练策略期望维度一致；不再裁剪/补零。
        """
        if not self.freeze_human:
            raise RuntimeError("_predict_human_muscle_cmd() 仅在 freeze_human=True 时使用")

        self._load_human_policy_if_needed()

        obs_h = self._get_human_obs_vec()  # 当前拼接结果（按 obs_keys_human）
        obs_h = np.asarray(obs_h, dtype=np.float32).ravel()

        # 预训练 human_policy 期望的输入维度
        exp_dim = int(
            getattr(self, "human_policy_obs_dim", None)
            or self.human_policy.observation_space.shape[0]
        )
        if obs_h.size != exp_dim:
            # 打印一次 breakdown，帮助你把“是哪一个 obs_key 维度变了”定位到具体项
            if not hasattr(self, "_warn_hobs_dim_once"):
                print(f"[ENV DEBUG] human obs dim mismatch: got={obs_h.size}, expected={exp_dim}")
                try:
                    od = self.get_obs_dict(self.sim)
                    cum = 0
                    for k in getattr(self, "obs_keys_human", []):
                        n = int(np.asarray(od[k]).ravel().size)
                        cum += n
                        print(f"  - {k:20s}: {n:4d} | cum={cum}")
                except Exception as e:
                    print("[ENV DEBUG] breakdown failed:", repr(e))
                self._warn_hobs_dim_once = True

            # 强约束：直接抛错，不允许继续跑
            raise RuntimeError(
                f"[WalkEnvV4Multi] human obs dim mismatch: got={obs_h.size}, expected={exp_dim}. "
                f"请对齐 get_obs_dict() 与 obs_keys_human（尤其是 qpos_without_xy/qvel 是否混入 exo 维度、"
                f"qvel 是否乘 dt、com_vel/torso_angle 等是否改了维度）。"
            )

        act, _ = self.human_policy.predict(obs_h, deterministic=self.human_deterministic)
        return np.asarray(act, dtype=np.float32).ravel()

    
    # ========== EXO policy helpers ==========
    def _load_exo_policy_if_needed(self):
        if self.exo_policy is not None:
            return
        if not self.exo_pretrained_path:
            raise RuntimeError("[WalkEnvV4Multi] freeze_exo=True 但未提供 exo_pretrained_path")
        p = Path(self.exo_pretrained_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"[WalkEnvV4Multi] exo_pretrained_path 不存在: {p}")
        from stable_baselines3 import PPO
        custom_objects = {"lr_schedule": ConstantFn(0.0), "clip_range": ConstantFn(0.2)}
        self.exo_policy = PPO.load(str(p), device=self.exo_policy_device, custom_objects=custom_objects)
        self.exo_policy_loaded = True
        if hasattr(self.exo_policy, "policy") and hasattr(self.exo_policy.policy, "set_training_mode"):
            self.exo_policy.policy.set_training_mode(False)
        try:
            act_dim = int(self.exo_policy.action_space.shape[0])
            if act_dim != int(self.n_exo_act):
                raise RuntimeError(f"[WalkEnvV4Multi] exo_policy action_dim={act_dim} != n_exo_act={self.n_exo_act}")
        except Exception:
            pass
        try:
            self.exo_policy_obs_dim = int(self.exo_policy.observation_space.shape[0])
        except Exception:
            self.exo_policy_obs_dim = None

    def _predict_exo_cpg_cmd(self) -> 'np.ndarray':
        if not self.freeze_exo:
            raise RuntimeError("_predict_exo_cpg_cmd() 仅在 freeze_exo=True 时使用")
        self._load_exo_policy_if_needed()
        obs_e = np.asarray(self._get_exo_obs_vec(), dtype=np.float32).ravel()
        exp_dim = int(self.exo_policy_obs_dim) if self.exo_policy_obs_dim is not None else None
        if exp_dim is not None and obs_e.size != exp_dim:
            if not hasattr(self, '_warn_eobs_dim_once'):
                print(f"[ENV DEBUG] exo obs dim mismatch: got={obs_e.size}, expected={exp_dim}")
                self._warn_eobs_dim_once = True
            obs_e = obs_e[:exp_dim] if obs_e.size > exp_dim else np.pad(obs_e, (0, exp_dim - obs_e.size))
        act, _ = self.exo_policy.predict(obs_e, deterministic=self.exo_deterministic)
        act = np.asarray(act, dtype=np.float32).ravel()
        # 说明：本环境 EXO 动作为 2 维 setpoint；不再做旧版 5/6 维 CPG 参数兼容转换。

        # 更稳：保证输出维度严格等于 n_exo_act（防止未来又变维度/策略输出异常）
        n_e = int(self.n_exo_act)
        if act.size > n_e:
            act = act[:n_e]
        elif act.size < n_e:
            act = np.pad(act, (0, n_e - act.size), mode="constant")

        return act

    # ========== step：联合动作 ==========
    # ========== step：联合动作 ==========
    def step(self, a, **kwargs):
        """
        Gymnasium step.
    
        动作向量约定（EXO 为“角度 setpoint”，环境内部用 PD 生成扭矩）：
    
        - freeze_human = False 且 freeze_exo = False（联合训练）：
            a = [muscle_cmd(n_muscle_act), qsp_raw_L, qsp_raw_R]
    
        - freeze_human = True（冻结人体，只训练外骨骼）：
            a = [qsp_raw_L, qsp_raw_R]
            肌肉动作 muscle_cmd 由预训练人体策略 human_policy 产生。
    
        - freeze_exo = True（冻结外骨骼，只训练人体）：
            a = [muscle_cmd(n_muscle_act)]
            外骨骼动作 exo_act 由预训练/固定 exo 策略在环境内部产生（其输出同样为 2 维 setpoint）。
    
        其中 qsp_raw_* 的取值范围建议为 [-1, 1]（由 PPO 输出），会线性映射到 XML joint range:
            rr_joint / lr_joint range = [-1.571, 1.571] rad
        """
        self._ensure_exo_pd_cache()
    
        # 兼容：若环境还未完整初始化，退回父类
        if not hasattr(self, "n_muscle_act"):
            return super().step(a, **kwargs)
    
        # 互斥性防呆：不能同时冻结 human 与 exo
        if bool(getattr(self, "freeze_human", False)) and bool(getattr(self, "freeze_exo", False)):
            raise RuntimeError("[WalkEnvV4Multi] freeze_human 与 freeze_exo 不能同时为 True")
    
        a = np.asarray(a, dtype=float).ravel()
    
        # --------- debug cache：先初始化为 NaN，避免后续 callback 读到 None ---------
        self._debug_cache = {
            "debug/muscle_mean": float("nan"),
            "debug/muscle_std":  float("nan"),
            "debug/tau_L":       float("nan"),
            "debug/tau_R":       float("nan"),
            "debug/qsp_L":       float("nan"),
            "debug/qsp_R":       float("nan"),
            "debug/q_L":         float("nan"),
            "debug/q_R":         float("nan"),
            "debug/dq_L":        float("nan"),
            "debug/dq_R":        float("nan"),
            "debug/human_policy_used": int(bool(getattr(self, "freeze_human", False))),
            "debug/exo_policy_used":   int(bool(getattr(self, "freeze_exo", False))),
            "debug/interleave_phase_id": int(getattr(self, "interleave_phase_id", 0)),
        }
    
        n_m = int(self.n_muscle_act)
        n_e = int(self.n_exo_act)  # 期望为 2
    
        # ========== 1) 拆分动作（强互锁 + 强校验） ==========
        if bool(getattr(self, "freeze_human", False)):
            # RL 仅输出 exo(n_e)，肌肉由预训练人体策略输出
            if a.size != n_e:
                raise AssertionError(
                    f"[WalkEnvV4Multi] freeze_human=True 时动作维度应为 {n_e}，但收到 {a.size}"
                )
            muscle_cmd = self._predict_human_muscle_cmd()
            exo_act = a
    
        elif bool(getattr(self, "freeze_exo", False)):
            # RL 仅输出 muscle(n_m)，exo 由环境内部 exo 策略输出
            if a.size != n_m:
                raise AssertionError(
                    f"[WalkEnvV4Multi] freeze_exo=True 时动作维度应为 {n_m}，但收到 {a.size}"
                )
            muscle_cmd = a
            exo_act = self._predict_exo_cpg_cmd()  # 该函数会输出 2 维 setpoint 动作
    
        else:
            # 联合训练：RL 输出 [muscle(n_m), exo(n_e)]
            if a.size != (n_m + n_e):
                raise AssertionError(
                    f"[WalkEnvV4Multi] 动作维度应为 n_muscle_act + {n_e} = {n_m + n_e}，但收到 {a.size}"
                )
            muscle_cmd = a[:n_m]
            exo_act = a[n_m:n_m + n_e]
    
        # 统一形状 + 强校验（避免 silent mismatch）
        muscle_cmd = np.asarray(muscle_cmd, dtype=float).ravel()
        exo_act = np.asarray(exo_act, dtype=float).ravel()
    
        if muscle_cmd.size != n_m:
            raise AssertionError(
                f"[WalkEnvV4Multi] muscle_cmd 维度应为 {n_m}，但得到 {muscle_cmd.size}。"
                f"请检查 freeze_exo/freeze_human 分支与策略输出。"
            )
        if exo_act.size != n_e:
            raise AssertionError(
                f"[WalkEnvV4Multi] exo_act 维度应为 {n_e}，但得到 {exo_act.size}。"
                f"请检查 freeze_exo/freeze_human 分支与 exo 策略输出。"
            )
    
        # 内部推理动作同样裁剪到 [-1, 1]（避免越界污染映射）
        muscle_cmd = np.clip(muscle_cmd, -1.0, 1.0)
        exo_act = np.clip(exo_act, -1.0, 1.0)
    
        # 缓存一下（便于日志/奖励扩展）
        self.last_muscle_cmd = np.asarray(muscle_cmd, dtype=float).copy()
    
        # ========== 2) EXO 动作：[-1,1] -> 关节角 setpoint（rad） ==========
        qmin = float(getattr(self, "exo_q_min", -1.571))
        qmax = float(getattr(self, "exo_q_max",  1.571))
        # 线性映射
        q_sp = qmin + (exo_act + 1.0) * 0.5 * (qmax - qmin)  # (2,)
        q_sp_L, q_sp_R = float(q_sp[0]), float(q_sp[1])
    
        # （可选）对 setpoint 做低通，抑制 PPO 高频抖动
        alpha_sp = float(getattr(self, "exo_sp_alpha", 0.2))  # 0~1
        if alpha_sp > 0.0:
            if not hasattr(self, "_exo_qsp_filt"):
                self._exo_qsp_filt = np.array([q_sp_L, q_sp_R], dtype=np.float32)
            else:
                cur = np.array([q_sp_L, q_sp_R], dtype=np.float32)
                self._exo_qsp_filt = (1.0 - alpha_sp) * self._exo_qsp_filt + alpha_sp * cur
            q_sp_L, q_sp_R = float(self._exo_qsp_filt[0]), float(self._exo_qsp_filt[1])
    
        # ========== 3) 读当前 exo 髋关节状态 ==========
        q_L  = float(self.sim.data.qpos[self.exo_hip_qadr[0]])
        q_R  = float(self.sim.data.qpos[self.exo_hip_qadr[1]])
        dq_L = float(self.sim.data.qvel[self.exo_hip_dadr[0]])
        dq_R = float(self.sim.data.qvel[self.exo_hip_dadr[1]])
    
        # ========== 4) PD：setpoint -> 扭矩 ==========
        # 参考论文：PD 增益可设 kp=50, kd=14.14（你可在 YAML/kwargs 中覆盖）
        self.exo_kp = float(getattr(self, "exo_kp", 50.0))
        self.exo_kd = float(getattr(self, "exo_kd", 14.14))
    
        # 目标速度若未提供，可取 0（更稳）
        dq_sp_L = 0.0
        dq_sp_R = 0.0
    
        tau_L = self.exo_kp * (q_sp_L - q_L) + self.exo_kd * (dq_sp_L - dq_L)
        tau_R = self.exo_kp * (q_sp_R - q_R) + self.exo_kd * (dq_sp_R - dq_R)
    
        # --------- debug：记录 setpoint 与当前状态 ---------
        self._debug_cache["debug/qsp_L"] = float(q_sp_L)
        self._debug_cache["debug/qsp_R"] = float(q_sp_R)
        self._debug_cache["debug/q_L"]   = float(q_L)
        self._debug_cache["debug/q_R"]   = float(q_R)
        self._debug_cache["debug/dq_L"]  = float(dq_L)
        self._debug_cache["debug/dq_R"]  = float(dq_R)
    
        # 扭矩限幅（优先用 XML actuator ctrlrange）
        if getattr(self, "exo_tau_range", None) is not None:
            loL, hiL = float(self.exo_tau_range[0, 0]), float(self.exo_tau_range[0, 1])
            loR, hiR = float(self.exo_tau_range[1, 0]), float(self.exo_tau_range[1, 1])
        else:
            loL, hiL = -1000.0, 1000.0
            loR, hiR = -1000.0, 1000.0
    
        tau_L = float(np.clip(tau_L, loL, hiL))
        tau_R = float(np.clip(tau_R, loR, hiR))
    
        # ========== 5) Soft-start gate（可选，默认启用但可通过 exo_softstart_* 关闭/调参） ==========
        dt = float(getattr(self, "dt", 0.0))
        if dt <= 0.0:
            dt = float(self.sim.model.opt.timestep)
        self._exo_soft_t = float(getattr(self, "_exo_soft_t", 0.0) + dt)
    
        delay_s = float(getattr(self, "exo_softstart_delay", 0.0))   # 默认不延迟
        ramp_s  = float(getattr(self, "exo_softstart_ramp", 0.5))    # 默认 0.5s 拉到 1
        mode    = str(getattr(self, "exo_softstart_mode", "linear")) # "linear" 或 "exp"
        t = float(getattr(self, "_exo_soft_t", 0.0))
    
        if ramp_s <= 1e-6:
            g = 1.0
        elif t <= delay_s:
            g = 0.0
        else:
            x = (t - delay_s) / max(ramp_s, 1e-6)
            if mode == "linear":
                g = float(np.clip(x, 0.0, 1.0))
            elif mode == "exp":
                k = float(getattr(self, "exo_softstart_k", 4.0))
                g = float(1.0 - np.exp(-k * max(x, 0.0)))
                g = float(np.clip(g, 0.0, 1.0))
            else:
                g = 1.0
    
        tau_L *= g
        tau_R *= g
    
        # ========== 6) （可选）禁止负功输出 ==========
        # 若你希望严格贴合“assist-only”，可开启 exo_no_negative_power=True：
        #   当 tau * qdot < 0 时置零
        if bool(getattr(self, "exo_no_negative_power", False)):
            if tau_L * dq_L < 0.0:
                tau_L = 0.0
            if tau_R * dq_R < 0.0:
                tau_R = 0.0
    
        # 最终输出扭矩写入 debug
        self._debug_cache["debug/tau_L"] = float(tau_L)
        self._debug_cache["debug/tau_R"] = float(tau_R)
    
        # ========== 7) 组装 actuator 控制向量 ==========
        ctrl = np.zeros(self.sim.model.nu, dtype=float)
        # 肌肉
        ctrl[self.muscle_act_ids] = np.asarray(muscle_cmd, dtype=float)
        # 外骨骼髋扭矩 actuator
        ctrl[self.exo_hip_ids[0]] = tau_L
        ctrl[self.exo_hip_ids[1]] = tau_R
    
        # 二阶差分平滑奖励扭矩历史缓存（供 _get_exo_smooth_reward 使用）
        try:
            tau_vec = np.array([float(ctrl[self.exo_hip_ids[0]]), float(ctrl[self.exo_hip_ids[1]])], dtype=np.float32)
            self._exo_tau_hist.append(tau_vec)
        except Exception:
            pass
    
        # ========== 8) 完全沿用 BaseV0 的后半段逻辑（肌肉归一化/疲劳/仿真推进） ==========
        muscle_act_ind = (self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
    
        if self.sim.model.na and self.normalize_act:
            ctrl[muscle_act_ind] = 1.0 / (1.0 + np.exp(-5.0 * (ctrl[muscle_act_ind] - 0.5)))
            isNormalized = False
        else:
            isNormalized = self.normalize_act
    
        if self.muscle_condition == "fatigue":
            ctrl[muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(ctrl[muscle_act_ind])
        elif self.muscle_condition == "reafferentation":
            ctrl[self.EPLpos] = ctrl[self.EIPpos].copy()
            ctrl[self.EIPpos] = 0.0
    
        # --------- debug cache：从最终写入 ctrl 的链路读取（包含 normalize/fatigue 影响）---------
        try:
            _m = np.asarray(ctrl[self.muscle_act_ids], dtype=float).ravel()
            self._debug_cache["debug/muscle_mean"] = float(np.mean(_m)) if _m.size else float("nan")
            self._debug_cache["debug/muscle_std"]  = float(np.std(_m))  if _m.size else float("nan")
            self._debug_cache["debug/tau_L"]       = float(ctrl[self.exo_hip_ids[0]])
            self._debug_cache["debug/tau_R"]       = float(ctrl[self.exo_hip_ids[1]])
        except Exception as e:
            if not hasattr(self, "_dbg_err_once"):
                print("[ENV DEBUG] building _debug_cache failed:", repr(e))
                self._dbg_err_once = True
    
        self.last_ctrl = self.robot.step(
            ctrl_desired=ctrl,
            ctrl_normalized=isNormalized,
            step_duration=self.dt,
            realTimeSim=self.mujoco_render_frames,
            render_cbk=self.mj_render if self.mujoco_render_frames else None,
        )
    
        self.steps += 1
    
        # EXO 观测空间更新
        self._exo_hist_update()
    
        # BaseV0.forward (gymnasium style) returns: obs, reward, terminal, truncated, info
        obs, reward, terminal, truncated, info = self.forward(**kwargs)
    
        # 若父类返回非 dict，则强制转换成 Dict(human, exo)
        if not isinstance(obs, dict):
            obs = self._get_obs(self.sim)
    
        # 附加关键调试量（来自最终 ctrl 的链路）
        info = info or {}
        if not isinstance(info, dict):
            info = {}
        dc = getattr(self, "_debug_cache", None)
        if isinstance(dc, dict):
            info.update(dc)

        return obs, reward, terminal, truncated, info

    def _ensure_exo_pd_cache(self):
        """
        确保 PD 需要的缓存都存在：
        - exo_hip_ids: 两个髋扭矩 actuator id
        - exo_hip_joint_ids: 对应 joint id（由 actuator_trnid 反查）
        - exo_hip_qadr / exo_hip_dadr: 读取 qpos/qvel 的索引地址
        - exo_tau_range: ctrlrange（用于限幅）
        """
        # 已经建好就直接返回
        if hasattr(self, "exo_hip_qadr") and hasattr(self, "exo_hip_dadr") and hasattr(self, "exo_hip_ids"):
            return

        # 1) exo_hip_ids
        if not hasattr(self, "exo_hip_ids"):
            # 如果你 XML actuator 名字不同，把这里改成你的名字
            name_L, name_R = "exo_hip_l", "exo_hip_r"
            self.exo_hip_ids = np.array(
                [self.sim.model.actuator_name2id(name_L),
                self.sim.model.actuator_name2id(name_R)],
                dtype=int
            )

        # 2) actuator -> joint
        self.exo_hip_joint_ids = [int(self.sim.model.actuator_trnid[aid, 0]) for aid in self.exo_hip_ids]

        # 3) joint -> qpos/qvel address
        # MuJoCo 的字段名是 jnt_qposadr / jnt_dofadr（不是 joint_qposadr）
        self.exo_hip_qadr = [int(self.sim.model.jnt_qposadr[jid]) for jid in self.exo_hip_joint_ids]
        self.exo_hip_dadr = [int(self.sim.model.jnt_dofadr[jid]) for jid in self.exo_hip_joint_ids]

        # 4) ctrlrange（扭矩限幅）
        if hasattr(self.sim.model, "actuator_ctrlrange"):
            self.exo_tau_range = self.sim.model.actuator_ctrlrange[self.exo_hip_ids].copy()
        else:
            self.exo_tau_range = None

    def _finalize_observation_space(self):
        """保持 env.observation_space 与实际返回一致。

        reset/step 返回 dict(obs)。如需训练时用 Box(exo-only) 观测，
        请在训练脚本中使用 ObservationWrapper 将 dict->Box。
        """
        return
    
    def _exo_hist_reset(self):
        qL = float(self.sim.data.qpos[self.exo_hip_qadr[0]])
        qR = float(self.sim.data.qpos[self.exo_hip_qadr[1]])
        dqL = float(self.sim.data.qvel[self.exo_hip_dadr[0]])
        dqR = float(self.sim.data.qvel[self.exo_hip_dadr[1]])
        q  = np.array([qL, qR], dtype=np.float32)
        dq = np.array([dqL, dqR], dtype=np.float32)
        self._exo_hist_q  = deque([q.copy()  for _ in range(4)], maxlen=4)
        self._exo_hist_dq = deque([dq.copy() for _ in range(4)], maxlen=4)

    def _exo_hist_update(self):
        if not hasattr(self, "_exo_hist_q") or not hasattr(self, "_exo_hist_dq"):
            self._exo_hist_reset()
            return
        qL = float(self.sim.data.qpos[self.exo_hip_qadr[0]])
        qR = float(self.sim.data.qpos[self.exo_hip_qadr[1]])
        dqL = float(self.sim.data.qvel[self.exo_hip_dadr[0]])
        dqR = float(self.sim.data.qvel[self.exo_hip_dadr[1]])
        self._exo_hist_q.appendleft(np.array([qL, qR], dtype=np.float32))
        self._exo_hist_dq.appendleft(np.array([dqL, dqR], dtype=np.float32))