import collections
import mujoco
from myosuite.utils import gym
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2mat
from pathlib import Path
from cpg.params_12 import CMUParams
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
        # ----- DeepMimic-style imitation terms (recommended defaults) -----
        "imit_pose": 6.5,          # posture tracking (dominant)
        "imit_vel":  1.0,          # joint velocity tracking
        "imit_ee":   1.5,          # feet (end-effector) relative position tracking
        "imit_com":  1.0,          # COM height + torso orientation + COM velocity tracking

        # ----- task / regularization terms -----
        "vel_reward": 0.5,         # optional: forward-speed task (keep small during imitation warmup)
        "centerline": 0.2,         # keep near center line
        "head_stability": 0.5,     # stability / smoothness
        "joint_angle_rew": 0.5,    # discourage excessive hip adduction / rotation
        "act_mag": -0.2,           # activation cost (negative weight)
        "done": -100.0,            # termination penalty
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
        'qpos_without_xy',
        'qvel',
        'com_vel',
        'torso_angle',
        'feet_heights',
        'height',
        'feet_rel_positions',
        'phase_var',
        'muscle_length',
        'muscle_velocity',
        'muscle_force', 
        'imit_ref',      # 12 关节参考角（归一化）
        'imit_err',      # 12 关节误差（归一化）
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
            # 左髋：屈伸 / 内收 / 旋转
            np.deg2rad(45.0),   # hip_flexion_l    行走中大约 -20~30°，给个 45° 量级
            np.deg2rad(15.0),   # hip_adduction_l  约 -10~10°
            np.deg2rad(20.0),   # hip_rotation_l   约 -15~15°

            # 右髋：屈伸 / 内收 / 旋转
            np.deg2rad(45.0),   # hip_flexion_r
            np.deg2rad(15.0),   # hip_adduction_r
            np.deg2rad(20.0),   # hip_rotation_r

            # 膝：左右屈伸
            np.deg2rad(60.0),   # knee_angle_l     行走中 0~60° 左右
            np.deg2rad(60.0),   # knee_angle_r

            # 踝 + 跟距：左右
            np.deg2rad(25.0),   # ankle_angle_l    背屈/跖屈 -15~15°，给个 25°
            np.deg2rad(15.0),   # subtalar_angle_l 内翻/外翻 -10~10°

            np.deg2rad(25.0),   # ankle_angle_r
            np.deg2rad(15.0),   # subtalar_angle_r
        ], dtype=np.float32)

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
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['time'] = np.array([sim.data.time])  # 时间
        obs_dict['qpos_without_xy'] = sim.data.qpos[2:].copy()  # 关节位置
        obs_dict['qvel'] = sim.data.qvel[:].copy() * self.dt    # 关节速度
        obs_dict['com_vel'] = np.array([self._get_com_velocity().copy()])  # 质心速度
        obs_dict['torso_angle'] = np.array([self._get_torso_angle().copy()])  # 躯干角度
        obs_dict['feet_heights'] = self._get_feet_heights().copy()  # 双脚高度
        obs_dict['height'] = np.array([self._get_height()]).copy()  # 躯干高度
        obs_dict['feet_rel_positions'] = self._get_feet_relative_position().copy()  # 脚部相对骨盆位置
        obs_dict['phase_var'] = np.array([(self.steps / self.hip_period) % 1.0]).copy()  # 相位变量
        obs_dict['muscle_length'] = self.muscle_lengths()
        obs_dict['muscle_velocity'] = self.muscle_velocities()
        obs_dict['muscle_force'] = self.muscle_forces()

        if sim.model.na > 0:  # 肌肉激活状态
            obs_dict['act'] = sim.data.act[:].copy()
        
        # ===== 在这里加入 12DOF 参考与误差 =====
        phase_var = (self.steps / self.hip_period) % 1.0

        des = np.array(
            [self.cmu_params.eval_joint_phase(jn, phase_var) for jn in self.imit_joint_names],
            dtype=np.float32
        )  # rad
        cur = self._get_angle(self.imit_joint_names).astype(np.float32)  # rad

        # 推荐使用和奖励里一样的归一化尺度
        des_norm = des / self.imit_scale
        err_norm = (des - cur) / self.imit_scale

        obs_dict['imit_ref'] = des_norm.copy()   # (12,)
        obs_dict['imit_err'] = err_norm.copy()   # (12,)
        return obs_dict

    # 计算奖励字典
    def get_reward_dict(self, obs_dict):
        # --- DeepMimic-style imitation rewards (phase-driven reference) ---
        imit_pose = self._get_imit_pose_rew()
        imit_vel  = self._get_imit_vel_rew()
        imit_ee   = self._get_imit_ee_rew()
        imit_com  = self._get_imit_com_rew()

        # --- task / regularization rewards ---
        vel_reward = self._get_vel_reward()  # optional forward speed objective (keep small during warmup)
        joint_angle_rew = self._get_joint_angle_rew(
            ['hip_adduction_l', 'hip_adduction_r', 'hip_rotation_l', 'hip_rotation_r']
        )

        # muscle activation cost (act_mag is a cost-like signal; use negative weight in YAML)
        if self.sim.model.na != 0 and hasattr(self, "obs_dict") and "act" in self.obs_dict:
            act = self.obs_dict["act"]
            act_rms = np.linalg.norm(act, axis=-1) / np.sqrt(self.sim.model.na)
            act_mag = float(act_rms)
        else:
            act_mag = 0.0

        done = self._get_done()
        centerline = self._get_track_centerline_rew()
        head_stability = self._get_head_stability_rew()

        # build reward dict
        rwd_dict = collections.OrderedDict((
            # imitation terms
            ('imit_pose', imit_pose),
            ('imit_vel',  imit_vel),
            ('imit_ee',   imit_ee),
            ('imit_com',  imit_com),

            # task / regularization terms
            ('vel_reward', vel_reward),
            ('centerline', centerline),
            ('head_stability', head_stability),
            ('joint_angle_rew', joint_angle_rew),
            ('act_mag', act_mag),

            # required fields
            ('sparse', imit_pose),  # imitation warmup is usually more stable than speed-based sparse reward
            ('solved', (imit_pose > 0.8) and (imit_vel > 0.6) and (done == 0)),
            ('done', done),
        ))

        # dense reward (safe: ignore extra keys in YAML that are not present in rwd_dict)
        rwd_dict['dense'] = np.sum(
            [wt * rwd_dict.get(key, 0.0) for key, wt in self.rwd_keys_wt.items()],
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

    # ----------------- 模仿奖励：12DOF 关节轨迹 -----------------

    # def _get_imit_pose_rew(self, sigma=0.25):
    #     """
    #     12 个关节的模仿学习奖励：
    #     - 使用 phase_var = (steps / hip_period) % 1 对应 mocap 步态相位
    #     - 通过 CMUParams.eval_joint_phase 生成 12 DOF 的期望角度
    #     - 和当前关节角做 MSE，然后用 exp(-MSE / sigma) 压缩到 (0,1]
    #     """
    #     phase_var = (self.steps / self.hip_period) % 1.0

    #     # 期望角度：来自 CMU mocap 傅立叶（rad）
    #     des = np.array(
    #         [self.cmu_params.eval_joint_phase(jn, phase_var) for jn in self.imit_joint_names],
    #         dtype=np.float32
    #     )
    #     # 当前角度：来自 MuJoCo qpos（rad）
    #     cur = self._get_angle(self.imit_joint_names).astype(np.float32)

    #     err = des - cur
    #     mse = float(np.mean(err ** 2))  # rad^2

    #     # 误差越小，奖励越接近 1；误差大时奖励衰减
    #     rew = np.exp(-mse / sigma)
    #     return rew

    def _get_imit_pose_rew(self, sigma=1.0):
        phase_var = (self.steps / self.hip_period) % 1.0

        des = np.array(
            [self.cmu_params.eval_joint_phase(jn, phase_var) for jn in self.imit_joint_names],
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

        # sigma 建议先取 1.0 左右，比之前的 0.25 “温和”一点 
        rew = np.exp(-mse / sigma)
        return rew



    # ----------------- DeepMimic-style imitation rewards -----------------

    def _get_sim_dt(self) -> float:
        """Return MuJoCo simulation timestep (seconds)."""
        try:
            return float(self.sim.model.opt.timestep)
        except Exception:
            return 0.01

    def _init_mimic_cache(self):
        """Lazy init for reference kinematics/velocity computation."""
        if getattr(self, "_mimic_cache_inited", False):
            return

        # reference data for kinematics (same model, separate data buffer)
        # self._ref_data = mujoco.MjData(self.sim.model)
        # 兼容 dm_control wrapper 与原生 mujoco
        model = self.sim.model
        raw_model = model.ptr if hasattr(model, "ptr") else model  # dm_control: model.ptr 是 mujoco.MjModel
        self._ref_data = mujoco.MjData(raw_model)
        self._ref_model = raw_model

        # body ids for end-effector / torso
        self._pelvis_bid = self.sim.model.body_name2id('pelvis')
        self._talus_l_bid = self.sim.model.body_name2id('talus_l')
        self._talus_r_bid = self.sim.model.body_name2id('talus_r')
        self._torso_bid = self.sim.model.body_name2id('torso')

        # joint address caches (qpos and qvel)
        self._imit_jnt_ids = [self.sim.model.joint_name2id(n) for n in self.imit_joint_names]
        self._imit_qpos_adrs = [int(self.sim.model.jnt_qposadr[jid]) for jid in self._imit_jnt_ids]
        self._imit_dof_adrs = [int(self.sim.model.jnt_dofadr[jid]) for jid in self._imit_jnt_ids]
        self._imit_dof_nums = [int(self.sim.model.jnt_dofnum[jid]) for jid in self._imit_jnt_ids]

        # per-axis scaling for feet relative positions (meters)
        #   x: forward, y: lateral, z: vertical
        self.imit_ee_scale = np.array([0.25, 0.15, 0.15], dtype=np.float32)

        # precompute a reasonable joint-velocity scale from the reference gait (rad/s)
        phases = np.linspace(0.0, 1.0, 200, endpoint=False, dtype=np.float32)
        vel_samples = np.stack([self._get_ref_joint_vel(float(p)) for p in phases], axis=0)  # (200, 12)
        self.imit_vel_scale = np.maximum(np.std(vel_samples, axis=0), 0.5).astype(np.float32)

        self._mimic_cache_inited = True

    def _get_joint_vel(self, names):
        """Get generalized joint velocities for a list of named joints (rad/s)."""
        v = []
        for name in names:
            jid = self.sim.model.joint_name2id(name)
            adr = int(self.sim.model.jnt_dofadr[jid])
            dofnum = int(self.sim.model.jnt_dofnum[jid])
            if dofnum == 1:
                v.append(float(self.sim.data.qvel[adr]))
            else:
                # multi-DOF joints are not expected for the 12-DOF imitation set;
                # fall back to the norm for robustness.
                vv = self.sim.data.qvel[adr:adr + dofnum]
                v.append(float(np.linalg.norm(vv)))
        return np.array(v, dtype=np.float32)

    def _get_ref_joint_angles(self, phase_var: float) -> np.ndarray:
        """Reference joint angles (rad) from CMU/Fourier params at a given phase."""
        return np.array(
            [self.cmu_params.eval_joint_phase(jn, phase_var) for jn in self.imit_joint_names],
            dtype=np.float32
        )

    def _get_ref_joint_vel(self, phase_var: float) -> np.ndarray:
        """Reference joint velocities (rad/s) via central difference over phase."""
        dt =  float(getattr(self, "dt", 1.0/50.0))
        dphase = 1.0 / float(self.hip_period)

        p_p = (phase_var + dphase) % 1.0
        p_m = (phase_var - dphase) % 1.0

        q_p = self._get_ref_joint_angles(p_p)
        q_m = self._get_ref_joint_angles(p_m)

        return (q_p - q_m) / (2.0 * dt)

    def _update_ref_data(self, phase_var: float):
        """Write reference state into self._ref_data and forward kinematics."""
        self._init_mimic_cache()

        # start from current state to keep non-imitation joints consistent
        self._ref_data.qpos[:] = self.sim.data.qpos
        self._ref_data.qvel[:] = self.sim.data.qvel

        q_ref = self._get_ref_joint_angles(phase_var)
        dq_ref = self._get_ref_joint_vel(phase_var)

        # override imitation joints qpos/qvel
        for i, (qadr, vadr, dofnum) in enumerate(zip(self._imit_qpos_adrs, self._imit_dof_adrs, self._imit_dof_nums)):
            self._ref_data.qpos[qadr] = q_ref[i]
            if dofnum == 1:
                self._ref_data.qvel[vadr] = dq_ref[i]

        mujoco.mj_forward(self._ref_data, self._ref_data)
        return q_ref, dq_ref

    def _get_imit_vel_rew(self, sigma=1.0):
        """DeepMimic-style joint velocity imitation reward."""
        phase_var = (self.steps / self.hip_period) % 1.0
        self._init_mimic_cache()

        dq_ref = self._get_ref_joint_vel(phase_var)
        dq_cur = self._get_joint_vel(self.imit_joint_names)

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
        return float(np.exp(-mse / sigma))

    def _get_imit_ee_rew(self, sigma=1.0):
        """DeepMimic-style end-effector (feet) imitation reward using relative positions."""
        phase_var = (self.steps / self.hip_period) % 1.0
        self._update_ref_data(phase_var)

        # current feet rel positions (2, 3)
        ee_cur = self._get_feet_relative_position().astype(np.float32)

        # reference feet rel positions computed via reference kinematics
        pelvis_pos_ref = self._ref_data.body_xpos[self._pelvis_bid]
        ee_ref = np.array([
            self._ref_data.body_xpos[self._talus_l_bid] - pelvis_pos_ref,
            self._ref_data.body_xpos[self._talus_r_bid] - pelvis_pos_ref,
        ], dtype=np.float32)

        diff = (ee_ref - ee_cur) / self.imit_ee_scale  # broadcast over xyz
        mse = float(np.mean(diff**2))
        return float(np.exp(-mse / sigma))

    def _get_com_from_data(self, data):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com = data.xipos
        return (np.sum(mass * com, 0) / np.sum(mass))

    def _get_com_vel_from_data(self, data):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = -data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]

    def _get_imit_com_rew(self, sigma=1.0):
        """DeepMimic-style COM / torso imitation reward.

        Note: since the reference trajectory here is phase-driven joint motion (no reference root translation),
        we imitate: (i) COM height, (ii) torso orientation, (iii) COM planar velocity.
        """
        phase_var = (self.steps / self.hip_period) % 1.0
        self._update_ref_data(phase_var)

        # COM height
        com_cur = self._get_com()
        com_ref = self._get_com_from_data(self._ref_data)
        dh = float((com_ref[2] - com_cur[2]) / 0.05)  # 5 cm scale

        # torso orientation (quaternion similarity)
        q_cur = self.sim.data.body_xquat[self._torso_bid].astype(np.float32)
        q_ref = self._ref_data.body_xquat[self._torso_bid].astype(np.float32)
        ori_err = 1.0 - float(np.clip(np.abs(np.dot(q_cur, q_ref)), 0.0, 1.0))
        dor = ori_err / 0.10  # scale

        # COM planar velocity
        v_cur = self._get_com_velocity()
        v_ref = self._get_com_vel_from_data(self._ref_data)
        dv = (v_ref - v_cur) / 1.0  # 1 m/s scale
        mse = float(dh**2 + dor**2 + np.mean(dv**2))

        return float(np.exp(-mse / sigma))
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

    def _get_com_velocity(self):
        """
        Compute the center of mass velocity of the model.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = - self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]

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

