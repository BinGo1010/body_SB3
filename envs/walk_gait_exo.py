import collections
import mujoco
from myosuite.utils import gym
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2mat
from cpg.params_12_bias import CMUParams
from cpg.params import PaperParams
from cpg.model import HipTorqueCPG
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

# With CPG-based hip torque control
class WalkEnvV3(BaseV0):
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
        'muscle_force'
    ]
    # 默认奖励及其权重
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "vel_reward": 5.0,
        "done": -100,
        "cyclic_hip": -10,
        "ref_rot": 10.0,
        "joint_angle_rew": 5.0
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    # 环境的具体配置和初始化
    def _setup(self,
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               # 核心超参数
               min_height = 0.8,
               max_rot = 0.8,
               hip_period = 100,
               reset_type='init',
               target_x_vel=0.0,
               target_y_vel=1.2,
               target_rot = None,
               **kwargs,
               ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.steps = 0  # 初始化步数计数器
        # 通用的环境设置
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs
                       )
        # 初始化状态
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0
         # === 1) 初始化扭矩 CPG ===
        self.exo_cpg = HipTorqueCPG(PaperParams(), dt=self.dt)

        # === 2) 肌肉 actuator 索引 & 数量 ===
        self.muscle_act_mask = (
            self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        )
        self.muscle_act_ids = np.where(self.muscle_act_mask)[0]
        self.n_muscle_act = int(self.muscle_act_ids.size)

        # === 3) 外骨骼髋扭矩 actuator 索引 ===
        # 注意：这里 actuator 名字要和你 xml 里一致
        self.exo_hip_ids = np.array(
            [
                self.sim.model.actuator_name2id("exo_hip_l"),
                self.sim.model.actuator_name2id("exo_hip_r"),
            ],
            dtype=int,
        )
        # 保险：确认这两个不是 muscle 类型
        assert not np.any(self.muscle_act_mask[self.exo_hip_ids])

        # === 4) 设置动作空间：[-1, 1]^(n_muscle_act + 5) ===
        act_dim = self.n_muscle_act + 5   # [muscle, Ω, amp_L, amp_R, off_L, off_R]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32,
        )

        # 调整地形（如果未使用）
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])

    # 计算观测字典
    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['time'] = np.array([sim.data.time]) # 时间
        obs_dict['qpos_without_xy'] = sim.data.qpos[2:].copy() # 关节位置
        obs_dict['qvel'] = sim.data.qvel[:].copy() * self.dt # 关节速度
        obs_dict['com_vel'] = np.array([self._get_com_velocity().copy()]) # 质心速度
        obs_dict['torso_angle'] = np.array([self._get_torso_angle().copy()]) # 躯干角度
        obs_dict['feet_heights'] = self._get_feet_heights().copy() # 双脚高度
        obs_dict['height'] = np.array([self._get_height()]).copy() # 躯干高度
        obs_dict['feet_rel_positions'] = self._get_feet_relative_position().copy() # 脚部相对位置
        obs_dict['phase_var'] = np.array([(self.steps/self.hip_period) % 1]).copy() # 相位变量
        obs_dict['muscle_length'] = self.muscle_lengths() # 肌肉长度
        obs_dict['muscle_velocity'] = self.muscle_velocities() # 肌肉速度
        obs_dict['muscle_force'] = self.muscle_forces() # 肌肉力量

        if sim.model.na>0: # 肌肉激活状态
            obs_dict['act'] = sim.data.act[:].copy()

        return obs_dict
    # 计算奖励字典
    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_vel_reward() # 计算速度奖励
        cyclic_hip = self._get_cyclic_rew() # 计算周期性奖励
        ref_rot = self._get_ref_rotation_rew() # 计算参考旋转奖励
        joint_angle_rew = self._get_joint_angle_rew(['hip_adduction_l', 'hip_adduction_r', 'hip_rotation_l',
                                                       'hip_rotation_r']) # 计算关节角度惩罚/奖励
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0 # 计算动作幅值成本（Action Magnitude Cost）
        #  构建奖励字典
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('vel_reward', vel_reward),
            ('cyclic_hip',  cyclic_hip),
            ('ref_rot',  ref_rot),
            ('joint_angle_rew', joint_angle_rew),
            ('act_mag', act_mag),
            # Must keys
            ('sparse',  vel_reward),
            ('solved',    vel_reward >= 1.0),
            ('done',  self._get_done()),
        ))
        # 计算综合奖励
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict
    
    # 随机初始状态-2种姿势
    def get_randomized_initial_state(self):
        # randomly start with flexed left or right knee
        if  self.np_random.uniform() < 0.5:
            qpos = self.sim.model.key_qpos[2].copy()
            qvel = self.sim.model.key_qvel[2].copy()
        else:
            qpos = self.sim.model.key_qpos[3].copy()
            qvel = self.sim.model.key_qvel[3].copy()

        # randomize qpos coordinates
        # but dont change height or rot state
        rot_state = qpos[3:7]
        height = qpos[2]
        qpos[:] = qpos[:] + self.np_random.normal(0, 0.02, size=qpos.shape)
        qpos[3:7] = rot_state
        qpos[2] = height
        return qpos, qvel

    def reset(self, **kwargs):
        self.steps = 0
        if self.reset_type == 'random':
            qpos, qvel = self.get_randomized_initial_state()
        elif self.reset_type == 'init':
                qpos, qvel = self.sim.model.key_qpos[2], self.sim.model.key_qvel[2]
        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        # 重置 CPG 内部状态（可选，确保每个 episode 一样）
        self.exo_cpg.reset()
        return obs

    def muscle_lengths(self):
        return self.sim.data.actuator_length

    def muscle_forces(self):
        return np.clip(self.sim.data.actuator_force / 1000, -100, 100)

    def muscle_velocities(self):
        return np.clip(self.sim.data.actuator_velocity, -100, 100)

    def _get_done(self):
        if not hasattr(self, "n_muscle_act") or not hasattr(self, "exo_cpg"):
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
        """
        Get a reward proportional to the specified joint angles.
        """
        mag = 0
        joint_angles = self._get_angle(joint_names)
        mag = np.mean(np.abs(joint_angles))
        return np.exp(-5 * mag)

    def _get_feet_heights(self):
        """
        Get the height of both feet.
        """
        foot_id_l = self.sim.model.body_name2id('talus_l')
        foot_id_r = self.sim.model.body_name2id('talus_r')
        return np.array([self.sim.data.body_xpos[foot_id_l][2], self.sim.data.body_xpos[foot_id_r][2]])

    def _get_feet_relative_position(self):
        """
        Get the feet positions relative to the pelvis.
        """
        foot_id_l = self.sim.model.body_name2id('talus_l')
        foot_id_r = self.sim.model.body_name2id('talus_r')
        pelvis = self.sim.model.body_name2id('pelvis')
        return np.array([self.sim.data.body_xpos[foot_id_l]-self.sim.data.body_xpos[pelvis], self.sim.data.body_xpos[foot_id_r]-self.sim.data.body_xpos[pelvis]])

    def _get_vel_reward(self):
        """
        Gaussian that incentivizes a walking velocity. Going
        over only achieves flat rewards.
        """
        vel = self._get_com_velocity()
        return np.exp(-np.square(self.target_y_vel - vel[1])) + np.exp(-np.square(self.target_x_vel - vel[0]))

    def _get_cyclic_rew(self):
        """
        Cyclic extension of hip angles is rewarded to incentivize a walking gait.
        """
        phase_var = (self.steps/self.hip_period) % 1
        des_angles = np.array([0.8 * np.cos(phase_var * 2 * np.pi + np.pi), 0.8 * np.cos(phase_var * 2 * np.pi)], dtype=np.float32)
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
        MuJoCo specifies the orientation as a quaternion representing the rotation
        from the [1,0,0] vector to the orientation vector. To check if
        a body is facing in the right direction, we can check if the
        quaternion when applied to the vector [1,0,0] as a rotation
        yields a vector with a strong x component.
        """
        # quaternion of root
        quat = self.sim.data.qpos[3:7].copy()
        return [1 if np.abs((quat2mat(quat) @ [1, 0, 0])[0]) > self.max_rot else 0][0]

    def _get_com(self):
        """
        Compute the center of mass of the robot.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com =  self.sim.data.xipos
        return (np.sum(mass * com, 0) / np.sum(mass))

    def _get_angle(self, names):
        """
        Get the angles of a list of named joints.
        """
        return np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(name)]] for name in names])

    # 重写 step：动作 = [muscle_actions, Omega, amp_L, amp_R, off_L, off_R]
    def step(self, a, **kwargs):
        """
        参数
        ----
        a : np.ndarray
            RL 动作向量：
            [0 : n_muscle_act)     -> 肌肉控制（原始 BaseV0 的 a，建议范围 [-1, 1]）
            [n_muscle_act + 0]     -> Omega_raw        in [-1, 1]
            [n_muscle_act + 1 : +3]-> amp_raw (L, R)   in [-1, 1]
            [n_muscle_act + 3 : +5]-> off_raw (L, R)   in [-1, 1]
        """
        # 用 BaseV0.step 初始化
        if (not hasattr(self, "n_muscle_act")) or (not hasattr(self, "exo_cpg")):
            return super().step(a, **kwargs)
        
        a = np.asarray(a, dtype=float)
        # 1) 拆分动作
        n_m = self.n_muscle_act
        assert a.size == n_m + 5, \
            f"动作维度应为 n_muscle_act + 5 = {n_m + 5}，但收到 {a.size}"

        muscle_cmd = a[:n_m]
        Omega_raw = a[n_m + 0]
        amp_raw = a[n_m + 1: n_m + 3]   # [L, R]
        off_raw = a[n_m + 3: n_m + 5]   # [L, R]

        # 2) [-1,1] → 物理区间
        Omega_min, Omega_max = 0.3, 1.5
        Omega_target = Omega_min + (Omega_raw + 1.0) * 0.5 * (Omega_max - Omega_min)

        amp_min, amp_max = 0.0, 0.1       # Nm/deg
        amp_target = amp_min + (amp_raw + 1.0) * 0.5 * (amp_max - amp_min)

        off_min, off_max = -2.0, 2.0      # Nm
        offset_target = off_min + (off_raw + 1.0) * 0.5 * (off_max - off_min)

        # 3) 扭矩 CPG
        cpg_out = self.exo_cpg.step(
            Omega_target=Omega_target,
            amp_target=amp_target,
            offset_target=offset_target,
        )
        tau_L, tau_R = cpg_out["tau"]

        # 4) 组装 actuator 控制向量
        muscle_a = np.zeros(self.sim.model.nu, dtype=float)
        # 肌肉部分
        muscle_a[self.muscle_act_ids] = muscle_cmd
        # 外骨骼髋扭矩
        muscle_a[self.exo_hip_ids[0]] = tau_L
        muscle_a[self.exo_hip_ids[1]] = tau_R

        # 5) 完全沿用 BaseV0 后半段逻辑
        muscle_act_ind = (
            self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        )

        if self.sim.model.na and self.normalize_act:
            muscle_a[muscle_act_ind] = 1.0 / (
                1.0 + np.exp(-5.0 * (muscle_a[muscle_act_ind] - 0.5))
            )
            isNormalized = False
        else:
            isNormalized = self.normalize_act

        if self.muscle_condition == "fatigue":
            muscle_a[muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                muscle_a[muscle_act_ind]
            )
        elif self.muscle_condition == "reafferentation":
            muscle_a[self.EPLpos] = muscle_a[self.EIPpos].copy()
            muscle_a[self.EIPpos] = 0.0

        self.last_ctrl = self.robot.step(
            ctrl_desired=muscle_a,
            ctrl_normalized=isNormalized,
            step_duration=self.dt,
            realTimeSim=self.mujoco_render_frames,
            render_cbk=self.mj_render if self.mujoco_render_frames else None,
        )

        self.steps += 1
        return self.forward(**kwargs)


class WalkEnvV4Multi(BaseV0):
    """
    联合训练用环境：
    - 观测：Dict(human, exo)
        human: 与原 WalkEnvV3 / 旧人体行走策略完全一致的观测拼接（便于对齐 model.zip）
        exo:   只包含 exo 关心的状态（qpos, qvel, com_vel, feet_rel_positions, phase_var），维度不同
    - 动作：连续 Box 向量
        a = [ a_human (n_muscle_act), a_exo (5维 CPG 参数) ]
    """
    # 人体观测
    DEFAULT_OBS_KEYS_HUMAN = [
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
        'muscle_force'
    ]
    # 外骨骼观测(需要修改为两个电机相关)
    DEFAULT_OBS_KEYS_EXO = [
        'qpos_without_xy',
        'qvel',
        'com_vel',
        'feet_rel_positions',
        'phase_var',
    ]

    # 奖励：沿用 V3 的默认设置（你原来那套 vel_reward + cyclic_hip + ref_rot + joint_angle_rew）
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "vel_reward": 5.0,
        "done": -100,
        "cyclic_hip": -10,
        "ref_rot": 10.0,
        "joint_angle_rew": 5.0
    }

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
               **kwargs):

        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.steps = 0

        # BaseV0 里的 _setup 会保存 self.obs_keys / rwd_keys_wt 等，
        # 这里把 obs_keys 设置为 “人体那一套”，保证与旧模型完全一致
        super()._setup(
            obs_keys=self.obs_keys_human,
            weighted_reward_keys=weighted_reward_keys,
            **kwargs
        )

        # 初始状态
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0

        # 1) 初始化外骨骼髋扭矩 CPG
        self.exo_cpg = HipTorqueCPG(PaperParams(), dt=self.dt)

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

        # 4) exo 动作维度（这里沿用 5 维 CPG 参数：Ω, amp_L, amp_R, off_L, off_R）
        self.n_exo_act = 5

        # 5) 联合动作空间：肌肉 + exo
        act_dim = self.n_muscle_act + self.n_exo_act
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32,
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

    # ========== 观测相关 ==========

    def get_obs_dict(self, sim):
        """
        与 WalkEnvV3 完全一致的 obs_dict，保证人体分支与旧模型对齐。
        """
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['qpos_without_xy'] = sim.data.qpos[2:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy() * self.dt
        obs_dict['com_vel'] = np.array([self._get_com_velocity().copy()])
        obs_dict['torso_angle'] = np.array([self._get_torso_angle().copy()])
        obs_dict['feet_heights'] = self._get_feet_heights().copy()
        obs_dict['height'] = np.array([self._get_height()]).copy()
        obs_dict['feet_rel_positions'] = self._get_feet_relative_position().copy()
        obs_dict['phase_var'] = np.array([(self.steps / self.hip_period) % 1]).copy()
        obs_dict['muscle_length'] = self.muscle_lengths()
        obs_dict['muscle_velocity'] = self.muscle_velocities()
        obs_dict['muscle_force'] = self.muscle_forces()
        if sim.model.na > 0:
            obs_dict['act'] = sim.data.act[:].copy()
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

    # ========== 奖励、终止等：与 V3 保持一致 ==========

    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_vel_reward()
        cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_rotation_rew()
        joint_angle_rew = self._get_joint_angle_rew([
            'hip_adduction_l', 'hip_adduction_r',
            'hip_rotation_l', 'hip_rotation_r'
        ])
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1) / self.sim.model.na \
            if self.sim.model.na != 0 else 0

        rwd_dict = collections.OrderedDict((
            ('vel_reward', vel_reward),
            ('cyclic_hip', cyclic_hip),
            ('ref_rot', ref_rot),
            ('joint_angle_rew', joint_angle_rew),
            ('act_mag', act_mag),
            ('sparse', vel_reward),
            ('solved', vel_reward >= 1.0),
            ('done', self._get_done()),
        ))
        rwd_dict['dense'] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()],
            axis=0
        )
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
        if self.reset_type == 'random':
            qpos, qvel = self.get_randomized_initial_state()
        elif self.reset_type == 'init':
            qpos, qvel = self.sim.model.key_qpos[2], self.sim.model.key_qvel[2]
        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]

        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        self.exo_cpg.reset()
        self.last_muscle_cmd = np.zeros(self.n_muscle_act, dtype=float)
        return obs  # 这里的 obs 已经是 Dict(human, exo)，因为 _get_obs 被覆盖了

    # === 一些工具函数（直接沿用 V3） ===

    def muscle_lengths(self):
        return self.sim.data.actuator_length

    def muscle_forces(self):
        return np.clip(self.sim.data.actuator_force / 1000, -100, 100)

    def muscle_velocities(self):
        return np.clip(self.sim.data.actuator_velocity, -100, 100)

    def _get_done(self):
        if not hasattr(self, "n_muscle_act") or not hasattr(self, "exo_cpg"):
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

    def _get_vel_reward(self):
        vel = self._get_com_velocity()
        return (
            np.exp(-np.square(self.target_y_vel - vel[1])) +
            np.exp(-np.square(self.target_x_vel - vel[0]))
        )

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
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]

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

    # ========== step：联合动作 ==========

    def step(self, action, **kwargs):
        """
        action: np.ndarray, shape = (n_muscle_act + 5,)
        - [0 : n_muscle_act) -> 人体肌肉激活 raw （统一 [-1,1]，后面通过 BaseV0 的 normalize_act 等做映射）
        - [n_muscle_act + 0] -> exo CPG: Omega_raw
        - [n_muscle_act + 1 : n_muscle_act + 3] -> exo CPG: amp_raw (L, R)
        - [n_muscle_act + 3 : n_muscle_act + 5] -> exo CPG: off_raw (L, R)
        """
        if (not hasattr(self, "n_muscle_act")) or (not hasattr(self, "exo_cpg")):
            return super().step(action, **kwargs)

        a = np.asarray(action, dtype=float)
        n_m = self.n_muscle_act
        n_e = self.n_exo_act
        assert a.size == n_m + n_e, \
            f"动作维度应为 n_muscle_act + n_exo_act = {n_m + n_e}，收到 {a.size}"

        # 1) 拆分动作
        muscle_cmd = a[:n_m]
        exo_cmd = a[n_m:]

        Omega_raw = exo_cmd[0]
        amp_raw = exo_cmd[1:3]
        off_raw = exo_cmd[3:5]

        # 2) [-1,1] → 物理区间
        Omega_min, Omega_max = 0.3, 1.5
        Omega_target = Omega_min + (Omega_raw + 1.0) * 0.5 * (Omega_max - Omega_min)

        amp_min, amp_max = 0.0, 0.1
        amp_target = amp_min + (amp_raw + 1.0) * 0.5 * (amp_max - amp_min)

        off_min, off_max = -2.0, 2.0
        offset_target = off_min + (off_raw + 1.0) * 0.5 * (off_max - off_min)

        # 3) CPG 生成 exo 髋扭矩
        cpg_out = self.exo_cpg.step(
            Omega_target=Omega_target,
            amp_target=amp_target,
            offset_target=offset_target,
        )
        tau_L, tau_R = cpg_out["tau"]

        # 4) 组装 actuator 控制向量
        muscle_a = np.zeros(self.sim.model.nu, dtype=float)
        muscle_a[self.muscle_act_ids] = muscle_cmd
        self.last_muscle_cmd = muscle_cmd.copy()
        muscle_a[self.exo_hip_ids[0]] = tau_L
        muscle_a[self.exo_hip_ids[1]] = tau_R

        muscle_act_ind = (
            self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        )

        if self.sim.model.na and self.normalize_act:
            muscle_a[muscle_act_ind] = 1.0 / (
                1.0 + np.exp(-5.0 * (muscle_a[muscle_act_ind] - 0.5))
            )
            isNormalized = False
        else:
            isNormalized = self.normalize_act

        if self.muscle_condition == "fatigue":
            muscle_a[muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                muscle_a[muscle_act_ind]
            )
        elif self.muscle_condition == "reafferentation":
            muscle_a[self.EPLpos] = muscle_a[self.EIPpos].copy()
            muscle_a[self.EIPpos] = 0.0

        self.last_ctrl = self.robot.step(
            ctrl_desired=muscle_a,
            ctrl_normalized=isNormalized,
            step_duration=self.dt,
            realTimeSim=self.mujoco_render_frames,
            render_cbk=self.mj_render if self.mujoco_render_frames else None,
        )

        self.steps += 1
        return self.forward(**kwargs)
