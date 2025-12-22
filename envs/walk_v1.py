""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Pierre Schumacher (schumacherpier@gmail.com), Cameron Berg (cam.h.berg@gmail.com)
================================================= """

import collections
import mujoco
from myosuite.utils import gym
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2mat
from cpg.model import HipTorqueCPG
from cpg.params import PaperParams
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

class WalkEnvV1(BaseV0):
    
    DEFAULT_OBS_KEYS = [
        "qpos_without_xy",
        "qvel",
        "com_vel",
        "torso_angle",
        "feet_heights",
        "height",
        "feet_rel_positions",
        "phase_var",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = load_reward_weights_from_yaml()

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
        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            env_credits=self.MYO_CREDIT,
        )
        self._setup(**kwargs)

    def _setup(
        self,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        min_height=0.8,
        max_rot=0.8,
        hip_period=100,
        reset_type="init",
        target_x_vel=0.0,
        target_y_vel=1.2,
        target_rot=None,
        **kwargs,
    ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.steps = 0
        super()._setup(
            obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, **kwargs
        )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0

        # move heightfield down if not used
        self.sim.model.geom_rgba[self.sim.model.geom_name2id("terrain")][-1] = 0.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id("terrain")] = np.array(
            [0, 0, -10]
        )

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["t"] = np.array([sim.data.time])
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["qpos_without_xy"] = sim.data.qpos[2:].copy()
        obs_dict["qvel"] = sim.data.qvel[:].copy() * self.dt
        obs_dict["com_vel"] = np.array([self._get_com_velocity().copy()])
        obs_dict["torso_angle"] = np.array([self._get_torso_angle().copy()])
        obs_dict["feet_heights"] = self._get_feet_heights().copy()
        obs_dict["height"] = np.array([self._get_height()]).copy()
        obs_dict["feet_rel_positions"] = self._get_feet_relative_position().copy()
        obs_dict["phase_var"] = np.array([(self.steps / self.hip_period) % 1]).copy()
        obs_dict["muscle_length"] = self.muscle_lengths()
        obs_dict["muscle_velocity"] = self.muscle_velocities()
        obs_dict["muscle_force"] = self.muscle_forces()
        # 计算当前相位对应的目标关节角
        phase_var = (self.steps / self.hip_period) % 1.0
        des_angles = np.array(
            [self.cmu_params.eval_joint_phase(jn, phase_var) for jn in self.imit_joint_names],
            dtype=np.float32
        )
        current_angles = self._get_angle(self.imit_joint_names)
        # 将误差加入观测
        # 注意：需要确保 des_angles 和 current_angles 单位一致（都是弧度）
        obs_dict['ref_error'] = (des_angles - current_angles).copy()

        # 也可以选择把 des_angles 直接加进去
        obs_dict['ref_angles'] = des_angles.copy()

        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_vel_reward()
        cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_rotation_rew()
        joint_angle_rew = self._get_joint_angle_rew(
            ["hip_adduction_l", "hip_adduction_r", "hip_rotation_l", "hip_rotation_r"]
        )
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )

        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("vel_reward", vel_reward),
                ("cyclic_hip", cyclic_hip),
                ("ref_rot", ref_rot),
                ("joint_angle_rew", joint_angle_rew),
                ("act_mag", act_mag),
                # Must keys
                ("sparse", vel_reward),
                ("solved", vel_reward >= 1.0),
                ("done", self._get_done()),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def get_randomized_initial_state(self):
        # randomly start with flexed left or right knee
        if self.np_random.uniform() < 0.5:
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

    def step(self, *args, **kwargs):
        results = super().step(*args, **kwargs)
        self.steps += 1
        return results

    def reset(self, **kwargs):
        self.steps = 0
        if self.reset_type == "random":
            qpos, qvel = self.get_randomized_initial_state()
        elif self.reset_type == "init":
            qpos, qvel = self.sim.model.key_qpos[2], self.sim.model.key_qvel[2]
        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return obs

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
        foot_id_l = self.sim.model.body_name2id("talus_l")
        foot_id_r = self.sim.model.body_name2id("talus_r")
        return np.array(
            [
                self.sim.data.body_xpos[foot_id_l][2],
                self.sim.data.body_xpos[foot_id_r][2],
            ]
        )

    def _get_feet_relative_position(self):
        """
        Get the feet positions relative to the pelvis.
        """
        foot_id_l = self.sim.model.body_name2id("talus_l")
        foot_id_r = self.sim.model.body_name2id("talus_r")
        pelvis = self.sim.model.body_name2id("pelvis")
        return np.array(
            [
                self.sim.data.body_xpos[foot_id_l] - self.sim.data.body_xpos[pelvis],
                self.sim.data.body_xpos[foot_id_r] - self.sim.data.body_xpos[pelvis],
            ]
        )

    def _get_vel_reward(self):
        """
        Gaussian that incentivizes a walking velocity. Going
        over only achieves flat rewards.
        """
        vel = self._get_com_velocity()
        return np.exp(-np.square(self.target_y_vel - vel[1])) + np.exp(
            -np.square(self.target_x_vel - vel[0])
        )

    # def _get_cyclic_rew(self):
    #     """
    #     Cyclic extension of hip angles is rewarded to incentivize a walking gait.
    #     """
    #     phase_var = (self.steps / self.hip_period) % 1
    #     des_angles = np.array(
    #         [
    #             0.8 * np.cos(phase_var * 2 * np.pi + np.pi),
    #             0.8 * np.cos(phase_var * 2 * np.pi),
    #         ],
    #         dtype=np.float32,
    #     )
    #     angles = self._get_angle(["hip_flexion_l", "hip_flexion_r"])
    #     return np.linalg.norm(des_angles - angles)

    def _get_cyclic_rew(self):
        """
        用 CPG 的傅里叶系数对 hip/knee 的周期性运动进行拟合，鼓励相位一致的轨迹跟踪。
        """
        self.p = PaperParams()
        phase = (self.steps / self.hip_period) % 1
        phi_L = 2 * np.pi * phase + np.pi  # 左腿滞后 π，相位差对齐
        phi_R = 2 * np.pi * phase

        # 使用傅里叶系数构建函数
        def fourier_eval(phi, a0, a, b):
            l = np.arange(1, len(a) + 1, dtype=float)
            return a0 + np.sum(a * np.cos(l * phi) + b * np.sin(l * phi))

        # 髋目标角度（单位：deg）
        des_hip_deg = np.array([
            fourier_eval(phi_L, self.p.hip_a0, self.p.hip_a, self.p.hip_b),
            fourier_eval(phi_R, self.p.hip_a0, self.p.hip_a, self.p.hip_b),
        ], dtype=np.float32)

        # 膝目标角度（单位：deg）
        des_knee_deg = np.array([
            fourier_eval(phi_L, self.p.knee_a0, self.p.knee_a, self.p.knee_b),
            fourier_eval(phi_R, self.p.knee_a0, self.p.knee_a, self.p.knee_b),
        ], dtype=np.float32)

        # ===== 在这里统一转弧度 =====
        des_hip = np.deg2rad(des_hip_deg)
        des_knee = np.deg2rad(des_knee_deg)

        # 当前关节角度：MuJoCo 本来就是 rad，不要再转成 deg
        hip_now = self._get_angle(['hip_flexion_l', 'hip_flexion_r'])
        knee_now = self._get_angle(['knee_angle_l', 'knee_angle_r'])

        # 误差（欧几里得距离，单位：rad）
        hip_err = np.linalg.norm(des_hip - hip_now)
        knee_err = np.linalg.norm(des_knee - knee_now)
        
        return hip_err + knee_err

    def _get_ref_rotation_rew(self):
        """
        Incentivize staying close to the initial reference orientation up to a certain threshold.
        """
        target_rot = [
            self.target_rot if self.target_rot is not None else self.init_qpos[3:7]
        ][0]
        return np.exp(-np.linalg.norm(5.0 * (self.sim.data.qpos[3:7] - target_rot)))

    def _get_torso_angle(self):
        body_id = self.sim.model.body_name2id("torso")
        return self.sim.data.body_xquat[body_id]

    def _get_com_velocity(self):
        """
        Compute the center of mass velocity of the model.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = -self.sim.data.cvel
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
        com = self.sim.data.xipos
        return np.sum(mass * com, 0) / np.sum(mass)

    def _get_angle(self, names):
        """
        Get the angles of a list of named joints.
        """
        return np.array(
            [
                self.sim.data.qpos[
                    self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(name)]
                ]
                for name in names
            ]
        )

class WalkEnvV1_1(BaseV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy",
        "qvel",
        "com_vel",
        "torso_angle",
        "feet_heights",
        "height",
        "feet_rel_positions",
        "phase_var",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = load_reward_weights_from_yaml()

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
        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            env_credits=self.MYO_CREDIT,
        )
        self._setup(**kwargs)

    def _setup(
        self,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        min_height=0.8,
        max_rot=0.8,
        hip_period=100,
        reset_type="init",
        target_x_vel=0.0,
        target_y_vel=1.2,
        target_rot=None,
        **kwargs,
    ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.steps = 0
        super()._setup(
            obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, **kwargs
        )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0

        # move heightfield down if not used
        self.sim.model.geom_rgba[self.sim.model.geom_name2id("terrain")][-1] = 0.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id("terrain")] = np.array(
            [0, 0, -10]
        )

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["t"] = np.array([sim.data.time])
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["qpos_without_xy"] = sim.data.qpos[2:].copy()
        obs_dict["qvel"] = sim.data.qvel[:].copy() * self.dt
        obs_dict["com_vel"] = np.array([self._get_com_velocity().copy()])
        obs_dict["torso_angle"] = np.array([self._get_torso_angle().copy()])
        obs_dict["feet_heights"] = self._get_feet_heights().copy()
        obs_dict["height"] = np.array([self._get_height()]).copy()
        obs_dict["feet_rel_positions"] = self._get_feet_relative_position().copy()
        obs_dict["phase_var"] = np.array([(self.steps / self.hip_period) % 1]).copy()
        obs_dict["muscle_length"] = self.muscle_lengths()
        obs_dict["muscle_velocity"] = self.muscle_velocities()
        obs_dict["muscle_force"] = self.muscle_forces()

        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_vel_reward()
        cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_rotation_rew()
        joint_angle_rew = self._get_joint_angle_rew(
            ["hip_adduction_l", "hip_adduction_r", "hip_rotation_l", "hip_rotation_r"]
        )
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )

        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("vel_reward", vel_reward),
                ("cyclic_hip", cyclic_hip),
                ("ref_rot", ref_rot),
                ("joint_angle_rew", joint_angle_rew),
                ("act_mag", act_mag),
                # Must keys
                ("sparse", vel_reward),
                ("solved", vel_reward >= 1.0),
                ("done", self._get_done()),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def get_randomized_initial_state(self):
        # randomly start with flexed left or right knee
        if self.np_random.uniform() < 0.5:
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

    def step(self, *args, **kwargs):
        results = super().step(*args, **kwargs)
        self.steps += 1
        return results

    def reset(self, **kwargs):
        self.steps = 0
        if self.reset_type == "random":
            qpos, qvel = self.get_randomized_initial_state()
        elif self.reset_type == "init":
            qpos, qvel = self.sim.model.key_qpos[2], self.sim.model.key_qvel[2]
        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return obs

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
        foot_id_l = self.sim.model.body_name2id("talus_l")
        foot_id_r = self.sim.model.body_name2id("talus_r")
        return np.array(
            [
                self.sim.data.body_xpos[foot_id_l][2],
                self.sim.data.body_xpos[foot_id_r][2],
            ]
        )

    def _get_feet_relative_position(self):
        """
        Get the feet positions relative to the pelvis.
        """
        foot_id_l = self.sim.model.body_name2id("talus_l")
        foot_id_r = self.sim.model.body_name2id("talus_r")
        pelvis = self.sim.model.body_name2id("pelvis")
        return np.array(
            [
                self.sim.data.body_xpos[foot_id_l] - self.sim.data.body_xpos[pelvis],
                self.sim.data.body_xpos[foot_id_r] - self.sim.data.body_xpos[pelvis],
            ]
        )

    def _get_vel_reward(self):
        """
        Gaussian that incentivizes a walking velocity. Going
        over only achieves flat rewards.
        """
        vel = self._get_com_velocity()
        return np.exp(-np.square(self.target_y_vel - vel[1])) + np.exp(
            -np.square(self.target_x_vel - vel[0])
        )

    def _get_cyclic_rew(self):
        """
        Cyclic extension of hip angles is rewarded to incentivize a walking gait.
        """
        phase_var = (self.steps / self.hip_period) % 1
        des_angles = np.array(
            [
                0.8 * np.cos(phase_var * 2 * np.pi + np.pi),
                0.8 * np.cos(phase_var * 2 * np.pi),
            ],
            dtype=np.float32,
        )
        angles = self._get_angle(["hip_flexion_l", "hip_flexion_r"])
        return np.linalg.norm(des_angles - angles)

    def _get_ref_rotation_rew(self):
        """
        Incentivize staying close to the initial reference orientation up to a certain threshold.
        """
        target_rot = [
            self.target_rot if self.target_rot is not None else self.init_qpos[3:7]
        ][0]
        return np.exp(-np.linalg.norm(5.0 * (self.sim.data.qpos[3:7] - target_rot)))

    def _get_torso_angle(self):
        body_id = self.sim.model.body_name2id("torso")
        return self.sim.data.body_xquat[body_id]

    def _get_com_velocity(self):
        """
        Compute the center of mass velocity of the model.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = -self.sim.data.cvel
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
        com = self.sim.data.xipos
        return np.sum(mass * com, 0) / np.sum(mass)

    def _get_angle(self, names):
        """
        Get the angles of a list of named joints.
        """
        return np.array(
            [
                self.sim.data.qpos[
                    self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(name)]
                ]
                for name in names
            ]
        )

class WalkEnvV2(BaseV0):
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
    DEFAULT_RWD_KEYS_AND_WEIGHTS = load_reward_weights_from_yaml()


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
        # act_mag = np.linalg.norm(obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0 # 计算动作幅值成本（Action Magnitude Cost）
        if self.sim.model.na != 0 and 'act' in obs_dict:
            act_mag = float(np.linalg.norm(obs_dict['act']) / self.sim.model.na)
        else:
            act_mag = 0.0
        # 终止信号与存活奖励
        done = self._get_done()
        alive = float(self.steps)
        centerline = self._get_track_centerline_rew()  # 计算沿中心线奖励
        #  构建奖励字典
        rwd_dict = collections.OrderedDict((
            # Custom Keys
            ('centerline', centerline),
            ('act_mag', act_mag),
            ('alive', alive),
            # Optional Keys
            ('vel_reward', vel_reward),
            ('cyclic_hip',  cyclic_hip),
            ('ref_rot',  ref_rot),
            ('joint_angle_rew', joint_angle_rew),
            # Must keys
            ('sparse',  vel_reward),
            ('solved',    vel_reward >= 1.0),
            ('done',  self._get_done()),
        ))
        # 计算综合奖励
        # for key in self.rwd_keys_wt.keys():
        #     val = rwd_dict[key]
        #     print(f"[DEBUG] reward[{key}]: {val}, type: {type(val)}, shape: {np.shape(val)}")
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

    def step(self, *args, **kwargs):
        results = super().step(*args, **kwargs)
        self.steps += 1
        return results

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
        return obs

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

    def _get_track_centerline_rew(self):
        pelvis_x = self.sim.data.body_xpos[self.sim.model.body_name2id('pelvis')][0]
        return np.exp(-5.0 * pelvis_x ** 2).item()
    
    def _get_head_stability_rew(self, sigma_v=0.25, sigma_r=0.1):
        """
        计算头部稳定性奖励：
        - 头部速度越平稳（Δv越小）奖励越高
        - 姿态越接近初始方向，奖励越高
        r = exp( - ||Δv||^2 / σ_v - ||θ||^2 / σ_r )
        """
        head_id = self.sim.model.body_name2id("head")

        # ===== Δv: 线速度变化率 =====
        if not hasattr(self, "_prev_head_vel"):
            self._prev_head_vel = np.zeros(3)

        curr_vel = self.sim.data.cvel[head_id][3:]   # 当前头部线速度
        delta_v = curr_vel - self._prev_head_vel     # 速度变化量
        self._prev_head_vel = curr_vel.copy()        # 保存供下次使用

        v_err = np.linalg.norm(delta_v)              # Δv 模长

        # ===== θ: 姿态误差（与初始参考姿态） =====
        current_quat = self.sim.data.body_xquat[head_id]
        target_quat = self.init_qpos[3:7]  # 使用初始 pelvis 四元数作为目标方向

        # 归一化
        current_quat = current_quat / np.linalg.norm(current_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)

        cos_theta = np.abs(np.dot(current_quat, target_quat))  # 防止翻转对称性干扰
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_diff = 2 * np.arccos(cos_theta)  # 四元数差异角度（弧度）

        # ===== 最终奖励 =====
        reward = np.exp(-(v_err**2 / sigma_v) - (angle_diff**2 / sigma_r))
        return float(reward)


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

# 特调奖励函数环境
class WalkEnvV4(BaseV0):
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
    DEFAULT_RWD_KEYS_AND_WEIGHTS = load_reward_weights_from_yaml()


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
        joint_angle_rew = self._get_joint_angle_rew(['hip_adduction_l', 'hip_adduction_r', 'hip_rotation_l',
                                                       'hip_rotation_r']) # 计算关节角度惩罚/奖励
        # act_mag = np.linalg.norm(obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0 # 计算动作幅值成本（Action Magnitude Cost）
        if self.sim.model.na != 0 and 'act' in obs_dict:
            act_mag = float(np.linalg.norm(obs_dict['act']) / self.sim.model.na)
        else:
            act_mag = 0.0
        # 终止信号与存活奖励
        done = self._get_done()
        centerline = self._get_track_centerline_rew()  # 计算沿中心线奖励
        head_stability = self._get_head_stability_rew() # (0,1]
        # alive = float(self.steps)
        # ref_rot = self._get_ref_rotation_rew() # 计算参考旋转奖励
        
        #  构建奖励字典
        rwd_dict = collections.OrderedDict((
            # Custom Keys
            ('centerline', centerline),
            ('act_mag', act_mag),
            ('head_stability', head_stability),
            # ('alive', alive),
            # Optional Keys
            ('vel_reward', vel_reward),
            ('cyclic_hip',  cyclic_hip),
            # ('ref_rot',  ref_rot),
            ('joint_angle_rew', joint_angle_rew),
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

    def step(self, *args, **kwargs):
        results = super().step(*args, **kwargs)
        self.steps += 1
        return results

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
        return obs

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

    def _get_track_centerline_rew(self):
        pelvis_x = self.sim.data.body_xpos[self.sim.model.body_name2id('pelvis')][0]
        return np.exp(-10.0 * pelvis_x ** 2).item()
    
    def _get_head_stability_rew(self, sigma_v=0.25, sigma_r=0.1):
        """
        计算头部稳定性奖励：
        - 头部速度越平稳（Δv越小）奖励越高
        - 姿态越接近初始方向，奖励越高
        r = exp( - ||Δv||^2 / σ_v - ||θ||^2 / σ_r )
        """
        head_id = self.sim.model.body_name2id("head")

        # ===== Δv: 线速度变化率 =====
        if not hasattr(self, "_prev_head_vel"):
            self._prev_head_vel = np.zeros(3)

        curr_vel = self.sim.data.cvel[head_id][3:]   # 当前头部线速度
        delta_v = curr_vel - self._prev_head_vel     # 速度变化量
        self._prev_head_vel = curr_vel.copy()        # 保存供下次使用

        v_err = np.linalg.norm(delta_v)              # Δv 模长

        # ===== θ: 姿态误差（与初始参考姿态） =====
        current_quat = self.sim.data.body_xquat[head_id]
        target_quat = self.init_qpos[3:7]  # 使用初始 pelvis 四元数作为目标方向

        # 归一化
        current_quat = current_quat / np.linalg.norm(current_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)

        cos_theta = np.abs(np.dot(current_quat, target_quat))  # 防止翻转对称性干扰
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_diff = 2 * np.arccos(cos_theta)  # 四元数差异角度（弧度）

        # ===== 最终奖励 =====
        reward = np.exp(-(v_err**2 / sigma_v) - (angle_diff**2 / sigma_r))
        return float(reward)

# 特调奖励函数环境 v5：12个髋膝踝关节模仿 CMU mocap 数据
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
        # --- 主模仿奖励：12个 DOF 的相位傅立叶关节轨迹 ---
        imit_pose = self._get_imit_pose_rew()

        # --- 其他“约束型”奖励 ---
        vel_reward = self._get_vel_reward()          # 前进速度约束
        joint_angle_rew = self._get_joint_angle_rew(
            ['hip_adduction_l', 'hip_adduction_r', 'hip_rotation_l', 'hip_rotation_r']
        )                                            # “不要乱甩腿”约束
        progress = self._get_progress_rew()            # 前进进度奖励
        # 肌肉激活幅值成本
        if self.sim.model.na != 0 and "act" in self.obs_dict:
            act = self.obs_dict["act"]
            # 版本 A：RMS 激活
            act_rms = np.linalg.norm(act, axis=-1) / np.sqrt(self.sim.model.na)
            act_mag = act_rms  # 越大越“费力”
        else:
            act_mag = 0.0

        # 终止、中心线、头稳定
        done = self._get_done()
        centerline = self._get_track_centerline_rew()
        head_stability = self._get_head_stability_rew()

        #  构建奖励字典
        rwd_dict = collections.OrderedDict((
            # 主模仿奖励
            ('imit_pose', imit_pose),

            # 约束与正则项
            ('centerline', centerline),
            ('act_mag', act_mag),
            ('head_stability', head_stability),
            ('progress', progress),

            # 原有项
            ('vel_reward', vel_reward),

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

    # def _get_vel_reward(self):
    #     """
    #     Gaussian that incentivizes a walking velocity.
    #     """
    #     vel = self._get_com_velocity()
    #     return np.exp(-np.square(self.target_y_vel - vel[1])) + \
    #            np.exp(-np.square(self.target_x_vel - vel[0]))

    def _get_vel_reward(self, sigma_y=0.7, sigma_x=0.25):
        """
        y 为前进方向：
        - vy 越接近 target_y_vel 奖励越大
        - vy<=0 时奖励接近 0（平滑门控，不用硬阈值）
        - vx 约束在 0 附近（防侧滑）
        返回约在 [0, 1]
        """
        vel = self._get_com_velocity()
        vx = float(vel[0])  # 侧向
        vy = float(vel[1])  # 前向（y）

        # 向前门控：vy<=0 时为 0；vy 增大时平滑趋近 1
        vy_pos = max(0.0, vy)
        gate = np.tanh(vy_pos / max(self.target_y_vel, 1e-6))  # 0~1

        # 目标速度跟踪：绝对误差高斯（好调）
        track = np.exp(-((vy - self.target_y_vel) / sigma_y) ** 2)

        # 侧向约束：vx 越接近 0 越好（target_x_vel 通常为 0）
        track_x = np.exp(-((vx - self.target_x_vel) / sigma_x) ** 2)

        return float(gate * track * track_x)


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
        cvel = -self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]
    
    def _get_vel_reward(self):
        """
        强调：向前走才有分，且速度越接近 target_y_vel 越好。
        - vy < 0 基本视为负奖励（往回走）
        - vy 在 [0, target_y_vel] 内，按比例给基础分
        - 再叠加一个 Gaussian，鼓励接近目标速度
        - 侧向速度 vx 用一个 Gaussian 约束在 0 附近
        返回值大致在 [0, 1] 范围
        """
        vel = self._get_com_velocity()
        vx = float(vel[0])  # 侧向
        vy = float(vel[1])  # 前向（你现在 target_y_vel=1.2）

        # 1) 只要往后走就直接给接近 0
        if vy <= 0.0:
            base = 0.0
        else:
            # 先按比例，把 [0, target_y_vel] 映到 [0,1]，超过 target 就截断
            base = np.clip(vy / max(self.target_y_vel, 1e-6), 0.0, 1.0)

        # 2) 再乘一个“跟踪误差”的 Gaussian，窗口窄一点
        #    vy 偏差超过 50% 以后 reward 明显下降
        rel_err = (vy - self.target_y_vel) / max(self.target_y_vel, 1e-6)
        track = np.exp(- (rel_err ** 2) / 0.25)   # 0.5^2=0.25，对 ±50% 偏差容忍

        # 3) 侧向约束：vx 越接近 0 越好
        lateral = np.exp(- 5.0 * (vx ** 2))

        vel_reward = base * track * lateral  # ∈ [0,1]
        return float(vel_reward)
    
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
    
    def _get_progress_rew(self, clip=0.01):
        y = self.sim.data.subtree_com[0][1]   # 或你自己的com_y
        if not hasattr(self, "_prev_com_y"):
            self._prev_com_y = y
        dy = y - self._prev_com_y
        self._prev_com_y = y
        # 正向位移奖励（裁剪+归一）
        return np.clip(dy, 0.0, clip) / clip


# class TerrainEnvV0(WalkEnvV0):
