import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# =========================
# 配置：XML 路径
# =========================
xml_path = '/home/lvchen/miniconda3/envs/myosuite/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml'
# xml_path = "/home/lvchen/body_SB3_11-7/resources/exo/hip_exo.xml"

# 输出图文件目录
out_dir = "./mujoco_logs"
os.makedirs(out_dir, exist_ok=True)

# 你在 XML <sensor> 里定义的名字（与问题中一致）
SENSOR_NAMES = {
    "rr_pos": "rr_joint_pos",
    "rr_vel": "rr_joint_vel",
    "lr_pos": "lr_joint_pos",
    "lr_vel": "lr_joint_vel",
    "rr_tau": "rr_act_torque",
    "lr_tau": "lr_act_torque",
}

# 关节/执行器名字（用于“传感器不存在时”的回退读取）
JOINT_NAMES = {"rr": "rr_joint", "lr": "lr_joint"}
ACTUATOR_NAMES = {"rr": "exo_hip_r", "lr": "exo_hip_l"}


def _sensor_adr(model: mujoco.MjModel, name: str):
    """返回 (adr, dim)；若不存在则返回 None。"""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid == -1:
        return None
    adr = int(model.sensor_adr[sid])
    dim = int(model.sensor_dim[sid])
    return adr, dim


def _joint_qpos_adr(model: mujoco.MjModel, joint_name: str):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid == -1:
        return None
    # 每个 joint 的 qpos 起始地址
    return int(model.jnt_qposadr[jid])


def _joint_qvel_adr(model: mujoco.MjModel, joint_name: str):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid == -1:
        return None
    return int(model.jnt_dofadr[jid])


def _actuator_id(model: mujoco.MjModel, actuator_name: str):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if aid == -1:
        return None
    return int(aid)


def main():
    if not os.path.exists(xml_path):
        print(f"错误: 找不到文件 '{xml_path}'")
        return

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # -------------------------
    # 解析传感器地址（优先用 sensor）
    # -------------------------
    sens_map = {}
    for k, nm in SENSOR_NAMES.items():
        sens_map[k] = _sensor_adr(model, nm)

    have_all_sensors = all(v is not None for v in sens_map.values())
    if have_all_sensors:
        print("检测到全部所需传感器：将使用 data.sensordata 记录。")
    else:
        missing = [k for k, v in sens_map.items() if v is None]
        print(f"警告：缺少传感器 {missing}，将尝试回退用 joint/actuator 直接读取。")

    # 回退地址（如果传感器缺失）
    rr_qpos_adr = _joint_qpos_adr(model, JOINT_NAMES["rr"])
    lr_qpos_adr = _joint_qpos_adr(model, JOINT_NAMES["lr"])
    rr_qvel_adr = _joint_qvel_adr(model, JOINT_NAMES["rr"])
    lr_qvel_adr = _joint_qvel_adr(model, JOINT_NAMES["lr"])
    rr_aid = _actuator_id(model, ACTUATOR_NAMES["rr"])
    lr_aid = _actuator_id(model, ACTUATOR_NAMES["lr"])

    # 若传感器缺失且回退也拿不到，提前提示
    if not have_all_sensors:
        if rr_qpos_adr is None or lr_qpos_adr is None or rr_qvel_adr is None or lr_qvel_adr is None:
            print("注意：回退读取 joint 状态失败（未找到 rr_joint/lr_joint）。")
        if rr_aid is None or lr_aid is None:
            print("注意：回退读取 actuator 力失败（未找到 exo_hip_r/exo_hip_l）。")

    # -------------------------
    # 记录缓冲区
    # -------------------------
    ts = []
    rr_pos, rr_vel, rr_tau = [], [], []
    lr_pos, lr_vel, lr_tau = [], [], []

    dt = float(model.opt.timestep) if model.opt.timestep > 0 else 0.001

    # -------------------------
    # 启动 viewer（passive）并自行推进仿真
    # -------------------------
    print(f"正在为 '{xml_path}' 启动 MuJoCo Viewer（关闭窗口后自动出图）...")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                mujoco.mj_step(model, data)

                # 时间戳
                ts.append(float(data.time))

                if have_all_sensors:
                    # 从 sensordata 读
                    adr, _ = sens_map["rr_pos"]; rr_pos.append(float(data.sensordata[adr]))
                    adr, _ = sens_map["rr_vel"]; rr_vel.append(float(data.sensordata[adr]))
                    adr, _ = sens_map["lr_pos"]; lr_pos.append(float(data.sensordata[adr]))
                    adr, _ = sens_map["lr_vel"]; lr_vel.append(float(data.sensordata[adr]))
                    adr, _ = sens_map["rr_tau"]; rr_tau.append(float(data.sensordata[adr]))
                    adr, _ = sens_map["lr_tau"]; lr_tau.append(float(data.sensordata[adr]))
                else:
                    # 回退：joint qpos/qvel + data.actuator_force
                    rr_pos.append(float(data.qpos[rr_qpos_adr]) if rr_qpos_adr is not None else np.nan)
                    lr_pos.append(float(data.qpos[lr_qpos_adr]) if lr_qpos_adr is not None else np.nan)
                    rr_vel.append(float(data.qvel[rr_qvel_adr]) if rr_qvel_adr is not None else np.nan)
                    lr_vel.append(float(data.qvel[lr_qvel_adr]) if lr_qvel_adr is not None else np.nan)

                    # MuJoCo: data.actuator_force 是执行器产生的“广义力”（对关节的力矩/力）
                    rr_tau.append(float(data.actuator_force[rr_aid]) if rr_aid is not None else np.nan)
                    lr_tau.append(float(data.actuator_force[lr_aid]) if lr_aid is not None else np.nan)

                viewer.sync()

                # 尝试按 real-time 运行（不严格）
                elapsed = time.time() - step_start
                sleep_t = dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

    except Exception as e:
        print(f"运行或启动 viewer 出错: {e}")
        return

    # -------------------------
    # viewer 关闭后：出图
    # -------------------------
    ts_np = np.array(ts)
    rr_pos_np = np.array(rr_pos); lr_pos_np = np.array(lr_pos)
    rr_vel_np = np.array(rr_vel); lr_vel_np = np.array(lr_vel)
    rr_tau_np = np.array(rr_tau); lr_tau_np = np.array(lr_tau)

    def plot_two_curves(x, y1, y2, title, ylab, save_name):
        plt.figure()
        plt.plot(x, y1, label="Right (rr)")
        plt.plot(x, y2, label="Left (lr)")
        plt.xlabel("Time (s)")
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"已保存: {save_path}")

    plot_two_curves(ts_np, rr_pos_np, lr_pos_np,
                    "Hip Joint Position", "Position (rad)",
                    "hip_pos.png")
    plot_two_curves(ts_np, rr_vel_np, lr_vel_np,
                    "Hip Joint Velocity", "Velocity (rad/s)",
                    "hip_vel.png")
    plot_two_curves(ts_np, rr_tau_np, lr_tau_np,
                    "Hip Actuator Torque", "Torque (N·m)",
                    "hip_tau.png")

    plt.show()


if __name__ == "__main__":
    main()
