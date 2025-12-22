# tools/gen_keyframe.py
import mujoco
import numpy as np
from pathlib import Path

XML_PATH = Path("/home/lvchen/miniconda3/envs/myosuite/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml")  # 按实际路径改

def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    print("nq =", model.nq, "nv =", model.nv)

    # 1) 默认初始状态（全 0 + model.xml 里设置的默认关节）
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()  # 一般全 0

    # 2) 你也可以在这里手动调整某些关节，生成第二个 key
    # 例如：让右髋屈曲 0.3 rad
    # hip_r_id = model.joint_name2id("hip_flexion_r")
    # qpos1 = qpos0.copy()
    # qpos1[model.jnt_qposadr[hip_r_id]] += 0.3

    def to_str(arr):
        return " ".join(f"{v:.8g}" for v in arr)

    print("\n===== 新的 keyframe 段，可以直接粘到 myolegs.xml 里 =====\n")
    print("<keyframe>")
    print(f"    <key name=\"init\" qpos='{to_str(qpos0)}' qvel='{to_str(qvel0)}'/>")
    # 如果要第二个 key，就再打印一行：
    # print(f"    <key name=\"hip_r_flexed\" qpos='{to_str(qpos1)}' qvel='{to_str(qvel0)}'/>")
    print("</keyframe>")

if __name__ == "__main__":
    main()
