# jointdata2mat_batch.py
# 批量把 /home/lvchen/allasfamc/数字编号子目录 下的 asf+amc
# 转成包含 6 个关节(12 DOF) 的 qpos/qvel 的 .mat 文件
# 统一保存到 /home/lvchen/allasfamc/all_mat

import os
import numpy as np
from amc_parser import parse_asf, parse_amc
from scipy.io import savemat

# ========= 配置部分 =========
# 根目录：下面按数字命名的子目录里放 asf+amc
ROOT_DIR = "/home/lvchen/allasfamc/allasfamc/"          # 根据你的实际情况改
# 统一输出 .mat 的目录
OUT_DIR  = "/home/lvchen/allasfamc/all_mat/"            # 根据你的实际情况改

# 关键关节，6 个关节，共 12 个自由度
JOINT_ORDER = ['lfemur', 'rfemur', 'ltibia', 'rtibia', 'lfoot', 'rfoot']

FPS = 120.0
DT = 1.0 / FPS


def extract_qpos_qvel_from_amc(asf_path, amc_path, joint_order=JOINT_ORDER,
                               fps=FPS, dt=DT):
    """从单个 asf/amc 文件对中提取下肢 6 关节 12-DOF 的 qpos/qvel 序列"""

    # 解析骨架和动作
    joints = parse_asf(asf_path)
    motions = parse_amc(amc_path)

    if len(motions) == 0:
        raise RuntimeError(f"No frames in {amc_path}")

    # ---- 先用第 1 帧确定每一列的 DOF 标签 ----
    frame0 = motions[0]
    axis_names = ['rx', 'ry', 'rz']
    dim_labels = []
    cur_dim = 0

    for jname in joint_order:
        angles0 = frame0.get(jname, [])
        if isinstance(angles0, (int, float)):
            angles0 = [angles0]
        dof_num = len(angles0)
        for k in range(dof_num):
            dim_labels.append(f"{jname}_{axis_names[k]}")
        cur_dim += dof_num

    # ---- 提取角度序列（单位：rad）----
    qpos_list = []
    for frame in motions:
        qpos_frame = []
        for jname in joint_order:
            angles = frame.get(jname, [])
            if isinstance(angles, (int, float)):
                angles = [angles]
            qpos_frame.extend(angles)
        qpos_list.append(qpos_frame)

    qpos_seq_deg = np.array(qpos_list, dtype=float)   # T x DOF，单位：deg
    qpos_seq = np.deg2rad(qpos_seq_deg)               # 转成 rad

    # ---- 计算角速度（中心差分）----
    qvel_seq = np.gradient(qpos_seq, dt, axis=0)

    return qpos_seq, qvel_seq, dim_labels


def process_subject_folder(sub_dir, subject_name):
    """处理一个数字编号子目录：找到 asf + 多个 amc，分别生成 .mat"""

    print(f"\n=== 处理目录: {sub_dir} ===")

    # 找 asf
    asf_files = [f for f in os.listdir(sub_dir)
                 if f.lower().endswith(".asf")]
    if not asf_files:
        print("  [跳过] 没有 .asf 文件")
        return
    if len(asf_files) > 1:
        print(f"  [警告] 发现多个 .asf，默认为第一个: {asf_files[0]}")
    asf_path = os.path.join(sub_dir, asf_files[0])

    # 找所有 .amc
    amc_files = sorted(
        f for f in os.listdir(sub_dir)
        if f.lower().endswith(".amc")
    )
    if not amc_files:
        print("  [跳过] 没有 .amc 文件")
        return

    print(f"  使用 asf: {asf_files[0]}")
    print(f"  发现 {len(amc_files)} 个 amc 文件")

    for amc_name in amc_files:
        amc_path = os.path.join(sub_dir, amc_name)
        base_name, _ = os.path.splitext(amc_name)
        # 输出文件名：仍用原来的 base_name_lower_body.mat
        # base_name 本身就带 01_01 / 02_03 这种前缀，不必再加 subject
        mat_name = base_name + "_lower_body.mat"
        mat_path = os.path.join(OUT_DIR, mat_name)

        try:
            qpos_seq, qvel_seq, dim_labels = extract_qpos_qvel_from_amc(
                asf_path, amc_path
            )
        except Exception as e:
            print(f"  [错误] 处理 {amc_name} 失败: {e}")
            continue

        # 保存为 .mat
        mat_dict = {
            "qpos_seq": qpos_seq,
            "qvel_seq": qvel_seq,
            "joint_order": JOINT_ORDER,
            "dim_labels": np.array(dim_labels, dtype=object),
            "fps": FPS,
            "dt": DT,
            "subject": subject_name,
            "amc_file": amc_name,
        }
        savemat(mat_path, mat_dict)
        print(f"  [OK] {amc_name} -> {mat_name}, qpos shape={qpos_seq.shape}")


def main():
    # 确保输出目录存在
    os.makedirs(OUT_DIR, exist_ok=True)

    # 遍历 ROOT_DIR 下所有数字编号的子目录
    for name in sorted(os.listdir(ROOT_DIR)):
        sub_dir = os.path.join(ROOT_DIR, name)
        if not os.path.isdir(sub_dir):
            continue
        # 只处理纯数字命名的文件夹，如 "01", "02", ...
        if not name.isdigit():
            continue
        process_subject_folder(sub_dir, subject_name=name)


if __name__ == "__main__":
    main()
    print("\n全部处理完成。输出目录:", OUT_DIR)
