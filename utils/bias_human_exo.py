# -*- coding: utf-8 -*-
"""
bias_human_exo.py

目标：
1) 画出 model.py(HipAngleCPG + PaperParams) “默认参数下”的左右髋输出曲线
2) 与 params_12_bias.py(CMUParams) 的 hip_flexion_l / hip_flexion_r 对比
3) 通过扫描 Δ(cycle) 找到让两者相位最对齐的偏置 Δ*

用法：
python /home/lvchen/body_SB3/utils/bias_human_exo.py
"""

import importlib.util
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import types

# ========= 需要你按实际路径改这里 =========
CPG_DIR     = Path("/home/lvchen/body_SB3/cpg")                     # 包含 model.py / params.py / params_12_bias.py 的目录
PARAMS_12_PY = Path("/home/lvchen/body_SB3/cpg/params_12_bias.py")  # 你的 params_12_bias.py（CMUParams 在这里）
CSV_PATH     = Path("/home/lvchen/body_SB3/cpg/fourier_coefficients_order6_132.csv")
# =========================================


def load_cpg_modules(cpg_dir: Path, pkg_name: str = "cpg_runtime_pkg"):
    """
    把 cpg_dir 伪装成一个包 pkg_name，并按 pkg_name.params / pkg_name.model 的名字加载，
    这样 model.py 里的 `from .params import PaperParams` 就能正常工作。
    """
    cpg_dir = Path(cpg_dir).resolve()

    # 1) 创建伪包（关键：__path__）
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(cpg_dir)]  # 标记为 package
        sys.modules[pkg_name] = pkg

    def _load(subname: str, filename: str):
        full_name = f"{pkg_name}.{subname}"
        file_path = cpg_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"找不到文件: {file_path}")

        spec = importlib.util.spec_from_file_location(full_name, str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法创建 spec: {full_name} <- {file_path}")

        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
        return mod

    # 注意顺序：先 params 再 model（model 里要 import .params）
    params_mod = _load("params", "params.py")
    model_mod  = _load("model",  "model.py")
    return model_mod, params_mod


def load_module(path: Path, name: str):
    """用于加载不含相对导入的独立 .py（例如 params_12_bias.py）。"""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def zscore(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    return x / (x.std() + eps)


def roll_by_cycle(y, delta_cycle):
    """
    把 y(phase) 按 delta_cycle 平移；delta_cycle>0 表示相位向前（提前）。
    实现：phase -> phase + delta_cycle
    """
    y = np.asarray(y)
    N = y.shape[0]
    k = int(np.round((delta_cycle % 1.0) * N))
    return np.roll(y, -k)


def maybe_deg2rad_for_paper_params(p):
    """
    PaperParams 的 hip 系数单位在不同项目里可能是“度”或“弧度”。
    这里用一个温和的经验判断：如果系数量级明显大于 2π，则更像“度”，做 deg->rad。
    若你非常确定单位，请直接删掉这段自动判断，手动转换/不转换。
    """
    vals = []
    vals.append(float(getattr(p, "hip_a0", 0.0)))
    a = np.asarray(getattr(p, "hip_a", np.zeros(0)), dtype=float).ravel()
    b = np.asarray(getattr(p, "hip_b", np.zeros(0)), dtype=float).ravel()
    vals += a.tolist()
    vals += b.tolist()
    vmax = float(np.max(np.abs(vals))) if len(vals) else 0.0

    if vmax > 2.0 * np.pi + 1e-6:  # 更像“度”
        deg2rad = np.pi / 180.0
        p.hip_a0 = float(p.hip_a0) * deg2rad
        p.hip_a  = np.asarray(p.hip_a, dtype=float) * deg2rad
        p.hip_b  = np.asarray(p.hip_b, dtype=float) * deg2rad
        return True, vmax
    return False, vmax


def eval_model_default_hip_curves(cpg, p, phase, use_default_amp_offset=True):
    """
    用 model.py 内部一致的方式构造输出曲线：
      base(phi) = hip_fourier(phi)
      shape = base - a0
      q = offset + amp * shape

    右腿相对左腿做 π 相位差（等价于 cycle 上 +0.5）。
    """
    phase = np.asarray(phase, dtype=float)
    phi = 2.0 * np.pi * phase

    # 单侧基形（左）
    base_L = np.array([cpg.hip_fourier(ph) for ph in phi], dtype=float)
    # 右腿相位 +π
    base_R = np.array([cpg.hip_fourier(ph + np.pi) for ph in phi], dtype=float)

    a0 = float(getattr(p, "hip_a0"))
    shape_L = base_L - a0
    shape_R = base_R - a0

    if use_default_amp_offset:
        amp = np.asarray(cpg.amp, dtype=float).ravel()
        off = np.asarray(cpg.offset, dtype=float).ravel()
        if amp.size == 1:
            amp = np.repeat(amp, 2)
        if off.size == 1:
            off = np.repeat(off, 2)
        qL = off[0] + amp[0] * shape_L
        qR = off[1] + amp[1] * shape_R
    else:
        # 只看相位形状（与你旧脚本一致）
        qL = shape_L
        qR = shape_R

    return qL, qR


def main():
    # 中文显示（如缺字体可注释掉）
    plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 1) 以“伪包”方式加载 model.py/params.py（避免相对导入报错）
    model_mod, params_mod = load_cpg_modules(CPG_DIR)

    HipAngleCPG = model_mod.HipAngleCPG
    PaperParams = params_mod.PaperParams

    # 2) 加载 params_12_bias.py（它是独立文件，不要伪包也行）
    params12_mod = load_module(PARAMS_12_PY, "params12_mod")
    CMUParams = params12_mod.CMUParams

    cmu = CMUParams(csv_path=str(CSV_PATH), phase_bias=0.75)  # 按你当前参考设置
    # 如果你要对齐到“无偏置”的原始参考，把 phase_bias=0.0（见 params_12_bias.py 的 wrap 逻辑）:
    # cmu = CMUParams(csv_path=str(CSV_PATH), phase_bias=0.0)

    # 3) 实例化 PaperParams + CPG（用于调用 hip_fourier、并取默认 amp/offset）
    p = PaperParams()
    converted, vmax = maybe_deg2rad_for_paper_params(p)

    cpg = HipAngleCPG(p, dt=0.02, phase0=0.0, S_hip=6)

    # 4) 采样相位
    N = 2000
    phase = np.linspace(0.0, 1.0, N, endpoint=False)

    # 参考曲线（来自 CSV 的 hip_flexion_l/r）
    q_ref_L = cmu.eval_joint_phase("hip_flexion_l", phase)
    q_ref_R = cmu.eval_joint_phase("hip_flexion_r", phase)

    # model 默认输出曲线（真正“默认参数下”，不是占位全 0）
    q_model_L, q_model_R = eval_model_default_hip_curves(
        cpg, p, phase, use_default_amp_offset=True
    )

    # --- 基本诊断：如果这里仍接近 0，就说明 PaperParams 的 hip 系数本身就是 0 或没被正确设置 ---
    print(f"[PaperParams] coeff max|.|(before auto convert) ~= {vmax:.6f} ; applied deg->rad? {converted}")
    print(f"[ModelCurve] std(L)={np.std(q_model_L):.6e}, std(R)={np.std(q_model_R):.6e}")

    # 5) 搜索 Δ（cycle）让 model 与 ref 相位最对齐
    refL_n = zscore(q_ref_L)
    refR_n = zscore(q_ref_R)
    modL_n = zscore(q_model_L)
    modR_n = zscore(q_model_R)

    deltas = np.linspace(0.0, 1.0, N, endpoint=False)
    best = {"delta": 0.0, "score": -1e9}
    for d in deltas:
        mL = roll_by_cycle(modL_n, d)
        mR = roll_by_cycle(modR_n, d)
        score = float(np.dot(mL, refL_n) + np.dot(mR, refR_n))
        if score > best["score"]:
            best["score"] = score
            best["delta"] = float(d)

    delta_star = best["delta"]
    print(f"[OK] 最优相位偏置 Δ* = {delta_star:.6f} cycle  (phi_bias = {2*np.pi*delta_star:.6f} rad)")

    # 6) 可视化：原始 + 偏置后
    mL_best = roll_by_cycle(q_model_L, delta_star)
    mR_best = roll_by_cycle(q_model_R, delta_star)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(phase, q_ref_L, label="参考 hip_flexion_l")
    axes[0].plot(phase, q_model_L, "--", label="model 左髋（默认）")
    axes[0].plot(phase, mL_best, ":", label=f"model 左髋（平移 Δ*={delta_star:.3f}）")
    axes[0].set_title("左髋：参考 vs model（默认输出）")
    axes[0].set_ylabel("角度（单位随参数，通常 rad）")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(phase, q_ref_R, label="参考 hip_flexion_r")
    axes[1].plot(phase, q_model_R, "--", label="model 右髋（默认）")
    axes[1].plot(phase, mR_best, ":", label=f"model 右髋（平移 Δ*={delta_star:.3f}）")
    axes[1].set_title("右髋：参考 vs model（默认输出）")
    axes[1].set_xlabel("相位 phase（cycle）")
    axes[1].set_ylabel("角度（单位随参数，通常 rad）")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
