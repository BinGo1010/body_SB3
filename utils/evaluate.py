#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对 Stable-Baselines3 EvalCallback 生成的 evaluations.npz 画 3 张图：
1) 01_eval_return_vs_steps.png   ：平均评估回报 vs 总步数
2) 02_eval_ep_len_vs_steps.png   ：平均 episode 长度 vs 总步数（若有 ep_lengths）
3) 03_eval_return_curve.png      ：评估回报曲线（单 run）

特点：
- 支持多个 runs：对每个 evaluations.npz，分别在它的同级目录生成 01/02/03。
- --runs 可以是 npz 文件，或包含 evaluations.npz 的目录。
- --out 若指定，则所有图都输出到同一个 out 目录（文件名相同会被覆盖）。
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10


# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------

def load_eval_npz(npz_path: str) -> Dict[str, np.ndarray]:
    """
    读取 evaluations.npz，返回：
      - 'timesteps': (N_eval,)
      - 'results_mean': (N_eval,)
      - 'ep_len_mean': (N_eval,) or None
    """
    data = np.load(npz_path, allow_pickle=True)
    if "timesteps" not in data or "results" not in data:
        raise ValueError(f"{npz_path} 缺少必要键（timesteps/results）")

    timesteps = data["timesteps"].astype(np.int64)
    results = data["results"]
    if results.ndim == 1:
        results_mean = results.astype(np.float64)
    else:
        results_mean = results.mean(axis=1).astype(np.float64)

    if "ep_lengths" in data:
        ep_lengths = data["ep_lengths"]
        if ep_lengths.ndim == 1:
            ep_len_mean = ep_lengths.astype(np.float64)
        else:
            ep_len_mean = ep_lengths.mean(axis=1).astype(np.float64)
    else:
        ep_len_mean = None

    return {
        "timesteps": timesteps,
        "results_mean": results_mean,
        "ep_len_mean": ep_len_mean,
    }


def ensure_outdir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)


def resolve_run_paths(run_args: Optional[List[str]]) -> List[str]:
    """
    根据命令行参数解析 runs：
    - 若提供的是 npz 文件路径，直接使用；
    - 若提供的是目录，则在目录下优先找 evaluations.npz；
    - 若参数为空（None），则在 ./logs 以及当前目录下递归搜索所有 evaluations.npz。
    """
    paths: List[str] = []

    # 情况 1：用户显式提供了 runs
    if run_args:
        for r in run_args:
            p = Path(r).expanduser().resolve()
            if p.is_file() and p.suffix == ".npz":
                paths.append(str(p))
            elif p.is_dir():
                cand = p / "evaluations.npz"
                if cand.is_file():
                    paths.append(str(cand.resolve()))
                else:
                    # 退而求其次：该目录下所有 *.npz
                    npzs = sorted(p.glob("*.npz"))
                    if npzs:
                        paths.extend(str(x.resolve()) for x in npzs)
                    else:
                        print(f"[警告] 目录 {p} 下未找到任何 .npz 文件。")
            else:
                cand = p / "evaluations.npz"
                if cand.is_file():
                    paths.append(str(cand.resolve()))
                else:
                    print(f"[警告] 无法解析 runs 条目：{r}（既不是 .npz 文件也不是有效目录）")
        # 去重并排序
        paths = sorted(set(paths))
        return paths

    # 情况 2：用户未提供 runs，自动搜索 ./logs 下的 evaluations.npz
    cwd = Path.cwd()
    search_roots = []
    logs_dir = cwd / "logs"
    if logs_dir.is_dir():
        search_roots.append(logs_dir)
    search_roots.append(cwd)

    found = []
    for root in search_roots:
        for npz in root.rglob("evaluations.npz"):
            found.append(npz.resolve())

    paths = sorted(set(str(p) for p in found))
    return paths


# ------------------------------------------------------------
# 画图（单 run）
# ------------------------------------------------------------

def plot_single_run(npz_path: str, out_dir: str):
    """
    对单个 evaluations.npz，在 out_dir 里生成：
      01_eval_return_vs_steps.png
      02_eval_ep_len_vs_steps.png （若有 ep_len）
      03_eval_return_curve.png
    """
    d = load_eval_npz(npz_path)
    timesteps = d["timesteps"]
    results_mean = d["results_mean"]
    ep_len_mean = d["ep_len_mean"]

    ensure_outdir(out_dir)

    # 图 1：平均评估回报 vs 步数
    fig, ax = plt.subplots()
    ax.plot(timesteps, results_mean, linewidth=2)
    ax.set_title("Evaluation Return vs Total Timesteps")
    ax.set_xlabel("Total Timesteps")
    ax.set_ylabel("Eval Mean Return")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_eval_return_vs_steps.png"), dpi=200)
    plt.close(fig)

    # 图 2：平均 episode 长度 vs 步数
    if ep_len_mean is not None:
        fig, ax = plt.subplots()
        ax.plot(timesteps, ep_len_mean, linewidth=2)
        ax.set_title("Evaluation Episode Length vs Total Timesteps")
        ax.set_xlabel("Total Timesteps")
        ax.set_ylabel("Eval Episode Length (mean)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "02_eval_ep_len_vs_steps.png"), dpi=200)
        plt.close(fig)
    else:
        print(f"[提示] {npz_path} 中没有 ep_lengths，跳过 02 图。")

    # 图 3：评估回报曲线（跟图 1 类似，这里单独再画一张做对比或叠加）
    fig, ax = plt.subplots()
    ax.plot(timesteps, results_mean, linewidth=2)
    ax.set_title("Evaluation Return Curve")
    ax.set_xlabel("Total Timesteps")
    ax.set_ylabel("Eval Mean Return")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_eval_return_curve.png"), dpi=200)
    plt.close(fig)

    print(f"[完成] {npz_path} 已在 {out_dir} 生成 01/02/03 图。")


# ------------------------------------------------------------
# 主函数
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        help="多个 evaluations.npz 或目录路径；可省略（自动在 ./logs 下搜索 evaluations.npz）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出图片目录；若省略，则每个 npz 使用其所在目录作为输出目录。",
    )
    args = parser.parse_args()

    run_paths = resolve_run_paths(args.runs)
    if not run_paths:
        print("[错误] 未找到任何 evaluations.npz，请检查 --runs 参数或确保 ./logs 下有 eval 结果。")
        sys.exit(1)

    print("[信息] 找到的 evaluations.npz 列表：")
    for p in run_paths:
        print("   ", p)

    # 如果指定了 --out，则所有 runs 输出到同一个目录（会互相覆盖同名文件）
    # 如果没指定，则每个 npz 在自己的目录下生成 01/02/03
    if args.out is not None:
        global_out = Path(args.out).expanduser().resolve()
        ensure_outdir(str(global_out))
        for npz in run_paths:
            plot_single_run(npz, str(global_out))
    else:
        for npz in run_paths:
            out_dir = str(Path(npz).resolve().parent)
            plot_single_run(npz, out_dir)


if __name__ == "__main__":
    main()
