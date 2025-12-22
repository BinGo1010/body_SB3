import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from glob import glob

# 当前文件 utils/reward_functions.py -> 上两级就是 body_SB3/
BASE_DIR = Path(__file__).resolve().parents[1]
# reward_logs 根目录：body_SB3/logs/walkenv/reward_logs
REWARD_LOG_ROOT = BASE_DIR / "logs" / "walkenv" / "reward_logs"

WINDOW = 50  # 滑动窗口大小（用于平滑）

# 找到所有时间戳文件夹
subdirs = [d for d in glob(os.path.join(REWARD_LOG_ROOT, "*")) if os.path.isdir(d)]
if len(subdirs) == 0:
    raise FileNotFoundError("reward_logs 下没有任何训练记录文件夹。")
# 根据修改时间排序
latest = max(subdirs, key=os.path.getmtime)
CSV_PATH = os.path.join(latest, "reward_summary.csv")
print("使用的 reward_summary.csv 路径：", CSV_PATH)
OUT_DIR = os.path.join(latest, "plots")   # 图像输出目录自动跟随
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================
# 读取 CSV
# ==========================
df = pd.read_csv(CSV_PATH)

# 自动识别 reward 字段（排除 SB3 字段）
exclude_keywords = ["rollout", "train/", "time/", "eval/"]
reward_cols = []
for c in df.columns:
    if any(k in c for k in exclude_keywords):
        continue
    # 数值型字段才认为是 reward
    if df[c].dtype in [np.float64, np.float32, np.int64, float, int]:
        reward_cols.append(c)

print("识别到 reward 字段：", reward_cols)

# ==========================
# 1) 画 reward 曲线（原始曲线）
# ==========================
plt.figure(figsize=(14, 6))
for c in reward_cols:
    plt.plot(df[c], label=c)
plt.legend()
plt.title("Reward Curves (Raw)")
plt.xlabel("Steps")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/reward_raw.png", dpi=200)
plt.close()

# ==========================
# 2) 画 reward 平均趋势（滑动窗口 + EMA）
# ==========================
df_sma = df[reward_cols].rolling(WINDOW).mean()
df_ema = df[reward_cols].ewm(span=WINDOW).mean()

plt.figure(figsize=(14, 6))
for c in reward_cols:
    plt.plot(df_sma[c], label=f"{c} (SMA)")
plt.legend()
plt.title(f"Reward Moving Average (Window={WINDOW})")
plt.xlabel("Steps")
plt.ylabel("Mean Value")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/reward_moving_average.png", dpi=200)
plt.close()

plt.figure(figsize=(14, 6))
for c in reward_cols:
    plt.plot(df_ema[c], label=f"{c} (EMA)")
plt.legend()
plt.title(f"Reward EMA Trend (span={WINDOW})")
plt.xlabel("Steps")
plt.ylabel("Smoothed Value")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/reward_ema.png", dpi=200)
plt.close()

# ==========================
# 3) reward 贡献对比（堆叠占比图）
# ==========================
df_norm = df[reward_cols].clip(lower=0)  # 去掉负数，避免占比问题
df_norm = df_norm.div(df_norm.sum(axis=1) + 1e-9, axis=0)

plt.figure(figsize=(14, 7))
plt.stackplot(range(len(df_norm)), df_norm.T, labels=reward_cols)
plt.legend(loc='upper left')
plt.title("Reward Contribution Over Time (Stacked Proportion)")
plt.xlabel("Steps")
plt.ylabel("Proportion")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/reward_contribution_stacked.png", dpi=200)
plt.close()

# ==========================
# 4) 最后 1000 步 reward 的雷达图（占比分析）
# ==========================
last = df[reward_cols].tail(1000).mean()
angles = np.linspace(0, 2*np.pi, len(reward_cols), endpoint=False).tolist()
angles += angles[:1]
values = last.tolist()
values += values[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values, linewidth=2)
ax.fill(angles, values, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), reward_cols)
ax.set_title("Reward Contribution (Last 1000 Steps)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/reward_radar.png", dpi=200)
plt.close()

print(f"全部图像已保存到: {OUT_DIR}/")
