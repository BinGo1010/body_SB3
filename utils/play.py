# /home/lvchen/body_SB3/scripts/play_2.py

import os
import sys
import yaml
from pathlib import Path
from base64 import b64encode

import numpy as np
import myosuite
from IPython.display import HTML
from stable_baselines3 import PPO
import skvideo.io

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# =============================================================================
# 把项目根目录加入 sys.path（如果以后还要 import 自己写的包，会有用）
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 工具：在 Jupyter 里内嵌播放视频（命令行跑脚本可以不用）
# =============================================================================

def show_video(video_path, video_width=400):
    """
    在 Notebook 里内嵌一个 <video> 播放器。
    命令行跑脚本的话，这个函数可以不用。
    """
    with open(video_path, "rb") as f:
        video_file = f.read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(
        f"""<video autoplay loop muted width={video_width} controls>
                <source src="{video_url}">
            </video>"""
    )


# =============================================================================
# 工具：从 myosuite 环境拿一帧 RGB 图像
# =============================================================================

def get_frame(env, env_cfg, camera_id=None):
    """
    从 myosuite 环境里拿一帧 RGB 图像 (H, W, 3, uint8)

    - 优先使用 myosuite 的 sim.renderer.render_offscreen
    - 自动从 wrapper 里剥到最底层 env
    - 尺寸取自 YAML 的 env.render_size（默认 480x480）
    """
    base = env

    # 1) 往下剥 wrapper：Monitor / ObservationWrapper / TimeLimit / VecEnv 子环境等
    for _ in range(10):
        if hasattr(base, "env"):
            new_base = base.env
            if new_base is base:
                break
            base = new_base
        else:
            break

    # gymnasium / gym 的通用接口
    try:
        base = base.unwrapped
    except Exception:
        pass

    # 2) 通过 myosuite 的 sim.renderer.render_offscreen 拿离屏图像
    sim = getattr(base, "sim", None)
    renderer = getattr(sim, "renderer", None) if sim is not None else None

    if renderer is None or not hasattr(renderer, "render_offscreen"):
        print("[get_frame] 找不到 sim.renderer.render_offscreen，返回 None")
        return None

    # 从 YAML 里读渲染尺寸
    size = env_cfg.get("render_size", [480, 480])
    width, height = int(size[0]), int(size[1])

    # camera_id 可以从 YAML 的 env.render_camera 里读（若未指定则默认 0）
    if camera_id is None:
        camera_id = int(env_cfg.get("render_camera", 0))

    try:
        frame = renderer.render_offscreen(
            width=width,
            height=height,
            camera_id=camera_id,
        )
    except Exception as e:
        print(f"[get_frame] render_offscreen 失败: {type(e).__name__}: {e}")
        return None

    # 3) 转成 uint8，保证是 (H, W, 3)
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)

    # 有些版本返回 (3, H, W)，这里做一下通用处理
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))

    return frame


# =============================================================================
# YAML 读取
# =============================================================================

def load_yaml(path: Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML 文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


# =============================================================================
# 主函数：加载 best_model，rollout 一段，渲染成视频
# =============================================================================

def main():
    # 配置和模型路径
    config_path = PROJECT_ROOT / "configs" / "body_walk_config.yaml"
    policy_path = PROJECT_ROOT / "logs/walkenv/models/best/best_model.zip"

    if not config_path.is_file():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    if not policy_path.is_file():
        raise FileNotFoundError(f"未找到策略文件: {policy_path}")

    cfg = load_yaml(config_path)
    ecfg = cfg.get("env", {})

    # === 1. 创建环境（直接用 gym.make('myoLegWalk-v0')） ===
    env_id = ecfg.get("env_id", "myoLegWalk-v0")

    # 这里不需要 render_mode，因为我们走的是 sim.renderer.render_offscreen
    env = gym.make(env_id)

    # 如果训练时用的是 FlattenObservation，这里也要保持一致
    obs, _ = env.reset()
    if isinstance(obs, dict):
        env = FlattenObservation(env)
        obs, _ = env.reset()

    # === 2. 加载最优策略 ===
    # 如果你训练时用的是 GPU / CPU，这里会自动匹配当前设备
    model = PPO.load(str(policy_path))

    # === 3. rollout 一段轨迹，收集渲染帧 ===
    frames = []

    # 录制长度：可以用 max_episode_steps，也可以自己定
    max_steps = int(ecfg.get("max_episode_steps", 1000))

    for t in range(max_steps):
        # 用训练好的策略预测动作
        action, _ = model.predict(obs, deterministic=True)
        # cpg外骨骼
        obs, reward, terminated, truncated, info = env.step(action)

        frame = get_frame(env, ecfg)
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            break

    frames = np.asarray(frames)
    print("frames.shape =", frames.shape)

    if frames.size == 0:
        raise RuntimeError("没有采集到任何渲染帧，检查 get_frame 是否成功返回图像。")

    # === 4. 写视频 ===
    videos_dir = PROJECT_ROOT / "logs" / "walkenv" / "videos"
    os.makedirs(videos_dir, exist_ok=True)
    out_path = videos_dir / "walk.mp4"

    # skvideo 期望输入为 (T, H, W, 3)
    skvideo.io.vwrite(str(out_path), frames, outputdict={"-pix_fmt": "yuv420p"})
    print("saved to:", out_path)


if __name__ == "__main__":
    main()
