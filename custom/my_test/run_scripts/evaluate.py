"""
评估脚本: 使用 Diffusion Actor + 逆动力学模型在 MPE 环境中在线评估
==============================================================
功能:
  1. 加载训练好的 checkpoint
  2. 在 N 个 seed 上各跑 M 个 episode
  3. 生成 MP4 视频用于观察
  4. 输出评估指标 JSON
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# import custom modules
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from diffusion_actor import RiskGuidedDiffusion
from risk_measures import RiskMeasures


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_video_writer(video_dir: str, seed: int, fps: int = 5):
    """
    创建视频写入器: 使用 imageio 写入 MP4.
    返回一个函数: write_frame(render_result)  — render_result 是 (H, W, 3) np.uint8 图像
    以及 finalize() 函数.
    """
    import imageio

    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"seed_{seed}.mp4")
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8)

    def write_frame(frame: np.ndarray):
        """frame: (H, W, 3) uint8 RGB"""
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        writer.append_data(frame)

    def finalize():
        writer.close()
        print(f"    Video saved: {video_path}")

    return write_frame, finalize


def render_mpe_frame(env, mode="rgb_array"):
    """
    对 MPE 环境截取一帧 RGB 图像.
    修复: 优先使用 env.render("rgb_array"), 若 None 则尝试从 env.screen 直接读取像素.
    """
    try:
        frame = env.render(mode=mode)
    except TypeError:
        frame = env.render()

    # 如果 render_mode 设置正确, render() 应返回 (H, W, 3) ndarray
    if frame is not None and isinstance(frame, np.ndarray) and frame.ndim >= 2:
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        return frame

    # ---- fallback: 直接从 env.unwrapped.screen 读取 ----
    try:
        import pygame
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        screen = getattr(unwrapped, "screen", None)
        if screen is not None:
            frame = pygame.surfarray.array3d(screen)  # (W, H, 3)
            frame = np.transpose(frame, (1, 0, 2))   # (H, W, 3)
            return frame
    except Exception:
        pass

    # 最终 fallback: 空白帧 + 警告
    print("\n[Warn] render_mpe_frame: cannot obtain frame, returning blank")
    return np.zeros((480, 480, 3), dtype=np.uint8)


def evaluate(
    checkpoint_path: str,
    cfg: dict,
    device: str = "cuda",
    save_video: bool = True,
    video_dir: str = "videos",
):
    """
    在线评估。

    Args:
        checkpoint_path: 模型权重路径 (.pt)
        cfg: 完整 YAML 配置
        device: cuda / cpu
        save_video: 是否保存视频
        video_dir: 视频子目录

    Returns:
        dict: 评估结果
    """
    env_cfg = cfg["env"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluation"]
    path_cfg = cfg["paths"]

    n_seeds = eval_cfg["n_eval_seeds"]
    n_episodes = eval_cfg["n_eval_episodes"]
    target_return = eval_cfg["target_return"]

    # ---- 1. 构建模型 ----
    print(f"\n[Eval] Building model...")
    model = RiskGuidedDiffusion(
        horizon=model_cfg["horizon"],
        n_agents=env_cfg["n_agents"],
        obs_dim=env_cfg["obs_dim"],
        n_actions=env_cfg["n_actions"],
        history_horizon=model_cfg["history_horizon"],
        hidden_dim=model_cfg["hidden_dim"],
        n_timesteps=model_cfg["n_timesteps"],
        n_blocks=model_cfg["n_blocks"],
        use_returns_condition=model_cfg["use_returns_condition"],
        use_risk_guide=model_cfg["use_risk_guide"],
        condition_guidance_w=model_cfg["condition_guidance_w"],
        risk_guidance_w=model_cfg["risk_guidance_w"],
    )

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"])
        print(f"  Loaded EMA model from step {checkpoint.get('step', '?')}")
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print(f"  Loaded model from step {checkpoint.get('step', '?')}")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Loaded raw state_dict")
    model.to(device)
    model.eval()

    n_agents = env_cfg["n_agents"]
    obs_dim = env_cfg["obs_dim"]
    horizon = model_cfg["horizon"]
    history_horizon = model_cfg["history_horizon"]
    max_steps = env_cfg["max_path_length"]

    # ---- 2. 导入环境 ----
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from diffuser.datasets.mpe import make_env

    all_returns_sum = []
    all_returns_agents = []
    seed_returns = defaultdict(list)
    seed_returns_sum = defaultdict(list)

    save_dir = path_cfg["save_dir"].format(env_name=env_cfg["env_name"])
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    for seed_idx in range(n_seeds):
        seed = 100 * (seed_idx + 1)
        set_seed(seed)

        env = make_env(env_cfg["env_name"], render_mode="rgb_array")
        env.reset()

        # 视频写入器
        if save_video:
            vid_subdir = os.path.join(save_dir, video_dir)
            write_frame, finalize_video = make_video_writer(vid_subdir, seed)
        else:
            write_frame, finalize_video = None, lambda: None

        print(f"  Seed {seed}: ", end="", flush=True)

        for ep in range(n_episodes):
            obs = env.reset()
            done = np.array([False] * n_agents)
            ep_return = np.zeros(n_agents, dtype=np.float32)
            obs_history = []

            # 初始填充
            obs_ary = np.array(obs)  # [A, obs_dim]
            while len(obs_history) < history_horizon:
                obs_history.append(obs_ary)

            # 渲染初始帧
            if write_frame is not None:
                frame = render_mpe_frame(env)
                write_frame(frame)

            step_count = 0

            while (not np.all(done)) and step_count < max_steps:
                # 构建条件
                cond = np.stack(obs_history[-history_horizon:], axis=0)  # [H_hist, A, D]
                cond_t = torch.from_numpy(cond).float().to(device).unsqueeze(0)   # [1, H_hist, A, D]
                returns_t = torch.tensor([target_return], device=device).float()

                # 扩散采样 (无 risk gradient 时简化)
                with torch.no_grad():
                    traj = model.conditional_sample(
                        cond_t, returns_t, horizon=horizon
                    )  # [1, H_hist+H, A, D]

                # 逆动力学预测动作
                pred_obs = traj[0, history_horizon, :, :]   # [A, D]  (s_{t+1})
                cur_obs = obs_history[-1]                     # [A, D]  (s_t)
                s_t = torch.from_numpy(cur_obs).float().to(device).unsqueeze(0)
                s_t_1 = pred_obs.unsqueeze(0)
                pred_actions = model.inv_model(s_t, s_t_1)          # [1, A, action_dim]
                actions = pred_actions.detach()[0].cpu().numpy()  # [A, action_dim]
                actions = np.clip(actions, -1.0, 1.0)

                # 环境步进
                act_list = [actions[i] for i in range(n_agents)]
                next_obs, reward, done, info = env.step(act_list)
                obs_ary = np.array(next_obs)
                obs_history.append(obs_ary)
                if isinstance(reward, (list, np.ndarray)):
                    reward_arr = np.asarray(reward, dtype=np.float32)
                else:
                    reward_arr = np.full(n_agents, reward, dtype=np.float32)
                ep_return += reward_arr
                step_count += 1

                # 渲染帧
                if write_frame is not None:
                    frame = render_mpe_frame(env)
                    write_frame(frame)

                if np.all(done):
                    break

            seed_returns[seed].append(ep_return)
            seed_returns_sum[seed].append(float(ep_return.sum()))
            all_returns_agents.append(ep_return)
            all_returns_sum.append(float(ep_return.sum()))
            if (ep + 1) % 5 == 0:
                print(".", end="", flush=True)

        env.close()
        finalize_video()
        mean_sum = np.mean(seed_returns_sum[seed])
        print(f" {mean_sum:.1f}")

    # ---- 3. 统计 ----
    all_r_sum = np.array(all_returns_sum)
    all_r_agents = np.array(all_returns_agents)
    seed_stats = {}
    for seed, rets in seed_returns.items():
        rets_arr = np.stack(rets, axis=0)
        rets_sum = np.array(seed_returns_sum[seed])
        seed_stats[seed] = {
            "mean_per_agent": rets_arr.mean(axis=0).tolist(),
            "std_per_agent": rets_arr.std(axis=0).tolist(),
            "mean_sum": float(rets_sum.mean()),
            "std_sum": float(rets_sum.std()),
        }

    average_ep_reward = all_r_agents.mean(axis=0).tolist()
    std_ep_reward = all_r_agents.std(axis=0).tolist()
    overall_mean = float(all_r_sum.mean())
    overall_std = float(all_r_sum.std())
    print(f"\n  Final (sum): {overall_mean:.2f} ± {overall_std:.2f}")

    results = {
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "average_ep_reward": average_ep_reward,
        "std_ep_reward": std_ep_reward,
        "seed_stats": seed_stats,
        "all_returns_sum": all_r_sum.tolist(),
        "all_returns_agents": all_r_agents.tolist(),
        "checkpoint": checkpoint_path,
        "checkpoint_step": checkpoint.get("step", -1),
    }

    # 保存 JSON
    json_path = os.path.join(eval_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Results saved: {json_path}")

    return results


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Risk-Guided Diffusion")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_video", action="store_true",
                        help="Skip video generation")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

    set_seed(cfg["training"]["seed"])
    print(f"=== Risk-Guided Diffusion Evaluation ===")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {args.device}")

    results = evaluate(
        checkpoint_path=args.checkpoint,
        cfg=cfg,
        device=args.device,
        save_video=not args.no_video,
        video_dir=cfg["evaluation"].get("video_dir", "videos"),
    )

    print(f"\nDone. Mean return: {results['overall_mean']:.2f} ± {results['overall_std']:.2f}")


if __name__ == "__main__":
    main()
