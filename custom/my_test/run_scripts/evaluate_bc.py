#!/usr/bin/env python3
"""
Evaluate BC policy on MPE environments.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.bc_policy import BCPolicy
from normalizer import DatasetNormalizer, GaussianNormalizer, LimitsNormalizer


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
    import imageio

    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"seed_{seed}.mp4")
    writer = imageio.get_writer(
        video_path, fps=fps, codec="libx264", quality=8, macro_block_size=None
    )

    def write_frame(frame: np.ndarray):
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        writer.append_data(frame)

    def finalize():
        writer.close()
        print(f"    Video saved: {video_path}")

    return write_frame, finalize


def render_mpe_frame(env, mode="rgb_array"):
    try:
        frame = env.render(mode=mode)
    except TypeError:
        frame = env.render()

    if frame is not None and isinstance(frame, np.ndarray) and frame.ndim >= 2:
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        return frame

    try:
        import pygame

        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        screen = getattr(unwrapped, "screen", None)
        if screen is not None:
            frame = pygame.surfarray.array3d(screen)  # (W, H, 3)
            frame = np.transpose(frame, (1, 0, 2))  # (H, W, 3)
            return frame
    except Exception:
        pass

    print("\n[Warn] render_mpe_frame: cannot obtain frame, returning blank")
    return np.zeros((480, 480, 3), dtype=np.uint8)


def load_mpe_data(data_dir: str, n_agents: int, max_episodes: int = 5000):
    import glob

    seed_dirs = sorted(glob.glob(os.path.join(data_dir, "seed_*_data")))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed dirs found in {data_dir}")

    all_episodes = {"obs": [], "next_obs": [], "acs": [], "rews": []}
    for seed_dir in seed_dirs:
        obs_list, next_obs_list, acs_list, rews_list, dones_list = [], [], [], [], []
        for i in range(n_agents):
            obs_list.append(np.load(os.path.join(seed_dir, f"obs_{i}.npy")))
            next_obs_list.append(np.load(os.path.join(seed_dir, f"next_obs_{i}.npy")))
            acs_list.append(np.load(os.path.join(seed_dir, f"acs_{i}.npy")))
            rews_list.append(np.load(os.path.join(seed_dir, f"rews_{i}.npy")))
            dones_list.append(np.load(os.path.join(seed_dir, f"dones_{i}.npy")))

        obs = np.stack(obs_list, axis=1)  # [T, A, obs_dim]
        next_obs = np.stack(next_obs_list, axis=1)  # [T, A, obs_dim]
        acs = np.stack(acs_list, axis=1)  # [T, A, act_dim]
        rews = np.stack(rews_list, axis=1)  # [T, A]
        dones = np.stack(dones_list, axis=1)  # [T, A]

        T = obs.shape[0]
        start = 0
        for t in range(1, T):
            if dones[t].any() or (t - start) >= 50:
                ep_len = t - start
                if ep_len >= 4:
                    all_episodes["obs"].append(obs[start:t])
                    all_episodes["next_obs"].append(next_obs[start:t])
                    all_episodes["acs"].append(acs[start:t])
                    all_episodes["rews"].append(rews[start:t])
                start = t + 1
                if len(all_episodes["obs"]) >= max_episodes:
                    break
        if len(all_episodes["obs"]) >= max_episodes:
            break

    for k in all_episodes:
        all_episodes[k] = all_episodes[k][:max_episodes]
    print(f"  Loaded {len(all_episodes['obs'])} episodes")
    return all_episodes


def build_normalizer_from_state(state: dict) -> DatasetNormalizer:
    normalizer = DatasetNormalizer.__new__(DatasetNormalizer)
    normalizer.keys = state["keys"]
    normalizer.agent_share = state["agent_share"]
    normalizer.normalizers = {}

    for key in normalizer.keys:
        entry = state["normalizers"][key]
        if normalizer.agent_share:
            if entry["type"] == "LimitsNormalizer":
                n = LimitsNormalizer.__new__(LimitsNormalizer)
                n.mins = entry["mins"]
                n.maxs = entry["maxs"]
            elif entry["type"] == "GaussianNormalizer":
                n = GaussianNormalizer.__new__(GaussianNormalizer)
                n.mean = entry["mean"]
                n.std = entry["std"]
            else:
                raise ValueError(f"Unknown normalizer type: {entry['type']}")
            normalizer.normalizers[key] = n
        else:
            per_agent = []
            for sub in entry:
                if sub["type"] == "LimitsNormalizer":
                    n = LimitsNormalizer.__new__(LimitsNormalizer)
                    n.mins = sub["mins"]
                    n.maxs = sub["maxs"]
                elif sub["type"] == "GaussianNormalizer":
                    n = GaussianNormalizer.__new__(GaussianNormalizer)
                    n.mean = sub["mean"]
                    n.std = sub["std"]
                else:
                    raise ValueError(f"Unknown normalizer type: {sub['type']}")
                per_agent.append(n)
            normalizer.normalizers[key] = per_agent

    return normalizer


def build_normalizer_from_dataset(cfg: dict) -> DatasetNormalizer:
    env_cfg = cfg["env"]
    path_cfg = cfg["paths"]

    data_dir = path_cfg["data_dir"].format(
        env_name=env_cfg["env_name"],
        quality=env_cfg["quality"],
    )
    print(f"[Eval] Building normalizer from dataset: {data_dir}")
    episodes = load_mpe_data(data_dir, env_cfg["n_agents"], max_episodes=2000)
    all_obs = np.concatenate(episodes["obs"], axis=0)  # [T_total, A, obs_dim]
    all_acs = np.concatenate(episodes["acs"], axis=0)  # [T_total, A, act_dim]
    norm_dataset = {"obs": all_obs, "acs": all_acs}
    normalizer = DatasetNormalizer(
        norm_dataset,
        normalizer="LimitsNormalizer",
        keys=["obs", "acs"],
        agent_share=True,
    )
    return normalizer


def evaluate(
    checkpoint_path: str,
    cfg: dict,
    device: str = "cuda",
    save_video: bool = True,
    video_dir: str = "videos",
):
    env_cfg = cfg["env"]
    eval_cfg = cfg["evaluation"]
    path_cfg = cfg["paths"]
    model_cfg = cfg["model"]

    n_seeds = eval_cfg["n_eval_seeds"]
    n_episodes = eval_cfg["n_eval_episodes"]
    max_steps = env_cfg["max_path_length"]

    policy = BCPolicy(
        obs_dim=env_cfg["obs_dim"],
        action_dim=env_cfg["action_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        n_layers=model_cfg.get("n_blocks", 4),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    if "normalizer_state" in checkpoint:
        normalizer = build_normalizer_from_state(checkpoint["normalizer_state"])
    else:
        normalizer = build_normalizer_from_dataset(cfg)

    n_agents = env_cfg["n_agents"]

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from diffuser.datasets.mpe import make_env

    all_returns_sum = []
    all_returns_agents = []
    seed_returns = defaultdict(list)
    seed_returns_sum = defaultdict(list)

    base_save_dir = path_cfg["save_dir"].format(
        env_name=env_cfg["env_name"],
        quality=env_cfg["quality"],
    )
    save_dir = path_cfg.get("bc_save_dir", os.path.join(base_save_dir, "bc")).format(
        env_name=env_cfg["env_name"],
        quality=env_cfg["quality"],
    )
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    for seed_idx in range(n_seeds):
        seed = 100 * (seed_idx + 1)
        set_seed(seed)

        env = make_env(env_cfg["env_name"], render_mode="rgb_array")
        env.reset()

        if save_video:
            vid_subdir = os.path.join(save_dir, video_dir)
            write_frame, finalize_video = make_video_writer(vid_subdir, seed)
        else:
            write_frame, finalize_video = None, lambda: None

        print(f"  Seed {seed}: ", end="", flush=True)

        for _ in range(n_episodes):
            obs = env.reset()
            done = np.array([False] * n_agents)
            ep_return = np.zeros(n_agents, dtype=np.float32)

            if write_frame is not None:
                frame = render_mpe_frame(env)
                write_frame(frame)

            step_count = 0
            while (not np.all(done)) and step_count < max_steps:
                obs_arr = np.array(obs)  # [A, obs_dim]
                obs_norm = normalizer.normalize(obs_arr, "obs")  # [A, obs_dim]
                obs_t = (
                    torch.from_numpy(obs_norm)
                    .float()
                    .to(device)
                    .unsqueeze(0)
                )  # [A, D] -> [1, A, D]

                with torch.no_grad():
                    act_norm = policy(obs_t)[0].cpu().numpy()  # [1, A, act_dim] -> [A, act_dim]

                act = normalizer.unnormalize(act_norm, "acs")  # [A, act_dim]
                act = np.clip(act, -1.0, 1.0)

                obs, reward, done, _ = env.step(act)
                ep_return += np.array(reward, dtype=np.float32)

                if write_frame is not None:
                    frame = render_mpe_frame(env)
                    write_frame(frame)

                step_count += 1

            all_returns_agents.append(ep_return.copy())
            all_returns_sum.append(ep_return.sum())
            seed_returns[seed].append(ep_return.copy())
            seed_returns_sum[seed].append(ep_return.sum())

        finalize_video()
        print("done")

    all_returns_agents = np.array(all_returns_agents)  # [E, A]
    all_returns_sum = np.array(all_returns_sum)  # [E]

    avg_ep_reward = np.mean(all_returns_agents, axis=0)
    std_ep_reward = np.std(all_returns_agents, axis=0)
    overall_mean = np.mean(all_returns_sum)
    overall_std = np.std(all_returns_sum)

    results = {
        "overall_mean": float(overall_mean),
        "overall_std": float(overall_std),
        "average_ep_reward": avg_ep_reward.tolist(),
        "std_ep_reward": std_ep_reward.tolist(),
        "seed_stats": {
            str(k): {
                "agent_returns": np.array(v).tolist(),
                "sum_returns": np.array(seed_returns_sum[k]).tolist(),
            }
            for k, v in seed_returns.items()
        },
        "checkpoint": checkpoint_path,
        "checkpoint_step": int(checkpoint.get("step", -1)),
    }

    eval_path = os.path.join(eval_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal (sum): {overall_mean:.2f} ± {overall_std:.2f}")
    print(f"Average per-agent: {avg_ep_reward}, std: {std_ep_reward}")
    print(f"Saved eval results: {eval_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC policy")
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_eval_seeds", type=int, default=None)
    parser.add_argument("--n_eval_episodes", type=int, default=None)
    args = parser.parse_args()

    import yaml

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.n_eval_seeds is not None:
        cfg["evaluation"]["n_eval_seeds"] = args.n_eval_seeds
    if args.n_eval_episodes is not None:
        cfg["evaluation"]["n_eval_episodes"] = args.n_eval_episodes

    save_video = not args.no_video
    evaluate(
        checkpoint_path=args.checkpoint,
        cfg=cfg,
        device=args.device,
        save_video=save_video,
        video_dir=cfg["evaluation"]["video_dir"],
    )


if __name__ == "__main__":
    main()
