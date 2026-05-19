#!/usr/bin/env python3
"""
Behavior Cloning training for MPE offline datasets.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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


def export_normalizer_state(normalizer: DatasetNormalizer) -> dict:
    state = {
        "agent_share": normalizer.agent_share,
        "keys": normalizer.keys,
        "normalizers": {},
    }
    for key in normalizer.keys:
        n = normalizer.normalizers[key]
        if normalizer.agent_share:
            if isinstance(n, LimitsNormalizer):
                state["normalizers"][key] = {
                    "type": "LimitsNormalizer",
                    "mins": n.mins,
                    "maxs": n.maxs,
                }
            elif isinstance(n, GaussianNormalizer):
                state["normalizers"][key] = {
                    "type": "GaussianNormalizer",
                    "mean": n.mean,
                    "std": n.std,
                }
            else:
                raise ValueError(f"Unsupported normalizer type: {type(n)}")
        else:
            per_agent = []
            for sub in n:
                if isinstance(sub, LimitsNormalizer):
                    per_agent.append({"type": "LimitsNormalizer", "mins": sub.mins, "maxs": sub.maxs})
                elif isinstance(sub, GaussianNormalizer):
                    per_agent.append({"type": "GaussianNormalizer", "mean": sub.mean, "std": sub.std})
                else:
                    raise ValueError(f"Unsupported normalizer type: {type(sub)}")
            state["normalizers"][key] = per_agent
    return state


def infinite_cycle(dl):
    while True:
        for batch in dl:
            yield batch


def train(cfg: dict, device: str = "cuda"):
    env_cfg = cfg["env"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    path_cfg = cfg["paths"]

    data_dir = path_cfg["data_dir"].format(
        env_name=env_cfg["env_name"],
        quality=env_cfg["quality"],
    )
    print(f"[1/3] Loading MPE data: {data_dir}")
    episodes = load_mpe_data(data_dir, env_cfg["n_agents"], max_episodes=2000)

    print("  Building normalizer...")
    all_obs = np.concatenate(episodes["obs"], axis=0)  # [T_total, A, obs_dim]
    all_acs = np.concatenate(episodes["acs"], axis=0)  # [T_total, A, act_dim]
    norm_dataset = {"obs": all_obs, "acs": all_acs}
    normalizer = DatasetNormalizer(
        norm_dataset,
        normalizer="LimitsNormalizer",
        keys=["obs", "acs"],
        agent_share=True,
    )

    obs_norm = normalizer.normalize(all_obs, "obs")  # [T_total, A, obs_dim]
    acs_norm = normalizer.normalize(all_acs, "acs")  # [T_total, A, act_dim]
    obs_flat = obs_norm.reshape(-1, obs_norm.shape[-1])  # [T_total, A, D] -> [T_total*A, D]
    acs_flat = acs_norm.reshape(-1, acs_norm.shape[-1])  # [T_total, A, D] -> [T_total*A, D]

    obs_tensor = torch.from_numpy(obs_flat).float()
    acs_tensor = torch.from_numpy(acs_flat).float()
    dataset = TensorDataset(obs_tensor, acs_tensor)
    dataloader = infinite_cycle(
        DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
    )

    print(f"\n[2/3] Building BC policy...")
    policy = BCPolicy(
        obs_dim=env_cfg["obs_dim"],
        action_dim=env_cfg["action_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        n_layers=model_cfg.get("n_blocks", 4),
    ).to(device)

    optimizer = Adam(policy.parameters(), lr=train_cfg["learning_rate"])

    base_save_dir = path_cfg["save_dir"].format(
        env_name=env_cfg["env_name"],
        quality=env_cfg["quality"],
    )
    save_dir = path_cfg.get("bc_save_dir", os.path.join(base_save_dir, "bc")).format(
        env_name=env_cfg["env_name"],
        quality=env_cfg["quality"],
    )
    tb_dir = path_cfg.get("bc_tb_dir", os.path.join(save_dir, "tensorboard")).format(
        env_name=env_cfg["env_name"],
        quality=env_cfg["quality"],
    )
    checkpoint_dir = path_cfg.get(
        "bc_checkpoint_dir", os.path.join(save_dir, "checkpoint")
    ).format(
        env_name=env_cfg["env_name"],
        quality=env_cfg["quality"],
    )
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir)

    n_train_steps = train_cfg["n_train_steps"]
    log_freq = max(1, train_cfg["log_freq"])
    save_freq = max(1, train_cfg["save_freq"])

    print(f"\n[3/3] Training for {n_train_steps} steps...")
    pbar = tqdm(range(n_train_steps), ncols=100)
    start_time = time.time()

    for step in pbar:
        obs_b, acs_b = next(dataloader)
        obs_b = obs_b.to(device)
        acs_b = acs_b.to(device)

        loss = policy.loss(obs_b, acs_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_freq == 0:
            elapsed = time.time() - start_time
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "t": f"{elapsed:.1f}s"})
            writer.add_scalar("Loss/bc", loss.item(), step)

        if step % save_freq == 0 and step > 0:
            ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save(
                {
                    "step": step,
                    "model": policy.state_dict(),
                    "normalizer_state": export_normalizer_state(normalizer),
                },
                ckpt_path,
            )

    final_ckpt = os.path.join(checkpoint_dir, "checkpoint.pt")
    torch.save(
        {
            "step": n_train_steps,
            "model": policy.state_dict(),
            "normalizer_state": export_normalizer_state(normalizer),
        },
        final_ckpt,
    )
    writer.close()
    print(f"Saved checkpoint: {final_ckpt}")


def main():
    parser = argparse.ArgumentParser(description="Train BC policy on MPE offline data")
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_train_steps", type=int, default=None)
    args = parser.parse_args()

    import yaml

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.n_train_steps is not None:
        cfg["training"]["n_train_steps"] = args.n_train_steps

    set_seed(cfg["training"]["seed"])

    print("=" * 60)
    print(f"  Experiment: {cfg['meta_data'].get('exp_name', 'bc')}")
    print(f"  Env: {cfg['env']['env_name']} ({cfg['env']['quality']})")
    print(f"  Steps: {cfg['training']['n_train_steps']}")
    print(f"  Device: {args.device}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    train(cfg, device=args.device)


if __name__ == "__main__":
    main()
