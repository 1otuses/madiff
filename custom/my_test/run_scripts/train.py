"""
训练脚本: Risk-Guided CFG Diffusion (纯训练, step-based)
==========================================================
只负责:
  1. 加载 MPE 离线数据
  2. 构建/加载模型
  3. Step-based 训练循环 (n_train_steps)
  4. 定期 checkpoint 保存, TensorBoard 日志

评估和视频生成由 evaluate.py 负责。
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.diffusion_actor import RiskGuidedDiffusion
from normalizer import DatasetNormalizer


# ===========================================================================
# Utilities
# ===========================================================================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ===========================================================================
# Data loading
# ===========================================================================

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

        obs = np.stack(obs_list, axis=1)          # [T, A, obs_dim]
        next_obs = np.stack(next_obs_list, axis=1)
        acs = np.stack(acs_list, axis=1)           # [T, A, ac_dim]
        rews = np.stack(rews_list, axis=1)         # [T, A]
        dones = np.stack(dones_list, axis=1)       # [T, A]

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


def build_trajectory_batch(episodes, horizon, n_agents, obs_dim, ac_dim, batch_size,
                          discount: float = 1.0, return_scale: float = 1.0):
    """
    在 episode 段上采样并计算完整的 Return-to-Go（对齐 MADiff 的 RTG）。

    Args:
        episodes: 'obs', 'next_obs', 'acs', 'rews' 列表
        horizon: 采样时间长度 T
        n_agents: agent 数量
        obs_dim: 观测维度
        ac_dim: 动作维度
        batch_size: 批大小
        discount: 折扣因子 γ
        return_scale: RTG 归一化缩放 (用于将 RTG 压缩到 [0,1] 范围)

    Returns:
        x_batch: [B, T, A, obs_dim] 状态序列
        actions_batch: [B, T-1, A, ac_dim] 动作序列
        returns_batch: [B, n_agents] 归一化后的 RTG
    """
    n_episodes = len(episodes["obs"])
    B = batch_size
    x_batch = np.zeros((B, horizon, n_agents, obs_dim), dtype=np.float32)
    actions_batch = np.zeros((B, horizon - 1, n_agents, ac_dim), dtype=np.float32)
    returns_batch = np.zeros((B, n_agents), dtype=np.float32)

    for i in range(B):
        ep_idx = np.random.randint(0, n_episodes)
        ep_obs = episodes["obs"][ep_idx]      # [L, A, obs_dim]
        ep_acs = episodes["acs"][ep_idx]       # [L, A, ac_dim]
        ep_rews = episodes["rews"][ep_idx]     # [L, A]

        L = ep_obs.shape[0]
        start = np.random.randint(0, max(1, L - horizon)) if L > horizon else 0
        end = min(start + horizon, L)
        start = max(0, end - horizon)
        seg_len = end - start

        x_batch[i, :seg_len] = ep_obs[start:end]
        if seg_len > 1:
            actions_batch[i, :seg_len - 1] = ep_acs[start:end - 1]

        # RTG: 从此段起点开始的完整未来折扣回报
        rew_segment = ep_rews[start:]     # [L_remain, A]
        L_remain = len(rew_segment)
        rtg = np.zeros((L_remain, n_agents), dtype=np.float32)
        discounted = np.zeros(n_agents, dtype=np.float32)
        for t in reversed(range(L_remain)):
            discounted = rew_segment[t] + discount * discounted
            rtg[t] = discounted
        # 取片段起点处的 RTG
        rtg_start = rtg[0]  # [n_agents]
        # 归一化到 [0,1]
        returns_batch[i] = rtg_start / return_scale

    # 将 returns 从 [B, n_agents] reshape 成 [B, 1] 以配合 denoiser 输入
    returns_batch = returns_batch.mean(axis=-1, keepdims=True)  # [B, 1]
    return x_batch, actions_batch, returns_batch


# ===========================================================================
# EMA
# ===========================================================================

class EMA:
    def __init__(self, beta: float = 0.995):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for cur_p, ma_p in zip(current_model.parameters(), ma_model.parameters()):
            ma_p.data = self.update_average(ma_p.data, cur_p.data)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# ===========================================================================
# Training loop
# ===========================================================================

def train(cfg: dict, device: str = "cuda"):
    env_cfg = cfg["env"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    path_cfg = cfg["paths"]

    # 1. 数据
    data_dir = path_cfg["data_dir"].format(
        env_name=env_cfg["env_name"], quality=env_cfg["quality"]
    )
    print(f"[1/4] Loading MPE data: {data_dir}")
    episodes = load_mpe_data(data_dir, env_cfg["n_agents"], max_episodes=2000)

    # 1b. 构建归一化器 (整个训练集)
    print("  Building normalizer...")
    # 收集全部 obs 和 acs 用于拟合归一化参数
    all_obs = np.concatenate(episodes["obs"], axis=0)  # [total_L, A, obs_dim]
    all_acs = np.concatenate(episodes["acs"], axis=0)  # [total_L, A, ac_dim]
    norm_dataset = {"obs": all_obs, "acs": all_acs}
    normalizer = DatasetNormalizer(
        norm_dataset,
        normalizer="LimitsNormalizer",
        keys=["obs", "acs"],
        agent_share=True,
    )
    del all_obs, all_acs  # 释放内存

    # 2. 模型
    print(f"\n[2/4] Building Risk-Guided Diffusion Model...")
    def _make_model():
        return RiskGuidedDiffusion(
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

    model = _make_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_params:,}")
    model.to(device)

    ema = EMA(train_cfg["ema_decay"])
    ema_model = _make_model()
    ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())

    optimizer = Adam(model.parameters(), lr=train_cfg["learning_rate"])

    # 3. 路径
    save_dir = path_cfg["save_dir"].format(
        env_name=env_cfg["env_name"],
        quality=env_cfg["quality"],
    )
    tb_dir = path_cfg["tb_dir"].format(
        env_name=env_cfg["env_name"],
        quality=env_cfg["quality"],
    )
    checkpoint_dir = path_cfg.get(
        "checkpoint_dir", os.path.join(save_dir, "checkpoint")
    ).format(
        env_name=env_cfg["env_name"],
        quality=env_cfg["quality"],
    )
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_dir)
    print(f"  TensorBoard: {tb_dir}")

    n_train_steps = train_cfg["n_train_steps"]
    batch_size = train_cfg["batch_size"]
    accumulate_every = train_cfg["gradient_accumulate_every"]
    log_freq = train_cfg["log_freq"]
    save_freq = train_cfg["save_freq"]
    horizon = model_cfg["horizon"]
    n_agents = env_cfg["n_agents"]
    obs_dim = env_cfg["obs_dim"]
    ac_dim = env_cfg["action_dim"]
    # RTG 参数（对齐 MADiff）
    discount = train_cfg.get("discount", 0.997)
    return_scale = train_cfg.get("returns_scale", 1000.0)
    step_start_ema = 2000

    # 自适应 log_freq: 确保在 n_train_steps 中至少有 20 个日志点
    effective_log_freq = min(log_freq, max(1, n_train_steps // 20))
    checkpoint_count = 0
    if effective_log_freq != log_freq:
        print(f"  (Adaptive log_freq: {log_freq} → {effective_log_freq} for {n_train_steps} steps)")

    print(f"\n[3/4] Training ({n_train_steps:,} steps, batch={batch_size})...")

    pbar = tqdm(total=n_train_steps, desc="Training", unit="step")
    step = 0
    start_time = time.time()
    last_diff, last_inv, last_mae = 0.0, 0.0, 0.0

    while step < n_train_steps:
        optimizer.zero_grad()
        for _ in range(accumulate_every):
            x_np, actions_np, returns_np = build_trajectory_batch(
                episodes, horizon, n_agents, obs_dim, ac_dim, batch_size,
                discount=discount, return_scale=return_scale,
            )
            x = torch.from_numpy(x_np).float().to(device)
            actions = torch.from_numpy(actions_np).float().to(device)
            returns = torch.from_numpy(returns_np).float().to(device)

            diffuse_loss = model.loss(x, returns) / accumulate_every
            diffuse_loss.backward()

            inv_loss, inv_mae = model.compute_inv_loss(x, actions)
            inv_loss = inv_loss / accumulate_every
            inv_loss.backward()

            last_diff = diffuse_loss.item() * accumulate_every
            last_inv = inv_loss.item() * accumulate_every
            last_mae = inv_mae.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # EMA
        if step % 10 == 0:
            if step < step_start_ema:
                ema_model.load_state_dict(model.state_dict())
            else:
                ema.update_model_average(ema_model, model)

        # 日志 (自适应频率, 确保短训练也有输出)
        if step % effective_log_freq == 0:
            elapsed = time.time() - start_time
            total = last_diff + last_inv
            tb_writer.add_scalar("Loss/diffuse", last_diff, step)
            tb_writer.add_scalar("Loss/inv", last_inv, step)
            tb_writer.add_scalar("Loss/total", total, step)
            tb_writer.add_scalar("Stats/inv_mae", last_mae, step)
            tb_writer.add_scalar("Stats/elapsed_sec", elapsed, step)
            tb_writer.add_scalar("Stats/steps_per_sec", step / max(elapsed, 1e-6), step)

        # 进度条每次更新都显示最新值，确保可见变化
        elapsed_total = time.time() - start_time
        pbar.set_postfix({
            "diff": f"{last_diff:.4f}",
            "inv": f"{last_inv:.4f}",
            "mae": f"{last_mae:.3f}",
            "s/s": f"{step / max(elapsed_total, 1e-6):.1f}",
        })

        # 保存
        if step % save_freq == 0 and step > 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
            }, os.path.join(checkpoint_dir, "checkpoint.pt"))
            tqdm.write(f"  [step {step}] Checkpoint saved")

        step += 1
        pbar.update(1)

    pbar.close()

    # 最终保存
    final_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, final_path)
    print(f"\n[4/4] Training complete! {final_path}")

    tb_writer.close()
    return final_path, save_dir


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Risk-Guided Diffusion")
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
    print(f"=== Training | seed={cfg['training']['seed']} | {datetime.now():%Y-%m-%d %H:%M:%S}")
    model_path, save_dir = train(cfg, device=args.device)
    print(f"Done. Model: {model_path}")


if __name__ == "__main__":
    main()
