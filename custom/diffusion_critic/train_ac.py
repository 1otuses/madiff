"""
Actor-Critic 联合训练脚本 (MLP Actor + Diffusion Critic)
=======================================================
训练流程:
  1. Critic 更新: 从离线数据采样 (s, a, R), 最小化 DDPM 损失
  2. Actor 更新: 对采样动作 a~π(s), 用 Critic 预估 E[R|s,a],
                 最大化期望回报

评估:
  加载 Actor 在 SMAC 环境中多 seed 运行，统计累计回报均值±方差

用法:
  python train_ac.py --env 3m-Good --n_episodes 5000 --n_epochs 300
"""
# n_episodes: 训练使用的 episode 数量 (每 episode 包含多步 transition)
# n_epochs: 训练的 epoch 数量 (每 epoch 遍历完整数据集一次)

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from smac_dataset import load_smac_state_action_returns
from diffusion_critic_ac import DiffusionCriticAC, MLPActor


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(
    actor: MLPActor,
    critic: DiffusionCriticAC,
    dataloader: DataLoader,
    n_epochs: int,
    lr_actor: float,
    lr_critic: float,
    return_mean: float,
    return_std: float,
    n_agents: int,
    n_actions: int,
    device: str,
    tb_writer: SummaryWriter = None,
):
    """Actor-Critic 联合训练循环."""
    critic.to(device)
    actor.to(device)
    opt_critic = Adam(critic.parameters(), lr=lr_critic)
    opt_actor = Adam(actor.parameters(), lr=lr_actor)

    critic_losses = []
    actor_losses = []

    pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in pbar:
        epoch_c_loss = 0.0
        epoch_a_loss = 0.0

        for batch_s, batch_a, batch_r in dataloader:
            s = batch_s.to(device)
            a = batch_a.to(device)
            r = batch_r.to(device)
            r_norm = (r - return_mean) / return_std

            c_loss = critic.loss(r_norm, s, a)
            opt_critic.zero_grad()
            c_loss.backward()
            opt_critic.step()
            epoch_c_loss += c_loss.item()

            B = s.shape[0]
            sampled_a, probs = actor.sample_action(s)
            log_probs = torch.log(probs.gather(-1, sampled_a.unsqueeze(-1)) + 1e-8).squeeze(-1)

            with torch.no_grad():
                q_sampled = critic.expected_value(s, sampled_a, n_samples=30)

            a_loss = -(log_probs.sum(dim=-1) * q_sampled).mean()
            opt_actor.zero_grad()
            a_loss.backward()
            opt_actor.step()
            epoch_a_loss += a_loss.item()

        avg_c = epoch_c_loss / len(dataloader)
        avg_a = epoch_a_loss / len(dataloader)
        critic_losses.append(avg_c)
        actor_losses.append(avg_a)

        if tb_writer:
            tb_writer.add_scalar("Loss/critic", avg_c, epoch)
            tb_writer.add_scalar("Loss/actor", avg_a, epoch)

        pbar.set_postfix({"c_loss": f"{avg_c:.4f}", "a_loss": f"{avg_a:.4f}"})

    pbar.close()
    return critic_losses, actor_losses


def evaluate_online(
    actor: MLPActor,
    map_name: str,
    n_seeds: int,
    n_episodes: int,
    device: str,
):
    """MADiff 风格在线评估: 多 seed 运行累计回报."""
    from smac.env import StarCraft2Env

    all_returns = []
    seed_returns = defaultdict(list)

    for seed_idx in range(n_seeds):
        seed = 100 * (seed_idx + 1)
        set_seed(seed)
        env = StarCraft2Env(map_name=map_name, obs_last_action=False)

        print(f"  Seed {seed}: ", end="", flush=True)
        for ep in range(n_episodes):
            env.reset()
            obs = np.array(env.get_obs())
            done = False
            total_return = 0.0

            while not done:
                # 构建状态特征 (简化版: agent obs + 零填充 global state)
                state = obs.reshape(-1).astype(np.float32)  # [A*obs_dim]
                state = np.concatenate([state, np.zeros(48, dtype=np.float32)])  # [A*obs_dim+48]
                s_t = torch.from_numpy(state).unsqueeze(0).float().to(device)  # [1, D]

                with torch.no_grad():
                    actions, _ = actor.sample_action(s_t, deterministic=True)
                action_list = actions[0].cpu().numpy().astype(int).tolist()

                reward, done, _ = env.step(action_list)
                obs = np.array(env.get_obs())
                total_return += reward if isinstance(reward, (int, float)) else sum(reward)

            episode_returns = total_return
            seed_returns[seed].append(episode_returns)
            all_returns.append(episode_returns)
            if (ep + 1) % 5 == 0:
                print(".", end="", flush=True)

        env.close()
        mean_s = np.mean(seed_returns[seed])
        print(f" {mean_s:.1f}")

    all_r = np.array(all_returns)
    seed_stats = {}
    for seed, rets in seed_returns.items():
        rets_arr = np.array(rets)
        seed_stats[seed] = {
            "mean": float(rets_arr.mean()),
            "std": float(rets_arr.std()),
        }

    overall_mean = float(all_r.mean())
    overall_std = float(all_r.std())

    print(f"\n  🎯 Final: {overall_mean:.2f} ± {overall_std:.2f}")

    return {
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "final_result": f"{overall_mean:.2f} ± {overall_std:.2f}",
        "seed_stats": seed_stats,
        "all_returns": all_r.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Actor-Critic Training")
    parser.add_argument("--env", type=str, default="3m-Good")
    parser.add_argument("--n_episodes", type=int, default=5000)
    parser.add_argument("--n_timesteps", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--lr_actor", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seeds", type=int, default=3)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "runs"))
    args = parser.parse_args()

    set_seed(args.seed)

    # ---- 1. 数据 ----
    data_dir = f"diffuser/datasets/data/smac/{args.env.replace('-','/')}"
    print(f"[1/4] Loading data: {data_dir}")
    states_np, actions_np, returns_np, info = load_smac_state_action_returns(
        data_dir=data_dir, max_episodes=args.n_episodes,
    )
    return_mean = info["return_mean"]
    return_std = info["return_std"]

    dataset = TensorDataset(
        torch.from_numpy(states_np).float(),
        torch.from_numpy(actions_np).long(),
        torch.from_numpy(returns_np).float(),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ---- TensorBoard ----
    tb_dir = os.path.join(args.save_dir, "tensorboard", f"ac_{args.env}")
    tb_writer = SummaryWriter(log_dir=tb_dir)
    print(f"\n  TensorBoard: {tb_dir}")

    # ---- 2. 模型 ----
    print(f"\n[2/4] Building Actor-Critic models...")
    print(f"  State dim: {info['state_dim']}, Agents: {info['n_agents']}, "
          f"Actions: {info['n_actions']}")

    critic = DiffusionCriticAC(
        state_dim=info["state_dim"],
        n_agents=info["n_agents"],
        n_actions=info["n_actions"],
        n_timesteps=args.n_timesteps,
        hidden_dim=args.hidden_dim,
    )
    actor = MLPActor(
        state_dim=info["state_dim"],
        n_agents=info["n_agents"],
        n_actions=info["n_actions"],
        hidden_dim=args.hidden_dim,
    )
    print(f"  Critic: {sum(p.numel() for p in critic.parameters()):,} params")
    print(f"  Actor:  {sum(p.numel() for p in actor.parameters()):,} params")

    # ---- 3. 训练 ----
    print(f"\n[3/4] Training Actor-Critic ({args.n_epochs} epochs)...")
    c_losses, a_losses = train(
        actor, critic, dataloader,
        n_epochs=args.n_epochs,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        return_mean=return_mean,
        return_std=return_std,
        n_agents=info["n_agents"],
        n_actions=info["n_actions"],
        device=args.device,
        tb_writer=tb_writer,
    )

    # 保存模型
    model_dir = os.path.join(args.save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(critic.state_dict(), os.path.join(model_dir, f"critic_ac_{args.env}.pt"))
    torch.save(actor.state_dict(), os.path.join(model_dir, f"actor_ac_{args.env}.pt"))
    print(f"  ✅ Models saved: {model_dir}")

    # ---- 4. 在线评估 ----
    print(f"\n[4/4] Online Evaluation ({args.eval_seeds} seeds × {args.eval_episodes} eps)...")
    map_name = args.env.split("-")[0]  # "3m-Good" → "3m"
    try:
        results = evaluate_online(
            actor, map_name, args.eval_seeds, args.eval_episodes, args.device,
        )
    except FileNotFoundError as e:
        print(f"Skipping online evaluation: {e}")
        results = {
            "overall_mean": None,
            "overall_std": None,
            "final_result": "skipped",
            "seed_stats": {},
            "all_returns": [],
        }

    # 保存结果
    os.makedirs(os.path.join(args.save_dir, "eval"), exist_ok=True)
    json_path = os.path.join(args.save_dir, "eval", f"ac_{args.env}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    tb_writer.close()
    print(f"\n{'='*60}")
    print(f"✅ Training complete! Final: {results['final_result']}")
    print(f"   Results: {json_path}")
    print(f"   TensorBoard: tensorboard --logdir {tb_dir} --port 6006")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
