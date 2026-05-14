"""
SMAC Diffusion Critic 训练脚本
==============================
在 SMAC 离线数据集上训练 Diffusion Critic ,
并与标准 MLP Critic 对比价值分布预测能力。

用法:
  # 3m-Good (快速验证, 3 agents, 33D obs)
  python train_smac.py --env 3m-Good --n_episodes 5000 --n_epochs 200

  # 5m_vs_6m-Medium
  python train_smac.py --env 5m_vs_6m-Medium --n_episodes 10000 --n_epochs 500

  # 8m-Poor
  python train_smac.py --env 8m-Poor --n_episodes 5000 --n_epochs 300
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from smac_dataset import load_smac_returns
from diffusion_critic_adv import DiffusionCritic


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_diffusion_critic(
    model: DiffusionCritic,
    dataloader: DataLoader,
    n_epochs: int = 300,
    lr: float = 1e-3,
    device: str = "cpu",
    log_interval: int = 50,
) -> list:
    """训练 Diffusion Critic."""
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_states, batch_returns in dataloader:
            batch_states = batch_states.to(device)
            batch_returns = batch_returns.to(device)

            loss = model.loss(batch_returns, batch_states)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"  [Diff Critic] Epoch {epoch+1:4d}/{n_epochs} | Loss: {avg_loss:.6f}")

    return losses


class MLPCritic(torch.nn.Module):
    """标准 MLP Critic (Baseline)."""
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(n_layers - 1):
            layers.extend([
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, states):
        return self.net(states)

    def loss(self, returns, states):
        return torch.nn.functional.mse_loss(self.forward(states), returns.unsqueeze(-1))


def train_mlp_critic(
    model: MLPCritic,
    dataloader: DataLoader,
    n_epochs: int = 300,
    lr: float = 1e-3,
    device: str = "cpu",
    log_interval: int = 50,
) -> list:
    """训练 MLP Critic Baseline."""
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_states, batch_returns in dataloader:
            batch_states = batch_states.to(device)
            batch_returns = batch_returns.to(device)

            loss = model.loss(batch_returns, batch_states)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"  [MLP Critic]   Epoch {epoch+1:4d}/{n_epochs} | Loss: {avg_loss:.6f}")

    return losses


def visualize_smac_results(
    diff_critic: DiffusionCritic,
    mlp_critic: MLPCritic,
    states_sample: torch.Tensor,
    returns_sample: torch.Tensor,
    info: dict,
    loss_diff: list,
    loss_mlp: list,
    save_dir: str,
    n_samples: int = 1000,
):
    """可视化 SMAC 实验结果."""
    device = next(diff_critic.parameters()).device
    os.makedirs(save_dir, exist_ok=True)

    # 取前 4 个 sample states 做详细分析
    n_viz = min(4, len(states_sample))
    viz_states = states_sample[:n_viz].to(device)

    # Diffusion Critic 采样
    diff_samples = diff_critic.sample(viz_states, n_samples=n_samples)  # [n_s, B]
    diff_samples = diff_samples.cpu().numpy()

    # MLP 预测
    mlp_preds = mlp_critic(viz_states).cpu().detach().numpy()  # [B, 1]

    # 真实 returns
    true_returns = returns_sample[:n_viz].cpu().numpy()

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle(
        f"SMAC Diffusion Critic — {info.get('env_name', 'Unknown')}",
        fontsize=14, fontweight="bold",
    )

    # (1) Diffusion Critic 样本直方图 (4 states)
    for i in range(n_viz):
        ax = axes[i // 2][i % 2]
        ax.hist(diff_samples[:, i], bins=40, density=True, alpha=0.7,
                color="blue", edgecolor="black")
        ax.axvline(mlp_preds[i, 0], color="green", linestyle="--", linewidth=2,
                    label=f"MLP={mlp_preds[i,0]:.1f}")
        ax.axvline(true_returns[i], color="red", linestyle=":", linewidth=2,
                    label=f"True={true_returns[i]:.1f}")
        ax.set_title(f"State {i+1}: P(R|s) from Diffusion Critic", fontsize=11,
                      fontweight="bold")
        ax.set_xlabel("Return R")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    # (2) 训练损失
    ax_loss = axes[2][0]
    ax_loss.plot(loss_diff, label="Diffusion Critic (DDPM)", color="blue", linewidth=1.5)
    ax_loss.set_title("Diffusion Critic Training Loss", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_loss2 = axes[2][1]
    ax_loss2.plot(loss_mlp, label="MLP Critic (MSE)", color="green", linewidth=1.5)
    ax_loss2.set_title("MLP Critic Training Loss", fontsize=11, fontweight="bold")
    ax_loss2.set_xlabel("Epoch")
    ax_loss2.set_ylabel("Loss")
    ax_loss2.legend()
    ax_loss2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, "smac_diffusion_critic.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ 结果已保存: {save_path}")

    # 打印统计摘要
    print(f"\n{'='*60}")
    print(f"SMAC Diffusion Critic — 定量分析")
    print(f"{'='*60}")
    print(f"  环境: {info.get('env_name', 'N/A')}")
    print(f"  状态维度: {info['state_dim']}")
    print(f"  样本数: {info.get('n_samples', 'N/A'):,}")
    print(f"  Return 范围: [{info.get('return_min', 0):.2f}, {info.get('return_max', 0):.2f}]")
    print(f"  Return 均值: {info.get('return_mean', 0):.2f}")
    print()
    for i in range(n_viz):
        diff_mean = diff_samples[:, i].mean()
        diff_std = diff_samples[:, i].std()
        q10 = np.percentile(diff_samples[:, i], 10)
        q50 = np.percentile(diff_samples[:, i], 50)
        q90 = np.percentile(diff_samples[:, i], 90)
        print(f"  State {i+1}:")
        print(f"    True return:    {true_returns[i]:.2f}")
        print(f"    MLP predict:    {mlp_preds[i, 0]:.2f}")
        print(f"    Diff mean/std:  {diff_mean:.2f} / {diff_std:.2f}")
        print(f"    Diff quantiles: [{q10:.2f}, {q50:.2f}, {q90:.2f}]")
        print(f"    Interpretation: {'多峰分布 ✓' if diff_std > 1.0 else '近乎单峰'}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="SMAC Diffusion Critic Training")
    parser.add_argument("--env", type=str, default="3m-Good",
                        help="SMAC 环境名 (如 3m-Good, 8m-Medium)")
    parser.add_argument("--data_base", type=str,
                        default="diffuser/datasets/data/smac",
                        help="SMAC 数据根目录")
    parser.add_argument("--n_episodes", type=int, default=5000,
                        help="最大 episode 数")
    parser.add_argument("--n_timesteps", type=int, default=200,
                        help="扩散步数 (建议 200)")
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_blocks", type=int, default=4,
                        help="ResMLP 残差块数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str,
                        default="custom/diffusion_critic/results/smac")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SMAC Diffusion Critic 训练")
    print(f"{'='*60}")
    print(f"  Environment: {args.env}")
    print(f"  Device: {args.device}")
    print(f"  Max episodes: {args.n_episodes}")
    print(f"  Diffusion steps: {args.n_timesteps}")
    print(f"  Training epochs: {args.n_epochs}")
    print(f"{'='*60}\n")

    set_seed(args.seed)

    # ---- 1. 加载 SMAC 数据 ----
    data_dir = os.path.join(args.data_base, args.env)
    print(f"[1/4] Loading SMAC data from: {data_dir}")
    states_np, returns_np, info = load_smac_returns(
        data_dir=data_dir,
        discount=0.99,
        max_episodes=args.n_episodes,
    )
    info["env_name"] = args.env
    info["n_samples"] = len(returns_np)

    # 估算 return_scale (用于归一化) — 使用 max(abs(min), abs(max)) * 1.2
    return_scale = max(abs(returns_np.min()), abs(returns_np.max())) * 1.2
    print(f"  Return scale: {return_scale:.2f}")

    # 创建 DataLoader
    dataset = TensorDataset(
        torch.from_numpy(states_np).float(),
        torch.from_numpy(returns_np).float(),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print()

    # ---- 2. 训练 Diffusion Critic ----
    print("[2/4] Training Diffusion Critic...")
    diff_critic = DiffusionCritic(
        state_dim=info["state_dim"],
        n_timesteps=args.n_timesteps,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        return_scale=return_scale,
    )
    loss_diff = train_diffusion_critic(
        diff_critic, dataloader,
        n_epochs=args.n_epochs, lr=args.lr, device=args.device,
    )
    print(f"  Final Diffusion Loss: {loss_diff[-1]:.6f}\n")

    # ---- 3. 训练 MLP Critic Baseline ----
    print("[3/4] Training MLP Critic Baseline...")
    mlp_critic = MLPCritic(state_dim=info["state_dim"], hidden_dim=args.hidden_dim)
    loss_mlp = train_mlp_critic(
        mlp_critic, dataloader,
        n_epochs=args.n_epochs, lr=args.lr, device=args.device,
    )
    print(f"  Final MLP Loss: {loss_mlp[-1]:.6f}\n")

    # ---- 4. 可视化 ----
    print("[4/4] Visualizing results...")
    # 采样一些状态用于可视化
    viz_idx = np.random.choice(len(states_np), min(4, len(states_np)), replace=False)
    viz_states = torch.from_numpy(states_np[viz_idx]).float()
    viz_returns = torch.from_numpy(returns_np[viz_idx]).float()

    visualize_smac_results(
        diff_critic, mlp_critic,
        viz_states, viz_returns,
        info, loss_diff, loss_mlp,
        save_dir=args.save_dir,
    )

    print(f"\n{'='*60}")
    print("SMAC Diffusion Critic 训练完成!")
    print(f"结果保存至: {args.save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
