"""
Diffusion Critic 训练脚本 — Toy Environment
===========================================
在双峰回报环境上训练 Diffusion Critic，并与标准 MLP Critic 对比。

Pipeline:
1. 生成离线数据集 (states, returns)
2. 训练 Diffusion Critic (DDPM 建模 P(R|s))
3. 训练 MLP Critic Baseline (MSE 建模 E[R|s])
4. 可视化对比: 真分布 vs 学习分布 vs Baseline 点估计
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from toy_env import BimodalToyEnv, generate_dataset
from diffusion_critic import DiffusionCritic


def set_seed(seed: int = 42):
    """固定全局随机种子 (遵循 MADiff 确定性约束)。"""
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
    n_epochs: int = 500,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list:
    """训练 Diffusion Critic (DDPM 建模 P(R|s))。"""
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_states, batch_returns in dataloader:
            batch_states = batch_states.to(device)    # [B, state_dim]
            batch_returns = batch_returns.to(device)   # [B]

            loss = model.loss(batch_returns, batch_states)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  [Diffusion Critic] Epoch {epoch+1:4d}/{n_epochs} | Loss: {avg_loss:.6f}")

    return losses


class MLPCritic(torch.nn.Module):
    """标准 MLP Critic: 学习 E[R|s] (MSE 回归)。"""
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor):
        # states: [B, state_dim] → returns: [B, 1]
        return self.net(states)

    def loss(self, returns: torch.Tensor, states: torch.Tensor):
        # returns: [B, 1], states: [B, state_dim]
        pred = self.forward(states)  # [B, 1]
        return torch.nn.functional.mse_loss(pred, returns)


def train_mlp_critic(
    model: MLPCritic,
    dataloader: DataLoader,
    n_epochs: int = 500,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list:
    """训练 MLP Critic (MSE 回归 E[R|s])。"""
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_states, batch_returns in dataloader:
            batch_states = batch_states.to(device)     # [B, D]
            batch_returns = batch_returns.to(device).unsqueeze(-1)  # [B, 1]

            loss = model.loss(batch_returns, batch_states)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  [MLP Critic]     Epoch {epoch+1:4d}/{n_epochs} | Loss: {avg_loss:.6f}")

    return losses


def visualize(
    diff_critic: DiffusionCritic,
    mlp_critic: MLPCritic,
    env: BimodalToyEnv,
    loss_hist_diff: list,
    loss_hist_mlp: list,
    save_dir: str,
    n_samples: int = 5000,
):
    """可视化: 真分布 vs 学习分布 vs Baseline。"""
    device = next(diff_critic.parameters()).device
    os.makedirs(save_dir, exist_ok=True)

    # ---- 1. 购买 S₀ 状态 ----
    s0 = torch.zeros(1, env.n_states, device=device)
    s0[0, env.S0] = 1.0  # [1, 4]

    # ---- 2. Diffusion Critic 采样 ----
    samples = diff_critic.sample(s0, n_samples=n_samples)  # [n_samples, 1]
    samples = samples.squeeze(-1).cpu().numpy()  # [n_samples]

    # ---- 3. MLP Critic 预测 ----
    mlp_pred = mlp_critic(s0).item()  # 标量

    # ---- 4. 真实分布 ----
    # 理论上: 50% -10, 50% +10
    true_samples = np.where(
        np.random.rand(n_samples) < 0.5, -10.0, 10.0
    )

    # ---- 5. 绘图 ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Diffusion Critic vs MLP Critic — Bimodal Return Distribution",
        fontsize=14,
        fontweight="bold",
    )

    # (a) 真实分布直方图
    axes[0, 0].hist(true_samples, bins=40, density=True, alpha=0.7, color="gray",
                     edgecolor="black")
    axes[0, 0].axvline(0.0, color="red", linestyle="--", linewidth=2, label="E[R] = 0")
    axes[0, 0].set_title("True Return Distribution P(R|S₀)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Return R")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].set_xlim(-15, 15)

    # (b) Diffusion Critic 学习的分布
    axes[0, 1].hist(samples, bins=40, density=True, alpha=0.7, color="blue",
                     edgecolor="black")
    axes[0, 1].axvline(samples.mean(), color="red", linestyle="--", linewidth=2,
                        label=f"E[R] ≈ {samples.mean():.2f}")
    axes[0, 1].set_title(
        "Diffusion Critic: Learned P(R|S₀)", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Return R")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].set_xlim(-15, 15)

    # (c) MLP Critic 预测
    axes[0, 2].bar(["MLP Critic\n(point estimate)"], [mlp_pred], color="green",
                    alpha=0.7, width=0.3)
    axes[0, 2].axhline(0.0, color="red", linestyle="--", linewidth=2, label="True E[R]=0")
    axes[0, 2].set_title("MLP Critic: Point Estimate", fontsize=12, fontweight="bold")
    axes[0, 2].set_ylabel("Predicted E[R]")
    axes[0, 2].set_ylim(-15, 15)
    axes[0, 2].legend()

    # (d) 叠加对比
    axes[1, 0].hist(true_samples, bins=40, density=True, alpha=0.4, color="gray",
                     label="True Dist.", edgecolor="black")
    axes[1, 0].hist(samples, bins=40, density=True, alpha=0.4, color="blue",
                     label="Diffusion Critic", edgecolor="black")
    axes[1, 0].axvline(mlp_pred, color="green", linestyle="-", linewidth=3,
                        label=f"MLP pred = {mlp_pred:.2f}")
    axes[1, 0].set_title("Overlay Comparison", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Return R")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].set_xlim(-15, 15)

    # (e) 训练损失曲线
    axes[1, 1].plot(loss_hist_diff, label="Diffusion Critic (DDPM Loss)", color="blue",
                    linewidth=1.5)
    axes[1, 1].set_title("Diffusion Critic Training Loss", fontsize=12,
                          fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(loss_hist_mlp, label="MLP Critic (MSE Loss)", color="green",
                    linewidth=1.5)
    axes[1, 2].set_title("MLP Critic Training Loss", fontsize=12, fontweight="bold")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Loss")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存
    save_path = os.path.join(save_dir, "diffusion_critic_vs_mlp.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ 对比图已保存: {save_path}")

    # 同时保存独立的扩散过程可视化
    # 采样过程中间步骤
    _, trajectory = diff_critic.sample(s0, n_samples=50, return_all_steps=True)
    # trajectory: [50, 1, T+1]
    traj_np = trajectory.squeeze(1).cpu().numpy()  # [50, T+1]
    T_diff = traj_np.shape[1] - 1
    steps_to_plot = [0, T_diff//5, 2*T_diff//5, 3*T_diff//5, 4*T_diff//5, T_diff]

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 8))
    fig2.suptitle("Diffusion Denoising Process (50 sample trajectories)", fontsize=14,
                   fontweight="bold")
    for i, step in enumerate(steps_to_plot):
        ax = axes2[i // 3][i % 3]
        ax.hist(traj_np[:, step], bins=30, alpha=0.7, color="blue", edgecolor="black")
        ax.set_title(f"t={step}", fontsize=11)
        ax.set_xlim(-15, 15)
        ax.set_xlabel("Return R")
        ax.set_ylabel("Count")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path2 = os.path.join(save_dir, "diffusion_process.png")
    plt.savefig(save_path2, dpi=150, bbox_inches="tight")
    print(f"✅ 扩散过程图已保存: {save_path2}")


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Critic on Toy Env")
    parser.add_argument("--n_episodes", type=int, default=20000,
                        help="离线数据集大小")
    parser.add_argument("--n_timesteps", type=int, default=100,
                        help="扩散步数")
    parser.add_argument("--n_epochs", type=int, default=1000,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str,
                        default="custom/diffusion_critic/results")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Diffusion Critic — Toy Environment Experiment")
    print(f"{'='*60}")
    print(f"  Device: {args.device}")
    print(f"  Dataset size: {args.n_episodes} episodes")
    print(f"  Diffusion timesteps: {args.n_timesteps}")
    print(f"  Training epochs: {args.n_epochs}")
    print(f"{'='*60}\n")

    # ---- 固定随机种子 ----
    set_seed(args.seed)

    # ---- 1. 生成数据集 ----
    print("[1/4] Generating offline dataset...")
    states, returns = generate_dataset(n_episodes=args.n_episodes, seed=args.seed)
    # states: [N, 4], returns: [N]
    # 过滤出 S₀ 的数据用于训练 (终止状态的价值为 0 无信息量，但也可以保留)
    dataset = TensorDataset(
        torch.from_numpy(states).float(),
        torch.from_numpy(returns).float(),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"  Dataset: {len(dataset)} samples (states shape: {states.shape})\n")

    # ---- 2. 训练 Diffusion Critic ----
    print("[2/4] Training Diffusion Critic (DDPM modeling P(R|s))...")
    diff_critic = DiffusionCritic(
        state_dim=4,  # one-hot S₀, S_trap, S_safe, S_treasure
        n_timesteps=args.n_timesteps,
        hidden_dim=args.hidden_dim,
    )
    loss_diff = train_diffusion_critic(
        diff_critic, dataloader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=args.device,
    )
    print(f"  Final Diffusion Loss: {loss_diff[-1]:.6f}\n")

    # ---- 3. 训练 MLP Critic Baseline ----
    print("[3/4] Training MLP Critic (MSE regression E[R|s])...")
    mlp_critic = MLPCritic(state_dim=4, hidden_dim=args.hidden_dim)
    loss_mlp = train_mlp_critic(
        mlp_critic, dataloader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=args.device,
    )
    print(f"  Final MLP Loss: {loss_mlp[-1]:.6f}\n")

    # ---- 4. 可视化对比 ----
    print("[4/4] Visualizing results...")
    env = BimodalToyEnv(seed=args.seed)
    visualize(
        diff_critic,
        mlp_critic,
        env,
        loss_diff,
        loss_mlp,
        save_dir=args.save_dir,
        n_samples=5000,
    )

    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"Results saved to: {args.save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
