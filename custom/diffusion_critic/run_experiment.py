"""
统一实验运行器 — Diffusion Critic + TensorBoard
===============================================
特性:
  1. 一键运行 Toy Env / SMAC 实验
  2. 自动保存模型 checkpoints + TensorBoard 日志
  3. 训练后自动评估 (计算 return 分布的统计指标)
  4. 高质量对比可视化

用法:
  # Toy Environment
  python run_experiment.py --env toy --n_epochs 500

  # SMAC 3m-Good
  python run_experiment.py --env smac_3m --n_episodes 3000 --n_epochs 300

  # 启动 TensorBoard 查看训练
  tensorboard --logdir custom/diffusion_critic/runs/
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# 根据实验类型加载对应模块
# Toy Env 相关
sys.path.insert(0, os.path.dirname(__file__))
from toy_env import BimodalToyEnv, generate_dataset as gen_toy_data
from diffusion_critic import DiffusionCritic as ToyDiffusionCritic

# SMAC 相关
from smac_dataset import load_smac_returns
from diffusion_critic_ac import DiffusionCriticAC


# ============================================================================
# 通用工具
# ============================================================================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MLPCritic(torch.nn.Module):
    """标准 MLP Critic Baseline."""
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, states):
        return self.net(states)  # [B, state_dim] -> [B, 1]

    def loss(self, returns, states):
        return torch.nn.functional.mse_loss(self.forward(states), returns.unsqueeze(-1))


def train_model(model, dataloader, n_epochs, lr, device, tb_writer, prefix, log_interval=50):
    """
    通用训练循环 — 同时写 TensorBoard 日志。
    """
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

        # TensorBoard
        if tb_writer is not None:
            tb_writer.add_scalar(f"Loss/{prefix}", avg_loss, epoch)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"  [{prefix}] Epoch {epoch+1:4d}/{n_epochs} | Loss: {avg_loss:.6f}")

    return losses


def save_model(model, save_path: str):
    """保存模型 state_dict."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  ✅ Model saved: {save_path}")


def load_model(model, load_path: str, device: str):
    """加载模型 state_dict."""
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    print(f"  ✅ Model loaded: {load_path}")
    return model


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_diffusion_critic(model, states, returns_true, device, n_samples=1000):
    """
    评估 Diffusion Critic: 采样回报并与真值对比。
    返回 dict 包含: mean_abs_error, std_error, quantiles, 等.
    """
    model.eval()
    states_t = states.to(device)

    with torch.no_grad():
        samples = model.sample(states_t, n_samples=n_samples)  # [n_samples, B]

    samples_np = samples.cpu().numpy()
    returns_np = returns_true.cpu().numpy()

    B = returns_np.shape[0]
    metrics = {
        "n_states": B,
        "true_return_mean": float(returns_np.mean()),
        "true_return_std": float(returns_np.std()),
        "diff_sample_mean": float(samples_np.mean()),
        "diff_sample_std": float(samples_np.std()),
        "per_state": [],
    }

    for i in range(B):
        si = samples_np[:, i]
        ri = returns_np[i]
        # 计算 EMD (Earth Mover Distance) 简化版: 均值+分位数差异
        q_diff = [np.percentile(si, q) for q in [10, 25, 50, 75, 90]]
        metrics["per_state"].append({
            "state_idx": i,
            "true_return": float(ri),
            "sampled_mean": float(si.mean()),
            "sampled_std": float(si.std()),
            "quantiles_10_90": [float(q_diff[0]), float(q_diff[-1])],
            "mae": float(abs(si.mean() - ri)),
        })

    # 汇总指标
    maes = [m["mae"] for m in metrics["per_state"]]
    metrics["mean_mae"] = float(np.mean(maes))
    metrics["std_mae"] = float(np.std(maes))

    model.train()
    return metrics


def evaluate_mlp_critic(model, states, returns_true, device):
    """评估 MLP Critic: 计算 MSE/MAE."""
    model.eval()
    states_t = states.to(device)
    returns_t = returns_true.to(device)

    with torch.no_grad():
        preds = model(states_t).squeeze(-1)  # [B]
        mse = torch.nn.functional.mse_loss(preds, returns_t)
        mae = (preds - returns_t).abs().mean()

    metrics = {
        "mse": float(mse.item()),
        "mae": float(mae.item()),
        "pred_mean": float(preds.mean().item()),
        "pred_std": float(preds.std().item()),
    }
    model.train()
    return metrics


# ============================================================================
# 可视化
# ============================================================================

def make_comparison_figure(
    diff_samples, true_returns, mlp_preds,
    loss_diff, loss_mlp, eval_diff, eval_mlp,
    env_name, save_path,
):
    """制作高质量对比图 (Toy + SMAC 通用)."""
    n_viz = min(4, diff_samples.shape[1])

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Row 0: Distribution histograms
    for i in range(n_viz):
        if n_viz == 4:
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[0, :n_viz][i])

        ax.hist(diff_samples[:, i], bins=50, density=True, alpha=0.7,
                color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axvline(true_returns[i], color="crimson", linestyle="--", linewidth=2,
                    label=f"True R={true_returns[i]:.2f}")
        ax.axvline(mlp_preds[i], color="darkorange", linestyle=":", linewidth=2,
                    label=f"MLP={mlp_preds[i]:.2f}")
        ax.set_title(f"State #{i+1}: P(R|s)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Return R")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper right")

    # Row 1: Training loss curves
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_loss.plot(loss_diff, color="steelblue", linewidth=1.5, label="Diffusion Critic")
    ax_loss.set_title("Diffusion Critic Loss", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("DDPM Loss")
    ax_loss.grid(True, alpha=0.3); ax_loss.legend(fontsize=8)

    ax_mlp = fig.add_subplot(gs[1, 1])
    ax_mlp.plot(loss_mlp, color="darkorange", linewidth=1.5, label="MLP Critic")
    ax_mlp.set_title("MLP Critic Loss", fontsize=11, fontweight="bold")
    ax_mlp.set_xlabel("Epoch"); ax_mlp.set_ylabel("MSE Loss")
    ax_mlp.grid(True, alpha=0.3); ax_mlp.legend(fontsize=8)

    # Row 1 col 2: Metrics comparison
    ax_met = fig.add_subplot(gs[1, 2])
    met_names = ["Diff MAE", "MLP MAE", "Diff Quant\nSpread"]
    met_vals = [
        eval_diff.get("mean_mae", 0),
        eval_mlp.get("mae", 0),
        eval_diff.get("diff_sample_std", 0) / max(eval_diff.get("true_return_std", 1), 0.01),
    ]
    bars = ax_met.bar(met_names, met_vals, color=["steelblue", "darkorange", "gray"], alpha=0.8)
    ax_met.set_title("Evaluation Metrics", fontsize=11, fontweight="bold")
    ax_met.set_ylabel("Value")
    for bar, val in zip(bars, met_vals):
        ax_met.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", fontsize=9)

    # Row 2: Summary text
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis("off")
    summary_lines = [
        f"Environment: {env_name}",
        f"Diffusion Critic — Sampling MAE: {eval_diff.get('mean_mae', 0):.3f}  |  "
        f"Sample std: {eval_diff.get('diff_sample_std', 0):.3f}  (True std: {eval_diff.get('true_return_std', 0):.3f})",
        f"MLP Critic       — MSE: {eval_mlp.get('mse', 0):.3f}  |  MAE: {eval_mlp.get('mae', 0):.3f}",
        "",
        "Interpretation: Diffusion Critic captures full return distribution → provides uncertainty estimates.",
        "MLP Critic only provides point estimate → loses distribution structure (important for risk-sensitive decisions).",
    ]
    ax_text.text(0.01, 0.9, "\n".join(summary_lines), fontsize=10, fontfamily="monospace",
                 verticalalignment="top")

    fig.suptitle(f"Diffusion Critic vs MLP Critic — {env_name}", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Figure saved: {save_path}")


# ============================================================================
# 主入口
# ============================================================================

def run_toy_experiment(args):
    """运行 Toy Environment 实验."""
    print(f"\n{'='*60}")
    print(f"Toy Environment — Diffusion Critic Experiment")
    print(f"{'='*60}\n")

    set_seed(args.seed)
    device = args.device

    # 数据
    states, returns = gen_toy_data(n_episodes=args.n_episodes, seed=args.seed)
    dataset = TensorDataset(torch.from_numpy(states).float(), torch.from_numpy(returns).float())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # TensorBoard
    tb_dir = os.path.join(args.save_dir, "tensorboard", f"toy_seed{args.seed}")
    tb_writer = SummaryWriter(log_dir=tb_dir)
    print(f"  TensorBoard: {tb_dir}")

    # 模型
    diff_critic = ToyDiffusionCritic(
        state_dim=4, n_timesteps=args.n_timesteps, hidden_dim=args.hidden_dim,
    )
    mlp_critic = MLPCritic(state_dim=4, hidden_dim=args.hidden_dim)

    # 训练
    print("\n[Training Diffusion Critic]")
    loss_diff = train_model(diff_critic, dataloader, args.n_epochs, args.lr,
                            device, tb_writer, "diff_critic", args.log_interval)

    print("\n[Training MLP Critic]")
    loss_mlp = train_model(mlp_critic, dataloader, args.n_epochs, args.lr,
                           device, tb_writer, "mlp_critic", args.log_interval)

    # 保存
    save_model(diff_critic, os.path.join(args.save_dir, "models", "diff_critic_toy.pt"))
    save_model(mlp_critic, os.path.join(args.save_dir, "models", "mlp_critic_toy.pt"))

    # 评估
    print("\n[Evaluation]")
    eval_states = torch.from_numpy(states[:100]).float()
    eval_returns = torch.from_numpy(returns[:100]).float()

    eval_diff = evaluate_diffusion_critic(diff_critic, eval_states, eval_returns, device)
    eval_mlp = evaluate_mlp_critic(mlp_critic, eval_states, eval_returns, device)

    # 详细评估: S₀ 状态
    s0 = torch.zeros(1, 4, device=device)
    s0[0, 0] = 1.0
    with torch.no_grad():
        s0_samples = diff_critic.sample(s0, n_samples=5000).squeeze(-1).cpu().numpy()

    print(f"\n  === S₀ 价值分布 (Diffusion Critic) ===")
    print(f"  True distribution: 50% -10, 50% +10")
    print(f"  Sampled mean/std: {s0_samples.mean():.2f} / {s0_samples.std():.2f}")
    neg_mask = s0_samples < 0
    pos_mask = s0_samples > 0
    print(f"  Negative peak (R≈-10): {neg_mask.mean():.1%}, mean={s0_samples[neg_mask].mean():.2f}")
    print(f"  Positive peak (R≈+10): {pos_mask.mean():.1%}, mean={s0_samples[pos_mask].mean():.2f}")

    # 保存评估结果
    results = {
        "s0_negative_peak_ratio": float(neg_mask.mean()),
        "s0_negative_peak_mean": float(s0_samples[neg_mask].mean()) if neg_mask.any() else 0,
        "s0_positive_peak_mean": float(s0_samples[pos_mask].mean()) if pos_mask.any() else 0,
        "diff_mae": eval_diff["mean_mae"],
        "mlp_mae": eval_mlp["mae"],
        "diff_sample_std": eval_diff["diff_sample_std"],
    }
    with open(os.path.join(args.save_dir, "results_toy.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 可视化
    print("\n[Visualization]")
    s0_samples_2d = s0_samples.reshape(-1, 1)
    true_returns_2d = np.array([0.0])  # S₀ 真值均值
    mlp_preds_2d = mlp_critic(s0).cpu().detach().numpy().flatten()

    make_comparison_figure(
        diff_samples=s0_samples_2d,
        true_returns=true_returns_2d,
        mlp_preds=mlp_preds_2d,
        loss_diff=loss_diff,
        loss_mlp=loss_mlp,
        eval_diff=eval_diff,
        eval_mlp=eval_mlp,
        env_name="Toy (Bimodal)",
        save_path=os.path.join(args.save_dir, "comparison_toy.png"),
    )

    tb_writer.close()
    print(f"\n✅ Toy experiment complete! Results in: {args.save_dir}")


def run_smac_experiment(args):
    """运行 SMAC 实验."""
    env_name = args.env  # 如 "3m-Good"
    print(f"\n{'='*60}")
    print(f"SMAC {env_name} — Diffusion Critic Experiment")
    print(f"{'='*60}\n")

    set_seed(args.seed)
    device = args.device

    # 数据
    data_dir = os.path.join(args.data_base, env_name) if args.data_base else env_name
    states_np, returns_np, info = load_smac_returns(
        data_dir=data_dir, discount=0.99, max_episodes=args.n_episodes,
    )

    # 更好的归一化: z-score
    return_mean = returns_np.mean()
    return_std = returns_np.std() + 1e-6
    returns_norm = (returns_np - return_mean) / return_std

    dataset = TensorDataset(
        torch.from_numpy(states_np).float(),
        torch.from_numpy(returns_norm).float(),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # TensorBoard
    tb_dir = os.path.join(args.save_dir, "tensorboard", f"smac_{env_name}_seed{args.seed}")
    tb_writer = SummaryWriter(log_dir=tb_dir)
    print(f"  TensorBoard: {tb_dir}")
    print(f"  State dim: {info['state_dim']}, Samples: {info.get('n_samples', 0):,}")
    print(f"  Return range: [{info.get('return_min', 0):.1f}, {info.get('return_max', 0):.1f}]")
    print(f"  Return mean/std: {return_mean:.2f} / {return_std:.2f}")

    # 模型
    diff_critic = AdvDiffusionCritic(
        state_dim=info["state_dim"],
        n_timesteps=args.n_timesteps,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        return_scale=1.0,  # 已 z-score 归一化，不需要额外 scale
    )
    mlp_critic = MLPCritic(state_dim=info["state_dim"], hidden_dim=args.hidden_dim)

    # 训练
    print("\n[Training Diffusion Critic]")
    loss_diff = train_model(diff_critic, dataloader, args.n_epochs, args.lr,
                            device, tb_writer, "diff_critic", args.log_interval)

    print("\n[Training MLP Critic]")
    loss_mlp = train_model(mlp_critic, dataloader, args.n_epochs, args.lr,
                           device, tb_writer, "mlp_critic", args.log_interval)

    # 保存
    model_dir = os.path.join(args.save_dir, "models")
    save_model(diff_critic, os.path.join(model_dir, f"diff_critic_smac_{env_name}.pt"))
    save_model(mlp_critic, os.path.join(model_dir, f"mlp_critic_smac_{env_name}.pt"))

    # 评估 (反归一化)
    print("\n[Evaluation]")
    n_eval = min(100, len(states_np))
    eval_idx = np.random.choice(len(states_np), n_eval, replace=False)
    eval_states = torch.from_numpy(states_np[eval_idx]).float()
    eval_returns_norm = torch.from_numpy(returns_norm[eval_idx]).float()
    eval_returns_true = eval_returns_norm * return_std + return_mean

    eval_diff_norm = evaluate_diffusion_critic(diff_critic, eval_states, eval_returns_norm, device)
    eval_mlp_norm = evaluate_mlp_critic(mlp_critic, eval_states, eval_returns_norm, device)

    # 反归一化 MLP 评估
    with torch.no_grad():
        mlp_preds_norm = mlp_critic(eval_states.to(device)).squeeze(-1)
    mlp_preds = mlp_preds_norm * return_std + return_mean
    eval_mlp = {
        "mse": float(((mlp_preds - eval_returns_true.to(device)) ** 2).mean().item()),
        "mae": float((mlp_preds - eval_returns_true.to(device)).abs().mean().item()),
        "pred_mean": float(mlp_preds.mean().item()),
        "pred_std": float(mlp_preds.std().item()),
    }

    # Diffusion 采样 (反归一化)
    with torch.no_grad():
        diff_samples_norm = diff_critic.sample(eval_states[:4].to(device),
                                                n_samples=1000).cpu().numpy()
    diff_samples = diff_samples_norm * return_std + return_mean
    true_returns_sel = eval_returns_true[:4].cpu().numpy()
    mlp_preds_sel = mlp_preds[:4].cpu().detach().numpy()

    print(f"  Diffusion Critic MAE (normed): {eval_diff_norm['mean_mae']:.3f}")
    print(f"  MLP Critic MAE (real): {eval_mlp['mae']:.3f}")

    # 平滑损失曲线用于显示
    def smooth(y, w=5):
        if len(y) < w: return y
        return np.convolve(y, np.ones(w)/w, mode='valid')

    make_comparison_figure(
        diff_samples=diff_samples,
        true_returns=true_returns_sel,
        mlp_preds=mlp_preds_sel,
        loss_diff=smooth(loss_diff, 10),
        loss_mlp=smooth(loss_mlp, 10),
        eval_diff={"mean_mae": eval_diff_norm["mean_mae"] * return_std,
                    "diff_sample_std": float(diff_samples.std()),
                    "true_return_std": return_std},
        eval_mlp=eval_mlp,
        env_name=f"SMAC {env_name}",
        save_path=os.path.join(args.save_dir, f"comparison_smac_{env_name}.png"),
    )

    # 保存结果
    results = {
        "env": env_name,
        "state_dim": info["state_dim"],
        "n_samples": info.get("n_samples", 0),
        "return_mean": float(return_mean),
        "return_std": float(return_std),
        "diff_mae_real": float(eval_diff_norm["mean_mae"] * return_std),
        "mlp_mae_real": eval_mlp["mae"],
        "diff_sample_std_real": float(diff_samples.std()),
    }
    with open(os.path.join(args.save_dir, f"results_smac_{env_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    tb_writer.close()
    print(f"\n✅ SMAC experiment complete! Results in: {args.save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Diffusion Critic 统一实验运行器")
    parser.add_argument("--env", type=str, default="toy",
                        help="实验: toy, smac_3m, smac_8m, smac_5m6m")
    parser.add_argument("--data_base", type=str, default="diffuser/datasets/data/smac")
    parser.add_argument("--n_episodes", type=int, default=5000)
    parser.add_argument("--n_timesteps", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "runs"))
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 实验路由
    env_map = {
        "toy": lambda: run_toy_experiment(args),
        "smac_3m": lambda: (setattr(args, "env", "3m-Good"), run_smac_experiment(args))[-1],
        "smac_8m": lambda: (setattr(args, "env", "8m-Good"), run_smac_experiment(args))[-1],
        "smac_5m6m": lambda: (setattr(args, "env", "5m_vs_6m-Good"), run_smac_experiment(args))[-1],
    }

    assert args.env in env_map, f"Unknown env: {args.env}. Choose from: {list(env_map.keys())}"
    env_map[args.env]()

    # 提示
    print(f"\n📊 启动 TensorBoard 查看训练曲线:")
    print(f"   tensorboard --logdir {os.path.join(args.save_dir, 'runs')} --port 6006")


if __name__ == "__main__":
    main()
