"""
SMAC Offline 数据分析和 Diffusion Critic 训练脚本
==================================================
加载 SMAC 离线数据集，提取 (state, return) 对，训练 Diffusion Critic
并和 MLP Critic 对比。
"""

import argparse
import json
import os
import sys
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

warnings.filterwarnings("ignore")

# ===================== 路径解析 =====================
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # madiff/
_model_path = _PROJECT_ROOT / "custom" / "my_test" / "model"
sys.path.insert(0, str(_model_path))
from diffusion_actor import RiskGuidedDiffusion

# ===================== SMAC Dataset Loader =====================

def load_smac_data(
    data_dir: str,
    max_transitions: int = None,
    gamma: float = 0.99,
):
    """
    加载 SMAC 离线数据，计算每个时间步的 return-to-go。

    Args:
        data_dir: 如 diffuser/datasets/data/smac/3m/Good/
        max_transitions: 最大样本数
        gamma: 折扣因子

    Returns:
        states: [N, D] 全局状态
        returns: [N] return-to-go
    """
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = _PROJECT_ROOT / data_dir

    # 加载数据
    states = np.load(str(data_dir / "states.npy")).astype(np.float32)     # [T, D]
    rewards = np.load(str(data_dir / "rewards.npy")).astype(np.float32)   # [T, A]
    path_lengths = np.load(str(data_dir / "path_lengths.npy"))           # [n_episodes]

    # 所有 agent 共享 reward → 取第一个 agent
    team_rewards = rewards[:, 0]  # [T]

    # 路径分割 → 每个 episode 单独计算 return-to-go
    returns_list = []
    start = 0
    for ep_len in path_lengths:
        ep_rewards = team_rewards[start : start + ep_len]  # [L]
        L = len(ep_rewards)
        # 反向计算 return-to-go
        rtg = np.zeros(L, dtype=np.float32)
        discounted = 0.0
        for t in reversed(range(L)):
            discounted = ep_rewards[t] + gamma * discounted
            rtg[t] = discounted
        returns_list.extend(rtg.tolist())
        start += ep_len

    # 如果数据量大则降采样
    if max_transitions is not None and len(states) > max_transitions:
        idx = np.random.choice(len(states), max_transitions, replace=False)
        states = states[idx]
        returns = np.array(returns_list, dtype=np.float32)[idx]
    else:
        returns = np.array(returns_list, dtype=np.float32)

    print(f"  SMAC data loaded: {len(states)} samples, state_dim={states.shape[1]}")
    print(f"  Return stats: mean={returns.mean():.4f}, std={returns.std():.4f}")
    print(f"  Return min={returns.min():.4f}, max={returns.max():.4f}")

    return states, returns


def normalize_data(states, returns):
    """对状态和回报做简单归一化，返回归一化器参数用于反归一化。"""
    s_mean = states.mean(axis=0)
    s_std = states.std(axis=0) + 1e-8
    states_norm = (states - s_mean) / s_std
    r_mean = returns.mean()
    r_std = returns.std() + 1e-8
    returns_norm = (returns - r_mean) / r_std
    norm_info = {
        "s_mean": s_mean, "s_std": s_std,
        "r_mean": r_mean, "r_std": r_std,
    }
    return states_norm, returns_norm, norm_info


# ===================== Diffusion Critic =====================

class TimeEmbedding(nn.Module):
    """时间步嵌入 (同 DDPM 标准实现)。"""
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.net(emb)


class ResidualMLPBlock(nn.Module):
    """残差 MLP 块。"""
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class DiffusionCriticNet(nn.Module):
    """
    条件扩散去噪网络。
    输入: 带噪回报 R_t (scalar), 时间步 t, 状态 s
    输出: 预测噪声 ε
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.return_proj = nn.Linear(1, hidden_dim)
        self.time_emb = TimeEmbedding(hidden_dim)

        input_dim = hidden_dim * 3  # state + return + time
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])

        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, s, r_t, t):
        """
        Args:
            s: [B, state_dim]
            r_t: [B, 1] 带噪回报
            t: [B] 时间步
        """
        s_h = self.state_proj(s)           # [B, H]
        r_h = self.return_proj(r_t)        # [B, H]
        t_h = self.time_emb(t)             # [B, H]
        h = torch.cat([s_h, r_h, t_h], dim=-1)  # [B, 3H]
        h = self.input_proj(h)             # [B, H]
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)         # [B, 1]


# ===================== DDPM Scheduler =====================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DDPMScheduler:
    """DDPM 噪声调度器 (用于标量回报扩散)。"""
    def __init__(self, n_timesteps: int = 100):
        self.n_timesteps = n_timesteps
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer = lambda name, val: setattr(self, name, val)
        for name, val in [
            ("betas", betas),
            ("alphas", alphas),
            ("alphas_cumprod", alphas_cumprod),
            ("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)),
            ("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)),
        ]:
            self.register_buffer(name, val)

    def to(self, device):
        for attr in ["betas", "alphas", "alphas_cumprod",
                      "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self, r0, t):
        """前向加噪: r_t = sqrt(ᾱ_t)·r_0 + sqrt(1-ᾱ_t)·ε"""
        eps = torch.randn_like(r0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        r_t = sqrt_alpha * r0 + sqrt_one_minus * eps
        return r_t, eps

    def p_sample(self, model, s, r_t, t_idx, returns_clamp: float = 10.0):
        """
        单步逆向采样。
        model: DiffusionCriticNet
        t_idx: int (当前步索引, 从 T-1 到 0)
        """
        B = s.shape[0]
        device = s.device
        t = torch.full((B,), t_idx, device=device, dtype=torch.long)
        eps_pred = model(s, r_t, t)

        alpha = self.alphas[t_idx]
        alpha_bar = self.alphas_cumprod[t_idx]
        beta = self.betas[t_idx]

        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        x0_pred = (r_t - torch.sqrt(1.0 - alpha_bar) * eps_pred) / sqrt_alpha_bar
        x0_pred = torch.clamp(x0_pred, -returns_clamp, returns_clamp)

        if t_idx > 0:
            alpha_bar_prev = self.alphas_cumprod[t_idx - 1]
            coef1 = torch.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar)
            coef2 = torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            mean = coef1 * x0_pred + coef2 * r_t
            sigma = torch.sqrt(beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
            return mean + sigma * torch.randn_like(r_t)
        else:
            return x0_pred

    @torch.no_grad()
    def sample(self, model, s, n_samples: int = 1):
        """从模型生成 n_samples 个回报样本。"""
        B = s.shape[0]
        device = s.device
        r_t = torch.randn(B, n_samples if n_samples > 1 else 1, device=device).squeeze(-1)

        # 扩展状态以匹配 n_samples
        if n_samples > 1:
            s = s.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, s.shape[-1])
            r_t = r_t.reshape(-1, 1)

        for t_idx in reversed(range(self.n_timesteps)):
            r_t = self.p_sample(model, s, r_t, t_idx)

        return r_t.squeeze(-1)


# ===================== MLP Critic (Baseline) =====================

class MLPCritic(nn.Module):
    """标准标量值函数。"""
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)


# ===================== Training =====================

def train_diffusion_critic(
    model, scheduler, train_loader, epochs: int = 100, lr: float = 1e-3, device: str = "cpu"
):
    """训练 Diffusion Critic (DDPM 噪声预测)。"""
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    losses = []

    t_range = trange(epochs, desc="DiffusionCritic")
    for epoch in t_range:
        total_loss = 0.0
        n_batches = 0
        for s, r in train_loader:
            s, r = s.to(device), r.to(device).unsqueeze(-1)  # [B, D], [B, 1]
            B = s.shape[0]

            t = torch.randint(0, scheduler.n_timesteps, (B,), device=device).long()
            r_t, eps = scheduler.q_sample(r, t)  # [B, 1]
            eps_pred = model(s, r_t, t)           # [B, 1]
            loss = F.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        losses.append(avg_loss)
        t_range.set_postfix(loss=f"{avg_loss:.6f}")

    return losses


def train_mlp_critic(
    model, train_loader, epochs: int = 100, lr: float = 1e-3, device: str = "cpu"
):
    """训练标准 MLP Critic (MSE 回归)。"""
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    losses = []

    t_range = trange(epochs, desc="MLPCritic")
    for epoch in t_range:
        total_loss = 0.0
        n_batches = 0
        for s, r in train_loader:
            s, r = s.to(device), r.to(device)
            pred = model(s)                      # [B]
            loss = F.mse_loss(pred, r)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        losses.append(avg_loss)
        t_range.set_postfix(loss=f"{avg_loss:.4f}")

    return losses


# ===================== Evaluation =====================

@torch.no_grad()
def evaluate_critics(
    diffusion_model, scheduler, mlp_model, states, returns, norm_info, device: str = "cpu"
):
    """
    评估两个 Critic。

    Returns:
        dict: 评估结果字典
    """
    s = torch.from_numpy(states).to(device)
    r_true = torch.from_numpy(returns).to(device)

    # MLP 预测
    mlp_pred = mlp_model(s).cpu().numpy()

    # Diffusion 采样
    n_samples = 64
    diffusion_samples = []
    batch_size = 256
    for i in range(0, len(states), batch_size):
        s_batch = s[i : i + batch_size]
        samples = scheduler.sample(diffusion_model, s_batch, n_samples=n_samples)
        diffusion_samples.append(samples.cpu().numpy())
    diffusion_samples = np.concatenate(diffusion_samples, axis=0)

    # 反归一化
    r_mean, r_std = norm_info["r_mean"], norm_info["r_std"]
    mlp_pred = mlp_pred * r_std + r_mean
    diffusion_samples = diffusion_samples * r_std + r_mean
    r_true_np = returns * r_std + r_mean

    # 统计
    diff_mean = diffusion_samples.mean(axis=-1)
    diff_std = diffusion_samples.std(axis=-1)

    return {
        "true_returns": r_true_np.tolist(),
        "mlp_predictions": mlp_pred.tolist(),
        "diffusion_mean": diff_mean.tolist(),
        "diffusion_std": diff_std.tolist(),
        "diffusion_samples": diffusion_samples.tolist(),
    }


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="e.g. diffuser/datasets/data/smac/3m/Good")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_timesteps", type=int, default=50,
                        help="DDPM 扩散步数")
    parser.add_argument("--max_transitions", type=int, default=50000,
                        help="最大样本数（降采样）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or str(
        _PROJECT_ROOT / "custom" / "diffusion_critic" / "results" / "smac_analysis"
    )
    os.makedirs(output_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Device: {device}")

    # 1. 加载数据
    print("[1/4] Loading SMAC data...")
    states, returns = load_smac_data(args.data_dir, max_transitions=args.max_transitions)

    # 2. 归一化
    print("[2/4] Normalizing...")
    states_norm, returns_norm, norm_info = normalize_data(states, returns)

    # 3. 构建 DataLoader
    print(f"[3/4] Building models (state_dim={states.shape[1]})...")
    dataset = TensorDataset(
        torch.from_numpy(states_norm).float(),
        torch.from_numpy(returns_norm).float(),
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    state_dim = states.shape[1]

    # Diffusion Critic
    diff_model = DiffusionCriticNet(
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=3,
    ).to(device)
    scheduler = DDPMScheduler(n_timesteps=args.n_timesteps).to(device)

    # MLP Critic (baseline)
    mlp_model = MLPCritic(state_dim=state_dim, hidden_dim=args.hidden_dim).to(device)

    # 4. 训练
    print("[4/4] Training...")
    print("  Training Diffusion Critic...")
    diff_losses = train_diffusion_critic(
        diff_model, scheduler, train_loader,
        epochs=args.epochs, lr=args.lr, device=device,
    )

    print("  Training MLP Critic...")
    mlp_losses = train_mlp_critic(
        mlp_model, train_loader,
        epochs=args.epochs, lr=args.lr, device=device,
    )

    # 5. 评估
    print("  Evaluating...")
    eval_results = evaluate_critics(
        diff_model, scheduler, mlp_model, states_norm[:500], returns_norm[:500],
        norm_info, device=device,
    )

    # 保存结果
    env_name = Path(args.data_dir).parts[-2] if "/" in args.data_dir else args.data_dir.replace("/", "_")
    summary = {
        "data_dir": args.data_dir,
        "state_dim": state_dim,
        "n_samples": len(states),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "diffusion_loss_final": float(diff_losses[-1]) if diff_losses else None,
        "mlp_loss_final": float(mlp_losses[-1]) if mlp_losses else None,
        "return_stats": {
            "mean": float(returns.mean()),
            "std": float(returns.std()),
            "min": float(returns.min()),
            "max": float(returns.max()),
        },
        "eval_samples": eval_results["true_returns"][:20],
        "mlp_predictions": eval_results["mlp_predictions"][:20],
        "diffusion_mean": eval_results["diffusion_mean"][:20],
        "diffusion_std": eval_results["diffusion_std"][:20],
    }

    summary_path = os.path.join(output_dir, f"{env_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSummary saved: {summary_path}")
    print(f"  Diffusion MSE: {summary['diffusion_loss_final']:.6f}")
    print(f"  MLP MSE: {summary['mlp_loss_final']:.6f}")


if __name__ == "__main__":
    main()
