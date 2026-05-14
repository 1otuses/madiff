"""
升级版 Diffusion Critic — 支持高维状态 (SMAC/MPE/MAMuJoCo)
=========================================================
相比 toy_env 版本的核心改进:
  1. 去噪网络从简单 MLP → 残差 MLP (ResMLP)
  2. 支持高维状态条件 (SMAC: 80~200 维; MPE: 20~60 维)
  3. 条件嵌入维度可配置
  4. 支持 return 归一化/反归一化
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 基础模块 (与 toy version 相同的辅助模块)
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]  # [B, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, dim]
        return emb


class ResidualBlock(nn.Module):
    """带 LayerNorm 的残差块."""
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # [B, dim] -> [B, dim]


class ResMLPDenoiser(nn.Module):
    """
    残差 MLP 去噪网络 ε_θ(R_noisy, t, s) → ε_pred.

    架构:
      s ──→ Linear(state_dim, hidden_dim) ──┐
      t ──→ SinusoidalPosEmb → MLP ─────────┤
                                             ├── + ──→ ResBlock × n_blocks ──→ Linear → ε_pred
      R_t ──→ Linear(1, hidden_dim) ─────────┘

    维度标注:
      B = batch, D_s = state_dim, D_h = hidden_dim, T = diffusion timesteps
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)  # [D_s] -> [D_h]
        self.return_proj = nn.Linear(1, hidden_dim)          # [1] -> [D_h]
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.output = nn.Linear(hidden_dim, 1)  # [D_h] -> [1]

    def forward(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        """
        x: 带噪回报    [B, 1]
        t: 扩散时间步  [B]
        s: 状态条件    [B, state_dim]
        """
        # 三个输入各自投影到 hidden_dim
        h_return = self.return_proj(x)       # [B, 1] -> [B, D_h]
        h_state = self.state_proj(s)         # [B, D_s] -> [B, D_h]
        h_time = self.time_mlp(t)            # [B] -> [B, D_h]

        # 融合: 加法 (也可用 concat + linear，但加法参数效率更高)
        h = h_return + h_state + h_time      # [B, D_h]

        # 残差块
        for block in self.blocks:
            h = block(h)                      # [B, D_h]

        # 输出噪声预测
        epsilon = self.output(h)              # [B, D_h] -> [B, 1]
        return epsilon


# ============================================================================
# 扩散 Critic (与 toy version 完全相同的 DDPM 逻辑)
# ============================================================================

class DiffusionCritic(nn.Module):
    """
    用 DDPM 建模 P(R|s) 的 Critic (升级版).

    训练: loss()  → L_simple (DDPM 噪声预测损失)
    推理: sample() → 从 p_θ(R|s) 采样价值分布
    """

    def __init__(
        self,
        state_dim: int,
        n_timesteps: int = 200,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.0,
        return_scale: float = 1.0,  # return 缩放因子 (用于归一化)
    ):
        super().__init__()
        self.denoiser = ResMLPDenoiser(state_dim, hidden_dim, n_blocks, dropout)
        self.n_timesteps = n_timesteps
        self.return_scale = return_scale

        # β schedule: 平方余弦 (与 MADiff 一致)
        betas = self._cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008):
        """余弦 beta 调度 (来自 improved DDPM)."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    # ---- 训练 ----

    def loss(
        self,
        returns: torch.Tensor,  # [B] 真实回报
        states: torch.Tensor,   # [B, state_dim] 状态
    ) -> torch.Tensor:
        """DDPM 训练损失 (噪声预测 MSE)."""
        B = returns.shape[0]
        if returns.ndim == 1:
            returns = returns.unsqueeze(-1)  # [B, 1]

        # 归一化 return
        returns = returns / self.return_scale

        # 采样扩散时间步
        t = torch.randint(0, self.n_timesteps, (B,), device=returns.device).long()

        # 采样噪声
        noise = torch.randn_like(returns)

        # 前向加噪: R_t = √(ᾱ_t)·R₀ + √(1-ᾱ_t)·ε
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        noisy_returns = sqrt_alpha_bar * returns + sqrt_one_minus * noise

        # 预测噪声
        noise_pred = self.denoiser(noisy_returns, t, states)

        # MSE
        loss_val = F.mse_loss(noise_pred, noise)
        return loss_val

    # ---- 推理 ----

    @torch.no_grad()
    def sample(
        self,
        states: torch.Tensor,
        n_samples: int = 500,
    ) -> torch.Tensor:
        """
        从 p_θ(R|s) 采样回报。
        返回: [n_samples, B] (已反归一化)
        """
        B = states.shape[0]
        device = states.device

        # 扩展状态
        states_exp = states.unsqueeze(0).expand(n_samples, B, -1)  # [n_s, B, D]
        states_exp = states_exp.reshape(n_samples * B, -1)         # [n_s*B, D]

        # 初始化: R_T ~ N(0, I)
        x = torch.randn(n_samples * B, 1, device=device)

        # DDPM 反向去噪
        for t_idx in reversed(range(self.n_timesteps)):
            t = torch.full((n_samples * B,), t_idx, device=device, dtype=torch.long)
            epsilon = self.denoiser(x, t, states_exp)

            alpha = self.alphas[t_idx]
            alpha_bar = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]

            # 估计 x₀
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar) * epsilon) / torch.sqrt(alpha_bar)

            # 后验均值
            if t_idx > 0:
                alpha_bar_prev = self.alphas_cumprod[t_idx - 1]
                coef1 = torch.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar)
                coef2 = torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                mean = coef1 * x0_pred + coef2 * x
            else:
                # t=0: 直接取 x0_pred
                mean = x0_pred

            # 添加方差 (t>0 时)
            if t_idx > 0:
                noise = torch.randn_like(x)
                # 后验方差
                if t_idx > 1:
                    beta_tilde = beta * (1.0 - self.alphas_cumprod[t_idx - 1]) / (1.0 - alpha_bar)
                else:
                    beta_tilde = beta
                sigma = torch.sqrt(beta_tilde)
                x = mean + sigma * noise
            else:
                x = mean

        x = x.reshape(n_samples, B)  # [n_samples, B]
        return x * self.return_scale  # 反归一化

    @torch.no_grad()
    def expected_value(self, states: torch.Tensor, n_samples: int = 500) -> torch.Tensor:
        """E[R|s] 估计 (用于与 MLP 对比)."""
        samples = self.sample(states, n_samples=n_samples)
        return samples.mean(dim=0)  # [B]

    @torch.no_grad()
    def value_distribution_stats(self, states: torch.Tensor, n_samples: int = 1000):
        """
        返回价值分布的统计量: mean, std, quantiles.
        """
        samples = self.sample(states, n_samples=n_samples)  # [n_samples, B]
        mean = samples.mean(dim=0)     # [B]
        std = samples.std(dim=0)       # [B]
        q10 = samples.quantile(0.1, dim=0)  # [B]
        q50 = samples.quantile(0.5, dim=0)  # [B]
        q90 = samples.quantile(0.9, dim=0)  # [B]
        return mean, std, q10, q50, q90, samples
