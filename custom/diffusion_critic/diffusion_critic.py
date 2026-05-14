"""
Diffusion Critic: 用扩散模型近似价值分布 P(R|s)
================================================
数学原理:
---------
传统 Critic: Q(s,a) = E[R|s,a]  — 仅学习价值期望
Diffusion Critic: p_θ(R|s) ≈ P_true(R|s)  — 学习完整价值分布

训练目标 (DDPM 损失):
  L(θ) = E_{t, ε, R₀, s} [ ‖ε - ε_θ(√(ᾱ_t)·R₀ + √(1-ᾱ_t)·ε, t, s)‖² ]

采样 (推理时):
  从噪声开始 R_T ~ N(0,1)，逐步去噪至 R₀,即得到价值分布样本。

条件机制:
  将状态 s 作为条件嵌入到去噪网络的时间嵌入中，
  类似 MADiff 中的 returns_condition 的逆过程。
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# 1. 基础模块
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """正弦位置编码，与 MADiff 中 models.helpers.SinusoidalPosEmb 一致。"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        # x: [B] 扩散时间步
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]  # [B, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, dim]
        return emb


class TimeStateConditionEmbed(nn.Module):
    """
    将扩散时间步 t 和状态 s 融合为条件嵌入:
      emb = time_mlp(sinusoidal(t)) + state_mlp(s)
    """
    def __init__(self, state_dim: int, embed_dim: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor, s: torch.Tensor):
        # t: [B], s: [B, state_dim]
        t_emb = self.time_mlp(t)        # [B, embed_dim]
        s_emb = self.state_mlp(s)       # [B, embed_dim]
        return t_emb + s_emb            # [B, embed_dim]


class DiffusionDenoiser(nn.Module):
    """
    去噪网络 ε_θ(R_noisy, t, s) → ε_pred
    用于预测噪声，输入为:
      - R_noisy: [B, 1] 带噪价值
      - t: [B] 扩散时间步
      - s: [B, state_dim] 状态条件
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.condition_embed = TimeStateConditionEmbed(state_dim, embed_dim=64)

        # 输入: [B, 1] 价值 + [B, 64] 条件嵌入 → concat → [B, 65]
        self.net = nn.Sequential(
            nn.Linear(1 + 64, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),  # 输出 ε_pred: [B, 1]
        )

    def forward(
        self,
        x: torch.Tensor,          # [B, 1] noisy return
        t: torch.Tensor,          # [B] diffusion timestep
        s: torch.Tensor,          # [B, state_dim] state condition
    ) -> torch.Tensor:
        # 1. 构建条件嵌入
        cond = self.condition_embed(t, s)  # [B, 64]

        # 2. 拼接输入
        net_input = torch.cat([x, cond], dim=-1)  # [B, 1+64]

        # 3. MLP 预测噪声
        epsilon_pred = self.net(net_input)  # [B, 1]
        return epsilon_pred


# ============================================================================
# 2. 扩散 Critic 封装
# ============================================================================

class DiffusionCritic(nn.Module):
    """
    用 DDPM 建模 P(R|s) 的 Critic。

    训练: loss()  → L_simple (DDPM 噪声预测损失)
    推理: sample() → 从 p_θ(R|s) 采样价值
    """

    def __init__(
        self,
        state_dim: int,
        n_timesteps: int = 100,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.denoiser = DiffusionDenoiser(state_dim, hidden_dim)
        self.n_timesteps = n_timesteps

        # 构建 β schedule (线性, β ∈ [1e-4, 0.02])
        betas = torch.linspace(1e-4, 0.02, n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)                     # [T]
        self.register_buffer("alphas", alphas)                   # [T]
        self.register_buffer("alphas_cumprod", alphas_cumprod)   # [T]
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)
        )  # √(ᾱ_t)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )  # √(1-ᾱ_t)

    # ---- 训练 ----

    def loss(
        self,
        returns: torch.Tensor,    # [B] 或 [B, 1] 真实回报
        states: torch.Tensor,     # [B, state_dim] 状态
    ) -> torch.Tensor:
        """DDPM 训练损失 L_simple。"""
        B = returns.shape[0]
        if returns.ndim == 1:
            returns = returns.unsqueeze(-1)  # [B, 1]

        # 1. 随机采样扩散时间步
        t = torch.randint(0, self.n_timesteps, (B,), device=returns.device).long()

        # 2. 采样噪声
        noise = torch.randn_like(returns)  # [B, 1]

        # 3. 前向加噪: R_t = √(ᾱ_t)·R₀ + √(1-ᾱ_t)·ε
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].unsqueeze(-1)          # [B, 1]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)  # [B, 1]
        noisy_returns = sqrt_alpha_bar * returns + sqrt_one_minus_alpha_bar * noise  # [B, 1]

        # 4. 预测噪声
        noise_pred = self.denoiser(noisy_returns, t, states)  # [B, 1]

        # 5. MSE 损失
        loss_val = F.mse_loss(noise_pred, noise)
        return loss_val

    # ---- 推理 / 采样 ----

    @torch.no_grad()
    def sample(
        self,
        states: torch.Tensor,        # [B, state_dim] 状态条件
        n_samples: int = 1000,       # 每个状态采样的回报数量
        return_all_steps: bool = False,
    ) -> torch.Tensor:
        """
        从 p_θ(R|s) 采样回报。

        返回:
          如果 return_all_steps=False: [n_samples, B] 最终采样回报
          如果 return_all_steps=True: [n_samples, B, T+1] 完整扩散轨迹
        """
        B = states.shape[0]
        device = states.device

        # 将 states 扩展以适应多个采样
        states_expanded = states.unsqueeze(0).expand(n_samples, B, -1)  # [n_s, B, D]
        states_expanded = states_expanded.reshape(n_samples * B, -1)    # [n_s*B, D]

        # 初始化: R_T ~ N(0, 1)
        x = torch.randn(n_samples * B, 1, device=device)  # [n_s*B, 1]

        trajectory = [x] if return_all_steps else None

        # DDPM 反向采样 (逐步去噪)
        for t_idx in reversed(range(self.n_timesteps)):
            t = torch.full((n_samples * B,), t_idx, device=device, dtype=torch.long)

            # 预测噪声
            epsilon = self.denoiser(x, t, states_expanded)  # [n_s*B, 1]

            # 计算 p(x_{t-1} | x_t) 的均值
            alpha = self.alphas[t_idx]           # 标量
            alpha_bar = self.alphas_cumprod[t_idx]  # 标量
            beta = self.betas[t_idx]             # 标量

            # 估计 x₀
            x0_pred = (x - torch.sqrt(1 - alpha_bar) * epsilon) / torch.sqrt(
                alpha_bar
            )

            # 后验均值
            if t_idx > 0:
                alpha_bar_prev = self.alphas_cumprod[t_idx - 1]
                mean = (
                    torch.sqrt(alpha_bar_prev) * beta * x0_pred
                    + torch.sqrt(alpha) * (1 - alpha_bar_prev) * x
                ) / (1 - alpha_bar)
            else:
                mean = (
                    torch.sqrt(alpha) * (1 - alpha_bar) * x
                    + torch.sqrt(alpha_bar) * beta * x0_pred
                ) / (1 - alpha_bar)
                # 在 t=0 时简化: mean ≈ x0_pred (因为最后一步几乎无噪声)

            # 添加方差 (t>0 时)
            if t_idx > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean  # t=0 时无噪声

            if return_all_steps:
                trajectory.append(x)

        # 整理输出
        x = x.reshape(n_samples, B)  # [n_samples, B]

        if return_all_steps:
            trajectory = torch.stack(trajectory[::-1], dim=-1)  # [n_s*B, T+1]
            trajectory = trajectory.reshape(n_samples, B, -1)    # [n_samples, B, T+1]
            return x, trajectory

        return x  # [n_samples, B]

    @torch.no_grad()
    def expected_value(self, states: torch.Tensor, n_samples: int = 500) -> torch.Tensor:
        """估算 E[R|s] (用于与标准 Critic 对比)。"""
        samples = self.sample(states, n_samples=n_samples)  # [n_samples, B]
        return samples.mean(dim=0)  # [B]
