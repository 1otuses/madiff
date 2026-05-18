"""
Action-conditioned Diffusion Critic + MLP Actor (Actor-Critic 架构)
==================================================================
Critic:  DiffusionCriticAC — 建模 P(R|s, a), action-conditioned 价值分布
Actor:   MLPActor — 策略网络 π(a|s), 输出离散动作 logits

训练:
  1. Critic: 从离线数据采样 (s, a, R), 最小化 DDPM 噪声预测损失
  2. Actor:  对当前策略采样的动作 a ~ π(·|s)，用 Critic 预估 E[R|s,a],
             最大化期望回报 (策略梯度 / 直接优化)

评估:
  直接加载 Actor,在 SMAC 环境中多 seed 运行，统计累计回报均值±方差
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 基础模块
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResidualBlock(nn.Module):
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


# ============================================================================
# Action-conditioned Denoiser (Critic 核心)
# ============================================================================

class ActionConditionedDenoiser(nn.Module):
    """
    去噪网络 ε_θ(R_noisy, t, s, a) → ε_pred.
    将动作 a 作为额外条件嵌入。

    输入:
      x: 带噪回报 [B, 1]
      t: 扩散时间步 [B]
      s: 状态 [B, state_dim]
      a: 动作 [B, n_agents] (离散, int)

    架构:
      s → Linear(state_dim, D_h)
      a → Embedding(n_actions, D_h) × n_agents → sum → Linear(D_h, D_h)
      t → SinusoidalPosEmb → MLP → D_h
      x → Linear(1, D_h)
      融合: h = h_x + h_s + h_a + h_t
      h → ResBlock×N → Linear(D_h, 1) → ε_pred
    """

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        n_actions: int,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions

        # 各输入投影
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.return_proj = nn.Linear(1, hidden_dim)

        # 动作嵌入: 每个 agent 一个可学习 embedding
        self.act_embed_dim = hidden_dim // n_agents
        self.action_embed = nn.Embedding(n_actions, self.act_embed_dim)
        self.action_proj = nn.Linear(n_agents * self.act_embed_dim, hidden_dim)

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 残差块
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x, t, s, a):
        """
        x: [B, 1]      noisy return
        t: [B]          diffusion timestep
        s: [B, state_dim]  state feature
        a: [B, n_agents]   discrete actions
        """
        B = x.shape[0]

        # 状态投影
        h_s = self.state_proj(s)  # [B, D_h]

        # 动作嵌入: 每个 agent 独立 embedding → sum
        a_emb = self.action_embed(a.long())  # [B, n_agents, D_h//n_agents]
        a_flat = a_emb.reshape(B, -1)        # [B, D_h]
        h_a = self.action_proj(a_flat)       # [B, D_h]

        # 回报投影
        h_x = self.return_proj(x)  # [B, D_h]

        # 时间投影
        h_t = self.time_mlp(t)  # [B, D_h]

        # 融合
        h = h_x + h_s + h_a + h_t  # [B, D_h]

        # 残差块
        for block in self.blocks:
            h = block(h)

        # 输出噪声
        return self.output(h)  # [B, 1]


# ============================================================================
# Action-conditioned Diffusion Critic
# ============================================================================

class DiffusionCriticAC(nn.Module):
    """
    Action-conditioned Diffusion Critic: 建模 P(R|s, a)。

    训练: loss(returns, states, actions) — DDPM 噪声预测
    推理: sample(states, actions) → [n_samples, B] 采样回报
          expected_value(states, actions) → [B] 期望回报 (用于 Actor 更新)
    """

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        n_actions: int,
        n_timesteps: int = 100,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        return_scale: float = 1.0,
    ):
        super().__init__()
        self.denoiser = ActionConditionedDenoiser(
            state_dim, n_agents, n_actions, hidden_dim, n_blocks
        )
        self.n_timesteps = n_timesteps
        self.return_scale = return_scale

        # β schedule (cosine)
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
    def _cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    # ---- 训练 ----

    def loss(self, returns, states, actions):
        """
        returns: [B]       真实回报
        states:  [B, D]    状态
        actions: [B, A]    离散动作
        """
        B = returns.shape[0]
        if returns.ndim == 1:
            returns = returns.unsqueeze(-1)

        # 归一化
        returns = returns / self.return_scale

        # 扩散步采样
        t = torch.randint(0, self.n_timesteps, (B,), device=returns.device).long()
        noise = torch.randn_like(returns)

        # 前向加噪
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        noisy_returns = sqrt_alpha * returns + sqrt_one_minus * noise

        # 预测噪声
        noise_pred = self.denoiser(noisy_returns, t, states, actions)

        return F.mse_loss(noise_pred, noise)

    # ---- 推理 ----

    @torch.no_grad()
    def sample(self, states, actions, n_samples=500):
        """从 p(R|s,a) 采样回报 [n_samples, B]."""
        B = states.shape[0]
        device = states.device

        # 扩展 batch
        s_exp = states.unsqueeze(0).expand(n_samples, B, -1).reshape(n_samples * B, -1)
        a_exp = actions.unsqueeze(0).expand(n_samples, B, -1).reshape(n_samples * B, -1)
        x = torch.randn(n_samples * B, 1, device=device)

        for t_idx in reversed(range(self.n_timesteps)):
            t = torch.full((n_samples * B,), t_idx, device=device, dtype=torch.long)
            epsilon = self.denoiser(x, t, s_exp, a_exp)

            alpha = self.alphas[t_idx]
            alpha_bar = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]

            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar) * epsilon) / sqrt_alpha_bar
            x0_pred = torch.clamp(x0_pred, -5.0, 5.0)  # clamp 防异常值

            if t_idx > 0:
                alpha_bar_prev = self.alphas_cumprod[t_idx - 1]
                coef1 = torch.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar)
                coef2 = torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                mean = coef1 * x0_pred + coef2 * x
                beta_tilde = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                sigma = torch.sqrt(beta_tilde)
                x = mean + sigma * torch.randn_like(x)
            else:
                x = x0_pred

        return x.reshape(n_samples, B) * self.return_scale

    @torch.no_grad()
    def expected_value(self, states, actions, n_samples=200):
        """E[R|s,a] 估计 (用于 Actor 更新)."""
        samples = self.sample(states, actions, n_samples=n_samples)
        return samples.mean(dim=0)  # [B]


# ============================================================================
# MLP Actor (策略网络)
# ============================================================================

class MLPActor(nn.Module):
    """
    策略网络 π(a|s) — 输出离散动作 logits。
    支持多 agent 独立输出。
    """

    def __init__(self, state_dim: int, n_agents: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * n_actions),  # 每个 agent 独立输出
        )

    def forward(self, states):
        """states: [B, state_dim] → [B, n_agents, n_actions]"""
        B = states.shape[0]
        logits = self.net(states)  # [B, n_agents * n_actions]
        logits = logits.reshape(B, self.n_agents, self.n_actions)  # [B, A, n_act]
        return logits

    def sample_action(self, states, deterministic=False):
        """
        从策略中采样动作。
        states: [B, state_dim] → actions: [B, n_agents]
        """
        logits = self.forward(states)  # [B, A, n_act]
        probs = F.softmax(logits, dim=-1)  # [B, A, n_act]

        if deterministic:
            actions = probs.argmax(dim=-1)  # [B, A]
        else:
            # Gumbel-Softmax 采样 (可导) 或直接 categorical 采样
            actions = torch.multinomial(
                probs.reshape(-1, self.n_actions), 1
            ).reshape(-1, self.n_agents)  # [B, A]

        return actions, probs
