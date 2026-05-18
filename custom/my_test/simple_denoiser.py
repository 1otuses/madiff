"""
简化版扩散去噪器 (无注意力，仅 Conv1d + MLP)
==============================================
用于状态序列的去噪，去掉 SelfAttention / TemporalSelfAttention。
支持 classifier-free guidance 条件。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Conv1dBlock(nn.Module):
    """1D 卷积块 (替代注意力)"""
    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation),
            nn.Mish(),
        )

    def forward(self, x):
        # x: [B*T, D] 或 [B*T*A, D]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B*T*A, D, 1]
            x = self.block(x)
            return x.squeeze(-1)
        elif x.dim() == 3:
            return self.block(x)
        else:
            raise ValueError(f"Unexpected input dim: {x.dim()}")


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


class SimpleTemporalDenoiser(nn.Module):
    """
    简化版时序去噪器 (无注意力)。
    架构:
      - 时间嵌入 (Sinusoidal)
      - Returns 条件嵌入 (可选)
      - 堆叠的 Conv1d + Residual 块
      - 共享参数在所有 agent 间

    输入:
      x:     [B, T, A, obs_dim]  带噪状态序列
      t:     [B]  扩散时间步
      returns: [B]  回报条件 (可选, CFG 用)
      risk_grad: [B, T, A, obs_dim]  风险梯度 (可选, CFG 引导用)

    输出:
      epsilon: [B, T, A, obs_dim]  预测噪声
    """

    def __init__(
        self,
        horizon: int,
        n_agents: int,
        obs_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.0,
        use_returns_condition: bool = True,
        use_risk_guide: bool = False,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.use_returns_condition = use_returns_condition
        self.use_risk_guide = use_risk_guide

        # 输入投影：obs_dim → hidden_dim
        self.input_proj = nn.Linear(obs_dim, hidden_dim)

        # 时间嵌入
        self.time_emb = SinusoidalPosEmb(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Returns 条件嵌入
        if use_returns_condition:
            self.returns_proj = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # 风险梯度引导嵌入
        if use_risk_guide:
            self.risk_proj = nn.Linear(obs_dim, hidden_dim)

        # 逐时间步的卷积块 (替代注意力)
        self.conv_blocks = nn.ModuleList([
            Conv1dBlock(hidden_dim, kernel_size=3, dilation=1)
            for _ in range(n_blocks)
        ])

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(n_blocks)
        ])

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, obs_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        risk_grad: Optional[torch.Tensor] = None,
        force_dropout: bool = False,
    ):
        """
        x:     [B, T, A, obs_dim]
        t:     [B]
        returns: [B] 或 [B, 1]
        risk_grad: [B, T, A, obs_dim]
        force_dropout: 若 True, 忽略 returns 和 risk_grad (CFG 无条件分支)
        """
        B, T, A, obs_dim = x.shape

        # 输入投影
        h = self.input_proj(x)  # [B, T, A, hidden_dim]

        # 时间嵌入
        t_emb = self.time_emb(t)  # [B, hidden_dim]
        t_emb = self.time_proj(t_emb)  # [B, hidden_dim]
        t_emb = t_emb.unsqueeze(1).unsqueeze(2).expand(-1, T, A, -1)  # [B, T, A, hidden_dim]
        h = h + t_emb

        # Returns 条件 (CFG)
        if self.use_returns_condition and returns is not None and not force_dropout:
            if returns.ndim == 1:
                returns = returns.unsqueeze(-1)
            r_emb = self.returns_proj(returns)  # [B, hidden_dim]
            r_emb = r_emb.unsqueeze(1).unsqueeze(2).expand(-1, T, A, -1)
            h = h + r_emb

        # 风险梯度引导
        if self.use_risk_guide and risk_grad is not None and not force_dropout:
            rk_emb = self.risk_proj(risk_grad)  # [B, T, A, hidden_dim]
            h = h + rk_emb

        # 重塑为 [B*A, T, hidden_dim] 进行时间维卷积
        h = h.permute(0, 2, 1, 3).reshape(B * A, T, -1)  # [B*A, T, hidden_dim]
        h = h.permute(0, 2, 1)  # [B*A, hidden_dim, T]  (Conv1d 输入)

        for conv_block, res_block in zip(self.conv_blocks, self.res_blocks):
            h = conv_block(h)
            # ResBlock 需要 [*, hidden_dim] 形状
            h = h.permute(0, 2, 1)  # [B*A, T, hidden_dim]
            h_flat = h.reshape(B * A * T, -1)
            h_flat = res_block(h_flat)
            h = h_flat.reshape(B * A, T, -1).permute(0, 2, 1)  # [B*A, hidden_dim, T]

        # 恢复形状
        h = h.permute(0, 2, 1)  # [B*A, T, hidden_dim]
        h = h.reshape(B, A, T, -1).permute(0, 2, 1, 3)  # [B, T, A, hidden_dim]

        # 输出投影
        epsilon = self.output_proj(h)  # [B, T, A, obs_dim]
        return epsilon
