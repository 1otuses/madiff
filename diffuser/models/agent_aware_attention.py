"""将智能体与轨迹时间位置展开为 token,进行 AgentFormer 式时空注意力交互."""

import math
from typing import Optional

import torch
from torch import nn


def sinusoidal_pos_emb(length, dim, device, dtype=torch.float32, scale=10.0):
    """生成窗口内轨迹位置的正弦编码,所有智能体共用同一套位置编码."""
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    position = torch.arange(length, device=device, dtype=compute_dtype)
    position = position / max(length - 1, 1) * scale
    frequency = torch.arange(0, dim, 2, device=device, dtype=compute_dtype)
    frequency = torch.exp(-math.log(10000.0) * frequency / dim)
    angles = position[:, None] * frequency[None, :]
    encoding = torch.empty(length, dim, device=device, dtype=compute_dtype)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles[:, : dim // 2].cos()
    return encoding.to(dtype=dtype)


class AgentAwareSpatioTemporalAttention(nn.Module):
    """同一智能体与不同智能体分别投影 Q/K,选择分数后统一进行 softmax.

    ``history_horizon`` 表示当前模块时间尺度下的历史 token 数量.
    ``film`` 表示方案中的加性条件注入模式.
    ``adaln`` 使用自身零初始化的残差门控,避免与另一个零初始化 gamma 相乘.
    """

    def __init__(
        self,
        n_channels: int,
        n_agents: int,
        history_horizon: int = 0,
        nheads: int = 4,
        qk_ratio: int = 16,
        cond_dim: Optional[int] = None,
        cond_mode: str = "film",
        use_time_enc: bool = True,
        use_segment_emb: bool = True,
        use_ffn: bool = False,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        if nheads <= 0 or n_channels % nheads or qk_ratio <= 0:
            raise ValueError(
                "Channels must be divisible by positive nheads; qk_ratio must be positive."
            )
        if cond_mode not in ("film", "adaln", "none"):
            raise ValueError(f"Unknown condition mode: {cond_mode}")

        self.n_channels = n_channels
        self.n_agents = n_agents
        self.nheads = nheads
        self.history_horizon = history_horizon
        self.use_time_enc = use_time_enc
        self.use_segment_emb = use_segment_emb and history_horizon > 0
        self.use_ffn = use_ffn
        self.residual = residual
        self.cond_mode = cond_mode if cond_dim is not None else "none"
        self.dqk = max(n_channels // qk_ratio, 16)
        self.dv = n_channels // nheads

        self.q_self = nn.Linear(n_channels, nheads * self.dqk)
        self.k_self = nn.Linear(n_channels, nheads * self.dqk)
        self.q_other = nn.Linear(n_channels, nheads * self.dqk)
        self.k_other = nn.Linear(n_channels, nheads * self.dqk)
        self.v = nn.Linear(n_channels, n_channels)
        self.ln_in = nn.LayerNorm(
            n_channels, elementwise_affine=self.cond_mode != "adaln"
        )
        self.dropout = nn.Dropout(dropout)

        if self.use_segment_emb:
            self.seg_emb = nn.Embedding(2, n_channels)
        if self.cond_mode == "film":
            self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, n_channels))
        elif self.cond_mode == "adaln":
            self.modulation_parts = 3 if residual else 2
            modulation_dim = (
                self.modulation_parts * (2 if use_ffn else 1) * n_channels
            )
            self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, modulation_dim))
            nn.init.zeros_(self.adaln[-1].weight)
            nn.init.zeros_(self.adaln[-1].bias)

        if residual and self.cond_mode != "adaln":
            self.gamma = nn.Parameter(torch.zeros(1))
        if use_ffn:
            self.ln_ffn = nn.LayerNorm(
                n_channels, elementwise_affine=self.cond_mode != "adaln"
            )
            self.ffn = nn.Sequential(
                nn.Linear(n_channels, 4 * n_channels),
                nn.GELU(),
                nn.Linear(4 * n_channels, n_channels),
            )
            if residual and self.cond_mode != "adaln":
                self.gamma_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, x, cond=None, attention_masks=None):
        """输入特征 x 为 [B,A,C,T],条件 cond 为 [B,A,D],有效位置掩码为 [B,A,T]."""
        batch, agents, channels, length = x.shape
        if agents != self.n_agents or channels != self.n_channels:
            raise ValueError(
                f"Expected A={self.n_agents}, C={self.n_channels}; got {tuple(x.shape)}"
            )
        if self.cond_mode != "none" and cond is None:
            raise ValueError(f"cond is required for condition mode {self.cond_mode}")
        if attention_masks is None:
            valid = torch.ones(batch, agents, length, device=x.device, dtype=torch.bool)
        else:
            if attention_masks.shape != (batch, agents, length):
                raise ValueError(
                    "attention_masks must have shape [B,A,T] at the current scale"
                )
            valid = attention_masks.to(device=x.device, dtype=torch.bool)

        tokens = x.permute(0, 1, 3, 2)
        h = tokens
        if self.use_time_enc:
            h = h + sinusoidal_pos_emb(length, channels, x.device, x.dtype)[None, None]
        if self.use_segment_emb:
            segment = (
                torch.arange(length, device=x.device) >= self.history_horizon
            ).long()
            h = h + self.seg_emb(segment)[None, None]

        modulation = None
        if self.cond_mode == "film":
            h = h + self.cond_mlp(cond)[:, :, None, :]
        elif self.cond_mode == "adaln":
            modulation = self.adaln(cond)[:, :, None, :].split(channels, dim=-1)
        h = self.ln_in(h)
        if modulation is not None:
            h = h * (1 + modulation[0]) + modulation[1]

        def heads(projected, head_dim):
            return projected.reshape(
                batch, agents * length, self.nheads, head_dim
            ).transpose(1, 2)

        qs = heads(self.q_self(h), self.dqk)
        ks = heads(self.k_self(h), self.dqk)
        qo = heads(self.q_other(h), self.dqk)
        ko = heads(self.k_other(h), self.dqk)
        value = heads(self.v(h), self.dv)
        self_logits = (qs @ ks.transpose(-1, -2)) / math.sqrt(self.dqk)
        other_logits = (qo @ ko.transpose(-1, -2)) / math.sqrt(self.dqk)
        same_agent = torch.eye(agents, device=x.device, dtype=torch.bool)
        same_agent = same_agent[:, None, :, None].expand(
            agents, length, agents, length
        )
        same_agent = same_agent.reshape(agents * length, agents * length)
        logits = torch.where(same_agent, self_logits, other_logits)
        key_valid = valid.reshape(batch, 1, 1, agents * length)
        # 用有限最小值填充,避免全掩码时 softmax 出现 NaN,再将无效位置的权重置零.
        logits = logits.masked_fill(~key_valid, torch.finfo(logits.dtype).min)
        attention = logits.softmax(dim=-1) * key_valid
        attention = self.dropout(attention)
        update = (attention @ value).transpose(1, 2).reshape(
            batch, agents, length, channels
        )
        update = update * valid[..., None]
        if self.residual:
            gate = modulation[2] if modulation is not None else self.gamma
            output = tokens + gate * update
        else:
            output = update

        if self.use_ffn:
            h = self.ln_ffn(output)
            if modulation is not None:
                offset = self.modulation_parts
                h = h * (1 + modulation[offset]) + modulation[offset + 1]
            update = self.ffn(h) * valid[..., None]
            if self.residual:
                gate = (
                    modulation[offset + 2]
                    if modulation is not None
                    else self.gamma_ffn
                )
                update = gate * update
            output = output + update
        return output.permute(0, 1, 3, 2)
