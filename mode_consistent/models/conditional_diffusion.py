"""为 MADiff 共享时序 U-Net 注入离散 mode 条件的适配器。"""

from typing import Optional

import torch
from torch import nn


class ModeConditionedDenoiser(nn.Module):
    """在 MADiff 去噪器之前应用离散 mode 的 FiLM 调制。

    一维 ``mode_codes`` 表示 focal agent 局部推断出的 mode，只在该 agent
    自己生成的联合轨迹假设内部广播。诊断时也接受 ``[batch, agents]`` 形状；
    ``-1`` 表示未知或无条件分支。
    """

    def __init__(
        self,
        denoiser: nn.Module,
        transition_dim: int,
        n_agents: int,
        n_modes: int,
        mode_embedding_dim: int = 32,
        condition_dropout: float = 0.1,
    ):
        super().__init__()
        if not 0.0 <= condition_dropout < 1.0:
            raise ValueError("condition_dropout must be in [0, 1)")
        self.denoiser = denoiser
        self.transition_dim = transition_dim
        self.n_agents = n_agents
        self.n_modes = n_modes
        self.condition_dropout = condition_dropout
        self.mode_embedding = nn.Embedding(n_modes, mode_embedding_dim)
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mode_embedding_dim, 2 * transition_dim),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        mode_codes: Optional[torch.Tensor] = None,
        guidance_weight: Optional[float] = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if guidance_weight is not None:
            conditional = self._predict(
                x, time, mode_codes, use_dropout=False, force_dropout=False, **kwargs
            )
            unconditional = self._predict(
                x, time, mode_codes, use_dropout=False, force_dropout=True, **kwargs
            )
            return unconditional + guidance_weight * (conditional - unconditional)
        return self._predict(
            x,
            time,
            mode_codes,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
            **kwargs,
        )

    def _predict(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        mode_codes: Optional[torch.Tensor],
        use_dropout: bool,
        force_dropout: bool,
        **kwargs,
    ) -> torch.Tensor:
        mode_embedding = self._mode_embeddings(x, mode_codes)
        if force_dropout:
            mode_embedding = torch.zeros_like(mode_embedding)
        elif use_dropout and self.training and self.condition_dropout:
            # 每个样本共享一次条件丢弃决定，避免不同 agent 获得不一致的 CFG 条件。
            keep = (
                torch.rand(x.shape[0], 1, 1, device=x.device) >= self.condition_dropout
            )
            mode_embedding = mode_embedding * keep
        scale, shift = self.film(mode_embedding).chunk(2, dim=-1)
        conditioned = x * (1.0 + scale[:, None]) + shift[:, None]
        return self.denoiser(conditioned, time, **kwargs)

    def _mode_embeddings(
        self, x: torch.Tensor, mode_codes: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[2:] != (self.n_agents, self.transition_dim):
            raise ValueError(
                "x must have shape [batch, time, n_agents, transition_dim]"
            )
        if mode_codes is None:
            return torch.zeros(
                x.shape[0],
                self.n_agents,
                self.mode_embedding.embedding_dim,
                device=x.device,
                dtype=x.dtype,
            )
        if mode_codes.ndim == 1:
            mode_codes = mode_codes[:, None].expand(-1, self.n_agents)
        if mode_codes.shape != (x.shape[0], self.n_agents):
            raise ValueError("mode_codes must have shape [batch] or [batch, agents]")
        known = mode_codes >= 0
        if torch.any(mode_codes >= self.n_modes):
            raise ValueError("mode code is outside the codebook")
        safe_codes = mode_codes.clamp_min(0)
        embedding = self.mode_embedding(safe_codes)
        # 未知 code 虽临时映射到索引 0，但最终乘零，不泄漏 mode 0 的嵌入。
        return embedding * known[..., None]
