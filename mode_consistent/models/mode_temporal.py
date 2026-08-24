"""以 agent-specific 离散团队角色为原生上下文的 TemporalUnet。"""

from typing import Optional, Tuple

import einops
import torch
from torch import nn

from diffuser.models.temporal import TemporalUnet


class ModeTemporalUnet(nn.Module):
    """共享参数、无跨 agent attention 的 mode 条件时序 U-Net。

    输入仍使用 MADiff 的 ``[batch, time, agent, transition]`` 契约，但每个
    agent slice 在送入基础 :class:`TemporalUnet` 前独立展平。唯一的团队级
    耦合来自共享 mode ID；codebook 中每个 ID 含有不同的 agent role 向量。
    """

    agent_share_parameters = True

    def __init__(
        self,
        n_agents: int,
        horizon: int,
        transition_dim: int,
        history_horizon: int = 0,
        n_modes: int = 6,
        mode_dim: int = 32,
        agent_embedding_dim: int = 8,
        dim: int = 128,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        mode_condition_dropout: float = 0.1,
        kernel_size: int = 5,
        max_path_length: int = 100,
        freeze_mode_codebook: bool = True,
        residual_attn: bool = False,
        use_temporal_attention: bool = False,
        device=None,
    ):
        super().__init__()
        del device
        if residual_attn or use_temporal_attention:
            raise ValueError("ModeTemporalUnet does not support agent attention")
        if n_agents < 1:
            raise ValueError("n_agents must be positive")
        if n_modes < 2:
            raise ValueError("n_modes must be at least two")
        if not 0.0 <= mode_condition_dropout < 1.0:
            raise ValueError("mode_condition_dropout must be in [0, 1)")

        self.n_agents = n_agents
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.history_horizon = history_horizon
        self.n_modes = n_modes
        self.mode_dim = mode_dim
        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition
        self.mode_condition_dropout = mode_condition_dropout

        self.mode_codebook = nn.Embedding(n_modes, n_agents * mode_dim)
        nn.init.uniform_(self.mode_codebook.weight, -1.0 / n_modes, 1.0 / n_modes)
        self.mode_codebook.weight.requires_grad_(not freeze_mode_codebook)
        self.agent_embedding = nn.Embedding(n_agents, agent_embedding_dim)

        downsample_factor = 2 ** max(len(dim_mults) - 1, 0)
        self.padded_horizon = (
            (horizon + downsample_factor - 1) // downsample_factor
        ) * downsample_factor
        self.net = TemporalUnet(
            horizon=self.padded_horizon,
            transition_dim=transition_dim,
            history_horizon=history_horizon,
            dim=dim,
            dim_mults=dim_mults,
            returns_condition=returns_condition,
            env_ts_condition=env_ts_condition,
            condition_dropout=condition_dropout,
            context_dim=mode_dim + agent_embedding_dim,
            context_dropout=0.0,
            kernel_size=kernel_size,
            max_path_length=max_path_length,
        )

    @property
    def role_codebook(self) -> torch.Tensor:
        return self.mode_codebook.weight.reshape(
            self.n_modes, self.n_agents, self.mode_dim
        )

    def set_mode_codebook(self, weight: torch.Tensor, freeze: bool = True) -> None:
        if weight.shape == (self.n_modes, self.n_agents, self.mode_dim):
            weight = weight.reshape(self.n_modes, -1)
        if weight.shape != self.mode_codebook.weight.shape:
            raise ValueError(
                "mode codebook must have shape [mode, agent, mode_dim] or "
                "[mode, agent * mode_dim]"
            )
        with torch.no_grad():
            self.mode_codebook.weight.copy_(weight)
        self.mode_codebook.weight.requires_grad_(not freeze)

    def mode_context(
        self,
        mode_codes: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        use_mode_dropout: bool = True,
        force_mode_dropout: bool = False,
    ) -> torch.Tensor:
        roles = torch.zeros(
            batch_size,
            self.n_agents,
            self.mode_dim,
            device=device,
            dtype=dtype,
        )
        if mode_codes is not None:
            mode_codes = mode_codes.to(device=device, dtype=torch.long)
            if mode_codes.ndim == 1:
                if mode_codes.shape != (batch_size,):
                    raise ValueError("shared mode_codes must have shape [batch]")
                mode_codes = mode_codes[:, None].expand(-1, self.n_agents)
            if mode_codes.shape != (batch_size, self.n_agents):
                raise ValueError("mode_codes must have shape [batch] or [batch, agent]")
            if torch.any(mode_codes < -1):
                raise ValueError("unknown mode code must be represented by -1")
            if torch.any(mode_codes >= self.n_modes):
                raise ValueError("mode code is outside the codebook")
            known = mode_codes >= 0
            safe_codes = mode_codes.clamp_min(0)
            # 每个 agent 可以从自己的 local prior 得到不同 code。查询 code k
            # 后只取当前 agent slot 对应的 role，避免看到其他 agent 的向量。
            all_roles = self.mode_codebook(safe_codes).reshape(
                batch_size,
                self.n_agents,
                self.n_agents,
                self.mode_dim,
            )
            agent_ids = torch.arange(self.n_agents, device=device)
            roles = all_roles[:, agent_ids, agent_ids]
            roles = roles * known[..., None].to(dtype=roles.dtype)

        if force_mode_dropout:
            roles = torch.zeros_like(roles)
        elif use_mode_dropout and self.training and self.mode_condition_dropout > 0.0:
            # 同一 episode 的所有 agent 共享一次 mode CFG 丢弃决定。
            keep = (
                torch.rand(batch_size, 1, 1, device=device)
                >= self.mode_condition_dropout
            )
            roles = roles * keep.to(dtype=roles.dtype)

        agent_ids = torch.arange(self.n_agents, device=device)
        identities = self.agent_embedding(agent_ids)[None].expand(batch_size, -1, -1)
        return torch.cat([roles, identities.to(dtype=dtype)], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        env_timestep: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        mode_codes: Optional[torch.Tensor] = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        use_mode_dropout: bool = True,
        force_mode_dropout: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del attention_masks, kwargs
        if x.ndim != 4 or x.shape[2:] != (
            self.n_agents,
            self.transition_dim,
        ):
            raise ValueError("x must have shape [batch, time, agent, transition_dim]")
        if time.shape != (x.shape[0],):
            raise ValueError("time must have shape [batch]")
        original_horizon = x.shape[1]
        if original_horizon > self.padded_horizon:
            raise ValueError("input horizon exceeds configured padded horizon")

        context = self.mode_context(
            mode_codes=mode_codes,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            use_mode_dropout=use_mode_dropout,
            force_mode_dropout=force_mode_dropout,
        )
        if original_horizon < self.padded_horizon:
            padding = x.new_zeros(
                x.shape[0],
                self.padded_horizon - original_horizon,
                self.n_agents,
                self.transition_dim,
            )
            x = torch.cat([x, padding], dim=1)

        batch = x.shape[0]
        x = einops.rearrange(x, "b t a f -> (b a) t f")
        flat_time = time.repeat_interleave(self.n_agents)
        flat_returns = self._flatten_returns(returns, batch)
        flat_env_timestep = (
            env_timestep.repeat_interleave(self.n_agents, dim=0)
            if env_timestep is not None
            else None
        )
        prediction = self.net(
            x,
            time=flat_time,
            returns=flat_returns,
            env_timestep=flat_env_timestep,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
            context=context.reshape(batch * self.n_agents, -1),
            use_context_dropout=False,
            force_context_dropout=False,
        )
        prediction = einops.rearrange(
            prediction,
            "(b a) t f -> b t a f",
            b=batch,
            a=self.n_agents,
        )
        return prediction[:, :original_horizon]

    def _flatten_returns(
        self, returns: Optional[torch.Tensor], batch_size: int
    ) -> Optional[torch.Tensor]:
        if returns is None:
            return None
        if returns.ndim == 2 and returns.shape == (batch_size, 1):
            returns = returns[:, :, None].expand(-1, -1, self.n_agents)
        elif returns.ndim == 2 and returns.shape == (batch_size, self.n_agents):
            returns = returns[:, None]
        if returns.shape != (batch_size, 1, self.n_agents):
            raise ValueError(
                "returns must have shape [batch, 1, agent], [batch, agent], "
                "or [batch, 1]"
            )
        return returns.permute(0, 2, 1).reshape(batch_size * self.n_agents, 1)
