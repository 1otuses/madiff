"""表征学习后使用的 mode 条件集中式价值头。"""

import torch
from torch import nn


class ModeValueModel(nn.Module):
    """根据集中状态和匿名 mode 估计数据集回报。"""

    def __init__(
        self,
        n_agents: int,
        observation_dim: int,
        n_modes: int,
        hidden_dim: int = 128,
        mode_embedding_dim: int = 16,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.n_modes = n_modes
        self.register_buffer("return_mean", torch.tensor(0.0))
        self.register_buffer("return_std", torch.tensor(1.0))
        self.mode_embedding = nn.Embedding(n_modes, mode_embedding_dim)
        self.value = nn.Sequential(
            nn.Linear(n_agents * observation_dim + mode_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, initial_observations: torch.Tensor, mode_codes: torch.Tensor
    ) -> torch.Tensor:
        if initial_observations.shape[1:] != (
            self.n_agents,
            self.observation_dim,
        ):
            raise ValueError(
                "initial_observations must have shape [batch, agent, observation]"
            )
        if mode_codes.shape != (initial_observations.shape[0],):
            raise ValueError("mode_codes must have shape [batch]")
        features = torch.cat(
            [initial_observations.flatten(1), self.mode_embedding(mode_codes)], dim=-1
        )
        return self.value(features).squeeze(-1)

    @torch.no_grad()
    def all_mode_values(self, initial_observations: torch.Tensor) -> torch.Tensor:
        batch = initial_observations.shape[0]
        observations = initial_observations[:, None].expand(-1, self.n_modes, -1, -1)
        modes = torch.arange(self.n_modes, device=initial_observations.device)
        modes = modes[None].expand(batch, -1)
        return self(observations.flatten(0, 1), modes.flatten()).reshape(
            batch, self.n_modes
        )
