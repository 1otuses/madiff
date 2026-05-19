import torch
from torch import nn


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 4):
        super().__init__()
        n_layers = max(2, int(n_layers))

        layers = []
        in_dim = obs_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 3:
            bsz, n_agents, obs_dim = obs.shape
            x = obs.reshape(bsz * n_agents, obs_dim)  # [B, A, D] -> [B*A, D]
            out = self.net(x)  # [B*A, action_dim]
            out = torch.tanh(out)
            out = out.reshape(bsz, n_agents, -1)  # [B*A, action_dim] -> [B, A, action_dim]
            return out
        if obs.ndim == 2:
            x = obs  # [B, D]
            out = self.net(x)  # [B, action_dim]
            out = torch.tanh(out)
            return out
        raise ValueError(f"Unexpected obs shape: {obs.shape}")

    def loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pred = self.forward(obs)
        loss = (pred - actions).pow(2).mean()
        return loss
