"""用于时序 U-Net 单条跳连的非注意力智能体交互模块."""

import torch
from torch import nn


class IdentityInteraction(nn.Module):
    def forward(self, x, time_embed=None, node_mask=None):
        return x


class _NeighbourInteraction(nn.Module):
    def __init__(self, n_channels, embed_dim, hidden_dim, pairwise, residual):
        super().__init__()
        self.pairwise = pairwise
        self.residual = residual
        self.hidden_dim = hidden_dim
        self.phi = nn.Sequential(
            nn.Linear((2 if pairwise else 1) * n_channels + embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(n_channels + hidden_dim + embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, n_channels),
        )
        if residual:
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, time_embed, node_mask=None):
        batch, n_agents, _, horizon = x.shape
        if n_agents == 1:
            return x
        features = x.permute(0, 1, 3, 2)
        conditions = time_embed[:, :, None].expand(-1, -1, horizon, -1)
        if node_mask is None:
            valid = torch.ones(batch, n_agents, horizon, dtype=torch.bool, device=x.device)
        else:
            valid = node_mask.to(device=x.device, dtype=torch.bool)
            if valid.ndim == 2:
                valid = valid[:, :, None].expand(-1, -1, horizon)
        counts = valid.sum(dim=1, keepdim=True) - valid.long()

        if self.pairwise:
            neighbours = valid[:, None] & ~torch.eye(
                n_agents, dtype=torch.bool, device=x.device
            )[None, :, :, None]
            receivers = features[:, :, None].expand(-1, -1, n_agents, -1, -1)
            senders = features[:, None].expand(-1, n_agents, -1, -1, -1)
            receiver_conditions = conditions[:, :, None].expand(-1, -1, n_agents, -1, -1)
            messages = self.phi(torch.cat([receivers, senders, receiver_conditions], dim=-1))
            aggregate = messages.masked_fill(~neighbours[..., None], 0).sum(dim=2)
        else:
            messages = self.phi(torch.cat([features, conditions], dim=-1))
            messages = messages.masked_fill(~valid[..., None], 0)
            aggregate = messages.sum(dim=1, keepdim=True) - messages
        aggregate = aggregate / counts.clamp(min=1)[..., None]
        update = self.rho(torch.cat([features, aggregate, conditions], dim=-1))
        if self.residual:
            update = features + self.gamma.to(x.dtype) * update
        active = valid & (counts > 0)
        return torch.where(active[..., None], update, features).permute(0, 1, 3, 2)


class PoolInteraction(_NeighbourInteraction):
    def __init__(self, n_channels, embed_dim, hidden_dim, residual=True):
        super().__init__(n_channels, embed_dim, hidden_dim, False, residual)


class MPNNInteraction(_NeighbourInteraction):
    def __init__(self, n_channels, embed_dim, hidden_dim, residual=True):
        super().__init__(n_channels, embed_dim, hidden_dim, True, residual)


def matching_hidden_dim(kind, n_channels, embed_dim, target_parameters):
    """按被替换注意力模块的参数量,选择最接近的隐藏层宽度."""
    phi_channels = (2 if kind == "mpnn" else 1) * n_channels

    def difference(width):
        parameters = (
            2 * width * width
            + (phi_channels + 2 * n_channels + 2 * embed_dim + 3) * width
            + n_channels + 1
        )
        return abs(parameters - target_parameters)

    return min(range(16, 513, 16), key=difference)
