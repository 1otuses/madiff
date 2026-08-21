"""借鉴 VO-MASD、直接处理原始联合轨迹片段的离散 mode 教师。"""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def build_action_decoder(input_dim: int, hidden_dim: int, action_dim: int) -> nn.Module:
    """创建 P2 主模型与对照组共用的动作重建头。"""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, action_dim),
    )


def masked_action_mse(
    predicted_actions: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """只在有效时间步上计算每个动作维度的均方误差。"""

    valid = mask[..., None, None]
    dimensions = actions.shape[2] * actions.shape[3]
    return ((predicted_actions - actions).square() * valid).sum() / (
        valid.sum() * dimensions
    ).clamp_min(1.0)


class CentralModeVQVAE(nn.Module):
    """将一个无标签联合轨迹窗口编码为一个离散团队 mode。
    编码器只接收原始观测、动作和有效步掩码；接口刻意排除所有审计标签。
    """

    def __init__(
        self,
        n_agents: int,
        observation_dim: int,
        action_dim: int,
        n_modes: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        agent_embedding_dim: int = 8,
        vq_beta: float = 0.001,
        vq_loss_design: str = "vomasd",
    ):
        super().__init__()
        if n_agents < 2:
            raise ValueError("n_agents must be at least two")
        if n_modes < 2:
            raise ValueError("n_modes must be at least two")
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.n_modes = n_modes
        self.vq_beta = vq_beta
        self.vq_loss_design = vq_loss_design
        if vq_beta < 0.0:
            raise ValueError("vq_beta 不能为负数")
        if vq_loss_design not in {"vomasd", "standard"}:
            raise ValueError("vq_loss_design 必须为 vomasd 或 standard")

        # 所有 agent 共享轨迹编码器；量化前再聚合为一个团队表示。
        self.agent_encoder = nn.GRU(
            observation_dim + action_dim, hidden_dim, batch_first=True
        )
        self.agent_embedding = nn.Embedding(n_agents, agent_embedding_dim)
        self.team_encoder = nn.Sequential(
            nn.Linear(n_agents * (hidden_dim + agent_embedding_dim), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.codebook = nn.Embedding(n_modes, latent_dim)
        self.action_decoder = build_action_decoder(
            observation_dim + latent_dim + agent_embedding_dim,
            hidden_dim,
            action_dim,
        )

    def _encode(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ):
        self._validate_inputs(observations, actions, mask)
        # 每个 agent 的局部轨迹独立经过同一个 GRU，尚未进行离散量化。
        batch, horizon = observations.shape[:2]
        sequence = torch.cat([observations, actions], dim=-1)
        sequence = sequence.permute(0, 2, 1, 3).reshape(
            batch * self.n_agents, horizon, -1
        )
        lengths = mask.sum(dim=1).long()
        if torch.any(lengths < 1):
            raise ValueError("each episode must contain at least one valid step")
        lengths = lengths[:, None].expand(-1, self.n_agents).reshape(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.agent_encoder(packed)
        agent_features = hidden[-1].reshape(batch, self.n_agents, -1)
        agent_ids = torch.arange(self.n_agents, device=observations.device)
        identities = self.agent_embedding(agent_ids)[None].expand(batch, -1, -1)
        # 固定 agent slot 的 identity 保留“谁执行了哪段轨迹”的团队协调语义。
        encoded = self.team_encoder(
            torch.cat([agent_features, identities], dim=-1).flatten(1)
        )
        distances = (
            encoded.square().sum(dim=1, keepdim=True)
            + self.codebook.weight.square().sum(dim=1)
            - 2.0 * encoded @ self.codebook.weight.t()
        )
        codes = distances.argmin(dim=1)
        quantized = F.embedding(codes, self.codebook.weight)
        return encoded, quantized, distances, codes

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        encoded, quantized, distances, codes = self._encode(observations, actions, mask)
        # 直通估计器让重建梯度流向编码器，同时保持前向值来自离散码本。
        straight_through = encoded + (quantized - encoded).detach()
        predicted_actions = self._decode_embedding(observations, straight_through)

        reconstruction_loss = masked_action_mse(predicted_actions, actions, mask)
        # Decoder 输出固定方差高斯策略的动作均值；此时 MSE 与 NLL 只差常数和尺度。
        codebook_loss = F.mse_loss(quantized, encoded.detach())
        commitment_loss = F.mse_loss(encoded, quantized.detach())
        code_probabilities = F.softmax(-distances, dim=-1)
        if self.vq_loss_design == "vomasd":
            # VO-MASD 源码的权重方向：encoder 承诺项权重为 1，
            # codebook 更新项权重为 beta。
            loss = reconstruction_loss + commitment_loss + self.vq_beta * codebook_loss
        else:
            # 仅用于读取旧 H=5 checkpoint，新实验不应使用此分支。
            loss = reconstruction_loss + codebook_loss + self.vq_beta * commitment_loss
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "codes": codes,
            "code_probabilities": code_probabilities,
            "predicted_actions": predicted_actions,
        }

    def decode_actions(
        self, observations: torch.Tensor, codes: torch.Tensor
    ) -> torch.Tensor:
        """使用指定离散 code 解码动作，供冻结后的条件有效性审计使用。"""

        if codes.shape != (observations.shape[0],):
            raise ValueError("codes 必须为 [batch]")
        return self._decode_embedding(observations, self.codebook(codes.long()))

    def _decode_embedding(
        self, observations: torch.Tensor, mode_vectors: torch.Tensor
    ) -> torch.Tensor:
        batch, horizon = observations.shape[:2]
        agent_ids = torch.arange(self.n_agents, device=observations.device)
        agent_embedding = self.agent_embedding(agent_ids)[None, None].expand(
            batch, horizon, -1, -1
        )
        mode_embedding = mode_vectors[:, None, None].expand(
            -1, horizon, self.n_agents, -1
        )
        return self.action_decoder(
            torch.cat([observations, mode_embedding, agent_embedding], dim=-1)
        )

    @torch.no_grad()
    def predict_codes(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._encode(observations, actions, mask)[-1]

    def _validate_inputs(self, observations, actions, mask) -> None:
        expected = observations.shape[:3]
        if observations.ndim != 4 or expected[2] != self.n_agents:
            raise ValueError("observations 必须为 [batch, time, agent, obs]")
        if observations.shape[-1] != self.observation_dim:
            raise ValueError("observation 维度与模型配置不一致")
        if actions.shape != (*expected, self.action_dim):
            raise ValueError("actions 必须与 observations 和 action_dim 匹配")
        if mask.shape != observations.shape[:2]:
            raise ValueError("mask 必须为 [batch, time]")


class NoCodeActionModel(nn.Module):
    """不使用 mode 的动作重建对照，用于检验离散 code 是否有用。"""

    def __init__(
        self,
        n_agents: int,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        agent_embedding_dim: int = 8,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.agent_embedding = nn.Embedding(n_agents, agent_embedding_dim)
        self.action_decoder = build_action_decoder(
            observation_dim + agent_embedding_dim, hidden_dim, action_dim
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, horizon = observations.shape[:2]
        agent_ids = torch.arange(self.n_agents, device=observations.device)
        agent_embedding = self.agent_embedding(agent_ids)[None, None].expand(
            batch, horizon, -1, -1
        )
        predicted_actions = self.action_decoder(
            torch.cat([observations, agent_embedding], dim=-1)
        )
        reconstruction_loss = masked_action_mse(predicted_actions, actions, mask)
        return {
            "loss": reconstruction_loss,
            "reconstruction_loss": reconstruction_loss,
            "predicted_actions": predicted_actions,
        }


class OracleModeActionModel(nn.Module):
    """使用审计 mode 的特权上界；不属于可部署算法。"""

    def __init__(
        self,
        n_agents: int,
        observation_dim: int,
        action_dim: int,
        n_modes: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        agent_embedding_dim: int = 8,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.mode_embedding = nn.Embedding(n_modes, latent_dim)
        self.agent_embedding = nn.Embedding(n_agents, agent_embedding_dim)
        self.action_decoder = build_action_decoder(
            observation_dim + latent_dim + agent_embedding_dim,
            hidden_dim,
            action_dim,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
        modes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, horizon = observations.shape[:2]
        agent_ids = torch.arange(self.n_agents, device=observations.device)
        agent_embedding = self.agent_embedding(agent_ids)[None, None].expand(
            batch, horizon, -1, -1
        )
        mode_embedding = self.mode_embedding(modes)[:, None, None].expand(
            -1, horizon, self.n_agents, -1
        )
        predicted_actions = self.action_decoder(
            torch.cat([observations, mode_embedding, agent_embedding], dim=-1)
        )
        reconstruction_loss = masked_action_mse(predicted_actions, actions, mask)
        return {
            "loss": reconstruction_loss,
            "reconstruction_loss": reconstruction_loss,
            "predicted_actions": predicted_actions,
        }
