"""固定团队的离散 coordination-mode VQ-VAE。

训练期 posterior 可以读取完整联合轨迹；执行期 prior 只能读取每个 agent
自己的 observation 与过去动作。两个路径共享同一个离散 mode ID 语义。
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _masked_action_mse(
    predicted_actions: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask[..., None, None].to(dtype=actions.dtype)
    dimensions = actions.shape[2] * actions.shape[3]
    return ((predicted_actions - actions).square() * valid).sum() / (
        valid.sum() * dimensions
    ).clamp_min(1.0)


class LocalModePrior(nn.Module):
    """用严格局部历史预测中央 posterior 的匿名 mode ID。"""

    def __init__(
        self,
        n_agents: int,
        observation_dim: int,
        action_dim: int,
        n_modes: int,
        hidden_dim: int = 128,
        agent_embedding_dim: int = 8,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.n_modes = n_modes
        self.agent_embedding = nn.Embedding(n_agents, agent_embedding_dim)
        self.encoder = nn.GRU(
            observation_dim + action_dim + agent_embedding_dim,
            hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, n_modes)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(observations, actions, mask)
        batch, horizon = observations.shape[:2]

        # 当前动作是待决策结果，prior 只能看到前一时刻及更早的自身动作。
        previous_actions = torch.zeros_like(actions)
        previous_actions[:, 1:] = actions[:, :-1]
        agent_ids = torch.arange(self.n_agents, device=observations.device)
        identities = self.agent_embedding(agent_ids)[None, None].expand(
            batch, horizon, -1, -1
        )
        histories = torch.cat(
            [observations, previous_actions, identities], dim=-1
        ).permute(0, 2, 1, 3)
        histories = histories.reshape(batch * self.n_agents, horizon, -1)

        lengths = mask.sum(dim=1).long()
        if torch.any(lengths < 1):
            raise ValueError("each local history must contain a valid step")
        lengths = lengths[:, None].expand(-1, self.n_agents).reshape(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            histories,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        logits = self.classifier(hidden[-1])
        return logits.reshape(batch, self.n_agents, self.n_modes)

    @torch.no_grad()
    def infer(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
        confidence_threshold: float = 0.0,
        unknown_code: int = -1,
    ) -> Dict[str, torch.Tensor]:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        probabilities = F.softmax(self(observations, actions, mask), dim=-1)
        confidence, codes = probabilities.max(dim=-1)
        codes = codes.masked_fill(confidence < confidence_threshold, unknown_code)
        return {
            "codes": codes,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def _validate_inputs(self, observations, actions, mask) -> None:
        if observations.ndim != 4 or observations.shape[2] != self.n_agents:
            raise ValueError(
                "observations must have shape [batch, time, agent, observation]"
            )
        if observations.shape[-1] != self.observation_dim:
            raise ValueError("unexpected observation dimension")
        if actions.shape != (*observations.shape[:3], self.action_dim):
            raise ValueError("actions must match observations and action_dim")
        if mask.shape != observations.shape[:2]:
            raise ValueError("mask must have shape [batch, time]")


class TeamModeVQVAE(nn.Module):
    """把联合轨迹窗口量化为一个含 agent-specific role 的团队 mode。

    Codebook 逻辑形状为 ``[n_modes, n_agents, mode_dim]``。一个 code ID
    对应整支团队，最后两维分别给出每个固定 agent slot 的角色向量。
    """

    agent_share_parameters = True

    def __init__(
        self,
        n_agents: int,
        observation_dim: int,
        action_dim: int,
        n_modes: int,
        hidden_dim: int = 128,
        mode_dim: int = 32,
        agent_embedding_dim: int = 8,
        prior_hidden_dim: Optional[int] = None,
        codebook_weight: float = 1.0,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        if n_agents < 2:
            raise ValueError("n_agents must be at least two")
        if n_modes < 2:
            raise ValueError("n_modes must be at least two")
        if codebook_weight < 0.0 or commitment_weight < 0.0:
            raise ValueError("VQ loss weights must be non-negative")

        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.n_modes = n_modes
        self.mode_dim = mode_dim
        self.codebook_weight = codebook_weight
        self.commitment_weight = commitment_weight

        self.agent_encoder = nn.GRU(
            observation_dim + action_dim, hidden_dim, batch_first=True
        )
        self.agent_embedding = nn.Embedding(n_agents, agent_embedding_dim)
        self.team_encoder = nn.Sequential(
            nn.Linear(n_agents * (hidden_dim + agent_embedding_dim), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_agents * mode_dim),
        )
        self.codebook = nn.Embedding(n_modes, n_agents * mode_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_modes, 1.0 / n_modes)

        self.action_decoder = nn.Sequential(
            nn.Linear(observation_dim + mode_dim + agent_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.local_prior = LocalModePrior(
            n_agents=n_agents,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_modes=n_modes,
            hidden_dim=prior_hidden_dim or hidden_dim,
            agent_embedding_dim=agent_embedding_dim,
        )

    @property
    def role_codebook(self) -> torch.Tensor:
        return self.codebook.weight.reshape(self.n_modes, self.n_agents, self.mode_dim)

    def encode_modes(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(observations, actions, mask)
        batch, horizon = observations.shape[:2]
        sequence = torch.cat([observations, actions], dim=-1)
        sequence = sequence.permute(0, 2, 1, 3).reshape(
            batch * self.n_agents, horizon, -1
        )
        lengths = mask.sum(dim=1).long()
        if torch.any(lengths < 1):
            raise ValueError("each trajectory must contain a valid step")
        packed_lengths = lengths[:, None].expand(-1, self.n_agents).reshape(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            packed_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.agent_encoder(packed)
        agent_features = hidden[-1].reshape(batch, self.n_agents, -1)
        agent_ids = torch.arange(self.n_agents, device=observations.device)
        identities = self.agent_embedding(agent_ids)[None].expand(batch, -1, -1)
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
        encoded, quantized, distances, codes = self.encode_modes(
            observations, actions, mask
        )
        straight_through = encoded + (quantized - encoded).detach()
        predicted_actions = self._decode_role_vectors(
            observations,
            straight_through.reshape(-1, self.n_agents, self.mode_dim),
        )

        reconstruction_loss = _masked_action_mse(predicted_actions, actions, mask)
        codebook_loss = F.mse_loss(quantized, encoded.detach())
        commitment_loss = F.mse_loss(encoded, quantized.detach())
        loss = (
            reconstruction_loss
            + self.codebook_weight * codebook_loss
            + self.commitment_weight * commitment_loss
        )
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "codes": codes,
            "code_probabilities": F.softmax(-distances, dim=-1),
            "role_vectors": straight_through.reshape(-1, self.n_agents, self.mode_dim),
            "predicted_actions": predicted_actions,
        }

    def prior_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
        teacher_codes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits = self.local_prior(observations, actions, mask)
        if teacher_codes.shape != (observations.shape[0],):
            raise ValueError("teacher_codes must have shape [batch]")
        targets = teacher_codes[:, None].expand(-1, self.n_agents)
        loss = F.cross_entropy(logits.reshape(-1, self.n_modes), targets.reshape(-1))
        probabilities = F.softmax(logits, dim=-1)
        return {
            "loss": loss,
            "logits": logits,
            "probabilities": probabilities,
        }

    @torch.no_grad()
    def predict_codes(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_modes(observations, actions, mask)[-1]

    @torch.no_grad()
    def infer_local_modes(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
        confidence_threshold: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        return self.local_prior.infer(
            observations,
            actions,
            mask,
            confidence_threshold=confidence_threshold,
        )

    def decode_actions(
        self, observations: torch.Tensor, codes: torch.Tensor
    ) -> torch.Tensor:
        if codes.shape != (observations.shape[0],):
            raise ValueError("codes must have shape [batch]")
        if torch.any(codes < -1) or torch.any(codes >= self.n_modes):
            raise ValueError("codes contain an invalid codebook entry")
        known = codes >= 0
        role_vectors = self.codebook(codes.long().clamp_min(0)).reshape(
            -1, self.n_agents, self.mode_dim
        )
        role_vectors = role_vectors * known[:, None, None].to(role_vectors.dtype)
        return self._decode_role_vectors(observations, role_vectors)

    def _decode_role_vectors(
        self,
        observations: torch.Tensor,
        role_vectors: torch.Tensor,
    ) -> torch.Tensor:
        batch, horizon = observations.shape[:2]
        if role_vectors.shape != (batch, self.n_agents, self.mode_dim):
            raise ValueError("role_vectors must have shape [batch, agent, mode_dim]")
        agent_ids = torch.arange(self.n_agents, device=observations.device)
        identities = self.agent_embedding(agent_ids)[None, None].expand(
            batch, horizon, -1, -1
        )
        roles = role_vectors[:, None].expand(-1, horizon, -1, -1)
        return self.action_decoder(torch.cat([observations, roles, identities], dim=-1))

    def _validate_inputs(self, observations, actions, mask) -> None:
        if observations.ndim != 4 or observations.shape[2] != self.n_agents:
            raise ValueError(
                "observations must have shape [batch, time, agent, observation]"
            )
        if observations.shape[-1] != self.observation_dim:
            raise ValueError("unexpected observation dimension")
        if actions.shape != (*observations.shape[:3], self.action_dim):
            raise ValueError("actions must match observations and action_dim")
        if mask.shape != observations.shape[:2]:
            raise ValueError("mask must have shape [batch, time]")
