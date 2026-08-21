"""借鉴 CLS-DP、从集中教师到局部学生的离散 mode 蒸馏。"""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class LocalModeAligner(nn.Module):
    """每个 agent 仅根据自身局部历史推断离散团队 mode。"""

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
        # 时刻 t 只能使用 t 之前的自身动作，避免当前动作泄漏决策结果。
        previous_actions = torch.zeros_like(actions)
        previous_actions[:, 1:] = actions[:, :-1]
        agent_ids = torch.arange(self.n_agents, device=observations.device)
        agent_embedding = self.agent_embedding(agent_ids)[None, None].expand(
            batch, horizon, -1, -1
        )
        histories = torch.cat(
            [observations, previous_actions, agent_embedding], dim=-1
        ).permute(0, 2, 1, 3)
        # agent 维并入 batch，使所有 agent 共享同一个局部历史编码器。
        histories = histories.reshape(batch * self.n_agents, horizon, -1)
        lengths = mask.sum(dim=1).long()
        if torch.any(lengths < 1):
            raise ValueError("each history must contain at least one valid step")
        lengths = lengths[:, None].expand(-1, self.n_agents).reshape(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            histories,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        return self.classifier(hidden[-1]).reshape(batch, self.n_agents, self.n_modes)

    def distillation_loss(
        self,
        logits: torch.Tensor,
        teacher_codes: torch.Tensor,
        agreement_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        if logits.ndim != 3 or logits.shape[1:] != (self.n_agents, self.n_modes):
            raise ValueError("logits must have shape [batch, agent, mode]")
        if teacher_codes.shape != (logits.shape[0],):
            raise ValueError("teacher_codes must have shape [batch]")
        targets = teacher_codes[:, None].expand(-1, self.n_agents)
        distillation = F.cross_entropy(
            logits.reshape(-1, self.n_modes), targets.reshape(-1)
        )
        probabilities = F.softmax(logits, dim=-1)
        consensus = probabilities.mean(dim=1, keepdim=True)
        # 平均 KL 散度约束同一 episode 中各 agent 的后验分布趋于一致。
        agreement = (
            (
                probabilities
                * (
                    torch.log(probabilities.clamp_min(1e-8))
                    - torch.log(consensus.clamp_min(1e-8))
                )
            )
            .sum(dim=-1)
            .mean()
        )
        return {
            "loss": distillation + agreement_weight * agreement,
            "distillation_loss": distillation,
            "agreement_loss": agreement,
        }

    @torch.no_grad()
    def infer_codes(
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
            raise ValueError("observations must have shape [batch, time, agent, obs]")
        if observations.shape[-1] != self.observation_dim:
            raise ValueError("unexpected observation dimension")
        if actions.shape != (*observations.shape[:3], self.action_dim):
            raise ValueError("actions must match observations and action_dim")
        if mask.shape != observations.shape[:2]:
            raise ValueError("mask must have shape [batch, time]")
