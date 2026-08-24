"""把 TeamModeVQVAE 适配到 ``Trainer.model.loss(**batch)`` 协议。"""

from typing import Dict

import torch
from torch import nn

from mode_consistent.models.team_mode_vqvae import TeamModeVQVAE


class ModeVQObjective(nn.Module):
    """训练 privileged VQ teacher，并把其离散 code 蒸馏到 local prior。"""

    def __init__(
        self,
        mode_model: TeamModeVQVAE,
        train_teacher: bool = True,
        train_prior: bool = False,
        prior_loss_weight: float = 1.0,
        random_prior_prefix: bool = True,
    ):
        super().__init__()
        if not train_teacher and not train_prior:
            raise ValueError("at least one mode objective must be trainable")
        if train_prior and prior_loss_weight <= 0.0:
            raise ValueError("prior_loss_weight must be positive")
        self.mode_model = mode_model
        self.train_teacher = train_teacher
        self.train_prior = train_prior
        self.prior_loss_weight = prior_loss_weight
        self.random_prior_prefix = random_prior_prefix
        self.returns_condition = False
        self.configure_training()

    def configure_training(self) -> None:
        self.mode_model.requires_grad_(self.train_teacher)
        self.mode_model.local_prior.requires_grad_(self.train_prior)

    def loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ):
        total = observations.new_zeros(())
        info: Dict[str, torch.Tensor] = {}

        if self.train_teacher:
            teacher = self.mode_model(observations, actions, mask)
            teacher_codes = teacher["codes"].detach()
            total = total + teacher["loss"]
            info.update(
                reconstruction_loss=teacher["reconstruction_loss"],
                codebook_loss=teacher["codebook_loss"],
                commitment_loss=teacher["commitment_loss"],
            )
        else:
            with torch.no_grad():
                teacher_codes = self.mode_model.predict_codes(
                    observations, actions, mask
                )

        if self.train_prior:
            prior_observations = observations
            prior_actions = actions
            prior_mask = mask
            if self.random_prior_prefix and observations.shape[1] > 1:
                prefix = int(
                    torch.randint(
                        1,
                        observations.shape[1] + 1,
                        (),
                        device=observations.device,
                    ).item()
                )
                prior_observations = observations[:, :prefix]
                prior_actions = actions[:, :prefix]
                prior_mask = mask[:, :prefix]
            prior = self.mode_model.prior_loss(
                prior_observations,
                prior_actions,
                prior_mask,
                teacher_codes,
            )
            total = total + self.prior_loss_weight * prior["loss"]
            info["prior_loss"] = prior["loss"]

        info["active_code_fraction"] = (
            torch.unique(teacher_codes).numel() / float(self.mode_model.n_modes)
        ) * observations.new_ones(())
        return total, info

    def forward(self, *args, **kwargs):
        return self.mode_model(*args, **kwargs)
