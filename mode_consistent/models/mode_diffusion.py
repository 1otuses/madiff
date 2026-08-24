"""Return 与离散团队 mode 双条件的 Gaussian diffusion。"""

from pathlib import Path
from typing import Dict, Mapping, Optional

import torch

from diffuser.models.diffusion import GaussianDiffusion

from .team_mode_vqvae import TeamModeVQVAE


def _mode_state_dict(checkpoint: Mapping, use_ema: bool) -> Mapping:
    if use_ema and "ema" in checkpoint:
        state_dict = checkpoint["ema"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    expected_prefix = "mode_model."
    prefixed = {
        key[len(expected_prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(expected_prefix)
    }
    return prefixed or state_dict


def load_mode_checkpoint(
    mode_model: TeamModeVQVAE,
    checkpoint_path: str,
    use_ema: bool = True,
) -> None:
    """读取标准 Trainer 保存的 mode objective 或原始 model state dict。"""

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError("mode checkpoint does not exist: {}".format(path))
    checkpoint = torch.load(str(path), map_location="cpu")
    state_dict = _mode_state_dict(checkpoint, use_ema=use_ema)
    try:
        mode_model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "checkpoint is not compatible with TeamModeVQVAE: {}".format(path)
        ) from error


class ModeGaussianDiffusion(GaussianDiffusion):
    """以 RTG 为基础条件、mode 为增量条件的链式 CFG 扩散目标。"""

    def __init__(
        self,
        model,
        *args,
        mode_model_kwargs: Optional[Dict] = None,
        mode_source_train: str = "posterior",
        local_prefix_train: int = 1,
        local_confidence_threshold: float = 0.0,
        cfg_dropout_prob: float = 0.05,
        mode_guidance_w: float = 1.5,
        **kwargs,
    ):
        # 旧 checkpoint 的 Config 可能仍携带这些非参数字段。模型权重兼容，
        # 但推理语义统一切换为新的链式 CFG。
        for legacy_key in (
            "return_condition_dropout",
            "mode_condition_dropout",
            "interaction_guidance_w",
            "factorized_guidance",
        ):
            kwargs.pop(legacy_key, None)
        super().__init__(model, *args, **kwargs)
        if self.use_inv_dyn and self.joint_inv:
            raise ValueError(
                "ModeGaussianDiffusion requires local, not joint, inverse dynamics"
            )
        if mode_source_train not in {"posterior", "local", "none"}:
            raise ValueError("mode_source_train must be posterior, local, or none")
        if not 0.0 <= cfg_dropout_prob < 1.0 / 3.0:
            raise ValueError("cfg_dropout_prob must be in [0, 1/3)")
        if not 0.0 <= local_confidence_threshold <= 1.0:
            raise ValueError("local_confidence_threshold must be in [0, 1]")

        self.mode_source_train = mode_source_train
        if local_prefix_train < 1:
            raise ValueError("local_prefix_train must be positive")
        if mode_source_train == "local" and local_prefix_train != 1:
            raise ValueError(
                "local-conditioned diffusion currently requires "
                "local_prefix_train=1 to avoid target-action leakage"
            )
        self.local_prefix_train = local_prefix_train
        self.local_confidence_threshold = local_confidence_threshold
        self.cfg_dropout_prob = cfg_dropout_prob
        self.mode_guidance_w = mode_guidance_w

        self.mode_model = None
        if mode_model_kwargs is not None:
            self.mode_model = TeamModeVQVAE(
                n_agents=self.n_agents,
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                **dict(mode_model_kwargs),
            )
            self.mode_model.requires_grad_(False)
            self.mode_model.eval()
            self._sync_mode_codebook()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mode_model is not None:
            self.mode_model.eval()
        return self

    def load_pretrained_mode(self, checkpoint_path: str, use_ema: bool = True) -> None:
        if self.mode_model is None:
            raise ValueError(
                "mode_model_kwargs are required before loading a checkpoint"
            )
        load_mode_checkpoint(self.mode_model, checkpoint_path, use_ema=use_ema)
        self.mode_model.requires_grad_(False)
        self.mode_model.eval()
        self._sync_mode_codebook()

    def _sync_mode_codebook(self) -> None:
        if self.mode_model is None:
            return
        if not hasattr(self.model, "set_mode_codebook"):
            raise TypeError("denoiser must expose set_mode_codebook")
        self.model.set_mode_codebook(self.mode_model.role_codebook, freeze=True)

    @torch.no_grad()
    def infer_mode_codes(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
        source: str,
    ) -> torch.Tensor:
        if source == "none":
            return torch.full(
                (observations.shape[0], self.n_agents),
                -1,
                device=observations.device,
                dtype=torch.long,
            )
        if self.mode_model is None:
            raise RuntimeError("a frozen TeamModeVQVAE is required to infer modes")
        if source == "posterior":
            return self.mode_model.predict_codes(observations, actions, mask)
        if source == "local":
            return self.mode_model.infer_local_modes(
                observations,
                actions,
                mask,
                confidence_threshold=self.local_confidence_threshold,
            )["codes"]
        raise ValueError("source must be posterior, local, or none")

    def predict_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """用每个 agent 自身的相邻 observation 反演连续动作。"""

        if not self.use_inv_dyn:
            raise RuntimeError("predict_actions requires use_inv_dyn=True")
        if self.joint_inv:
            raise RuntimeError("joint inverse dynamics is not allowed")
        if observations.ndim != 4 or observations.shape[2:] != (
            self.n_agents,
            self.observation_dim,
        ):
            raise ValueError(
                "observations must have shape [batch, time, agent, observation]"
            )
        if observations.shape[1] < 2:
            raise ValueError("at least two observation steps are required")

        pairs = torch.cat([observations[:, :-1], observations[:, 1:]], dim=-1)
        if self.share_inv:
            return self.inv_model(pairs)
        return torch.stack(
            [self.inv_model[i](pairs[:, :, i]) for i in range(self.n_agents)],
            dim=2,
        )

    def loss(
        self,
        x: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        legal_actions: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        mode_mask: Optional[torch.Tensor] = None,
    ):
        model_kwargs = dict(model_kwargs or {})
        if "mode_codes" not in model_kwargs:
            if mode_mask is None:
                mode_mask = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
            observations = x[..., self.action_dim :]
            actions = x[..., : self.action_dim]
            if self.mode_source_train == "local":
                prefix = min(self.local_prefix_train, observations.shape[1])
                observations = observations[:, :prefix]
                actions = actions[:, :prefix]
                mode_mask = mode_mask[:, :prefix]
            model_kwargs["mode_codes"] = self.infer_mode_codes(
                observations,
                actions,
                mode_mask,
                source=self.mode_source_train,
            )
        return super().loss(
            x=x,
            cond=cond,
            loss_masks=loss_masks,
            attention_masks=attention_masks,
            returns=returns,
            env_ts=env_ts,
            states=states,
            legal_actions=legal_actions,
            model_kwargs=model_kwargs,
        )

    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        t: torch.Tensor,
        loss_masks: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
    ):
        cfg_kwargs = dict(model_kwargs or {})
        mode_codes = cfg_kwargs.get("mode_codes")
        has_mode = mode_codes is not None and bool(torch.any(mode_codes >= 0).item())
        force_return_dropout, force_mode_dropout = self._sample_cfg_dropout(
            device=x_start.device,
            has_return=self.returns_condition,
            has_mode=has_mode,
        )
        cfg_kwargs.update(
            use_dropout=False,
            force_dropout=force_return_dropout,
            use_mode_dropout=False,
            force_mode_dropout=force_mode_dropout,
        )
        loss, info = super().p_losses(
            x_start=x_start,
            cond=cond,
            t=t,
            loss_masks=loss_masks,
            attention_masks=attention_masks,
            returns=returns,
            env_ts=env_ts,
            states=states,
            model_kwargs=cfg_kwargs,
        )
        info["return_condition_dropped"] = x_start.new_tensor(
            float(force_return_dropout)
        )
        info["mode_condition_dropped"] = x_start.new_tensor(float(force_mode_dropout))
        return loss, info

    def _sample_cfg_dropout(self, device, has_return: bool, has_mode: bool):
        """按 InstructPix2Pix 的四种条件状态采样 minibatch mask。"""

        draw = float(torch.rand((), device=device).item())
        probability = self.cfg_dropout_prob
        if draw < probability:
            drop_return, drop_mode = True, True
        elif draw < 2.0 * probability:
            drop_return, drop_mode = False, True
        elif draw < 3.0 * probability:
            drop_return, drop_mode = True, False
        else:
            drop_return, drop_mode = False, False
        return drop_return and has_return, drop_mode and has_mode

    def get_model_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
    ):
        del states
        model_kwargs = dict(model_kwargs or {})
        mode_codes = model_kwargs.get("mode_codes")
        has_mode = mode_codes is not None and bool(torch.any(mode_codes >= 0).item())
        has_return = self.returns_condition

        if not has_mode and not has_return:
            return self._model_prediction(
                x,
                t,
                returns,
                env_ts,
                attention_masks,
                model_kwargs,
                force_return_dropout=False,
                force_mode_dropout=False,
            )

        epsilon_un = self._model_prediction(
            x,
            t,
            returns,
            env_ts,
            attention_masks,
            model_kwargs,
            force_return_dropout=has_return,
            force_mode_dropout=has_mode,
        )
        if has_return and has_mode:
            epsilon_r = self._model_prediction(
                x,
                t,
                returns,
                env_ts,
                attention_masks,
                model_kwargs,
                force_return_dropout=False,
                force_mode_dropout=True,
            )
            epsilon_all = self._model_prediction(
                x,
                t,
                returns,
                env_ts,
                attention_masks,
                model_kwargs,
                force_return_dropout=False,
                force_mode_dropout=False,
            )
            return (
                epsilon_un
                + self.condition_guidance_w * (epsilon_r - epsilon_un)
                + self.mode_guidance_w * (epsilon_all - epsilon_r)
            )

        epsilon_conditional = self._model_prediction(
            x,
            t,
            returns,
            env_ts,
            attention_masks,
            model_kwargs,
            force_return_dropout=False,
            force_mode_dropout=False,
        )
        guidance_weight = (
            self.condition_guidance_w if has_return else self.mode_guidance_w
        )
        return epsilon_un + guidance_weight * (epsilon_conditional - epsilon_un)

    def _model_prediction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor],
        env_ts: Optional[torch.Tensor],
        attention_masks: Optional[torch.Tensor],
        model_kwargs: Dict,
        force_return_dropout: bool,
        force_mode_dropout: bool,
    ) -> torch.Tensor:
        kwargs = dict(model_kwargs)
        for key in (
            "use_dropout",
            "force_dropout",
            "use_mode_dropout",
            "force_mode_dropout",
        ):
            kwargs.pop(key, None)
        return self.model(
            x,
            t,
            returns=returns,
            env_timestep=env_ts,
            attention_masks=attention_masks,
            use_dropout=False,
            force_dropout=force_return_dropout,
            use_mode_dropout=False,
            force_mode_dropout=force_mode_dropout,
            **kwargs,
        )
