"""CI-CoDiff 的端到端训练骨架。

本模块显式限定信息权限：表征学习和局部对齐只能接收
:class:`UnlabeledEpisodeView`；奖励与审计标签保留在外层编排逻辑中。
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from diffuser.models import GaussianDiffusion, SharedConvAttentionDeconv

from .data.offline import EpisodeStore, UnlabeledEpisodeView
from .evaluation.metrics import local_mode_evaluation, mode_discovery_metrics
from .models import (
    CentralModeVQVAE,
    LocalModeAligner,
    ModeConditionedDenoiser,
    ModeValueModel,
)


@dataclass(frozen=True)
class PipelineConfig:
    n_modes: int = 6
    hidden_dim: int = 128
    latent_dim: int = 32
    batch_size: int = 64
    central_steps: int = 500
    local_steps: int = 500
    value_steps: int = 200
    diffusion_steps: int = 1_000
    learning_rate: float = 3e-4
    local_prefix: int = 1
    agreement_weight: float = 0.1
    confidence_threshold: float = 0.5
    diffusion_dim: int = 64
    diffusion_dim_mults: Tuple[int, ...] = (1, 2, 4)
    diffusion_timesteps: int = 20
    mode_embedding_dim: int = 32
    condition_dropout: float = 0.1
    guidance_weight: float = 1.5
    eval_fraction: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class TrajectoryStandardizer:
    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @classmethod
    def fit(cls, view: UnlabeledEpisodeView) -> "TrajectoryStandardizer":
        valid = view.mask.astype(bool)
        observations = view.observations[valid].reshape(-1, view.observation_dim)
        actions = view.actions[valid].reshape(-1, view.action_dim)
        return cls(
            observation_mean=observations.mean(axis=0).astype(np.float32),
            observation_std=observations.std(axis=0).clip(1e-6).astype(np.float32),
            action_mean=actions.mean(axis=0).astype(np.float32),
            action_std=actions.std(axis=0).clip(1e-6).astype(np.float32),
        )

    def transform(self, view: UnlabeledEpisodeView) -> UnlabeledEpisodeView:
        valid = view.mask[..., None, None].astype(np.float32)
        return UnlabeledEpisodeView(
            observations=(
                ((view.observations - self.observation_mean) / self.observation_std)
                * valid
            ).astype(np.float32),
            actions=(
                ((view.actions - self.action_mean) / self.action_std) * valid
            ).astype(np.float32),
            mask=view.mask.astype(np.float32),
        )

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "observation_mean": self.observation_mean,
            "observation_std": self.observation_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
        }


@dataclass
class PipelineArtifacts:
    config: PipelineConfig
    standardizer: TrajectoryStandardizer
    central_mode: CentralModeVQVAE
    local_aligner: LocalModeAligner
    mode_value: ModeValueModel
    diffusion: GaussianDiffusion
    summary: Dict

    def checkpoint(self) -> Dict:
        return {
            "config": asdict(self.config),
            "standardizer": self.standardizer.state_dict(),
            "central_mode": self.central_mode.state_dict(),
            "local_aligner": self.local_aligner.state_dict(),
            "mode_value": self.mode_value.state_dict(),
            "diffusion": self.diffusion.state_dict(),
            "summary": self.summary,
        }


def scenario_split(
    store: EpisodeStore, eval_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")
    group_ids = (
        np.asarray(store.scenario_ids)
        if store.scenario_ids is not None
        else np.arange(store.n_episodes)
    )
    groups = np.unique(group_ids)
    if len(groups) < 2:
        raise ValueError("at least two scenarios or episodes are required for a split")
    rng = np.random.RandomState(seed)
    # 按场景整体划分，确保同一初始状态的不同 mode 不会跨越训练集与评估集。
    groups = rng.permutation(groups)
    n_eval = min(max(1, int(round(len(groups) * eval_fraction))), len(groups) - 1)
    eval_groups = groups[:n_eval]
    eval_mask = np.isin(group_ids, eval_groups)
    return np.flatnonzero(~eval_mask), np.flatnonzero(eval_mask)


def run_pipeline(
    store: EpisodeStore,
    config: PipelineConfig = PipelineConfig(),
    device: Optional[str] = None,
) -> PipelineArtifacts:
    """依次拟合当前各算法阶段，并执行一次分散采样。"""

    if store.horizon < 2:
        raise ValueError("diffusion training requires at least two trajectory steps")
    if not 1 <= config.local_prefix <= store.horizon:
        raise ValueError("local_prefix must be within the episode horizon")
    _set_seed(config.seed)
    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_indices, eval_indices = scenario_split(
        store, config.eval_fraction, config.seed
    )
    # 后续表征模块只能看到无奖励、无标签的训练视图。
    raw_train = store.training_view(train_indices)
    raw_eval = store.training_view(eval_indices)
    standardizer = TrajectoryStandardizer.fit(raw_train)
    train = standardizer.transform(raw_train)
    evaluation = standardizer.transform(raw_eval)

    central, central_loss = fit_central_mode(train, config, torch_device)
    train_codes = predict_central_codes(central, train, config.batch_size, torch_device)
    eval_codes = predict_central_codes(
        central, evaluation, config.batch_size, torch_device
    )

    local, local_loss = fit_local_aligner(
        train.prefix(config.local_prefix), train_codes, config, torch_device
    )
    local_output = infer_local_codes(
        local,
        evaluation.prefix(config.local_prefix),
        config.batch_size,
        config.confidence_threshold,
        torch_device,
    )

    returns = _team_returns(store)[train_indices]
    mode_value, value_loss = fit_mode_value(
        train, train_codes, returns, config, torch_device
    )
    diffusion, diffusion_loss = fit_mode_diffusion(
        train, train_codes, config, torch_device
    )

    n_sample = min(2, evaluation.n_episodes)
    sample_actions = sample_decentralized_first_actions(
        diffusion,
        evaluation.subset(np.arange(n_sample)),
        local_output["codes"][:n_sample],
        config,
        torch_device,
    )
    summary = {
        "status": "pipeline_smoke_complete",
        "device": str(torch_device),
        "train_episodes": int(len(train_indices)),
        "eval_episodes": int(len(eval_indices)),
        "central": {
            "final_loss": central_loss,
            "mode_usage": np.bincount(eval_codes, minlength=config.n_modes).tolist(),
        },
        "local": {
            "final_loss": local_loss,
            **local_mode_evaluation(local_output["codes"], eval_codes, config.n_modes),
        },
        "value": {"final_loss": value_loss},
        "diffusion": {
            "final_loss": diffusion_loss,
            "sample_shape": list(sample_actions.shape),
            "sample_finite": bool(np.isfinite(sample_actions).all()),
        },
    }
    if store.true_modes is not None:
        summary["central"].update(
            mode_discovery_metrics(eval_codes, store.true_modes[eval_indices])
        )
    return PipelineArtifacts(
        config=config,
        standardizer=standardizer,
        central_mode=central,
        local_aligner=local,
        mode_value=mode_value,
        diffusion=diffusion,
        summary=summary,
    )


def fit_central_mode(view, config, device):
    model = CentralModeVQVAE(
        n_agents=view.n_agents,
        observation_dim=view.observation_dim,
        action_dim=view.action_dim,
        n_modes=config.n_modes,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.RandomState(config.seed)
    final_loss = float("nan")
    model.train()
    for _ in range(config.central_steps):
        indices = _sample_indices(rng, view.n_episodes, config.batch_size)
        observations, actions, mask = _tensor_batch(view, indices, device)
        output = model(observations, actions, mask)
        optimizer.zero_grad()
        output["loss"].backward()
        optimizer.step()
        final_loss = float(output["loss"].detach().cpu())
    return model, final_loss


@torch.no_grad()
def predict_central_codes(model, view, batch_size, device):
    model.eval()
    outputs = []
    for start in range(0, view.n_episodes, batch_size):
        indices = np.arange(start, min(start + batch_size, view.n_episodes))
        batch = _tensor_batch(view, indices, device)
        outputs.append(model.predict_codes(*batch).cpu().numpy())
    return np.concatenate(outputs).astype(np.int64)


def fit_local_aligner(view, teacher_codes, config, device):
    model = LocalModeAligner(
        n_agents=view.n_agents,
        observation_dim=view.observation_dim,
        action_dim=view.action_dim,
        n_modes=config.n_modes,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.RandomState(config.seed + 1)
    final_loss = float("nan")
    model.train()
    for _ in range(config.local_steps):
        indices = _sample_indices(rng, view.n_episodes, config.batch_size)
        observations, actions, mask = _tensor_batch(view, indices, device)
        codes = torch.as_tensor(teacher_codes[indices], device=device, dtype=torch.long)
        losses = model.distillation_loss(
            model(observations, actions, mask), codes, config.agreement_weight
        )
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()
        final_loss = float(losses["loss"].detach().cpu())
    return model, final_loss


@torch.no_grad()
def infer_local_codes(model, view, batch_size, confidence_threshold, device):
    model.eval()
    codes, confidences = [], []
    for start in range(0, view.n_episodes, batch_size):
        indices = np.arange(start, min(start + batch_size, view.n_episodes))
        output = model.infer_codes(
            *_tensor_batch(view, indices, device),
            confidence_threshold=confidence_threshold,
        )
        codes.append(output["codes"].cpu().numpy())
        confidences.append(output["confidence"].cpu().numpy())
    return {"codes": np.concatenate(codes), "confidence": np.concatenate(confidences)}


def fit_mode_value(view, mode_codes, returns, config, device):
    model = ModeValueModel(
        view.n_agents, view.observation_dim, config.n_modes, config.hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.RandomState(config.seed + 2)
    returns = np.asarray(returns, dtype=np.float32)
    return_mean = float(returns.mean())
    return_std = float(max(returns.std(), 1e-6))
    targets = (returns - return_mean) / return_std
    final_loss = float("nan")
    model.train()
    for _ in range(config.value_steps):
        indices = _sample_indices(rng, view.n_episodes, config.batch_size)
        initial = torch.as_tensor(
            view.observations[indices, 0], device=device, dtype=torch.float32
        )
        codes = torch.as_tensor(mode_codes[indices], device=device, dtype=torch.long)
        target = torch.as_tensor(targets[indices], device=device)
        loss = F.mse_loss(model(initial, codes), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    model.return_mean.fill_(return_mean)
    model.return_std.fill_(return_std)
    return model, final_loss


def fit_mode_diffusion(view, mode_codes, config, device):
    transition_dim = view.action_dim + view.observation_dim
    base = SharedConvAttentionDeconv(
        horizon=view.horizon,
        transition_dim=transition_dim,
        dim=config.diffusion_dim,
        history_horizon=0,
        dim_mults=config.diffusion_dim_mults,
        n_agents=view.n_agents,
        returns_condition=False,
    )
    denoiser = ModeConditionedDenoiser(
        base,
        transition_dim=transition_dim,
        n_agents=view.n_agents,
        n_modes=config.n_modes,
        mode_embedding_dim=config.mode_embedding_dim,
        condition_dropout=config.condition_dropout,
    )
    diffusion = GaussianDiffusion(
        denoiser,
        n_agents=view.n_agents,
        horizon=view.horizon,
        history_horizon=0,
        observation_dim=view.observation_dim,
        action_dim=view.action_dim,
        use_inv_dyn=False,
        n_timesteps=config.diffusion_timesteps,
    ).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=config.learning_rate)
    rng = np.random.RandomState(config.seed + 3)
    final_loss = float("nan")
    diffusion.train()
    for _ in range(config.diffusion_steps):
        indices = _sample_indices(rng, view.n_episodes, config.batch_size)
        batch = view.subset(indices)
        x, cond, loss_masks, focal_codes = decentralized_diffusion_batch(
            batch, mode_codes[indices], device
        )
        loss, _ = diffusion.loss(
            x,
            cond,
            loss_masks,
            model_kwargs={"mode_codes": focal_codes},
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return diffusion, final_loss


def decentralized_diffusion_batch(view, team_codes, device):
    """为每个 focal agent 分别构造一个 MADiff 分散预测问题。"""

    observations, actions, episode_mask = _tensor_batch(
        view, np.arange(view.n_episodes), device
    )
    trajectories = torch.cat([actions, observations], dim=-1)
    trajectories = trajectories.repeat_interleave(view.n_agents, dim=0)
    cond_x = torch.zeros_like(trajectories)
    cond_masks = torch.zeros_like(trajectories, dtype=torch.bool)
    focal = torch.arange(view.n_agents, device=device).repeat(view.n_episodes)
    rows = torch.arange(len(trajectories), device=device)
    observation_slice = slice(view.action_dim, None)
    # 每个副本只公开 focal agent 在 t=0 的局部观测；其他真实观测和动作均不可见。
    cond_x[rows, 0, focal, observation_slice] = trajectories[
        rows, 0, focal, observation_slice
    ]
    cond_masks[rows, 0, focal, observation_slice] = True
    loss_masks = episode_mask[..., None, None].expand(
        -1, -1, view.n_agents, trajectories.shape[-1]
    )
    loss_masks = loss_masks.repeat_interleave(view.n_agents, dim=0).clone()
    loss_masks[cond_masks] = 0.0
    focal_codes = torch.as_tensor(team_codes, device=device, dtype=torch.long)
    focal_codes = focal_codes[:, None].expand(-1, view.n_agents).reshape(-1)
    return trajectories, {"x": cond_x, "masks": cond_masks}, loss_masks, focal_codes


@torch.no_grad()
def sample_decentralized_first_actions(diffusion, view, local_codes, config, device):
    _, cond, _, _ = decentralized_diffusion_batch(
        view, np.zeros(view.n_episodes, dtype=np.int64), device
    )
    focal_codes = torch.as_tensor(local_codes, device=device, dtype=torch.long).reshape(
        -1
    )
    diffusion.eval()
    samples = diffusion.conditional_sample(
        cond,
        verbose=False,
        model_kwargs={
            "mode_codes": focal_codes,
            "guidance_weight": config.guidance_weight,
        },
    )
    samples = samples.reshape(
        view.n_episodes,
        view.n_agents,
        view.horizon,
        view.n_agents,
        -1,
    )
    episodes = torch.arange(view.n_episodes, device=device)[:, None]
    focal = torch.arange(view.n_agents, device=device)[None]
    actions = samples[episodes, focal, 0, focal, : view.action_dim]
    return actions.cpu().numpy()


def _tensor_batch(view, indices: Sequence[int], device):
    return tuple(
        torch.as_tensor(array[indices], device=device, dtype=torch.float32)
        for array in (view.observations, view.actions, view.mask)
    )


def _sample_indices(rng, n_episodes, batch_size):
    return rng.randint(0, n_episodes, size=min(batch_size, n_episodes))


def _team_returns(store: EpisodeStore) -> np.ndarray:
    rewards = np.asarray(store.rewards)
    if rewards.ndim == 3:
        rewards = rewards.mean(axis=-1)
    return (rewards * store.mask).sum(axis=1).astype(np.float32)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
