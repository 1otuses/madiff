"""标准 train.py 可用的 mode discovery / diffusion 窗口数据集。"""

from types import SimpleNamespace
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from diffuser.datasets import normalization as dataset_normalization

from .offline import EpisodeStore, load_episode_store


class ModeDatasetNormalizer:
    """复用 MADiff normalizer 规范化 observation/action feature。"""

    def __init__(
        self,
        observation_normalizer,
        action_normalizer,
        normalizer_name: str,
    ):
        self.observation_normalizer = observation_normalizer
        self.action_normalizer = action_normalizer
        self.normalizer_name = normalizer_name

    @classmethod
    def fit(
        cls,
        store: EpisodeStore,
        episode_indices: Sequence[int],
        normalizer: str,
    ):
        indices = np.asarray(episode_indices, dtype=np.int64)
        valid = store.mask[indices].astype(bool)
        observations = store.observations[indices][valid].reshape(
            -1, store.observation_dim
        )
        actions = store.actions[indices][valid].reshape(-1, store.action_dim)
        normalizer_class = _resolve_normalizer(normalizer)
        return cls(
            observation_normalizer=normalizer_class(observations),
            action_normalizer=normalizer_class(actions),
            normalizer_name=normalizer_class.__name__,
        )

    def normalize(self, value: np.ndarray, key: str) -> np.ndarray:
        if key == "observations":
            return self.observation_normalizer.normalize(value)
        if key == "actions":
            return self.action_normalizer.normalize(value)
        raise KeyError(key)

    def unnormalize(self, value: np.ndarray, key: str) -> np.ndarray:
        if key == "observations":
            return self.observation_normalizer.unnormalize(value)
        if key == "actions":
            return self.action_normalizer.unnormalize(value)
        raise KeyError(key)


def _resolve_normalizer(name: str):
    normalizer = (
        getattr(dataset_normalization, name, None) if isinstance(name, str) else None
    )
    if not isinstance(normalizer, type) or not issubclass(
        normalizer, dataset_normalization.Normalizer
    ):
        choices = sorted(
            candidate
            for candidate, value in vars(dataset_normalization).items()
            if isinstance(value, type)
            and issubclass(value, dataset_normalization.Normalizer)
            and value is not dataset_normalization.Normalizer
        )
        raise ValueError(
            "unknown normalizer {!r}; expected one of {}".format(name, choices)
        )
    return normalizer


def split_episode_indices(
    store: EpisodeStore,
    eval_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """零比例返回完整训练集；非零时优先隔离 scenario，再按 collector 分层。"""

    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0, 1)")
    if eval_fraction == 0.0:
        return np.arange(store.n_episodes, dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.RandomState(seed)

    if store.scenario_ids is not None and len(np.unique(store.scenario_ids)) > 1:
        groups = rng.permutation(np.unique(store.scenario_ids))
        n_eval = min(max(1, int(round(len(groups) * eval_fraction))), len(groups) - 1)
        eval_mask = np.isin(store.scenario_ids, groups[:n_eval])
        return np.flatnonzero(~eval_mask), np.flatnonzero(eval_mask)

    if store.collector_ids is not None:
        train_parts, eval_parts = [], []
        for collector in np.unique(store.collector_ids):
            indices = np.flatnonzero(store.collector_ids == collector)
            indices = rng.permutation(indices)
            if len(indices) == 1:
                train_parts.append(indices)
                continue
            n_eval = min(
                max(1, int(round(len(indices) * eval_fraction))), len(indices) - 1
            )
            eval_parts.append(indices[:n_eval])
            train_parts.append(indices[n_eval:])
        train = np.sort(np.concatenate(train_parts))
        evaluation = (
            np.sort(np.concatenate(eval_parts))
            if eval_parts
            else np.empty(0, dtype=np.int64)
        )
        if len(evaluation) > 0:
            return train, evaluation

    indices = rng.permutation(store.n_episodes)
    n_eval = min(
        max(1, int(round(store.n_episodes * eval_fraction))),
        store.n_episodes - 1,
    )
    return np.sort(indices[n_eval:]), np.sort(indices[:n_eval])


class ModeSequenceDataset(torch.utils.data.Dataset):
    """从 EpisodeStore 生成 VQ/prior 或 mode diffusion 的固定长度窗口。"""

    def __init__(
        self,
        dataset_path: str,
        objective: str,
        horizon: int,
        split: str = "train",
        stride: int = 1,
        eval_fraction: float = 0.0,
        seed: int = 0,
        max_n_episodes: Optional[int] = None,
        use_padding: bool = True,
        normalizer: str = "CDFNormalizer",
        use_inv_dyn: bool = False,
        include_returns: bool = True,
        returns_scale: float = 400.0,
        discount: float = 0.99,
        history_horizon: int = 0,
        n_agents: Optional[int] = None,
        **kwargs,
    ):
        del kwargs
        if objective not in {"mode_stage1", "mode_stage2"}:
            raise ValueError("objective must be mode_stage1 or mode_stage2")
        if split not in {"train", "eval"}:
            raise ValueError("split must be train or eval")
        if horizon < 1 or stride < 1:
            raise ValueError("horizon and stride must be positive")
        if discount < 0.0:
            raise ValueError("discount must be non-negative")
        if returns_scale <= 0.0:
            raise ValueError("returns_scale must be positive")
        if history_horizon != 0:
            raise ValueError("ModeSequenceDataset currently requires history_horizon=0")
        if objective == "mode_stage2" and not use_inv_dyn:
            raise ValueError("mode_stage2 requires use_inv_dyn=True")

        self.dataset_path = str(dataset_path)
        self.store = load_episode_store(dataset_path)
        if n_agents is not None and n_agents != self.store.n_agents:
            raise ValueError("configured n_agents does not match EpisodeStore")
        train_indices, eval_indices = split_episode_indices(
            self.store, eval_fraction=eval_fraction, seed=seed
        )
        normalizer_indices = train_indices
        if max_n_episodes is not None:
            normalizer_indices = normalizer_indices[:max_n_episodes]
        episode_indices = train_indices if split == "train" else eval_indices
        if max_n_episodes is not None:
            episode_indices = episode_indices[:max_n_episodes]
        if len(episode_indices) == 0:
            if split == "eval" and eval_fraction == 0.0:
                raise ValueError(
                    "no offline evaluation split when eval_fraction=0; "
                    "use the online environment evaluator"
                )
            raise ValueError("selected dataset split is empty")

        self.objective = objective
        self.split = split
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.stride = stride
        self.use_padding = use_padding
        self.include_returns = include_returns
        self.returns_scale = float(returns_scale)
        self.discount = float(discount)
        self.n_agents = self.store.n_agents
        self.observation_dim = self.store.observation_dim
        self.action_dim = self.store.action_dim
        self.max_path_length = self.store.horizon
        self.use_inv_dyn = use_inv_dyn
        self.episode_indices = np.asarray(episode_indices, dtype=np.int64)
        self.env = SimpleNamespace(num_actions=0)
        self.normalizer = ModeDatasetNormalizer.fit(
            self.store,
            normalizer_indices,
            normalizer=normalizer,
        )

        rewards = np.asarray(self.store.rewards, dtype=np.float32)
        self.team_rewards = rewards.reshape(*rewards.shape[:2], -1).mean(axis=-1)
        self.discounts = self.discount ** np.arange(
            self.store.horizon, dtype=np.float32
        )

        windows = []
        for episode_index in episode_indices:
            valid_length = int(self.store.mask[episode_index].sum())
            if valid_length < 1:
                continue
            if valid_length < horizon:
                if use_padding:
                    windows.append((int(episode_index), 0))
                continue
            starts = list(range(0, valid_length - horizon + 1, stride))
            final_start = valid_length - horizon
            if starts[-1] != final_start:
                starts.append(final_start)
            windows.extend((int(episode_index), int(start)) for start in starts)
        if not windows:
            raise ValueError("no valid trajectory windows were produced")
        self.indices = windows
        self.window_episode_indices = np.asarray(
            [episode for episode, _ in windows], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        episode, start = self.indices[index]
        observations, actions, mask = self._window(episode, start)
        if self.objective == "mode_stage1":
            return {
                "observations": observations,
                "actions": actions,
                "mask": mask,
            }

        trajectories = np.concatenate([actions, observations], axis=-1)
        cond_x = np.zeros_like(observations)
        cond_masks = np.zeros_like(observations, dtype=bool)
        cond_x[0] = observations[0]
        cond_masks[0] = True
        loss_masks = mask[:, None, None] * np.ones(
            (1, self.n_agents, 1), dtype=np.float32
        )
        loss_masks[0] = 0.0
        attention_masks = loss_masks.copy()
        batch = {
            "x": trajectories,
            "cond": {"x": cond_x, "masks": cond_masks},
            "loss_masks": loss_masks,
            "attention_masks": attention_masks,
            "mode_mask": mask,
        }
        if self.include_returns:
            rewards = self.team_rewards[episode, start:]
            valid = self.store.mask[episode, start:]
            discounts = self.discounts[: len(rewards)]
            value = float(
                (rewards * valid * discounts).sum() / self.returns_scale
            )
            batch["returns"] = np.full((1, self.n_agents), value, dtype=np.float32)
        return batch

    def audit_labels(self, indices: Sequence[int], key: str):
        """仅供冻结 evaluator 使用；该信息永不出现在训练 batch 中。"""

        if key not in {"true_modes", "collector_ids", "scenario_ids", "quality_ids"}:
            raise KeyError(key)
        labels = getattr(self.store, key)
        if labels is None:
            return None
        windows = self.window_episode_indices[np.asarray(indices, dtype=np.int64)]
        return np.asarray(labels)[windows]

    def _window(self, episode: int, start: int):
        end = min(start + self.horizon, self.store.horizon)
        length = end - start
        observations = np.zeros(
            (self.horizon, self.n_agents, self.observation_dim), dtype=np.float32
        )
        actions = np.zeros(
            (self.horizon, self.n_agents, self.action_dim), dtype=np.float32
        )
        mask = np.zeros(self.horizon, dtype=np.float32)
        raw_mask = self.store.mask[episode, start:end].astype(np.float32)
        observations[:length] = self.normalizer.normalize(
            self.store.observations[episode, start:end], "observations"
        )
        actions[:length] = self.normalizer.normalize(
            self.store.actions[episode, start:end], "actions"
        )
        mask[:length] = raw_mask
        observations *= mask[:, None, None]
        actions *= mask[:, None, None]
        return observations, actions, mask
