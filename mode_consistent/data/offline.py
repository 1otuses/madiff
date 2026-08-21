"""为模式一致性研究保留完整 episode 结构的离线数据接口。"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

ArrayPath = Union[str, Path]


@dataclass(frozen=True)
class UnlabeledEpisodeView:
    """训练专用的无标签视图，不暴露审计标签或奖励。"""

    observations: np.ndarray
    actions: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        if self.observations.ndim != 4:
            raise ValueError(
                "observations must have shape [episode, time, agent, feature]"
            )
        if self.actions.shape[:3] != self.observations.shape[:3]:
            raise ValueError(
                "actions and observations must share [episode, time, agent]"
            )
        if self.mask.shape != self.observations.shape[:2]:
            raise ValueError("mask must have shape [episode, time]")

    @property
    def n_episodes(self) -> int:
        return self.observations.shape[0]

    @property
    def horizon(self) -> int:
        return self.observations.shape[1]

    @property
    def n_agents(self) -> int:
        return self.observations.shape[2]

    @property
    def observation_dim(self) -> int:
        return self.observations.shape[3]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[3]

    def subset(self, indices: Sequence[int]) -> "UnlabeledEpisodeView":
        indices = np.asarray(indices, dtype=np.int64)
        return UnlabeledEpisodeView(
            observations=self.observations[indices],
            actions=self.actions[indices],
            mask=self.mask[indices],
        )

    def prefix(self, length: int) -> "UnlabeledEpisodeView":
        if not 1 <= length <= self.horizon:
            raise ValueError("length must be within the episode horizon")
        return UnlabeledEpisodeView(
            observations=self.observations[:, :length],
            actions=self.actions[:, :length],
            mask=self.mask[:, :length],
        )


@dataclass
class EpisodeStore:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    mask: np.ndarray
    true_modes: Optional[np.ndarray] = None
    quality_ids: Optional[np.ndarray] = None
    scenario_ids: Optional[np.ndarray] = None
    collector_ids: Optional[np.ndarray] = None
    states: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.observations.ndim != 4:
            raise ValueError(
                "observations must have shape [episode, time, agent, feature]"
            )
        if self.actions.shape[:3] != self.observations.shape[:3]:
            raise ValueError(
                "actions and observations must share [episode, time, agent]"
            )
        if self.next_observations.shape != self.observations.shape:
            raise ValueError("next_observations must match observations")
        if self.mask.shape != self.observations.shape[:2]:
            raise ValueError("mask must have shape [episode, time]")
        if self.terminals.shape != self.mask.shape:
            raise ValueError("terminals must have shape [episode, time]")
        if self.rewards.shape[:2] != self.mask.shape:
            raise ValueError("rewards must share [episode, time]")
        if self.true_modes is not None and len(self.true_modes) != self.n_episodes:
            raise ValueError("true_modes must contain one label per episode")
        if self.quality_ids is not None and len(self.quality_ids) != self.n_episodes:
            raise ValueError("quality_ids must contain one label per episode")
        if self.scenario_ids is not None and len(self.scenario_ids) != self.n_episodes:
            raise ValueError("scenario_ids must contain one label per episode")
        if self.collector_ids is not None and len(self.collector_ids) != self.n_episodes:
            raise ValueError("collector_ids must contain one id per episode")

    @property
    def n_episodes(self) -> int:
        return self.observations.shape[0]

    @property
    def horizon(self) -> int:
        return self.observations.shape[1]

    @property
    def n_agents(self) -> int:
        return self.observations.shape[2]

    @property
    def observation_dim(self) -> int:
        return self.observations.shape[3]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[3]

    @property
    def state_dim(self) -> int:
        if self.states is not None:
            return self.states.shape[-1]
        return self.n_agents * self.observation_dim

    def joint_states(self) -> np.ndarray:
        if self.states is not None:
            return self.states
        return self.observations.reshape(self.n_episodes, self.horizon, -1)

    def next_joint_states(self) -> np.ndarray:
        if self.states is not None:
            # 自建 benchmark 的全局状态由观测导出。这里统一向前平移，也兼容
            # 未单独保存 next_state 的转换数据集。
            shifted = np.concatenate([self.states[:, 1:], self.states[:, -1:]], axis=1)
            return shifted
        return self.next_observations.reshape(self.n_episodes, self.horizon, -1)

    def training_view(
        self, indices: Optional[Sequence[int]] = None
    ) -> UnlabeledEpisodeView:
        """返回训练模块唯一允许接收的数据视图，以落实标签防火墙。"""

        view = UnlabeledEpisodeView(
            observations=self.observations,
            actions=self.actions,
            mask=self.mask,
        )
        return view if indices is None else view.subset(indices)

    def save(self, path: ArrayPath) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "terminals": self.terminals,
            "mask": self.mask,
            "metadata_json": np.asarray(json.dumps(self.metadata or {})),
        }
        if self.true_modes is not None:
            arrays["true_modes"] = self.true_modes
        if self.quality_ids is not None:
            arrays["quality_ids"] = self.quality_ids
        if self.scenario_ids is not None:
            arrays["scenario_ids"] = self.scenario_ids
        if self.collector_ids is not None:
            arrays["collector_ids"] = self.collector_ids
        if self.states is not None:
            arrays["states"] = self.states
        np.savez_compressed(path, **arrays)


def load_episode_store(path: ArrayPath) -> EpisodeStore:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = (
            json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
        )
        kwargs = {
            "observations": data["observations"].astype(np.float32),
            "actions": data["actions"].astype(np.float32),
            "rewards": data["rewards"].astype(np.float32),
            "next_observations": data["next_observations"].astype(np.float32),
            "terminals": data["terminals"].astype(np.float32),
            "mask": data["mask"].astype(np.float32),
            "metadata": metadata,
        }
        for key in (
            "true_modes",
            "quality_ids",
            "scenario_ids",
            "collector_ids",
            "states",
        ):
            if key in data:
                kwargs[key] = data[key].copy()
    return EpisodeStore(**kwargs)
