"""按 episode 读取 OMAR 中由不同采集器生成的 MPE 文件。

采集器目录只属于审计元数据，不能作为监督目标暴露给无标签 mode 发现模型。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from .offline import EpisodeStore


TASK_OBSERVATION_DIMS = {
    "simple_spread": 18,
    "simple_tag": 16,
    "simple_world": 24,
}


@dataclass(frozen=True)
class OMARCollectorEpisodes:
    task: str
    collector_index: int
    horizon: int
    observations: Tuple[np.ndarray, ...]
    actions: Tuple[np.ndarray, ...]
    rewards: Tuple[np.ndarray, ...]
    next_observations: Tuple[np.ndarray, ...]

    @property
    def n_steps(self) -> int:
        return len(self.observations[0])

    @property
    def n_episodes(self) -> int:
        return self.n_steps // self.horizon

    def joint_at_step(self, field: str, step: int) -> np.ndarray:
        if not 0 <= step < self.horizon:
            raise ValueError("step must be within the episode horizon")
        arrays = getattr(self, field)
        return np.stack([array[step :: self.horizon] for array in arrays], axis=1)

    def team_returns(self) -> np.ndarray:
        per_agent = [
            np.asarray(reward).reshape(self.n_episodes, self.horizon).sum(axis=1)
            for reward in self.rewards
        ]
        return np.stack(per_agent, axis=1).mean(axis=1)


def iter_expert_collectors(
    dataset_root: Path,
    task: str,
    horizon: int = 25,
    n_controlled_agents: int = 3,
) -> Iterator[OMARCollectorEpisodes]:
    """依次返回五个原始 expert 采集器，不人为构造 mode 标签。"""

    if task not in TASK_OBSERVATION_DIMS:
        raise ValueError(f"unsupported OMAR MPE task: {task}")
    task_root = Path(dataset_root) / task / "expert"
    collector_dirs = sorted(task_root.glob("seed_*_data"))
    if not collector_dirs:
        raise FileNotFoundError(f"no collector directories below {task_root}")

    for collector_index, collector_dir in enumerate(collector_dirs):
        fields = {}
        for field, prefix in (
            ("observations", "obs"),
            ("actions", "acs"),
            ("rewards", "rews"),
            ("next_observations", "next_obs"),
        ):
            arrays = tuple(
                np.load(collector_dir / f"{prefix}_{agent}.npy", mmap_mode="r")
                for agent in range(n_controlled_agents)
            )
            if len({len(array) for array in arrays}) != 1:
                raise ValueError(f"inconsistent {field} lengths in {collector_dir}")
            fields[field] = arrays

        n_steps = len(fields["observations"][0])
        if n_steps % horizon:
            raise ValueError(f"{collector_dir} is not divisible by horizon {horizon}")
        expected_dim = TASK_OBSERVATION_DIMS[task]
        if any(array.shape[1:] != (expected_dim,) for array in fields["observations"]):
            raise ValueError(f"unexpected observation shape in {collector_dir}")

        yield OMARCollectorEpisodes(
            task=task,
            collector_index=collector_index,
            horizon=horizon,
            **fields,
        )


def build_omar_episode_store(
    dataset_root: Path,
    task: str,
    horizon: int = 25,
    n_controlled_agents: int = 3,
    max_episodes_per_collector: Optional[int] = None,
) -> EpisodeStore:
    """将 OMAR collector 合并为不包含人工 mode 标签的联合 episode。"""

    if max_episodes_per_collector is not None and max_episodes_per_collector < 1:
        raise ValueError("max_episodes_per_collector 必须为正整数")

    observations = []
    actions = []
    rewards = []
    next_observations = []
    collector_ids = []
    for collector in iter_expert_collectors(
        dataset_root,
        task,
        horizon=horizon,
        n_controlled_agents=n_controlled_agents,
    ):
        n_episodes = (
            collector.n_episodes
            if max_episodes_per_collector is None
            else min(collector.n_episodes, max_episodes_per_collector)
        )
        observations.append(
            _joint_episodes(collector.observations, collector, n_episodes)
        )
        actions.append(_joint_episodes(collector.actions, collector, n_episodes))
        rewards.append(_joint_episodes(collector.rewards, collector, n_episodes))
        next_observations.append(
            _joint_episodes(collector.next_observations, collector, n_episodes)
        )
        collector_ids.append(
            np.full(n_episodes, collector.collector_index, dtype=np.int64)
        )

    joint_observations = np.concatenate(observations).astype(np.float32, copy=False)
    n_episodes = len(joint_observations)
    mask = np.ones((n_episodes, horizon), dtype=np.float32)
    terminals = np.zeros_like(mask)
    terminals[:, -1] = 1.0
    return EpisodeStore(
        observations=joint_observations,
        actions=np.concatenate(actions).astype(np.float32, copy=False),
        rewards=np.concatenate(rewards).astype(np.float32, copy=False),
        next_observations=np.concatenate(next_observations).astype(
            np.float32, copy=False
        ),
        terminals=terminals,
        mask=mask,
        collector_ids=np.concatenate(collector_ids),
        metadata={
            "benchmark": "omar_mpe",
            "task": task,
            "horizon": horizon,
            "n_collectors": len(collector_ids),
            "max_episodes_per_collector": max_episodes_per_collector,
            "mode_label_source": "none",
            "collector_ids_are_audit_only": True,
        },
    )


def _joint_episodes(
    arrays: Tuple[np.ndarray, ...], collector, n_episodes: int
) -> np.ndarray:
    """把逐 agent 的连续 step 数组恢复为联合 episode。"""

    return np.stack(
        [
            np.asarray(array).reshape(
                collector.n_episodes,
                collector.horizon,
                *array.shape[1:],
            )[:n_episodes]
            for array in arrays
        ],
        axis=2,
    )
