"""为模式协调问题定制、具有固定协作模式的 Simple Spread 数据集。"""

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from diffuser.datasets.mpe import make_env

from .offline import EpisodeStore


@dataclass(frozen=True)
class ControllerQuality:
    name: str
    gain: float
    damping: float
    noise_std: float
    dropout_probability: float


DEFAULT_QUALITIES: Tuple[ControllerQuality, ...] = (
    ControllerQuality(
        "expert", gain=3.0, damping=0.5, noise_std=0.0, dropout_probability=0.0
    ),
    ControllerQuality(
        "medium", gain=0.7, damping=0.25, noise_std=0.18, dropout_probability=0.08
    ),
    ControllerQuality(
        "poor", gain=0.4, damping=0.1, noise_std=0.4, dropout_probability=0.22
    ),
)


def coordination_modes(n_agents: int = 3) -> List[Tuple[int, ...]]:
    return list(itertools.permutations(range(n_agents)))


def assigned_landmark_controller(
    observation: np.ndarray,
    assignment: Sequence[int],
    quality: ControllerQuality,
    rng: np.random.RandomState,
) -> np.ndarray:
    """在整个 episode 内由 ``assignment`` 固定意图的 PD 控制器。"""

    n_agents = len(assignment)
    actions = np.zeros((n_agents, 2), dtype=np.float32)
    for agent, landmark in enumerate(assignment):
        velocity = observation[agent, :2]
        start = 4 + 2 * landmark
        target_vector = observation[agent, start : start + 2]
        action = quality.gain * target_vector - quality.damping * velocity
        action += rng.normal(0.0, quality.noise_std, size=2)
        if rng.rand() < quality.dropout_probability:
            action = rng.uniform(-1.0, 1.0, size=2)
        action_scale = max(1.0, float(np.max(np.abs(action))))
        actions[agent] = action / action_scale
    return actions


def _rollout(
    env,
    assignment: Sequence[int],
    quality: ControllerQuality,
    horizon: int,
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    observation = np.asarray(env.reset(), dtype=np.float32)
    observations, actions, rewards, next_observations = [], [], [], []
    terminals = []
    for timestep in range(horizon):
        action = assigned_landmark_controller(observation, assignment, quality, rng)
        next_observation, reward, _, _ = env.step(action)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        observations.append(observation)
        actions.append(action)
        rewards.append(np.asarray(reward, dtype=np.float32))
        next_observations.append(next_observation)
        terminals.append(float(timestep == horizon - 1))
        observation = next_observation
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.float32),
    }


def generate_paired_assignment_dataset(
    n_scenarios: int,
    mode_ids: Sequence[int] = tuple(range(6)),
    horizon: int = 25,
    seed: int = 0,
    qualities: Iterable[ControllerQuality] = DEFAULT_QUALITIES[:1],
    mode_qualities: Optional[Iterable[ControllerQuality]] = None,
) -> EpisodeStore:
    """在同一组初始场景上采集每一种指定的 agent-landmark 分配。"""

    if n_scenarios < 1:
        raise ValueError("n_scenarios must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    qualities = tuple(qualities)
    if not qualities:
        raise ValueError("qualities must not be empty")
    if mode_qualities is not None:
        mode_qualities = tuple(mode_qualities)
        if len(mode_qualities) != len(mode_ids):
            raise ValueError("mode_qualities must match mode_ids length")
        qualities_by_mode = {
            mode_id: [(0, quality)]
            for mode_id, quality in zip(mode_ids, mode_qualities)
        }
    else:
        qualities_by_mode = {
            mode_id: list(enumerate(qualities)) for mode_id in mode_ids
        }
    modes = coordination_modes(3)
    mode_ids = tuple(int(mode_id) for mode_id in mode_ids)
    if not mode_ids or len(set(mode_ids)) != len(mode_ids):
        raise ValueError("mode_ids must be non-empty and unique")
    if min(mode_ids) < 0 or max(mode_ids) >= len(modes):
        raise ValueError("mode_ids must index the six assignment permutations")

    env = make_env("simple_spread")
    episodes = []
    true_modes = []
    quality_ids = []
    scenario_ids = []
    for scenario_id in range(n_scenarios):
        scenario_seed = seed + scenario_id
        for mode_id in mode_ids:
            assignment = modes[mode_id]
            for quality_id, quality in qualities_by_mode[mode_id]:
                env.seed(scenario_seed)
                controller_seed = (
                    (seed + 1) * 1_000_003
                    + scenario_id * 10_007
                    + mode_id * 101
                    + quality_id
                ) % (2**32 - 1)
                controller_rng = np.random.RandomState(controller_seed)
                episodes.append(
                    _rollout(env, assignment, quality, horizon, controller_rng)
                )
                true_modes.append(mode_id)
                quality_ids.append(quality_id)
                scenario_ids.append(scenario_id)

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(episodes))
    fields = {
        key: np.stack([episodes[index][key] for index in order]) for key in episodes[0]
    }
    observations = fields["observations"]
    metadata = {
        "benchmark": "ci_codiff_assignment_v1",
        "environment": "simple_spread",
        "collection_design": "paired_scenarios",
        "mode_label_source": "collector_assignment",
        "seed": seed,
        "horizon": horizon,
        "n_scenarios": n_scenarios,
        "mode_ids": list(mode_ids),
        "mode_assignments": [list(mode) for mode in modes],
        "qualities": [quality.__dict__ for quality in qualities],
        "scenario_seed_rule": "seed + scenario_id",
        "explicit_agent_id_in_observation": False,
        "communication": "zero",
    }
    return EpisodeStore(
        observations=observations,
        actions=fields["actions"],
        rewards=fields["rewards"],
        next_observations=fields["next_observations"],
        terminals=fields["terminals"],
        mask=np.ones(observations.shape[:2], dtype=np.float32),
        true_modes=np.asarray(true_modes, dtype=np.int64)[order],
        quality_ids=np.asarray(quality_ids, dtype=np.int64)[order],
        scenario_ids=np.asarray(scenario_ids, dtype=np.int64)[order],
        metadata=metadata,
    )
