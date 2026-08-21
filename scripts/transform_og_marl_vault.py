"""将 OG-MARL Flashbax Vault 转换为 MADiff 的 NumPy 缓存格式。

Flashbax 采用延迟导入，使转换辅助函数及其测试可在普通 MADiff 环境中运行。
命令行转换需要在安装了 ``flashbax`` 和 ``jax`` 的环境中执行。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

CONVERTER_VERSION = 2
REQUIRED_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "terminals",
    "truncations",
)
OUTPUT_FILENAMES = (
    "obs.npy",
    "actions.npy",
    "rewards.npy",
    "discounts.npy",
    "legals.npy",
    "states.npy",
    "path_lengths.npy",
    "manifest.json",
)


def _as_numpy_tree(value: Any) -> Any:
    """将嵌套的 JAX/NumPy 经验树复制为主机端 NumPy 数组。"""

    if isinstance(value, Mapping):
        return {key: _as_numpy_tree(child) for key, child in value.items()}
    return np.asarray(value)


def _validate_leading_shape(
    name: str, array: np.ndarray, batch_size: int, time_size: int
) -> None:
    if array.ndim < 2 or array.shape[:2] != (batch_size, time_size):
        raise ValueError(
            f"{name} must start with (B, T)=({batch_size}, {time_size}); "
            f"received {array.shape}"
        )


def _has_agent_id_prefix(observations: np.ndarray) -> bool:
    """判断每个观测是否都以对应智能体的 one-hot ID 开头。"""

    n_agents = observations.shape[2]
    if observations.shape[-1] < n_agents:
        return False
    expected = np.eye(n_agents, dtype=observations.dtype)[None, None, :, :]
    return bool(np.allclose(observations[..., :n_agents], expected))


def _episode_layout(
    done: np.ndarray, drop_incomplete_tail: bool = False
) -> Tuple[List[int], List[int]]:
    """提取 episode 长度，以及每个 batch lane 中可保留的步数。"""

    batch_size, time_size, _ = done.shape
    lengths: List[int] = []
    valid_steps: List[int] = []
    for batch_idx in range(batch_size):
        episode_ends = np.flatnonzero(done[batch_idx, :, 0])
        has_incomplete_tail = (
            len(episode_ends) == 0 or int(episode_ends[-1]) != time_size - 1
        )
        if has_incomplete_tail and not drop_incomplete_tail:
            last_end = int(episode_ends[-1]) if len(episode_ends) else None
            raise ValueError(
                "Vault contains an incomplete trailing episode in batch lane "
                f"{batch_idx}: final done index is {last_end}, expected {time_size - 1}"
            )

        # 显式允许丢弃时，仅保留最后一个完整 episode 及其之前的数据。
        lane_valid_steps = int(episode_ends[-1]) + 1 if len(episode_ends) else 0
        valid_steps.append(lane_valid_steps)
        if len(episode_ends) == 0:
            continue

        starts = np.concatenate([np.array([0], dtype=np.int64), episode_ends[:-1] + 1])
        lane_lengths = episode_ends - starts + 1
        if np.any(lane_lengths <= 0):
            raise ValueError(f"Invalid episode boundary in batch lane {batch_idx}")
        lengths.extend(int(length) for length in lane_lengths)

    if not lengths:
        raise ValueError("Vault does not contain any complete episodes")
    return lengths, valid_steps


def convert_experience(
    experience: Mapping[str, Any],
    add_agent_id: bool = True,
    drop_incomplete_tail: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """将内存中的 Vault 经验 pytree 转换为 MADiff 数组。

    Vault 数组采用 ``(B, T, ...)`` 结构。输出数组按 batch-major 顺序展平
    batch lane，同时由 ``path_lengths`` 保留所有 episode 边界。
    """

    missing = [field for field in REQUIRED_FIELDS if field not in experience]
    if missing:
        raise KeyError(f"Vault experience is missing required fields: {missing}")

    arrays = _as_numpy_tree(experience)
    observations = arrays["observations"]
    if observations.ndim != 4:
        raise ValueError(
            "observations must have shape (B, T, N, O); "
            f"received {observations.shape}"
        )
    batch_size, time_size, n_agents, observation_dim = observations.shape

    for name in REQUIRED_FIELDS[1:]:
        _validate_leading_shape(name, arrays[name], batch_size, time_size)

    actions = arrays["actions"]
    if actions.ndim not in (3, 4) or actions.shape[2] != n_agents:
        raise ValueError(
            "actions must have shape (B, T, N) or (B, T, N, A); "
            f"received {actions.shape}"
        )
    rewards = arrays["rewards"]
    if rewards.shape != (batch_size, time_size, n_agents):
        raise ValueError(
            "rewards must have shape (B, T, N); " f"received {rewards.shape}"
        )

    terminals = arrays["terminals"]
    truncations = arrays["truncations"]
    expected_done_shape = (batch_size, time_size, n_agents)
    if terminals.shape != expected_done_shape:
        raise ValueError(
            f"terminals must have shape {expected_done_shape}; received {terminals.shape}"
        )
    if truncations.shape != expected_done_shape:
        raise ValueError(
            f"truncations must have shape {expected_done_shape}; received {truncations.shape}"
        )

    done = np.logical_or(terminals.astype(bool), truncations.astype(bool))
    synchronous = np.all(done == done[..., :1], axis=-1)
    if not np.all(synchronous):
        first_bad = tuple(int(i) for i in np.argwhere(~synchronous)[0])
        raise ValueError(
            "MADiff requires agents to end episodes synchronously; "
            f"first mismatch is at (batch, time)={first_bad}"
        )
    episode_lengths, valid_steps = _episode_layout(done, drop_incomplete_tail)
    path_lengths = np.asarray(episode_lengths, dtype=np.int64)

    already_has_agent_id = _has_agent_id_prefix(observations)
    if add_agent_id and already_has_agent_id:
        raise ValueError(
            "observations already have a one-hot agent-id prefix; rerun with "
            "--no-add-agent-id to avoid adding it twice"
        )

    if add_agent_id:
        agent_ids = np.eye(n_agents, dtype=observations.dtype)[None, None, :, :]
        agent_ids = np.broadcast_to(
            agent_ids, (batch_size, time_size, n_agents, n_agents)
        )
        observations = np.concatenate([agent_ids, observations], axis=-1)

    def flatten_time(array: np.ndarray) -> np.ndarray:
        _validate_leading_shape("output field", array, batch_size, time_size)
        if all(length == time_size for length in valid_steps):
            return array.reshape(batch_size * time_size, *array.shape[2:])
        if batch_size == 1:
            return array[0, : valid_steps[0]]
        # 每个 lane 可以丢弃不同长度的尾部，因此逐 lane 截取后再拼接。
        return np.concatenate(
            [array[index, :length] for index, length in enumerate(valid_steps)],
            axis=0,
        )

    outputs: Dict[str, np.ndarray] = {
        "obs": flatten_time(observations),
        "actions": flatten_time(arrays["actions"]),
        "rewards": flatten_time(arrays["rewards"]),
        "discounts": flatten_time((~done).astype(np.float32)),
        "path_lengths": path_lengths,
    }

    infos = arrays.get("infos", {})
    if not isinstance(infos, Mapping):
        raise ValueError("infos must be a mapping when present")
    if "legals" in infos:
        legals = np.asarray(infos["legals"])
        expected_legals_prefix = (batch_size, time_size, n_agents)
        if legals.ndim < 4 or legals.shape[:3] != expected_legals_prefix:
            raise ValueError(
                "infos.legals must have shape (B, T, N, num_actions); "
                f"received {legals.shape}"
            )
        outputs["legals"] = flatten_time(legals)
    if "state" in infos:
        state = np.asarray(infos["state"])
        _validate_leading_shape("infos.state", state, batch_size, time_size)
        outputs["states"] = flatten_time(state)

    retained_transitions = int(sum(valid_steps))
    source_transitions = batch_size * time_size
    if int(path_lengths.sum()) != retained_transitions:
        raise AssertionError("episode lengths do not cover all retained transitions")

    metadata: Dict[str, Any] = {
        "converter_version": CONVERTER_VERSION,
        "source_shapes": {
            key: list(np.asarray(value).shape)
            for key, value in arrays.items()
            if key != "infos"
        },
        "info_shapes": {
            key: list(np.asarray(value).shape) for key, value in infos.items()
        },
        "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
        "batch_size": batch_size,
        "timesteps_per_batch": time_size,
        "n_agents": n_agents,
        "source_observation_dim": observation_dim,
        "output_observation_dim": int(outputs["obs"].shape[-1]),
        "source_n_transitions": source_transitions,
        "n_transitions": retained_transitions,
        "n_episodes": int(len(path_lengths)),
        "drop_incomplete_tail": drop_incomplete_tail,
        "dropped_incomplete_tail_transitions": (
            source_transitions - retained_transitions
        ),
        "retained_timesteps_per_batch_lane": valid_steps,
        "agent_id": {
            "added": add_agent_id,
            "position": "prefix" if add_agent_id or already_has_agent_id else None,
            "already_present_in_source": already_has_agent_id,
        },
        "episode_end": "terminals OR truncations",
        "flatten_order": "batch-major, then time",
    }
    return outputs, metadata


def write_madiff_cache(
    outputs: Mapping[str, np.ndarray],
    output_dir: Path,
    manifest: Mapping[str, Any],
    overwrite: bool = False,
) -> None:
    """将转换后的数组和 JSON manifest 写入输出目录。"""

    output_dir = Path(output_dir)
    existing = [
        output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory already contains converted data: {output_dir}. "
            "Pass --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for path in existing:
            path.unlink()

    filename_by_key = {
        "obs": "obs.npy",
        "actions": "actions.npy",
        "rewards": "rewards.npy",
        "discounts": "discounts.npy",
        "legals": "legals.npy",
        "states": "states.npy",
        "path_lengths": "path_lengths.npy",
    }
    for key, array in outputs.items():
        if key not in filename_by_key:
            raise KeyError(f"Unknown MADiff output field: {key}")
        np.save(output_dir / filename_by_key[key], np.asarray(array))

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(dict(manifest), file, indent=2, sort_keys=True)
        file.write("\n")


def load_vault_experience(
    vault_root: Path, source: str, env_name: str, scenario: str, uid: str
) -> Tuple[Mapping[str, Any], Path]:
    """加载 Vault,同时避免将 Flashbax 变成 MADiff 的运行时依赖。"""

    try:
        from flashbax.vault import Vault
    except ImportError as error:
        raise RuntimeError(
            "flashbax is required to read .vlt datasets. Run this converter from "
            "the OG-MARL/Flashbax environment."
        ) from error

    vault_root = Path(vault_root).expanduser().resolve()
    vault_name = f"{source}/{env_name}/{scenario}.vlt"
    vault_path = vault_root / vault_name
    if not vault_path.is_dir():
        raise FileNotFoundError(f"Vault directory not found: {vault_path}")

    vault = Vault(rel_dir=os.fspath(vault_root), vault_name=vault_name, vault_uid=uid)
    return vault.read().experience, vault_path


def _madiff_env_name(env_name: str) -> str:
    aliases = {
        "smac_v1": "smac",
        "smac_v2": "smacv2",
        "mamujoco": "mamujoco",
        "gymnasium_mamujoco": "gymnasium_mamujoco",
    }
    if env_name not in aliases:
        raise ValueError(
            f"Unsupported OG-MARL environment {env_name!r}; expected one of {sorted(aliases)}"
        )
    return aliases[env_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an OG-MARL .vlt dataset to MADiff NumPy arrays."
    )
    parser.add_argument("--vault-root", type=Path, required=True, default="/home/lotus/lotus/lhh/offline_datasets/Vaults")
    parser.add_argument("--source", default="og_marl")
    parser.add_argument("--env", dest="env_name", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--output-root", type=Path, default="/home/lotus/lotus/lhh/offline_datasets/OG-MARL")
    parser.add_argument(
        "--no-add-agent-id",
        action="store_true",
        help="Keep Vault observations unchanged instead of prefixing one-hot agent ids.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--drop-incomplete-tail",
        action="store_true",
        help=(
            "Drop the unfinished final episode from each batch lane. "
            "The default is to reject incomplete data."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experience, vault_path = load_vault_experience(
        args.vault_root, args.source, args.env_name, args.scenario, args.uid
    )
    outputs, metadata = convert_experience(
        experience,
        add_agent_id=not args.no_add_agent_id,
        drop_incomplete_tail=args.drop_incomplete_tail,
    )
    output_dir = (
        args.output_root.expanduser()
        / _madiff_env_name(args.env_name)
        / args.scenario
        / args.uid
    )
    metadata.update(
        {
            "vault_path": os.fspath(vault_path),
            "vault_uid": args.uid,
            "dataset_source": args.source,
            "environment": args.env_name,
            "scenario": args.scenario,
            "output_directory": os.fspath(output_dir.resolve()),
        }
    )
    write_madiff_cache(outputs, output_dir, metadata, overwrite=args.overwrite)
    print(
        f"Converted {metadata['n_transitions']} transitions in "
        f"{metadata['n_episodes']} episodes to {output_dir}"
    )


if __name__ == "__main__":
    main()
