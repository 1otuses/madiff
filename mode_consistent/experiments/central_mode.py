"""P2 中央离散 mode 模型的训练、checkpoint 与审计评估协议。"""

import hashlib
import json
import os
import platform
import random
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import yaml

from mode_consistent.data import load_episode_store
from mode_consistent.data.offline import UnlabeledEpisodeView
from mode_consistent.evaluation.metrics import (
    adjusted_rand_index,
    all_assignments,
    assignment_success,
    best_mode_mapping,
    normalized_mutual_information,
    terminal_assignment,
)
from mode_consistent.models import CentralModeVQVAE
from mode_consistent.pipeline import TrajectoryStandardizer


LOSS_NAMES = (
    "loss",
    "reconstruction_loss",
    "codebook_loss",
    "commitment_loss",
)
CHECKPOINT_PATTERN = re.compile(r"state_(\d+)\.pt$")


@dataclass(frozen=True)
class CentralTrainingData:
    """P2 训练进程允许读取的最小数据集合。"""

    view: UnlabeledEpisodeView
    scenario_ids: Optional[np.ndarray] = None
    collector_ids: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CentralWindowData:
    """由完整 episode 生成、并保留来源关系的定长联合轨迹窗口。"""

    view: UnlabeledEpisodeView
    source_episode_indices: np.ndarray
    start_steps: np.ndarray


def sliding_window_view(
    view: UnlabeledEpisodeView,
    window_horizon: int,
    window_stride: int = 1,
) -> CentralWindowData:
    """提取只包含完整有效时间步的滑动窗口。"""

    if not 1 <= window_horizon <= view.horizon:
        raise ValueError("window_horizon 必须位于 [1, episode horizon] 内")
    if window_stride < 1:
        raise ValueError("window_stride 必须为正整数")

    episode_indices = []
    start_steps = []
    for episode_index in range(view.n_episodes):
        for start in range(0, view.horizon - window_horizon + 1, window_stride):
            end = start + window_horizon
            if np.all(view.mask[episode_index, start:end] > 0):
                episode_indices.append(episode_index)
                start_steps.append(start)
    if not episode_indices:
        raise ValueError("当前数据无法生成一个完整有效的滑动窗口")

    episode_indices = np.asarray(episode_indices, dtype=np.int64)
    start_steps = np.asarray(start_steps, dtype=np.int64)
    offsets = np.arange(window_horizon, dtype=np.int64)[None, :]
    time_indices = start_steps[:, None] + offsets
    return CentralWindowData(
        view=UnlabeledEpisodeView(
            observations=view.observations[episode_indices[:, None], time_indices],
            actions=view.actions[episode_indices[:, None], time_indices],
            mask=view.mask[episode_indices[:, None], time_indices],
        ),
        source_episode_indices=episode_indices,
        start_steps=start_steps,
    )


def central_trajectory_view(
    view: UnlabeledEpisodeView,
    trajectory_scope: str,
    window_horizon: int,
    window_stride: int,
) -> CentralWindowData:
    """按中央模型协议构造轨迹输入。

    full_episode 要求数据 horizon 与配置完全一致，从而保证每个
    episode 只产生一个团队 code；sliding_window 仅供历史实验复现。
    """

    if trajectory_scope == "full_episode":
        if window_horizon != view.horizon:
            raise ValueError("full_episode 要求 window_horizon 等于数据 horizon")
        if window_stride != 1:
            raise ValueError("full_episode 的 window_stride 必须为 1")
    elif trajectory_scope != "sliding_window":
        raise ValueError("trajectory_scope 必须为 full_episode 或 sliding_window")
    return sliding_window_view(view, window_horizon, window_stride)


def train_central_experiment(
    config: Mapping,
    run_dir: Path,
    device: torch.device,
) -> Dict:
    """训练 P2 中央模型；该函数不会读取奖励或任何审计标签。"""

    config = dict(config)
    _validate_train_config(config)
    run_dir = Path(run_dir)
    dataset_path = Path(config["dataset_path"]).expanduser().resolve()
    dataset_sha256 = _file_sha256(dataset_path)
    data = _load_central_training_data(
        dataset_path, config.get("split_strategy", "scenario_grouped")
    )
    split = _build_three_way_split(data, config)
    split_sha256 = _split_sha256(split)
    standardizer = TrajectoryStandardizer.fit(data.view.subset(split["train_indices"]))
    episode_views = {
        name: standardizer.transform(data.view.subset(split[f"{name}_indices"]))
        for name in ("train", "validation", "test")
    }
    windows = {
        name: central_trajectory_view(
            view,
            str(config["trajectory_scope"]),
            int(config["window_horizon"]),
            int(config["window_stride"]),
        )
        for name, view in episode_views.items()
    }

    model_config = {
        "n_agents": data.view.n_agents,
        "observation_dim": data.view.observation_dim,
        "action_dim": data.view.action_dim,
        "n_modes": int(config["n_modes"]),
        "hidden_dim": int(config["hidden_dim"]),
        "latent_dim": int(config["latent_dim"]),
        "agent_embedding_dim": int(config["agent_embedding_dim"]),
        "vq_beta": float(config["vq_beta"]),
        "vq_loss_design": str(config["vq_loss_design"]),
    }
    _set_seed(int(config["seed"]))
    model = CentralModeVQVAE(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    batch_rng = np.random.RandomState(int(config["seed"]))

    checkpoint_dir = run_dir / "checkpoint"
    start_step = 0
    if bool(config.get("continue_training", False)):
        latest = _latest_checkpoint(checkpoint_dir)
        if latest is not None:
            checkpoint = torch.load(latest, map_location=device)
            _validate_checkpoint_lineage(
                checkpoint,
                dataset_sha256=dataset_sha256,
                split_sha256=split_sha256,
                model_config=model_config,
                trajectory_scope=str(config["trajectory_scope"]),
                window_horizon=int(config["window_horizon"]),
                window_stride=int(config["window_stride"]),
            )
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            batch_rng.set_state(checkpoint["rng_state"]["batch_numpy"])
            torch.set_rng_state(checkpoint["rng_state"]["torch_cpu"].cpu())
            if device.type == "cuda" and checkpoint["rng_state"]["torch_cuda"]:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in checkpoint["rng_state"]["torch_cuda"]]
                )
            random.setstate(checkpoint["rng_state"]["python"])
            start_step = int(checkpoint["step"])
    elif _latest_checkpoint(checkpoint_dir) is not None:
        raise FileExistsError(
            f"{checkpoint_dir} 已存在 checkpoint；继续训练需显式设置 continue_training"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _save_resolved_config(run_dir / "resolved_config.yaml", config)
    np.savez_compressed(run_dir / "split.npz", **split)
    provenance = _build_provenance(
        config=config,
        dataset_path=dataset_path,
        dataset_sha256=dataset_sha256,
        split_sha256=split_sha256,
        data=data,
        split=split,
        device=device,
    )
    _write_json(run_dir / "provenance.json", provenance)

    n_steps = int(config["n_train_steps"])
    if start_step > n_steps:
        raise ValueError("n_train_steps 不能小于已恢复 checkpoint 的 step")
    batch_size = int(config["batch_size"])
    log_freq = int(config["log_freq"])
    save_freq = int(config["save_freq"])
    metrics_path = run_dir / "train_metrics.jsonl"
    model.train()
    final_losses = None
    for step in range(start_step + 1, n_steps + 1):
        indices = batch_rng.randint(
            0,
            windows["train"].view.n_episodes,
            size=min(batch_size, windows["train"].view.n_episodes),
        )
        observations, actions, mask = _tensor_batch(
            windows["train"].view, indices, device
        )
        output = model(observations, actions, mask)
        optimizer.zero_grad()
        output["loss"].backward()
        optimizer.step()
        final_losses = {name: float(output[name].detach().cpu()) for name in LOSS_NAMES}

        if step == 1 or step % log_freq == 0 or step == n_steps:
            _append_jsonl(metrics_path, {"step": step, **final_losses})
        if step % save_freq == 0 or step == n_steps:
            _save_checkpoint(
                checkpoint_dir / f"state_{step}.pt",
                step=step,
                model=model,
                optimizer=optimizer,
                config=config,
                model_config=model_config,
                standardizer=standardizer,
                split=split,
                dataset_sha256=dataset_sha256,
                split_sha256=split_sha256,
                batch_rng=batch_rng,
            )

    if final_losses is None:
        final_losses = {name: float("nan") for name in LOSS_NAMES}
    evidence = {
        name: evaluate_unlabeled_partition(
            model, window, int(config.get("eval_batch_size", batch_size)), device
        )
        for name, window in windows.items()
    }
    summary = {
        "stage": "central",
        "status": "training_complete",
        "review_status": "pending_user_discussion",
        "step": n_steps,
        "seed": int(config["seed"]),
        "split_seed": int(config["split_seed"]),
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
        "final_batch_losses": final_losses,
        "unlabeled_evidence": evidence,
    }
    _write_json(run_dir / "train_summary.json", summary)
    return summary


def evaluate_central_experiment(
    log_dir: Path,
    load_step: int,
    device: torch.device,
    batch_size: int = 256,
    overwrite: bool = False,
) -> Dict:
    """加载冻结 checkpoint，并在 validation/test 上生成待讨论的审计证据。"""

    log_dir = Path(log_dir).expanduser().resolve()
    result_path = log_dir / "results" / f"step_{load_step}.json"
    if result_path.exists() and not overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))

    checkpoint_path = log_dir / "checkpoint" / f"state_{load_step}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("stage") != "central":
        raise ValueError("checkpoint stage 不是 central")

    config = checkpoint["config"]
    dataset_path = Path(config["dataset_path"]).expanduser().resolve()
    dataset_sha256 = _file_sha256(dataset_path)
    if dataset_sha256 != checkpoint["dataset_sha256"]:
        raise ValueError("当前数据 SHA-256 与训练 checkpoint 不一致")

    store = load_episode_store(dataset_path)
    standardizer = TrajectoryStandardizer(**checkpoint["standardizer"])
    split = checkpoint["split"]
    if _split_sha256(split) != checkpoint["split_sha256"]:
        raise ValueError("checkpoint 中的数据 split 已损坏或被修改")

    model = CentralModeVQVAE(**_compatible_model_config(checkpoint["model_config"])).to(
        device
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    partitions = {}
    codes_by_partition = {}
    windows_by_partition = {}
    for name in ("validation", "test"):
        indices = np.asarray(split[f"{name}_indices"], dtype=np.int64)
        episode_view = standardizer.transform(store.training_view(indices))
        window = central_trajectory_view(
            episode_view,
            str(config.get("trajectory_scope", "sliding_window")),
            int(config["window_horizon"]),
            int(config["window_stride"]),
        )
        partition = evaluate_unlabeled_partition(
            model, window, batch_size, device, include_codes=True
        )
        codes = np.asarray(partition.pop("codes"), dtype=np.int64)
        partition.update(
            _episode_code_statistics(
                codes,
                window.source_episode_indices,
                window.start_steps,
                model.n_modes,
            )
        )
        partition.update(
            evaluate_shuffled_code_reconstruction(
                model,
                window.view,
                codes,
                batch_size,
                device,
                seed=int(config["seed"]) + (0 if name == "validation" else 1),
            )
        )
        partitions[name] = partition
        codes_by_partition[name] = codes
        windows_by_partition[name] = window

    mapping = None
    if store.true_modes is not None:
        validation_indices = np.asarray(split["validation_indices"], dtype=np.int64)
        validation_targets = store.true_modes[validation_indices][
            windows_by_partition["validation"].source_episode_indices
        ]
        mapping = best_mode_mapping(
            codes_by_partition["validation"], validation_targets
        )

    for name in ("validation", "test"):
        indices = np.asarray(split[f"{name}_indices"], dtype=np.int64)
        codes = codes_by_partition[name]
        audit = {}
        if store.true_modes is not None:
            targets = np.asarray(store.true_modes[indices], dtype=np.int64)[
                windows_by_partition[name].source_episode_indices
            ]
            aligned = np.asarray([mapping.get(int(code), -1) for code in codes])
            audit.update(
                {
                    "mode_nmi": normalized_mutual_information(codes, targets),
                    "mode_ari": adjusted_rand_index(codes, targets),
                    "mapped_accuracy": float((aligned == targets).mean()),
                }
            )
        if store.quality_ids is not None:
            quality = np.asarray(store.quality_ids[indices], dtype=np.int64)[
                windows_by_partition[name].source_episode_indices
            ]
            audit["code_quality_nmi"] = normalized_mutual_information(codes, quality)
        source_indices = indices[windows_by_partition[name].source_episode_indices]
        if store.collector_ids is not None:
            collectors = np.asarray(store.collector_ids[source_indices], dtype=np.int64)
            audit["collector_nmi"] = normalized_mutual_information(codes, collectors)
        returns = (
            (store.rewards * store.mask[..., None]).sum(axis=1).mean(axis=1)
        )[source_indices]
        audit["code_return_means"] = [
            float(returns[codes == code].mean()) if np.any(codes == code) else None
            for code in range(model.n_modes)
        ]
        if (store.metadata or {}).get("task") == "simple_spread":
            assignments = all_assignments(store.n_agents)
            terminal_observations = store.next_observations[indices, -1]
            episode_assignment_ids = np.asarray(
                [
                    assignments.index(terminal_assignment(observation))
                    for observation in terminal_observations
                ],
                dtype=np.int64,
            )
            assignment_ids = episode_assignment_ids[
                windows_by_partition[name].source_episode_indices
            ]
            audit["terminal_assignment_nmi"] = normalized_mutual_information(
                codes, assignment_ids
            )
            audit["terminal_assignment_counts"] = np.bincount(
                episode_assignment_ids, minlength=len(assignments)
            ).tolist()
            audit["terminal_success_rate"] = float(
                np.mean(
                    [
                        assignment_success(observation)
                        for observation in terminal_observations
                    ]
                )
            )
        partitions[name]["audit"] = audit

    result = {
        "stage": "central",
        "checkpoint_step": int(load_step),
        "checkpoint": str(checkpoint_path),
        "dataset_sha256": dataset_sha256,
        "split_sha256": checkpoint["split_sha256"],
        "window_horizon": int(config["window_horizon"]),
        "window_stride": int(config["window_stride"]),
        "trajectory_scope": str(config.get("trajectory_scope", "sliding_window")),
        "mapping_partition": "validation" if mapping is not None else None,
        "mode_mapping": mapping,
        "review_status": "pending_user_discussion",
        "documentation_updated": False,
        "partitions": partitions,
    }
    _write_json(result_path, result)
    return result


def scenario_three_way_split(
    scenario_ids: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """按 scenario 整体生成互斥的 train/validation/test 三分划分。"""

    scenario_ids = np.asarray(scenario_ids)
    if scenario_ids.ndim != 1:
        raise ValueError("scenario_ids 必须是一维数组")
    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("validation_fraction 和 test_fraction 必须为正数")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction 必须小于 1")
    scenarios = np.unique(scenario_ids)
    if len(scenarios) < 3:
        raise ValueError("正式 P2 划分至少需要三个不同 scenario")

    scenarios = np.random.RandomState(seed).permutation(scenarios)
    n_validation = max(1, int(round(len(scenarios) * validation_fraction)))
    n_test = max(1, int(round(len(scenarios) * test_fraction)))
    if n_validation + n_test >= len(scenarios):
        raise ValueError("scenario 数量不足以生成非空三分划分")
    validation_scenarios = np.sort(scenarios[:n_validation])
    test_scenarios = np.sort(scenarios[n_validation : n_validation + n_test])
    train_scenarios = np.sort(scenarios[n_validation + n_test :])

    return {
        "train_indices": np.flatnonzero(np.isin(scenario_ids, train_scenarios)),
        "validation_indices": np.flatnonzero(
            np.isin(scenario_ids, validation_scenarios)
        ),
        "test_indices": np.flatnonzero(np.isin(scenario_ids, test_scenarios)),
        "train_scenarios": train_scenarios,
        "validation_scenarios": validation_scenarios,
        "test_scenarios": test_scenarios,
    }


def collector_stratified_three_way_split(
    collector_ids: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """在每个 collector 内独立划分 episode，避免把来源当作监督标签。"""

    collector_ids = np.asarray(collector_ids)
    if collector_ids.ndim != 1:
        raise ValueError("collector_ids 必须是一维数组")
    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("validation_fraction 和 test_fraction 必须为正数")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction 必须小于 1")

    collectors = np.unique(collector_ids)
    rng = np.random.RandomState(seed)
    partitions = {name: [] for name in ("train", "validation", "test")}
    for collector in collectors:
        indices = rng.permutation(np.flatnonzero(collector_ids == collector))
        n_validation = max(1, int(round(len(indices) * validation_fraction)))
        n_test = max(1, int(round(len(indices) * test_fraction)))
        if n_validation + n_test >= len(indices):
            raise ValueError("每个 collector 至少需要三个 episode 才能三分划分")
        partitions["validation"].append(indices[:n_validation])
        partitions["test"].append(indices[n_validation : n_validation + n_test])
        partitions["train"].append(indices[n_validation + n_test :])

    return {
        **{
            f"{name}_indices": np.sort(np.concatenate(chunks))
            for name, chunks in partitions.items()
        },
        "train_collectors": collectors,
        "validation_collectors": collectors,
        "test_collectors": collectors,
    }


@torch.no_grad()
def evaluate_unlabeled_partition(
    model: CentralModeVQVAE,
    window: CentralWindowData,
    batch_size: int,
    device: torch.device,
    include_codes: bool = False,
) -> Dict:
    """计算不依赖审计标签的损失、hard/soft usage 与 perplexity。"""

    model.eval()
    loss_sums = {name: 0.0 for name in LOSS_NAMES}
    hard_usage = np.zeros(model.n_modes, dtype=np.int64)
    soft_usage = np.zeros(model.n_modes, dtype=np.float64)
    view = window.view
    timestep_squared_error = np.zeros(view.horizon, dtype=np.float64)
    timestep_count = np.zeros(view.horizon, dtype=np.float64)
    codes = []
    for start in range(0, view.n_episodes, batch_size):
        indices = np.arange(start, min(start + batch_size, view.n_episodes))
        batch = _tensor_batch(view, indices, device)
        output = model(*batch)
        count = len(indices)
        for name in LOSS_NAMES:
            loss_sums[name] += float(output[name].detach().cpu()) * count
        batch_codes = output["codes"].detach().cpu().numpy()
        hard_usage += np.bincount(batch_codes, minlength=model.n_modes)
        soft_usage += output["code_probabilities"].detach().cpu().numpy().sum(axis=0)
        squared_error = (
            (output["predicted_actions"] - batch[1])
            .square()
            .sum(dim=(2, 3))
            .detach()
            .cpu()
            .numpy()
        )
        batch_mask = view.mask[indices]
        timestep_squared_error += (squared_error * batch_mask).sum(axis=0)
        timestep_count += batch_mask.sum(axis=0) * view.n_agents * view.action_dim
        codes.append(batch_codes)

    hard_probabilities = hard_usage / max(hard_usage.sum(), 1)
    soft_probabilities = soft_usage / max(soft_usage.sum(), 1.0)
    result = {
        **{name: value / view.n_episodes for name, value in loss_sums.items()},
        "n_windows": int(view.n_episodes),
        "n_source_episodes": int(np.unique(window.source_episode_indices).size),
        "window_horizon": int(view.horizon),
        "hard_code_usage": hard_usage.tolist(),
        "hard_perplexity": _perplexity(hard_probabilities),
        "active_codes": int(np.count_nonzero(hard_usage)),
        "soft_code_usage": soft_probabilities.tolist(),
        "soft_perplexity": _perplexity(soft_probabilities),
        "first_action_mse": float(
            timestep_squared_error[0] / max(timestep_count[0], 1.0)
        ),
        "action_mse_by_offset": (
            timestep_squared_error / np.maximum(timestep_count, 1.0)
        ).tolist(),
    }
    all_codes = np.concatenate(codes).astype(np.int64)
    same_episode = window.source_episode_indices[1:] == window.source_episode_indices[:-1]
    temporal_pairs = int(same_episode.sum())
    result["temporal_code_pairs"] = temporal_pairs
    result["temporal_code_agreement"] = (
        float((all_codes[1:][same_episode] == all_codes[:-1][same_episode]).mean())
        if temporal_pairs
        else None
    )
    if include_codes:
        result["codes"] = all_codes.tolist()
    return result


@torch.no_grad()
def evaluate_shuffled_code_reconstruction(
    model: CentralModeVQVAE,
    view: UnlabeledEpisodeView,
    codes: np.ndarray,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Dict:
    """打乱窗口与 code 的对应关系，同时保持 code 边际频率不变。"""

    shuffled = np.random.RandomState(seed).permutation(np.asarray(codes))
    squared_error = np.zeros(view.horizon, dtype=np.float64)
    counts = np.zeros(view.horizon, dtype=np.float64)
    for start in range(0, view.n_episodes, batch_size):
        positions = np.arange(start, min(start + batch_size, view.n_episodes))
        observations, actions, _ = _tensor_batch(view, positions, device)
        batch_codes = torch.as_tensor(shuffled[positions], device=device).long()
        predicted = model.decode_actions(observations, batch_codes)
        errors = (predicted - actions).square().sum(dim=(2, 3)).cpu().numpy()
        batch_mask = view.mask[positions]
        squared_error += (errors * batch_mask).sum(axis=0)
        counts += batch_mask.sum(axis=0) * view.n_agents * view.action_dim
    by_offset = squared_error / np.maximum(counts, 1.0)
    return {
        "code_shuffled_reconstruction_loss": float(
            squared_error.sum() / max(counts.sum(), 1.0)
        ),
        "code_shuffled_first_action_mse": float(by_offset[0]),
        "code_shuffled_action_mse_by_offset": by_offset.tolist(),
    }


def _episode_code_statistics(
    codes: np.ndarray,
    source_episode_indices: np.ndarray,
    start_steps: np.ndarray,
    n_modes: int,
) -> Dict:
    entropies = []
    modal_fractions = []
    unique_counts = []
    for episode_index in np.unique(source_episode_indices):
        usage = np.bincount(
            codes[source_episode_indices == episode_index], minlength=n_modes
        )
        probabilities = usage[usage > 0] / usage.sum()
        entropy = -(probabilities * np.log(probabilities)).sum()
        entropies.append(float(entropy / np.log(n_modes)))
        modal_fractions.append(float(usage.max() / usage.sum()))
        unique_counts.append(int(np.count_nonzero(usage)))
    return {
        "code_nmi_with_window_start": normalized_mutual_information(
            codes, start_steps
        ),
        "mean_normalized_episode_code_entropy": float(np.mean(entropies)),
        "mean_episode_modal_code_fraction": float(np.mean(modal_fractions)),
        "mean_unique_codes_per_episode": float(np.mean(unique_counts)),
        "single_code_episode_fraction": float(
            np.mean(np.asarray(unique_counts) == 1)
        ),
    }


def _load_central_training_data(
    path: Path, split_strategy: str = "scenario_grouped"
) -> CentralTrainingData:
    # 有意只读取训练张量和划分字段，避免接触 reward、true mode 或 quality。
    with np.load(path, allow_pickle=False) as data:
        split_field = {
            "scenario_grouped": "scenario_ids",
            "collector_stratified": "collector_ids",
        }.get(split_strategy)
        if split_field is None:
            raise ValueError(f"未知 split_strategy：{split_strategy}")
        required = {"observations", "actions", "mask", split_field}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"正式 P2 数据缺少字段：{sorted(missing)}")
        view = UnlabeledEpisodeView(
            observations=data["observations"].astype(np.float32),
            actions=data["actions"].astype(np.float32),
            mask=data["mask"].astype(np.float32),
        )
        split_ids = data[split_field].copy()
    if split_ids.shape != (view.n_episodes,):
        raise ValueError(f"{split_field} 必须为每个 episode 提供一个编号")
    return CentralTrainingData(
        view=view,
        scenario_ids=split_ids if split_field == "scenario_ids" else None,
        collector_ids=split_ids if split_field == "collector_ids" else None,
    )


def _build_three_way_split(data: CentralTrainingData, config: Mapping) -> Dict:
    arguments = {
        "validation_fraction": float(config["validation_fraction"]),
        "test_fraction": float(config["test_fraction"]),
        "seed": int(config["split_seed"]),
    }
    strategy = config.get("split_strategy", "scenario_grouped")
    if strategy == "scenario_grouped":
        return scenario_three_way_split(data.scenario_ids, **arguments)
    if strategy == "collector_stratified":
        return collector_stratified_three_way_split(data.collector_ids, **arguments)
    raise ValueError(f"未知 split_strategy：{strategy}")


def _validate_train_config(config: Mapping) -> None:
    required = {
        "stage",
        "dataset_path",
        "seed",
        "split_seed",
        "validation_fraction",
        "test_fraction",
        "n_modes",
        "hidden_dim",
        "latent_dim",
        "agent_embedding_dim",
        "vq_beta",
        "vq_loss_design",
        "trajectory_scope",
        "window_horizon",
        "window_stride",
        "n_train_steps",
        "batch_size",
        "learning_rate",
        "log_freq",
        "save_freq",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"P2 训练配置缺少字段：{sorted(missing)}")
    if config["stage"] != "central":
        raise ValueError("当前只开放 stage=central；其他阶段需讨论确认后实现")
    if config["vq_loss_design"] != "vomasd":
        raise ValueError("新 P2 训练只允许 vq_loss_design=vomasd")
    if float(config["vq_beta"]) < 0.0:
        raise ValueError("vq_beta 不能为负数")
    if config["trajectory_scope"] not in {"full_episode", "sliding_window"}:
        raise ValueError("trajectory_scope 必须为 full_episode 或 sliding_window")
    for name in (
        "n_train_steps",
        "batch_size",
        "log_freq",
        "save_freq",
        "window_horizon",
        "window_stride",
    ):
        if int(config[name]) < 1:
            raise ValueError(f"{name} 必须为正整数")


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    model: CentralModeVQVAE,
    optimizer: torch.optim.Optimizer,
    config: Mapping,
    model_config: Mapping,
    standardizer: TrajectoryStandardizer,
    split: Mapping,
    dataset_sha256: str,
    split_sha256: str,
    batch_rng: np.random.RandomState,
) -> None:
    checkpoint = {
        "schema_version": 1,
        "stage": "central",
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": dict(config),
        "model_config": dict(model_config),
        "standardizer": standardizer.state_dict(),
        "split": {name: np.asarray(values) for name, values in split.items()},
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
        "rng_state": {
            "python": random.getstate(),
            "batch_numpy": batch_rng.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def _validate_checkpoint_lineage(
    checkpoint: Mapping,
    *,
    dataset_sha256: str,
    split_sha256: str,
    model_config: Mapping,
    trajectory_scope: str,
    window_horizon: int,
    window_stride: int,
) -> None:
    if checkpoint.get("stage") != "central":
        raise ValueError("不能从非 central checkpoint 恢复 P2")
    if checkpoint.get("dataset_sha256") != dataset_sha256:
        raise ValueError("恢复训练时数据 SHA-256 不一致")
    if checkpoint.get("split_sha256") != split_sha256:
        raise ValueError("恢复训练时数据 split 不一致")
    if checkpoint.get("model_config") != dict(model_config):
        raise ValueError("恢复训练时模型配置不一致")
    checkpoint_config = checkpoint.get("config", {})
    if (
        str(checkpoint_config.get("trajectory_scope", "sliding_window"))
        != trajectory_scope
        or int(checkpoint_config.get("window_horizon", -1)) != window_horizon
        or int(checkpoint_config.get("window_stride", -1)) != window_stride
    ):
        raise ValueError("恢复训练时轨迹输入协议不一致")


def _compatible_model_config(model_config: Mapping) -> Dict:
    """将旧 H=5 checkpoint 显式标记为 standard VQ 语义后读取。"""

    compatible = dict(model_config)
    if "commitment_cost" in compatible:
        if "vq_beta" in compatible or "vq_loss_design" in compatible:
            raise ValueError("checkpoint 同时包含新旧 VQ 配置字段")
        compatible["vq_beta"] = float(compatible.pop("commitment_cost"))
        compatible["vq_loss_design"] = "standard"
    return compatible


def _latest_checkpoint(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    checkpoints = []
    for path in directory.iterdir():
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(None, None))[1]


def _tensor_batch(view, indices: Sequence[int], device: torch.device):
    return tuple(
        torch.as_tensor(array[indices], device=device, dtype=torch.float32)
        for array in (view.observations, view.actions, view.mask)
    )


def _perplexity(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    return float(np.exp(-(positive * np.log(positive)).sum()))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_sha256(split: Mapping) -> str:
    digest = hashlib.sha256()
    for name in sorted(split):
        values = np.asarray(split[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _build_provenance(
    *,
    config: Mapping,
    dataset_path: Path,
    dataset_sha256: str,
    split_sha256: str,
    data: CentralTrainingData,
    split: Mapping,
    device: torch.device,
) -> Dict:
    return {
        "dataset": {
            "path": str(dataset_path),
            "size_bytes": dataset_path.stat().st_size,
            "sha256": dataset_sha256,
            "episodes": data.view.n_episodes,
            "horizon": data.view.horizon,
            "agents": data.view.n_agents,
            "observation_dim": data.view.observation_dim,
            "action_dim": data.view.action_dim,
        },
        "split": {
            "strategy": config.get("split_strategy", "scenario_grouped"),
            "seed": int(config["split_seed"]),
            "sha256": split_sha256,
            **(
                {
                    "train_collectors": len(split["train_collectors"]),
                    "validation_collectors": len(split["validation_collectors"]),
                    "test_collectors": len(split["test_collectors"]),
                }
                if config.get("split_strategy") == "collector_stratified"
                else {
                    "train_scenarios": len(split["train_scenarios"]),
                    "validation_scenarios": len(split["validation_scenarios"]),
                    "test_scenarios": len(split["test_scenarios"]),
                }
            ),
        },
        "seed": int(config["seed"]),
        "config_sha256": hashlib.sha256(
            json.dumps(dict(config), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "command": shlex.join(sys.argv),
    }


def _git_output(*arguments: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *arguments], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _save_resolved_config(path: Path, config: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, record: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
