"""P2 central model 的无 code、oracle code 与 KMeans 最小对照。"""

import json
import os
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from scipy.cluster.vq import kmeans2

from mode_consistent.data import load_episode_store
from mode_consistent.evaluation.metrics import (
    adjusted_rand_index,
    best_mode_mapping,
    normalized_mutual_information,
)
from mode_consistent.models import NoCodeActionModel, OracleModeActionModel
from mode_consistent.pipeline import TrajectoryStandardizer

from .central_mode import (
    _build_provenance,
    _build_three_way_split,
    _file_sha256,
    _load_central_training_data,
    _save_resolved_config,
    _set_seed,
    _split_sha256,
    _tensor_batch,
    _write_json,
    central_trajectory_view,
    evaluate_central_experiment,
)

NEURAL_BASELINES = {"no_code", "oracle_code"}
BASELINES = NEURAL_BASELINES | {"kmeans"}


def train_central_validation_baseline(
    config: Mapping,
    run_dir: Path,
    device: torch.device,
) -> Dict:
    """训练一个 P2 对照组；只有 oracle_code 可读取审计 mode。"""

    config = dict(config)
    model_kind = config.get("model_kind")
    if model_kind not in BASELINES:
        raise ValueError(f"未知 P2 对照类型：{model_kind}")
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
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
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
    provenance["model_kind"] = model_kind
    _write_json(run_dir / "provenance.json", provenance)

    if model_kind == "kmeans":
        summary = _fit_kmeans(
            config,
            run_dir,
            windows,
            standardizer,
            split,
            dataset_sha256,
            split_sha256,
        )
    else:
        summary = _fit_neural_baseline(
            config,
            run_dir,
            device,
            windows,
            standardizer,
            split,
            dataset_path,
            dataset_sha256,
            split_sha256,
        )
    _write_json(run_dir / "train_summary.json", summary)
    return summary


def evaluate_central_validation_baseline(
    log_dir: Path,
    load_step: int,
    device: torch.device,
    batch_size: int = 256,
    overwrite: bool = False,
) -> Dict:
    """评估单个对照 checkpoint，不使用 test 标签重新拟合映射。"""

    log_dir = Path(log_dir).expanduser().resolve()
    result_path = log_dir / "results" / f"step_{load_step}.json"
    if result_path.exists() and not overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_path = log_dir / "checkpoint" / f"state_{load_step}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_kind = checkpoint["model_kind"]
    dataset_path = Path(checkpoint["config"]["dataset_path"]).expanduser().resolve()
    if _file_sha256(dataset_path) != checkpoint["dataset_sha256"]:
        raise ValueError("当前数据与对照 checkpoint 不一致")
    if _split_sha256(checkpoint["split"]) != checkpoint["split_sha256"]:
        raise ValueError("对照 checkpoint 中的 split 校验失败")

    store = load_episode_store(dataset_path)
    standardizer = TrajectoryStandardizer(**checkpoint["standardizer"])
    episode_views = {
        name: standardizer.transform(
            store.training_view(checkpoint["split"][f"{name}_indices"])
        )
        for name in ("validation", "test")
    }
    windows = {
        name: central_trajectory_view(
            view,
            str(checkpoint["config"].get("trajectory_scope", "sliding_window")),
            int(checkpoint["config"]["window_horizon"]),
            int(checkpoint["config"]["window_stride"]),
        )
        for name, view in episode_views.items()
    }
    if model_kind == "kmeans":
        result = _evaluate_kmeans(checkpoint, store, windows)
    else:
        model = _build_neural_model(model_kind, checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        labels = store.true_modes if model_kind == "oracle_code" else None
        result = {
            "partitions": {
                name: _evaluate_reconstruction(
                    model,
                    window.view,
                    batch_size,
                    device,
                    None
                    if labels is None
                    else labels[checkpoint["split"][f"{name}_indices"]][
                        window.source_episode_indices
                    ],
                )
                for name, window in windows.items()
            }
        }
        for name, window in windows.items():
            result["partitions"][name].update(
                n_windows=int(window.view.n_episodes),
                n_source_episodes=int(
                    np.unique(window.source_episode_indices).size
                ),
                window_horizon=int(window.view.horizon),
            )
    result.update(
        {
            "stage": "central",
            "model_kind": model_kind,
            "checkpoint_step": int(load_step),
            "checkpoint": str(checkpoint_path),
            "dataset_sha256": checkpoint["dataset_sha256"],
            "split_sha256": checkpoint["split_sha256"],
            "window_horizon": int(checkpoint["config"]["window_horizon"]),
            "window_stride": int(checkpoint["config"]["window_stride"]),
            "trajectory_scope": str(
                checkpoint["config"].get("trajectory_scope", "sliding_window")
            ),
            "privileged": model_kind == "oracle_code",
            "review_status": "pending_user_discussion",
            "documentation_updated": False,
        }
    )
    _write_json(result_path, result)
    return result


def evaluate_central_comparison(
    runs: Mapping,
    output_path: Path,
    device: torch.device,
    batch_size: int = 256,
    overwrite: bool = False,
) -> Dict:
    """聚合四个冻结运行，计算 VQ 相对于无 code 和 oracle 的改善。"""

    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() and not overwrite:
        return json.loads(output_path.read_text(encoding="utf-8"))
    required = {"vq", "no_code", "oracle_code", "kmeans"}
    if set(runs) != required:
        raise ValueError(f"runs 必须严格包含：{sorted(required)}")

    evidence = {
        "vq": evaluate_central_experiment(
            runs["vq"]["log_dir"],
            int(runs["vq"]["load_step"]),
            device,
            batch_size,
            overwrite,
        )
    }
    for name in ("no_code", "oracle_code", "kmeans"):
        evidence[name] = evaluate_central_validation_baseline(
            runs[name]["log_dir"],
            int(runs[name]["load_step"]),
            device,
            batch_size,
            overwrite,
        )

    hashes = {value["dataset_sha256"] for value in evidence.values()}
    split_hashes = {value["split_sha256"] for value in evidence.values()}
    if len(hashes) != 1 or len(split_hashes) != 1:
        raise ValueError("四个运行的数据或 split lineage 不一致")
    window_protocols = {
        (
            value.get("trajectory_scope", "sliding_window"),
            value["window_horizon"],
            value["window_stride"],
        )
        for value in evidence.values()
    }
    if len(window_protocols) != 1:
        raise ValueError("四个运行的滑动窗口协议不一致")
    usefulness = {
        partition: _mode_usefulness(evidence, partition)
        for partition in ("validation", "test")
    }
    result = {
        "stage": "central_comparison",
        "review_status": "pending_user_discussion",
        "documentation_updated": False,
        "dataset_sha256": hashes.pop(),
        "split_sha256": split_hashes.pop(),
        "evidence": evidence,
        "mode_usefulness": usefulness,
    }
    _write_json(output_path, result)
    return result


def evaluate_unlabeled_central_comparison(
    runs: Mapping,
    output_path: Path,
    device: torch.device,
    batch_size: int = 256,
    overwrite: bool = False,
) -> Dict:
    """比较无标签 VQ、no-code 与冻结 VQ 的 code-shuffle 诊断。"""

    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() and not overwrite:
        return json.loads(output_path.read_text(encoding="utf-8"))
    if set(runs) != {"vq", "no_code"}:
        raise ValueError("无标签比较必须严格包含 vq 和 no_code")
    evidence = {
        "vq": evaluate_central_experiment(
            runs["vq"]["log_dir"],
            int(runs["vq"]["load_step"]),
            device,
            batch_size,
            overwrite,
        ),
        "no_code": evaluate_central_validation_baseline(
            runs["no_code"]["log_dir"],
            int(runs["no_code"]["load_step"]),
            device,
            batch_size,
            overwrite,
        ),
    }
    if len({value["dataset_sha256"] for value in evidence.values()}) != 1:
        raise ValueError("VQ 与 no-code 的数据 lineage 不一致")
    if len({value["split_sha256"] for value in evidence.values()}) != 1:
        raise ValueError("VQ 与 no-code 的 split lineage 不一致")
    if len(
        {
            (
                value.get("trajectory_scope", "sliding_window"),
                value["window_horizon"],
                value["window_stride"],
            )
            for value in evidence.values()
        }
    ) != 1:
        raise ValueError("VQ 与 no-code 的滑动窗口协议不一致")

    usefulness = {
        partition: _unlabeled_usefulness(evidence, partition)
        for partition in ("validation", "test")
    }
    result = {
        "stage": "central_unlabeled_comparison",
        "review_status": "pending_user_discussion",
        "documentation_updated": False,
        "dataset_sha256": evidence["vq"]["dataset_sha256"],
        "split_sha256": evidence["vq"]["split_sha256"],
        "evidence": evidence,
        "usefulness": usefulness,
    }
    _write_json(output_path, result)
    return result


def _unlabeled_usefulness(evidence: Mapping, partition: str) -> Dict:
    vq = evidence["vq"]["partitions"][partition]
    no_code = evidence["no_code"]["partitions"][partition]
    return {
        "reconstruction_improvement_fraction": float(
            (no_code["reconstruction_loss"] - vq["reconstruction_loss"])
            / max(no_code["reconstruction_loss"], 1e-12)
        ),
        "first_action_improvement_fraction": float(
            (no_code["first_action_mse"] - vq["first_action_mse"])
            / max(no_code["first_action_mse"], 1e-12)
        ),
        "shuffle_degradation_fraction": float(
            (
                vq["code_shuffled_reconstruction_loss"]
                - vq["reconstruction_loss"]
            )
            / max(vq["reconstruction_loss"], 1e-12)
        ),
    }


def _fit_neural_baseline(
    config,
    run_dir,
    device,
    windows,
    standardizer,
    split,
    dataset_path,
    dataset_sha256,
    split_sha256,
):
    model_kind = config["model_kind"]
    model_config = {
        "n_agents": windows["train"].view.n_agents,
        "observation_dim": windows["train"].view.observation_dim,
        "action_dim": windows["train"].view.action_dim,
        "hidden_dim": int(config["hidden_dim"]),
        "agent_embedding_dim": int(config["agent_embedding_dim"]),
    }
    if model_kind == "oracle_code":
        model_config.update(
            n_modes=int(config["n_modes"]), latent_dim=int(config["latent_dim"])
        )
    _set_seed(int(config["seed"]))
    model = _build_neural_model(model_kind, model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    batch_rng = np.random.RandomState(int(config["seed"]))
    labels = None
    if model_kind == "oracle_code":
        # oracle 是唯一允许读取 true_modes 的特权诊断分支。
        with np.load(dataset_path, allow_pickle=False) as raw:
            if "true_modes" not in raw.files:
                raise ValueError("oracle_code 需要 true_modes 审计字段")
            labels = raw["true_modes"].astype(np.int64)
        if labels.min() < 0 or labels.max() >= int(config["n_modes"]):
            raise ValueError("true_modes 超出配置的 codebook 范围")

    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "train_metrics.jsonl"
    n_steps = int(config["n_train_steps"])
    batch_size = int(config["batch_size"])
    for step in range(1, n_steps + 1):
        positions = batch_rng.randint(
            0,
            windows["train"].view.n_episodes,
            size=min(batch_size, windows["train"].view.n_episodes),
        )
        batch = _tensor_batch(windows["train"].view, positions, device)
        if labels is None:
            output = model(*batch)
        else:
            original_indices = split["train_indices"][
                windows["train"].source_episode_indices[positions]
            ]
            modes = torch.as_tensor(labels[original_indices], device=device).long()
            output = model(*batch, modes)
        optimizer.zero_grad()
        output["loss"].backward()
        optimizer.step()
        if step == 1 or step % int(config["log_freq"]) == 0 or step == n_steps:
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(output["loss"].detach().cpu()),
                            "reconstruction_loss": float(
                                output["reconstruction_loss"].detach().cpu()
                            ),
                        }
                    )
                    + "\n"
                )
        if step % int(config["save_freq"]) == 0 or step == n_steps:
            _save_baseline_checkpoint(
                checkpoint_dir / f"state_{step}.pt",
                config,
                model_kind,
                model_config,
                model.state_dict(),
                optimizer.state_dict(),
                standardizer,
                split,
                dataset_sha256,
                split_sha256,
                step,
            )

    return {
        "stage": "central",
        "model_kind": model_kind,
        "status": "training_complete",
        "review_status": "pending_user_discussion",
        "step": n_steps,
        "privileged": model_kind == "oracle_code",
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
    }


def _fit_kmeans(
    config,
    run_dir,
    windows,
    standardizer,
    split,
    dataset_sha256,
    split_sha256,
):
    features = _trajectory_features(windows["train"].view)
    centers, codes = kmeans2(
        features,
        int(config["n_modes"]),
        iter=int(config.get("kmeans_iterations", 100)),
        minit="++",
        missing="raise",
        seed=int(config["seed"]),
    )
    checkpoint = {
        "schema_version": 1,
        "stage": "central",
        "model_kind": "kmeans",
        "step": 0,
        "config": dict(config),
        "centers": centers.astype(np.float32),
        "standardizer": standardizer.state_dict(),
        "split": {name: np.asarray(value) for name, value in split.items()},
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
    }
    checkpoint_path = run_dir / "checkpoint" / "state_0.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    usage = np.bincount(codes, minlength=int(config["n_modes"]))
    return {
        "stage": "central",
        "model_kind": "kmeans",
        "status": "training_complete",
        "review_status": "pending_user_discussion",
        "step": 0,
        "privileged": False,
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
        "train_code_usage": usage.tolist(),
    }


def _save_baseline_checkpoint(
    path,
    config,
    model_kind,
    model_config,
    model_state,
    optimizer_state,
    standardizer,
    split,
    dataset_sha256,
    split_sha256,
    step,
):
    checkpoint = {
        "schema_version": 1,
        "stage": "central",
        "model_kind": model_kind,
        "step": int(step),
        "config": dict(config),
        "model_config": dict(model_config),
        "model": model_state,
        "optimizer": optimizer_state,
        "standardizer": standardizer.state_dict(),
        "split": {name: np.asarray(value) for name, value in split.items()},
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def _evaluate_reconstruction(
    model,
    view,
    batch_size,
    device,
    modes: Sequence[int] = None,
):
    model.eval()
    squared_error = np.zeros(view.horizon, dtype=np.float64)
    counts = np.zeros(view.horizon, dtype=np.float64)
    modes = None if modes is None else np.asarray(modes, dtype=np.int64)
    for start in range(0, view.n_episodes, batch_size):
        positions = np.arange(start, min(start + batch_size, view.n_episodes))
        batch = _tensor_batch(view, positions, device)
        if modes is None:
            output = model(*batch)
        else:
            batch_modes = torch.as_tensor(modes[positions], device=device).long()
            output = model(*batch, batch_modes)
        errors = (
            (output["predicted_actions"] - batch[1])
            .square()
            .sum(dim=(2, 3))
            .cpu()
            .numpy()
        )
        batch_mask = view.mask[positions]
        squared_error += (errors * batch_mask).sum(axis=0)
        counts += batch_mask.sum(axis=0) * view.n_agents * view.action_dim
    by_timestep = squared_error / np.maximum(counts, 1.0)
    return {
        "reconstruction_loss": float(squared_error.sum() / max(counts.sum(), 1.0)),
        "first_action_mse": float(by_timestep[0]),
        "action_mse_by_offset": by_timestep.tolist(),
    }


def _evaluate_kmeans(checkpoint, store, windows):
    codes = {
        name: _nearest_centers(
            _trajectory_features(window.view), checkpoint["centers"]
        )
        for name, window in windows.items()
    }
    validation_indices = checkpoint["split"]["validation_indices"]
    mapping = None
    if store.true_modes is not None:
        mapping = best_mode_mapping(
            codes["validation"],
            store.true_modes[validation_indices][
                windows["validation"].source_episode_indices
            ],
        )
    partitions = {}
    n_modes = int(checkpoint["config"]["n_modes"])
    for name in ("validation", "test"):
        current = codes[name]
        usage = np.bincount(current, minlength=n_modes)
        probabilities = usage[usage > 0] / max(usage.sum(), 1)
        audit = {}
        indices = checkpoint["split"][f"{name}_indices"]
        if store.true_modes is not None:
            targets = store.true_modes[indices][windows[name].source_episode_indices]
            aligned = np.asarray([mapping.get(int(code), -1) for code in current])
            audit = {
                "mode_nmi": normalized_mutual_information(current, targets),
                "mode_ari": adjusted_rand_index(current, targets),
                "mapped_accuracy": float((aligned == targets).mean()),
            }
        partitions[name] = {
            "n_windows": int(len(current)),
            "n_source_episodes": int(
                np.unique(windows[name].source_episode_indices).size
            ),
            "window_horizon": int(windows[name].view.horizon),
            "hard_code_usage": usage.tolist(),
            "active_codes": int(np.count_nonzero(usage)),
            "hard_perplexity": float(
                np.exp(-(probabilities * np.log(probabilities)).sum())
            ),
            "audit": audit,
        }
    return {
        "dataset_sha256": checkpoint["dataset_sha256"],
        "split_sha256": checkpoint["split_sha256"],
        "mapping_partition": "validation" if mapping is not None else None,
        "mode_mapping": mapping,
        "partitions": partitions,
    }


def _trajectory_features(view):
    sequence = np.concatenate([view.observations, view.actions], axis=-1)
    sequence = sequence.reshape(view.n_episodes, -1)
    return np.concatenate([sequence, view.mask], axis=1).astype(np.float32)


def _nearest_centers(features, centers, batch_size=512):
    predictions = []
    center_norm = np.square(centers).sum(axis=1)
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size]
        distances = (
            np.square(batch).sum(axis=1, keepdims=True)
            + center_norm[None]
            - 2.0 * batch @ centers.T
        )
        predictions.append(distances.argmin(axis=1))
    return np.concatenate(predictions).astype(np.int64)


def _build_neural_model(model_kind, model_config):
    if model_kind == "no_code":
        return NoCodeActionModel(**model_config)
    if model_kind == "oracle_code":
        return OracleModeActionModel(**model_config)
    raise ValueError(f"未知神经网络对照：{model_kind}")


def _mode_usefulness(evidence, partition):
    no_code = evidence["no_code"]["partitions"][partition]
    oracle = evidence["oracle_code"]["partitions"][partition]
    vq = evidence["vq"]["partitions"][partition]
    return {
        "reconstruction_oracle_gap_fraction": _gap_fraction(
            no_code["reconstruction_loss"],
            vq["reconstruction_loss"],
            oracle["reconstruction_loss"],
        ),
        "first_action_oracle_gap_fraction": _gap_fraction(
            no_code["first_action_mse"],
            vq["first_action_mse"],
            oracle["first_action_mse"],
        ),
        "offset_oracle_gap_fraction": [
            _gap_fraction(no_value, vq_value, oracle_value)
            for no_value, vq_value, oracle_value in zip(
                no_code["action_mse_by_offset"],
                vq["action_mse_by_offset"],
                oracle["action_mse_by_offset"],
            )
        ],
    }


def _gap_fraction(no_code, vq, oracle):
    denominator = no_code - oracle
    if denominator <= 1e-12:
        return None
    return float((no_code - vq) / denominator)
