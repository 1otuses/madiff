"""分别衡量 mode 恢复、数据支持度和任务成功率的评估指标。"""

import itertools
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def best_mode_mapping(predicted: np.ndarray, target: np.ndarray) -> Dict[int, int]:
    predicted = np.asarray(predicted, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    predicted_values, predicted_inverse = np.unique(predicted, return_inverse=True)
    target_values, target_inverse = np.unique(target, return_inverse=True)
    counts = np.zeros((len(predicted_values), len(target_values)), dtype=np.int64)
    np.add.at(counts, (predicted_inverse, target_inverse), 1)
    rows, columns = linear_sum_assignment(-counts)
    return {
        int(predicted_values[row]): int(target_values[column])
        for row, column in zip(rows, columns)
    }


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """计算两个硬聚类划分之间的调整兰德指数。"""

    labels_a = np.asarray(labels_a, dtype=np.int64)
    labels_b = np.asarray(labels_b, dtype=np.int64)
    if labels_a.ndim != 1 or labels_b.shape != labels_a.shape:
        raise ValueError("labels must be 1-D arrays of equal length")
    if len(labels_a) < 2:
        return 1.0

    _, a_inverse = np.unique(labels_a, return_inverse=True)
    _, b_inverse = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros((a_inverse.max() + 1, b_inverse.max() + 1), dtype=np.float64)
    np.add.at(contingency, (a_inverse, b_inverse), 1.0)
    sum_combinations = lambda counts: (counts * (counts - 1.0) / 2.0).sum()
    sum_ij = sum_combinations(contingency)
    sum_a = sum_combinations(contingency.sum(axis=1))
    sum_b = sum_combinations(contingency.sum(axis=0))
    total = len(labels_a) * (len(labels_a) - 1.0) / 2.0
    expected = sum_a * sum_b / total
    maximum = 0.5 * (sum_a + sum_b)
    if maximum == expected:
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


def calibration_metrics(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> Dict:
    """使用等宽置信度分箱计算期望校准误差。"""

    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidences.shape != correct.shape or confidences.ndim != 1:
        raise ValueError("confidences and correct must be equal-length 1-D arrays")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(confidences, boundaries[1:-1]), 0, n_bins - 1)
    error = 0.0
    rows = []
    for index in range(n_bins):
        mask = bin_indices == index
        if not mask.any():
            continue
        mean_confidence = float(confidences[mask].mean())
        accuracy = float(correct[mask].mean())
        error += float(mask.sum() * abs(accuracy - mean_confidence))
        rows.append(
            {
                "bin": index,
                "count": int(mask.sum()),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "ece": float(error / len(confidences)),
        "n_samples": int(len(confidences)),
        "bins": rows,
    }


def confidence_coverage_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    thresholds: Sequence[float] = (0.5, 0.7, 0.9, 0.95),
) -> list:
    """计算各置信度阈值下拒绝预测后的覆盖率与准确率。"""

    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    rows = []
    for threshold in thresholds:
        kept = confidences >= threshold
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(kept.mean()),
                "accuracy": (float(correct[kept].mean()) if kept.any() else None),
            }
        )
    return rows


def local_mode_evaluation(
    codes: np.ndarray,
    targets: np.ndarray,
    n_modes: int,
    unknown: int = -1,
) -> Dict:
    """联合评估各 agent 的局部 mode code，并允许低置信度时拒绝预测。

    ``codes`` 的形状为 ``[episodes, agents]``，``unknown`` 表示拒绝预测。
    覆盖率指所有 agent 都给出 code 的 episode 比例，其余指标只在已覆盖
    episode 上统计。
    """

    codes = np.asarray(codes, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if codes.ndim != 2 or codes.shape[1] < 2:
        raise ValueError("codes must have shape [episodes, agents]")
    if targets.shape != (codes.shape[0],):
        raise ValueError("targets must have shape [episodes]")
    covered = np.all(codes != unknown, axis=1)
    covered_codes = codes[covered]
    covered_targets = targets[covered]
    agreement = np.all(covered_codes == covered_codes[:, :1], axis=1)
    consensus = covered_codes[:, 0]
    counts = np.bincount(consensus, minlength=n_modes).astype(np.float64)
    probabilities = counts[counts > 0] / max(counts.sum(), 1.0)
    entropy = (
        float(-(probabilities * np.log(probabilities)).sum() / np.log(n_modes))
        if n_modes > 1 and counts.sum() > 0
        else 0.0
    )
    forced_correct = codes == targets[:, None]
    return {
        "coverage": float(covered.mean()),
        "agreement_rate": float(agreement.mean()) if covered.any() else 0.0,
        "consensus_accuracy": (
            float(np.mean(agreement & (consensus == covered_targets)))
            if covered.any()
            else 0.0
        ),
        "per_agent_accuracy": (
            float((covered_codes == covered_targets[:, None]).mean())
            if covered.any()
            else 0.0
        ),
        "mmr": float(1.0 - agreement.mean()) if covered.any() else 0.0,
        "consensus_entropy": entropy,
        "forced_accuracy": float(forced_correct.mean()),
        "unknown_rate": float((1.0 - covered).mean()),
    }


def normalized_mutual_information(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    joint = np.zeros((predicted.max() + 1, target.max() + 1), dtype=np.float64)
    np.add.at(joint, (predicted, target), 1.0)
    joint /= joint.sum()
    p_pred = joint.sum(axis=1, keepdims=True)
    p_target = joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    mutual_information = (
        joint[nonzero] * np.log(joint[nonzero] / (p_pred @ p_target)[nonzero])
    ).sum()
    h_pred = -(p_pred[p_pred > 0] * np.log(p_pred[p_pred > 0])).sum()
    h_target = -(p_target[p_target > 0] * np.log(p_target[p_target > 0])).sum()
    return float(2.0 * mutual_information / max(h_pred + h_target, 1e-12))


def mode_discovery_metrics(predicted: np.ndarray, target: np.ndarray) -> Dict:
    mapping = best_mode_mapping(predicted, target)
    aligned = np.asarray([mapping.get(int(value), -1) for value in predicted])
    counts = np.bincount(predicted, minlength=int(predicted.max()) + 1)
    probabilities = counts / counts.sum()
    perplexity = np.exp(
        -(
            probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])
        ).sum()
    )
    return {
        "mode_accuracy": float((aligned == target).mean()),
        "mode_nmi": normalized_mutual_information(predicted, target),
        "mode_perplexity": float(perplexity),
        "mode_usage": counts.tolist(),
        "mode_mapping": mapping,
    }


def coordination_code_metrics(
    codes: np.ndarray,
    targets: Optional[np.ndarray] = None,
    n_modes: Optional[int] = None,
) -> Dict[str, float]:
    """衡量各 agent 的局部 mode code 是否形成有效的联合约定。"""

    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2 or codes.shape[1] < 2:
        raise ValueError("codes must have shape [samples, agents] with >= 2 agents")
    if codes.shape[0] == 0:
        raise ValueError("codes must contain at least one sample")
    if n_modes is None:
        n_modes = int(codes.max()) + 1
    if n_modes < 1 or np.any(codes < 0) or np.any(codes >= n_modes):
        raise ValueError("codes must be in [0, n_modes)")

    agreement = np.all(codes == codes[:, :1], axis=1)
    consensus_codes = codes[agreement, 0]
    counts = np.bincount(consensus_codes, minlength=n_modes).astype(np.float64)
    probabilities = counts[counts > 0] / max(counts.sum(), 1.0)
    entropy = (
        float(-(probabilities * np.log(probabilities)).sum() / np.log(n_modes))
        if n_modes > 1
        else 0.0
    )
    metrics = {
        "mode_mismatch_rate": float(1.0 - agreement.mean()),
        "agreement_rate": float(agreement.mean()),
        "consensus_mode_coverage": float(np.count_nonzero(counts) / n_modes),
        "consensus_mode_entropy": entropy,
    }
    if targets is not None:
        targets = np.asarray(targets, dtype=np.int64)
        if targets.shape != (codes.shape[0],):
            raise ValueError("targets must have shape [samples]")
        metrics.update(
            {
                "per_agent_accuracy": float((codes == targets[:, None]).mean()),
                "consensus_accuracy": float(
                    np.mean(agreement & (codes[:, 0] == targets))
                ),
            }
        )
    return metrics


def joint_action_support_distance(
    query_states: np.ndarray,
    query_actions: np.ndarray,
    query_modes: np.ndarray,
    reference_states: np.ndarray,
    reference_actions: np.ndarray,
    reference_modes: np.ndarray,
    state_weight: float = 0.1,
    max_references_per_mode: int = 2048,
    seed: int = 0,
) -> float:
    """近似计算 mode 条件下的最近邻数据支持距离。

    输入应预先归一化。状态距离项用于避免把无关状态下出现过的动作误判为
    当前查询的完美数据支持。
    """

    rng = np.random.RandomState(seed)
    distances = []
    query_actions = query_actions.reshape(len(query_actions), -1)
    reference_actions = reference_actions.reshape(len(reference_actions), -1)
    for mode in np.unique(query_modes):
        query_indices = np.flatnonzero(query_modes == mode)
        reference_indices = np.flatnonzero(reference_modes == mode)
        if not len(reference_indices):
            distances.extend([np.inf] * len(query_indices))
            continue
        if len(reference_indices) > max_references_per_mode:
            reference_indices = rng.choice(
                reference_indices, max_references_per_mode, replace=False
            )
        state_difference = (
            query_states[query_indices, None]
            - reference_states[reference_indices][None]
        )
        action_difference = (
            query_actions[query_indices, None]
            - reference_actions[reference_indices][None]
        )
        squared = state_weight * np.mean(
            np.square(state_difference), axis=-1
        ) + np.mean(np.square(action_difference), axis=-1)
        distances.extend(np.sqrt(squared.min(axis=1)).tolist())
    return float(np.mean(distances))


def _feature_offset(observation: np.ndarray, n_agents: int) -> int:
    if observation.shape[-1] >= 18 + n_agents:
        prefix = observation[..., :n_agents]
        if np.allclose(prefix.sum(axis=-1), 1.0, atol=1e-4):
            return n_agents
    return 0


def landmark_vectors(observation: np.ndarray, n_landmarks: int = 3) -> np.ndarray:
    offset = _feature_offset(observation, observation.shape[-2])
    start = offset + 4
    return observation[..., start : start + 2 * n_landmarks].reshape(
        *observation.shape[:-1], n_landmarks, 2
    )


def terminal_assignment(observation: np.ndarray) -> Tuple[int, ...]:
    vectors = landmark_vectors(observation)
    costs = np.linalg.norm(vectors, axis=-1)
    rows, columns = linear_sum_assignment(costs)
    assignment = np.zeros(observation.shape[-2], dtype=np.int64)
    assignment[rows] = columns
    return tuple(int(value) for value in assignment)


def assignment_success(
    observation: np.ndarray,
    expected: Optional[Sequence[int]] = None,
    distance_threshold: float = 0.25,
) -> bool:
    vectors = landmark_vectors(observation)
    assignment = terminal_assignment(observation)
    distances = np.asarray(
        [
            np.linalg.norm(vectors[agent, landmark])
            for agent, landmark in enumerate(assignment)
        ]
    )
    return bool(
        np.all(distances < distance_threshold)
        and (expected is None or tuple(expected) == assignment)
    )


def instantaneous_assignment(
    observation: np.ndarray, action: np.ndarray
) -> Tuple[int, ...]:
    vectors = landmark_vectors(observation)
    vector_norm = np.linalg.norm(vectors, axis=-1).clip(1e-6)
    action_norm = np.linalg.norm(action, axis=-1, keepdims=True).clip(1e-6)
    similarity = (vectors * action[:, None, :]).sum(-1) / (vector_norm * action_norm)
    return tuple(int(value) for value in similarity.argmax(axis=-1))


def cross_mode_splice_rate(
    observations: np.ndarray,
    actions: np.ndarray,
    expected: Optional[Sequence[int]] = None,
) -> float:
    invalid = 0
    for observation, action in zip(observations, actions):
        assignment = instantaneous_assignment(observation, action)
        is_permutation = len(set(assignment)) == len(assignment)
        invalid += int(
            not is_permutation
            or (expected is not None and tuple(expected) != assignment)
        )
    return invalid / max(len(observations), 1)


def all_assignments(n_agents: int = 3):
    return list(itertools.permutations(range(n_agents)))
