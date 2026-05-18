"""
SMAC 离线数据加载器 — 从 .npy 文件提取 (state, return) 对
=========================================================
对齐 MADiff 的数据加载方式。
"""

import os
import sys
from pathlib import Path

import numpy as np

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_smac_dataset(
    scenario: str,
    quality: str = "Good",
    max_samples: int = None,
    gamma: float = 0.99,
    data_base: str = None,
):
    """
    加载 SMAC 离线数据集，计算每个时间步的 return-to-go。

    Args:
        scenario: "3m", "8m", "2s3z", "5m_vs_6m" 等
        quality: "Good" | "Medium" | "Poor" | "Replay"
        max_samples: 最大样本数 (降采样)
        gamma: 折扣因子
        data_base: 数据根目录 (默认 diffuser/datasets/data/smac)

    Returns:
        states: np.ndarray [N, D] 全局状态
        returns_rtg: np.ndarray [N] return-to-go
        info: dict 包含统计信息和路径长度
    """
    if data_base is None:
        data_base = _PROJECT_ROOT / "diffuser" / "datasets" / "data" / "smac"
    data_dir = Path(data_base) / scenario / quality

    if not data_dir.exists():
        raise FileNotFoundError(f"SMAC 数据目录不存在: {data_dir}")

    # 加载数据
    states = np.load(str(data_dir / "states.npy")).astype(np.float32)     # [T, D]
    rewards = np.load(str(data_dir / "rewards.npy")).astype(np.float32)   # [T, A]
    path_lengths = np.load(str(data_dir / "path_lengths.npy"))           # [n_episodes]

    # 所有 agent 共享 reward → 取第一个 agent
    team_rewards = rewards[:, 0]  # [T]

    # 按 episode 分割计算 return-to-go
    returns_list = []
    start = 0
    for ep_len in path_lengths:
        ep_rewards = team_rewards[start : start + ep_len]  # [L]
        L = len(ep_rewards)
        rtg = np.zeros(L, dtype=np.float32)
        discounted = 0.0
        for t in reversed(range(L)):
            discounted = ep_rewards[t] + gamma * discounted
            rtg[t] = discounted
        returns_list.append(rtg)
        start += ep_len

    returns_rtg = np.concatenate(returns_list, axis=0).astype(np.float32)

    # 降采样
    if max_samples is not None and len(states) > max_samples:
        idx = np.random.choice(len(states), max_samples, replace=False)
        states = states[idx]
        returns_rtg = returns_rtg[idx]

    info = {
        "scenario": scenario,
        "quality": quality,
        "n_agents": rewards.shape[1],
        "state_dim": states.shape[1],
        "total_transitions": len(rewards),
        "loaded_samples": len(states),
        "n_episodes": len(path_lengths),
        "return_mean": float(returns_rtg.mean()),
        "return_std": float(returns_rtg.std()),
        "return_min": float(returns_rtg.min()),
        "return_max": float(returns_rtg.max()),
    }

    print(
        f"[smac_dataset] {scenario}/{quality}: "
        f"{len(states)} samples, state_dim={states.shape[1]}, "
        f"return_mean={info['return_mean']:.2f}±{info['return_std']:.2f}"
    )

    return states, returns_rtg, info


def normalize(states, returns):
    """归一化为零均值单位方差。"""
    s_mean = states.mean(axis=0, keepdims=True)
    s_std = states.std(axis=0, keepdims=True) + 1e-8
    r_mean = returns.mean()
    r_std = returns.std() + 1e-8

    states_norm = (states - s_mean) / s_std
    returns_norm = (returns - r_mean) / r_std

    norm_info = {
        "s_mean": s_mean.squeeze(0),
        "s_std": s_std.squeeze(0),
        "r_mean": r_mean,
        "r_std": r_std,
    }
    return states_norm, returns_norm, norm_info
