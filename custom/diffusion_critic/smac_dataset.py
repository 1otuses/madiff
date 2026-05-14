"""
SMAC 离线数据集 → (state, return) 对提取器
==========================================
从 MADiff 的 SMAC .npy 文件中加载 episode 数据，
计算每个 timestep 的 return-to-go 作为价值标签。

数据格式:
  obs.npy:          [total_steps, n_agents, obs_dim]
  rewards.npy:      [total_steps, n_agents]
  path_lengths.npy: [n_episodes,]
  states.npy:       [total_steps, global_state_dim]
  actions.npy:      [total_steps, n_agents]  (discrete, int64)
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path


def load_smac_returns(
    data_dir: str,
    discount: float = 0.99,
    use_global_state: bool = True,
    use_agent_obs: bool = True,
    n_agents: int = None,
    max_episodes: int = None,
) -> tuple:
    """
    从 SMAC 离线数据加载 (state, return) 数据集。

    参数:
      data_dir: SMAC 数据目录。支持两种格式:
                - 完整路径: 'diffuser/datasets/data/smac/3m/Good'
                - 简写: '3m-Good' (自动转换为 '3m/Good')
    """
    import os as _os
    data_dir_str = str(data_dir)
    # project_root = madiff/  (custom/diffusion_critic/smac_dataset.py 的上三级)
    project_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.realpath(__file__))))
    if not _os.path.isabs(data_dir_str):
        # 支持简写 '3m-Good' → 'diffuser/datasets/data/smac/3m/Good'
        env_name = data_dir_str.split("/")[-1]
        if "-" in env_name and "/" not in env_name:
            scenario, quality = env_name.split("-", 1)
            data_dir = Path(project_root) / "diffuser/datasets/data/smac" / scenario / quality
        else:
            data_dir = Path(project_root) / data_dir_str
    else:
        data_dir = Path(data_dir_str)

    if not data_dir.exists():
        raise FileNotFoundError(f"SMAC data directory not found: {data_dir}")

    # 1. 加载所有数据
    obs = np.load(data_dir / "obs.npy")            # [T, A, obs_dim]
    rewards = np.load(data_dir / "rewards.npy")    # [T, A]
    path_lengths = np.load(data_dir / "path_lengths.npy")  # [E]
    states = np.load(data_dir / "states.npy")      # [T, global_dim]
    actions = np.load(data_dir / "actions.npy")    # [T, A]

    total_steps, n_agents_file, obs_dim = obs.shape
    if n_agents is None:
        n_agents = n_agents_file
    assert n_agents == n_agents_file, f"n_agents mismatch: {n_agents} vs {n_agents_file}"

    print(f"[SMAC Dataset] {data_dir.name}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Episodes: {len(path_lengths):,}")
    print(f"  Agents: {n_agents}, Obs dim: {obs_dim}")
    print(f"  Actions: discrete ({actions.max()+1})")
    print(f"  Global state dim: {states.shape[-1]}")
    print(f"  Discount: {discount}")

    # 2. 构建状态特征
    state_parts = []
    if use_agent_obs:
        # 拼接所有 agent 的观测: [T, A*obs_dim]
        agent_obs_flat = obs.reshape(total_steps, -1)  # [T, A*obs_dim]
        state_parts.append(agent_obs_flat)
    if use_global_state:
        state_parts.append(states)  # [T, global_dim]

    state_features = np.concatenate(state_parts, axis=-1)  # [T, D]
    state_dim = state_features.shape[-1]
    print(f"  State feature dim: {state_dim}")

    # 3. 按 episode 计算 return-to-go
    n_episodes = min(len(path_lengths), max_episodes or len(path_lengths))

    all_returns = []
    all_states = []

    pos = 0
    for ep_idx in range(n_episodes):
        ep_len = path_lengths[ep_idx]

        # 提取该 episode 的 reward 和 state
        ep_rewards = rewards[pos:pos + ep_len]       # [L, A]
        ep_states = state_features[pos:pos + ep_len]  # [L, D]

        # 计算 return-to-go: G_t = Σ_{k=t}^{L-1} γ^{k-t} · r_k
        # 使用 sum of team rewards (shared reward 假设)
        team_rewards = ep_rewards.sum(axis=-1)  # [L] — SMAC 中 reward 共享

        returns_to_go = np.zeros(ep_len, dtype=np.float32)
        cum_return = 0.0
        for t in reversed(range(ep_len)):
            cum_return = team_rewards[t] + discount * cum_return
            returns_to_go[t] = cum_return

        all_returns.append(returns_to_go)
        all_states.append(ep_states)

        pos += ep_len

    # 4. 拼接所有 episode
    all_returns = np.concatenate(all_returns, axis=0)   # [N]
    all_states = np.concatenate(all_states, axis=0)     # [N, D]

    # 5. 做简单的统计归一化（return 缩放到合理范围）
    return_mean = all_returns.mean()
    return_std = all_returns.std() + 1e-6

    print(f"  Total samples: {len(all_returns):,}")
    print(f"  Return range: [{all_returns.min():.2f}, {all_returns.max():.2f}]")
    print(f"  Return mean: {return_mean:.2f}, std: {return_std:.2f}")

    info = {
        "state_dim": state_dim,
        "n_agents": n_agents,
        "obs_dim": obs_dim,
        "n_episodes": n_episodes,
        "return_mean": float(return_mean),
        "return_std": float(return_std),
        "return_min": float(all_returns.min()),
        "return_max": float(all_returns.max()),
    }

    return all_states, all_returns, info


def create_smac_dataloader(
    data_dir: str,
    batch_size: int = 256,
    discount: float = 0.99,
    max_episodes: int = None,
    device: str = "cpu",
) -> tuple:
    """便捷函数：创建 SMAC DataLoader。"""
    states, returns, info = load_smac_returns(
        data_dir=data_dir,
        discount=discount,
        max_episodes=max_episodes,
    )

    dataset = TensorDataset(
        torch.from_numpy(states).float(),
        torch.from_numpy(returns).float(),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, info
