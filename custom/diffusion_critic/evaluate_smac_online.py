"""
SMAC Diffusion Critic 在线评估脚本
==================================
加载预训练的 Diffusion Critic 模型，在 SMAC 环境中做在线 rollout,
统计多 seed 的累计回报均值±方差。

评估流程 (类似 MADiff 的 evaluator.py):
  1. 加载预训练 Diffusion Critic (learns P(R|s))
  2. 在 SMAC 环境中，每个 step:
     - 对当前观测 s_t,用 Diffusion Critic 为每个合法动作 a 采样 value 分布
     - 选择具有最高 E[R|s,a] 的动作
     - 执行动作，获取 s_{t+1}, r_{t+1}
  3. 统计每个 episode 的累计回报

用法:
  python evaluate_smac_online.py --env 3m --n_seeds 5 --n_episodes 20
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from diffusion_critic_ac import DiffusionCriticAC


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_smac_env(map_name: str):
    """加载 SMAC 环境 (使用项目中的 wrapper)."""
    from smac.env import StarCraft2Env
    env = StarCraft2Env(map_name=map_name, obs_last_action=False)
    return env


def build_state_from_obs(obs: np.ndarray, env) -> np.ndarray:
    """
    从 agent 局部观测 + 全局状态构建状态特征 (匹配训练时的 state_dim=147).
    obs: [n_agents, obs_dim]
    """
    n_agents = env.n_agents
    agent_obs_flat = obs.reshape(-1)  # [A*obs_dim]
    try:
        global_state = env.get_state()
    except Exception:
        global_state = np.zeros(48, dtype=np.float32)
    return np.concatenate([agent_obs_flat, global_state], axis=-1).astype(np.float32)


def select_action_ac(model, actor, obs, env, legal_actions, device):
    """使用 Actor 选择动作 (AC 版本)."""
    state = build_state_from_obs(obs, env)
    s_t = torch.from_numpy(state).unsqueeze(0).float().to(device)
    with torch.no_grad():
        actions, _ = actor.sample_action(s_t, deterministic=True)
    return actions[0].cpu().numpy().astype(int)


def run_episode_ac(actor, env, device):
    """Actor-Critic 在线评估: 运行一个 episode."""
    env.reset()
    obs = np.array(env.get_obs())
    done = False
    total_return = 0.0
    while not done:
        state = build_state_from_obs(obs, env)
        s_t = torch.from_numpy(state).unsqueeze(0).float().to(device)
        with torch.no_grad():
            actions, _ = actor.sample_action(s_t, deterministic=True)
        action_list = actions[0].cpu().numpy().astype(int).tolist()
        reward, done, _ = env.step(action_list)
        obs = np.array(env.get_obs())
        total_return += reward if isinstance(reward, (int, float)) else sum(reward)
    return total_return


def evaluate_online(
    model_path: str,
    actor_path: str,
    map_name: str,
    n_seeds: int,
    n_episodes: int,
    state_dim: int,
    n_agents: int,
    n_actions: int,
    device: str,
):
    """多 seed 在线评估 (Actor-Critic 版)."""
    from smac.env import StarCraft2Env

    actor = MLPActor(state_dim, n_agents, n_actions, hidden_dim=256)
    state_dict = torch.load(actor_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.to(device)
    actor.eval()

    all_returns = []
    seed_returns = defaultdict(list)

    for seed_idx in range(n_seeds):
        seed = 100 * (seed_idx + 1)
        set_seed(seed)
        env = StarCraft2Env(map_name=map_name, obs_last_action=False)
        print(f"  Seed {seed}: ", end="", flush=True)
        for ep in range(n_episodes):
            ret = run_episode_ac(actor, env, device)
            seed_returns[seed].append(ret)
            all_returns.append(ret)
            if (ep + 1) % 5 == 0:
                print(".", end="", flush=True)
        env.close()
        print(f" {np.mean(seed_returns[seed]):.1f}")

    all_r = np.array(all_returns)
    overall_mean = float(all_r.mean())
    overall_std = float(all_r.std())
    result = f"{overall_mean:.2f} ± {overall_std:.2f}"

    seed_stats = {}
    for seed, rets in seed_returns.items():
        arr = np.array(rets)
        seed_stats[seed] = {"mean": float(arr.mean()), "std": float(arr.std())}

    return {"overall_mean": overall_mean, "overall_std": overall_std,
            "final_result": result, "seed_stats": seed_stats, "all_returns": all_r.tolist()}


def main():
    parser = argparse.ArgumentParser(description="SMAC Online Eval (AC version)")
    parser.add_argument("--env", type=str, default="3m")
    parser.add_argument("--model", type=str, default="runs/models/critic_ac_3m-Good.pt")
    parser.add_argument("--actor", type=str, default="runs/models/actor_ac_3m-Good.pt")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="runs/eval_smac")
    args = parser.parse_args()

    results = evaluate_online(
        model_path=args.model, actor_path=args.actor,
        map_name=args.env, n_seeds=args.n_seeds, n_episodes=args.n_episodes,
        state_dim=147, n_agents=3, n_actions=9, device=args.device,
    )

    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, f"online_eval_{args.env}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n✅ {results['final_result']}\n   Saved: {json_path}")


if __name__ == "__main__":
    from diffusion_critic_ac import MLPActor
    main()
