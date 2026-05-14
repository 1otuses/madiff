"""
Toy Environment: Bimodal Return Distribution
=============================================
状态空间: S₀ (初始), S_trap (陷阱), S_safe (安全), S_treasure (宝藏)
动作空间: S₀ 处只有 a_forward
动态:   S₀ --a_forward--> 50% S_trap (R=-10) / 50% S_treasure (R=+10)
        所有终止状态仅自环, R=0
折扣因子: γ = 1.0 (无折扣)

价值分布: V(S₀) ~ 0.5·δ(R+10) + 0.5·δ(R-10)  → 完美双峰分布
"""

import numpy as np


class BimodalToyEnv:
    """
    简单的双峰回报环境，用于验证 Diffusion Critic 是否能学习多峰价值分布。
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.n_states = 4
        self.n_actions = 1  # only a_forward at S₀

        # 状态索引
        self.S0 = 0        # 初始状态
        self.S_TRAP = 1    # 陷阱（终止, R=-10）
        self.S_SAFE = 2    # 安全（终止, R=0）
        self.S_TREASURE = 3 # 宝藏（终止, R=+10）

        self.reset()

    def reset(self):
        self.state = self.S0
        self.done = False
        return self._get_obs()

    def step(self, action):
        """执行动作，返回 (obs, reward, done, info)。"""
        if self.done:
            return self._get_obs(), 0.0, True, {}

        if self.state == self.S0:
            # S₀ → 50% 陷阱 / 50% 宝藏
            if self.rng.rand() < 0.5:
                self.state = self.S_TRAP
                reward = -10.0
            else:
                self.state = self.S_TREASURE
                reward = 10.0
            self.done = True
        else:
            # 终止状态自环
            reward = 0.0

        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        """返回 one-hot 编码的状态向量 [4]."""
        obs = np.zeros(self.n_states, dtype=np.float32)
        obs[self.state] = 1.0
        return obs

    def generate_dataset(self, n_episodes: int = 10000):
        """
        生成离线数据集: [(state_onehot, return), ...]
        对 S₀ 收集完整 episode 回报，对终止状态回报=0。
        """
        data = []
        for _ in range(n_episodes):
            obs = self.reset()
            total_return = 0.0
            done = False
            while not done:
                # 唯一动作
                obs, reward, done, _ = self.step(0)
                total_return += reward
            # 记录 S₀ 的 one-hot 和 episode 回报
            s0_onehot = np.zeros(self.n_states, dtype=np.float32)
            s0_onehot[self.S0] = 1.0
            data.append((s0_onehot, total_return))

            # 对终止状态，回报为 0（它们没有后续奖励）
            for s_idx in [self.S_TRAP, self.S_SAFE, self.S_TREASURE]:
                s_onehot = np.zeros(self.n_states, dtype=np.float32)
                s_onehot[s_idx] = 1.0
                data.append((s_onehot, 0.0))

        return data


def generate_dataset(n_episodes: int = 10000, seed: int = 42) -> tuple:
    """便捷函数：生成 (states, returns) 的 numpy 数组。"""
    env = BimodalToyEnv(seed=seed)
    raw = env.generate_dataset(n_episodes)
    states = np.stack([r[0] for r in raw], axis=0)   # [N, 4]
    returns = np.array([r[1] for r in raw], dtype=np.float32)  # [N]
    return states, returns
