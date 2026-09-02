import os
from typing import Any, Dict, List, Optional

import gym
import gym.spaces
import numpy as np
from smac.env import StarCraft2Env

from diffuser.datasets.paths import get_dataset_path


class SMAC(gym.Env):
    """Environment wrapper SMAC."""

    metadata = {}

    def __init__(self, map_name: str, add_agent_ids_to_obs: bool = True):
        self._environment = StarCraft2Env(map_name=map_name, obs_last_action=False)
        self._agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self.num_agents = len(self._agents)
        self.num_actions = self._environment.n_actions
        self._done = False
        self.max_episode_length = self._environment.episode_limit
        self.add_agent_ids_to_obs = add_agent_ids_to_obs

        if add_agent_ids_to_obs: # 为obs增加agent_id one-hot编码
            self.one_hot_agent_ids = []
            for i in range(self.num_agents):
                agent_id = np.eye(self.num_agents)[i]
                self.one_hot_agent_ids.append(agent_id)
            self.one_hot_agent_ids = np.stack(self.one_hot_agent_ids, axis=0)

        self.observation_space = [
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    self._environment.get_obs_size() + self.num_agents
                    if add_agent_ids_to_obs
                    else self._environment.get_obs_size(),
                ),
            )
            for _ in range(self.num_agents)
        ] # obs空间
        self.action_space = [
            gym.spaces.Discrete(n=self.num_actions) for _ in range(self.num_agents)
        ] # action空间 离散

    def reset(self):
        """Resets the env."""

        # 重置环境
        self._environment.reset()
        self._done = False

        observation = np.array(self.environment.get_obs())
        if self.add_agent_ids_to_obs:
            observation = np.concatenate([self.one_hot_agent_ids, observation], axis=1)
        return observation

    def step(self, actions: np.ndarray):
        """Steps in env."""

        # 执行环境
        reward, self._done, self._info = self._environment.step(actions)
        reward_n = np.array([reward for _ in range(self.num_agents)]) # 共享奖励
        done_n = np.array([self._done for _ in range(self.num_agents)]) # 共享done

        # Get the next observation
        next_observation = np.array(self._environment.get_obs())
        if self.add_agent_ids_to_obs:
            next_observation = np.concatenate(
                [self.one_hot_agent_ids, next_observation], axis=1
            )
        return next_observation, reward_n, done_n, self._info # o',r,done,info

    def env_done(self) -> bool:
        """Check if env is done."""
        return self._done

    def get_legal_actions(self) -> List:
        """Get legal actions from the environment."""
        # 合法动作
        legal_actions = []
        for i, _ in enumerate(self._agents):
            legal_actions.append(
                np.array(self._environment.get_avail_agent_actions(i), dtype="float32")
            )
        return np.array(legal_actions)

    def get_stats(self) -> Optional[Dict]:
        """Return extra stats to be logged."""
        # 需要额外记录的信息
        return self._environment.get_stats()

    @property
    def agents(self) -> List:
        """Agents still alive in env (not done)."""
        return self._agents

    @property
    def possible_agents(self) -> List:
        """All possible agents in env."""
        return self._agents

    @property
    def environment(self):
        """Returns the wrapped environment."""
        return self._environment

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)


def load_environment(name, **kwargs):
    if type(name) is not str:
        # name is already an environment
        return name

    idx = name.find("-")
    env_name, data_split = name[:idx], name[idx + 1 :]

    env = SMAC(env_name, **kwargs)
    if hasattr(env, "metadata"):
        assert isinstance(env.metadata, dict)
    else:
        env.metadata = {}
    env.metadata["data_split"] = data_split
    env.metadata["name"] = env_name
    env.metadata["global_feats"] = ["states"]
    return env


def sequence_dataset(env, preprocess_fn):
    dataset_path = get_dataset_path(
        "smac",
        env.metadata["name"],
        env.metadata["data_split"],
    ) # 加载数据地址
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset directory not found: {}".format(dataset_path))

    observations = np.load(os.path.join(dataset_path, "obs.npy")) # 状态
    legal_actions = np.load(os.path.join(dataset_path, "legals.npy")) # 合法动作
    rewards = np.load(os.path.join(dataset_path, "rewards.npy")) # 奖励
    actions = np.load(os.path.join(dataset_path, "actions.npy")) # 动作
    path_lengths = np.load(os.path.join(dataset_path, "path_lengths.npy")) # 轨迹长度 episode

    start = 0
    for path_length in path_lengths:
        end = start + path_length
        episode_data = {}
        episode_data["observations"] = observations[start:end]
        episode_data["legal_actions"] = legal_actions[start:end]
        episode_data["rewards"] = rewards[start:end]
        episode_data["actions"] = actions[start:end]
        episode_data["terminals"] = np.zeros(
            (path_length, observations.shape[1]), dtype=bool
        )
        episode_data["terminals"][-1] = True
        yield episode_data
        start = end
