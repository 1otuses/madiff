import collections
import os

import gym
import numpy as np
import torch
from ddpg_agent import DDPGAgent

from diffuser.datasets.paths import get_dataset_path


class StackWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_n = np.array(obs)
        reward_n = np.array(reward)
        done_n = np.array(done)
        return obs_n, reward_n, done_n, info

    def reset(self):
        obs = self.env.reset()
        obs_n = np.array(obs)
        return obs_n


class PretrainedPreyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, scenario_name: str):
        # 对于simple_tag/simple_world场景，需要加载预训练的 prey 模型
        assert scenario_name in ["simple_tag", "simple_world"], scenario_name
        self.scenario_name = scenario_name
        self.prey_load_path = get_dataset_path(
            "mpe", scenario_name, "pretrained_adv_model.pt"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prey = None

        super().__init__(env)

        self.prey_obs = None
        # rewrite env attributes to remove prey
        self.n = env.n - 1
        self.action_space = env.action_space[:-1]
        self.observation_space = env.observation_space[:-1]

    def _load_prey(self):
        if self.prey is not None:
            return
        if not os.path.exists(self.prey_load_path):
            raise FileNotFoundError(
                "Missing pretrained prey policy for MPE "
                f"{self.scenario_name}: {self.prey_load_path}. "
                "Offline dataset construction does not need this file, but online "
                "rollout/evaluation for simple_tag/simple_world does. Copy "
                "pretrained_adv_model.pt into this directory or set "
                "MADIFF_OFFLINE_DATA_ROOT to the offline dataset root that contains it."
            )

        self.prey = DDPGAgent(
            num_in_pol=self.env.observation_space[-1].shape[0],
            num_out_pol=self.env.action_space[-1].shape[0],
            num_in_critic=self.env.observation_space[-1].shape[0]
            + self.env.action_space[-1].shape[0],
        )
        self.prey.to(self.device)

        prey_params = torch.load(
            self.prey_load_path,
            map_location=self.device,
        )["agent_params"][-1]
        self.prey.load_params_without_optims(prey_params)
        self.prey.policy.eval()
        self.prey.target_policy.eval()

    def step(self, action):
        self._load_prey()
        prey_obs = torch.tensor(
            self.prey_obs, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        prey_action = self.prey.step(prey_obs, explore=False)[0].detach().cpu().numpy()
        action = [*action, prey_action]
        obs, reward, done, info = self.env.step(action)
        self.prey_obs = obs[-1]
        return obs[:-1], reward[:-1], done[:-1], info

    def reset(self):
        obs = self.env.reset()
        self.prey_obs = obs[-1]
        return obs[:-1]


def make_env(scenario_name, benchmark=False, **kwargs):
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    """
    import multiagent.scenarios as scenarios
    from multiagent.environment import MultiAgentEnv

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
            **kwargs,
        )
    else:
        env = MultiAgentEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation, **kwargs
        )
    if scenario_name in ["simple_tag", "simple_world"]:
        env = PretrainedPreyWrapper(env, scenario_name)
    return StackWrapper(env)


def load_environment(name, **kwargs):
    if type(name) != str:
        # name is already an environment
        return name

    idx = name.find("-")
    env_name, data_split = name[:idx], name[idx + 1 :]

    env = make_env(env_name, **kwargs)
    if hasattr(env, "metadata"):
        assert isinstance(env.metadata, dict)
    else:
        env.metadata = {}
    env.metadata["data_split"] = data_split
    env.metadata["name"] = env_name
    env.metadata["global_feats"] = []
    return env


def sequence_dataset(env, preprocess_fn, seed: int = None):
    """
    Returns an iterator through trajectories.
    Args:
        env: An MultiAgentEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """

    dataset_path = get_dataset_path(
        "mpe",
        env.metadata["name"],
        env.metadata["data_split"],
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset directory not found: {}".format(dataset_path))

    if seed is None:
        print("\n USE ALL SEED DATASET \n") # 将所有种子数据集都加载
        seed_dirs = os.listdir(dataset_path)
    else:
        print(f"\n USE SEED {seed} DATASET \n")
        seed_dirs = [f"seed_{seed}_data"]

    n_agents = len(env.action_space)
    for idx, seed_dir in enumerate(seed_dirs):
        seed_path = os.path.join(dataset_path, seed_dir)
        if not os.path.isdir(seed_path):
            continue

        missing_files = [
            f"obs_{agent_idx}.npy"
            for agent_idx in range(n_agents)
            if not os.path.exists(os.path.join(seed_path, f"obs_{agent_idx}.npy"))
        ]
        if missing_files:
            has_tfrecord = any(
                filename.endswith(".tfrecord") for filename in os.listdir(seed_path)
            )
            hint = (
                " 当前目录包含 .tfrecord 文件，请先转换为 MADiff 需要的 "
                "obs_*.npy/actions_*.npy/rewards_*.npy 格式。"
                if has_tfrecord
                else ""
            )
            raise FileNotFoundError(
                f"MPE dataset seed directory is missing {missing_files[0]}: "
                f"{seed_path}.{hint}"
            )

        observations = np.stack(
            [
                np.load(os.path.join(seed_path, "obs_{}.npy".format(agent_idx)))
                for agent_idx in range(n_agents)
            ],
            axis=1,
        ) # 合并后的obs
        actions = np.stack(
            [
                np.load(os.path.join(seed_path, "acs_{}.npy".format(agent_idx)))
                for agent_idx in range(n_agents)
            ],
            axis=1,
        ) # 合并后的动作
        rewards = np.stack(
            [
                np.load(os.path.join(seed_path, "rews_{}.npy".format(agent_idx)))
                for agent_idx in range(n_agents)
            ],
            axis=1,
        ) # 合并后的奖励
        dones = np.stack(
            [
                np.load(os.path.join(seed_path, "dones_{}.npy".format(agent_idx)))
                for agent_idx in range(n_agents)
            ],
            axis=1,
        ) # 合并后的done
        data_ = collections.defaultdict(list)
        for obs, act, rew, done in zip(observations, actions, rewards, dones):
            data_["observations"].append(obs)
            data_["actions"].append(act)
            data_["rewards"].append(rew)
            data_["terminals"].append(done)

            if done.all() or len(data_["observations"]) == env.unwrapped.max_timestep:
                data_["timeouts"] = np.zeros_like(data_["terminals"])
                if len(data_["observations"]) == env.unwrapped.max_timestep:
                    data_["terminals"][-1][:] = 0.0
                    data_["timeouts"][-1][:] = 1.0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                yield episode_data
                data_ = collections.defaultdict(list)


if __name__ == "__main__":
    env = make_env("simple_tag")

    obs = env.reset()
    for _ in range(5):
        obs, reward, done, info = env.step(
            [act_space.sample() for act_space in env.action_space]
        )
