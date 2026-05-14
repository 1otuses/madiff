import os

import numpy as np
from multiagent_mujoco.mujoco_multi import MujocoMulti


def load_environment(name):
    if type(name) is not str:
        # name is already an environment
        return name

    idx = name.find("-")
    env_name, data_split = name[:idx], name[idx + 1 :]

    env_kwargs = {
        "agent_obsk": 1,
        "episode_limit": 1000,
        "global_categories": "qvel,qpos",
    }
    if env_name == "4ant":
        env_kwargs["scenario"] = "Ant-v2"
        env_kwargs["agent_conf"] = "4x2"
    elif env_name == "2ant":
        env_kwargs["scenario"] = "Ant-v2"
        env_kwargs["agent_conf"] = "2x4"
    elif env_name == "2halfcheetah":
        env_kwargs["scenario"] = "HalfCheetah-v2"
        env_kwargs["agent_conf"] = "2x3"
    else:
        raise NotImplementedError(
            f"Multi-agent Mujoco environment {env_name} not supported."
        )

    env = MujocoMulti(env_args=env_kwargs, add_agent_ids_to_obs=True)
    if hasattr(env, "metadata"):
        assert isinstance(env.metadata, dict)
    else:
        env.metadata = {}
    env.metadata["data_split"] = data_split
    env.metadata["name"] = env_name
    env.metadata["global_feats"] = []
    return env


def sequence_dataset(env, preprocess_fn):
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "data/mamujoco",
        env.metadata["name"],
        env.metadata["data_split"],
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset directory not found: {}".format(dataset_path))

    observations = np.load(os.path.join(dataset_path, "obs.npy"))  # 加载所有观察
    rewards = np.load(os.path.join(dataset_path, "rewards.npy"))  # 加载所有奖励
    actions = np.load(os.path.join(dataset_path, "actions.npy"))  # 加载所有动作
    path_lengths = np.load(os.path.join(dataset_path, "path_lengths.npy"))  # 加载每个episode的长度

    start = 0
    for path_length in path_lengths:
        end = start + path_length  # 计算当前episode的结束索引
        episode_data = {}
        episode_data["observations"] = observations[start:end]  # 提取当前episode的观察
        episode_data["rewards"] = rewards[start:end]  # 提取当前episode的奖励
        episode_data["actions"] = actions[start:end]  # 提取当前episode的动作
        episode_data["terminals"] = np.zeros(
            (path_length, observations.shape[1]), dtype=bool
        )  # 初始化terminals为False
        episode_data["terminals"][-1] = True  # 最后一个时间步为terminal状态
        yield episode_data
        start = end
