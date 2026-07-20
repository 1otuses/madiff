import os

import numpy as np

import diffuser.datasets.mpe as mpe


class _FakeUnwrapped:
    n = 4
    max_timestep = 2


class _FakeEnv:
    metadata = {
        "name": "simple_world",
        "data_split": "expert",
    }
    action_space = [object(), object(), object()]
    unwrapped = _FakeUnwrapped()


def test_mpe_sequence_dataset_uses_wrapped_training_agents(tmpdir, monkeypatch):
    seed_path = tmpdir.mkdir("seed_0_data")

    for agent_idx in range(3):
        np.save(os.path.join(str(seed_path), f"obs_{agent_idx}.npy"), np.zeros((2, 5)))
        np.save(os.path.join(str(seed_path), f"acs_{agent_idx}.npy"), np.zeros((2, 2)))
        np.save(os.path.join(str(seed_path), f"rews_{agent_idx}.npy"), np.zeros(2))
        np.save(
            os.path.join(str(seed_path), f"dones_{agent_idx}.npy"),
            np.array([0.0, 1.0]),
        )

    np.save(os.path.join(str(seed_path), "obs_3.npy"), np.zeros((2, 4)))
    np.save(os.path.join(str(seed_path), "acs_3.npy"), np.zeros((2, 2)))
    np.save(os.path.join(str(seed_path), "rews_3.npy"), np.zeros(2))
    np.save(os.path.join(str(seed_path), "dones_3.npy"), np.array([0.0, 1.0]))

    monkeypatch.setattr(mpe, "get_dataset_path", lambda *parts: str(tmpdir))

    episode = next(mpe.sequence_dataset(_FakeEnv(), lambda data: data, seed=0))

    assert episode["observations"].shape == (2, 3, 5)
    assert episode["actions"].shape == (2, 3, 2)
    assert episode["rewards"].shape == (2, 3)
    assert episode["terminals"].shape == (2, 3)
