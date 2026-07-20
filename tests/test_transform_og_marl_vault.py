import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.transform_og_marl_vault import convert_experience, write_madiff_cache


def make_experience():
    batch_size, time_size, n_agents, obs_dim = 2, 5, 2, 3
    observations = np.arange(
        batch_size * time_size * n_agents * obs_dim, dtype=np.float32
    ).reshape(batch_size, time_size, n_agents, obs_dim)
    actions = np.arange(batch_size * time_size * n_agents).reshape(
        batch_size, time_size, n_agents
    )
    rewards = np.ones((batch_size, time_size, n_agents), dtype=np.float32)
    terminals = np.zeros_like(rewards)
    truncations = np.zeros_like(rewards)

    terminals[0, 1] = 1
    truncations[0, 4] = 1
    terminals[1, 0] = 1
    truncations[1, 4] = 1

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "truncations": truncations,
        "infos": {
            "legals": np.ones((batch_size, time_size, n_agents, 4), dtype=np.float32),
            "state": np.arange(batch_size * time_size * 6).reshape(
                batch_size, time_size, 6
            ),
        },
    }


def test_convert_vault_experience_preserves_episode_boundaries_and_fields():
    experience = make_experience()
    outputs, metadata = convert_experience(experience)

    np.testing.assert_array_equal(outputs["path_lengths"], [2, 3, 1, 4])
    assert outputs["obs"].shape == (10, 2, 5)
    assert outputs["actions"].shape == (10, 2)
    assert outputs["legals"].shape == (10, 2, 4)
    assert outputs["states"].shape == (10, 6)

    expected_ids = np.broadcast_to(np.eye(2), (10, 2, 2))
    np.testing.assert_array_equal(outputs["obs"][..., :2], expected_ids)
    np.testing.assert_array_equal(
        outputs["obs"][..., 2:], experience["observations"].reshape(10, 2, 3)
    )
    np.testing.assert_array_equal(
        outputs["discounts"][:, 0], [1, 0, 1, 1, 0, 0, 1, 1, 1, 0]
    )
    assert metadata["n_transitions"] == 10
    assert metadata["n_episodes"] == 4
    assert metadata["agent_id"]["added"] is True


def test_convert_rejects_asynchronous_agent_endings():
    experience = make_experience()
    experience["terminals"][0, 1, 1] = 0

    with pytest.raises(ValueError, match="synchronously"):
        convert_experience(experience)


def test_convert_rejects_incomplete_trailing_episode():
    experience = make_experience()
    experience["truncations"][1, 4] = 0

    with pytest.raises(ValueError, match="incomplete trailing episode"):
        convert_experience(experience)


def test_convert_can_explicitly_drop_incomplete_trailing_episodes():
    experience = make_experience()
    experience["truncations"][1, 4] = 0

    outputs, metadata = convert_experience(
        experience, drop_incomplete_tail=True
    )

    np.testing.assert_array_equal(outputs["path_lengths"], [2, 3, 1])
    assert outputs["obs"].shape == (6, 2, 5)
    np.testing.assert_array_equal(
        outputs["actions"],
        np.concatenate(
            [experience["actions"][0], experience["actions"][1, :1]], axis=0
        ),
    )
    assert metadata["source_n_transitions"] == 10
    assert metadata["n_transitions"] == 6
    assert metadata["dropped_incomplete_tail_transitions"] == 4


def test_convert_rejects_duplicate_agent_ids():
    experience = make_experience()
    observations = experience["observations"]
    ids = np.broadcast_to(np.eye(2)[None, None], (2, 5, 2, 2))
    experience["observations"] = np.concatenate([ids, observations], axis=-1)

    with pytest.raises(ValueError, match="already have a one-hot agent-id prefix"):
        convert_experience(experience)

    outputs, metadata = convert_experience(experience, add_agent_id=False)
    assert outputs["obs"].shape[-1] == 5
    assert metadata["agent_id"]["already_present_in_source"] is True


def test_write_cache_writes_manifest_and_protects_existing_output():
    outputs, metadata = convert_experience(make_experience())
    with tempfile.TemporaryDirectory() as directory:
        output_dir = Path(directory) / "smac" / "test" / "Good"
        write_madiff_cache(outputs, output_dir, metadata)

        np.testing.assert_array_equal(
            np.load(output_dir / "path_lengths.npy"), [2, 3, 1, 4]
        )
        with (output_dir / "manifest.json").open(encoding="utf-8") as file:
            manifest = json.load(file)
        assert manifest["converter_version"] == 2
        assert manifest["output_shapes"]["obs"] == [10, 2, 5]

        with pytest.raises(FileExistsError):
            write_madiff_cache(outputs, output_dir, metadata)

        write_madiff_cache(outputs, output_dir, metadata, overwrite=True)
