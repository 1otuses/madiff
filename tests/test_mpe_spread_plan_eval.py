import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

from diffuser.utils.mpe_plan_visualization import (
    _mpe_agent_positions,
    _mpe_landmark_positions,
    _format_returns_text,
    _last_episode_indices,
    plot_mpe_plan_rollout,
)
from run_scripts.evaluate import should_save_eval_plan_images_for_step


def test_mpe_position_helpers_extract_agents_and_landmarks():
    obs = np.zeros((3, 18), dtype=np.float32)
    obs[:, 2:4] = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    obs[0, 4:10] = np.array([0.5, 0.0, 0.0, 0.5, -0.5, -0.5])

    np.testing.assert_allclose(
        _mpe_agent_positions(obs),
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    np.testing.assert_allclose(
        _mpe_landmark_positions(obs, n_landmarks=3),
        np.array([[1.5, 2.0], [1.0, 2.5], [0.5, 1.5]]),
    )


def test_evaluator_plot_mpe_eval_plan_rollout_saves_png_in_2x2_grid(tmpdir):
    actual = np.zeros((6, 3, 18), dtype=np.float32)
    planned = np.zeros((4, 4, 3, 18), dtype=np.float32)

    for timestep in range(actual.shape[0]):
        actual[timestep, :, 2:4] = np.array(
            [[timestep, 0.0], [0.0, timestep], [timestep, timestep]],
            dtype=np.float32,
        )
        actual[timestep, 0, 4:10] = np.array(
            [0.5, 0.0, 0.0, 0.5, -0.5, -0.5],
            dtype=np.float32,
        )

    for env_step in range(planned.shape[0]):
        for plan_step in range(planned.shape[1]):
            planned[env_step, plan_step, :, 2:4] = (
                actual[min(env_step + plan_step, actual.shape[0] - 1), :, 2:4]
                + 0.1
            )

    savepath = tmpdir.join("eval_mpe_plan.png")
    returned_path = plot_mpe_plan_rollout(
        actual_observations=actual,
        planned_observations=planned,
        savepath=str(savepath),
        plot_steps=[0, 1, 2, 3],
        rollout_horizon=4,
        title="eval test",
        grid_cols=2,
    )

    assert returned_path == str(savepath)
    assert savepath.check()


def test_evaluator_plot_mpe_eval_plan_rollout_saves_ctde_agent_grid(tmpdir):
    actual = np.zeros((6, 3, 18), dtype=np.float32)
    planned = np.zeros((4, 3, 4, 3, 18), dtype=np.float32)

    for timestep in range(actual.shape[0]):
        actual[timestep, :, 2:4] = np.array(
            [[timestep, 0.0], [0.0, timestep], [timestep, timestep]],
            dtype=np.float32,
        )
        actual[timestep, 0, 4:10] = np.array(
            [0.5, 0.0, 0.0, 0.5, -0.5, -0.5],
            dtype=np.float32,
        )

    for env_step in range(planned.shape[0]):
        for planning_agent in range(planned.shape[1]):
            for plan_step in range(planned.shape[2]):
                planned[env_step, planning_agent, plan_step, :, 2:4] = (
                    actual[min(env_step + plan_step, actual.shape[0] - 1), :, 2:4]
                    + 0.1 * (planning_agent + 1)
                )

    savepath = tmpdir.join("eval_mpe_ctde_plan.png")
    returned_path = plot_mpe_plan_rollout(
        actual_observations=actual,
        planned_observations=planned,
        savepath=str(savepath),
        plot_steps=[0, 1, 2, 3],
        rollout_horizon=4,
        title="ctde eval test",
        returns_text=(
            "Online cumulative returns\nmean: 1.0\nagent: [1.0, 1.0, 1.0]"
        ),
    )

    assert returned_path == str(savepath)
    assert savepath.check()


def test_eval_plan_image_selects_last_episode_by_default():
    assert _last_episode_indices(
        num_available=10,
        batch_start_episode_idx=0,
        max_to_save=1,
    ) == [(9, 9)]
    assert _last_episode_indices(
        num_available=2,
        batch_start_episode_idx=8,
        max_to_save=1,
    ) == [(1, 9)]


def test_eval_plan_image_only_enabled_for_last_load_step():
    class Config:
        save_eval_plan_images = True
        load_steps = [0, 100000, 1000000]

    assert not should_save_eval_plan_images_for_step(Config, 0)
    assert not should_save_eval_plan_images_for_step(Config, 100000)
    assert should_save_eval_plan_images_for_step(Config, 1000000)


def test_eval_plan_returns_text_uses_online_cumulative_returns():
    text = _format_returns_text(np.array([406.0, 381.0, 441.0]))

    assert "Online cumulative returns" in text
    assert "mean: 409.3" in text
    assert "agent: [406.0, 381.0, 441.0]" in text
