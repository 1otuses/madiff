import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
from PIL import Image

from diffuser.utils.mpe_plan_visualization import (
    _mpe_agent_positions,
    _mpe_landmark_positions,
    _format_returns_text,
    _last_episode_indices,
    plot_mpe_plan_rollout,
)
from diffuser.utils.evaluator import (
    MADEvaluator,
    MADEvaluatorWorker,
    _build_eval_metrics,
    _eval_tensorboard_log_dir,
    _reset_eval_environments,
)
from run_scripts.evaluate import (
    build_evaluator_kwargs,
    should_save_eval_plan_images_for_step,
)


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


def test_evaluator_plot_mpe_centralized_plan_uses_two_by_two_grid(tmpdir):
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
    with Image.open(str(savepath)) as image:
        width, height = image.size
    assert 0.8 < width / height < 1.4


def test_evaluator_plot_mpe_lines_start_at_the_selected_environment_step(
    tmpdir,
    monkeypatch,
):
    from matplotlib.axes import Axes

    actual = np.zeros((6, 3, 18), dtype=np.float32)
    planned = np.zeros((6, 2, 3, 18), dtype=np.float32)
    actual[:, :, 2] = np.arange(6, dtype=np.float32)[:, None]
    planned[:, :, :, 2] = np.arange(2, dtype=np.float32)[None, :, None]

    plotted_lines = []
    original_plot = Axes.plot

    def capture_plot(self, *args, **kwargs):
        plotted_lines.append((np.asarray(args[0]).copy(), args[2]))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture_plot)
    plot_mpe_plan_rollout(
        actual_observations=actual,
        planned_observations=planned,
        savepath=str(tmpdir.join("selected_step.png")),
        plot_steps=[2],
        rollout_horizon=2,
    )

    actual_x, actual_style = plotted_lines[0]
    planned_x, planned_style = plotted_lines[1]
    np.testing.assert_array_equal(actual_x, [2.0, 3.0])
    np.testing.assert_array_equal(planned_x, [2.0, 1.0])
    assert actual_style == "-"
    assert planned_style == "--"


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
    with Image.open(str(savepath)) as image:
        width, height = image.size
    assert width < height < 1.5 * width


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


def test_fixed_eval_seeds_reproduce_the_same_initial_states():
    class RandomResetEnv:
        def reset(self):
            return np.random.uniform(size=3)

    first = _reset_eval_environments(
        [RandomResetEnv(), RandomResetEnv()],
        episode_seeds=[100, 200],
    )
    second = _reset_eval_environments(
        [RandomResetEnv(), RandomResetEnv()],
        episode_seeds=[100, 200],
    )

    np.testing.assert_allclose(first, second)
    assert not np.allclose(first[0], first[1])


def test_eval_metrics_report_mean_over_the_same_episodes_and_agents():
    metrics = _build_eval_metrics(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
    )

    np.testing.assert_allclose(metrics["average_ep_reward"], [2.5, 3.5, 4.5])
    assert metrics["mean_ep_reward"] == 3.5


def test_eval_tensorboard_uses_existing_run_directories():
    assert _eval_tensorboard_log_dir("logs/example", False) == os.path.join(
        "logs/example", "tensorboard", "eval"
    )
    assert _eval_tensorboard_log_dir("logs/example", True) == os.path.join(
        "logs/example", "tensorboard", "eval-ddim"
    )
    assert _eval_tensorboard_log_dir(
        "logs/example",
        True,
        condition_guidance_w=1.2,
    ) == os.path.join("logs/example", "tensorboard", "eval-ddim-cg_1.2")


def test_eval_tensorboard_uses_existing_scalar_tags():
    class RecordingWriter:
        def __init__(self):
            self.scalars = []
            self.flushed = False

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

        def flush(self):
            self.flushed = True

    worker = MADEvaluatorWorker.__new__(MADEvaluatorWorker)
    worker.eval_tb_writer = RecordingWriter()

    worker._write_eval_tensorboard(
        {
            "average_ep_reward": np.array([1.0, 2.0, 3.0]),
            "std_ep_reward": np.array([0.25, 0.5, 0.75]),
            "mean_ep_reward": 2.0,
            "win_rate": 0.8,
        },
        load_step=200000,
    )

    assert worker.eval_tb_writer.scalars == [
        ("eval/mean", 2.0, 200000),
        ("eval/std", 0.5, 200000),
        ("eval/win_rate", 0.8, 200000),
    ]
    assert worker.eval_tb_writer.flushed


def test_eval_seeds_are_forwarded_to_the_evaluator_worker():
    class Config:
        log_dir = "logs/example"
        num_eval = 5
        num_envs = 5
        condition_guidance_w = None
        use_ddim_sample = True
        n_ddim_steps = 15
        eval_seeds = [100, 200, 300, 400, 500]

    kwargs = build_evaluator_kwargs(Config)

    assert kwargs["eval_seeds"] == [100, 200, 300, 400, 500]


def test_evaluator_close_force_kills_a_worker_ignoring_terminate():
    class FakeQueue:
        def __init__(self):
            self.messages = []
            self.closed = False

        def put(self, message):
            self.messages.append(message)

        def close(self):
            self.closed = True

    class FakeRemote:
        def __init__(self):
            self.closed = False

        def recv(self):
            return "closed"

        def close(self):
            self.closed = True

    class StuckWorker:
        def __init__(self):
            self.alive = True
            self.join_timeouts = []
            self.terminated = False
            self.killed = False

        def join(self, timeout=None):
            assert timeout is not None, "an unbounded join would hang forever"
            self.join_timeouts.append(timeout)

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self.alive = False

    evaluator = MADEvaluator.__new__(MADEvaluator)
    evaluator.queue = FakeQueue()
    evaluator.parent_remote = FakeRemote()
    evaluator._worker_process = StuckWorker()

    evaluator.close(join_timeout=0.01)

    assert evaluator.queue.messages == [["close", None]]
    assert evaluator._worker_process.join_timeouts == [0.01, 0.01, 0.01]
    assert evaluator._worker_process.terminated
    assert evaluator._worker_process.killed
    assert evaluator.queue.closed
    assert evaluator.parent_remote.closed
    assert evaluator._closed
