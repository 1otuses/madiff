from typing import Iterable, Optional, Sequence
import math
import os

import numpy as np


def _mpe_agent_positions(observations: np.ndarray) -> np.ndarray:
    """从 MPE 观测中提取智能体绝对坐标。"""
    observations = np.asarray(observations)
    return observations[..., 2:4]


def _mpe_landmark_positions(observation: np.ndarray, n_landmarks: int) -> np.ndarray:
    """从 simple_spread 单步观测中还原地标绝对坐标。"""
    observation = np.asarray(observation)
    if observation.ndim == 3:
        observation = observation[0]
    agent0_pos = observation[0, 2:4]
    landmark_rel = observation[0, 4 : 4 + 2 * n_landmarks].reshape(n_landmarks, 2)
    return landmark_rel + agent0_pos


def _normalise_plot_steps(plot_steps: Iterable[int], max_step: int) -> list:
    """过滤越界和重复的待绘制时间步。"""
    steps = []
    for step in plot_steps:
        if 0 <= step < max_step and step not in steps:
            steps.append(step)
    return steps


def _anchor_planned_positions_to_current(
    planned_positions: np.ndarray,
    current_positions: np.ndarray,
) -> np.ndarray:
    """仅在绘图时把 DM 预测轨迹起点锚定到真实当前坐标。"""
    anchored_positions = np.array(planned_positions, copy=True)
    if anchored_positions.shape[0] > 0:
        anchored_positions[0] = current_positions
    return anchored_positions


def _format_returns_text(episode_rewards: np.ndarray) -> str:
    """将单个评估 episode 的累计回报整理为图中显示的文本。"""
    episode_rewards = np.asarray(episode_rewards, dtype=np.float32)
    mean_reward = float(episode_rewards.mean())
    per_agent = ", ".join(f"{reward:.1f}" for reward in episode_rewards)
    return f"Online cumulative returns\nmean: {mean_reward:.1f}\nagent: [{per_agent}]"


def _last_episode_indices(
    num_available: int,
    batch_start_episode_idx: int,
    max_to_save: int,
) -> list:
    """返回最后一批评估中需要保存图片的本地/全局 episode 下标。"""
    num_to_save = min(max(0, max_to_save), num_available)
    first_local_idx = num_available - num_to_save
    return [
        (local_idx, batch_start_episode_idx + local_idx)
        for local_idx in range(first_local_idx, num_available)
    ]


def plot_mpe_plan_rollout(
    actual_observations: np.ndarray,
    planned_observations: np.ndarray,
    savepath: str,
    plot_steps: Sequence[int],
    rollout_horizon: int,
    title: Optional[str] = None,
    returns_text: Optional[str] = None,
    anchor_plan_start: bool = True,
    grid_cols: Optional[int] = None,
) -> str:
    """绘制 MPE 真实 rollout(实线)和 DM 预测轨迹(虚线)。"""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    actual_observations = np.asarray(actual_observations)
    planned_observations = np.asarray(planned_observations)
    if planned_observations.ndim == 4:
        planned_panels = planned_observations[:, None]
        panel_titles = ["joint DM plan"]
        has_planning_agent_panels = False
    elif planned_observations.ndim == 5:
        planned_panels = planned_observations
        panel_titles = [
            f"planning agent {agent_idx}"
            for agent_idx in range(planned_observations.shape[1])
        ]
        has_planning_agent_panels = True
    else:
        raise ValueError(
            "planned_observations 形状应为 [T,H,A,D] 或 [T,P,H,A,D]。"
        )

    n_agents = actual_observations.shape[1]
    n_panels = planned_panels.shape[1]
    plot_steps = _normalise_plot_steps(plot_steps, planned_panels.shape[0])
    if not plot_steps:
        plot_steps = [0]

    grid_cols = grid_cols or n_panels
    use_step_grid = n_panels == 1 and grid_cols > 1
    if use_step_grid:
        n_rows = int(math.ceil(len(plot_steps) / grid_cols))
        n_cols = min(grid_cols, len(plot_steps))
    else:
        n_rows = len(plot_steps)
        n_cols = n_panels

    agent_colors = ["crimson", "forestgreen", "purple"]
    if n_agents > len(agent_colors):
        agent_colors.extend(plt.get_cmap("tab10").colors[len(agent_colors) : n_agents])
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.7 * n_rows),
        squeeze=False,
    )

    all_positions = [_mpe_agent_positions(actual_observations)]
    for env_step in plot_steps:
        current_positions = _mpe_agent_positions(actual_observations[env_step])
        horizon = min(
            rollout_horizon,
            planned_panels.shape[2],
            actual_observations.shape[0] - env_step,
        )
        for panel_idx in range(n_panels):
            planned_positions = _mpe_agent_positions(
                planned_panels[env_step, panel_idx, :horizon]
            )
            if anchor_plan_start:
                planned_positions = _anchor_planned_positions_to_current(
                    planned_positions,
                    current_positions,
                )
            all_positions.append(planned_positions)

    stacked_positions = np.concatenate(
        [positions.reshape(-1, 2) for positions in all_positions],
        axis=0,
    )
    minimum = stacked_positions.min(axis=0)
    maximum = stacked_positions.max(axis=0)
    span = max(float((maximum - minimum).max()), 1.0)
    margin = 0.1 * span
    center = (minimum + maximum) / 2
    half_span = span / 2 + margin
    xy_min = center - half_span
    xy_max = center + half_span

    used_axes = set()
    for step_idx, env_step in enumerate(plot_steps):
        actual_step_obs = actual_observations[env_step]
        landmarks = _mpe_landmark_positions(actual_step_obs, n_agents)
        current_positions = _mpe_agent_positions(actual_step_obs)
        horizon = min(
            rollout_horizon,
            planned_panels.shape[2],
            actual_observations.shape[0] - env_step,
        )
        actual_positions = _mpe_agent_positions(
            actual_observations[env_step : env_step + horizon]
        )

        for panel_idx in range(n_panels):
            if use_step_grid:
                row_idx = step_idx // n_cols
                col_idx = step_idx % n_cols
            else:
                row_idx = step_idx
                col_idx = panel_idx
            used_axes.add((row_idx, col_idx))

            ax = axes[row_idx, col_idx]
            planned_positions = _mpe_agent_positions(
                planned_panels[env_step, panel_idx, :horizon]
            )
            if anchor_plan_start:
                planned_positions = _anchor_planned_positions_to_current(
                    planned_positions,
                    current_positions,
                )

            ax.scatter(
                landmarks[:, 0],
                landmarks[:, 1],
                marker="s",
                s=70,
                color="royalblue",
                zorder=4,
            )
            for agent_idx in range(n_agents):
                line_width = 1.65
                alpha = 0.9
                ax.plot(
                    actual_positions[:, agent_idx, 0],
                    actual_positions[:, agent_idx, 1],
                    "-",
                    color=agent_colors[agent_idx],
                    lw=line_width,
                    alpha=alpha,
                    zorder=2,
                )
                ax.plot(
                    planned_positions[:, agent_idx, 0],
                    planned_positions[:, agent_idx, 1],
                    "--",
                    color=agent_colors[agent_idx],
                    lw=line_width,
                    alpha=alpha,
                    zorder=3,
                )
                marker = (
                    "^"
                    if has_planning_agent_panels and agent_idx == panel_idx
                    else "o"
                )
                ax.scatter(
                    current_positions[agent_idx, 0],
                    current_positions[agent_idx, 1],
                    marker=marker,
                    s=80 if marker == "^" else 60,
                    color=agent_colors[agent_idx],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                )

            if use_step_grid:
                ax.set_title(f"t = {env_step}", fontsize=12, fontweight="bold")
            else:
                ax.set_title(panel_titles[panel_idx], fontsize=12)
            ax.set_xlim(xy_min[0], xy_max[0])
            ax.set_ylim(xy_min[1], xy_max[1])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.18, linewidth=0.6)
            ax.tick_params(labelsize=9)
            if not use_step_grid and panel_idx == 0:
                ax.set_ylabel(f"t = {env_step}", fontsize=12, fontweight="bold")

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            if (row_idx, col_idx) not in used_axes:
                axes[row_idx, col_idx].axis("off")

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="royalblue",
            markersize=9,
            label="Landmarks",
        ),
        Line2D([0], [0], color="black", lw=1.65, label="Real rollout"),
        Line2D([0], [0], color="black", lw=1.65, ls="--", label="DM plan"),
    ]
    if has_planning_agent_panels:
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor="gray",
                    markersize=8,
                    label="Planning agent",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="gray",
                    markersize=8,
                    label="Other agents",
                ),
            ]
        )
    else:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label="Agents",
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    if title:
        fig.suptitle(title, fontsize=14, y=0.985)
    if returns_text:
        fig.text(
            0.985,
            0.94 if title else 0.985,
            returns_text,
            ha="right",
            va="top",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.9,
            },
        )
    top_margin = 0.90 if (title or returns_text) else 1
    fig.tight_layout(rect=(0, 0.07, 1, top_margin))

    os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
    fig.savefig(savepath, dpi=180)
    plt.close(fig)
    return savepath


def save_mpe_plan_visualizations(
    log_dir: str,
    load_step: Optional[int],
    plan_rollouts: dict,
    batch_start_episode_idx: int,
    max_to_save: int,
    plot_steps: Sequence[int],
    rollout_horizon: int,
    anchor_plan_start: bool = True,
    grid_cols: Optional[int] = None,
    save_npz: bool = False,
) -> list:
    """保存 MPE 评估规划轨迹图，返回已保存的 PNG 路径列表。"""
    actual_observations = plan_rollouts["actual_observations"]
    planned_observations = plan_rollouts["planned_observations"]
    episode_rewards = plan_rollouts["episode_rewards"]
    episode_indices = _last_episode_indices(
        num_available=actual_observations.shape[0],
        batch_start_episode_idx=batch_start_episode_idx,
        max_to_save=max_to_save,
    )

    saved_paths = []
    for local_idx, episode_idx in episode_indices:
        image_path = os.path.join(
            log_dir,
            "images",
            f"step_{load_step}-eval_ep_{episode_idx}-mpe_spread_plans.png",
        )
        plot_mpe_plan_rollout(
            actual_observations=actual_observations[local_idx],
            planned_observations=planned_observations[local_idx],
            savepath=image_path,
            plot_steps=plot_steps,
            rollout_horizon=rollout_horizon,
            title=f"MPE Spread eval plan | checkpoint {load_step} | episode {episode_idx}",
            returns_text=_format_returns_text(episode_rewards[local_idx]),
            anchor_plan_start=anchor_plan_start,
            grid_cols=grid_cols,
        )
        saved_paths.append(image_path)

        if save_npz:
            npz_path = image_path.replace(".png", ".npz")
            np.savez_compressed(
                npz_path,
                actual_observations=actual_observations[local_idx],
                planned_observations=planned_observations[local_idx],
                episode_rewards=episode_rewards[local_idx],
            )

    return saved_paths
