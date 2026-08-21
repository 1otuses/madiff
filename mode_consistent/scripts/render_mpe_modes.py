#!/usr/bin/env python
"""复现并渲染受控 Simple Spread 数据中的团队协调 mode。"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from diffuser.datasets.mpe import make_env
from mode_consistent.data.mpe_modes import (
    DEFAULT_QUALITIES,
    assigned_landmark_controller,
    coordination_modes,
)


@dataclass(frozen=True)
class ModeRollout:
    mode_id: int
    assignment: Tuple[int, ...]
    agent_positions: np.ndarray
    landmark_positions: np.ndarray
    team_return: float
    collision_steps: int


def collect_mode_rollouts(
    seed: int,
    horizon: int,
    mode_ids: Sequence[int] = tuple(range(6)),
) -> Tuple[ModeRollout, ...]:
    """使用数据生成器的无噪声专家控制器复现指定 mode。"""

    if seed < 0 or horizon < 1:
        raise ValueError("seed 不能为负数，horizon 必须为正数")
    assignments = coordination_modes(3)
    mode_ids = tuple(int(mode_id) for mode_id in mode_ids)
    if not mode_ids or len(set(mode_ids)) != len(mode_ids):
        raise ValueError("mode_ids 必须非空且不能重复")
    if min(mode_ids) < 0 or max(mode_ids) >= len(assignments):
        raise ValueError("mode_ids 必须位于 [0, 5]")

    rollouts = []
    for mode_id in mode_ids:
        assignment = assignments[mode_id]
        env = make_env("simple_spread", render_mode="rgb_array")
        env.seed(seed)
        observation = np.asarray(env.reset(), dtype=np.float32)
        landmarks = (
            observation[0, 4:10].reshape(3, 2) + observation[0, 2:4]
        )
        positions = [observation[:, 2:4].copy()]
        team_return = 0.0
        collision_steps = 0
        controller_seed = ((seed + 1) * 1_000_003 + mode_id * 101) % (
            2**32 - 1
        )
        rng = np.random.RandomState(controller_seed)
        for _ in range(horizon):
            action = assigned_landmark_controller(
                observation, assignment, DEFAULT_QUALITIES[0], rng
            )
            observation, reward, _, _ = env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            positions.append(observation[:, 2:4].copy())
            team_return += float(np.asarray(reward).sum())
            current = observation[:, 2:4]
            if any(
                np.linalg.norm(current[i] - current[j]) < 0.30
                for i in range(3)
                for j in range(i + 1, 3)
            ):
                collision_steps += 1
        env.close()
        rollouts.append(
            ModeRollout(
                mode_id=mode_id,
                assignment=assignment,
                agent_positions=np.asarray(positions),
                landmark_positions=landmarks,
                team_return=team_return,
                collision_steps=collision_steps,
            )
        )
    return tuple(rollouts)


def render_mode_rollouts(
    rollouts: Sequence[ModeRollout], output_dir: Path, fps: int = 6
) -> dict:
    """输出逐 mode 视频、同屏比较视频、终态轨迹图和数值摘要。"""

    if fps < 1:
        raise ValueError("fps 必须为正整数")
    if not rollouts:
        raise ValueError("rollouts 不能为空")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon = len(rollouts[0].agent_positions) - 1
    seed_label = output_dir.name
    colors = ("#d62728", "#1f77b4", "#2ca02c")
    landmark_names = ("A", "B", "C")

    def draw(axis, rollout, frame):
        axis.clear()
        axis.set_xlim(-1.15, 1.15)
        axis.set_ylim(-1.15, 1.15)
        axis.set_aspect("equal")
        axis.grid(alpha=0.18)
        mapping = ", ".join(
            f"{agent + 1}->{landmark_names[target]}"
            for agent, target in enumerate(rollout.assignment)
        )
        axis.set_title(f"Mode {rollout.mode_id}: {mapping}", fontsize=10)
        for landmark_id, point in enumerate(rollout.landmark_positions):
            axis.scatter(*point, marker="X", s=150, c="#333333", zorder=2)
            axis.text(
                point[0] + 0.035,
                point[1] + 0.035,
                landmark_names[landmark_id],
                weight="bold",
            )
        for agent_id, color in enumerate(colors):
            path = rollout.agent_positions[: frame + 1, agent_id]
            axis.plot(path[:, 0], path[:, 1], color=color, linewidth=1.8, alpha=0.75)
            axis.scatter(
                *rollout.agent_positions[0, agent_id],
                s=45,
                facecolors="none",
                edgecolors=color,
                linewidths=1.4,
            )
            axis.scatter(
                *path[-1],
                s=90,
                c=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            axis.text(
                path[-1, 0] + 0.035,
                path[-1, 1] + 0.035,
                str(agent_id + 1),
                color=color,
                weight="bold",
            )
        axis.set_xlabel(f"t={frame}/{horizon}", fontsize=8)

    # 每个 mode 单独保存，便于逐个观察。
    video_paths = []
    for rollout in rollouts:
        figure, axis = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
        path = output_dir / f"mode{rollout.mode_id}.mp4"
        writer = FFMpegWriter(fps=fps, metadata={"title": path.stem})
        with writer.saving(figure, str(path), dpi=130):
            for frame in range(horizon + 1):
                draw(axis, rollout, frame)
                writer.grab_frame()
        plt.close(figure)
        video_paths.append(str(path))

    # 多个 mode 同屏显示，保证比较时使用完全相同的坐标范围。
    columns = min(3, len(rollouts))
    rows = int(np.ceil(len(rollouts) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.7 * columns, 3.7 * rows), squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(f"MPE Simple Spread: coordination modes ({seed_label})", fontsize=15)
    combined_path = output_dir / "all_modes.mp4"
    writer = FFMpegWriter(fps=fps, metadata={"title": "MPE coordination modes"})
    with writer.saving(figure, str(combined_path), dpi=130):
        for frame in range(horizon + 1):
            for axis, rollout in zip(axes.flat, rollouts):
                draw(axis, rollout, frame)
            for axis in axes.flat[len(rollouts) :]:
                axis.set_visible(False)
            writer.grab_frame()
    plt.close(figure)

    figure, axes = plt.subplots(
        rows, columns, figsize=(3.7 * columns, 3.7 * rows), squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle("MPE Simple Spread: complete mode trajectories", fontsize=15)
    for axis, rollout in zip(axes.flat, rollouts):
        draw(axis, rollout, horizon)
    for axis in axes.flat[len(rollouts) :]:
        axis.set_visible(False)
    paths_image = output_dir / "all_modes_final_paths.png"
    figure.savefig(paths_image, dpi=160)
    plt.close(figure)

    mode_summaries = []
    for rollout in rollouts:
        distances = np.linalg.norm(
            rollout.agent_positions[-1, :, None, :]
            - rollout.landmark_positions[None, :, :],
            axis=-1,
        )
        nearest = distances.argmin(axis=1)
        mode_summaries.append(
            {
                "mode_id": rollout.mode_id,
                "assignment": list(rollout.assignment),
                "final_nearest_landmarks": nearest.tolist(),
                "unique_coverage": len(set(nearest.tolist())) == 3,
                "assigned_final_distances": distances[
                    np.arange(3), rollout.assignment
                ].tolist(),
                "team_return": rollout.team_return,
                "collision_steps": rollout.collision_steps,
            }
        )
    summary = {
        "combined_video": str(combined_path),
        "individual_videos": video_paths,
        "final_paths": str(paths_image),
        "modes": mode_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mode_consistent/artifacts/mpe_mode_videos/seed0"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--mode-ids", type=int, nargs="+", default=tuple(range(6)))
    args = parser.parse_args()

    summary = render_mode_rollouts(
        collect_mode_rollouts(args.seed, args.horizon, args.mode_ids),
        args.output_dir,
        args.fps,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
