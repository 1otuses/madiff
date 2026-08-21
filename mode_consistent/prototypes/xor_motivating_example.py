"""研究原型：可视化分散 mode 采样造成的相关性缺口。

这是 oracle 实验：局部边缘分布完全准确，因此失配只来自分散采样，而非优化
误差或模型容量不足。
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2673B8"
ORANGE = "#D97706"
DARK = "#263238"
GRID = "#D9E0E6"


def theoretical_mmr(n_modes, n_agents, cue_availability=0.0):
    independent_gap = 1.0 - float(n_modes) ** (1 - int(n_agents))
    return (1.0 - cue_availability) * independent_gap


def sample_modes(rng, episodes, n_agents, n_modes, cue_availability):
    """在共同线索可见或缺失的混合条件下采样各 agent 的局部 mode。"""

    modes = rng.integers(n_modes, size=(episodes, n_agents))
    cue_present = rng.random(episodes) < cue_availability
    shared_modes = rng.integers(n_modes, size=episodes)
    modes[cue_present] = shared_modes[cue_present, None]
    return modes


def mismatch_rate(modes):
    return float(np.mean(np.any(modes != modes[:, :1], axis=1)))


def xor_actions(modes):
    if modes.shape[1] != 2:
        raise ValueError("Role-ID XOR requires exactly two agents")
    return np.column_stack([modes[:, 0], 1 - modes[:, 1]])


def action_matrix(actions):
    matrix = np.zeros((2, 2), dtype=np.float64)
    np.add.at(matrix, (actions[:, 0], actions[:, 1]), 1.0)
    return matrix / matrix.sum()


def simulate(seed, episodes, example_rho):
    rng = np.random.default_rng(seed)
    behavior_modes = sample_modes(rng, episodes, 2, 2, 1.0)
    independent_modes = sample_modes(rng, episodes, 2, 2, 0.0)
    coupled_modes = sample_modes(rng, episodes, 2, 2, example_rho)

    agent_counts = np.arange(2, 9)
    mode_counts = (2, 3, 6)
    scaling_rows = []
    for n_modes in mode_counts:
        for n_agents in agent_counts:
            modes = sample_modes(rng, episodes, n_agents, n_modes, 0.0)
            scaling_rows.append(
                {
                    "n_modes": int(n_modes),
                    "n_agents": int(n_agents),
                    "empirical_mmr": mismatch_rate(modes),
                    "theoretical_mmr": theoretical_mmr(n_modes, n_agents),
                }
            )

    rho_values = np.linspace(0.0, 1.0, 11)
    rho_rows = []
    for n_agents in (2, 3, 4):
        for rho in rho_values:
            modes = sample_modes(rng, episodes, n_agents, 2, float(rho))
            rho_rows.append(
                {
                    "rho": float(rho),
                    "n_modes": 2,
                    "n_agents": n_agents,
                    "empirical_mmr": mismatch_rate(modes),
                    "theoretical_mmr": theoretical_mmr(2, n_agents, float(rho)),
                }
            )

    return {
        "seed": seed,
        "episodes_per_condition": episodes,
        "example_rho": example_rho,
        "matrices": {
            "behavior_joint_policy": action_matrix(xor_actions(behavior_modes)),
            "independent_marginals": action_matrix(xor_actions(independent_modes)),
            "ci_conditioned": action_matrix(xor_actions(coupled_modes)),
        },
        "example_mmr": {
            "behavior_joint_policy": mismatch_rate(behavior_modes),
            "independent_marginals": mismatch_rate(independent_modes),
            "ci_conditioned": mismatch_rate(coupled_modes),
            "ci_conditioned_theory": theoretical_mmr(2, 2, example_rho),
        },
        "scaling": scaling_rows,
        "cue_sweep": rho_rows,
    }


def _plot_matrix(axis, matrix, title, subtitle):
    axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=0.5)
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.1%}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > 0.3 else DARK,
                fontweight="bold",
            )
    axis.set_xticks((0, 1), labels=("0", "1"))
    axis.set_yticks((0, 1), labels=("0", "1"))
    axis.set_xlabel("Agent 2 action")
    axis.set_ylabel("Agent 1 action")
    axis.set_title(title, loc="left", pad=27, fontweight="bold", color=DARK)
    axis.text(0.0, 1.015, subtitle, transform=axis.transAxes, color="#5F6B73")


def render(result, output_path):
    figure = plt.figure(figsize=(15, 8), facecolor="white")
    grid = figure.add_gridspec(2, 6, height_ratios=(1.0, 1.15))

    matrices = result["matrices"]
    mmr = result["example_mmr"]
    heatmaps = (
        (
            "Behavior joint policy",
            "Only the two supported XOR modes",
            matrices["behavior_joint_policy"],
        ),
        (
            "Independent local marginals",
            f"Oracle marginals, MMR = {mmr['independent_marginals']:.1%}",
            matrices["independent_marginals"],
        ),
        (
            "CI-conditioned sampling",
            f"rho = {result['example_rho']:.2f}, MMR = {mmr['ci_conditioned']:.1%}",
            matrices["ci_conditioned"],
        ),
    )
    for index, (title, subtitle, matrix) in enumerate(heatmaps):
        _plot_matrix(
            figure.add_subplot(grid[0, 2 * index : 2 * index + 2]),
            matrix,
            title,
            subtitle,
        )

    scaling_axis = figure.add_subplot(grid[1, :3])
    scaling = result["scaling"]
    line_styles = {2: "-", 3: "--", 6: ":"}
    colors = {2: BLUE, 3: ORANGE, 6: DARK}
    for n_modes in (2, 3, 6):
        rows = [row for row in scaling if row["n_modes"] == n_modes]
        scaling_axis.plot(
            [row["n_agents"] for row in rows],
            [row["theoretical_mmr"] for row in rows],
            color=colors[n_modes],
            linestyle=line_styles[n_modes],
            linewidth=2,
            label=f"K = {n_modes} theory",
        )
        scaling_axis.scatter(
            [row["n_agents"] for row in rows],
            [row["empirical_mmr"] for row in rows],
            color=colors[n_modes],
            marker="o",
            s=24,
            zorder=3,
        )
    scaling_axis.set_title(
        "Independent-sampling mismatch by team size",
        loc="left",
        pad=27,
        fontweight="bold",
    )
    scaling_axis.text(
        0.0,
        1.015,
        "Lines: 1 - K^(1-N); markers: Monte Carlo estimates",
        transform=scaling_axis.transAxes,
        color="#5F6B73",
    )
    scaling_axis.set_xlabel("Number of agents (N)")
    scaling_axis.set_ylabel("Mode mismatch rate (MMR)")
    scaling_axis.set_ylim(-0.02, 1.02)
    scaling_axis.legend(frameon=False, loc="lower right")

    cue_axis = figure.add_subplot(grid[1, 3:])
    cue_rows = result["cue_sweep"]
    styles = {2: ("-", "o"), 3: ("--", "s"), 4: (":", "^")}
    for n_agents in (2, 3, 4):
        rows = [row for row in cue_rows if row["n_agents"] == n_agents]
        line_style, marker = styles[n_agents]
        cue_axis.plot(
            [row["rho"] for row in rows],
            [row["theoretical_mmr"] for row in rows],
            color=BLUE,
            linestyle=line_style,
            linewidth=2,
            label=f"N = {n_agents} theory",
        )
        cue_axis.scatter(
            [row["rho"] for row in rows],
            [row["empirical_mmr"] for row in rows],
            color=BLUE,
            marker=marker,
            facecolors="white",
            s=28,
            zorder=3,
        )
    cue_axis.set_title(
        "Mismatch by common-cue availability",
        loc="left",
        pad=27,
        fontweight="bold",
    )
    cue_axis.text(
        0.0,
        1.015,
        "K = 2; cue presence is jointly observable",
        transform=cue_axis.transAxes,
        color="#5F6B73",
    )
    cue_axis.set_xlabel("Common-cue availability (rho)")
    cue_axis.set_ylabel("Mode mismatch rate (MMR)")
    cue_axis.set_xlim(-0.02, 1.02)
    cue_axis.set_ylim(-0.02, 1.02)
    cue_axis.legend(frameon=False, loc="upper right")

    for axis in (scaling_axis, cue_axis):
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Role-ID XOR mode consistency under decentralized sampling",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=DARK,
    )
    figure.text(
        0.06,
        0.947,
        (
            f"Oracle local marginals; {result['episodes_per_condition']:,} episodes "
            "per condition; invalid XOR actions are the heatmap diagonal"
        ),
        color="#5F6B73",
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.09, hspace=0.42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def serializable(result):
    return {
        **result,
        "matrices": {
            name: matrix.tolist() for name, matrix in result["matrices"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--example-rho", type=float, default=0.75)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mode_consistent/prototypes/artifacts"),
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not 0.0 <= args.example_rho <= 1.0:
        parser.error("--example-rho must be in [0, 1]")

    result = simulate(args.seed, args.episodes, args.example_rho)
    figure_path = args.output_dir / "xor_motivating_example.png"
    data_path = args.output_dir / "xor_motivating_example.json"
    render(result, figure_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(serializable(result), indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "state": {
                    "seed": args.seed,
                    "episodes_per_condition": args.episodes,
                    "example_rho": args.example_rho,
                },
                "observed_mmr": result["example_mmr"],
                "figure": str(figure_path),
                "source_data": str(data_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
