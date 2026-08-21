#!/usr/bin/env python
"""生成带真实任务分配采集标签的配对 Simple Spread 数据。"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/lotus/lotus/lhh/offline_datasets/CI-CoDiff/mpe/"
            "simple_spread/balanced_6mode_expert_seed0.npz"
        ),
    )
    parser.add_argument("--n-scenarios", type=int, default=1_000)
    parser.add_argument("--mode-ids", type=int, nargs="+", default=tuple(range(6)))
    parser.add_argument(
        "--qualities",
        nargs="+",
        choices=("expert", "medium", "poor"),
        default=("expert",),
    )
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在：{args.output}；如需替换请显式传入 --overwrite")
    from mode_consistent.data.mpe_modes import (
        DEFAULT_QUALITIES,
        generate_paired_assignment_dataset,
    )

    quality_by_name = {quality.name: quality for quality in DEFAULT_QUALITIES}
    qualities = tuple(quality_by_name[name] for name in args.qualities)
    store = generate_paired_assignment_dataset(
        n_scenarios=args.n_scenarios,
        mode_ids=args.mode_ids,
        horizon=args.horizon,
        seed=args.seed,
        qualities=qualities,
    )
    store.save(args.output)
    print(
        json.dumps(
            {
                "输出": str(args.output),
                "episode数": store.n_episodes,
                "scenario数": len(np.unique(store.scenario_ids)),
                "mode计数": np.bincount(store.true_modes, minlength=6).tolist(),
                "quality": list(args.qualities),
                "agent数": store.n_agents,
                "horizon": store.horizon,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
