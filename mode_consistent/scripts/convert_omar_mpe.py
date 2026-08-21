#!/usr/bin/env python
"""将 OMAR MPE 原始 collector 转为无标签联合 episode 数据。"""

import argparse
import json
from pathlib import Path

import numpy as np

from mode_consistent.data.omar_mpe import build_omar_episode_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/lotus/lotus/lhh/offline_datasets/OMAR/mpe"),
    )
    parser.add_argument(
        "--task",
        choices=("simple_spread", "simple_tag", "simple_world"),
        default="simple_spread",
    )
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--max-episodes-per-collector", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在：{args.output}；如需替换请显式传入 --overwrite")
    store = build_omar_episode_store(
        args.dataset_root,
        task=args.task,
        horizon=args.horizon,
        n_controlled_agents=args.n_agents,
        max_episodes_per_collector=args.max_episodes_per_collector,
    )
    store.save(args.output)
    counts = np.bincount(store.collector_ids).tolist()
    print(
        json.dumps(
            {
                "输出": str(args.output),
                "task": args.task,
                "episode数": store.n_episodes,
                "collector计数": counts,
                "agent数": store.n_agents,
                "horizon": store.horizon,
                "包含true_modes": store.true_modes is not None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
