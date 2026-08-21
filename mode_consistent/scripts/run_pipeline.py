#!/usr/bin/env python
"""训练并贯通 CI-CoDiff 的中央 mode、局部对齐、价值与条件 MADiff。"""

import argparse
import json
from pathlib import Path

import torch

from mode_consistent.data import load_episode_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ci_codiff_pipeline.pt"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-modes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--central-steps", type=int, default=500)
    parser.add_argument("--local-steps", type=int, default=500)
    parser.add_argument("--value-steps", type=int, default=200)
    parser.add_argument("--diffusion-steps", type=int, default=1_000)
    parser.add_argument("--diffusion-timesteps", type=int, default=20)
    parser.add_argument("--local-prefix", type=int, default=1)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from mode_consistent.pipeline import PipelineConfig, run_pipeline

    config = PipelineConfig(
        n_modes=args.n_modes,
        batch_size=args.batch_size,
        central_steps=args.central_steps,
        local_steps=args.local_steps,
        value_steps=args.value_steps,
        diffusion_steps=args.diffusion_steps,
        diffusion_timesteps=args.diffusion_timesteps,
        local_prefix=args.local_prefix,
        confidence_threshold=args.confidence_threshold,
        seed=args.seed,
    )
    artifacts = run_pipeline(load_episode_store(args.dataset), config, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts.checkpoint(), args.output)
    summary_path = args.summary or args.output.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(artifacts.summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(artifacts.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
