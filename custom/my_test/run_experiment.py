#!/usr/bin/env python3
"""
run_experiment.py — 统一实验管理脚本
======================================
功能:
  1. 读取 YAML 配置文件
  2. 调用 train.py 进行训练
  3. 按 eval_freq 频率调用 evaluate.py 进行评估 + 视频生成
  4. 支持多 seed 并行运行 (可选)

使用示例:
  # 单 seed 训练 + 自动评估
  python custom/my_test/run_experiment.py -c custom/my_test/config/mpe_simple_spread.yaml

  # 多 seed 运行
  python custom/my_test/run_experiment.py -c custom/my_test/config/mpe_simple_spread.yaml --seeds 100,200,300
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime


def resolve_python():
    """使用 conda madiff 环境的 python"""
    madiff_python = "/home/lotus/anaconda3/envs/madiff/bin/python3"
    if os.path.exists(madiff_python):
        return madiff_python
    return sys.executable


def run_command(cmd: list, step_desc: str = ""):
    """运行一条命令并等待完成."""
    prefix = f"[{step_desc}] " if step_desc else ""
    print(f"{prefix}Running: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"{prefix} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
    else:
        print(f"{prefix} Done in {elapsed:.1f}s")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment: train + periodic evaluation"
    )
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (e.g. 100,200,300). Overrides config seed.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_train_steps", type=int, default=None,
                        help="Override total training steps")
    parser.add_argument("--eval_freq", type=int, default=None,
                        help="Override evaluation frequency")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training, only evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Manually specify checkpoint for evaluation")
    args = parser.parse_args()

    # ---- 加载配置 ----
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 重载参数
    if args.n_train_steps is not None:
        cfg["training"]["n_train_steps"] = args.n_train_steps
    if args.eval_freq is not None:
        cfg["evaluation"]["eval_freq"] = args.eval_freq

    n_train_steps = cfg["training"]["n_train_steps"]
    eval_freq = cfg["evaluation"]["eval_freq"]
    env_name = cfg["env"]["env_name"]
    save_dir = cfg["paths"]["save_dir"].format(env_name=env_name)
    checkpoint_dir = cfg["paths"].get(
        "checkpoint_dir", os.path.join(save_dir, "checkpoint")
    ).format(env_name=env_name)
    python = resolve_python()

    # ---- seeds ----
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = [cfg["training"]["seed"]]

    print("=" * 60)
    print(f"  Experiment: {cfg['meta_data']['exp_name']}")
    print(f"  Seeds: {seeds}")
    print(f"  Total training steps: {n_train_steps:,}")
    print(f"  Eval every: {eval_freq:,} steps")
    print(f"  Python: {python}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ---- 逐 seed 运行 ----
    all_results = {}

    for seed in seeds:
        print(f"\n{'─' * 50}")
        print(f"  Seed: {seed}")
        print(f"{'─' * 50}")

        cfg["training"]["seed"] = seed

        # ---- 训练 ----
        if not args.skip_train:
            train_cmd = [
                python, "custom/my_test/run_scripts/train.py",
                "-c", args.config,
                "--device", args.device,
                "--seed", str(seed),
            ]
            if args.n_train_steps is not None:
                train_cmd.extend(["--n_train_steps", str(args.n_train_steps)])

            ok = run_command(train_cmd, f"Train seed={seed}")
            if not ok:
                print(f"  Training failed for seed={seed}, skipping evaluation.")
                continue

        # ---- 评估 ----
        checkpoint = args.checkpoint or os.path.join(checkpoint_dir, "checkpoint.pt")
        if not os.path.exists(checkpoint):
            # fallback: 旧命名
            checkpoint = os.path.join(save_dir, "checkpoint_final.pt")
        if not os.path.exists(checkpoint):
            checkpoint = os.path.join(save_dir, "checkpoint_latest.pt")
        if not os.path.exists(checkpoint):
            print(f"  No checkpoint found at {save_dir}, skipping evaluation.")
            continue

        eval_cmd = [
            python, "custom/my_test/run_scripts/evaluate.py",
            "-c", args.config,
            "--checkpoint", checkpoint,
            "--device", args.device,
        ]
        ok = run_command(eval_cmd, f"Eval seed={seed}")
        if not ok:
            print(f"  Evaluation failed for seed={seed}")

        # 收集结果
        eval_json = os.path.join(save_dir, "eval", "eval_results.json")
        if os.path.exists(eval_json):
            with open(eval_json) as f:
                results = json.load(f)
            all_results[str(seed)] = results

    # ---- 汇总 ----
    if all_results:
        print(f"\n{'=' * 60}")
        print(f"  Summary")
        print(f"{'=' * 60}")
        means = [r["overall_mean"] for r in all_results.values()]
        overall_mean = sum(means) / len(means)
        print(f"  Seeds: {list(all_results.keys())}")
        print(f"  Per-seed means: {[f'{m:.2f}' for m in means]}")
        print(f"  Overall mean: {overall_mean:.2f}")

        summary_path = os.path.join(save_dir, "experiment_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "config": args.config,
                "seeds": seeds,
                "per_seed": all_results,
                "overall_mean": overall_mean,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2, default=float)
        print(f"  Summary saved: {summary_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
