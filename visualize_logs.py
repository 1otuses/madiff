#!/usr/bin/env python3
"""
可视化 MADiff 训练日志（兼容 ml_logger 格式输出 + TensorBoard）。
支持两种模式：
1. --tb_dir: 直接读取 TensorBoard events 文件绘图
2. --log_dir: 读取 ml_logger 的 outputs.log 和 metrics.pkl，输出训练曲线

用法：
  # 查看 TensorBoard 日志 (推荐)
  python visualize_logs.py --tb_dir logs/mad_mpe/simple_spread-expert/.../tensorboard

  # 查看 ml_logger 原始日志
  python visualize_logs.py --log_dir logs/mad_mpe/simple_spread-expert/.../seed_100

  # 批量对比多个 seed
  python visualize_logs.py --tb_dirs logs/mad_mpe/simple_spread-expert/seed_100/tensorboard \
                           logs/mad_mpe/simple_spread-expert/seed_200/tensorboard
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ===========================================================================
# 1. 从 TensorBoard events 文件解析时间序列
# ===========================================================================

def parse_tensorboard_events(tb_dir: str):
    """
    使用 tensorboard 的数据读取 API 解析 events 文件。
    返回 dict: {tag: [(step, value), ...]}
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[Error] tensorboard 未安装。请运行: pip install tensorboard")
        sys.exit(1)

    if not os.path.isdir(tb_dir):
        print(f"[Warn] TensorBoard 目录不存在: {tb_dir}")
        return {}

    ea = EventAccumulator(tb_dir)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"[Warn] 在 {tb_dir} 中未找到 scalar 数据")
        return {}

    result = {}
    for tag in tags:
        events = ea.Scalars(tag)
        result[tag] = [(e.step, e.value) for e in events]

    return result


# ===========================================================================
# 2. 从 ml_logger outputs.log 解析训练曲线
# ===========================================================================

def parse_outputs_log(log_dir: str):
    """
    从 ml_logger 的 outputs.log 中解析出 step 和 loss 等指标。
    匹配格式: "0:    0.3684 | a0_loss:    1.1387 | inv_loss:    0.3362 | t:   0.5814"
    """
    outputs_path = os.path.join(log_dir, "outputs.log")
    if not os.path.exists(outputs_path):
        print(f"[Warn] outputs.log 不存在: {outputs_path}")
        return {}

    pattern = re.compile(
        r"^\s*(?P<step>\d+):\s+(?P<loss>[\d.eE+-]+)\s*"
        r"(?:\|\s*(?P<key>\w+):\s+(?P<val>[\d.eE+-]+))*"
    )

    records = {}  # {step: {tag: value}}
    with open(outputs_path, "r") as f:
        for line in f:
            line = line.strip()
            # 主匹配: step: loss | k1: v1 | k2: v2 ...
            m = re.match(
                r"^(\d+):\s+([\d.eE+-]+)((?:\s*\|\s*\w+\s*:\s*[\d.eE+-]+)*)",
                line,
            )
            if not m:
                continue

            step = int(m.group(1))
            loss = float(m.group(2))
            rest = m.group(3)

            entry = {"Loss/total": loss}
            # 解析 | key: val 对
            pairs = re.findall(r"\|\s*(\w+)\s*:\s*([\d.eE+-]+)", rest)
            for key, val_str in pairs:
                entry[f"Loss/{key}"] = float(val_str)

            records[step] = entry

    # 合并为时间序列
    if not records:
        return {}

    tags = set()
    for entry in records.values():
        tags.update(entry.keys())

    result = {}
    for tag in tags:
        sorted_steps = sorted(records.keys())
        steps = [s for s in sorted_steps if tag in records[s]]
        result[tag] = [(s, records[s][tag]) for s in steps]

    return result


def parse_metrics_pkl(log_dir: str):
    """
    从 ml_logger 的 metrics.pkl 读取最新的指标（快照）。
    仅用于查看最后一步的值。
    """
    import pickle

    metrics_path = os.path.join(log_dir, "metrics.pkl")
    if not os.path.exists(metrics_path):
        return {}

    try:
        with open(metrics_path, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"[Warn] 无法解析 metrics.pkl: {e}")
        return {}


# ===========================================================================
# 3. 从 eval results JSON 解析评估结果
# ===========================================================================

def parse_eval_results(log_dir: str):
    """读取 {log_dir}/results/step_*.json 中的评估指标。"""
    results_dir = os.path.join(log_dir, "results")
    if not os.path.isdir(results_dir):
        return {}

    json_files = sorted(glob.glob(os.path.join(results_dir, "step_*.json")))
    if not json_files:
        return {}

    records = {}
    for jf in json_files:
        basename = os.path.basename(jf)
        m = re.match(r"step_(\d+)", basename)
        step = int(m.group(1)) if m else 0
        with open(jf, "r") as f:
            data = json.load(f)
        records[step] = data

    return records


# ===========================================================================
# 4. 绘图函数
# ===========================================================================

def plot_time_series(data_dict: dict, title: str, save_path: str):
    """
    绘制时间序列曲线。
    data_dict: {tag: [(step, value), ...], ...}
    """
    if not data_dict:
        print(f"[Warn] 无数据可绘图: {title}")
        return

    # 按 tag 前缀分组
    groups = {}
    for tag in data_dict:
        prefix = tag.split("/")[0] if "/" in tag else "other"
        groups.setdefault(prefix, {})[tag] = data_dict[tag]

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 4), squeeze=False)
    ax_list = axes[0]

    for idx, (group_name, tags_dict) in enumerate(groups.items()):
        ax = ax_list[idx]
        for tag, steps_vals in tags_dict.items():
            steps = [s for s, v in steps_vals]
            vals = [v for s, v in steps_vals]
            short_tag = tag.split("/")[-1]
            ax.plot(steps, vals, label=short_tag, linewidth=1.5)

        ax.set_xlabel("Step")
        ax.set_ylabel(group_name)
        ax.set_title(f"{title} - {group_name}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"  Saved plot: {save_path}")
    plt.close()


def print_stats(data_dict: dict, title: str):
    """打印时间序列的统计信息。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for tag, steps_vals in data_dict.items():
        vals = [v for _, v in steps_vals]
        if len(vals) > 0:
            print(f"  {tag:30s}: last={vals[-1]:.4f}  "
                  f"min={min(vals):.4f}  max={max(vals):.4f}  "
                  f"n={len(vals)}")
        else:
            print(f"  {tag:30s}: (空)")


# ===========================================================================
# 5. 主入口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="可视化 MADiff 训练日志"
    )
    parser.add_argument("--tb_dir", type=str, default=None,
                        help="TensorBoard events 目录")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="ml_logger 日志目录 (含 outputs.log)")
    parser.add_argument("--output", type=str, default="plots",
                        help="输出图片目录")
    parser.add_argument("--title", type=str, default="Training",
                        help="图表标题")
    parser.add_argument("--no_plot", action="store_true",
                        help="只打印统计，不生成图片")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ---- 读取数据 ----
    data = {}

    if args.tb_dir:
        print(f"\n[1/2] 读取 TensorBoard: {args.tb_dir}")
        tb_data = parse_tensorboard_events(args.tb_dir)
        if tb_data:
            data.update(tb_data)
            print(f"  找到 {len(tb_data)} 个 tag")
        else:
            print("  (无数据)")

    if args.log_dir:
        print(f"\n[1/2] 读取 ml_logger 日志: {args.log_dir}")
        log_data = parse_outputs_log(args.log_dir)
        if log_data:
            data.update(log_data)
            print(f"  从 outputs.log 解析到 {len(log_data)} 个 tag")
        else:
            print("  outputs.log 无可解析的训练记录")

        # 最新快照
        metrics = parse_metrics_pkl(args.log_dir)
        if metrics:
            print(f"  metrics.pkl 最后快照:")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")

        # 评估结果
        eval_data = parse_eval_results(args.log_dir)
        if eval_data:
            print(f"  评估结果 ({len(eval_data)} 次):")
            for step, results in sorted(eval_data.items()):
                avg = results.get("overall_mean", "?")
                std = results.get("overall_std", "?")
                print(f"    step {step}: overall_mean={avg}, overall_std={std}")
                # 将评估结果加入时间序列
                for k, v in results.items():
                    if isinstance(v, (int, float, np.floating)):
                        data.setdefault(f"Eval/{k}", []).append((step, float(v)))
                    elif isinstance(v, list) and all(isinstance(x, (int, float, np.floating)) for x in v):
                        for i, val in enumerate(v):
                            data.setdefault(f"Eval/{k}_agent{i}", []).append((step, float(val)))

        # 检查 checkpoint 目录
        ckpt_dir = os.path.join(args.log_dir, "checkpoint")
        if os.path.isdir(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            print(f"  Checkpoints: {len(ckpts)} 个文件 ({', '.join(sorted(ckpts)[:5])}...)")
        else:
            print(f"  无 checkpoint 目录")

    # ---- 绘图/统计 ----
    if data:
        if not args.no_plot:
            plot_path = os.path.join(args.output, f"{args.title.replace(' ', '_')}.png")
            plot_time_series(data, args.title, plot_path)
        print_stats(data, args.title)
    else:
        print("\n[Info] 未读取到任何训练数据。请检查 --tb_dir 或 --log_dir 参数。")
        print(" 示例:")
        print("   python visualize_logs.py --log_dir logs/mad_mpe/simple_spread-expert/.../seed_100")
        print("   python visualize_logs.py --tb_dir logs/mad_mpe/simple_spread-expert/.../tensorboard")


if __name__ == "__main__":
    main()
