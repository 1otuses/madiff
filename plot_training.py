#!/usr/bin/env python3
"""
Madiff 训练日志可视化脚本
=============================
支持两种数据源:
  1. 从 outputs.log 解析打印的训练步骤日志  (方案1: 解析文本)
  2. 从 TensorBoard event 文件读取          (方案2: 读取TB)
  3. 从 ml_logger metrics.pkl 读取快照     (方案3: 读取最后快照)

使用方式:
  # 可视化单个 seed 目录
  python plot_training.py -d logs/mad_mpe/simple_spread-expert/.../100/

  # 对比多个 seed
  python plot_training.py -d logs/mad_mpe/simple_spread-expert/.../100/ -d .../200/

  # 指定输出文件名
  python plot_training.py -d logs/.../100/ -o my_plot.png --no-show
"""

import argparse
import pickle
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_outputs_log(log_path: Path) -> dict:
    """
    从 outputs.log 中解析训练步骤日志行。
    格式示例: "  0:   0.3687 | a0_loss:   1.1381 | inv_loss:   0.3363 | t:   3.2571"
    """
    if not log_path.exists():
        return {}

    pattern = re.compile(
        r"^\s*(?P<step>\d+):\s+(?P<loss>[\d.]+)\s*\|"
        r"(\s*(?P<key>\w+):\s+(?P<val>[\d.]+)\s*\|?)*"
        r"\s*t:\s*[\d.]+\s*$"
    )

    # 更简单的逐行解析
    data = defaultdict(list)
    with open(log_path) as f:
        for line in f:
            # 匹配 "  0:   0.3687 | key1: val1 | key2: val2 | ..."
            m = re.match(
                r"^\s*(?P<step>\d+):\s+(?P<loss>[\d.e+\-]+)",
                line
            )
            if not m:
                continue
            step = int(m.group("step"))
            data["step"].append(step)
            data["loss"].append(float(m.group("loss")))

            # 提取 | key: val |
            rest = line[line.index("|"):] if "|" in line else ""
            for kv in re.finditer(r"\|\s*(?P<key>\w+):\s+(?P<val>[\d.e+\-]+)", rest):
                key = kv.group("key")
                val = float(kv.group("val"))
                if key not in data:
                    data[key] = [np.nan] * (len(data["step"]) - 1)
                data[key].append(val)
            # 补齐缺失的 key
            max_len = len(data["step"])
            for k in data:
                if k != "step" and len(data[k]) < max_len:
                    data[k].extend([np.nan] * (max_len - len(data[k])))

    return dict(data)


def parse_tensorboard(log_dir: Path) -> dict:
    """
    从 TensorBoard event 文件中读取数据 (需要 tensorboard 已安装)。
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {}

    if not log_dir.exists():
        return {}

    ea = EventAccumulator(str(log_dir))
    ea.Reload()

    data = {"step": []}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        if "step" not in data or len(steps) > len(data["step"]):
            data["step"] = steps
        data[tag.replace("Loss/", "").replace("/", "_")] = values

    return data


def plot_metrics(
    log_dirs: list,
    labels: list = None,
    output_path: str = None,
    show: bool = True,
    smooth: int = 0,
):
    """绘制训练指标曲线。"""
    if labels is None:
        labels = [str(d) for d in log_dirs]

    all_data = []
    for d in log_dirs:
        d = Path(d)

        # 优先尝试 outputs.log 解析
        data = parse_outputs_log(d / "outputs.log")

        # 其次尝试 TensorBoard
        if not data.get("step"):
            tb_dir = d / "tensorboard"
            if tb_dir.exists():
                data = parse_tensorboard(tb_dir)

        all_data.append(data)

    # 找出所有可用的指标
    all_keys = set()
    for data in all_data:
        all_keys.update(data.keys())
    all_keys.discard("step")

    if not all_keys:
        print("⚠ 未找到任何可绘制的训练指标数据。")
        print("  请确认 outputs.log 或 tensorboard 目录是否存在。")
        return

    # 确定哪些指标有数据
    plot_keys = sorted(all_keys)

    n_cols = min(3, len(plot_keys))
    n_rows = (len(plot_keys) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )
    fig.suptitle("Madiff Training Metrics", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(log_dirs)))

    for idx, key in enumerate(plot_keys):
        ax = axes[idx // n_cols][idx % n_cols]
        for i, data in enumerate(all_data):
            if key in data and len(data[key]) > 0:
                x = data.get("step", range(len(data[key])))
                y = np.array(data[key], dtype=float)
                # 平滑
                if smooth > 1 and len(y) > smooth:
                    kernel = np.ones(smooth) / smooth
                    y_smooth = np.convolve(y, kernel, mode="valid")
                    x_smooth = x[smooth - 1:]
                    ax.plot(x_smooth, y_smooth, color=colors[i], alpha=0.7)
                ax.plot(x, y, color=colors[i], alpha=0.3, linewidth=0.5)
                ax.scatter(x[-1:], y[-1:], color=colors[i], s=20, zorder=5)

        ax.set_title(key, fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(len(plot_keys), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    # 图例
    if len(log_dirs) > 1:
        fig.legend(
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            fontsize=8,
            frameon=True,
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ 图片已保存: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Madiff 训练日志可视化")
    parser.add_argument(
        "-d", "--log-dir", action="append", required=True, help="日志目录路径 (可多次指定)"
    )
    parser.add_argument(
        "-l", "--label", action="append", help="图例标签 (与 -d 一一对应)"
    )
    parser.add_argument("-o", "--output", help="输出图片路径")
    parser.add_argument("--no-show", action="store_true", help="不显示图片")
    parser.add_argument("--smooth", type=int, default=0, help="平滑窗口大小 (0=不平滑)")
    args = parser.parse_args()

    plot_metrics(
        log_dirs=args.log_dir,
        labels=args.label,
        output_path=args.output,
        show=not args.no_show,
        smooth=args.smooth,
    )


if __name__ == "__main__":
    main()
