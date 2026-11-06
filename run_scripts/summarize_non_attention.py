"""汇总单个训练种子下,MADiff 单处交互模块替换实验的已保存结果."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

METHODS = ("attention", "identity", "pool", "mpnn")
COLORS = {"attention": "#253858", "identity": "#888888",
          "pool": "#d98b00", "mpnn": "#008768"}
LABELS = {"attention": "attention (retrained control)", "identity": "identity",
          "pool": "pool", "mpnn": "mpnn"}


def load_json(path):
    return json.loads(path.read_text())


def interval(values, seed=20261104):
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(seed).integers(0, len(values), size=(5000, len(values)))
    return np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()


def load_method(directory, method, seed, stage):
    run_dir = directory / stage / method / str(seed)
    if not (run_dir / "run_manifest.json").exists():
        return {"method": method, "status": "not_started", "evaluations": []}
    manifest = load_json(run_dir / "run_manifest.json")
    evaluations = [load_json(path) for path in (run_dir / "results").glob("step_*-ep_*.json")]
    evaluations.sort(key=lambda record: record["step"])
    result = {"method": method, "status": manifest["status"],
              "run_dir": str(run_dir), "manifest": manifest, "evaluations": evaluations}
    if manifest["status"] != "complete":
        return result
    final = next(record for record in evaluations if record["step"] == manifest["completed_updates"])
    team_returns = np.mean(final["episode_returns"], axis=1)
    parameters = manifest["parameters"]
    metrics = [json.loads(line) for line in (run_dir / "train_metrics.jsonl").read_text().splitlines()]
    diagnostics = load_json(run_dir / "diagnostics.json")[-1]
    result.update(
        updates=manifest["completed_updates"], episode_seeds=final["episode_seeds"],
        team_returns=team_returns.tolist(), mean_return=float(team_returns.mean()),
        std_episode_return=float(team_returns.std()), episode_return_mean_ci95=interval(team_returns),
        episode_lengths=final["episode_lengths"], latency=final["decision_latency"],
        training_seconds=manifest["training_seconds"], parameters=parameters,
        final_train_metrics=metrics[-1] if metrics else {}, final_diagnostics=diagnostics,
        checkpoint=final["checkpoint"],
    )
    return result


def summarize(directory, stage, seed):
    results = {method: load_method(directory, method, seed, stage) for method in METHODS}
    baseline = results["attention"]
    comparisons = {}
    if baseline["status"] == "complete":
        for method, record in results.items():
            if method != "attention" and record["status"] == "complete":
                np.testing.assert_array_equal(record["episode_seeds"], baseline["episode_seeds"])
                assert record["updates"] == baseline["updates"]
                assert record["manifest"]["shared_weights_sha256"] == baseline["manifest"]["shared_weights_sha256"]
                delta = np.asarray(record["team_returns"]) - baseline["team_returns"]
                comparisons[method] = {
                    "mean_paired_episode_difference": float(delta.mean()),
                    "paired_episode_differences": delta.tolist(),
                    "episode_bootstrap_ci95": interval(delta),
                    "positive_episode_fraction": float((delta > 0).mean()),
                }
    return {
        "stage": stage, "training_seed": seed, "methods": results, "paired_vs_attention": comparisons,
        "uncertainty_scope": "episode bootstrap conditional on one trained checkpoint per method; not multi-training-seed evidence",
        "baseline_scope": "attention is a fresh same-budget control, not the historical MADiff reproduction checkpoint",
        "comparison_scope": "one actually used decoder skip; other attention positions retained",
        "bootstrap": {"replicates": 5000, "seed": 20261104, "unit": "paired environment episode seed"},
    }


def write_tables(summary, directory):
    rows = []
    for method, record in summary["methods"].items():
        if record["status"] == "complete":
            latency = record["latency"]
            row = {
                "stage": summary["stage"], "method": method, "training_seed": summary["training_seed"],
                "updates": record["updates"], "mean_return": record["mean_return"],
                "std_episode_return": record["std_episode_return"],
                "mean_ci95_low": record["episode_return_mean_ci95"][0],
                "mean_ci95_high": record["episode_return_mean_ci95"][1],
                "training_seconds": record["training_seconds"],
                "registered_parameters": record["parameters"]["registered_diffusion_parameters"],
                "active_parameters": record["parameters"]["active_diffusion_parameters"],
                "site_parameters": record["parameters"]["site_parameters"],
                "latency_median_ms": latency["median_ms"], "latency_p95_ms": latency["p95_ms"],
                "inference_peak_extra_mib": latency["peak_extra_bytes"] / 2 ** 20,
            }
            row.update(record["final_train_metrics"])
            for timestep, value in record["final_diagnostics"]["masked_epsilon_mse"].items():
                row[f"diagnostic_epsilon_mse_t{timestep}"] = value
            rows.append(row)
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (directory / "comparison_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    with (directory / "comparison_episode_returns.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "method", "training_seed", "environment_seed", "team_mean_return", "length"])
        for method, record in summary["methods"].items():
            if record["status"] == "complete":
                writer.writerows(
                    [summary["stage"], method, summary["training_seed"], env_seed, reward, length]
                    for env_seed, reward, length in zip(record["episode_seeds"], record["team_returns"], record["episode_lengths"])
                )


def plot_comparison(summary, directory):
    if not any(record["evaluations"] for record in summary["methods"].values()):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for method, record in summary["methods"].items():
        if record["evaluations"]:
            axes[0].plot([row["step"] / 1000 for row in record["evaluations"]],
                         [row["mean_ep_reward"] for row in record["evaluations"]],
                         marker="o", ms=3, lw=1.8, label=LABELS[method], color=COLORS[method])
    axes[0].set(xlabel="Optimizer updates (thousands)", ylabel="Mean episodic team return",
                title="Saved-checkpoint evaluation")
    axes[0].grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=9)
    completed = [(method, record) for method, record in summary["methods"].items()
                 if record["status"] == "complete"]
    for index, (method, record) in enumerate(completed):
        mean = record["mean_return"]
        low, high = record["episode_return_mean_ci95"]
        axes[1].bar(index, mean, color=COLORS[method], width=.65)
        axes[1].errorbar(index, mean, yerr=[[mean - low], [high - mean]], color="#222222", capsize=4)
        axes[1].text(index, high + 3, f"{mean:.1f}", ha="center", fontsize=9)
    axes[1].set(xticks=range(len(completed)), xticklabels=[method for method, _ in completed],
                ylabel="Mean episodic team return", title="Final checkpoints; episode 95% CI")
    axes[1].grid(axis="y", alpha=.2)
    axes[1].set_axisbelow(True)
    fig.suptitle(f"Single-skip interaction comparison | training seed {summary['training_seed']} | {summary['stage']}", fontsize=11)
    fig.tight_layout()
    target = directory / f"comparison_{summary['stage']}_seed_{summary['training_seed']}"
    fig.savefig(target.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return target.with_suffix(".png").name


def report(summary, directory, image):
    methods = summary["methods"]
    complete = [method for method, row in methods.items() if row["status"] == "complete"]
    lines = ["# MADiff 单位置非注意力交互实验比较", "",
             f"阶段：**{summary['stage']}**；训练 seed：**{summary['training_seed']}**；已完成方法：**{len(complete)}/4**。", ""]
    lines += ["**来源说明：本报告的 attention 是本次从共同训练前初值、按相同预算重新训练的对照组，"
              "不是 `logs/mad_mpe` 中用户已完成的 MADiff 复现 checkpoint。"
              "各行只用于当前阶段内的等预算比较。**", ""]
    if (directory / "baseline_provenance.md").exists():
        lines += ["已有复现结果、原始数据来源及设计说明见[基线来源核对](baseline_provenance.md)。", ""]
    if summary["stage"] == "smoke":
        lines += ["本阶段用于核对训练和真实环境闭环，200 次 update 的回报只表示 smoke 状态。性能 pilot 需要独立从共同初值重新训练。", ""]
    if "mpnn" in summary["paired_vs_attention"]:
        difference = summary["paired_vs_attention"]["mpnn"]
        low, high = difference["episode_bootstrap_ci95"]
        label = "高于" if difference["mean_paired_episode_difference"] > 0 else "低于或等于"
        lines += [f"在本次固定最终 checkpoint 上，MPNN 平均回报{label} attention；配对 episode 回报差为 "
                  f"**{difference['mean_paired_episode_difference']:+.2f}**，episode bootstrap 95% 区间为 "
                  f"**[{low:+.2f}, {high:+.2f}]**。该区间条件于当前单个训练 seed；跨训练 seed 及更长训练预算的排序尚未验证。", ""]
    lines += ["## 方法与公平条件", "",
              "四组分别在一个实际执行的 decoder skip 使用 attention、identity、pool、mpnn。"
              "替换位置为 `self_attn[2]`，特征 `[B,3,512,12]`，条件维度 256；其余跨 agent attention 保留。"
              "MPNN 读取 receiver/sender/已处理条件，沿 sender 轴对无自环邻居取均值；图表示单个执行端想象轨迹内部的计算依赖。", "",
              "共有参数在构造 optimizer/EMA 前从同一个训练前初始化载入，并核对排除替换位置后的 SHA-256。"
              "四组使用相同训练数据、独立但同 seed 的数据生成器、相同扩散加噪/条件规则和原 inverse model。"
              "评估从磁盘重新加载 EMA，使用相同初态与采样 seed；评估和固定窗口诊断隔离训练随机流。", "",
              "主指标是各 agent 的 25 步累计回报再取 agent 均值，保持原环境奖励尺度。"
              "主比较固定最终 update checkpoint，未根据测试回报挑选 best。", "",
              "## 最终 checkpoint 回报", "",
              "| 方法 | 状态 | updates | 平均回报 | episode 标准差 | 均值 95% CI | 训练循环小时 |",
              "|---|---|---:|---:|---:|---|---:|"]
    for method, record in methods.items():
        if record["status"] == "complete":
            low, high = record["episode_return_mean_ci95"]
            label = "attention（本次重训对照）" if method == "attention" else method
            lines.append(f"| {label} | 完成 | {record['updates']} | {record['mean_return']:.2f} | "
                         f"{record['std_episode_return']:.2f} | [{low:.2f}, {high:.2f}] | {record['training_seconds']/3600:.3f} |")
        else:
            lines.append(f"| {method} | {record['status']} | — | — | — | — | — |")
    lines += ["", "区间按相同环境初态的 episode 重采样 5000 次；当前只有一个训练 seed。"
              "训练时间包含原训练循环与其周期 checkpoint 保存，数据构建、评估和诊断耗时另计。", ""]
    if image:
        lines += [f"![回报曲线与最终 checkpoint]({image})", ""]
    lines += ["## 相对 attention 的配对差", "",
              "| 方法 | 平均配对差 | episode bootstrap 95% CI | 回报更高的初态比例 |",
              "|---|---:|---|---:|"]
    for method, record in summary["paired_vs_attention"].items():
        low, high = record["episode_bootstrap_ci95"]
        lines.append(f"| {method} | {record['mean_paired_episode_difference']:+.2f} | [{low:+.2f}, {high:+.2f}] | "
                     f"{record['positive_episode_fraction']*100:.1f}% |")
    lines += ["", "## 参数与推理成本", "",
              "| 方法 | 位置参数 | 全部注册参数 | 实际路径参数 | latency batch/次数 | latency median/p95 ms | 推理额外峰值 MiB |",
              "|---|---:|---:|---:|---|---|---:|"]
    for method in complete:
        row = methods[method]
        params, latency = row["parameters"], row["latency"]
        lines.append(f"| {method} | {params['site_parameters']} | {params['registered_diffusion_parameters']} | "
                     f"{params['active_diffusion_parameters']} | {latency['batch_size']}/{latency['measurements']} | "
                     f"{latency['median_ms']:.2f}/{latency['p95_ms']:.2f} | "
                     f"{latency['peak_extra_bytes']/2**20:.1f} |")
    lines += ["", "计时覆盖观测归一化、扩散采样、inverse 和动作反归一化，排除环境 step。"
              "pilot 评估以 batch=5，预热 5 次后记录 30 次决策，并在 GPU 计时边界同步。"
              "smoke 只有 3 个 episode，对应 batch=3、预热后 20 次记录；计时规模见表。"
              "显存列为独立推理对象构建后相对已有分配的峰值增量，绝对值在原始结果 JSON。"
              "设备存在共享负载，耗时只代表本次运行条件；图的计算边不表示新增真实通信。", "",
              "## 生成拟合与交互诊断", "",
              "| 方法 | 实际总 loss | 去噪 loss | inverse loss | gamma | 交互梯度范数 | 固定 MSE t20/t100/t180 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for method in complete:
        row = methods[method]
        metric, diagnostic = row["final_train_metrics"], row["final_diagnostics"]["masked_epsilon_mse"]
        gamma = "—" if metric.get("gamma") is None else f"{metric['gamma']:.5f}"
        lines.append(f"| {method} | {metric['total_loss']:.5f} | {metric['denoising_loss']:.5f} | "
                     f"{metric['inverse_loss']:.5f} | {gamma} | {metric['interaction_gradient_norm']:.5f} | "
                     + "/".join(f"{diagnostic[str(t)]:.5f}" for t in [20, 100, 180]) + " |")
    lines += ["", "总 loss 是实际优化的 `0.5*(denoising+inverse)`，日志按梯度累积的 microstep 求均值。"
              "gamma 和交互梯度记录于该 update 的 optimizer.step 前；固定 MSE 使用保存后的 EMA。"
              "固定 MSE 使用四个训练窗口、固定噪声及原有效位置 mask，包含原 padding 规则，不能视为 held-out 泛化误差。"
              "`message_rms/update_rms` 另存于训练 CSV/JSONL，表示 phi/rho 输出的原始 RMS；MPNN 的 phi RMS 在删除自环消息前测量。", "",
              "## 证据与复现", "",
              "[预检与协议适配](preflight.md)、[环境](environment_summary.txt)、[数据清单](data_manifest.json)、"
              "[源代码快照清单](source_manifest.json)、[完整数值 CSV](comparison_metrics.csv)、"
              "[逐 episode 回报](comparison_episode_returns.csv)。各方法目录保留配置 pickle、"
              "run manifest、checkpoint、每次评估 JSON、固定窗口诊断和训练指标。", "",
              "```bash", "/home/lotus/miniconda3/envs/madiff/bin/python run_scripts/summarize_non_attention.py \\",
              f"  --output-root '{directory}' --stage {summary['stage']} --seed {summary['training_seed']}", "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=Path.home() / ".cache/codex/madiff/non_attention_v1")
    parser.add_argument("--stage", choices=["smoke", "pilot", "confirm"], default="pilot")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    summary = summarize(args.output_root, args.stage, args.seed)
    write_tables(summary, args.output_root)
    image = plot_comparison(summary, args.output_root)
    target = args.output_root / f"comparison_{args.stage}_seed_{args.seed}.json"
    target.write_text(json.dumps(summary, indent=2))
    report_text = report(summary, args.output_root, image)
    (args.output_root / "comparison_report.md").write_text(report_text)
    archive_report = report_text
    for filename in ("comparison_metrics.csv", "comparison_episode_returns.csv"):
        source = args.output_root / filename
        if source.exists():
            archived = target.with_name(target.stem + "." + filename)
            archived.write_bytes(source.read_bytes())
            archive_report = archive_report.replace(filename, archived.name)
    target.with_suffix(".md").write_text(archive_report)
    print(target)
