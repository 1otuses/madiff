import os
import argparse
import json
from pathlib import Path
import numpy as np


def analyze_map(data_root: Path, map_name: str, qualities, out_dir: Path):
    summary = {"map": map_name, "qualities": {}}
    for q in qualities:
        q_dir = data_root / map_name / q
        if not q_dir.exists():
            print(f"  [WARN] {q_dir} not found")
            continue
        print(f"\nAnalyzing {map_name}/{q} -> {q_dir}")
        info = {}
        # expected files
        files = {
            "obs": q_dir / "obs.npy",
            "rewards": q_dir / "rewards.npy",
            "states": q_dir / "states.npy",
            "actions": q_dir / "actions.npy",
            "path_lengths": q_dir / "path_lengths.npy",
            "legals": q_dir / "legals.npy",
            "discounts": q_dir / "discounts.npy",
        }
        for k, p in files.items():
            info[k] = {"exists": p.exists()}
            if p.exists():
                try:
                    arr = np.load(p, mmap_mode='r')
                    info[k]["shape"] = list(arr.shape)
                    info[k]["dtype"] = str(arr.dtype)
                    if k == "actions":
                        try:
                            unique = np.unique(arr)
                            info[k]["n_unique"] = int(unique.size)
                            info[k]["unique_sample"] = unique[:10].tolist()
                        except Exception:
                            pass
                    if k == "rewards":
                        # compute per-timestep team reward stats
                        if arr.ndim == 3:
                            # maybe [T, A, 1] or similar
                            arr_flat = arr.reshape(arr.shape[0], -1)
                        else:
                            arr_flat = arr
                        info[k]["min"] = float(np.min(arr_flat))
                        info[k]["max"] = float(np.max(arr_flat))
                        info[k]["mean"] = float(np.mean(arr_flat))
                        info[k]["std"] = float(np.std(arr_flat))
                    # legal action masks: summarize availability
                    if k == "legals":
                        try:
                            # expected shape: [T, A, n_actions]
                            if arr.ndim >= 3:
                                avail_counts = arr.sum(axis=-1)  # [T, A]
                                info[k]["avg_available_per_agent"] = float(
                                    np.mean(avail_counts)
                                )
                                info[k]["min_available_per_agent"] = float(
                                    np.min(avail_counts)
                                )
                                info[k]["max_available_per_agent"] = float(
                                    np.max(avail_counts)
                                )
                            else:
                                info[k]["note"] = "unexpected shape for legals"
                        except Exception:
                            pass
                    # discounts (if present): basic stats
                    if k == "discounts":
                        try:
                            d = np.array(arr)
                            info[k]["min"] = float(np.min(d))
                            info[k]["max"] = float(np.max(d))
                            info[k]["mean"] = float(np.mean(d))
                            info[k]["std"] = float(np.std(d))
                        except Exception:
                            pass
                except Exception as e:
                    info[k]["load_error"] = str(e)
        # compute episode returns using path_lengths if available
        ep_stats = {}
        if files["path_lengths"].exists() and files["rewards"].exists():
            try:
                lengths = np.load(files["path_lengths"])  # [n_episodes]
                rew = np.load(files["rewards"], mmap_mode='r')  # [T, A] or [T, A, ...]
                total_steps = rew.shape[0]
                pos = 0
                ep_returns = []
                for L in lengths:
                    L = int(L)
                    if pos + L > total_steps:
                        break
                    ep_r = rew[pos:pos + L]
                    # sum team rewards per step then sum
                    if ep_r.ndim > 1:
                        team_r = ep_r.sum(axis=-1)
                    else:
                        team_r = ep_r
                    ep_returns.append(float(team_r.sum()))
                    pos += L
                if len(ep_returns) > 0:
                    ep_stats["n_episodes"] = len(ep_returns)
                    ep_stats["return_min"] = float(np.min(ep_returns))
                    ep_stats["return_max"] = float(np.max(ep_returns))
                    ep_stats["return_mean"] = float(np.mean(ep_returns))
                    ep_stats["return_std"] = float(np.std(ep_returns))
                    ep_stats["returns_sample"] = ep_returns[:10]
            except Exception as e:
                ep_stats["error"] = str(e)
        info["episode_returns"] = ep_stats

        summary["qualities"][q] = info

    # save JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"smac_{map_name}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {out_path}\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze SMAC dataset folders")
    parser.add_argument("--data_root", type=str, default="diffuser/datasets/data/smac",
                        help="path to smac data root")
    parser.add_argument("--maps", type=str, default="3m,8m,2s3z,5m_vs_6m",
                        help="comma separated map names")
    parser.add_argument("--qualities", type=str, default="Good,Medium,Poor",
                        help="comma separated qualities")
    parser.add_argument("--out_dir", type=str, default="custom/diffusion_critic/results/smac_analysis",
                        help="where to save summaries")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    qualities = [q.strip() for q in args.qualities.split(",") if q.strip()]
    out_dir = Path(args.out_dir)

    print(f"Analyzing SMAC data under: {data_root}")
    for m in maps:
        analyze_map(data_root, m, qualities, out_dir)


if __name__ == "__main__":
    main()
