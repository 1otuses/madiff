#!/usr/bin/env python
"""评估 CI-CoDiff 的冻结阶段 checkpoint，只生成待讨论证据。"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_eval_configs(spec):
    if {"meta_data", "variables", "constants"}.issubset(spec):
        from diffuser.utils.launcher_util import build_nested_variant_generator

        return list(build_nested_variant_generator(spec)())
    return [spec]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--experiment", required=True, help="评估 YAML")
    parser.add_argument("-g", "--gpu", default="0", help="物理 GPU 编号")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import yaml

    from mode_consistent.experiments import (
        evaluate_central_comparison,
        evaluate_central_experiment,
        evaluate_unlabeled_central_comparison,
    )

    with Path(args.experiment).open("r", encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for config in _build_eval_configs(spec):
        if config.get("stage") != "central":
            parser.error("当前只开放 stage=central；其他阶段需讨论确认后实现")
        if config.get("protocol") == "comparison":
            runs = {
                name: {
                    **run,
                    "log_dir": str(
                        Path(run["log_dir"])
                        if Path(run["log_dir"]).is_absolute()
                        else PROJECT_ROOT / run["log_dir"]
                    ),
                }
                for name, run in config["runs"].items()
            }
            output_path = Path(config["output_path"])
            if not output_path.is_absolute():
                output_path = PROJECT_ROOT / output_path
            result = evaluate_central_comparison(
                runs=runs,
                output_path=output_path,
                device=device,
                batch_size=int(config.get("batch_size", 256)),
                overwrite=bool(config.get("overwrite", False)),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            continue
        if config.get("protocol") == "unlabeled_comparison":
            runs = {
                name: {
                    **run,
                    "log_dir": str(
                        Path(run["log_dir"])
                        if Path(run["log_dir"]).is_absolute()
                        else PROJECT_ROOT / run["log_dir"]
                    ),
                }
                for name, run in config["runs"].items()
            }
            output_path = Path(config["output_path"])
            if not output_path.is_absolute():
                output_path = PROJECT_ROOT / output_path
            result = evaluate_unlabeled_central_comparison(
                runs=runs,
                output_path=output_path,
                device=device,
                batch_size=int(config.get("batch_size", 256)),
                overwrite=bool(config.get("overwrite", False)),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            continue
        log_dir = Path(config["log_dir"])
        if not log_dir.is_absolute():
            log_dir = PROJECT_ROOT / log_dir
        for load_step in config["load_steps"]:
            result = evaluate_central_experiment(
                log_dir=log_dir,
                load_step=int(load_step),
                device=device,
                batch_size=int(config.get("batch_size", 256)),
                overwrite=bool(config.get("overwrite", False)),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
