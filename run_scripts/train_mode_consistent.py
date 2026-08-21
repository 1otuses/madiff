#!/usr/bin/env python
"""按实验 YAML 训练 CI-CoDiff 的单个已开放阶段。"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--experiment", required=True, help="展开后的实验 YAML")
    parser.add_argument("-g", "--gpu", default="0", help="物理 GPU 编号")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import yaml
    from ml_logger import RUN, logger

    from mode_consistent.experiments import (
        train_central_experiment,
        train_central_validation_baseline,
    )

    with Path(args.experiment).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if {"meta_data", "variables", "constants"}.intersection(config):
        parser.error("训练入口只接受 run_experiment.py 展开后的单个 variant YAML")
    if config.get("stage") != "central":
        parser.error("当前只开放 stage=central；其他阶段需讨论确认后实现")

    job_name = config["job_name"].format(**config)
    RUN.prefix, RUN.job_name, _ = RUN(
        script_path=__file__,
        exp_name=config["exp_name"],
        job_name=f"{job_name}/{config['seed']}",
    )
    logger.configure(RUN.prefix, root=RUN.script_root)
    logger.remove("traceback.err")
    logger.log_params(Config=config, RUN=vars(RUN))
    logger.save_yaml(config, "exp_specs.yml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(RUN.script_root) / RUN.prefix
    model_kind = config.get("model_kind", "vq")
    if model_kind == "vq":
        summary = train_central_experiment(config, run_dir, device)
    else:
        summary = train_central_validation_baseline(config, run_dir, device)
    logger.print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
