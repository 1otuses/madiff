"""通过原 MADiff 训练入口启动四种智能体交互方法的实验."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
METHODS = ("attention", "identity", "pool", "mpnn")
PRIVATE_DIR = Path.home() / ".cache/codex/madiff/non_attention_v1/launch_1m"


def make_config(settings, method, seed=None, updates=None, resume=False):
    from diffuser.utils.launcher_util import build_nested_variant_generator

    spec = yaml.safe_load((PROJECT_ROOT / settings["baseline_spec"]).read_text())
    config = dict(next(build_nested_variant_generator(spec)()))
    config.update(deepcopy(settings["overrides"]))
    config.update(agent_interaction=method, continue_training=resume)
    config["initialization_path"] = str(Path(config["initialization_path"]).expanduser())
    config["job_name"] += "-interaction_{agent_interaction}"
    config["eval_seeds"] = list(range(20000, 20000 + config["num_eval"]))
    if seed is not None:
        config["seed"] = seed
    if updates is not None:
        config["n_train_steps"] = updates
    return config


def run(args):
    settings = yaml.safe_load(args.config.read_text())
    methods = METHODS if args.method == "all" else (args.method,)
    gpus = args.gpu.split(",")
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for offset in range(0, len(methods), len(gpus)):
        running = []
        for method, gpu in zip(methods[offset:offset + len(gpus)], gpus):
            config = make_config(settings, method, args.seed, args.updates, args.resume)
            run_dir = PROJECT_ROOT / "logs" / config["exp_name"] / (
                config["job_name"].format(**config)
            ) / str(config["seed"])
            if not args.dry_run and not args.resume and run_dir.exists():
                raise FileExistsError(f"Run already exists: {run_dir}; use --resume to continue.")
            name = f"{method}_{config['seed']}"
            config_path = PRIVATE_DIR / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            command = [sys.executable, "-u", "run_scripts/train.py", "-e", str(config_path), "-g", gpu]
            print(f"{method}: {config['n_train_steps']} updates, GPU {gpu}, {run_dir}", flush=True)
            if args.dry_run:
                continue
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
            with (PRIVATE_DIR / f"{name}.stdout.log").open("a") as stdout, (
                PRIVATE_DIR / f"{name}.stderr.log"
            ).open("a") as stderr:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                           stdout=stdout, stderr=stderr, start_new_session=True)
            record = {"method": method, "gpu": gpu, "pid": process.pid,
                      "run_dir": str(run_dir), "command": command, "status": "running"}
            records.append(record)
            running.append((process, record))
            (PRIVATE_DIR / "processes.json").write_text(json.dumps(records, indent=2))
        for process, record in running:
            record.update(exit_code=process.wait(), status="finished")
            (PRIVATE_DIR / "processes.json").write_text(json.dumps(records, indent=2))
    return int(any(record.get("exit_code", 0) != 0 for record in records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument("--gpu", default="3", help="One GPU for serial runs, or comma-separated GPUs.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "exp_specs/mpe/spread/non_attention.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write private input configs without training.")
    sys.exit(run(parser.parse_args()))
