import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import diffuser.utils as utils
import yaml
from diffuser.utils.launcher_util import (
    build_config_from_dict,
    build_nested_variant_generator,
)


def build_eval_configs(exp_specs):
    """兼容直接参数 yaml 和 launcher 风格 exp spec。"""
    if {"meta_data", "variables", "constants"}.issubset(exp_specs.keys()):
        variant_generator = build_nested_variant_generator(exp_specs)
        return [build_config_from_dict(variant) for variant in variant_generator()]
    return [build_config_from_dict(exp_specs)]


def should_save_eval_plan_images_for_step(Config, load_step):
    """只在配置中最后一个 load_step 保存评估轨迹图。"""
    return (
        getattr(Config, "save_eval_plan_images", False)
        and len(getattr(Config, "load_steps", [])) > 0
        and load_step == Config.load_steps[-1]
    )


def evaluate(Config):
    evaluator = None
    Config.condition_guidance_w = getattr(Config, "condition_guidance_w", None)
    Config.mode_guidance_w = getattr(Config, "mode_guidance_w", None)

    for load_step in Config.load_steps:
        ckpt_file_path = os.path.join(
            Config.log_dir, f"checkpoint/state_{load_step}.pt"
        )
        if not os.path.exists(ckpt_file_path):
            print(f"Checkpoint file {ckpt_file_path} not found. Skipping evaluation.")
            continue

        results_file_path = os.path.join(
            Config.log_dir,
            f"results/step_{load_step}-ep_{Config.num_eval}-ddim.json"
            if getattr(Config, "use_ddim_sample", False)
            else f"results/step_{load_step}-ep_{Config.num_eval}.json",
        )
        if Config.condition_guidance_w is not None:
            results_file_path = results_file_path.replace(
                ".json", f"-cg_{Config.condition_guidance_w}.json"
            )
        if Config.mode_guidance_w is not None:
            results_file_path = results_file_path.replace(
                ".json", f"-mg_{Config.mode_guidance_w}.json"
            )
        if getattr(Config, "use_return_to_go", False):
            results_file_path = results_file_path.replace(".json", "-rtg.json")
        if not Config.overwrite and os.path.exists(results_file_path):
            print(
                f"Results file {results_file_path} already exist. Skipping evaluation."
            )
            continue

        if evaluator is None:
            evaluator_config = utils.Config(Config.evaluator, verbose=True)
            evaluator = evaluator_config()
            evaluator_kwargs = dict(
                log_dir=Config.log_dir,
                num_eval=Config.num_eval,
                num_envs=getattr(Config, "num_envs", Config.num_eval),
                condition_guidance_w=Config.condition_guidance_w,
                mode_guidance_w=Config.mode_guidance_w,
                use_ddim_sample=Config.use_ddim_sample,
                n_ddim_steps=Config.n_ddim_steps,
            )
            for key in [
                "save_eval_plan_images",
                "eval_plan_num_episodes",
                "eval_plan_plot_steps",
                "eval_plan_rollout_horizon",
                "eval_plan_anchor_start",
                "eval_plan_grid_cols",
                "save_eval_plan_npz",
                "use_tensorboard",
                "local_prefix_eval",
                "local_prefixes",
                "local_confidence_threshold",
                "environment",
                "max_episode_steps",
                "num_videos",
                "video_fps",
                "eval_seed",
                "test_ret",
                "use_return_to_go",
            ]:
                if hasattr(Config, key):
                    evaluator_kwargs[key] = getattr(Config, key)
            evaluator.init(**evaluator_kwargs)

        evaluator.evaluate(
            load_step=load_step,
            save_eval_plan_images=should_save_eval_plan_images_for_step(
                Config,
                load_step,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)
    for Config in build_eval_configs(exp_specs):
        evaluate(Config)
