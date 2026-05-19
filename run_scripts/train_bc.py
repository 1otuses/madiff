import argparse
import os

import diffuser.utils as utils
import torch
import yaml
from diffuser.utils.launcher_util import build_config_from_dict, discover_latest_checkpoint_path


def main(Config, RUN):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        Config.loader,
        savepath="dataset_config.pkl",
        env_type=Config.env_type,
        env=Config.dataset,
        n_agents=Config.n_agents,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        max_n_episodes=Config.max_n_episodes,
        max_path_length=Config.max_path_length,
        agent_share_parameters=utils.config.import_class(
            Config.model
        ).agent_share_parameters,
    )

    dataset = dataset_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    model_kwargs = {
        "n_agents": Config.n_agents,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
    }
    if hasattr(Config, "dim"):
        model_kwargs["dim"] = Config.dim
    if hasattr(Config, "dim_mults"):
        model_kwargs["dim_mults"] = Config.dim_mults
    if hasattr(Config, "init_w"):
        model_kwargs["init_w"] = Config.init_w
    if hasattr(Config, "conditioned_std"):
        model_kwargs["conditioned_std"] = Config.conditioned_std

    model_config = utils.Config(
        Config.model,
        savepath="model_config.pkl",
        **model_kwargs,
    )

    bc_config = utils.Config(
        utils.BehaviorClone,
        savepath="bc_config.pkl",
        observation_dim=observation_dim,
        action_dim=action_dim,
    )

    trainer_config = utils.Config(
        utils.BCTrainer,
        savepath="trainer_config.pkl",
        train_batch_size=getattr(Config, "batch_size", 32),
        train_lr=getattr(Config, "learning_rate", 2e-5),
        log_freq=getattr(Config, "log_freq", 100),
        save_freq=getattr(Config, "save_freq", 1000),
        eval_freq=getattr(Config, "eval_freq", 10000),
        bucket=logger.root,
        train_device=Config.device,
        save_checkpoints=getattr(Config, "save_checkpoints", False),
        use_tensorboard=getattr(Config, "use_tensorboard", True),
    )

    evaluator_config = utils.Config(
        utils.BCEvaluator,
        savepath="evaluator_config.pkl",
        verbose=False,
    )

    model = model_config()
    bc = bc_config(model)
    trainer = trainer_config(bc, dataset)

    if Config.eval_freq > 0:
        evaluator = evaluator_config()
        evaluator.init(log_dir=logger.prefix)
        trainer.set_evaluator(evaluator)

    if getattr(Config, "continue_training", False):
        loadpath = discover_latest_checkpoint_path(
            os.path.join(trainer.bucket, logger.prefix, "checkpoint")
        )
        if loadpath is not None:
            state_dict = torch.load(loadpath, map_location=Config.device)
            logger.print(
                f"\nLoaded checkpoint from {loadpath} (step {state_dict['step']})\n",
                color="green",
            )
            trainer.step = state_dict["step"]
            trainer.model.load_state_dict(state_dict["model"])

    utils.report_parameters(model)

    n_train_steps = getattr(Config, "n_train_steps", None)
    if n_train_steps is None:
        raise ValueError("Config.n_train_steps must be set for BC training")

    trainer.train(n_train_steps=n_train_steps)
    trainer.finish_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=str, default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)

    from ml_logger import RUN, logger

    Config = build_config_from_dict(exp_specs)

    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    job_name = Config.job_name.format(**vars(Config))
    RUN.prefix, RUN.job_name, _ = RUN(
        script_path=__file__,
        exp_name=exp_specs["exp_name"],
        job_name=job_name + f"/{Config.seed}",
    )

    logger.configure(RUN.prefix, root=RUN.script_root)
    logger.remove("traceback.err")
    logger.remove("parameters.pkl")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.save_yaml(exp_specs, "exp_specs.yml")

    main(Config, RUN)
