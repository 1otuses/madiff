"""通过原项目 ``run_scripts/train.py`` 训练 mode-consistent 两个阶段。"""

import math
import os

import torch
from ml_logger import logger

import diffuser.utils as utils
from diffuser.utils.launcher_util import discover_latest_checkpoint_path


MODE_TRAINING_STAGES = {"mode_stage1", "mode_stage2"}


class NullRenderer:
    """Mode 离线训练不生成环境渲染；保留 Trainer 的 renderer 契约。"""

    def __init__(self, **kwargs):
        del kwargs


def is_mode_training_stage(stage: str) -> bool:
    return stage in MODE_TRAINING_STAGES


def train_mode_stage(Config, RUN) -> None:
    """构造新架构组件，并复用 MADiff 的 Config、Trainer 和 checkpoint。"""

    del RUN
    stage = Config.training_stage
    if stage not in MODE_TRAINING_STAGES:
        raise ValueError("unknown mode training stage: {}".format(stage))

    dataset_config = utils.Config(
        getattr(Config, "loader", "mode_consistent.data.ModeSequenceDataset"),
        savepath="dataset_config.pkl",
        dataset_path=Config.dataset_path,
        objective=stage,
        horizon=Config.horizon,
        split="train",
        stride=getattr(Config, "stride", 1),
        eval_fraction=getattr(Config, "eval_fraction", 0.0),
        seed=Config.seed,
        max_n_episodes=getattr(Config, "max_n_episodes", None),
        use_padding=getattr(Config, "use_padding", True),
        normalizer=getattr(Config, "normalizer", "CDFNormalizer"),
        use_inv_dyn=stage == "mode_stage2",
        include_returns=(
            stage == "mode_stage2" and getattr(Config, "returns_condition", True)
        ),
        returns_scale=getattr(Config, "returns_scale", 400.0),
        discount=getattr(Config, "discount", 0.99),
        history_horizon=getattr(Config, "history_horizon", 0),
        n_agents=getattr(Config, "n_agents", None),
    )
    render_config = utils.Config(
        getattr(Config, "renderer", "mode_consistent.training.NullRenderer"),
        savepath="render_config.pkl",
    )
    dataset = dataset_config()
    renderer = render_config()

    mode_model_kwargs = {
        "n_modes": Config.n_modes,
        "hidden_dim": getattr(Config, "mode_hidden_dim", 128),
        "mode_dim": getattr(Config, "mode_dim", 32),
        "agent_embedding_dim": getattr(Config, "agent_embedding_dim", 8),
        "prior_hidden_dim": getattr(Config, "prior_hidden_dim", None),
        "codebook_weight": getattr(Config, "codebook_weight", 1.0),
        "commitment_weight": getattr(Config, "commitment_weight", 0.25),
    }

    if stage == "mode_stage1":
        model_config = utils.Config(
            getattr(Config, "model", "mode_consistent.models.TeamModeVQVAE"),
            savepath="model_config.pkl",
            n_agents=dataset.n_agents,
            observation_dim=dataset.observation_dim,
            action_dim=dataset.action_dim,
            **mode_model_kwargs,
        )
        objective_config = utils.Config(
            getattr(
                Config,
                "diffusion",
                "mode_consistent.objectives.ModeVQObjective",
            ),
            savepath="diffusion_config.pkl",
            train_teacher=True,
            train_prior=True,
            prior_loss_weight=getattr(Config, "prior_loss_weight", 1.0),
            random_prior_prefix=getattr(Config, "random_prior_prefix", True),
        )
        model = model_config().to(Config.device)
        objective = objective_config(model).to(Config.device)
    else:
        if not getattr(Config, "use_inv_dyn", True):
            raise ValueError("mode_stage2 requires use_inv_dyn=True")
        if getattr(Config, "joint_inv", False):
            raise ValueError("mode_stage2 forbids joint inverse dynamics")
        model_config = utils.Config(
            getattr(Config, "model", "mode_consistent.models.ModeTemporalUnet"),
            savepath="model_config.pkl",
            n_agents=dataset.n_agents,
            horizon=Config.horizon,
            history_horizon=getattr(Config, "history_horizon", 0),
            transition_dim=dataset.observation_dim,
            n_modes=Config.n_modes,
            mode_dim=getattr(Config, "mode_dim", 32),
            agent_embedding_dim=getattr(Config, "agent_embedding_dim", 8),
            dim=getattr(Config, "dim", 128),
            dim_mults=tuple(getattr(Config, "dim_mults", (1, 2, 4, 8))),
            returns_condition=getattr(Config, "returns_condition", True),
            env_ts_condition=False,
            condition_dropout=0.0,
            mode_condition_dropout=0.0,
            kernel_size=getattr(Config, "kernel_size", 5),
            max_path_length=dataset.max_path_length,
            freeze_mode_codebook=True,
        )
        objective_config = utils.Config(
            getattr(
                Config,
                "diffusion",
                "mode_consistent.models.ModeGaussianDiffusion",
            ),
            savepath="diffusion_config.pkl",
            n_agents=dataset.n_agents,
            horizon=Config.horizon,
            history_horizon=0,
            observation_dim=dataset.observation_dim,
            action_dim=dataset.action_dim,
            use_inv_dyn=True,
            discrete_action=False,
            train_only_inv=False,
            share_inv=getattr(Config, "share_inv", True),
            joint_inv=False,
            n_timesteps=getattr(Config, "n_diffusion_steps", 200),
            clip_denoised=getattr(Config, "clip_denoised", True),
            predict_epsilon=getattr(Config, "predict_epsilon", True),
            action_weight=getattr(Config, "action_weight", 1.0),
            hidden_dim=getattr(Config, "hidden_dim", 256),
            loss_discount=getattr(Config, "loss_discount", 1.0),
            returns_condition=getattr(Config, "returns_condition", True),
            condition_guidance_w=getattr(Config, "condition_guidance_w", 1.2),
            data_encoder=utils.IdentityEncoder(),
            mode_model_kwargs=mode_model_kwargs,
            mode_source_train=getattr(Config, "mode_source_train", "posterior"),
            local_prefix_train=getattr(Config, "local_prefix_train", 1),
            local_confidence_threshold=getattr(
                Config, "local_confidence_threshold", 0.0
            ),
            cfg_dropout_prob=getattr(Config, "cfg_dropout_prob", 0.05),
            mode_guidance_w=getattr(Config, "mode_guidance_w", 1.5),
        )
        model = model_config().to(Config.device)
        objective = objective_config(model).to(Config.device)
        objective.load_pretrained_mode(
            _required_mode_checkpoint(Config),
            use_ema=getattr(Config, "mode_checkpoint_use_ema", True),
        )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath="trainer_config.pkl",
        train_batch_size=getattr(Config, "batch_size", 64),
        train_lr=getattr(Config, "learning_rate", 3e-4),
        gradient_accumulate_every=getattr(Config, "gradient_accumulate_every", 1),
        ema_decay=getattr(Config, "ema_decay", 0.995),
        sample_freq=0,
        save_freq=getattr(Config, "save_freq", 1000),
        log_freq=getattr(Config, "log_freq", 100),
        label_freq=max(1, int(getattr(Config, "n_train_steps", 1))),
        eval_freq=getattr(Config, "eval_freq", 0),
        save_parallel=False,
        bucket=logger.root,
        n_reference=0,
        train_device=Config.device,
        save_checkpoints=getattr(Config, "save_checkpoints", True),
        use_tensorboard=getattr(Config, "use_tensorboard", True),
        show_progress=getattr(Config, "show_progress", True),
        progress_position=getattr(Config, "exp_id", 0),
        progress_desc="{} | seed={}".format(stage, Config.seed),
    )
    evaluator_name = getattr(
        Config,
        "evaluator",
        (
            "mode_consistent.evaluation.ModeConditionedEvaluator"
            if stage == "mode_stage2"
            else "mode_consistent.evaluation.ModeVQEvaluator"
        ),
    )
    evaluator_config = utils.Config(
        evaluator_name,
        savepath="evaluator_config.pkl",
        verbose=False,
    )

    trainer = trainer_config(objective, dataset, renderer)
    if getattr(Config, "eval_freq", 0) > 0:
        evaluator = evaluator_config()
        evaluator.init(log_dir=logger.prefix)
        trainer.set_evaluator(evaluator)

    if getattr(Config, "continue_training", False):
        loadpath = discover_latest_checkpoint_path(
            os.path.join(trainer.bucket, logger.prefix, "checkpoint")
        )
        if loadpath is not None:
            state_dict = torch.load(loadpath, map_location=Config.device)
            trainer.step = state_dict["step"]
            trainer.model.load_state_dict(state_dict["model"])
            trainer.ema_model.load_state_dict(state_dict["ema"])

    utils.report_parameters(objective)
    logger.print("Testing mode stage forward...", end=" ", flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    loss, _ = objective.loss(**batch)
    loss.backward()
    objective.zero_grad(set_to_none=True)
    logger.print("✓")

    n_train_steps = int(getattr(Config, "n_train_steps", 0))
    steps_per_epoch = int(getattr(Config, "n_steps_per_epoch", n_train_steps or 1))
    remaining = max(0, n_train_steps - trainer.step)
    total_epochs = int(math.ceil(remaining / steps_per_epoch)) if remaining else 0
    for epoch in range(total_epochs):
        current_steps = min(steps_per_epoch, n_train_steps - trainer.step)
        trainer.train(
            n_train_steps=current_steps,
            epoch=epoch,
            total_epochs=total_epochs,
        )
    trainer.finish_training()


def _required_mode_checkpoint(Config) -> str:
    checkpoint = getattr(Config, "pretrained_mode_checkpoint", None)
    if not checkpoint:
        raise ValueError("mode_stage2 requires pretrained_mode_checkpoint")
    return checkpoint
