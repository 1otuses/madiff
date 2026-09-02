import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 Python 路径中
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import diffuser.utils as utils
import torch
import yaml
from diffuser.utils.launcher_util import (
    build_config_from_dict,
    discover_latest_checkpoint_path,
)


def check_forward_backward(diffusion, dataset, device):
    batch = utils.batchify(dataset[0], device)
    diffusion.zero_grad(set_to_none=True)
    loss, _ = diffusion.loss(**batch)
    loss.backward()
    diffusion.zero_grad(set_to_none=True)


def main(Config, RUN):
    # 禁用cudnn的benchmark模式以确保结果的可重复性
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    utils.set_seed(Config.seed)
    dataset_extra_kwargs = dict()

    # 旧版 YAML 文件中不存在的配置项
    Config.discrete_action = getattr(Config, "discrete_action", False) # 是否使用离散动作空间
    Config.state_loss_weight = getattr(Config, "state_loss_weight", None) # 状态损失权重
    Config.opponent_loss_weight = getattr(Config, "opponent_loss_weight", None) # 对手状态损失权重
    Config.use_seed_dataset = getattr(Config, "use_seed_dataset", False)
    Config.residual_attn = getattr(Config, "residual_attn", True) # 是否使用残差注意力机制
    Config.use_temporal_attention = getattr(Config, "use_temporal_attention", True) # 是否使用时间注意力机制
    Config.env_ts_condition = getattr(Config, "env_ts_condition", False) # 是否使用环境时间步作为条件
    Config.use_return_to_go = getattr(Config, "use_return_to_go", False) # 是否使用回报作为条件
    Config.joint_inv = getattr(Config, "joint_inv", False)
    Config.use_zero_padding = getattr(Config, "use_zero_padding", True)
    Config.use_inv_dyn = getattr(Config, "use_inv_dyn", True) # 是否使用逆动力学模型
    Config.pred_future_padding = getattr(Config, "pred_future_padding", False)
    if not hasattr(Config, "agent_condition_type"):
        if Config.decentralized_execution: # DE=True,agent只使用自己局部观测作为策略条件
            Config.agent_condition_type = "single"
        else: # DE=False,表示中心化执行
            Config.agent_condition_type = "all"

    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#
    # 创建数据集配置对象
    dataset_config = utils.Config(
        Config.loader,
        savepath="dataset_config.pkl",
        env_type=Config.env_type,
        env=Config.dataset,
        n_agents=Config.n_agents,
        horizon=Config.horizon,
        history_horizon=Config.history_horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        max_n_episodes=Config.max_n_episodes,
        use_padding=Config.use_padding,
        use_action=Config.use_action,
        discrete_action=Config.discrete_action,
        max_path_length=Config.max_path_length,
        include_returns=Config.returns_condition,
        include_env_ts=Config.env_ts_condition,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
        agent_share_parameters=utils.config.import_class(
            Config.model
        ).agent_share_parameters,
        use_seed_dataset=Config.use_seed_dataset,
        seed=Config.seed,
        use_inv_dyn=Config.use_inv_dyn,
        decentralized_execution=Config.decentralized_execution,
        use_zero_padding=Config.use_zero_padding,
        agent_condition_type=Config.agent_condition_type,
        pred_future_padding=Config.pred_future_padding,
        **dataset_extra_kwargs,
    )
    # 创建渲染器配置对象
    render_config = utils.Config(
        Config.renderer,
        savepath="render_config.pkl",
        env_type=Config.env_type,
        env=Config.dataset,
    )
    # 创建数据编码器配置对象
    data_encoder_config = utils.Config(
        getattr(Config, "data_encoder", "utils.IdentityEncoder"),
        savepath="data_encoder_config.pkl",
    )

    dataset = dataset_config()
    renderer = render_config()
    data_encoder = data_encoder_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    # 创建模型配置对象
    model_config = utils.Config(
        Config.model,
        savepath="model_config.pkl",
        n_agents=Config.n_agents,
        horizon=Config.horizon + Config.history_horizon, # 历史条件轨迹+未来规划轨迹
        history_horizon=Config.history_horizon,
        transition_dim=observation_dim,
        dim_mults=Config.dim_mults,
        returns_condition=Config.returns_condition,
        env_ts_condition=Config.env_ts_condition,
        dim=Config.dim,
        condition_dropout=Config.condition_dropout,
        residual_attn=Config.residual_attn,
        max_path_length=Config.max_path_length,
        use_temporal_attention=Config.use_temporal_attention,
        device=Config.device,
    )
    # 创建扩散模型配置对象
    diffusion_config = utils.Config(
        Config.diffusion,
        savepath="diffusion_config.pkl",
        n_agents=Config.n_agents,
        horizon=Config.horizon,
        history_horizon=Config.history_horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        discrete_action=Config.discrete_action,
        num_actions=getattr(dataset.env, "num_actions", 0),
        n_timesteps=Config.n_diffusion_steps,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        hidden_dim=Config.hidden_dim,
        train_only_inv=Config.train_only_inv,
        share_inv=Config.share_inv,
        joint_inv=Config.joint_inv,
        # loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        state_loss_weight=Config.state_loss_weight,
        opponent_loss_weight=Config.opponent_loss_weight,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        condition_guidance_w=Config.condition_guidance_w,
        data_encoder=data_encoder,
        use_inv_dyn=Config.use_inv_dyn,
        device=Config.device,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath="trainer_config.pkl",
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        eval_freq=Config.eval_freq,
        save_parallel=Config.save_parallel,
        bucket=logger.root,
        n_reference=Config.n_reference,
        train_device=Config.device,
        use_tensorboard=getattr(Config, "use_tensorboard", True),
        save_checkpoints=Config.save_checkpoints,
        show_progress=getattr(Config, "show_progress", True),
        progress_position=getattr(Config, "exp_id", 0),
        progress_desc=f"{Config.dataset} | seed={Config.seed}",
    )

    evaluator_config = utils.Config(
        Config.evaluator,
        savepath="evaluator_config.pkl",
        verbose=False,
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)
    # Trainer
    # ├── diffusion: GaussianDiffusion
    # │   └── model: SharedConvAttentionDeconv
    # ├── ema_model: GaussianDiffusion 的 EMA 副本
    # ├── optimizer
    # ├── dataset / dataloader
    # ├── renderer
    # └── evaluator

    if Config.eval_freq > 0:
        evaluator = evaluator_config()
        evaluator.init(log_dir=logger.prefix)
        trainer.set_evaluator(evaluator) # 转至evaluate状态

    if Config.continue_training:
        loadpath = discover_latest_checkpoint_path( # 最大保存模型步数开始,从该位置开始训练
            os.path.join(trainer.bucket, logger.prefix, "checkpoint")
        )
        if loadpath is not None:
            trainer.load(loadpath)
            logger.print(
                f"\nLoaded checkpoint from {loadpath} (step {trainer.step})\n",
                color="green",
            )

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    logger.print("Testing forward...", end=" ", flush=True) # 检查前向和后向pipline是否打通
    check_forward_backward(diffusion, dataset, Config.device)
    logger.print("✓")

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#
    # 主训练循环
    n_epochs = int((Config.n_train_steps - trainer.step) // Config.n_steps_per_epoch)
    # n_train_steps: 总训练步数
    # step: 当前训练步数
    # n_steps_per_epoch: 每个epoch的训练步数

    for i in range(n_epochs):
        logger.print(f"Epoch {i + 1} / {n_epochs} | {logger.prefix}")
        trainer.train(
            n_train_steps=Config.n_steps_per_epoch,
            epoch=i,
            total_epochs=n_epochs,
        )
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
    # logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.remove("parameters.pkl")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text(
        """
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: a0_loss
                      xKey: steps
                    """,
        filename=".charts.yml",
        dedent=True,
        overwrite=True,
    )
    logger.save_yaml(exp_specs, "exp_specs.yml")

    main(Config, RUN)
