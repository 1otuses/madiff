from typing import Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler # DDIM采样调度器(加速采样，通常15~50步)
from diffusers.schedulers.scheduling_consistency_models import CMStochasticIterativeScheduler  # 一致性模型调度器(单步/少步采样)


import diffuser.utils as utils
from diffuser.models.helpers import Losses, apply_conditioning


class GaussianDiffusion(nn.Module):
    # 实现高斯扩散模型
    def __init__(
        self,
        model,
        n_agents: int,
        horizon: int,
        history_horizon: int,
        observation_dim: int,
        action_dim: int,
        use_inv_dyn: bool = True,
        discrete_action: bool = False,
        num_actions: int = 0,  # 离散动作空间的动作数量
        n_timesteps: int = 1000,  # 扩散步长
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
        action_weight: float = 1.0,
        hidden_dim: int = 256,
        loss_discount: float = 1.0,
        loss_weights: np.ndarray = None,
        state_loss_weight: float = None,
        opponent_loss_weight: float = None,
        returns_condition: bool = False,
        condition_guidance_w: float = 1.2,
        returns_loss_guided: bool = False,
        loss_guidence_w: float = 0.1,  # 损失引导权重，默认0.1
        value_diffusion_model: nn.Module = None,
        train_only_inv: bool = False,
        share_inv: bool = True,  # 是否共享逆动力学模型（所有智能体使用同一个模型），默认True
        joint_inv: bool = False,  # 是否采用联合逆动力学模型，默认False
        data_encoder: utils.Encoder = utils.IdentityEncoder(),
        **kwargs,
    ):
        assert action_dim > 0
        assert (
            not returns_condition or not returns_loss_guided
        ), "Can't do both returns conditioning and returns loss guidence"

        super().__init__()
        # ========== 基础信息 ==========
        self.n_agents = n_agents  # N
        self.horizon = horizon  # T
        self.history_horizon = history_horizon  # H
        self.observation_dim = observation_dim  # O
        self.action_dim = action_dim  # A
        self.state_loss_weight = state_loss_weight
        self.opponent_loss_weight = opponent_loss_weight
        self.discrete_action = discrete_action
        self.num_actions = num_actions  # 离散动作空间的动作数量
        self.transition_dim = observation_dim + action_dim  # O + A
        # ========== 模型参数 ==========
        self.model = model
        self.use_inv_dyn = use_inv_dyn
        self.train_only_inv = train_only_inv
        self.share_inv = share_inv  # 是否共享逆动力学模型,默认True
        self.joint_inv = joint_inv  # 是否采用联合逆动力学模型,默认False
        self.data_encoder = data_encoder

        if self.use_inv_dyn:
            self.inv_model = self._build_inv_model(
                hidden_dim,
                output_dim=action_dim if not discrete_action else num_actions,
            )
        # ========== 条件生成设置 ==========
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        # returns_condition 与 returns_loss_guided 不能同时为 True
        self.returns_loss_guided = returns_loss_guided 
        self.loss_guidence_w = loss_guidence_w
        self.value_diffusion_model = value_diffusion_model
        if self.value_diffusion_model is not None:
            self.value_diffusion_model.requires_grad_(False)
        # ========== 扩散过程参数 ==========
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.n_timesteps,
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
        )
        self.use_ddim_sample = False  # 是否使用DDIM采样
        self.use_consistency_models_sample = False  # 是否使用一致性模型采样

        # ========== 损失函数设置 ==========
        # 获取损失权重并初始化优化目标
        loss_weights = self.get_loss_weights(loss_discount, action_weight)
        loss_type = "state_l2" if self.use_inv_dyn else "l2"
        self.loss_fn = (
            Losses[loss_type](loss_weights)
            if self.use_inv_dyn
            else Losses[loss_type](loss_weights, self.action_dim)
        )

    def _build_inv_model(self, hidden_dim: int, output_dim: int):
        # 构建逆动力学模型
        r'''
        \hat a_t ≈ inv_model( [o_t, o_{t+1}] )
        joint_inv: 联合动力学模型,将agents的观测o_i拼接,同时输出[a_1, ..., a_N].
        share_inv: 共享逆动力学模型,agents共享模型参数,agent独立输入o_i,输出动作a_i.
        independent_inv: 独立逆动力学模型,agent独立分配网络参数,独立输入o_i,输出动作a_i.
        '''
        if self.joint_inv:
            print("\n USE JOINT INV \n")
            # [B*T, N*2*O]--->[B*T, N*A]
            inv_model = nn.Sequential(
                nn.Linear(self.n_agents * (2 * self.observation_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_agents * output_dim),
            )

        elif self.share_inv:
            print("\n USE SHARED INV \n")
            # [B*T*N, 2*O]--->[B*T*N, A]
            inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        else:
            print("\n USE INDEPENDENT INV \n")
            inv_model = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * self.observation_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, output_dim),
                        nn.Softmax(dim=-1) if self.discrete_action else nn.Identity(),
                    )
                    for _ in range(self.n_agents)
                ]
            )

        return inv_model

    def set_ddim_scheduler(self, n_ddim_steps: int = 15):
        # 设置DDIM调度器
        # DDIM调度器用于在采样过程中生成高斯噪声，以模拟数据的生成过程
        # 该调度器通过调整噪声的方差，使模型在不同时间步长上生成的样本更符合真实数据的分布
        # 这允许模型在生成样本时，根据噪声的方差来控制样本的随机性，从而实现更真实的样本生成
        self.ddim_noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.n_timesteps,
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
        )
        self.ddim_noise_scheduler.set_timesteps(n_ddim_steps)
        self.use_ddim_sample = True
    
    def set_consistency_models_scheduler(self, n_consistency_model_steps: int = 5):
        # 设置一致性模型调度器
        # 一致性模型调度器用于在采样过程中生成高斯噪声，以模拟数据的生成过程
        # 该调度器通过调整噪声的方差，使模型在不同时间步长上生成的样本更符合真实数据的分布
        # 这允许模型在生成样本时，根据噪声的方差来控制样本的随机性，从而实现更真实的样本生成
        self.consistency_models_scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps = self.n_timesteps,
            sigma_min = 0.002,   # 最小噪声水平
            sigma_max = 80,      # 最大噪声水平
            rho = 7.0            # 调度曲线形状参数
        )
        self.consistency_models_scheduler.set_timesteps(n_consistency_model_steps)
        self.use_consistency_models_sample = True

    def get_loss_weights(self, discount: float, action_weight: Optional[float] = None):
        """
        设置轨迹各时间步和维度的损失权重。

        discount : float
            第 t 个轨迹时间步的损失乘以 discount**t。
        """

        if self.use_inv_dyn:
            dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)
        else:
            dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # 按轨迹时间步衰减损失：discount**t。
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        discounts = torch.cat([torch.zeros(self.history_horizon), discounts])
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        loss_weights = loss_weights.unsqueeze(1).expand(-1, self.n_agents, -1).clone()

        # 手动设置第一个动作位置的权重。
        if not self.use_inv_dyn:
            loss_weights[self.history_horizon, :, : self.action_dim] = action_weight
        return loss_weights

    # ------------------------------------------ 采样 ------------------------------------------#

    def get_model_output(
        self,
        x: torch.Tensor, # [B, T+H, N, O]带噪输入x_t
        t: torch.Tensor, # [B] 当前扩散时间步长t
        returns: Optional[torch.Tensor] = None, # [B, 1, N]条件return-to-go编码
        env_ts: Optional[torch.Tensor] = None,  # [B, T+H]真实环境时间步编码
        attention_masks: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ):
        if self.returns_condition: # CFG
            # 根据预测目标不同,epsilon 也可能表示 数据x0 本身
            epsilon_cond = self.model(
                x, t,
                returns=returns,
                env_timestep=env_ts,
                attention_masks=attention_masks,
                use_dropout=False,
            )  # 不通过use和force,表示不使用dropout概率机制,而是强制使用条件编码信息
            epsilon_uncond = self.model(
                x, t,
                returns=returns,
                env_timestep=env_ts,
                attention_masks=attention_masks,
                force_dropout=True,
            )  # 通过use和force,表示强制不使用条件编码信息(force=True会覆盖use)
            epsilon = epsilon_uncond + self.condition_guidance_w * (
                epsilon_cond - epsilon_uncond
            )

        else:  # 采用DM去噪学习数据分布~BC思想
            epsilon = self.model(
                x, t, env_timestep=env_ts, attention_masks=attention_masks
            )  # 采用默认模式: use_dropout=True,force_dropout=False,表示部分mask

        return epsilon

    @torch.no_grad()
    def conditional_sample(
        self,
        cond: Dict[str, torch.Tensor],
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        horizon: int = None,
        attention_masks: Optional[torch.Tensor] = None,
        verbose: bool = True,
        return_diffusion: bool = False,
    ):
        """
        conditions : [ (time, state), ... ]
        """

        batch_size = cond["x"].shape[0]
        horizon = horizon or self.horizon + self.history_horizon
        sample_dim = self.observation_dim if self.use_inv_dyn else self.transition_dim
        shape = (batch_size, horizon, self.n_agents, sample_dim)

        device = list(cond.values())[0].device
        if self.use_ddim_sample:
            scheduler = self.ddim_noise_scheduler
        elif self.use_consistency_models_sample:
            scheduler = self.consistency_models_scheduler
        else:
            scheduler = self.noise_scheduler

        x = 0.5 * torch.randn(shape, device=device)  # 0.5 用于低温采样

        if return_diffusion:
            diffusion = [x]

        # 设置采样步
        # scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps

        progress = utils.Progress(len(timesteps)) if verbose else utils.Silent()
        for t in timesteps:
            # 1. 应用条件约束
            x = apply_conditioning(x, cond)
            x = self.data_encoder(x)

            # 2. 预测模型输出
            ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
            model_output = self.get_model_output(
                x, ts, returns, env_ts, attention_masks
            )

            # 3. 计算上一扩散状态：x_t -> x_t-1
            x = scheduler.step(model_output, t, x).prev_sample

            progress.update({"t": t})
            if return_diffusion:
                diffusion.append(x)

        # 最后再次确保条件约束被强制满足
        x = apply_conditioning(x, cond)
        x = self.data_encoder(x)

        progress.close()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # ------------------------------------------ 训练 ------------------------------------------#

    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        t: torch.Tensor,
        loss_masks: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ):
        noise = torch.randn_like(x_start)

        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t)
        x_noisy = apply_conditioning(x_noisy, cond)
        x_noisy = self.data_encoder(x_noisy)

        epsilon = self.model(
            x_noisy, t,
            returns=returns,
            env_timestep=env_ts,
            attention_masks=attention_masks,
        )

        if not self.predict_epsilon:
            epsilon = apply_conditioning(epsilon, cond)
            epsilon = self.data_encoder(epsilon)

        assert noise.shape == epsilon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(epsilon, noise)
        else:
            loss, info = self.loss_fn(epsilon, x_start)

        if "agent_idx" in cond.keys() and self.opponent_loss_weight is not None:
            opponent_loss_weight = torch.ones_like(loss) * self.opponent_loss_weight
            indices = (
                cond["agent_idx"]
                .to(torch.long)[..., None]
                .repeat(
                    1, opponent_loss_weight.shape[1], 1, opponent_loss_weight.shape[-1]
                )
            )
            opponent_loss_weight.scatter_(dim=2, index=indices, value=1)
            loss = loss * opponent_loss_weight

        # TODO(zbzhu): 检查这里两个 `.mean()` 是否合理
        loss = (
            (loss * loss_masks).mean(dim=[1, 2]) / loss_masks.mean(dim=[1, 2])
        ).mean()

        if self.returns_loss_guided:
            returns_loss = self.r_losses(x_noisy, t, epsilon, cond)
            info["returns_loss"] = returns_loss
            loss = loss + returns_loss * self.loss_guidence_w

        return loss, info

    def r_losses(self, x_t, t, noise, cond):
        b = x_t.shape[0]
        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x_t, t, noise)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, _, model_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x_t, t=t
        )

        noise = 0.5 * torch.randn_like(x_t)
        # t == 0 时不添加噪声。
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))

        x_t_minus_1 = (
            model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        )
        x_t_minus_1 = apply_conditioning(x_t_minus_1, cond)
        x_t_minus_1 = self.data_encoder(x_t_minus_1)

        # value_diffusion_model 中的 t 按 t - 1 训练。
        value_pred = self.value_diffusion_model(x_t_minus_1, t)

        # value_pred = torch.clamp(value_pred, 0.0, 400.0)
        return -1.0 * value_pred.mean()  # 最大化 value。

    def compute_inv_loss(
        self,
        x: torch.Tensor,
        loss_masks: torch.Tensor,
        legal_actions: Optional[torch.Tensor] = None,
    ):
        info = {}
        # 计算逆动力学损失
        x_t = x[:, :-1, :, self.action_dim :]
        a_t = x[:, :-1, :, : self.action_dim]
        x_t_1 = x[:, 1:, :, self.action_dim :]
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, x_comb_t.shape[2], 2 * self.observation_dim)
        a_t = a_t.reshape(-1, a_t.shape[2], self.action_dim)
        masks_t = loss_masks[:, 1:].reshape(-1, loss_masks.shape[2])
        if legal_actions is not None:
            legal_actions_t = legal_actions[:, :-1].reshape(
                -1, *legal_actions.shape[2:]
            )

        if self.joint_inv or self.share_inv:
            if self.joint_inv:
                pred_a_t = self.inv_model(
                    x_comb_t.reshape(x_comb_t.shape[0], -1)  # (b a) f
                ).reshape(x_comb_t.shape[0], x_comb_t.shape[1], -1)
            else:
                pred_a_t = self.inv_model(x_comb_t)

            if legal_actions is not None:
                pred_a_t[legal_actions_t == 0] = -1e10
            if self.discrete_action:
                inv_loss = (
                    F.cross_entropy(
                        pred_a_t.reshape(-1, pred_a_t.shape[-1]),
                        a_t.reshape(-1).long(),
                        reduction="none",
                    )
                    * masks_t.reshape(-1)
                ).mean() / masks_t.mean()
                inv_acc = (
                    (pred_a_t.argmax(dim=-1, keepdim=True) == a_t)
                    .to(dtype=float)
                    .squeeze(-1)
                    * masks_t
                ).mean() / masks_t.mean()
                info["inv_acc"] = inv_acc
            else:
                inv_loss = (
                    F.mse_loss(pred_a_t, a_t, reduction="none") * masks_t.unsqueeze(-1)
                ).mean() / masks_t.mean()

        else:
            inv_loss = 0.0
            for i in range(self.n_agents):
                pred_a_t = self.inv_model[i](x_comb_t[:, i])
                if self.discrete_action:
                    inv_loss += (
                        F.cross_entropy(
                            pred_a_t, a_t[:, i].reshape(-1).long(), reduction="none"
                        )
                        * masks_t[:, i]
                    ).mean() / masks_t[:, i].mean()
                else:
                    inv_loss += (
                        F.mse_loss(pred_a_t, a_t[:, i]) * masks_t[:, i].unsqueeze(-1)
                    ).mean() / masks_t[:, i].mean()

        return inv_loss, info

    def loss(
        self,
        x: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        legal_actions: Optional[torch.Tensor] = None,
    ):
        if self.train_only_inv:
            assert self.use_inv_dyn, "If train_only_inv, must use inv_dyn"
            info = {}
        else:
            batch_size = len(x)
            t = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=x.device,
            ).long()

            if self.use_inv_dyn:
                diffuse_loss, info = self.p_losses(
                    x[..., self.action_dim :],
                    cond,
                    t,
                    loss_masks,
                    attention_masks,
                    returns,
                    env_ts,
                    states,
                )
            else:
                diffuse_loss, info = self.p_losses(
                    x,
                    cond,
                    t,
                    loss_masks,
                    attention_masks,
                    returns,
                    env_ts,
                    states,
                )

        if self.use_inv_dyn:
            inv_loss, inv_info = self.compute_inv_loss(x, loss_masks, legal_actions)
            info = {**info, **inv_info}
            info["inv_loss"] = inv_loss

            if self.train_only_inv:
                return inv_loss, info

            loss = (1 / 2) * (diffuse_loss + inv_loss)
        else:
            loss = diffuse_loss

        return loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)


class ValueDiffusion(GaussianDiffusion):
    # 值扩散模型
    # 用于学习智能体的奖励函数（例如，奖励函数可以是智能体的奖励）
    # 值扩散模型通过学习奖励函数的分布，来生成符合奖励函数的样本
    def __init__(self, *args, clean_only=False, **kwargs):
        assert "value" in kwargs["loss_type"]
        super().__init__(*args, **kwargs)
        if clean_only:
            print("[ models/diffusion ] Info: Only train on clean samples!")
        self.clean_only = clean_only
        self.sqrt_alphas_cumprod = torch.cat(
            [
                torch.ones(1, device=self.betas.device),
                torch.sqrt(self.alphas_cumprod[:-1]),
            ]
        )
        self.sqrt_one_minus_alphas_cumprod = torch.cat(
            [
                torch.zeros(1, device=self.betas.device),
                torch.sqrt(1 - self.alphas_cumprod[:-1]),
            ]
        )

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        value_loss, info = self.p_losses(x, cond, returns, t - 1)
        value_loss = value_loss.mean()
        return value_loss, info

    def p_losses(self, x_start, cond, target, t):
        if self.clean_only:
            pred = self.model(x_start, torch.zeros_like(t))

        else:
            t = t + 1
            noise = torch.randn_like(x_start)

            # 因为 self.sqrt_alphas_cumprod 等变量在 __init__() 中被调整，
            # 这里的 x_noisy 实际对应 x_t_minus_1。
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_noisy = apply_conditioning(x_noisy, cond)
            x_noisy = self.data_encoder(x_noisy)
            pred = self.model(x_noisy, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, t):
        return self.model(x, t)
