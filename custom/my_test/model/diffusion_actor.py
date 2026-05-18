"""
主扩散模型: CFG + 逆动力学 + 风险梯度引导
==========================================
基于 Simplified Temporal Denoiser (无注意力).
集成 Diffusion Critic 输出风险度量作为 CFG 引导.
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use absolute imports for subpackage usage
try:
    from .simple_denoiser import SimpleTemporalDenoiser
    from .inverse_dynamics import InverseDynamicsModel
    from .risk_measures import RiskMeasures
except ImportError:
    from simple_denoiser import SimpleTemporalDenoiser
    from inverse_dynamics import InverseDynamicsModel
    from risk_measures import RiskMeasures


class RiskGuidedDiffusion(nn.Module):
    """
    风险引导扩散模型 (~ MADIFF 简化版)。

        训练:
            diffuse_loss = MSE(ε_pred, ε_true) on state sequences
            inv_loss = MSE( inv_model(s_t, s_{t+1}), a_t )

    推理 (conditional_sample):
      条件引导 = CFG(returns) + CFG(risk_gradient)
      ε = ε_uncond + w_ret*(ε_cond_ret - ε_uncond)
                    + w_risk*(ε_cond_risk - ε_uncond)  [新增风险项]
    """

    def __init__(
        self,
        horizon: int,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        history_horizon: int = 0,
        hidden_dim: int = 256,
        n_timesteps: int = 100,
        n_blocks: int = 4,
        share_inv: bool = True,
        use_returns_condition: bool = True,
        use_risk_guide: bool = True,
        condition_guidance_w: float = 1.2,
        risk_guidance_w: float = 0.5,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.history_horizon = history_horizon
        self.use_returns_condition = use_returns_condition
        self.use_risk_guide = use_risk_guide
        self.condition_guidance_w = condition_guidance_w
        self.risk_guidance_w = risk_guidance_w

        # 去噪器
        self.denoiser = SimpleTemporalDenoiser(
            horizon=horizon + history_horizon,
            n_agents=n_agents,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            use_returns_condition=use_returns_condition,
            use_risk_guide=use_risk_guide,
        )

        # 逆动力学模型
        self.inv_model = InverseDynamicsModel(
            obs_dim=obs_dim,
            n_agents=n_agents,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            share=share_inv,
        )

        # β schedule (cosine)
        self.n_timesteps = n_timesteps
        betas = self._cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    # ----------------------- 训练损失 -----------------------

    def loss(
        self,
        x: torch.Tensor,
        returns: torch.Tensor,
        risk_grad: Optional[torch.Tensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
    ):
        """
        x:   [B, T, A, obs_dim]  真实状态序列
        returns: [B]  回报条件
        risk_grad: [B, T, A, obs_dim]  风险梯度 (可选)
        loss_masks: [B, T, A]  损失掩码 (1=有效)
        """
        B, T, A, D = x.shape
        device = x.device

        # 1. 扩散损失
        t = torch.randint(0, self.n_timesteps, (B,), device=device).long()
        noise = torch.randn_like(x)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)
        x_noisy = sqrt_alpha * x + sqrt_one_minus * noise  # [B, T, A, D]

        # 始终预测噪声 (不使用 CFG 分支，因为训练阶段只学习条件分布)
        epsilon_pred = self.denoiser(x_noisy, t, returns=returns, risk_grad=risk_grad)

        diffuse_loss = F.mse_loss(epsilon_pred, noise)

        # 2. 逆动力学损失
        s_t = x[:, :-1, :, :]     # [B, T-1, A, D]
        s_t_1 = x[:, 1:, :, :]    # [B, T-1, A, D]
        # actions 不在 x 中，需要单独提供
        # 这里返回 diffuse_loss 和 inv_loss 的训练由外部循环处理
        # (inv_loss 需要外部分段调用)

        return diffuse_loss

    def compute_inv_loss(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        loss_masks: Optional[torch.Tensor] = None,
    ):
        """
        x:       [B, T, A, obs_dim]  真实状态序列
        actions: [B, T-1, A, action_dim]  连续动作 (与 s_t->s_{t+1} —— 对应)
        loss_masks: [B, T, A]  掩码 (可选)
        """
        B, T, A, D = x.shape
        s_t = x[:, :-1, :, :]     # [B, T-1, A, D]
        s_t_1 = x[:, 1:, :, :]    # [B, T-1, A, D]
        # actions 已经由 build_trajectory_batch 提供为 (B, T-1, A, action_dim)
        a_t = actions  # [B, T-1, A, action_dim]

        pred_actions = self.inv_model(
            s_t.reshape(-1, A, D), s_t_1.reshape(-1, A, D)
        )  # [B*(T-1), A, action_dim]

        pred_flat = pred_actions.reshape(-1, pred_actions.shape[-1])  # [B*(T-1)*A, action_dim]
        target_flat = a_t.reshape(-1, a_t.shape[-1])  # [B*(T-1)*A, action_dim]

        # MSE loss
        inv_loss = F.mse_loss(pred_flat, target_flat)
        # MAE metric
        inv_mae = (pred_flat - target_flat).abs().mean()
        return inv_loss, inv_mae

    # ----------------------- 采样 -----------------------

    def get_model_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        risk_grad: Optional[torch.Tensor] = None,
    ):
        """
        CFG + 风险引导的噪声预测。

        ε = ε_uncond + w_ret*(ε_cond_ret - ε_uncond)
                    + w_risk*(ε_cond_risk - ε_uncond)
        """
        # 无条件
        epsilon_uncond = self.denoiser(
            x, t, returns=None, risk_grad=None, force_dropout=True
        )

        # 条件 (returns)
        if self.use_returns_condition and returns is not None:
            epsilon_cond_ret = self.denoiser(
                x, t, returns=returns, risk_grad=risk_grad
                if self.use_risk_guide else None
            )
            epsilon = epsilon_uncond + self.condition_guidance_w * (
                epsilon_cond_ret - epsilon_uncond
            )
        else:
            epsilon = epsilon_uncond

        # 风险梯度引导 (额外项)
        if self.use_risk_guide and risk_grad is not None:
            epsilon_cond_risk = self.denoiser(
                x, t, returns=returns, risk_grad=risk_grad
            )
            epsilon = epsilon + self.risk_guidance_w * (
                epsilon_cond_risk - epsilon_cond_ret
                if self.use_returns_condition and returns is not None
                else epsilon_cond_risk - epsilon_uncond
            )

        return epsilon

    @torch.no_grad()
    def conditional_sample(
        self,
        cond_obs: torch.Tensor,
        returns: torch.Tensor,
        risk_grad: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
        ddim_steps: Optional[int] = None,
        ddim_eta: float = 0.0,
        verbose: bool = False,
    ):
        """
        条件扩散采样，支持 DDIM 加速。

        cond_obs: [B, history_horizon, A, obs_dim]  历史观测条件
        returns:  [B]  目标回报
        risk_grad: [B, horizon, A, obs_dim]  风险梯度引导 (可选)
        horizon:  预测时间长度 (默认 self.horizon)
        ddim_steps: DDIM 加速步数 (None=用完整 DDPM)
        ddim_eta: DDIM η 参数 (0=确定性)
        """
        B = cond_obs.shape[0]
        H = horizon or self.horizon
        total_h = self.history_horizon + H
        device = cond_obs.device

        shape = (B, total_h, self.n_agents, self.obs_dim)
        x = 0.5 * torch.randn(shape, device=device)

        # 将条件观测复制到序列开头
        if self.history_horizon > 0:
            x[:, :self.history_horizon] = cond_obs

        # DDIM 采样时间序列
        if ddim_steps is not None and ddim_steps < self.n_timesteps:
            # 等间隔选取 ddim_steps 个时间步
            skip = self.n_timesteps // ddim_steps
            seq = list(range(0, self.n_timesteps, skip))
            if seq[-1] != self.n_timesteps - 1:
                seq.append(self.n_timesteps - 1)
            timesteps = sorted(seq, reverse=True)
            ddim = True
        else:
            timesteps = reversed(range(self.n_timesteps))
            ddim = False

        # 逆向扩散采样
        for t_idx in timesteps:
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)

            # 构建全序列 risk_grad (若提供)
            rg = None
            if risk_grad is not None:
                rg = torch.zeros_like(x)
                rg[:, self.history_horizon:] = risk_grad

            epsilon = self.get_model_output(x, t, returns=returns, risk_grad=rg)

            alpha_bar = self.alphas_cumprod[t_idx]
            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar) * epsilon) / sqrt_alpha_bar
            x0_pred = torch.clamp(x0_pred, -10.0, 10.0)

            if t_idx > 0:
                if ddim:
                    # DDIM 更新 (去噪 + 无随机噪声)
                    alpha_bar_prev = self.alphas_cumprod[timesteps[timesteps.index(t_idx) + 1]] \
                        if t_idx != timesteps[-1] else torch.tensor(1.0, device=device)
                    sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
                    sigma = ddim_eta * torch.sqrt(
                        (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
                    )
                    c1 = torch.sqrt(1.0 - alpha_bar_prev - sigma ** 2)
                    x = sqrt_alpha_bar_prev * x0_pred + c1 * epsilon + sigma * torch.randn_like(x)
                else:
                    # DDPM 更新
                    alpha = self.alphas[t_idx]
                    beta = self.betas[t_idx]
                    alpha_bar_prev = self.alphas_cumprod[t_idx - 1]
                    coef1 = torch.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar)
                    coef2 = torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                    mean = coef1 * x0_pred + coef2 * x
                    beta_tilde = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                    sigma = torch.sqrt(beta_tilde)
                    x = mean + sigma * torch.randn_like(x)
            else:
                x = x0_pred

            # 保持条件不变
            if self.history_horizon > 0:
                x[:, :self.history_horizon] = cond_obs

        # 返回预测的完整序列
        return x  # [B, total_h, A, obs_dim]
