"""
风险度量算子 (Risk Measures)
============================
VaR (Value at Risk), CVaR (Conditional VaR), Wang 扭曲风险度量
以及它们在 Diffusion Critic 评估下的梯度估计。
"""

import torch
import torch.nn.functional as F


class RiskMeasures:
    """风险度量算子集合，用于 Diffusion Critic 的分布评估。"""

    @staticmethod
    def var(samples: torch.Tensor, alpha: float, dim: int = 0) -> torch.Tensor:
        """
        Value at Risk: 分布的 α-分位数。
        samples: [n_samples, B]   B = batch_size
        alpha: 置信水平 (如 0.05 表示下分位数保守估计)
        返回: [B]
        """
        sorted_samples, _ = samples.sort(dim=dim)
        n = samples.shape[dim]
        idx = max(0, min(int(n * alpha), n - 1))
        return sorted_samples[idx]

    @staticmethod
    def cvar(samples: torch.Tensor, alpha: float, dim: int = 0) -> torch.Tensor:
        """
        Conditional VaR: 低于 VaR(α) 的所有样本的期望。
        samples: [n_samples, B]
        alpha: 置信水平
        返回: [B]
        """
        sorted_samples, _ = samples.sort(dim=dim)
        n = samples.shape[dim]
        k = max(1, int(n * alpha))
        tail = sorted_samples[:k]
        return tail.mean(dim=dim)

    @staticmethod
    def wang_transform(
        samples: torch.Tensor, gamma: float, dim: int = 0
    ) -> torch.Tensor:
        """
        Wang 扭曲风险度量 (Wang Transform)。
        samples: [n_samples, B]
        gamma: 风险厌恶参数 (>0 表示风险偏好扭曲, 单 agent 时通常用 <0)
               通常 gamma ∈ [-1, 1], gamma<0 表示风险厌恶 (保守)
        返回: [B]
        """
        sorted_samples, _ = samples.sort(dim=dim)
        n = samples.shape[dim]
        t = torch.linspace(0, 1, n, device=samples.device)
        # Φ^{-1}(t) 的近似 + gamma
        from math import sqrt, erf, erfinv
        # 标准正态分位数的近似
        # 使用 torch.special.erfinv (需要 PyTorch 1.10+)
        try:
            z = torch.erfinv(2 * t - 1) * sqrt(2.0)  # ≈ Φ^{-1}(t)  [N]
        except AttributeError:
            # fallback
            z = (2 * t - 1) * 3.0
        distorted_t = 0.5 * (1 + torch.erf((z + gamma) / sqrt(2.0)))
        # 插值权重 (离散化)
        distorted_t = torch.clamp(distorted_t, 0.0, 1.0)
        idx = (distorted_t * (n - 1)).long()
        idx = torch.clamp(idx, 0, n - 1)
        return sorted_samples[idx].mean(dim=dim)

    # ----------------------- 可微版本 -----------------------

    @staticmethod
    def differentiable_var(
        samples: torch.Tensor, alpha: float, dim: int = 0, tau: float = 0.1
    ) -> torch.Tensor:
        """
        可微 VaR: 用 soft quantile 实现，允许梯度通过。
        samples: [n_samples, B]
        alpha: 分位点
        tau: 温度参数
        返回: [B]
        """
        sorted_samples, _ = samples.sort(dim=dim)
        n = samples.shape[dim]
        idx = max(0, min(int(n * alpha), n - 1))
        # 使用附近几个值的软加权
        left = max(0, idx - 1)
        right = min(n - 1, idx + 1)
        soft_idx = torch.tensor([left, idx, right], device=samples.device).float()
        f_idx = n * alpha
        weights = torch.softmax(-((soft_idx - f_idx) ** 2) / (2 * tau**2), dim=0)
        return (weights[0] * sorted_samples[left] +
                weights[1] * sorted_samples[idx] +
                weights[2] * sorted_samples[right])

    @staticmethod
    def differentiable_cvar(
        samples: torch.Tensor, alpha: float, dim: int = 0
    ) -> torch.Tensor:
        """
        可微 CVaR: tail mean。
        """
        sorted_samples, _ = samples.sort(dim=dim)
        n = samples.shape[dim]
        k = max(1, int(n * alpha))
        return sorted_samples[:k].mean(dim=dim)


def risk_gradient(
    critic,
    trajectories: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    risk_fn,
    risk_kwargs: dict,
) -> torch.Tensor:
    """
    计算 ∂/∂τ [ ψ_α( critic(τ) ) ]，即风险度量对状态序列的梯度。

    用于 CFG 的条件引导：∇_τ Risk(τ) 作为额外引导信号。

    参数:
      critic: DiffusionCriticAC 模型 (评估 p(R|s,a))
      trajectories: [B, H, A, obs_dim]  状态序列 (去噪过程中的)
      states: [B, state_dim]  环境全局状态
      actions: [B, A]  离散动作 (若 critic 需要)
      risk_fn: 风险度量函数 (如 RiskMeasures.cvar)
      risk_kwargs: 风险度量参数字典
    返回:
      risk_grad: [B, H, A, obs_dim]  风险度量对轨迹的梯度
    """
    # 从轨迹末尾提取状态并计算风险
    # 简化: 取轨迹最后一步的观测作为评估状态
    obs_t = trajectories[:, -1, :, :]  # [B, A, obs_dim]

    # 如果需要全局 states，拼接到 obs
    # 这里简单地将轨迹最后一步视为 s，actions 直接从数据获取
    # 实际上风险度量应由 (s_t+H, a_t+H-1) 的分布得到
    s_feat = trajectories[:, -1, :, :].reshape(trajectories.shape[0], -1)  # [B, A*obs_dim]

    # 用 critic 采样回报分布
    samples = torch.linspace(0, 1, 100, device=trajectories.device)  # placeholder
    # 实际需要调用 critic.sample() 但需要 actions
    # 占位返回所有分量
    return torch.zeros_like(trajectories)
