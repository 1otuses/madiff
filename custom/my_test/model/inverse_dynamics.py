"""
逆动力学模型 (Inverse Dynamics Model)
=====================================
从相邻状态 (s_t, s_{t+1}) 预测动作 a_t。
当前用于连续动作回归 (MPE 连续动作)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseDynamicsModel(nn.Module):
    """
    从 (s_t, s_{t+1}) 预测 a_t。
    支持共享参数 (所有 agent 使用同一个模型) 或独立建模。
    """

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_actions: int,
        hidden_dim: int = 256,
        share: bool = True,
    ):
        """
        obs_dim: 每个 agent 的观测维度
        n_agents: agent 数量
        n_actions: 动作维度 (连续动作)
        """
        super().__init__()
        self.n_agents = n_agents
        self.share = share

        input_dim = 2 * obs_dim  # s_t + s_{t+1} 拼接

        if share:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
            )
        else:
            self.nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_actions),
                )
                for _ in range(n_agents)
            ])

    def forward(self, s_t, s_t_1):
        """
        s_t:   [B, A, obs_dim]
        s_t_1: [B, A, obs_dim]
        返回:  [B, A, n_actions]  连续动作预测
        """
        B, A, D = s_t.shape
        x = torch.cat([s_t, s_t_1], dim=-1)  # [B, A, 2*obs_dim]

        if self.share:
            x = x.reshape(B * A, -1)  # [B*A, 2*obs_dim]
            logits = self.net(x)      # [B*A, n_actions]
            return logits.reshape(B, A, -1)  # [B, A, n_actions]
        else:
            logits_list = []
            for i in range(A):
                logits_i = self.nets[i](x[:, i, :])  # [B, n_actions]
                logits_list.append(logits_i.unsqueeze(1))  # [B, 1, n_actions]
            return torch.cat(logits_list, dim=1)  # [B, A, n_actions]

    def compute_loss(self, s_t, s_t_1, a_t):
        """
        s_t:   [B, A, obs_dim]
        s_t_1: [B, A, obs_dim]
        a_t:   [B, A, n_actions]  连续动作
        返回:  scalar loss (MSE)
        """
        pred = self.forward(s_t, s_t_1)  # [B, A, n_actions]
        return F.mse_loss(pred, a_t)
