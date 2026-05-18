"""
my_test - CFG + Inverse Dynamics + Risk-Operator-Guided Diffusion
=================================================================
移除注意力机制，使用简化 MLP/Conv1d 去噪器。
Diffusion Critic 提供风险度量 (VaR, CVaR, Wang 扭曲风险)，
风险梯度 ∇τ ψα(Z_critic(τ)) 作为 CFG 引导条件。
"""
