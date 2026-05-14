# Diffusion Critic 实验套件 — 使用指南
# =========================================

## 文件结构
```
custom/diffusion_critic/
├── toy_env.py                  # Toy Environment (双峰回报)
├── diffusion_critic.py         # Diffusion Critic (基础版，MLP去噪)
├── diffusion_critic_adv.py     # Diffusion Critic (升级版，ResMLP去噪)
├── smac_dataset.py             # SMAC 数据集加载器
├── train.py                    # Toy Env 独立训练脚本
├── train_smac.py               # SMAC 独立训练脚本
├── run_experiment.py           # 统一实验运行器 (TensorBoard + 模型保存 + 评估)
├── results/                    # 历史实验结果
│   ├── diffusion_critic_vs_mlp.png
│   ├── diffusion_process.png
│   └── smac/
└── runs/                       # TensorBoard 日志 (由 run_experiment.py 生成)
    └── tensorboard/
```

## 快速使用

### 1. Toy Environment (验证 Diffusion Critic 双峰学习)
```bash
python run_experiment.py --env toy --n_epochs 300
```

### 2. SMAC 3m-Good (高维状态)
```bash
python run_experiment.py --env smac_3m --n_episodes 3000 --n_epochs 300
```

### 3. 查看训练曲线
```bash
tensorboard --logdir custom/diffusion_critic/runs/tensorboard --port 6006
```

### 4. 查看评估结果
```bash
# Toy Env 评估
cat custom/diffusion_critic/runs/results_toy.json

# SMAC 评估
cat custom/diffusion_critic/runs/results_smac_3m-Good.json
```

## 恢复训练 / 加载模型
```python
from diffusion_critic_adv import DiffusionCritic
model = DiffusionCritic(state_dim=147, n_timesteps=100, hidden_dim=256)
model.load_state_dict(torch.load("runs/models/diff_critic_smac_3m-Good.pt"))
model.eval()
```

## 关键指标
- **Diffusion Critic Loss**: DDPM 噪声预测 MSE (< 0.1 表示收敛)
- **MLP Critic Loss**: MSE 回归误差 (不可约减的为 return 方差)
- **Diffusion 采样 MAE**: 与真实 returns 的平均绝对误差
- **分位数 spread**: q10/q90 区间，指示学习到的分布宽度
