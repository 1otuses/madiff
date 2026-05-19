# my_test Pipeline

## 1. 架构总览

```
                 ┌──────────────────────────────────┐
                 │       RiskGuidedDiffusion         │
                 │  (state diffusion + inv_model)    │
                 └──────────┬───────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
     ┌────────────┐ ┌──────────┐ ┌──────────────┐
     │ Denoiser   │ │ InvModel │ │ DDPM Scheduler│
     │ (Conv1D    │ │(MLP 共享) │ │(cosine β)    │
     │  + ResMLP) │ │          │ │              │
     └────────────┘ └──────────┘ └──────────────┘
```

## 2. 数据流

### 训练阶段

```
离线数据 (MPE)                          离线数据 (SMAC)
  seed_x_data/obs_i.npy                  obs.npy
  seed_x_data/acs_i.npy                  states.npy
  seed_x_data/rews_i.npy                 rewards.npy
        │                                  path_lengths.npy
        ▼                                  │
  load_mpe_data()                    smac_dataset.py
  → ep: [obs, acs, rews]            → (states, returns_rtg)
        │                                  │
        ▼                                  ▼
  build_trajectory_batch()     Train/Val split
  → x: [B, T, A, D]            → (s_train, r_train)
  → actions: [B, T-1, A, d]    → (s_val, r_val)
  → returns: [B]                     │
        │                             ▼
        ▼                     Diffusion Critic
  RiskGuidedDiffusion.loss()      model + scheduler
  → diffuse_loss (MSE ε)          → train with DDPM
  → inv_loss (MSE a_t)            → compare with MLP Critic
        │
        ▼
  backprop + Adam + EMA + TB logging
```

### 评估阶段

```
加载 checkpoint.pt
  → model (或 ema_model)
        │
        ▼
    make_env(render_mode="rgb_array")
        │
        ▼
    循环: s_t → conditional_sample     
             ↓                       
          s_{t+1:H}                  可选择 DDIM 加速
             ↓
          inv_model(s_t, s_{t+1})    
             ↓
          a_t (连续, clip [-1,1])    
             ↓
          env.step(a_t)             
             ↓
          render (存帧→MP4)
```

## 3. 文件结构

| 文件 | 职责 | 依赖 |
|------|------|------|
| `model/diffusion_actor.py` | 扩散模型核心: loss, sample, CFG, DDIM | model/simple_denoiser, model/inverse_dynamics |
| `model/simple_denoiser.py` | Conv1D+ResMLP 去噪网络 | - |
| `model/inverse_dynamics.py` | (s_t, s_{t+1}) → a_t | - |
| `model/risk_measures.py` | VaR/CVaR/Wang 风险度量 | - |
| `model/__init__.py` | 统一导出 | model/* |
| `run_scripts/train.py` | 训练循环 + TensorBoard + EMA | model/* + normalizer |
| `run_scripts/evaluate.py` | 在线评估 + MP4 + 逐 agent 奖励统计 | model/* |
| `run_experiment.py` | 自动串联 train→eval | run_scripts/* |
| `normalizer.py` | obs/action 归一化 (LimitsNormalizer) | - |
| `smac_dataset.py` | SMAC 数据加载 + return-to-go 计算 | - |
| `config/mpe_spread_exp.yaml` | 超参数 (see other env/quality configs) | - |

## 4. 评估指标格式

`runs/my_test/{env}/eval/eval_results.json`:
```
{
  "overall_mean": 0.0,          ← 所有 agent 回报求和后的均值
  "overall_std": 2.5,           ← 所有 agent 回报求和后的标准差
  "average_ep_reward": [0.1, -0.3, 0.2],  ← 逐 agent 均值 (MADiff 风格)
  "std_ep_reward": [1.2, 1.5, 1.1],       ← 逐 agent 标准差
  "seed_stats": { ... },
  "checkpoint": "runs/my_test/.../checkpoint.pt",
  "checkpoint_step": 100000
}
```

## 5. 关键改进对照

| 模块 | 原版 my_test | 当前 my_test |
|------|-------------|-------------|
| 去噪器 | SimpleTemporalDenoiser (无 attention) | 同左 |
| 逆动力学 | 离散 CE → 连续 MSE | 连续 MSE 回归 |
| 条件引导 | 仅 CFG(returns) | CFG(returns) + risk_grad |
| 归一化 | ❌ 无 | ✅ LimitsNormalizer |
| 采样 | DDPM full | DDPM full + **DDIM 加速** |
| 评估指标 | scalar sum | per-agent + sum 双重输出 |
| 保存路径 | runs/{env}/checkpoint_final.pt | runs/{env}/checkpoint/checkpoint.pt |
