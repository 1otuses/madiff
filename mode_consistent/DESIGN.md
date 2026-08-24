# Mode-consistent 模块详细设计

本文档定义 2026-08-23 两阶段重构后的唯一实现契约。旧的单体 pipeline、`K×D` 中央 code、输入层 FiLM 和专用 run scripts 已移除。

## 1. 设计目标与非目标

目标：

- 从无标签联合离线窗口学习一个匿名离散团队 code；
- 一个 code 同时包含每个固定 agent slot 的角色向量；
- 将训练期 privileged posterior 蒸馏为严格局部 prior；
- 在不使用跨 agent attention 的 `TemporalUnet` 中原生注入 role context；
- 只扩散每个 agent 的 observation trajectory，再用 local inverse dynamics 生成动作；
- 让 return 与 mode 通过有序链式 CFG 训练，并可分别调节引导强度；
- 复用 MADiff 的 `run_experiment.py`、`train.py`、`evaluate.py`、Config、Trainer、EMA 和 checkpoint。

第一版非目标：动态 subgroup、通信学习、CPA shared key、离散动作环境、在线 MAPPO 和自动研究决策。

## 2. 统一张量契约

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `observations` | `[B,H,N,O]` | 规范化局部观测 |
| `actions` | `[B,H,N,A]` | 规范化连续动作 |
| `mask` | `[B,H]` | 有效时间步 |
| `codes_posterior` | `[B]` | 一个联合窗口一个匿名团队 ID |
| `codes_local` | `[B,N]` | 每个 agent 独立推断的 ID，`-1` 为 unknown |
| `role_codebook` | `[K,N,D]` | 团队 mode 到 agent role 的映射 |
| `x` | `[B,H,N,A+O]` | 训练容器；动作供 posterior/inverse loss，观测供 diffusion |
| `cond.x` | `[B,H,N,O]` | observation-only diffusion condition |
| `x_diffusion` | `[B,H,N,O]` | U-Net 实际去噪和采样的局部观测轨迹 |
| `returns` | `[B,1,N]` | 按 `returns_scale` 缩放的团队 RTG，复制到 agent 维 |

代码只允许连续动作。引入离散动作前必须单独定义 decoder likelihood、trajectory encoding 和 inverse-dynamics 分类目标。

## 3. TeamModeVQVAE

### 3.1 Privileged posterior

每个 agent 的 `[o_i^t,a_i^t]` 序列通过同一个 GRU，得到 `f_i`。拼接固定 agent identity 后，team encoder 读取所有 `f_i` 并输出：

```text
z_e ∈ R^(N×D)
```

量化时展平最后两维，距离对全队角色共同求和：

```text
k = argmin_j || flatten(z_e) - flatten(E[j]) ||²
E ∈ R^(K×N×D)
```

因此只产生一个团队 ID，避免先为每个 agent 独立量化再产生新的 mode mismatch。

### 3.2 Observation-conditioned decoder

Decoder 对每个时间步和 agent 共享参数：

```text
a_hat_i^t = Dec(o_i^t, E[k,i], agent_id_i)
```

Observation 已由 decoder 显式接收，code 只需要解释状态条件下剩余的行为选择。第一版不重建 observation，避免 code 优先编码场景或位置。

### 3.3 VQ 损失

```text
L_teacher = L_action
          + lambda_codebook ||sg(z_e) - E[k]||²
          + lambda_commit   ||z_e - sg(E[k])||²
```

默认 `lambda_codebook=1.0`、`lambda_commit=0.25`。两个权重显式命名，不再使用方向含糊的单一 `beta`。Codebook 初始化为 `Uniform(-1/K,1/K)`。

## 4. LocalModePrior

共享 GRU 对每个 agent 独立运行：

```text
p_i(k | o_i^0:t, a_i^0:t-1, agent_id_i)
```

实现先把 action 右移一拍，保证当前动作不会泄漏。Stage1 使用同一次前向得到的 posterior 匿名 code，并停止其梯度：

```text
L_prior = mean_i CE(p_i, stop_gradient(k_posterior))
```

不默认加入跨 agent agreement loss。若局部历史不可辨识，agreement loss 可能仅制造高置信度 collapse。执行时先按置信度阈值拒绝，低置信度输出 `-1`。

训练阶段随机选择 `1..H` 的 prefix，使同一 prior 支持随执行时间更新。Evaluator 通过 `local_prefixes` 输出 prefix–coverage、prefix–accuracy 和 prefix–agreement 曲线，并用 `local_prefix_eval` 指定主报告窗口。

## 5. ModeTemporalUnet

### 5.1 基础 U-Net seam

原 `TemporalUnet` 增加：

```text
context_dim=0
forward(..., context=None,
        use_context_dropout=True,
        force_context_dropout=False)
```

`context_dim=0` 时不创建新参数，原 checkpoint key 不变。启用时，context 经 MLP 投影后与 timestep、return、environment timestep embedding 拼接，送入所有 residual temporal blocks。

### 5.2 Agent-independent forward

`ModeTemporalUnet` 接收 MADiff 的四维输入，但内部执行：

```text
[B,H,N,F] -> [B*N,H,F] -> shared TemporalUnet -> [B,H,N,F]
```

展平顺序使用 `repeat_interleave` 对齐 timestep，return 从 `[B,1,N]` 转为 `[B*N,1]`。没有任何 agent attention 或 joint convolution。

对 agent `i`，上下文为：

```text
context_i = concat(E[k_i,i], agent_embedding(i))
```

若所有 agent 使用 posterior/shared code，则 `k_i=k`；若严格 local prior 产生分歧，每个 agent 只读取自己所选 code 的第 `i` 个 role，不读取其他 agent role。

Horizon 自动补零到 U-Net 下采样倍率的整数倍，输出再裁回原长度。

## 6. ModeGaussianDiffusion 与 inverse dynamics

### 6.1 冻结 mode checkpoint

Stage2 checkpoint 内包含一份冻结的 `TeamModeVQVAE`，并将其 role codebook 复制到 `ModeTemporalUnet`。训练构造时读取 stage1 checkpoint；此后模型权重自包含，evaluator 按保存的模型形状构造后加载完整 state dict。输入预处理仍依赖训练时的 `dataset_config.pkl` 和原数据文件，以重建相同的 CDF/return 统计；checkpoint 本身不重复保存可能很大的经验 CDF 表。

`mode_source_train`：

- `posterior`：默认的 privileged upper bound；
- `local`：只有 local identifiability 通过后才使用；当前实现只允许 `local_prefix_train=1`，避免用待预测动作反推同一窗口的 code；
- `none`：无 mode 对照。

CPA `shared_key` 和真实标签 `oracle` 不属于训练主线，应由未来 evaluator extension 明确实现并标注权限。

### 6.2 Observation diffusion 与 local inverse dynamics

`ModeTemporalUnet.transition_dim=O`。`GaussianDiffusion` 从训练容器中取出 observation 后加噪，采样输出：

```text
o_hat ∈ R^(B×H×N×O)
```

动作不由 U-Net 直接生成。共享 inverse-dynamics MLP 对 agent slice 独立运行：

```text
a_hat_i^t = g_omega(o_i^t, o_i^(t+1))
```

`share_inv=true` 只表示 homogeneous agents 共享 MLP 参数，不会拼接 agent 维。`joint_inv=true` 在 mode-consistent 入口中被拒绝。总 stage2 loss 沿用 MADiff：

```text
L_stage2 = 0.5 * (L_diffusion_observation + L_inverse_action)
```

### 6.3 `RTG → mode` 链式 CFG

四个条件状态的定义为：

```text
epsilon_un  = epsilon_theta(x_t, t, empty, empty)  # all unconditional
epsilon_r   = epsilon_theta(x_t, t, RTG,   empty)  # return only
epsilon_m   = epsilon_theta(x_t, t, empty, z)      # mode only
epsilon_all = epsilon_theta(x_t, t, RTG,   z)      # all conditions
```

训练每个 minibatch 只采样一个条件状态。前三种 ablation 状态各以 `cfg_dropout_prob` 出现，完整状态以 `1 - 3 * cfg_dropout_prob` 出现，所以参数必须满足 `0 <= cfg_dropout_prob < 1/3`。默认值 `0.05` 对应 5% 无条件、5% RTG-only、5% mode-only 和 85% 完整条件。模型调用显式设置所有 `use/force` 标志，不依赖 U-Net 内部随机 dropout。

链式 CFG 使用三次前向：

```text
epsilon_hat = epsilon_un
            + condition_guidance_w * (epsilon_r   - epsilon_un)
            + mode_guidance_w      * (epsilon_all - epsilon_r)
```

链式分解对应 `empty → RTG → RTG+z`。其第二项表示 RTG 的边际引导，第三项表示给定 RTG 后 mode 的条件引导；因此允许 RTG 与 mode 耦合，不要求 `RTG ⟂ z | x_t`。链顺序不是可交换的超参数，第一版固定 RTG 在前。双条件采样不计算 `epsilon_m`，每个扩散步成本为三次 U-Net 前向；只有单一可用条件时退化为标准两次前向 CFG。

Decision Diffuser 附录 D 的并行组合需要给定 `x_t` 后各条件相互独立，形式为：

```text
epsilon_hat_ind = epsilon_un
                + w * [(epsilon_r - epsilon_un)
                     + (epsilon_m - epsilon_un)]
```

RTG 与 mode 都由同一行为轨迹产生，当前没有依据支持该独立性假设。因此删除旧的 `factorized_guidance` 与 `interaction_guidance_w` 执行分支；旧 checkpoint 配置中的这些字段只为构造兼容而忽略，加载后的推理语义统一为链式 CFG。

## 7. ModeSequenceDataset

数据源必须是 `EpisodeStore`。训练 batch 永不包含审计标签。

正式配置使用 `eval_fraction=0.0`，训练索引覆盖数据文件中的全部 episode。`eval_fraction>0` 仅作为显式离线诊断选项，其划分规则为 scenario group holdout > collector-stratified > episode-random。Normalizer 由实际训练索引中的全部有效 observation/action 拟合。

Stage2 的 return 条件严格沿用 MADiff：窗口从 episode 时刻 `t` 开始时，先计算到 episode 末尾 `T` 的团队折扣回报，再除以固定尺度：

```text
G_t = sum_(k=t)^(T-1) discount^(k-t) * mean_i(reward_k,i)
returns_t = G_t / returns_scale
```

规划窗口长度 `H` 不截断 RTG。正式 Simple Spread 配置使用 `discount=0.99`、`returns_scale=700`；不再对 episode return 做 z-score。

两个 stage 使用相同窗口和 normalization：

- `mode_stage1` 返回 `observations/actions/mask`，同时计算 VQ 与 prior loss；
- `mode_stage2` 返回 `x/cond/loss_masks/attention_masks/mode_mask/returns`；
- `x` 保留 action 是为了 posterior 和 inverse loss，U-Net 只接收 observation；
- diffusion condition 只公开每个 agent 自己的 `t=0` observation，动作不可见。

`normalizer` 通过名称选择 `diffuser/datasets/normalization.py` 中的 `Normalizer` 子类。正式配置使用 `CDFNormalizer`；`GaussianNormalizer`、`LimitsNormalizer`、`SafeLimitsNormalizer` 和 `DebugNormalizer` 可用于消融。在线 evaluator 使用保存的 `dataset_config.pkl` 以 `split=train` 重建完全相同的 full-data normalizer，不创建离线 eval split。

## 8. 标准训练与 checkpoint

`run_scripts/train.py` 读取 `training_stage`：

| stage | model config | objective/diffusion config | 前置 checkpoint |
| --- | --- | --- | --- |
| `mode_stage1` | `TeamModeVQVAE` | `ModeVQObjective(train_teacher, train_prior)` | 无 |
| `mode_stage2` | `ModeTemporalUnet` | `ModeGaussianDiffusion(use_inv_dyn)` | stage1 |

两个阶段都复用 `utils.Config`、`utils.Trainer`、EMA、TensorBoard 和 `state_<step>.pt`。Mode 表征 checkpoint 的 state key 统一为 `mode_model.*`。

## 9. 冻结评估与阶段门

`ModeVQEvaluator`：

- reconstruction MSE、usage、active codes、perplexity；
- true assignment NMI/accuracy（若有标签，仅审计）；
- collector NMI；
- local coverage、agreement、MMR、posterior-code accuracy、confidence。

`ModeVQEvaluator` 和 `ModeConditionedEvaluator` 继续提供 code 与 first-action MSE 离线诊断，但当前正式配置不再为它们预留数据。

`ModeOnlineEvaluator` 以固定 scenario seeds 在真实 `simple_spread` 环境中进行 receding-horizon rollout。时刻 `t` 的 local prior 输入严格为 `o_i^0:t,a_i^0:t-1`；各 agent 独立输出 `k_i`，没有 privileged posterior、联合投票或共享随机 key。Diffusion 只以当前 observation 重新锚定，inverse dynamics 产生第一步动作。Evaluator 累积环境 reward，报告 mean-per-agent team return、逐 agent/episode return 与 local mode agreement，并把真实 RGB 环境帧编码为 MP4。`test_ret=0.9` 是缩放后的初始 RTG 条件，不表示最优回报等于 1；默认 `use_return_to_go=false` 时整条 rollout 固定为 0.9，启用后才按 MADiff 的 `(G_t-r_t)/discount` 递推。该条件必须与实测 return 分开解释。

进入下一阶段所需证据：

1. VQ code 使用不塌缩，且正确 code 的重建优于 shuffle/no-code；
2. paired same-state 数据上的 code 与 assignment 关系高于标签置乱基线；
3. posterior-conditioned diffusion 优于 shuffle/no-mode；
4. local prior 在有信息的 prefix 上达到可接受 coverage/agreement；
5. 环境 rollout 已按用户要求开放；只有实际运行、检查视频并完成多 seed 后才能形成正式 return 结论。
