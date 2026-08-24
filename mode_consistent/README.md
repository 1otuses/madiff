# CI-CoDiff：离散协调 mode 条件扩散

> 更新日期：2026-08-24。本文档保留稳定的研究问题、整体算法和复现入口；精确张量契约与模块设计见 [`DESIGN.md`](DESIGN.md)，历史实验与待讨论结论见 [`RESEARCH_PROGRESS.md`](RESEARCH_PROGRESS.md)。

当前代码实现了新的两阶段训练闭环，但尚未证明离散 code 等价于协调 mode、局部历史足以识别 mode，或 mode 条件策略能够提高环境 return。所有研究结论继续保持`pending_user_discussion`。

## 1. 研究问题

离线多智能体数据可能混合多个高回报协调模态，例如角色分配、目标顺序或阵型：

```text
mu(a_k | s_k) = sum_z p(z | s_k) prod_i mu_i(a_i | h_i^k, z)
```

若各 agent 只拟合自己的边缘多模态分布并独立采样，可能分别选择不同的 `z_i`。每个局部动作都来自数据，组合后的联合动作却不属于任何已见协调策略。

当 `K` 个 mode 等概率、`N` 个 agent 独立选择 mode 时：

```text
Pr(z_1 = ... = z_N) = K^(1-N)
MMR = 1 - K^(1-N)
```

本项目在 CTDE 框架下依次研究：

1. **mode discovery**：能否从无标签联合轨迹中学到代表协调策略的离散变量 `z`；
2. **decentralized identification**：执行时每个 agent 能否仅由局部历史恢复同一个匿名 code；
3. **conditional generation**：离散 mode 能否作为 CFG 条件改变 diffusion 的联合角色，同时保持数据支持；
4. **policy result**：以上条件是否最终改善 return、success 或 MMR。

Mode 默认描述长度为 `H` 的协调片段，不强制整条 episode 使用同一个 mode。`H=25` 是当前 OMAR Simple Spread 的首个诊断点，不是固定结论。

严格 Dec-POMDP 下存在不可绕过的信息边界：若相同局部历史对应多个 mode，且不存在公共观测、通信、共享随机性或可观察行为前缀，local prior 不可能恢复采集时的 `z`。此时只能学习确定性约定、输出 `unknown`，或把 CPA agreement key 明确标成额外 correlation device。

## 2. 与现有方法的关系

| 方法 | 借鉴 | 不直接照搬的部分 |
| --- | --- | --- |
| MADiff | Diffusion、Temporal U-Net、return CFG、训练与评估通道 | 移除跨 agent attention；不再宣称具有原 MADiff 的逐步 teammate interaction |
| CPA | policy mismatch、VQ action pool、条件负对照 | shared seed/time agreement key 只作为 privileged 扩展 |
| VO-MASD | `H` 步轨迹编码、VQ、agent-specific skill、动作重建 | 不引入动态 subgroup、全局 grouper 或在线 MAPPO |
| CLS-DP Contextualizer | privileged posterior 到 local prior 的蒸馏 | 使用匿名离散团队 code，并显式评估 coverage/agreement |

VQ 的动作重建目标只说明 code 对行为有预测价值，不自动说明 code 是协调 convention。因此 reconstruction、code-shuffle、assignment NMI、collector NMI 和局部可辨识性必须分开报告。

## 3. 推荐模型流程

```text
无标签联合窗口 (o[1:N,1:H], a[1:N,1:H], mask)
                     │
                     ▼
      TeamModeVQVAE privileged posterior
   shared agent GRU → team encoder → VQ nearest code k
                     │
              E ∈ R^(K×N×D)
            E[k,i] = agent i role
              ┌──────┴────────┐
              ▼               ▼
 observation-conditioned   LocalModePrior
 action decoder            p_i(k | o_i^≤t, a_i^<t)
              │               │
              └──────┬────────┘
                     ▼
      return embedding + E[k,i] + agent identity
                     │
                     ▼
        ModeTemporalUnet（无 agent attention）
                     │
                     ▼
  ModeGaussianDiffusion 生成局部 observation trajectory
                     │
                     ▼
 Local inverse dynamics g(o_i^t,o_i^{t+1}) → a_i^t
                     │
                     ▼
  MPE Simple Spread 真实 rollout → return + MP4
```

训练分为两个 checkpoint：

1. `mode_stage1`：联合训练 privileged posterior、`K×N×D` codebook、动作 decoder 和 local prior；prior 的监督 code 使用 `stop-gradient`，并随机抽取可见历史 prefix。
2. `mode_stage2`：加载并冻结 stage1，复制 role codebook 到 U-Net；Diffusion 只生成 observation，local inverse-dynamics 网络由每个 agent 自身的相邻 observation 生成动作。

只有 posterior 条件相对 shuffle/no-mode 有稳定增益后，才应把 `mode_source_train` 改为 `local` 或进行 local-conditioned 微调。

## 4. 实现模块

- [`TeamModeVQVAE`](models/team_mode_vqvae.py)：privileged posterior、`K×N×D` role codebook、observation-conditioned action decoder 与严格局部 prior。
- [`ModeTemporalUnet`](models/mode_temporal.py)：共享参数、无跨 agent attention；正确展平 `batch×agent`，将 `E[k,i]` 作为原生 residual context。
- [`ModeGaussianDiffusion`](models/mode_diffusion.py)：冻结 mode checkpoint、observation-only diffusion、local inverse dynamics、四状态训练掩码和 `RTG → mode` 三分支链式 CFG。
- [`ModeSequenceDataset`](data/sequence.py)：EpisodeStore 全量训练窗口、可选离线 holdout、原 MADiff normalizer 选择和训练标签防火墙。
- [`ModeVQObjective`](objectives/mode_vq.py)：在 stage1 联合训练 VQ teacher 与 local prior，并用 `stop-gradient` 隔离 prior target。
- [`ModeOnlineEvaluator`](evaluation/mode_online_evaluator.py)：通过原 `evaluate.py` 加载 stage2 checkpoint，在真实 Simple Spread 环境中滚动规划，输出环境 return 和实际 rollout 视频。
- [`ModeVQEvaluator` 与 `ModeConditionedEvaluator`](evaluation/mode_evaluator.py)：保留为显式离线诊断接口，不属于当前正式评估配置。

基础 [`TemporalUnet`](../diffuser/models/temporal.py) 只增加了默认关闭的 `context_dim=0` 接口；原 MADiff 配置不提供 context 时，参数结构和调用方式不变。

## 5. 信息权限

训练 batch 允许：

```text
observations, actions, valid mask
```

VQ posterior 在训练期可读取完整联合窗口。Local prior 只能读取第 `i` 个 agent 的 observation 和滞后一拍的自身 action；当前动作、其他 agent 轨迹、true mode、collector、scenario、quality 和 reward label 均不得进入 prior。

Stage2 的 U-Net 将 agent 维展平后独立去噪，每个 slice 只生成该 agent 的 observation trajectory。Inverse dynamics 只接收 `[o_i^t,o_i^{t+1}]`；`joint_inv=true` 被明确拒绝。

`true_modes`、collector 和 scenario 只由冻结 evaluator 通过 `audit_labels()` 读取。Reward 只用于 diffusion 的团队 RTG 和最终评估，不监督 VQ code。

## 6. CFG 条件

采用 [InstructPix2Pix](https://arxiv.org/html/2211.09800v2) 的链式 CFG 结构，将原 MADiff 的 RTG 作为第一层基础条件，将离散 mode 作为给定 RTG 后的增量条件。定义四个可训练的噪声预测：

```text
epsilon_un  = epsilon_theta(x_t, t, empty, empty)  # all unconditional
epsilon_r   = epsilon_theta(x_t, t, RTG,   empty)  # return only
epsilon_m   = epsilon_theta(x_t, t, empty, z)      # mode only
epsilon_all = epsilon_theta(x_t, t, RTG,   z)      # all conditions
```

训练时不是分别进行两个独立 Bernoulli dropout，而是每个 minibatch 采样一个四状态掩码。`empty/empty`、`RTG/empty`、`empty/z` 各以 `p_cfg` 出现，完整条件以 `1-3p_cfg` 出现；默认 `p_cfg=0.05`。该掩码只是保证所需分支都接受监督，不代表 RTG 与 mode 在概率上独立。

双条件推理固定执行三次 U-Net 前向：

```text
epsilon_hat = epsilon_un
            + w_R (epsilon_r   - epsilon_un)
            + w_z (epsilon_all - epsilon_r)
```

第二个差分是在 RTG 已知时增加 mode 的效果，因此不需要假设 `RTG` 与 `z` 在给定噪声轨迹后相互独立。链的顺序有意义：当前固定为 `empty → RTG → RTG+z`，因为 RTG 是 MADiff 原有的任务质量条件，mode 是新增的协调选择；当 `w_R=w_z=1` 时公式恰好退化为 `epsilon_all`。`epsilon_m` 仍在训练中出现，用于 mode-only 单条件回退和诊断，但双条件链不计算它。

[Decision Diffuser 附录 D](https://arxiv.org/html/2211.15657v4) 的并行组合可写成：

```text
epsilon_hat_ind = epsilon_un
                + w [epsilon_r - epsilon_un
                     + epsilon_m - epsilon_un]
```

其推导假设各条件在给定噪声轨迹 `x_t` 后条件独立。RTG 来自轨迹回报，而 mode 又概括同一轨迹的协调行为，两者预期存在耦合，因此当前实现不提供该并行/因子化开关；它只保留为未来受控消融，不作为默认框架。

`unknown=-1` 使用零 role 向量；若整个 batch 都是 unknown，推理退化为标准 RTG-only CFG。Agent identity 属于共享策略的结构信息，不随 mode 一起丢弃。评估配置分别用 `condition_guidance_w` 和 `mode_guidance_w` 控制 `w_R`、`w_z`。

## 7. 全量训练与在线评估

正式配置设置 `eval_fraction=0.0`，stage1 和 stage2 都使用数据文件中的全部 episode；不再从离线数据中扣除 validation/test episode。Normalizer 由完整训练文件的有效 observation/action 拟合。RTG 不拟合 z-score 统计，而是沿用 MADiff：从窗口起点 `t` 累积到 episode 末尾并除以 `returns_scale`；`H` 只控制模型轨迹窗口，不截断 RTG。正式 Simple Spread 配置为 `discount=0.99`、`returns_scale=700`。`eval_fraction>0` 只保留给显式离线诊断：有 `scenario_ids` 时按 scenario group holdout，否则优先在 collector 内分层，最后才按固定 seed 随机划分。

正式策略评估不读取离线 episode，而是在新建的 `simple_spread` 环境中执行。每个环境步：

1. 每个 agent 用自己的 `o_i^0:t,a_i^0:t-1` 更新 local mode；
2. observation-only Diffusion 以当前局部观测重新规划；
3. local inverse dynamics 从生成的 `o_i^t,o_i^(t+1)` 得到动作；
4. 将动作送入真实环境，累计实际 reward，并渲染实际环境帧。

至少报告：

| 层级 | 指标 |
| --- | --- |
| posterior | action reconstruction、code usage、perplexity、assignment NMI/ARI、collector NMI |
| local prior | posterior-code accuracy、coverage、unknown rate、agreement、MMR、prefix 曲线 |
| diffusion | posterior/local/shuffle/no-mode，固定 observation/return/noise 下的条件差异 |
| environment | return、success、mode fidelity、support distance、多 seed 方差 |

结果中的 team return 定义为各 agent 环境累计 return 的均值。MP4 记录实际环境 rollout，不是将预测 observation 直接渲染成“真实结果”。离线 first-action MSE 仍只是一种诊断，不等价于环境 return。

## 8. 复现入口

从仓库根目录运行软件测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run --no-capture-output -n madiff \
  pytest tests/mode_consistent -q
```

当前配置文件 `expert_unlabeled_h25.npz` 的实际头信息是 20 万条 horizon-25 episode，即 5M 个联合环境步或 15M 个 agent-step；按常用 transition 口径不是 100M。正式配置不设置 `max_n_episodes` 上限，并设置 `eval_fraction=0.0`，因此完整文件全部参与训练。若 100M 指另一份数据，只需替换两个 stage 的 `dataset_path`，并先核对相同的 episode tensor contract。两个阶段统一使用 `CDFNormalizer`，必须按顺序执行：

| 阶段 | 模型规模与优化预算 | 最终 checkpoint |
| --- | --- | --- |
| stage1：VQ + local prior | `hidden=128, mode_dim=32, batch=128, 10,000 joint updates` | `state_10000.pt` |
| stage2：DM + inverse dynamics | `TemporalUnet dim=128, [1,2,4,8], T=200, batch=32×accumulation 2, 1,000,000 updates` | `state_1000000.pt` |

```bash
GPU_ID=0

conda run -n madiff python run_experiment.py \
  -e exp_specs/mode_consistent/stage1_omar_spread_h25_seed0.yaml -g "$GPU_ID"

conda run -n madiff python run_experiment.py \
  -e exp_specs/mode_consistent/stage2_omar_spread_h25_seed0.yaml -g "$GPU_ID"
```

每个正式训练配置都设置了 `continue_training: true`；重复执行同一命令时会从该实验目录内的最新 checkpoint 继续。Stage2 固定加载 stage1 的 `state_10000.pt`，因此不能跳过或交换阶段。完整数据会整体载入内存，启动正式训练前应确认系统内存充足。

`normalizer` 可选择原 [`diffuser/datasets/normalization.py`](../diffuser/datasets/normalization.py) 中的 `CDFNormalizer`、`GaussianNormalizer`、`LimitsNormalizer`、`SafeLimitsNormalizer` 或 `DebugNormalizer`。正式配置默认 CDF；在线 evaluator 从训练配置重新构造同一完整数据统计。完整数据和精确 CDF 会整体占用较多内存，正式运行前应预留足够 RAM。

训练完成后，真实环境评估仍走原 `run_scripts/evaluate.py`：

```bash
conda run -n madiff python run_scripts/evaluate.py \
  -e exp_specs/mode_consistent/eval_stage2_omar_spread_h25_seed0.yaml -g "$GPU_ID"
```

正式 evaluator 加载 `state_1000000.pt`，使用 seeds 1000–1009 执行 10 个 horizon-25 episode，并为前三个 episode 保存 MP4。JSON 写入 stage2 实验目录的 `results/`，视频写入带 sampler/CFG 权重后缀的 `videos/step_1000000-*/`，避免不同引导权重互相覆盖。`test_ret=0.9` 表示缩放后的初始 RTG 条件（对应初始未缩放目标 `0.9×700=630`），不是 z-score，也不表示最优 return 为 1。正式配置保持 MADiff 默认的 `use_return_to_go=false`，因此每一步固定使用 0.9；只有显式启用后，才会减去已获得团队 reward 并按 `discount` 更新剩余 RTG。实测值单独以 `average_team_return`、逐 agent return 和逐 episode return 保存。当前不在 rollout 中使用 privileged posterior，也不从多个 agent 的 mode 投票。

正式配置已经可运行，但本仓库尚未执行这些新训练，因此不产生新的研究结论。`n_train_steps=1,000,000` 是 optimizer update 数；不要与 `n_diffusion_steps=200` 个噪声时间步混淆。

当前只保留两个正式训练配置和一个 stage2 在线评估配置；软件级快速验证由 `tests/mode_consistent` 覆盖。正式训练与 rollout 尚未实际执行，seed 0/十个评估场景仍只用于先观察信号，不构成多 seed 论文结论。

Motivating example 与受控 paired MPE 数据生成入口继续保留：

```bash
conda run -n madiff python -m mode_consistent.prototypes.xor_motivating_example

conda run -n madiff python -m mode_consistent.scripts.generate_mpe_dataset \
  --output /path/to/balanced_6mode_expert_seed0.npz \
  --n-scenarios 1000 --mode-ids 0 1 2 3 4 5 \
  --qualities expert --horizon 25 --seed 0
```
