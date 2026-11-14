# mad_mpe_aa 实验结果汇总

核对日期: 2026-11-12。范围: `logs/mad_mpe_aa`。共 3 组环境/数据质量、10 次训练，均已保存 1,000,000 步 checkpoint 及对应最终评估。其中 Tag Expert 于 2026-11-12 使用 DDIM 15 步、100 回合重新评估；其余结果读取已有记录。本次未训练。

## 最终 1M 步结果

Legacy 是此目录中 `attn_type=legacy` 的 MADiff 注意力基线。Agent-Aware 对应 `attn_type=agent_aware`。所有结果统一取最终 1M checkpoint 的 EMA 模型评估，不按历史最优 checkpoint 挑选。

| 环境 | 数据质量 | 训练 seeds | MADiff (Legacy) | Agent-Aware | 回报变化 | 相对变化 |
|---|---|---|---:|---:|---:|---:|
| Spread | Expert | 100,200,300 | 455.94 ± 22.12 | 502.97 ± 7.93 | +47.03 | +10.32% |
| Tag | Expert | 100 | 245.84 | 258.37 | +12.53 | +5.10% |
| Tag | Medium-Replay | 100 | 97.04 | 140.75 | +43.71 | +45.05% |

回报取结果 JSON 的 `mean_ep_reward`，与 `average_ep_reward` 在 agent 维度取均值一致，表示每个 agent 的回合累计回报对 agent 和评估回合取平均，数值越大越好。结果为原始回报，未进行论文分数归一化。

`±` 表示训练 seeds 间的样本标准差 (ddof=1)，只有 Spread Expert 有 3 个训练 seeds。Tag 的两组各只有 1 个训练 seed，因此不报告 seed 标准差。JSON 中的 `std_ep_reward` 是评估回合间波动，未混入此表。相对变化 = (Agent-Aware 均值 - Legacy 均值) / Legacy 均值 × 100%。

## 评估条件

| 环境/数据质量 | 每个训练 seed 的评估回合数 | 采样方式 | returns_scale |
|---|---:|---|---:|
| Spread / Expert | 100 | DDIM 15 | 700 |
| Tag / Expert | 100 | DDIM 15 | 700 |
| Tag / Medium-Replay | 100 | DDIM 15 | 600 |

共同条件: H=24、CTDE=True、guidance=1.2、eval_model_seed=0、num_envs=5、训练扩散步数=200。Agent-Aware 使用 all / film / 时间编码开启 / FFN 关闭。各行均使用 DDIM 15 步、100 回合；同一行中的两种方法评估条件一致。不同环境和数据质量的绝对回报不合并平均。

## 逐 seed 结果与来源

| 数据集 | seed | Legacy | Agent-Aware | 差值 | 评估结果来源 |
|---|---:|---:|---:|---:|---|
| simple_spread-expert | 100 | 481.26 | 510.56 | +29.30 | [Legacy](../../logs/mad_mpe_aa/simple_spread-expert/h_24-models.SharedConvAttentionDeconv-legacy-all-film-pe_True-ffn_False-r_700-guidew_1.2-ctde_True/100/results/step_1000000-ep_100-ddim.json) / [Agent-Aware](../../logs/mad_mpe_aa/simple_spread-expert/h_24-models.SharedConvAttentionDeconv-agent_aware-all-film-pe_True-ffn_False-r_700-guidew_1.2-ctde_True/100/results/step_1000000-ep_100-ddim.json) |
| simple_spread-expert | 200 | 446.20 | 503.62 | +57.42 | [Legacy](../../logs/mad_mpe_aa/simple_spread-expert/h_24-models.SharedConvAttentionDeconv-legacy-all-film-pe_True-ffn_False-r_700-guidew_1.2-ctde_True/200/results/step_1000000-ep_100-ddim.json) / [Agent-Aware](../../logs/mad_mpe_aa/simple_spread-expert/h_24-models.SharedConvAttentionDeconv-agent_aware-all-film-pe_True-ffn_False-r_700-guidew_1.2-ctde_True/200/results/step_1000000-ep_100-ddim.json) |
| simple_spread-expert | 300 | 440.35 | 494.74 | +54.38 | [Legacy](../../logs/mad_mpe_aa/simple_spread-expert/h_24-models.SharedConvAttentionDeconv-legacy-all-film-pe_True-ffn_False-r_700-guidew_1.2-ctde_True/300/results/step_1000000-ep_100-ddim.json) / [Agent-Aware](../../logs/mad_mpe_aa/simple_spread-expert/h_24-models.SharedConvAttentionDeconv-agent_aware-all-film-pe_True-ffn_False-r_700-guidew_1.2-ctde_True/300/results/step_1000000-ep_100-ddim.json) |
| simple_tag-expert | 100 | 245.84 | 258.37 | +12.53 | [Legacy](../../logs/mad_mpe_aa/simple_tag-expert/h_24-models.SharedConvAttentionDeconv-legacy-all-film-pe_True-ffn_False-r_700-guidew_1.2-ctde_True/100/results/step_1000000-ep_100-ddim.json) / [Agent-Aware](../../logs/mad_mpe_aa/simple_tag-expert/h_24-models.SharedConvAttentionDeconv-agent_aware-all-film-pe_True-ffn_False-r_700-guidew_1.2-ctde_True/100/results/step_1000000-ep_100-ddim.json) |
| simple_tag-medium-replay | 100 | 97.04 | 140.75 | +43.71 | [Legacy](../../logs/mad_mpe_aa/simple_tag-medium-replay/h_24-models.SharedConvAttentionDeconv-legacy-all-film-pe_True-ffn_False-r_600-guidew_1.2-ctde_True/100/results/step_1000000-ep_100-ddim.json) / [Agent-Aware](../../logs/mad_mpe_aa/simple_tag-medium-replay/h_24-models.SharedConvAttentionDeconv-agent_aware-all-film-pe_True-ffn_False-r_600-guidew_1.2-ctde_True/100/results/step_1000000-ep_100-ddim.json) |

## 可用于汇报的结论

Agent-Aware 在 Spread Expert 的 3 个训练 seeds 上均获得提升，最终平均回报从 455.94 提高至 502.97 (+10.32%)。在 Tag Medium-Replay 的 seed=100 上，回报从 97.04 提高至 140.75 (+45.05%)。在 Tag Expert 的 seed=100 上，统一 DDIM 15 步、100 回合后，回报从 245.84 提高至 258.37 (+5.10%)。

在当前统一 DDIM 协议下，三组数据上的 Agent-Aware 最终平均回报均高于 Legacy。Spread Expert 有 3 个训练 seeds 的一致提升；Tag 的两组各只有一个训练 seed，仍需补充训练 seeds，才能判断收益是否稳定，当前不主张统计显著或所有环境普遍有效。

这些是 mad_mpe_aa 历史训练记录。后续训练流程修复对应的 mad_mpe_aa_trainfix 和 mad_mpe_aware 未纳入本表；本表不能替代修复后重新训练的对照实验。保存的配置可核对方法和评估参数，但无法仅凭日志证明训练源码逐字一致。

## Tag Expert 重评估与历史协议

本次评估配置: [eval_expert_ddim.yaml](../../exp_specs/mpe/tag/eval_expert_ddim.yaml)。两种方法均取训练 seed=100 的最终 1M EMA 模型，H=24、CTDE=True、guidance=1.2、returns_scale=700、test_ret=0.9、eval_model_seed=0、num_envs=5。eval seed=0 表示同一全局种子下的 100 回合序列，并非 100 个预指定的独立 seeds。

| 方法 | 历史 DDPM 200 / 10 回合 | 本次 DDIM 15 / 100 回合 | 本次评估回合标准差 |
|---|---:|---:|---:|
| MADiff (Legacy) | 276.19 | 245.84 | 87.87 |
| Agent-Aware | 243.18 | 258.37 | 82.27 |

本次 Agent-Aware 相对 Legacy 的变化为 +12.53 (+5.10%)。原 DDPM 10 回合结果文件保留。采样器和回合数同时改变，不能将历史与本次结果的差异全部归因于 DDIM。表中的回合标准差不表示训练 seed 间标准差。

评估模型分别运行于 GPU 3 (Legacy) 和 GPU 2 (Agent-Aware)。为适配现有剩余显存，评估入口将 checkpoint 先读取到 CPU，再向既有模型复制权重并释放 checkpoint；这项调整不改变权重数值、EMA 选择、采样参数或随机数设置。cuDNN benchmark 保持项目原设置，因此不承诺跨 GPU 逐位确定性。
