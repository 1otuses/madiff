# AgentFormer 时空注意力接入 MADiff

## 本次范围

按 `agentformer_to_madiff_plan.md` 和本轮确认实现：`all` 替换、`film` 默认、`mid` 消融。有效性 mask 与历史/未来 segment 已接入；精确 observed/clamped 标记留到后续。实验继续采用 CTDE，训练预算为 1M。

新增模块位于 `diffuser/models/agent_aware_attention.py`。`SharedConvAttentionDeconv` 通过构造配置选择模块，`run_scripts/train.py` 转发并保存配置。扩散过程、数据管道、逆动力学和 evaluator 沿用现有实现。

本轮完成代码和数值验证，未启动训练或环境回报评估。原方案中的 2,000 步 smoke、显存/速度阈值及多 seed 性能判据尚未验证。

## 模块与数据流

输入为 `x: [B,A,C,T]`，条件为 `cond: [B,A,D]`，输出保持 `[B,A,C,T]`。内部使用 `[B,A,T,C]`，按 agent-major 展平为 `[B,A*T,C]`；通道始终是 LayerNorm 和 FFN 的最后一维。

每个 token 加窗口位置正弦编码，位置为 `t / max(T-1,1) * 10`，频率基数为 10000。所有 agent 共用位置编码，不添加 agent-ID 编码。这个编码描述当前尺度的轨迹位置，扩散步与回报继续使用 MADiff 原有条件嵌入。`env_ts_condition` 只有开启时才提供环境时间条件。

同 agent 与跨 agent 分别使用独立的 Q/K 投影，共享 V。先通过显式 agent 维度构造的块对角关系选择分数，再应用有效性 mask，最后统一 softmax。所有有效的跨时跨 agent 连接均开放；本版没有 restricted/full 分组。

`film` 配置沿用文档命名，实际操作为 token 加上 `MLP(cond)`，随后 LayerNorm。残差使用一个初始化为零的 `gamma`。`adaln` 使用条件生成 scale、shift 和 gate，gate 所在的最终线性层零初始化；它直接控制残差，不再叠加另一个零初始化 `gamma`。这与 [DiT 的单道残差门控结构](https://github.com/facebookresearch/DiT/blob/main/models.py) 一致。

FFN 默认关闭。开启时，film/none 使用独立的零初始化 FFN 门控；adaLN 使用第二组 scale、shift、gate。没有 FFN 时，adaLN 仅生成 attention 所需的 3C 个条件量。`residual=False` 返回 attention 分支；如果另行启用 FFN，则在该结果上执行 FFN 残差细化。

## 实际替换位置

Spread 的 `H=24, history_horizon=0, dim=128, dim_mults=[1,4,8]` 前向 trace 已确认：

| 索引 | 输入 `[B,A,C,T]` | all | mid |
|---|---|---|---|
| 0 | `[B,3,1024,6]`，bottleneck 后 | 新模块 | 新模块 |
| 1 | `[B,3,1024,6]`，第一条 decoder skip | 新模块 | 原模块 |
| 2 | `[B,3,512,12]`，第二条 decoder skip | 新模块 | 原模块 |
| 3 | 注册但 forward 未使用 | 新模块 | 原模块 |

保留原 ModuleList 长度与索引。默认 `attn_type=legacy` 保持原路径。已有 identity/pool/MPNN 实验仍可使用；为避免同一位置被重复替换，`agent_aware` 与非 attention 的 `agent_interaction` 组合会明确报错。

## Mask 与 segment 的准确含义

数据与 evaluator 提供的 `attention_masks: [B,H,A,1]` 表示现有管道允许作为注意力信息源的位置。它不等于条件 clamp mask：有效的未知未来 token 仍应参与联合去噪。

模型将 mask 转为 `[B,A,H]`，按对应 encoder 的下采样次数执行 `max_pool1d(kernel_size=3, stride=2, padding=1)`。一个粗尺度窗口包含至少一个有效位置时，该 key 可用。模块收到 `[B,A,T]` 的 mask 后屏蔽无效 key，并将无效 query 的分支更新置零；全无效样本也保持有限值，残差模式返回原输入。

segment 的边界使用 `ceil(history_horizon / stride)`。粗尺度 token 的中心位置位于历史边界之前时标记为历史，其余属于当前及预测区间。`history_horizon=0` 时不创建 segment embedding。该标记不表示当前观测是否属于执行 agent，也不能区分真实历史与被固定为零的不可见历史。

有效性 mask 的隔离保证适用于新 attention 模块。已有卷积会混合相邻位置，粗尺度 token 也具有时间感受野，因此不能将这个保证扩展为整个 U-Net 对 padding 完全隔离。

## 修正原文参考实现

| 原文问题 | 实现处理 |
|---|---|
| adaLN gate 与 gamma 同为零，梯度无法开启分支 | adaLN 保留一道零初始化 gate |
| gate 已展平，但输出已还原，形状不匹配 | gate、残差和 FFN 完成后统一还原 |
| 在 `[B,A,C,T]` 上直接做 `LayerNorm(C)` 和 FFN | 统一在末维为 C 的 token 布局执行 |
| `residual=False` 仍加原输入 | 保持原 TemporalSelfAttention 的无残差语义 |
| 改 head 数须同步修改 v_ratio | 使用 `dv=C//nheads`，取消冗余 v_ratio |
| 置换测试在 gamma=0 时只检查到恒等映射 | 打开门控并同时置换输入、条件和 mask |

模型是否提升回报需要实验。没有时间编码时 attention 的时间置换等变性，可以通过数值检查说明；它不能直接证明整个卷积模型的回报必然退化。

## 配置与运行入口

| 配置 | 数据 | 采样器 |
|---|---|---|
| `exp_specs/mpe/spread/agent_aware_exp.yaml` | Spread Expert | DDIM，15 步 |
| `exp_specs/mpe/spread/agent_aware_mdreplay.yaml` | Spread Medium-Replay | DDPM，200 步 |
| `exp_specs/mpe/tag/agent_aware_mdreplay.yaml` | Tag Medium-Replay | DDPM，200 步 |

这些文件基于仓库实际存在的 CTDE 配置。每份展开为 `legacy/agent_aware × seeds 100/200/300` 共 6 个实验；同环境、同 seed 的两组只有 `attn_type` 不同。每组 1M 步、`continue_training=False`、每次评估 100 回合、每个环境步重新规划。Medium-Replay 沿用原配置的 DDPM；跨数据集比较时应注意采样器差异。

新模型字段为：

```yaml
attn_type: agent_aware       # legacy / agent_aware
attn_heads: 4
attn_cond_mode: film        # film / adaln / none
attn_use_time_enc: True
attn_use_ffn: False
attn_levels: all            # all / mid
```

后续启动 Spread Expert 对照时，在仓库根目录、已激活的 MADiff 环境执行：

```bash
python run_experiment.py -e exp_specs/mpe/spread/agent_aware_exp.yaml -g 0
```

该命令会启动配置中的 6 个实验，本轮未执行。只跑 mid 消融时，可复制配置，将 `attn_levels` 改为 `mid`，并将 `variables.attn_type` 设为 `["agent_aware"]`。条件模式与 FFN 等选择已进入 job name。

训练和评估继续保存原项目格式，输出位于 `logs/mad_mpe_aa/`，配置展开记录使用原 `logs/variants/` 机制。新模型需要重新训练；本轮未创建模型 checkpoint，也未将诊断或测试记录写入 logs。

## 验证状态

验证环境为现有 `madiff` Conda 环境，PyTorch `2.0.1+cu118`。本轮数值检查在 CPU 上完成。

- 新测试 25 项、既有 interaction 测试 16 项全部通过。执行顺序为先 40 项，再单独验证新增的 diffusion 集成测试 1 项。
- 覆盖独立逐 token 数值参考、统一 softmax、agent 置换等变、位置/segment 信号、有效性 mask、门控可学习性、FFN、all/mid、CFG 和配置/权重重载。
- 实际 `GaussianDiffusion.loss().backward()` 与两步 DDIM CFG 采样通过，训练与采样调用都正确传递 mask，条件 clamp 逐值保持。
- 完整 Spread 尺寸的 all/film 与 all/adaln 均通过前向/反向检查；打开门控后，三个实际调用层的跨 agent Q 投影均有非零梯度。
- 默认 legacy 与改动前快照的参数量、206 个状态项的形状和值、初始化后的 RNG 状态及打开门控后的前向输出完全一致。
- 三份配置展开为 18 个唯一实验路径；9 对 baseline/新模块配置逐键比较只有 `attn_type` 不同。训练脚本的新参数转发也已检查。

复查数值测试的命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n madiff \
  python -m pytest tests/test_agent_aware_attention.py tests/test_agent_interaction.py \
  -q -p no:cacheprovider
```

禁用第三方 pytest 插件自动加载，是为避开当前环境中 typeguard 插件与旧 pytest 的兼容问题；未安装或修改依赖。

完整 denoiser 的实际参数量如下，包含原有注册但未调用的注意力层，不含逆动力学和 EMA：

| 模式，all、无 FFN | 参数量 |
|---|---:|
| legacy | 64,342,550 |
| agent_aware / film | 65,208,214 |
| agent_aware / adaln | 66,584,466 |

film 的完整 denoiser 参数增加约 1.35%。原文的“+33%”只考虑部分投影权重，不能代表完整模型。本次对照也同时改变了 token 组织、投影、条件注入、归一化和有效性 mask，回报差异应归于这个模块整体；尚无新模型回报或延迟结果。
