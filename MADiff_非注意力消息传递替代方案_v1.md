# MADiff 非注意力消息传递替代方案与实验验证

本方案供 Codex 在实际 MADiff 仓库中实施。目标是在固定任务离线多智能体强化学习中，用非注意力消息传递网络替换 MADiff 的跨智能体 attention，检验交互建模方式对去噪与任务回报的影响。第一阶段仅替换一个实际执行的交互位置，保留其余结构，以小预算实验判断是否值得扩展。

这是一份待实施方案。当前没有完成算法实现、训练或性能验证；不得把本文中的假设、示例和验收规则写成已有实验结果。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-11-04
- Verification Status: UNVERIFIED（实施和算法效果未验证；公开源码接口已静态核对）
- Version Label: madiff_non_attention_message_passing_v1
- 产物类型：实现任务书与实验计划
- 主 baseline：MADiff-D，固定任务，无额外执行通信

## 1 给执行 Codex 的任务

请读取本文件以及实际仓库中的 AGENTS.md，先检查本地环境与调用链，再按阶段完成最小修改、必要测试和预算内实验。不要只返回建议，应在可用环境中执行到对应阶段的验收点，并保存命令、配置和结果。

本任务只研究交互模块替代。不要自行加入动态图学习、边分类、消息贡献监督、稀疏 top-k、额外通信、辅助动力学、奖励模型、critic、MH 校正、异构类型网络或 meta-learning。不得为获得正结果连续堆叠模块。

默认执行范围是阶段 A–D：仓库核查、实现、测试与四组 pilot。正式多种子确认是阶段 E，使用明确的正式实验阶段选择执行，避免默认启动数十个百万步任务。阶段 D 得到负结果也算完成一次有效验证，应如实交付，不必为了得到提升擅自改变研究问题。

不要覆盖原有 checkpoint、训练日志、数据集或用户未提交的修改。新实验使用独立输出目录，默认从头训练，禁止意外恢复其他实验。不要打印或复制密码、token、私钥、Cookie 或任何认证信息。

## 2 研究问题与成功标准

研究问题：在保持 MADiff-D 的数据、局部可见条件、时序骨干、CFG、逆动力学和采样配置一致时，用接收端相关的非注意力消息传递替换一个跨智能体 attention 位置，是否改善实际任务回报？

待检验假设：成对消息函数可以学习有助于联合轨迹去噪的依赖，其表现可能优于简单团队池化，并达到或超过原 attention。该假设允许被否定。

分别报告以下三种结论，不能混用：

1. 实现正确：接口、梯度、信息条件、保存加载和环境闭环通过必要检查。
2. 结构可行：在指定数据与预算下，GNN 可以稳定训练、输出可执行动作。
3. 性能改善：实际回报具有多训练种子支持，且增益不能仅由参数量或额外计算解释。

训练 loss 下降不等于改善协调；一次 seed 的正向结果不等于稳定提升；把 attention 换成 GNN 不自动构成论文创新。

## 3 必须保持的信息条件

MADiff-D 的每个执行端 v 只使用自身真实观测或自身历史 h_v，内部生成一份全队想象轨迹。GNN 中其他 agent 的节点表示来自这份本地生成状态，不是接收其他执行端的真实历史或真实计划。

训练可以使用离线同步联合轨迹作为监督目标和扩散加噪样本，但条件掩码必须遵循原有分散设定。不得将其他智能体真实历史直接固定为端 v 的条件，也不得以 GNN 邻居输入的名义引入额外真实信息。

本版不新增真实通信。报告的图边是模型内部计算依赖；记录推理耗时与显存，不声称节约真实通信字节。

特别区分以下掩码：

- condition mask：哪些观测是执行端真实已知、必须固定的条件；
- loss/padding mask：哪些轨迹位置有有效监督；
- graph node/edge mask：哪些节点或消息可以参与聚合。

未知队友不等于无效节点。不能把 condition mask 当 graph mask，删掉需要想象的队友。公开版本虽接收 attention_masks，但部分模型路径未使用，必须检查实际调用，不要因为参数存在就认定其已生效。新增模块默认使用全部有效生成节点，保留原有条件与损失处理。

## 4 源码核查与基线冻结

参考项目：https://github.com/zbzhu99/madiff 。本文依据 2026-11-04 可读取的公开 main 源码制定，实际本地版本可能不同。不要依赖本方案中的固定行号。

优先查找：

| 文件或符号 | 核查内容 |
|---|---|
| `diffuser/models/ma_temporal.py:SharedConvAttentionDeconv` | 共享时序骨干与跨 agent attention 的实际调用 |
| `diffuser/models/temporal.py:TemporalSelfAttention` | 输入、扩散条件、残差与 gamma 初始化 |
| `diffuser/models/__init__.py` | 模型导出与配置解析 |
| `diffuser/models/diffusion.py` | 加噪、条件固定、CFG、逆动力学、loss mask |
| `diffuser/datasets/sequence.py` | 同步窗口、局部条件、回报尺度、padding |
| `diffuser/utils/evaluator.py` | 各端局部条件生成、提取自身计划、动作解码 |
| `run_scripts/train.py`、`run_experiment.py` | 参数传递、日志路径、自动恢复与启动方式 |
| `exp_specs/mpe/spread/*ctde*` | MADiff-D 的实际数据和训练配置 |

公开实现的一个重要细节：跨 agent attention 不只在 decoder 的 skip 特征上，也在 bottleneck 上执行。模型列表可能存在未被 forward 使用的条目，不能直接用 `self_attn[-1]` 猜测待替换位置。请先用 forward hook 或等价追踪记录每个实际调用的索引、调用次数、输入形状。

第一版选择“实际调用的、时间长度最大的 decoder skip 交互位置”。它未必等于原始输入的完整时间分辨率。若有并列，选 forward 中最后执行的那个。记录映射，例如 `last_used_skip -> self_attn[index]`。如果没有 decoder skip 位置，再使用 bottleneck，并明确记录这个适配。

保存以下基线证据：

```text
git_status.txt
source_manifest.json       # commit、dirty状态、修改文件，不含敏感信息
environment_summary.txt    # Python/PyTorch/CUDA/GPU与相关包版本
interaction_trace.json     # 真正执行的交互位置、形状、残差配置
resolved_baseline.yaml     # 解析后生效配置，敏感字段过滤
data_manifest.json         # 数据来源、文件标识/哈希、episode数、窗口数
```

原始配置中可能重复或跨层定义 returns_scale；不要手工选择一个看起来合理的数值。追踪 loader 与 override 优先级，将最终解析值固定在新实验配置中，所有方法保持一致。`continue_training` 可能默认开启；新实验显式关闭。

不要顺手修复无关历史问题。如果发现直接阻断本任务的问题，先定位并用最小公共修复处理，确保四组方法都使用相同修复，单独记录。

## 5 四组核心实验

第一版所有方法只在选定位置发生差别，其余跨 agent attention 保留原样。因此，以下 identity 组不叫“全局无交互”，mpnn 组也不叫“完全去除 attention”。

| 标识 | 选定位置的处理 | 目的 |
|---|---|---|
| `attention` | 保留原 attention | 主 baseline |
| `identity` | 原样返回输入 | 衡量该位置的净作用 |
| `pool` | 不含 attention 的邻居池化与节点更新 | 区分团队摘要与成对消息 |
| `mpnn` | 接收端相关的成对消息、均值聚合、节点更新 | 本方案核心 |

默认模型配置为共享时序骨干，`decentralized_execution=True`，`agent_condition_type=single`（以实际项目等价字段为准），`joint_inv=False`。逆动力学的参数共享策略沿用核验后的分散 baseline，不能顺手从 per-agent 改成共享或联合逆模型。

## 6 非注意力消息传递模块

### 6.1 输入输出与图

模块接口建议与被替换位置一致：

```python
forward(x, time_embed=None, node_mask=None) -> x_updated
```

- x：`[B, N, C, T_r]`；
- time_embed：`[B, N, E]`，来自原模型已处理的扩散条件向量；
- 输出：与 x 的 shape、dtype、device 一致；
- 每个插入位置使用自己的参数，同一位置的参数跨 agent 和时间共享；
- 默认图是无自环的完全有向图：每个 i 从所有 j != i 接收消息；
- 同一时间位置上的节点交换特征，跨时间上下文由保留的时序骨干提供；
- 不计算 Q/K/V、不使用 attention softmax、不学习标量边权、不扩散图结构；
- 一次 forward 在该位置执行一轮消息传递，不新增循环推理。

E 使用现有条件维度。若原向量包含已 dropout 的回报或环境时刻信息，可继续使用；不得重新读取原始回报绕过 `force_dropout`，否则会破坏 CFG 的无条件分支。各节点条件随 agent 置换同步置换。

### 6.2 明确的计算公式

在扩散步 k，位置 ell，将节点特征写为 c_i，原条件向量写为 e_i：

\[
m_{j\to i}=\phi_\psi([c_i,c_j,e_i]),\quad j\ne i.
\]

\[
\bar m_i=\frac{1}{|\mathcal N(i)|}\sum_{j\in\mathcal N(i)}m_{j\to i}.
\]

\[
\tilde c_i=c_i+\gamma\,\rho_\omega([c_i,\bar m_i,e_i]).
\]

消息网络读取接收端和发送端，因而同一个发送节点可以对不同接收节点产生不同消息；均值聚合并不等于所有邻居产生相同信息。

默认网络：

```text
phi: Linear(2*C + E, d) -> Mish -> Linear(d, d)
rho: Linear(C + d + E, d) -> Mish -> Linear(d, C)
```

gamma 是单个可训练残差标量。若原 attention 使用 `x + gamma * update` 且 gamma 初始为 0，则新模块也使用相同初始化。第一版不额外引入 LayerNorm、消息 dropout 或新的时间编码器；原模型外部已有的归一化保留。

邻居数为 0 时，该位置直接返回原输入，避免除零和虚假的邻居更新。默认 N=3 不会出现该情况，但单节点测试必须处理。

若使用 node_mask，必须在消息求和前屏蔽无效发送节点，并按实际有效邻居数归一化；无效接收节点不产生更新。第一版不必从现有 attention_masks 推导该 mask，除非调用链已经明确证明其语义适用。

### 6.3 张量实现要求

可先转换 x 为 `[B,N,T_r,C]`。构造成对输入时：

```text
receiver: [B,N,1,T_r,C]
sender:   [B,1,N,T_r,C]
message:  [B,N_receiver,N_sender,T_r,d]
```

只沿 sender 轴聚合，排除对角线。不要误沿时间、通道或 receiver 轴求平均。小规模 N=3 可直接广播实现；不引入 PyG/DGL 等新依赖。广播仍有 O(N²) 成本，不声称天然降低复杂度。

### 6.4 时间依赖的边界

本版图拓扑固定，消息内容随输入、扩散步和时间位置变化。它不显式构造跨时间边，也不声称学习真实因果图。时序卷积保留意味着每个位置的特征通常具有时间上下文，但实际感受野需依据本地网络计算，不能直接宣称覆盖整个 H。

## 7 池化与 identity 对照的定义

identity：无参数，直接返回 x；保持外部 skip 拼接、归一化和解码操作不变。

pool：先独立编码每个发送节点，再对其他节点求平均：

\[
v_j=\phi_{\rm pool}([c_j,e_j]),\quad
\bar v_i=\frac1{N-1}\sum_{j\ne i}v_j,\quad
\tilde c_i=c_i+\gamma\rho_{\rm pool}([c_i,\bar v_i,e_i]).
\]

```text
phi_pool: Linear(C + E, d_pool) -> Mish -> Linear(d_pool, d_pool)
rho_pool: Linear(C + d_pool + E, d_pool) -> Mish -> Linear(d_pool, C)
```

pool 与 mpnn 都使用相同无自环邻居集合、残差方式、条件信息和一轮更新。区别是 pool 的消息编码不依赖接收端；mpnn 的消息编码依赖接收端。pool 可以用总和减去自身实现，不必实际广播 N² 张量。

## 8 参数量与初始化控制

先统计被替换 attention 的有效参数量，包含时间条件投影和残差参数。mpnn、pool 的隐藏宽度分别从 16 到 512、步长 16 的离散集合中，按参数量差异最小选择；并列选择更小宽度。选择只依赖参数量，不允许查看任务回报后调宽度。

优先使新位置参数量与原位置差异不超过 10%。若无法满足，扩大一次合理搜索范围并记录；仍无法满足时，报告差异并补充容量对照，不填充未参与计算的参数来伪造匹配。identity 作为移除消融不要求参数匹配。

同时报告：位置参数、总注册参数、实际执行路径上的参数，以及模型中可能存在的未使用模块。不能用未调用的 attention 参数制造相同容量的表象。

若实现通过替换 ModuleList 条目完成，新模块替代旧条目；不要同时注册一份不再使用的原 attention 来填充参数量。参数统计必须区分注册与实际使用。

主实验从头训练，不用已训练的 attention checkpoint 初始化 GNN 后再与从头训练 baseline 比较。

对同一训练 seed，保存一个共同的“训练前初始化状态”，让四组共有的时序骨干、其余 attention、逆动力学等权重完全一致。可以先构建基线初始权重，再按精确键名复制共有权重；仅允许待替换位置的预期键不同。检查 shared keys 的张量哈希或数值一致性。不要用笼统 `strict=False` 隐藏其他不匹配。

新模块初始化使用独立、可记录的 RNG 上下文，之后恢复训练随机流；避免仅因多初始化几层网络，就让各组数据顺序、加噪或逆模型初值改变。加载共同初值后再初始化 optimizer 和 EMA；若项目初始化顺序不能如此安排，显式同步对应状态并验证。

## 9 训练与推理

保留原 `GaussianDiffusion` 的加噪、去噪目标、loss weighting、条件覆盖、低温/采样实现和 CFG。GNN 的参数包含在原去噪网络内，由原去噪损失端到端更新：

\[
\mathcal L_{\rm diff}=\mathbb E\|\epsilon-\epsilon_{\theta,\psi,\omega}(Y^{(k)},k,h,c)\|^2.
\]

上式是简式；实现以仓库原有有效位置、padding 和 loss 权重为准，不重写成未经加权的损失。逆动力学保留原训练损失和总损失组合。

本版不需要关系真值、消息贡献标签或额外 BCE。训练时梯度从去噪误差经更新网络、消息网络回到时序编码器。不要 detach 邻居特征，也不要先离线提取固定特征来训练一个分离的 GNN。

执行时，对每端自身历史生成本地联合想象轨迹，取自身当前真实观测与自身计划下一帧，经原逆动力学得到动作。保持原 evaluator 的信息条件与动作裁剪。EMA 中必须包含并更新新模块，评估不能意外加载旧 attention 网络。

## 10 代码修改范围与配置

优先新增一个轻量交互模块文件，例如 `diffuser/models/agent_interaction.py`，包含 Identity、Pool、MPNN。原模型新增可选配置，默认值完整保持原 attention 路径。只在已确定的一个位置替换模块，不复制整个 U-Net 或另造完整训练框架。

建议字段：

```yaml
# 新字段；实际配置层级以本地 loader 为准
agent_interaction: attention       # attention | identity | pool | mpnn
interaction_site: last_used_skip  # 运行时解析为具体索引并写入 manifest
interaction_hidden_dim: null      # 构建前按参数量确定，保存解析后的整数
```

这些字段必须沿 YAML -> Config -> model constructor 贯通，不能只写配置不使用。原有 `use_temporal_attention` 在部分路径中并非真正控制模块构造，不能将它直接当作新增方法开关。保留它的原语义。

所有组共有配置：

```yaml
decentralized_execution: true
joint_inv: false
continue_training: false
# 其余值从核查后的原 ctde 配置继承
```

每组独立文件或同一结构化实验矩阵均可，选择更符合现有项目的方法。不要改变全局配置解析器来支持一个局部模块。

## 11 阶段 A 仓库与环境核查

1. 定位实际 MADiff 仓库。优先当前任务目录和已有明确项目路径，不扫描密码目录或读取认证配置。若不存在仓库，使用官方项目新建独立目录，不覆盖已有项目。
2. 记录 git、运行环境、GPU 和可用数据。优先已有 Linux/CUDA 训练环境，旧版依赖以项目要求为准；不要一开始升级整个 PyTorch 栈。
3. 优先使用已有数据。缺少数据时，从 MADiff/OMAR 官方说明确认路径与版本，不能用临时在线生成的数据冒充标准数据集。不可用时完成代码与无数据测试，并明确记录缺失条件。
4. 用原配置跑通 dataset -> model -> loss -> sample -> inverse -> environment 的最小闭环。关闭自动恢复，使用独立 smoke 目录。
5. 完成交互位置追踪和最终配置解析。将核查结果写入 `preflight.md`。

通过标准：确认真实生效的 MADiff-D 配置；原模型可 forward/backward；数据和环境版本匹配；明确唯一替换位置。若原始闭环失败，先定位基线问题，不先修改 GNN。

## 12 阶段 B 必要的正确性检查

此处测试针对可导致错误结论的行为，不机械重复实现。

1. 形状与数值：四组在实际各维度和选定位置 shape 一致，输出有限，CPU 和可用 GPU 路径兼容。
2. agent 置换等变：只测共享交互模块，将 x、节点条件、有效节点 mask 同步置换，输出应对应置换。FP32 默认 `atol=1e-5, rtol=1e-4`。不要求具有 per-agent 逆模型的整套策略天然等变。
3. 有向消息轴：构造非对称输入并只扰动发送端 j，确认变化能通过 j→i 进入接收端，避免 receiver/sender 轴反转。
4. 梯度与残差：原 attention gamma=0 时，新模块也可从零开始。第一步 phi/rho 梯度为零可能是正常的，不能误判为断图；应检查 gamma 梯度及更新后消息参数梯度。独立单测可把 gamma 设为 1 验证 off-diagonal 依赖；不得因此修改实际训练初始化。
5. 边界与 mask：无自环、N=1 不除零、无有效邻居返回原输入；无效发送节点的任意大扰动不影响有效接收端；分母等于有效邻居数。
6. 未修改 baseline 回归：相同权重、输入、随机状态与 eval 模式下，默认 attention 路径与修改前输出一致。不要因重构改变原 baseline 的行为。
7. 序列化与 EMA：保存/加载后输出一致；optimizer 覆盖新模块；EMA 新模块确实更新；实际 evaluator 可重建对应方法。
8. CFG：conditional、force_dropout 两支可运行。检查新模块只接收原有处理后的条件，不存在原始回报旁路。
9. 局部信息隔离：固定端 v 自身历史和采样 RNG，扰动其他端真实历史，端 v 的独立条件样本构造及输出应保持一致。应在 per-view 生成函数上测试，不把批处理重排或全局 RNG 消耗误当信息泄漏。

主比较使用相同的训练数据、采样与加噪规则。不把 condition mask 未知位置当无效 graph node。

## 13 阶段 C 小批量与环境 smoke

四组各执行 200 个 optimizer updates，单训练 seed，使用相同数据配置；单组硬时限默认 30 分钟，不包括首次合法依赖安装与数据准备。小数据过拟合诊断可固定一组有效轨迹窗口，使用固定诊断噪声检查拟合能力；不能把该曲线称为泛化误差。

至少完成：

- loss 有限，新模块梯度与参数更新可观测；
- checkpoint 和 EMA 可加载；
- conditional sampling 正常，自身已知观测在每步采样中按原规则固定；
- 每组至少 3 个完整环境 episode，动作维度、范围和 episode 终止正常；
- 新模块的 hook 证明实际调用，而非仅注册未使用。

发生 crash 时保存报错、退出码、最后 checkpoint 与配置，定位后只作必要修复；确定性配置/代码问题修复后可复跑 smoke，记录原因。不要反复重启相同崩溃配置，或悄悄减少实验组。

## 14 阶段 D 默认 pilot

### 14.1 数据与训练预算

- 数据：OMAR `simple_spread-expert`，原数据对应的 MPE fork。
- 方法：attention、identity、pool、mpnn。
- 训练 seed：100。
- 每组：50,000 个 optimizer updates，总计 200,000 updates；梯度累积 microstep 不得误计为 update。
- 评估：训练前 sanity，随后每 10,000 updates；每次每组 20 个相同初始环境 seed 的 episode。
- 主 checkpoint：固定 50,000 updates 的最终 checkpoint，不能按测试回报挑 best。
- 日志：每 100 或 1,000 updates 输出进度，避免逐 batch 大量打印。
- 单组 wall-clock 上限：默认 8 小时。根据阶段 C 速度估计是否可完成；如预计超出，先记录可执行的等预算下调方案并让所有组一致使用，不只缩短较慢方法。

原配置中的 batch size、累积步数、学习率、EMA、H、normalizer、回报尺度、CFG 强度、采样步数沿用核查后的值。不要为了 pilot 单独启用 DDIM；如果统一选择 DDIM，则必须明确这是统一适配协议，不能把原 DDPM baseline 分数混入比较。

50,000 updates 是早期机制筛查，不能用它与论文百万步最终分数直接比较。各组都在相同较短预算下比较。

### 14.2 随机性与数据使用

四组使用相同训练 episode 集和数据顺序规则。尽量将 data/noise/condition-dropout 随机流与模型初始化区分记录，保证新增模块不会任意改变其他随机过程。评估使用独立 RNG，不应改变随后训练的样本/噪声序列。

主实验沿用原训练数据使用方式。建立一批固定离线诊断窗口，并明确它可能来自训练集；不要将其称为 held-out test。若本地已经有正规的 episode 级 train/validation 划分，则所有方法统一沿用；若新增划分，需为四组同步重新训练，并仅用训练部分拟合 normalizer。不得先按窗口随机拆分同一 episode 来制造独立验证集。

### 14.3 必须保存的指标

主指标为原 evaluator 的 episodic return，保留原始奖励聚合与尺度。先记录其究竟是 team sum、mean 还是其他约定，不能在方法间变化。可以附加统一的 team mean，但不能偷换主指标。

辅助指标：

- 原总 loss、去噪损失、逆动力学损失（以真实代码拆分为准）；
- 固定诊断窗口上的有效位置去噪 MSE，可按噪声区间分桶；
- 选定位置 gamma、消息/更新范数、新模块梯度范数；
- 每个 episode 的原始回报和长度；
- 固定 N、H、batch、采样配置下的端到端动作决策延迟；
- GPU 推理预热后至少 30 次测量，计时前后同步，报告 median/p95、峰值显存；
- 参数量、训练用时、实际去噪步数、设备与 batch size。

Spread 的地标覆盖和争抢是解释指标，不自动等于标准 success rate。可使用未归一化环境坐标计算各地标最近 agent 距离、按原碰撞规则统计碰撞；若定义终点目标分配与重复目标数量，须固定最近地标与并列规则，并在报告中标为自定义指标。不应随意设一个距离阈值然后把它冒充官方成功率。

### 14.4 Pilot 判断

- mpnn 稳定且固定最终 checkpoint 回报优于 attention：有继续验证价值，尚未证实稳定提升。
- mpnn 与 attention 相近，但优于 pool/identity：说明成对交互可能有作用，需要多种子确认。
- mpnn loss 更低但回报不升：记录生成拟合与决策收益分离，不能宣称改善协调。
- 四组表现相近：可能位置作用弱或训练过短；先检查该位置 gamma 和扰动敏感性，不立即叠加新模块。
- mpnn 明显退化：检查 residual、参数量、时间上下文、局部条件、EMA 与环境版本；没有实现问题则保留负结果。

## 15 阶段 E 正式确认与全替代扩展

该阶段用于明确选择正式实验任务时，不是默认 pilot 的隐含无限预算。

正式确认建议：Spread Expert 与 Medium，两组数据分别训练；四种方法；训练 seeds 100/200/300/400/500；每组使用原配置相同训练预算，通常为 1,000,000 updates，实际以核验基线为准。正式实验从共同初值重新开始，不将 pilot 训练后的权重仅用于某个方法。

每个最终 checkpoint 至少 100 个固定、各方法共享的独立评估 episode seed。定期评估频率与次数在各组完全一致；主比较使用固定最终 checkpoint。报告全部训练 seed，而非最优 seed。

统计基本单位是训练 seed。先对每个 seed 的 episode 求平均，再报告跨 seed 的均值、标准差与配对回报差。bootstrap 时重采样训练 seed；可另做层次 bootstrap，但不能把 500 个 episode 当 500 个独立训练结果。5 个训练 seed 的区间仍可能不稳定，应展示原始逐 seed 数值。

只有当 paired mean difference 的 95% 区间支持正向、增益达到预先说明的实用程度且不靠某个 seed 独撑时，才写“在这些任务上得到稳定改进”。不使用对负回报不适用的简单百分比变化；优先报告原始差值。没有明确 score 归一化参考就不自行构造 normalized score。

如果要进一步声称增益来自“跨 agent 消息”而不是节点自身新增的非线性变换，补充 receiver-only 容量对照：保留同等宽度的节点更新网络，但在训练和执行中都将邻居聚合向量固定为 0，让更新仅依赖自身特征和自身条件。该对照不等于 identity，也不适用无邻居直接返回输入的边界分支。仅在训练好的模型中临时打乱邻居，可作为敏感性诊断，但存在分布偏移，不能替代从头训练的 receiver-only 对照。四组 pilot 本身不承担这一更强归因结论。

如果单位置结果值得继续，才新增“全部实际调用交互位置替换”的扩展。需要重新追踪所有位置并在各尺度构建相应 C/E 的模块。完整 mpnn、完整 pool、全部 identity 与原 attention 对照另列，保持相同训练协议。单位置实验不能被写成完全去除了 attention。

先完成 MPE，再考虑 OG-MARL 的固定 MA-MuJoCo 拆分或 SMAC 同构单位场景。不要在第一版任务中同时承担数据转换、离散动作适配、多场景训练和新算法四件事。

## 16 运行方式与可执行脚本

官方启动形式为：

```bash
python run_experiment.py -e exp_specs/mpe/spread/mad_mpe_spread_ctde_exp.yaml
```

上面是基线入口示例，不代表本地配置已经安全可直接启动；先确认 num_workers、seed 列表、设备、continue_training、覆盖规则与日志目录。

实施时新增最小运行入口或脚本，建议支持：

```text
--stage smoke | pilot | confirm
--method attention | identity | pool | mpnn | all
--dataset expert | medium
--seed <int>
--output-root <path>
```

这些是要求实现的接口，不是声称仓库已有这些 CLI 参数。脚本负责写出项目真正接受的唯一化配置并调用现有入口，不能给原 CLI 传不存在的参数。Windows 用 PowerShell 脚本，Linux 用 shell/Python 脚本，按实际运行环境选择，不混用 shell 语法。

运行脚本默认串行，避免四组争抢同一 GPU 造成资源和时间比较失真。优先使用已有 Python 环境。必须提交实际可复制命令与 resolved config，不能在最终报告中保留 `<repo>`、`TODO` 或未填路径作为“已经运行”的证据。

远程环境只使用已有明确授权和可达的连接；不自动申请或消耗新的付费算力。缺失数据、权限或设备时，区分“实现已完成”“本地检查已通过”“训练未执行”的状态，列出具体未完成项，不生成虚假训练记录。

## 17 输出目录与数据格式

遵循仓库现有日志规范，可增加一个统一汇总根目录：

```text
outputs/non_attention_v1/
  preflight.md
  source_manifest.json
  interaction_trace.json
  configs/
  tests/
  smoke/<method>/
  pilot/<dataset>/<method>/seed_<seed>/
    resolved_config.yaml
    run_manifest.json
    train_metrics.csv
    eval_episodes.csv
    latency.json
    checkpoints/
    stdout.log
    stderr.log
  summary/
    per_seed.csv
    comparison.csv
    return_vs_updates.png
    diagnostics.png
    report.md
```

已有 logger 采用不同结构时，使用 manifest 映射实际路径，不为满足目录树重写 logger。保留原 checkpoint 格式与 optimizer 状态，便于检查和正式恢复。

建议最小记录字段：

```text
train_metrics.csv:
run_id,method,dataset,seed,optimizer_update,wall_seconds,
total_loss,diffusion_loss,inverse_loss,gamma,message_norm,grad_norm

eval_episodes.csv:
run_id,method,dataset,train_seed,optimizer_update,eval_seed,
episode_return,episode_length

run_manifest.json:
git_commit,dirty_patch_id,method,site_index,site_shape,
train_seed,init_seed,eval_seeds,config_path,data_manifest,
parameter_counts,device,start_time,end_time,status,exit_code
```

某个指标无法从现有损失中可靠分离时，记录缺失原因，不能伪造为 0。时间使用带时区的 ISO 8601；用户阅读的日期按 Asia/Shanghai 解释。

## 18 完成后交付的报告

`summary/report.md` 应包含：

1. 修改文件与唯一替换位置，说明本次是单位置还是全替代。
2. 原 baseline 的配置、源码版本、数据和环境版本。
3. 每项正确性检查结果，包括失败后如何修复。
4. 真正执行过的命令、预算、输出文件与退出状态。
5. 四组方法的参数量、训练量、推理延迟和回报。
6. 区分 pilot 趋势与正式多种子证据。
7. 可以支持的结论、不能支持的结论，以及下一步是否值得做。

允许结论为“实现正确，但没有观察到优势”。不得把未完成训练或未显著结果改写为预期提升。不得为了让表格好看删掉失败 seed；资源中断和算法数值失败分开记录。

最终验收清单：

- [ ] 原 attention 默认路径保持原行为；
- [ ] 真正定位并替换一个实际调用位置；
- [ ] Pool 和 MPNN 均不使用 Q/K/V、attention softmax 或学习标量边权；
- [ ] 核心检查、序列化、EMA 和局部信息隔离通过；
- [ ] 四组使用一致的数据、初始化共有部分和训练/评估预算；
- [ ] 默认阶段实际完成或准确记录外部阻断；
- [ ] 汇总包含原始逐 episode/逐 seed 数据与可复制命令；
- [ ] 算法效果、实现正确性和新颖性分开描述。

## 19 参考来源与定位

- MADiff 原论文，§3.1、§3.3：跨智能体特征交互与无通信分散执行。https://arxiv.org/abs/2305.17330
- 官方项目与数据/训练说明：https://github.com/zbzhu99/madiff
- 模型与交互调用：https://github.com/zbzhu99/madiff/blob/main/diffuser/models/ma_temporal.py
- 原 attention 与时序骨干：https://github.com/zbzhu99/madiff/blob/main/diffuser/models/temporal.py
- 参数传递入口：https://github.com/zbzhu99/madiff/blob/main/run_scripts/train.py
- 分散 Spread 配置：https://github.com/zbzhu99/madiff/blob/main/exp_specs/mpe/spread/mad_mpe_spread_ctde_exp.yaml
- MPNN 基本范式：https://proceedings.mlr.press/v70/gilmer17a.html
- OMAR 数据与环境：https://github.com/ling-pan/OMAR
- OG-MARL：https://github.com/instadeepai/og-marl

这些来源支持基线事实与消息传递范式。本文件的四组对照、插入位置选择、参数匹配和分阶段预算是本次建议，不是 MADiff 作者已完成的实验。

## 20 可直接复制给 Codex 的执行指令

```text
请按《MADiff 非注意力消息传递替代方案与实验验证》执行本次工作。

先定位实际 MADiff 仓库、读取 AGENTS.md、核查数据和运行环境，记录实际
attention 调用及配置。保留无额外通信的 MADiff-D 信息条件，只替换一个
实际执行的 decoder skip 交互位置，新增 identity、pool、mpnn 三种选项，
并确保默认 attention 路径完全保留。

本次默认执行阶段 A–D：小范围实现，完成必要正确性检查和环境 smoke，
再运行 Spread Expert 的四组、seed=100、每组50,000 optimizer updates
的等预算 pilot。严格按文档的配置、初始化、EMA、评估和输出要求执行。
正式多种子和全位置替换通过单独阶段执行，不默认扩张计算预算。

不要额外加入动态图、边监督、通信协议、critic或其他算法模块。
遇到原 baseline 问题先定位再最小修复；缺少外部条件时完成可独立工作，
准确报告剩余项。最后提交实际改动、验证、运行命令、输出路径与比较报告。
即使结果没有提升，也应保留并解释，不为了正结果擅自改变方案。
```
