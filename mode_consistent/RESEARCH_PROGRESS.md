# CI-CoDiff 研究进程、实验规划与证据审计

> 更新日期：2026-08-21。本文档记录动态研究过程，包括阶段状态、实验决策、数据审计、脚本、产物、失败结果与下一步计划。稳定的研究问题、算法定义、benchmark 和复现入口见 [`README.md`](README.md)。

实验状态、研究决策、脚本或产物变化时更新本文档。算法接口、模块定义、benchmark 或复现命令变化时更新 README。

一次修改同时影响两类信息时，必须同步更新两份文件。

## 1. 当前状态

当前结论是 **Conditional GO**：算法工程框架 v0 已贯通，但没有论文级效果证据。

继续研究的前提是中央 code 能形成稳定行为语义，且目标决策时刻的局部历史包含足以让各 agent 一致恢复 mode 的信息。

已实现的工程链路包括：无标签训练 view、中央 VQ teacher、局部 categorical aligner、置信度拒绝、mode 条件 MADiff、集中式 mode value head、场景级划分与 focal-agent 分散采样。

截至本次更新，`tests/mode_consistent` 为 `34 passed`。旧 `H=5` seed 0 证据保留为历史对照；当前 P2 已改为 VO-MASD 权重方向的 VQ 损失，且一条 `H=25` 完整 episode 只产生一个团队 code。真实 OMAR 40-episode 子集上的 VQ/no-code 2-step GPU smoke 与冻结评估已通过；全量 20 万 episode 的 H=25 正式评估尚未运行。所有结果仍标记为 `pending_user_discussion`。

P2 当前协议为 `trajectory_scope=full_episode, H=25, stride=1`。共享 GRU 先分别编码 agent 轨迹，再聚合为一个 `z_e`并查询单个 `K x D` 团队 codebook；Decoder 用该 code、局部观测和 agent identity 重建全轨迹连续动作。损失为 `reconstruction + commitment + 0.001 * codebook`。旧 H=5 checkpoint 可以显式 `standard` 语义读取，但不与 H=25 证据合并。

### 1.1 共同决策协议

每次验证后先展示原始证据、异常、备选解释和建议。任何 GO/NO-GO、超参数冻结、实验路线变化或下一阶段开放，都必须与用户讨论并得到确认后才能写入阶段结论。

评估入口只生成 `review_status=pending_user_discussion` 的事实报告，不自动修改 README 或本文档，也不自动启动后续阶段。

| 阶段 | 当前证据 | 审计结论 |
| --- | --- | --- |
| P0：相关性缺口 | XOR 模拟符合 `1-K^(1-N)`；公共 cue 可降低 MMR | **GO，可复现** |
| P1：OMAR 数据 | Spread 的 5 个 collector 主要覆盖 5 个不同 assignment，collector-assignment NMI=0.939；Tag/World 只有结果代理 | **初步 GO，不是同状态反事实证据** |
| P2：mode discovery | H=25 整轨迹、VO-MASD VQ 权重的训练/评估入口已通过 smoke；正式 seed 0 尚未运行 | **待确认预算，不自动推进 P3** |
| P3：局部共同信息 | `LocalModeAligner` 已实现局部历史蒸馏、agreement 与 unknown | **待决策时刻可辨识性曲线** |
| P4：条件策略 | FiLM、CFG、训练与分散采样路径已贯通 | **待完整 rollout 与负对照** |
| P5：价值改进 | `ModeValueModel` 可拟合数据回报 | **仅 value seam，未启用优势加权** |
| P6：外部效度 | Tag/World 与 SMAC 尚未进入正式模式定义和训练 | **暂停** |

测试通过只证明模块接口的软件行为。论文结论还必须具备 held-out 协议、负对照、标签权限审计、数据哈希、完整 rollout 和多 seed 统计。

### 1.2 H=5 seed 0 原始证据

四个分支使用相同 dataset SHA-256、scenario split、`H=5` 和 `stride=1`。validation/test 各含 900 条来源 episode、18,900 个窗口。

| test 指标 | VQ | no-code | oracle-code | KMeans |
| --- | ---: | ---: | ---: | ---: |
| 动作 reconstruction MSE | 0.3519 | 0.4071 | 0.4236 | 不适用 |
| first-action MSE | 0.5116 | 0.6067 | 0.6197 | 不适用 |
| assignment NMI | 0.0012 | 不适用 | 特权输入 | 0.0049 |
| assignment ARI | 0.0002 | 不适用 | 特权输入 | 0.0032 |
| validation 映射后的 test accuracy | 0.1703 | 不适用 | 特权输入 | 0.1696 |
| hard code perplexity | 3.1217 / 6 | 不适用 | 不适用 | 5.9843 / 6 |

VQ 相对 no-code 的 reconstruction MSE 降低 13.55%，但该收益不能解释为恢复了 assignment mode：chance accuracy 为 `1/6=0.1667`，而 VQ 仅为 0.1703。额外只读诊断显示，VQ code 与窗口起点的 NMI 为 0.1555；每条 episode 平均使用 2.92 个 code，归一化 code entropy 为 0.4811，只有 1.56% 的 episode 全程保持单一 code。

使用数据生成器的已知 PD 规则，可以从每个时间步的 observation/action 以 100% accuracy 反推出目标 landmark，并以 100% accuracy 恢复 episode mode。这排除了“受控数据没有 mode 信号”的解释。当前 learned oracle-code 的 test MSE 反而比 no-code 高 4.06%，因此现有 oracle-gap 指标无定义；在修正或充分训练该对照前，不能用 reconstruction 提升宣称 P2 有效。以上是待讨论证据，不构成新的 GO/NO-GO 决策。

### 1.3 OMAR Spread H=5 seed 0 原始证据

全量数据包含 20 万条 horizon 25 的 episode，5 个 collector 各 4 万条。训练文件不含 `true_modes`；数据 SHA-256 为 `00989b5e8c240c30b0a0f5239cb0aa7cdb828548820ee75f117bc803ac291ef1`。每个 collector 内按 70%/15%/15% 划分，test 包含 3 万条 episode、63 万个窗口。VQ 与 no-code 使用相同 split SHA-256 `093f56a5d739a2f0c5d391c0e9387257f6540c93e9380d06a410903fe44f2d9d`。

| test 指标 | VQ | no-code / 审计对照 |
| --- | ---: | ---: |
| 动作 reconstruction MSE | 0.4011 | 0.4480 |
| first-action MSE | 0.5056 | 0.5714 |
| code-shuffle reconstruction MSE | 0.6016 | 不适用 |
| hard code perplexity | 2.6661 / 6 | 不适用 |
| 主 code 窗口占比 | 74.30% | 不适用 |
| 相邻窗口 code agreement | 0.9047 | 不适用 |
| code-terminal assignment NMI | 0.0025 | 近随机 |
| code-collector NMI | 0.0017 | 近随机 |
| terminal assignment 成功率 | 94.39% | 数据审计 |

VQ 相对 no-code 的 reconstruction MSE 降低 10.47%，first-action MSE 降低 11.51%；打乱 VQ code 后 reconstruction MSE 上升 49.99%。因此 code 对动作重建有信息，但该信息没有对应最终 agent-landmark assignment。

额外只读分层诊断显示，code-terminal assignment NMI 在窗口起点 0 至 20 始终只有 0.0024--0.0059，并非被早期不可辨识窗口简单稀释。主 code 占比从起点 0 的 23.00% 持续增至起点 20 的 97.04%，说明当前 code 更接近运动阶段或动作难度表征，并在轨迹后段坍缩。

相反，episode 粒度的 collector-terminal assignment NMI 为 0.9392：5 个 collector 分别主要收敛到 5 个不同 permutation，第 6 个 permutation 在当前混合数据中几乎缺失。这证明 OMAR Spread 存在强多 collector 协调结构，但不是六类平衡数据，也不是相同初始状态下的 paired 反事实数据。当前证据把主要问题指向 VQ reconstruction 目标缺少跨窗口/整条轨迹的 mode 一致性约束，而不是数据完全没有 mode。该解释仍需与用户讨论后才能转化为模型修改决策。

### 1.4 VO-MASD 源码审计与设计边界

本轮直接核对了本地 VO-MASD 源码，而不是只沿用 `README copy.md` 的二手总结。关键执行链为：

```text
H 步 per-agent observation/action
  -> 实体分解与注意力
  -> 双向 GRU TrajEncoder，得到每个 agent 的 z_e^i
  -> global-state MAT Grouper 动态产生 subgroup
  -> 按 subgroup 大小选择 K × (mD) codebook 并联合量化
  -> 将组 code 拆回 agent-specific skill component
  -> recurrent Decoder 重建 H 步离散动作概率
```

VO-MASD 的 VAE 预训练最小化动作负对数似然与 VQ 距离；Grouper 另将重建 log-probability 和负量化误差作为内在回报，通过 PPO 学习分组。预训练后冻结 skill 组件，再由在线 MAPPO 高层策略选择 skill。它发现的是可动态切换、可按子组组合的 multi-agent skill，不是直接面向单环境对称最优 assignment 的共享 team mode。

对当前项目形成的设计边界如下：

- **保留**：`H` 步 observation/action 编码、离散 VQ、完整片段动作解码，以及 discovery 与下游策略分阶段冻结；
- **不引入当前 P2**：3D subgroup 层、group-size codebooks、global-state grouper PPO、稀有 code 频率筛选和在线 MAPPO；
- **维持当前建模假设**：所有 agent 先视为一个固定团队，使用单个 `K × D` codebook；同一 code ID 配合 agent identity 解码不同角色；
- **未解决点**：VO-MASD 的 reconstruction/VQ 目标同样不保证 assignment 语义，不能直接修复当前 code 的运动阶段混淆；
- **已确认最小修正**：取消同 episode 的滑动窗口独立量化，改为全轨迹唯一 code；当前只完成工程 smoke，有效性需正式证据判断。

正式 README 已据此更新框架和四个子模型的比较边界。原 `README copy.md` 同时混合了 VO-MASD 复现说明、未实现的 RTG/inverse-dynamics 构想和当前项目描述，已在信息归并后删除。

## 2. 可行性判断与研究出口

paired MPE 中，同一初始局部观测可对应六种 collector assignment。若执行前没有公共 cue、共享随机性、通信或可见行为前缀，`g_i(h_i^0)` 不可能知道采集时采用了哪个 `z`。

这属于信息不可识别，而不是模型容量不足。严格 Dec-POMDP 主线只允许确定性公共约定、置信度拒绝，或先执行可观察的信息动作。

共享 seed、公共 beacon 和 agreement key 必须作为 correlation-device extension 单独报告。

项目保留两个研究出口：

1. **算法路线**：在存在可用共同信息的区域，证明 learned code 能降低 MMR 并提高 return；
2. **诊断路线**：建立不可识别性边界、数据协议和 benchmark，说明何时任何方法都不应承诺相关多模态执行。

若 P2 或 P3 阶段门失败，应停止大规模 diffusion 与 SMAC 训练，转向诊断路线。

## 3. 数据审计与采集计划

### 3.1 OMAR MPE

数据根目录为 `/home/lotus/lotus/lhh/offline_datasets/OMAR/mpe`。

全量正式审计确认，Spread expert 的五个 collector 各自高度集中于一个 agent-landmark assignment，且不同 collector 覆盖五个不同 permutation。冻结 test 集的 collector-assignment NMI 为 0.9392，六个 assignment 的 episode 计数为 `[92, 6002, 5978, 5969, 5949, 6010]`；当前数据缺少平衡的第六种模式。

这支持“多次在线训练分别收敛，再混合策略数据”能形成多个行为簇。

该结果不能证明相同初始状态下存在多种反事实最优动作：原 collector 没有共享 scenario，collector 身份也不是真实 mode 标签。

Tag 和 World 的 first-capture agent 只属于结果代理，尚未建立稳定 tactic 定义。

`artifacts/omar_mode_audit.json/.png` 是先前审计快照。当前正式 P2 evaluator 可重建 collector、return、终态 assignment、code usage 和 code-shuffle 指标；按窗口起点的额外诊断暂时只记录在本文档，若成为论文指标则需先纳入最小正式入口。

当前正式入口不把 collector 当作 mode 标签。`scripts/convert_omar_mpe.py` 将五个 collector 合并为无 `true_modes` 的 `EpisodeStore`；训练只读取 observations、actions 和 mask。每个 collector 内独立完成 70%/15%/15% 划分，冻结后的 evaluator 才读取 collector、return 和终态 assignment 代理。

真实 OMAR Spread smoke 每个 collector 取 8 条 episode，共 40 条。转换后所有张量有限、mask 全有效、terminal 协议一致；train/validation/test 分别为每个 collector 的 6/1/1 条。H=25 VQ 与 no-code 的 2-step GPU 训练及 unlabeled comparison 已通过。VQ 在 2 steps 后只激活 1 个 code，这仅表明极小预算不能用于判定有效性，不得解读为正式 code collapse 结论。H=5 的全量历史结果见 1.3 节；H=25 全量评估尚未执行。

### 3.2 受控 paired Simple Spread

`generate_paired_assignment_dataset` 在相同 scenario seed 上分别执行六种 agent-landmark assignment，用于验证“同状态、不同协调约定”。assignment、quality 和 scenario 字段只允许用于划分与评估。

该数据由无噪声固定 assignment PD 控制器与 MPE 环境交互 rollout 生成，不是 OMAR 专家数据，也不是 MAPPO、QMIX 等 online MARL 策略收集的数据。`expert` 只表示控制器无噪声和 dropout；六种 assignment 在随机场景分布上的平均回报接近，但固定场景内可能因路径和碰撞产生明显回报差异。因此目前应称为“六种可行协调约定”，不能直接称为“每个状态下六种严格等价最优策略”。

`scripts/render_mpe_modes.py` 可复现固定 seed 的六种 rollout。seed 0、horizon 25 下六种 mode 均实现一一地标覆盖，最大目标终点距离约 `0.0015`；对应视频、轨迹图与摘要保存在 `artifacts/mpe_mode_videos/`。

计划中的正式切片包括 balanced、imbalanced、single-mode、mixed-quality 和 complete-support。mode coverage 与控制器质量必须正交，避免模型只学习“某些 mode 永远更优”的固定先验。

### 3.3 benchmark 不足时的采集协议

1. 独立训练 `K` 个 MAPPO 或 MATD3 collector，使用不同 seed、初始化或轻量采集期 shaping；
2. 仅保留高回报且行为不同的策略，并按 assignment 或 tactic 去重；
3. 冻结策略，在同一组 scenario seeds 上重新 rollout，生成 paired counterfactual 数据；
4. 构造平衡主数据、不平衡数据和混合质量消融；
5. 保存 collector、tactic 和 outcome 元数据，但训练 loader 不暴露这些字段；
6. headline result 使用原始环境奖励，采集期 shaping 不进入评估回报。

单个在线 joint policy 收敛到一个最优附近策略不是阻碍。采集时应利用多个独立收敛策略形成混合数据，再通过共享 scenario rollout 控制状态分布差异。

## 4. 阶段路线与停止条件

| 阶段 | 任务 | 完成门槛 | 状态 |
| --- | --- | --- | --- |
| P0 | XOR 的 `K,N,rho` 与 MMR | 理论和模拟一致；entropy 揭示 mode collapse | 已完成 |
| P1 | OMAR 与 paired MPE 数据审计 | mode 与 quality 分离；明确 paired 限制 | 初审完成 |
| P2 | H=25 整轨迹团队 mode learner | held-out collector 稳定；usage 非坍缩；code 有助于动作且对齐行为结构 | 实现与 smoke 完成，正式 seed 0 待确认 |
| P3 | `g_i(h_i^t)` 可辨识性 | 联合报告 coverage、MMR、entropy；`t=0` 负对照正确失败 | framework v0，待实证 |
| P4 | learned-code 条件 MADiff | learned 接近 oracle，优于 random/no-code；完成环境 rollout | train/sample 已贯通 |
| P5 | value 或 advantage weighting | 在 P4 固定后提高 return，且不恶化 mismatch/support | weighting 未启用 |
| P6 | Tag/World、SMAC/SMACv2 | 预先定义 tactic；原始奖励；多 seed | 暂停 |

停止条件：若 P2 不能在 held-out 数据上稳定恢复行为结构，或 P3 在实际决策时刻没有非平凡 coverage，则不进入大规模 P4–P6。

## 5. 正式实验矩阵

| 实验 | 唯一变化 | 主指标 | 必要负对照 |
| --- | --- | --- | --- |
| E1 XOR | `K`、`N` | MMR | 理论 `1-K^(1-N)` |
| E2 contextual XOR | common cue 可用率 `rho` | MMR-rho 曲线 | 无 cue、固定 code、随机 code |
| E3 mode discovery | 轨迹表征与 `K` | NMI、ARI、usage、perplexity | raw 特征、oracle assignment feature |
| E4 local identifiability | prefix 长度与可见信息 | coverage、MMR、entropy | `t=0`、agent permutation、label shuffle |
| E5 conditional policy | oracle、learned、random、no-code | return、support、mode fidelity | 同预算 MADiff-D |
| E6 value improvement | weighting on/off、beta | return、MMR、support | 随机化 mode-quality 绑定 |

方法之间固定 dataset seed 和 evaluation scenario seed 做配对。训练随机性与采集随机性分层报告；先估计方差，再确定正式 seed 数，默认目标不少于 5 个 seed。

## 6. 标签权限与评估协议

`EpisodeStore` 可保存 `true_modes`、`quality_ids`、`scenario_ids` 和 `collector_ids`，但所有表示学习函数只接受 `UnlabeledEpisodeView`。

该 view 仅含 observations、actions 和 mask；contract test 检查它不暴露 reward 或审计标签。

标签只允许用于以下环节：

- `scenario_ids`：防止 paired episode 跨越训练集和评估集；
- `true_modes`：训练后计算 NMI、ARI 与 best-mapping accuracy；
- `quality_ids`：检查 mode 与数据质量是否混淆；
- collector metadata：数据审计与 held-out collector 划分。

禁止使用 true mode 选择 codebook、调节模型超参数或生成训练目标。阈值和温度必须在验证集确定，不能在最终测试标签上调优。

## 7. 研究脚本、代码与产物状态

```text
mode_consistent/
├── data/
│   ├── offline.py       # EpisodeStore 与无标签训练 view
│   ├── mpe_modes.py     # paired Simple Spread collector
│   └── omar_mpe.py      # OMAR adapter 与无标签 EpisodeStore 转换
├── models/
│   ├── central_mode.py          # 联合轨迹 VQ teacher
│   ├── local_context.py         # 严格局部 categorical student
│   ├── conditional_diffusion.py # mode FiLM 与 CFG adapter
│   └── value.py                 # 集中式 mode value head
├── experiments/central_mode.py  # P2 整轨迹训练、checkpoint 与审计协议
├── pipeline.py                  # 划分、标准化、训练与分散采样
├── evaluation/metrics.py
├── prototypes/xor_motivating_example.py
└── scripts/
    ├── generate_mpe_dataset.py
    ├── convert_omar_mpe.py
    ├── render_mpe_modes.py
    └── run_pipeline.py
```

| 入口或产物 | 用途 | 当前状态 |
| --- | --- | --- |
| `prototypes/xor_motivating_example.py` | 生成 XOR JSON/PNG | 可复现 P0 |
| `scripts/generate_mpe_dataset.py` | 生成受控 paired MPE | 可复现，默认拒绝覆盖 |
| `scripts/convert_omar_mpe.py` | 转换 OMAR 为无标签联合 episode | 可复现，collector 只用于分层与审计 |
| `scripts/render_mpe_modes.py` | 复现并可视化固定 seed 的 assignment mode | 可复现视频、轨迹图和数值摘要 |
| `scripts/run_pipeline.py` | 贯通 P2–P5 工程路径 | 可运行，但不是阶段隔离的正式实验 runner |
| `run_scripts/train_mode_consistent.py` | 原 MADiff 风格的阶段训练入口 | 当前只开放 P2 central |
| `run_scripts/evaluate_mode_consistent.py` | 加载冻结 checkpoint 并生成 evidence | 不自动给出研究结论 |
| `exp_specs/mode_consistent/*smoke.yaml` | P2 入口级 CPU/GPU smoke | 2 steps，不是正式预算 |
| `artifacts/omar_mode_audit.*` | 保存先前 OMAR 审计 | 静态快照；当前正式 evaluator 已另行实现无标签代理指标 |
| `prototypes/artifacts/xor_motivating_example.*` | 保存 motivating example | 有源码入口 |

`diffuser/models/diffusion.py` 只增加了 `model_kwargs` 透传、direct-action loss 初始化和完整 transition 采样维度，没有复制第二套 diffusion 实现。

## 8. 已清理的研究分支

此前 P2–P5 的部分结果依赖 MPE 专用手工 assignment 特征、终局几何 probe、容量模型或固定 mode-quality 映射。

它们无法证明严格 Dec-POMDP 下的完整算法，因此相关中间代码、测试和 artifacts 已从正式路径删除。

当前保留的测试是稳定模块 contract，不是一次性实验脚本。后续新增研究 runner 时，应在阶段结论冻结后删除重复 wrapper，只保留能复现论文表格或图的最小入口。

## 9. 下一阶段：P2 中央 mode discovery

当前 `run_pipeline.py` 继续只承担整链工程 smoke。P2 的独立训练与评估入口已经建立；在 P2 通过之前不扩大 diffusion 或 SMAC 预算。

P2 的最小任务为：

1. H=25 全轨迹单 code、VO-MASD VQ 权重、产物和标签防火墙已通过 contract test 与 GPU smoke；
2. 与用户确认是否接受 `1600 updates, batch 128`（约 1.46 train pass）作为 OMAR H=25 seed 0 首次正式预算；
3. 确认后只运行 VQ/no-code 和冻结 code-shuffle，生成 held-out 证据后停止并讨论；
4. 只有 hard/soft usage、reconstruction 改善和 assignment/collector 行为对齐同时合格，才讨论 P3 冻结 teacher 蒸馏；
5. P3 通过决策时刻的局部可辨识性阶段门后，再实现无 attention U-Net 的 RTG+mode 双条件 diffusion。
