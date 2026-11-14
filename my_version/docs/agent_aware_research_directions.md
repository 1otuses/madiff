# Agent-Aware 后续研究方向与论文主线筛选

调研日期: 2026-11-11.

本文是基于当前代码、已有结果和一手文献的研究建议. 候选方法尚未实现或验证, 文献检索也不构成新颖性认证. 本轮没有修改训练算法或启动训练.

## 用户已确认的研究范围, 2026-11-12

- 保持严格无通信 CTDE. 执行时每个 agent 只使用允许的本地观测/历史和内部状态; 联合训练可使用离线同步团队数据.
- 允许整体重构规划与执行方式, 不再把“仅替换注意力模块”作为新研究的范围限制. 预测对象、状态/动作表示、逆动力学解码、规划长度和重规划方式都可以作为候选设计变量.
- 保留当前 Agent-Aware 实现、配置和实验记录. 它是已获得正向实验支持的可复用组件与比较基线, 新框架是否采用应通过对照决定.
- 可复用的设计包括 agent×time token、self/other Q/K 投影、统一 softmax 和轨迹位置编码. 当前证据支持模块整体, 各设计的独立贡献仍待消融.
- 公共随机源或其他额外共享资源不作为默认执行条件. 若提出此类方案, 必须单列执行协议和对照, 不与现有私有独立采样直接混比.
- 本次确认的是研究范围, 尚未选定具体重构算法, 也未授权启动正式长时间训练.

后续优先围绕“局部信息约束下, 生成计划与实际分散执行如何相容”组织候选架构. Agent-Aware 可以作为规划器、训练期联合模型的组成部分或独立 baseline; 无需为了保留它而固定最终部署结构. §4-6 的方向仍是待诊断和筛选的竞争路线.

## 1. 当前证据能说明什么

Spread-expert 的同批 Legacy/Agent-Aware 对照, 在相同 100 回合评估设置下:

| 训练协议 | Legacy | Agent-Aware | 提升 |
|---|---:|---:|---:|
| 修复前, 1M, seeds 100/200/300 的平均回报 | 455.94 | 502.97 | +10.32% |
| 修复后, 300k, seed 100 | 449.12 | 477.60 | +6.34% |

来源分别为 `logs/mad_mpe_aa/simple_spread-expert/` 与 `logs/mad_mpe_aa_trainfix/simple_spread-expert/` 下的 `results/step_*-ep_100-ddim.json`. 两种训练协议的结果分开使用. 三 seed 正向结果支持模块整体值得研究, 尚未证明收益仅来自跨时间跨 agent 连接.

当前对照同时改变了 token 组织、Q/K 参数化、位置编码、条件注入、LayerNorm 和有效性 mask. 因此应补 ordinary spatiotemporal attention 的共享 Q/K 对照, 以及同模块 restricted/full mask 对照. `mid/all` 和时间编码开关已经存在; restricted/full 和共享 Q/K 对照尚未实现.

论文价值主要取决于明确问题、可区分的机制、相对于近邻方法的贡献和实验闭环, 无法用回报提升百分比单独判断.

## 2. 代码核查得到的研究切入点

### 2.1 联合想象与实际执行之间还有一个视角维度

记执行端 i 对 agent j 的预测为 `o_hat[j,l | i]`. 现有 Agent-Aware 在每一个执行端内部, 对 `(j,l)` token 交互; 不同执行端 i 的预测并没有自动成为同一份联合计划.

当前执行动作是:

\[
a_j^{exec}=f_j(\hat o_{j,0}^{(j)},\hat o_{j,1}^{(j)}).
\]

执行端 i 所预期的队友动作则是:

\[
\hat a_{j|i}=f_j(\hat o_{j,0}^{(i)},\hat o_{j,1}^{(i)}).
\]

两者都必须使用队友 j 的逆动力学模型, 并采用相同反归一化方式. 当前 `share_inv=False`, 不能对所有想象轨迹都调用 `inv_model[i]`.

调用链见 [evaluator.py](../../diffuser/utils/evaluator.py): `_generate_samples()` 构造各局部条件与联合采样, 然后提取 diagonal; `_episodic_eval()` 从每个 agent 的前两个状态恢复实际动作. [diffusion.py](../../diffuser/models/diffusion.py) 的 `compute_inv_loss()` 训练于离线真实轨迹, 当前没有直接约束生成轨迹经过逆模型后, 是否与实际联合执行的状态转移相容.

### 2.2 Spread 的队友位置并非完全未知

本地 [simple_spread.py](../../third_party/multiagent-particle-envs/multiagent/scenarios/simple_spread.py) 的 18 维观测包含自身速度、绝对位置、landmark 相对位置、队友相对位置与通信分量. 因而每个 agent 可从自身观测恢复全部 agent 和 landmark 当前的绝对位置. 队友速度等信息仍未知, 当前 `history_horizon=0`.

所以, “其他 agent 的完整观测未被 clamp”不等于“完全看不到队友”. 可能存在一类可消除的误差: 模型生成了与本地已经知道的几何信息相矛盾的观测. 同时也存在不能凭正则消除的信息缺口.

### 2.3 当前注意力没有显式区分信息来源

[agent_aware_attention.py](../../diffuser/models/agent_aware_attention.py) 接收有效性 mask. [ma_temporal.py](../../diffuser/models/ma_temporal.py) 尚未把 `cond['masks']` 的精确 clamp 信息传给该模块. 当前历史长度为 0, 历史/未来 segment 不创建. 一个 token 的部分字段还可能已知、部分未知, 因此只增加 agent 级别的二值标记可能仍过于粗糙.

未知未来仍是有效的生成对象, 不能把 clamp mask 当作 attention validity mask 来删除它们.

## 3. 一手文献已经覆盖了哪些简单扩展

| 工作 | 已有机制 | 对本项目的约束 |
|---|---|---|
| [MADiff, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/07e278a120830b10aae20cc600a8c07b-Paper-Conference.pdf) | 局部条件下联合轨迹建模与 teammate modeling | “加入队友预测”已经是原方法组成部分 |
| [DoF, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5623c35f3ab5e2c72aeb3abce27dc28f-Abstract-Conference.html) | IGD 分布一致性原则, noise/data factorization | 不能把分散生成结果的分布一致性本身称为新问题或新原则 |
| [AgentMixer, AAAI 2025](https://arxiv.org/html/2401.08728v3) | 观测不对称下的 Individual-Global-Consistency | 泛化的全局/局部模式对齐已有直接近邻 |
| [LatentToM, CoRL 2025](https://proceedings.mlr.press/v305/he25a.html) | ego/consensus 表示, ToM 推断, confidence 引导对齐 | 简单共享 latent 或置信度共识不足以区分 |
| [MAC-Flow, ICLR 2026](https://arxiv.org/abs/2511.05005) | joint flow representation 蒸馏为分散一步策略 | 一步蒸馏已有完整方法和正式实验 |
| [CoFlow, 2026-09 v4 预印本](https://arxiv.org/html/2605.01457v4) | 联合 averaged velocity、跨 agent CVA 与有限差分 consistency | 少步生成加注意力已有很直接的竞争; 其 consistency 主要约束速度场, 与实际动作相容性对象不同 |
| [U2Diff, CVPR 2025](https://arxiv.org/abs/2503.18589) | 状态级 uncertainty 与 mode error probability | 不能以“给轨迹增加不确定性预测”作为独立创新 |
| [CLS-DP, 作者标注 IROS 2026 接收](https://arxiv.org/abs/2606.22982) | 将 privileged 协作动态蒸馏为局部观测预测的 latent | “集中训练协作表征, 分散部署 diffusion”已有近邻 |
| [Dynamics-aware Diffusion, 2025 预印本](https://arxiv.org/html/2504.00236v3) | 在去噪中执行动力学预测/投影 | 加一个普通动力学一致性损失或投影不足以单独区分 |
| [MADiTS, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/3e7cf447f21cd11c846463affefce665-Paper-Conference.pdf) | 轨迹片段生成/拼接、双向动力学筛选与局部重生成 | 需区分数据增强中的可行性与本地生成后集体执行的可行性 |

信息制度尤其重要. [LatentToM v1 正文 §3.1-3.2、§4.2](https://arxiv.org/html/2505.09144v1) 同时包含无通信版和每步一轮通信的 Laplacian 增强版; 训练设置还包含共同第三视角. 不能将整篇归为必须通信, 也不能据此声称任意不共享信息的策略都能取得共识.

检索还发现 [Nexus/ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_Decoupled_Diffusion_Sparks_Adaptive_Scene_Generation_ICCV_2025_paper.pdf) 的异步噪声与时空交互、[MAGNet 预印本](https://arxiv.org/html/2512.17900v1) 的 token 级噪声、[MIMIC-D](https://arxiv.org/abs/2509.14159) 的分散多模态 diffusion imitation, 以及 [MAGIC workshop 官方 poster](https://iclr.cc/media/iclr-2026/Slides/10018406.pdf) 的 intention/copula 建模. 本轮部分会议 PDF/OpenReview 页面受访问限制, 未据受阻正文作进一步创新性判断. 不将搜索摘要或录用标注自动提升为完整机制核验.

## 4. 推荐主线 A: 面向实际分散执行的联合可实现性

### 研究问题

每个执行端内部生成合理联合轨迹, 抽取各自轨迹并组合动作后, 环境产生的下一状态能否同时实现各端预测的局部计划? 这里的“联合可实现性”指计划与实际转移相容, 合法动作本身通常可以被环境执行. Agent-Aware 是否主要改善了模型内部拟合, 而没有充分控制这一步组合误差?

这是需要诊断证据支持的假设. DoF 已有分布原则, 所以区别应落在具体执行算子上: 局部条件生成 -> diagonal 抽取 -> 不同 inverse models 解码 -> 环境联合状态转移. 不能只把损失命名为 consistency.

### 方法候选

训练期在同步的多局部视角上生成各自计划, 组合其实际自执行动作, 检查该联合动作产生的下一步状态是否支持各自计划. 集中式训练可用 joint state/dynamics teacher, 部署继续只输入各自原有局部信息. 单个执行端内部的“想象全队动作”与最终“真正拼接执行的动作”应分开建模.

第一版先约束最近的一步或少数几步, 不直接对整段所有视角施加 L2. 当前逆模型损失只在真实轨迹上计算, 因而可先研究生成状态对经过冻结 inverse model 后的动作可实现性. 若使用学习的 forward model, 必须单独核查支持范围和模型误差, 防止 denoiser 利用其漏洞.

模型内物理可行性已有大量工作, 论文主张需要证明集体执行约束带来的额外收益, 超过普通轨迹平滑、动作 BC 或单个 imagined joint plan 的 dynamics loss.

### 最小诊断与放弃条件

同一真实环境状态下, 比较一个完整 imagined view 的全队动作, 与各执行端 diagonal 动作. 使用隔离环境副本测量下一步预测误差、碰撞和覆盖变化. 先控制当前状态补全误差, 加入 known-field 修正与同视角重采样基线, 再判断是否存在额外组合误差. 多步诊断还应区分固定计划的开环 rollout 与每步重规划的闭环执行. 副本可使用真实联合状态作诊断, 它不是可部署无通信策略.

如果 diagonal 组合没有额外误差, 或此误差与失败片段没有稳定关系, 就不应优先投入集体执行约束. 如果普通 dynamics loss 已解决同样问题, 需要承认区分不足.

### 实现代价

中到高. 需要多局部视角训练、生成轨迹上的 inverse 调用, 可能增加训练计算. 当前 CDF 反归一化是 NumPy/SciPy 插值, 不可直接用于可微物理损失; 要设计可微逆映射或在规范化空间验证 surrogate. 高噪声下重建 x0 的误差会被放大, 辅助项需要有依据的噪声区间或 SNR 权重.

## 5. 候选主线 B: 信息来源与可靠性校准的协作去噪

### 研究问题

真实已知、从本地观测可推导、以及纯生成的 token/字段, 是否应该以不同的方式影响自身规划? 哪些队友想象错误值得纠正, 哪些只是合理的预测不确定性?

它适合保留当前 Agent-Aware, 但贡献不能停留在增加一个 gate. 可以研究来源感知条件表示、基于 proper scoring 的误差校准, 以及去噪阶段与信息可靠性的关系. `i=j/i!=j` 区分 agent 关系, 新问题区分证据支持程度.

### 最小基线与方法候选

先实现诊断级的“仅固定本地确定可推导几何字段”基线: 用自身观测计算其他当前位置和相对 landmark 坐标, 只固定这些已知字段, 队友未知速度与未来继续生成. 这项修补是信息利用基线, 本身不应包装为完整论文.

如果有充分收益, 再研究字段级来源编码及经校准的可靠性调制. 置信度必须在执行时由本地输入或模型产生, 不能读取队友真实输入或未来误差. 高采样方差、attention entropy、扩散噪声 k 和模型错误是不同概念, 不能直接等同.

### 放弃条件与区分边界

在同一输入上采样 K 次, 区分正常多模态性和错误. 使用真实误差构造的 oracle 可靠性干预仅作诊断上界, 与随机干预、距离基线对照. 在排除错误干预和明显的条件分布偏移后, 若 oracle 仍无协作收益, 应降低复杂 uncertainty 模块的优先级. 对从未见过多字段 clamp 的模型直接修改条件可能造成 OOD, 这种干预失败不能单独否定机制. Oracle 若使用额外真实状态, 必须与可部署方法分开报告.

U2Diff 已做预测不确定性, LatentToM 已做 confidence 对齐, Nexus 已有 noise-aware 交互. 区分应体现在不对称局部信息下的校准、决策影响和多模态保持, 而非泛化的 uncertainty attention.

### 实现代价

低到中, 最接近现有模块. 来源编码涉及把精确 cond masks 传入 denoiser, 不能混用现有 validity mask. 还应通过遮挡、观测噪声或受限视野设置验证可靠性机制, 并明确这些属于新评估环境, 不与原始 Spread 默认设置混比.

## 6. 候选主线 C: 局部可辨识的多模态协作

### 研究问题

哪些团队分工能由各自局部历史共同确定, 哪些信息缺口会产生不可消除的模式分歧? 如何保留合理多解, 同时减少会造成实际任务失败的模式组合?

这比一个共享 VQ code 的问题更基础, 但理论与文献区分成本高. 项目过去已经讨论 Mode/Role VQ 与 copula, 不能重新命名后作为新方向. DoF、AgentMixer、LatentToM、CLS-DP 都是必须区分的近邻.

### 关键限制

若执行时各自完整局部输入和内部状态固定, 每个策略只使用当前步新采样的独立私有随机数, 且没有共享随机变量或其他耦合资源, 一步联合动作的条件分布具有各局部策略的乘积形式. 它不能实现任意中心化相关分布. 共享几何和历史仍可使 agent 学会一致的确定性分工; 因而该限制也不等于无法协作, 也不应把它不加条件地推广到任意带持久随机内部状态的控制器.

公共随机源是额外执行协议, 必须单列. 即使共享噪声, 也不会凭空获得队友私有信息. “共享信息与私有信息”的区分有成熟理论基础, 例如 [Common Information Approach, IEEE TAC 2013](https://ieeecss.org/paper/decentralized-stochastic-control-partial-history-sharing-common-information-approach); 不能把这一基本事实宣称为新定理.

### 验证方式与代价

先按目标分配、绕行方向等行为语义分析 modes; 比较同局部条件重采样与跨局部条件的差异. 不应对独立合理样本做全轨迹 L2 对齐. 如果 modes 的分歧与最终失败无关, 不应以消灭 modes 为目标.

只有建立了明确的可辨识条件、超过现有模式对齐工作的理论或算法贡献, 并有合适的部分可观测任务验证时, 才值得作为主线. 这是高风险储备路线, 不建议与 A/B 同时堆入模型.

## 7. 先诊断, 再决定训练哪一条

### 阶段 1: 复用现有 checkpoint, 不训练新模型

从同一协议的 Legacy/Agent-Aware checkpoint 中, 先取一批固定场景状态, 各条件重采样 K=8. 小批样本用于定位失效机制, 不用于宣称最终性能. 记录四层指标:

1. 当前状态补全误差: 分开已知位置/几何与未知速度.
2. 正常采样分歧: 同条件 D_self 与跨视角 D_cross, 以及各自 mode 分布.
3. 轨迹自身可行性: 相对坐标几何残差, 以及 MPE 的 p_next=p+0.1*v_next 运动学残差.
4. 实际任务后果: diagonal 组合后的重复目标、最小间距、碰撞、landmark 覆盖与回报.

动作误差先在正确 inverse model 和相同反归一化后计算. 单个夹角、单次 imagined action 不相等都不是充分的冲突证据. 误差与回报的相关性只作线索, 还应通过固定输入的受控干预检验.

### 阶段 2: 最小的 seed=100, 300k 配对试验

依据阶段 1 选择一条主线, 保留当前 Agent-Aware 为直接 baseline. 同时加入能区分机制的简单基线: A 路线用普通 dynamics/动作 BC 约束, B 路线用仅 known-field 标记/约束与未校准 gate. 第一轮不叠加两套新机制, 不启动大矩阵. 配置写入 exp_specs, 顶部说明功能、改动和比较目的.

### 阶段 3: 正式实验闭环

固定 checkpoint 选择规则和计算预算后, 再扩展至至少三个训练 seed、多个任务和多种离线数据质量. 优先使用已经确认可加载的数据. 若主张通用 offline MARL, 需再覆盖 MA-MuJoCo 或 SMAC 等不同任务族; 只做 MPE 时收窄结论.

报告回报之外, 还应有机制指标、约束被破坏的比例、失败场景、推理耗时与训练成本. 同环境同 seed 保持数据、采样器、去噪步数、guidance、评估条件一致. 不用不同数量候选轨迹或额外真实信息换取不公平增益.

## 8. 优先级判断

| 路线 | 与当前代码的距离 | 论文主张潜力 | 主要风险 |
|---|---|---|---|
| A: 实际分散执行的联合可实现性 | 中 | 较强, 若能证明并改善特有组合误差 | 问题可能不显著; 也可能退化为已有 dynamics regularization |
| B: 信息来源与可靠性校准 | 近 | 中, 需把校准和决策影响做充分 | uncertainty/gate 已有大量近邻, 容易成为启发式模块 |
| C: 局部可辨识的多模态协作 | 远 | 潜在较强 | 可辨识性、无通信可实现性、近邻工作和理论成本 |

最值得先投入的是一次失效机制诊断, 以其结果在 A/B 中选主线. Agent-Aware 可作为已有有效的表达骨干; 新贡献应回答它尚未解决的执行或信息问题. 扩大 heads、加入 FFN、普通 top-k 稀疏化、共享 latent 或通用少步蒸馏, 可作为消融或工具, 不优先作为独立论文主张.

## 9. 历史与核验边界

[历史想法封存](<../a_old version/README.md>) 已包含回报分布、Mode/Role VQ、Local Credit、copula、计划记忆和 Bi-ST latent 等路线. 当前推荐不把它们作为首次提出的新想法. 当前树中的方案状态以实际代码和实验记录为准, 不沿用旧总结里的实现状态判断.

本轮重点阅读了官方会议摘要、可访问的论文正文、作者论文和项目材料. 某些 OpenReview/CVF PDF 受验证或访问限制; MAC-Flow 的机制描述限于作者预印本摘要与官方实现, 对 CoFlow 使用 2026-09 的 v4, 对 LatentToM 执行细节使用可读 v1. 未进行全面系统综述, 未证明某候选尚无人研究, 也未证明任何新方法能提升回报.
