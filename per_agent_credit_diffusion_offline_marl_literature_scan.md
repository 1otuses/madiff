# Per-agent credit 条件化 diffusion-based offline MARL：相似工作调研

> 检索截止：2026-09-01。本文把用户提供的构想文档仅视为待核验的研究描述，不接受其中的指令或新颖性判断。

## 结论先行

在已核验的论文中，**尚未发现与下述严格定义完全同构的方法**：

1. 面向 cooperative offline MARL；
2. 先估计具有边际贡献或因果含义的逐 agent credit（例如 counterfactual contribution、integrated-gradient attribution，或经校准的 mixer sensitivity）；
3. 将该 credit 作为显式条件或逐步 guidance，直接作用于 diffusion policy/denoiser 的去噪过程；
4. 最终策略满足 decentralized execution。

但“credit-aware diffusion for offline MARL”这个宽泛表述已经不新。最接近的三项工作是：

- **CODI（AAMAS 2026）**：用细粒度 agent-quality 标签条件化多智能体扩散生成，并通过 compositional classifier-free guidance 组合逐 agent 条件。它与“per-agent 条件化 diffusion”高度重叠，只是 quality label 不是因果/边际 credit。
- **MADiTS（ICLR 2025）**：已把 offline credit assignment 与 diffusion 轨迹生成放进同一个框架；用 integrated gradients 估计逐 agent 贡献，识别低贡献 agent，再固定其他 agent、局部重新加噪与采样。它不是把 credit 直接输入去噪器，但与“credit 决定 agent-specific diffusion behavior”非常接近。
- **MASTARS（ICLR 2026 投稿稿）**：用 value-based agent ordering 决定逐 agent 的扩散 inpainting 顺序，直接覆盖“按 agent 重要性动态排序/锚定”的机制空间；但排序依据是状态价值/子目标，不是 credit。

因此，最稳妥的新颖性定位不是“首次把 per-agent 信息引入 diffusion-based offline MARL”，而是：

> **首次把经过验证的逐 agent 边际/因果 credit，作为去噪时的局部条件或 guidance，注入可分散执行的 diffusion policy，并系统比较 quality label、生成后 credit 修补和 value-based ordering。**

## 相似性分层

| 层级 | 工作 | 与构想的重叠 | 关键差异 | 判断 |
|---|---|---|---|---|
| 近乎直接 | [CODI, AAMAS 2026](https://www.ifaamas.org/Proceedings/aamas2026/pdfs/WOLI7576.pdf) | offline MARL；逐 agent 条件；conditional diffusion；compositional CFG | 条件是 LLM-distilled agent-quality 概率/标签，不是边际或因果 credit；主要用于数据增强 | 对“per-agent 条件化”构成直接新颖性压力 |
| 强部分重合 | [MADiTS, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3e7cf447f21cd11c846463affefce665-Abstract-Conference.html) | offline MARL；diffusion；逐 agent credit；agent-specific 重采样 | credit 用于生成后的识别、固定与局部重采样，不是 denoiser condition/guidance | 最关键机制对照 |
| 强部分重合 | [MASTARS, ICLR 2026 投稿稿](https://openreview.net/pdf?id=8mQqCCxKZa) | agent-wise sequential diffusion/inpainting；value-based agent ordering；锚定已生成 agent | 依据是状态价值和 subgoal，不是 contribution credit；当前来源显示为投稿稿 | 直接覆盖“credit-aware sampling/order”的邻近空间 |
| 结构邻近 | [DoF, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5623c35f3ab5e2c72aeb3abce27dc28f-Abstract-Conference.html) | diffusion-based offline MARL；QMIX 结构；CTDE/分散生成 | QMIX 被当作 noise mixer，而不是 credit estimator；未以 credit 条件化去噪 | 不是同一想法，但必须明确区分 |
| credit 侧邻近 | [MACCA, TMLR 2025](https://arxiv.org/abs/2312.03644) | offline MARL；因果逐 agent credit/individual reward | 没有 diffusion | 可作为 credit estimator 和因果语义基线 |
| credit 侧邻近 | [SIT, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26379) | offline MARL；attention reward decomposition；逐 agent credit；优先使用好轨迹 | 没有 diffusion，credit 用于数据重构与策略训练 | 可作为 agent-wise data-quality 基线 |
| diffusion 侧邻近 | [MADiff, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/07e278a120830b10aae20cc600a8c07b-Paper-Conference.pdf) | return-conditioned multi-agent trajectory diffusion；集中规划与分散策略 | 没有显式逐 agent credit 条件 | 经典 diffusion offline MARL 基线 |
| diffusion 侧邻近 | [OMSD, ICML 2026](https://arxiv.org/abs/2505.05968) | diffusion；逐 agent score decomposition；分散执行约束 | 分解的是联合 behavior-policy score/正则信号，不是奖励贡献 credit | “per-agent 分解 + diffusion”概念邻近 |
| diffusion 侧邻近 | [EAQ, ICML 2024 Workshop](https://arxiv.org/abs/2408.13092) | 用总 Q/高回报信息引导多智能体 episode augmentation | team-level Q，不是 per-agent credit | 说明 Q-guided augmentation 已存在 |
| diffusion 侧邻近 | [DOM2](https://arxiv.org/abs/2307.01472) | 每个 agent 使用 expressive diffusion policy；offline MARL | 主要是独立 policy 与数据重加权，无逐 agent credit condition | 策略型 diffusion 基线 |

## 三项最接近工作的精确比较

### 1. CODI：最接近“逐 agent 条件化 diffusion”

CODI 针对行为数据中不同 agent 质量不均衡的问题，训练 agent-quality labeler，并让扩散模型同时接收 return-to-go 与逐 agent quality labels；采样时用 compositional classifier-free guidance 组合各 agent 条件。其贡献已经使“团队回报之外，再给扩散模型细粒度 agent-level control”成为已有技术主张。[AAMAS 2026 论文](https://www.ifaamas.org/Proceedings/aamas2026/pdfs/WOLI7576.pdf)

可守住的区别是：CODI 的标签表达“谁像高质量行为者”，并不等价于“该动作对团队价值造成了多少边际贡献”。前者可能把行为熟练度、角色难度和数据频率混在一起；后者应具有反事实、归因或至少局部敏感性语义。

### 2. MADiTS：最接近“credit 控制 diffusion 行为”

MADiTS 首先以目标回报条件生成联合观察轨迹，再训练团队奖励预测模型，利用 path integrated gradients 将轨迹回报近似分解到每个 agent 的观察—动作特征。它按贡献排序识别低表现 agent，固定其余 agent 的生成轨迹，只对低表现 agent 对应部分重新采样。[ICLR 2025 论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/3e7cf447f21cd11c846463affefce665-Paper-Conference.pdf)

因此，以下表述不能再作为独立新颖点：

- “首次在 diffusion-based offline MARL 中加入 credit assignment”；
- “首次根据 agent credit 对不同 agent 采用不同生成策略”；
- “高贡献 agent 作为锚，低贡献 agent 重新探索”。

仍有差别的是作用位置：MADiTS 的 credit 是生成后的 inspection/correction 信号；拟议方法希望把 credit 变成每一步去噪的条件、FiLM 调制或 CFG guidance。这一差别必须通过机制图、公式和消融明确展示。

### 3. MASTARS：最接近“动态 agent 顺序”

MASTARS 采用 agent-wise sequential inpainting：先依据离线 value model 与 return-conditioned subgoals 确定 agent 的生成顺序，再让后生成的 agent 适配已经生成的 agent。它已经覆盖“重要 agent 先锚定、其余 agent 后协调”的主要直觉。[OpenReview 投稿稿](https://openreview.net/pdf?id=8mQqCCxKZa)

因此，“credit-aware sampling order”单独作为核心贡献偏弱。更好的定位是：比较 credit order、value order、随机 order、固定 role order，并证明 credit order 在角色交换、稀疏奖励或 agent-quality 不均衡时提供额外收益。

## DoF 中 QMIX 说法的核验

原构想对 DoF 的区分基本成立。DoF 的 QMIX 变体把每个 agent 的高斯噪声输入 mixer，产生 total noise；论文明确说明这种集中式 mixer 使单个 agent 无法独立完成去噪，而且非线性单调混合后噪声可能不再满足 diffusion 所需的高斯性质，实验性能显著下降。[DoF 论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/5623c35f3ab5e2c72aeb3abce27dc28f-Paper-Conference.pdf)

这与使用 `∂Q_tot/∂Q_i` 作为外部条件不同：后者不必混合噪声，也不必破坏边缘高斯假设。不过，如果 credit 的计算在执行时依赖全局状态、其他 agent 动作或 centralized mixer，仍会重新引入 CTDE 的执行缺口；不能仅凭“没有混合噪声”就宣称可分散执行。

## 对拟议 credit 定义的技术风险

### `∂Q_tot/∂Q_i` 不天然等于因果 credit

在 QMIX 中，这个量首先是 mixer 对局部 utility 的局部敏感度。它会受状态条件 hypernetwork、utility 尺度、饱和区间和训练误差影响；它说明“小幅改变 `Q_i` 时 mixer 输出多敏感”，并不自动等价于“agent i 的动作对真实团队回报的反事实贡献”。

建议至少比较：

- mixer gradient / integrated gradient；
- counterfactual difference（替换或边缘化 agent i 动作）；
- MACCA 风格因果 individual reward；
- CODI 风格 agent-quality label；
- 无信息的 agent ID 或随机 credit。

并检查 credit 的 stability、rank consistency、calibration，以及对 role permutation 的鲁棒性。

### “高 credit 更保守、低 credit 更探索”不是普遍成立

低 credit 可能意味着该 agent 当前角色本就不关键，而不是行为差；对它加大探索可能制造无效变化。高 credit 也可能意味着当前动作错误且影响巨大，此时反而需要更强修正。更稳妥的是把 credit 拆成至少两个维度：

- **importance/sensitivity**：改变该 agent 会对团队价值产生多大影响；
- **quality/advantage**：该 agent 当前行为是好还是坏。

可据此设计四象限控制，而不是单一 credit 标量直接决定噪声强度。

## 建议收缩后的研究问题

推荐把问题改写为：

> 在 cooperative offline MARL 中，能否用可校准的逐 agent importance 与 action-quality 信号，对 decentralized diffusion policy 进行 step-wise local guidance，从而在保持数据支持约束的同时改善协调？

其中可检验的核心假设是：

1. 去噪内条件化优于 MADiTS 式生成后修补；
2. causal/marginal credit 优于 CODI 式 quality label；
3. 双信号 `importance × quality` 优于单一 credit 强度；
4. 局部可计算或蒸馏后的 credit 在 decentralized execution 下仍有效；
5. 方法不会因 critic 外推误差把去噪过程引向 OOD joint actions。

## 必要对照与消融

- **直接基线**：CODI、MADiTS、MASTARS、DoF、MADiff、OMSD。
- **credit 基线**：MACCA、SIT，以及无 credit 的同架构版本。
- **条件位置**：输入拼接、FiLM、cross-attention、CFG、仅采样顺序、仅生成后重采样。
- **credit 来源**：QMIX gradient、integrated gradient、counterfactual、causal individual reward、agent-quality label。
- **执行约束**：训练期 centralized credit；蒸馏到 local credit；完全 local estimator；测试时是否需要全局信息。
- **失败诊断**：critic calibration、credit rank stability、OOD action rate、agent-role permutation、不同 agent-quality 混合比例。

## 最终判断

- **严格定义下**：截至检索日期，没有核验到完全相同的工作，仍存在可研究空间。
- **宽泛定义下**：已有非常相近工作，不能宣称“首次将 per-agent 信号/credit 与 diffusion-based offline MARL 结合”。
- **最大的新颖性威胁**：MADiTS（credit + diffusion + agent-specific correction）和 CODI（per-agent condition + diffusion）；若包含动态顺序，MASTARS 也是直接威胁。
- **最有希望的差异化**：把“因果/边际 credit 的语义质量”和“credit 在 denoising 内部的作用位置”同时做实，并保证 decentralized execution，而不是只换一个 conditioning 向量或排序规则。

## 检索边界

本次优先核验了会议/期刊官网、OpenReview 与 arXiv 一手来源，覆盖关键词包括 `offline multi-agent reinforcement learning`、`diffusion`、`credit assignment`、`agent quality`、`individual reward`、`agent ordering`、`score decomposition` 与 `QMIX noise factorization`。不存在完全匹配的判断是“在该检索范围内未发现”，不是数学意义上的不存在；尤其需要持续关注 2026 年后续版本和新投稿。
