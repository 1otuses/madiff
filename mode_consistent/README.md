# CI-CoDiff：分散 Diffusion MARL 的协调模态一致性

> 更新日期：2026-08-21。本文档只保留稳定的研究问题、相关工作、算法框架、benchmark 与复现入口。动态阶段状态、实验规划、脚本和证据审计见 [`RESEARCH_PROGRESS.md`](RESEARCH_PROGRESS.md)。

算法接口、模块定义、benchmark 或复现命令变化时更新本文档。实验状态、研究决策、脚本或产物变化时更新研究进程文档。

一次修改同时影响两类信息时，必须同步更新两份文件。

当前代码提供可运行的工程框架，但尚未证明 mode 语义、局部可辨识性或最终 return 提升。

## 1. 研究问题

离线多智能体数据可能混合多个高回报协调模态，例如角色分配、目标顺序或阵型：

```text
mu(a_k | s_k) = sum_z p(z | s_k) prod_i mu_i(a_i | h_i^k, z)
```

若各 agent 只拟合自己的边缘多模态分布并独立采样，就可能分别选择不同的 `z_i`。此时每个局部动作都来自数据，组合后的联合动作却不属于任何已见协调策略。

当 `K` 个模态等概率、`N` 个 agent 独立选取模态时：

```text
Pr(z_1 = ... = z_N) = K^(1-N)
MMR = 1 - K^(1-N)
```

本项目在 CTDE 框架下分别研究三个问题：

1. **模态发现**：能否从无标签联合离线轨迹中学习代表协调策略的离散隐藏变量 `z`；
2. **分散对齐**：执行时每个 agent 能否仅由局部历史 `h_i^t` 恢复一致的 mode；
3. **策略改进**：mode 条件 diffusion 是否能在保持数据支持和低失配率的同时提高任务回报。

严格 Dec-POMDP 下存在信息边界：若相同局部历史对应多个 mode，且没有公共观测、共享随机性、通信或可观察行为前缀，任何局部模型都无法恢复采集时的 `z`。

此时只能学习确定性公共约定、拒绝预测，或显式声明额外 correlation device。

## 2. 相关工作

| 方法 | 核心机制 | 本项目的借鉴与区别 |
| --- | --- | --- |
| MADiff | 跨 agent attention；每个 agent 预测联合轨迹并执行自己的动作分量 | 作为 diffusion policy 主干；额外研究独立反向采样时的 mode 一致性 |
| CPA | 自回归联合策略、VQ action pool、共享 agreement key | 借鉴 policy mismatch 问题与轨迹熵指标；共享 key 属于额外 correlation device |
| VO-MASD | 对每个 agent 的 `H` 步轨迹编码；用全局状态动态分组，并以组大小相关的 codebook 量化 multi-agent skill | 借鉴时间片段编码、VQ、动作解码和“预训练后冻结”；本项目学习全队共享 coordination mode，不照搬 3D subgroup、全局 grouper 或在线 MAPPO |
| CLS-DP Contextualizer | 用未来联合动力学 posterior 对齐局部 observation prior | 借鉴 privileged teacher 到 local student；改为共享离散 team code，并评估 agreement 与 unknown |
| DoF / OMSD | diffusion 分解、链式行为策略与条件 score 正则 | 可改善分布建模，但分解本身不保证多个 agent 在执行前选择同一协调模态 |
| MIMIC-D / CoDiMAD | joint supervision 与协作 latent | 属于最近邻工作；本项目重点区分离散 mode 的可识别性与严格信息权限 |

本项目的定位是：**相关协调模态的分散可实现性、诊断协议与 mode 条件 diffusion**。

### 2.1 VO-MASD 源码比较后的设计选择

VO-MASD 与本项目都使用轨迹编码、离散量化和行为解码，但隐藏变量的语义与执行路径不同：

| 组件 | VO-MASD | CI-CoDiff 的选择 |
| --- | --- | --- |
| 时间编码 | 每个 agent 的 `H` 步 observation/action 经实体注意力和双向 GRU 得到连续 skill | 保留 `H` 步原始轨迹输入；当前共享 GRU 编码每个 agent，再聚合为一个团队表示 |
| agent 结构 | 全局 state 驱动 MAT grouper；按动态 subgroup 量化 | 第一阶段将所有 agent 视为一个固定团队，不增加 3D subgroup 层 |
| codebook | 每个组大小 `m` 使用一个 `K × (mD)` codebook，量化后再拆成 agent-specific skill | 使用单个 `K × D` 团队 codebook；所有 agent 共享 code ID，再由 agent identity 解码各自行为 |
| 解码目标 | recurrent decoder 在 skill 条件下重建 `H` 步离散动作概率 | 用共享 mode、局部 observation 和 agent identity 重建 `H` 步连续动作均值 |
| 训练目标 | 动作负对数似然、VQ 距离；grouper 另以 PPO 优化重建/量化内在回报 | 动作 MSE 与 VO-MASD 权重方向的 VQ 损失；encoder commitment 权重为 1，codebook 更新权重 `beta=0.001` |
| 下游使用 | 冻结 skill 模块，由高层 MAPPO 周期性选择 skill | 冻结中央匿名 code，蒸馏到局部 aligner，再作为离线 diffusion 条件 |

VO-MASD 的 VQ 目标保证 code 有助于动作重建，但不保证 code 等价于协调 convention。当前 P2 因此将整条 `H=25` episode 绑定为一个 code，并把 assignment 对齐、code-shuffle 和 no-code 对照作为独立阶段门；重建误差下降本身不构成 mode discovery 成功。

## 3. 核心算法 pipeline

```text
无标签联合离线轨迹 D_train
        |
        v
1. CentralModeVQVAE q_psi(tau_joint[0:H]) -> 离散团队 mode z_episode
   per-agent 时间编码 -> 固定团队聚合 -> K×D codebook
   H=25 整条 episode 只产生一个共享 code ID
        | 冻结匿名 teacher code
        v
2. LocalModeAligner g_i(h_i^t) -> c_i / unknown
   每个 agent 只读取自身 observation 与过去动作
        | 局部 code 通过可辨识性阶段门
        v
3. 无 attention Temporal U-Net + Diffusion
   以离散 mode 和 team return-to-go 为双条件做 classifier-free guidance
        | 每个 focal agent 只执行自己的动作分量
        v
4. ModeValueModel V(s,z) 与集中 evaluator
   拟合数据内回报，并评估 MMR、coverage、support 与 return
        |
        v
严格分散执行
```

训练接口通过 `UnlabeledEpisodeView` 建立标签防火墙。表示学习模块只能接收 `observations`、`actions` 和 `mask`；`true_modes`、collector、quality、scenario 与奖励只保留在数据划分和集中评估层。

## 4. 模块说明

### 4.1 中央离散 mode 教师

[`CentralModeVQVAE`](models/central_mode.py) 先用共享 GRU 分别编码长度为 `H=25` 的各 agent observation/action 轨迹，再结合 agent identity 聚合为一个团队表示。完整 episode 只映射到一个共享团队 VQ code；共享的 per-agent decoder 接收当前局部观测、团队 code 和 agent identity，输出连续动作均值。

令 encoder 输出为 `z_e`、最近码本向量为 `e_z`，损失为：

```text
L = L_action + ||z_e - sg(e_z)||^2 + beta ||e_z - sg(z_e)||^2
beta = 0.001
```

Decoder 使用固定方差高斯解释时，动作 MSE 与负对数似然只差常数和尺度。受控数据中的 assignment 标签只用于训练结束后的 NMI、ARI 和 best-mapping accuracy，不进入模型输入或损失。

这里有意采用二维 `K × D` 团队 codebook，而不是 VO-MASD 按 subgroup 大小建立的三维结构。固定 agent identity 由 decoder 单独接收，使同一 mode ID 可以生成不同 agent 的兼容角色，同时避免为各 agent 独立量化后再次产生 mode mismatch。若未来研究动态 subgroup，必须作为独立扩展验证，不能混入当前 P2 的修正。

VO-MASD 的双向 GRU、recurrent action decoder 或更复杂 codebook 都可能提高重建容量，却不会自动解决当前 code 编码运动阶段而非 assignment 的问题。因此在中央 mode 语义通过前，不以增加这些容量模块作为默认修正。

P2 的正式训练和审计协议位于 [`experiments/central_mode.py`](experiments/central_mode.py)。它先冻结 train/validation/test，再强制 `trajectory_scope=full_episode`、`window_horizon=数据 horizon`和 `stride=1`；保存数据、split 和轨迹协议，并确保训练 checkpoint 不包含审计标签。旧 H=5 checkpoint 仅以显式 `standard` VQ 兼容语义读取，不与新证据混用。

### 4.2 局部 mode 对齐器

[`LocalModeAligner`](models/local_context.py) 为所有 agent 共享同一个 GRU。每个 agent 只输入自己的 observation、上一时刻自身 action 和本地可知的 agent identity，不读取其他 agent 的真实历史。

模型以中央匿名 code 为蒸馏目标，同时使用跨 agent agreement loss。执行时，低于置信度阈值的预测记为 `unknown=-1` 并进入无条件 diffusion 分支。

VO-MASD 的全局 grouper 属于集中训练中的 skill discovery 组件，不能替代该局部 aligner。执行期若使用 global state 重新分组或广播 group/code，就不再是本项目要求的严格 decentralized execution。

### 4.3 mode 条件 MADiff

目标策略模块沿用 MADiff 的时序 U-Net 尺度结构，但使用基础无 attention 版本，并以离散 mode 与团队 return-to-go 作为双条件 CFG。VO-MASD 的 recurrent decoder 只用于中央表示学习约束，不是最终策略。

仓内现有 [`ModeConditionedDenoiser`](models/conditional_diffusion.py) 是旧 v0 的 FiLM/CFG 工程 seam；它仍使用 attention 主干，且没有 RTG 双条件，因此不是上述最新算法的正式实现。在 P2 证据讨论前不继续扩写该模块。

### 4.4 mode 价值与集中评估

[`ModeValueModel`](models/value.py) 根据集中初始状态和匿名 mode 拟合数据集回报。当前模块只验证数据支持内的 value seam，不能把未出现的 `(s,z)` 外推值直接当作可靠 mode 选择依据。

它与 VO-MASD 中为 subgroup PPO 服务的 value head 不同：当前 value model 不参与中央 VQ 分组或 codebook 训练，也不构造内在奖励。

[`evaluation/metrics.py`](evaluation/metrics.py) 分开计算 mode recovery、分散一致性、行为支持和任务结果，避免用单一 return 掩盖 mode collapse 或联合动作失配。

## 5. Benchmark 数据集与指标

所有外部数据保存在 `/home/lotus/lotus/lhh/offline_datasets`。

| Benchmark | 数据位置 | 主要用途 | 当前适用范围 |
| --- | --- | --- | --- |
| Role-ID XOR | `mode_consistent/prototypes/` | 隔离验证独立 mode 采样的相关性缺口 | 验证理论 MMR、团队规模和公共 cue 可用率 |
| OMAR MPE Spread / Tag / World | `offline_datasets/OMAR/mpe/` | 无标签训练中央 VQ，并判断现有数据是否包含多种高回报行为结构 | Spread 的转换、分层划分和无标签评估已实现；Tag/World 的 tactic 定义仍需建立 |
| 受控 paired Simple Spread | `offline_datasets/CI-CoDiff/mpe/simple_spread/` | 确定性 PD 专家控制器在环境中在线 rollout，同一场景分别执行六种 assignment；不是 OMAR 数据或在线 MARL 策略 | 验证 mode discovery、局部可辨识性和条件策略 |
| OG-MARL SMAC | `offline_datasets/OG-MARL-Vault/smac/` | 检验复杂部分可观测协作任务上的外部效度 | 数据已保存，正式适配与 tactic 定义尚未完成 |

正式评估至少联合报告以下指标：

| 目标 | 指标 |
| --- | --- |
| mode recovery | NMI、ARI、best-mapping accuracy、code usage、perplexity、轨迹 code entropy |
| 无标签条件有效性 | VQ/no-code reconstruction、code-shuffle degradation、collector NMI、按 code 回报、终态 assignment NMI |
| decentralized realizability | MMR、agreement、coverage、consensus entropy、unknown rate |
| behavior support | mode 条件联合动作或轨迹最近邻距离、support violation |
| task result | return、success/win rate、time-to-agreement |
| cost | 训练预算、denoising steps、推理时间 |

## 6. 复现命令

以下命令均从仓库根目录执行。

### 6.1 软件回归与 motivating example

```bash
cd /home/lotus/lotus/lhh/madiff

PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n madiff pytest -q tests/mode_consistent

conda run -n madiff \
  python -m mode_consistent.prototypes.xor_motivating_example
```

### 6.2 生成受控 paired MPE 数据

```bash
DATA_ROOT=/home/lotus/lotus/lhh/offline_datasets/CI-CoDiff
PAIRED_DATA="$DATA_ROOT/mpe/simple_spread/balanced_6mode_expert_seed0.npz"

test -f "$PAIRED_DATA" || \
  conda run -n madiff python -m mode_consistent.scripts.generate_mpe_dataset \
    --output "$PAIRED_DATA" --n-scenarios 1000 \
    --mode-ids 0 1 2 3 4 5 --qualities expert --horizon 25 --seed 0
```

这里的 `expert` 表示无噪声、无 dropout 的固定 assignment PD 控制器。数据由控制器与 MPE 环境交互 rollout 得到，不来自 MAPPO/QMIX 等 online MARL 训练，也不是 OMAR 提供的专家数据。

### 6.3 可视化固定 seed 下的六种 mode

```bash
conda run -n madiff python -m mode_consistent.scripts.render_mpe_modes \
  --seed 0 --horizon 25 --mode-ids 0 1 2 3 4 5 \
  --output-dir mode_consistent/artifacts/mpe_mode_videos/seed0
```

该脚本复用数据生成器的同一个控制器，输出六个单独视频、同屏比较视频、终态轨迹图和包含 assignment、终点距离、回报与碰撞步数的 `summary.json`。

### 6.4 CPU 最小 pipeline

```bash
SMOKE_DATA=/tmp/ci_codiff_mpe_smoke.npz
test -f "$SMOKE_DATA" || \
  conda run -n madiff python -m mode_consistent.scripts.generate_mpe_dataset \
    --output "$SMOKE_DATA" \
    --n-scenarios 6 --mode-ids 0 1 --horizon 4 --seed 7

conda run -n madiff python -m mode_consistent.scripts.run_pipeline \
  "$SMOKE_DATA" --output /tmp/ci_codiff_pipeline.pt \
  --n-modes 2 --central-steps 2 --local-steps 2 \
  --value-steps 2 --diffusion-steps 2 --diffusion-timesteps 2 \
  --local-prefix 1 --confidence-threshold 0 --device cpu
```

该命令只验证数据形状、信息权限、梯度、checkpoint 和采样路径，不评价算法效果。

### 6.5 选择 GPU 运行冻结数据

```bash
nvtop

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

GPU_ID=3  # 示例；运行前改为当时的空闲卡
PAIRED_DATA=/home/lotus/lotus/lhh/offline_datasets/CI-CoDiff/mpe/simple_spread/balanced_6mode_expert_seed0.npz
RUN_DIR=mode_consistent/runs/pipeline_seed0
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run -n madiff \
  python -m mode_consistent.scripts.run_pipeline "$PAIRED_DATA" \
  --output "$RUN_DIR/checkpoint.pt" \
  --summary "$RUN_DIR/summary.json" \
  --n-modes 6 --seed 0 --device cuda
```

设置 `CUDA_VISIBLE_DEVICES` 后，所选物理卡映射为进程内 `cuda:0`，因此 runner 使用 `--device cuda`。

当前 GPU 命令仍是工程基线；论文实验需按研究进程文档中的阶段门冻结配置并运行多 seed。

### 6.6 P2 中央 mode 训练/评估入口 smoke

以下命令复用 6.4 按需生成的 `/tmp/ci_codiff_mpe_smoke.npz`。`-g -1` 隐藏 GPU 并使用 CPU；GPU smoke 应先用 `nvtop` 选卡，再替换为对应编号。

```bash
conda run -n madiff python run_experiment.py \
  -e exp_specs/mode_consistent/p2_central_smoke.yaml -g -1

conda run -n madiff python run_experiment.py \
  -e exp_specs/mode_consistent/eval_p2_central_smoke.yaml -g -1
```

该配置只有 2 个训练 step，只验证 YAML variant、三分 scenario split、标签防火墙、checkpoint、恢复信息和 evidence 输出。评估结果固定标记为 `pending_user_discussion`，不能据此判断 P2 是否有效。

### 6.7 OMAR Spread `H=25` 整轨迹 P2

转换后的训练文件只保存轨迹、回报、mask 和 `collector_ids`，不制造 `true_modes`。`collector_ids` 只用于分层划分和冻结后的审计，不进入模型输入或 checkpoint。

```bash
OMAR_ROOT=/home/lotus/lotus/lhh/offline_datasets/OMAR/mpe
OMAR_DATA=/home/lotus/lotus/lhh/offline_datasets/CI-CoDiff/omar/simple_spread/expert_unlabeled_h25.npz

test -f "$OMAR_DATA" || \
  conda run -n madiff python -m mode_consistent.scripts.convert_omar_mpe \
    --dataset-root "$OMAR_ROOT" --task simple_spread --horizon 25 \
    --n-agents 3 --output "$OMAR_DATA"

# 先用 nvtop 或 nvidia-smi 选择空闲卡，再替换 GPU_ID。
GPU_ID=0
conda run -n madiff python run_experiment.py \
  -e exp_specs/mode_consistent/p2_omar_spread_h25_seed0.yaml -g "$GPU_ID"
conda run -n madiff python run_experiment.py \
  -e exp_specs/mode_consistent/eval_p2_omar_spread_h25_seed0.yaml -g "$GPU_ID"
```

正式配置固定 `H=25`、`K=6`、`beta=0.001`、batch 128 和 1600 updates。14 万个 train episode 下约为 1.46 个 train pass，与旧 H=5 评估的数据暴露量对齐。评估只比较 VQ、no-code 和冻结 VQ 的 code-shuffle，不使用 oracle mode。输出仍标记为 `pending_user_discussion`。正式运行尚未执行，需先确认单 seed 预算。

只验证入口时，每个 collector 取 8 条 episode：

```bash
conda run -n madiff python -m mode_consistent.scripts.convert_omar_mpe \
  --dataset-root "$OMAR_ROOT" --task simple_spread --horizon 25 \
  --max-episodes-per-collector 8 \
  --output /tmp/ci_codiff_omar_spread_smoke.npz
conda run -n madiff python run_experiment.py \
  -e exp_specs/mode_consistent/p2_omar_spread_h25_smoke.yaml -g -1
conda run -n madiff python run_experiment.py \
  -e exp_specs/mode_consistent/eval_p2_omar_spread_h25_smoke.yaml -g -1
```

2-step smoke 只证明 `H=25 -> 单 code -> checkpoint -> 冻结评估` 可执行；其 code usage 或 reconstruction 数值不构成有效性证据。
