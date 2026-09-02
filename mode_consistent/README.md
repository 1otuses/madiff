# Mode-Consistent Offline MARL

> 状态：已完整重置，等待用户逐步确定研究方案。

本目录当前只保留研究核心，不保留此前的实现、配置、测试、实验结果或方法命名。

## 核心问题

Offline MARL 数据可能混合多种有效的联合协作方式。若各 agent 在 decentralized execution
时分别拟合或采样局部边缘行为，就可能把不同协作方式的动作错误组合，形成离线数据联合支持之外的行为，导致 coordination failure。

可用下面的潜变量分解表达核心设想：

$$
\mu(\mathbf a\mid\mathbf h)
=\sum_z p(z\mid\mathbf h)\prod_i\mu_i(a_i\mid h_i,z),
$$

其中 $z$ 表示尚未确定具体形式的 coordination representation。研究目标是判断：显式建模并在生成策略中使用该表示，能否减少跨协作方式的错误重组，并提升真实环境中的联合决策质量。

## 保留的研究原则

1. 研究基座仍是 **Offline MARL**，优先考虑 **diffusion-based policy / trajectory generation**。
2. 训练期可以利用联合轨迹发现协作结构，但执行期输入必须事先明确，不能隐式使用其他 agent 的私有信息。
3. 必须区分 private local history、public cue 和 shared key / communication；后三者带来的能力不能混称为严格 decentralized execution。
4. latent/code 的一致率、重建误差或训练 loss 下降都不是最终证据；需要验证它是否表达协作语义、是否在执行前可获得、是否真正改变联合行为。
5. 最终有效性以真实环境 rollout 的 return、success、coordination failure 和多 seed 结果为准。

## 当前不作出的承诺

- 不预设 coordination representation 必须是离散 code、VQ 或固定时长 mode。
- 不预设此前任何模型、损失、数据集或实验结论继续有效。
- 不预设 MADiff 是唯一可复用代码基座。
- 在明确问题设定和最小可证伪实验前，不开始大规模训练。

高层概念框架见 [FRAMEWORK.md](FRAMEWORK.md)。下一步研究内容由用户逐步指定。
