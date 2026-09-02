# 高层研究框架

> 本文只描述尚未实例化的研究骨架，不代表已经选定算法。

## 概念流程

```text
offline joint trajectories
          │
          ▼
coordination structure discovery
(training-time joint information is allowed)
          │
          ▼
execution-side recovery or selection
(only explicitly permitted information)
          │
          ▼
coordination-conditioned diffusion policy
          │
          ▼
decentralized actions and real-environment rollout
```

## 四个必要模块

1. **问题与数据机制**：证明数据中确实存在多个可行协作方式，并构造能够观察错误重组后果的对照。
2. **协作表示**：从联合离线行为中提取对联合决策有因果或预测意义的表示，同时排除 return、场景编号、采集器和时间阶段等伪相关。
3. **执行接口**：说明每个 agent 在动作产生前如何获得或推断该表示，并严格审计其信息来源。
4. **生成与验证**：将表示接入 diffusion-based policy，比较无表示、正确表示、扰动表示等对照，最后进行真实 rollout。

## 尚待用户确定的关键选择

- coordination representation 的粒度与形式：离散、连续、层级、静态或动态。
- 执行制度：仅私有历史、公共线索、共享随机变量或显式通信。
- diffusion 生成对象：动作、状态/观测轨迹或联合轨迹。
- 数据集与 motivating example。
- 第一阶段的可证伪假设、baseline、指标和停止条件。
- 优先复用的论文框架与开源代码库。

在这些选择被逐项确认前，本目录不增加实现代码。
