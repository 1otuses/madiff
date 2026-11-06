# 实施计划：把 AgentFormer 的 Agent-Aware 时空注意力迁移到 MADiff

> 交付对象：Codex（自动编码代理）
> 目标仓库：zbzhu99/madiff，本地根目录 `/home/lotus/lotus/lhh/madiff`
> 前期分析结论已固化在本文件，执行时不需要重新论证「为什么」，只需要「怎么做」。

## 环境前置

```
REPO=/home/lotus/lotus/lhh/madiff
cd $REPO                    # 所有命令、pytest、训练脚本都必须在该目录下执行
```

- Linux，仓库根目录 `/home/lotus/lotus/lhh/madiff`，下面所有路径以此为准。
- 根目录下有 `diffuser/`、`run_experiment.py`、`run_scripts/`、`exp_specs/`、`tests/`。
- `python -m pytest tests/...` 必须在 `$REPO` 下运行，否则 `from diffuser.models...` 会 ImportError。
- 不要写死 conda/venv 路径；沿用仓库当前可用的 Python 环境。

---

## 0. 一句话摘要

把 MADiff 里「agent 之间做 N×N 注意力、把时间轴压进 token」的 **TemporalSelfAttention**，
换成 AgentFormer 式的「在 (agent × time) 网格上做 (N·T)×(N·T) 注意力、同 agent / 跨 agent 用两套投影」的
**AgentAwareSpatioTemporalAttention**，保持 U-Net 主干、扩散流程、损失函数、评估流程完全不变。

**这是一个纯模块替换改造，不是重构。**

---

## 1. 目标 / 非目标

### 1.1 目标
- G1：新增一个即插即用的注意力模块，前向签名与 TemporalSelfAttention.forward(x, time) 兼容。
- G2：通过配置开关 attn_type: agent_aware 启用；默认 legacy 时不改变任何现有行为。
- G3：单元测试覆盖形状、恒等初始化、置换等变、self/other 掩码正确性。
- G4：在 MPE Spread / Tag 上跑通与 baseline 的对照实验。

### 1.2 非目标（明确禁止改动）
- 不改 diffuser/models/diffusion.py（损失、CFG、采样循环）
- 不改 diffuser/utils/evaluator.py（集中/分散 rollout 逻辑）
- 不改数据管道 diffuser/datasets/*
- 不改逆动力学模型
- 不引入新的第三方依赖（只用 torch / einops，仓库已有）
- 不改 U-Net 主干（Conv1d、下采样、上采样、FiLM 条件注入）

---

## 2. 现状基线（动手前必须核对的事实）

### 2.1 张量流

| 阶段 | 形状 | 备注 |
|---|---|---|
| 数据 batch | (B, T, A, F) | T = H + C（H 规划视野，C 历史视野）|
| 进入 SharedConvAttentionDeconv.forward | (B, T, A, F) | |
| rearrange 后 | (B, A, F, T) | agent 在前，通道 F 在时间 T 之前 |
| 折进 batch 过 U-Net | (B*A, F, T) | 每个 agent 独立做 1D 卷积 |
| 注意力调用点 | (B, A, F, T) + (B, A, embed_dim) | 只有这几个点跨 agent 混合 |

### 2.2 现有注意力的精确语义（关键）

diffuser/models/helpers.py 里的 SelfAttention 与 diffuser/models/temporal.py 里的 TemporalSelfAttention 是同一机制：

~~~
x_flat = rearrange(x, "b a f t -> (b a) f t")            # agent 折进 batch
q = rearrange(q, "(b a) (h d) t -> h b a (d t)", h=4)    # 每个 agent 的 (h*d*t) 压成一个 token
dots = einsum(q, k, "h b a1 f, h b a2 f -> h b a1 a2")   # 注意力矩阵 = A × A
attn = softmax(dots, dim=-1)                             # softmax 沿 agent 维
out  = einsum(attn, v, "h b a1 a2, h b a2 f -> h b a1 f")
~~~

**结论：token = 一个 agent 的整条时间轴；注意力矩阵是 A×A；时间段被内积求和掉了。**
它无法表达「agent j 在 t' 的动作影响 agent i 在 t 的规划」。

参数规模：q,k: F→F/4，v: F→F，合计 1.5·F²。
残差门控：gamma 初始化为 0（out = x + gamma * out），训练开始时是恒等映射。

### 2.3 插入点与调用签名

SharedConvAttentionDeconv.__init__ 构造 self.self_attn：

~~~
self.self_attn = [TemporalSelfAttention(in_out[-1][1], ...)]      # index 0 → mid_block2 之后
for dims in reversed(in_out):
    self.self_attn.append(TemporalSelfAttention(dims[1], ...))    # index 1..len(in_out) → 每个 upsample 前的 skip
self.self_attn = nn.ModuleList(self.self_attn)
~~~

**长度必须保持 1 + len(in_out)，索引映射必须不变**（forward 里写死了 self.self_attn[0] 和 self.self_attn[layer_idx+1]）。

MPE 配置（dim=128, dim_mults=[1,4,8], H=24, C=0）下的实际形状：

| 模块索引 | 位置 | F | T | token 数 (A=3) | 是否被调用 |
|---|---|---|---|---|---|
| 0 | mid_block2 后 | 1024 | 6 | 18 | 是 |
| 1 | ups[0] 前的 skip | 1024 | 6 | 18 | 是 |
| 2 | ups[1] 前的 skip | 512 | 12 | 36 | 是 |
| 3 | — | 128 | 24 | 72 | 否（构造但未使用，len(ups)=2）|

注意：由于 ups = len(in_out) - 1，索引 3 在 MPE 下是死代码。保持这个结构原样，不要顺手修掉，否则会破坏与其他模型类的一致性。

### 2.4 三条硬约束

1. 新模块的 forward(x, cond) 必须接受 x: (B, A, F, T)、cond: (B, A, embed_dim)，返回 (B, A, F, T)。
2. 类属性 agent_share_parameters = True 必须保留（run_scripts/train.py:61 会读它）。
3. 默认配置下行为必须与改造前逐比特一致（用于回归）。

---

## 3. 新模块设计

### 3.1 接口

~~~
class AgentAwareSpatioTemporalAttention(nn.Module):
    def __init__(
        self,
        n_channels: int,            # F
        n_agents: int,              # A
        history_horizon: int = 0,   # C
        nheads: int = 4,
        qk_ratio: int = 16,         # dqk = max(F // qk_ratio, 16)
        v_ratio: int = 4,           # dv  = F // v_ratio，且必须 nheads * dv == F
        cond_dim: int | None = None,# embed_dim（time + returns）
        cond_mode: str = "film",    # "film" | "adaln" | "none"
        use_time_enc: bool = True,
        use_segment_emb: bool = True,
        use_ffn: bool = False,
        dropout: float = 0.0,
        residual: bool = True,
    ): ...

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor: ...
~~~

### 3.2 前向流程（逐步骤 + 形状）

~~~
输入 x: (B, A, F, T), cond: (B, A, D)

1) token 化（a-major）
   h = x.permute(0,1,3,2).reshape(B, A*T, F)
   约定：token index i = a * T + t  → 同一个 agent 的 token 在序列里连续

2) 时间编码（相对窗口内位置 0..T-1）
   pe = sinusoidal(T, F)，t / max(T-1,1) * 10，base 10000
   h = h + pe.view(1,1,T,F).expand(B,A,T,F).reshape(B,A*T,F)

3) segment 编码（仅当 C > 0）
   seg = (arange(T) >= C).long()                  # 0 = 历史，1 = 未来
   h = h + seg_emb(seg).view(1,1,T,F).expand(...)

4) 条件注入
   film  : c = cond_mlp(cond)                     # (B, A, F)
           h = h + c.view(B,A,1,F).expand(B,A,T,F).reshape(B,A*T,F)
   adaln : c = adaln(cond).view(B,A,1,6F).expand(...).reshape(B,A*T,6F)
           s1,b1,g1,s2,b2,g2 = c.chunk(6, -1)

5) 归一化
   h = ln_in(h)
   if adaln: h = h * (1 + s1) + b1

6) 双投影
   q_self  = q_self_proj(h)   # (B, A*T, nheads*dqk)
   k_self  = k_self_proj(h)
   q_other = q_other_proj(h)
   k_other = k_other_proj(h)
   v       = v_proj(h)        # (B, A*T, nheads*dv) = (B, A*T, F)
   全部 view(B, A*T, nheads, d).transpose(1,2)  →  (B, nheads, A*T, d)

7) 双 logits + 掩码选择
   logit_self  = (q_self  @ k_self.transpose(-1,-2))  / sqrt(dqk)
   logit_other = (q_other @ k_other.transpose(-1,-2)) / sqrt(dqk)
   mask[i,j] = (i // T == j // T)                      # (A*T, A*T) bool，True = 同一 agent
   logits = torch.where(mask, logit_self, logit_other)

8) softmax + dropout + 加权求和
   attn = softmax(logits, dim=-1)                      # (B, H, A*T, A*T)
   out  = attn @ v                                     # (B, H, A*T, dv)

9) 合并 head 并还原形状
   out = out.transpose(1,2).reshape(B, A*T, F)
   out = out.reshape(B, A, T, F).permute(0,1,3,2)      # (B, A, F, T)

10) 残差
   if adaln: out = g1 * out
   out = x + gamma * out                               # gamma 初始化为 0

11) 可选 FFN（默认关闭）
   out = out + gamma2 * ffn(ln_ffn(out))               # gamma2 初始化为 0
~~~

### 3.3 掩码说明

- 用 a-major 顺序，掩码是块对角（每个 agent 一个 T×T 的块）。
- 不要用 AgentFormer 原文的 i mod N——那是 t-major 顺序下的等价写法，在这里既脆弱又难调试。
- 掩码每个 forward 现算即可（最大 A·T = 192，192² 个 bool，可忽略）。可加一个按 (A, T) 缓存的 dict。

### 3.4 位置编码

- 必须加。改成 (a,t) token 后注意力对 t 是置换不变的，不加一定退化。
- 用相对窗口位置（0..T-1），不依赖绝对 episode 时间——绝对时间已由 env_ts_condition 全局注入。
- scale=10.0 是经验值，让正弦在 T≤24 上展开得更开。

### 3.5 条件注入的两个版本

| 版本 | 做法 | 默认 | 风险 |
|---|---|---|---|
| V1 film | h = h + MLP(cond)，把扩散步与 RTG 的联合 embedding 平移到 token 上 | 是 | 低，最接近旧的「加到 q/k/v」 |
| V2 adaln | h = LN(h) * (1+s) + b，gate 控制残差 | 否 | 中，需要验证训练稳定性 |

两者都用零初始化的门控保证「初始恒等」。

### 3.6 参数与显存预算

| 项 | legacy | 新模块 | 比值 |
|---|---|---|---|
| q/k 参数 | 2 · F²/4 | 4 · F²/4 = F² | 2× |
| v 参数 | F² | F² | 1× |
| 合计 | 1.5 F² | 2 F² | 1.33× |

MPE 全部 4 个模块：legacy ≈ 3.5M → 新 ≈ 4.7M（+1.2M）。

注意力矩阵显存：B · nheads · (A·T)² · 4 bytes，self/other 两份。
- MPE B=32, A=3, T=24, H=4：32·4·72²·4 ≈ 2.65 MB ×2
- SMAC 8m A=8, T=24：32·4·192²·4 ≈ 18.9 MB ×2（若 OOM，对该层用 torch.utils.checkpoint）

---

## 4. 代码改动清单

### 4.1 新增文件：/home/lotus/lotus/lhh/madiff/diffuser/models/agent_aware_attention.py

完整参考实现（可直接用）：

~~~
"""Agent-aware spatio-temporal attention (AgentFormer-style) for MADiff.

Drop-in replacement for diffuser.models.temporal.TemporalSelfAttention.

legacy: (B, A, F, T) -> attention over A tokens, token = flatten(head_dim, T)
ours  : (B, A, F, T) -> attention over A*T tokens, self/other 两套投影
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_pos_emb(T: int, dim: int, device, dtype=torch.float32, scale: float = 10.0):
    """(T, dim) sinusoidal encoding over relative window positions."""
    pos = (torch.arange(T, device=device, dtype=dtype) / max(T - 1, 1) * scale).unsqueeze(1)
    i = torch.arange(0, dim, 2, device=device, dtype=dtype)
    div = torch.exp(-math.log(10000.0) * i / dim)
    pe = torch.zeros(T, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class AgentAwareSpatioTemporalAttention(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_agents: int,
        history_horizon: int = 0,
        nheads: int = 4,
        qk_ratio: int = 16,
        v_ratio: int = 4,
        cond_dim: Optional[int] = None,
        cond_mode: str = "film",
        use_time_enc: bool = True,
        use_segment_emb: bool = True,
        use_ffn: bool = False,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_agents = n_agents
        self.nheads = nheads
        self.history_horizon = history_horizon
        self.use_time_enc = use_time_enc
        self.use_segment_emb = bool(use_segment_emb and history_horizon > 0)

        self.dqk = max(n_channels // qk_ratio, 16)
        self.dv = n_channels // v_ratio
        assert self.nheads * self.dv == n_channels, (
            f"nheads({nheads}) * dv({self.dv}) != n_channels({n_channels})"
        )

        self.q_self = nn.Linear(n_channels, nheads * self.dqk)
        self.k_self = nn.Linear(n_channels, nheads * self.dqk)
        self.q_other = nn.Linear(n_channels, nheads * self.dqk)
        self.k_other = nn.Linear(n_channels, nheads * self.dqk)
        self.v = nn.Linear(n_channels, nheads * self.dv)

        self.ln_in = nn.LayerNorm(n_channels)
        self.dropout = nn.Dropout(dropout)

        self.cond_mode = cond_mode if cond_dim is not None else "none"
        if self.cond_mode == "film":
            self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, n_channels))
        elif self.cond_mode == "adaln":
            self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * n_channels))
            nn.init.zeros_(self.adaln[-1].weight)
            nn.init.zeros_(self.adaln[-1].bias)
        elif self.cond_mode != "none":
            raise ValueError(f"unknown cond_mode: {cond_mode}")

        if self.use_segment_emb:
            self.seg_emb = nn.Embedding(2, n_channels)

        self.use_ffn = use_ffn
        if use_ffn:
            self.ln_ffn = nn.LayerNorm(n_channels)
            self.ffn = nn.Sequential(
                nn.Linear(n_channels, 4 * n_channels),
                nn.GELU(),
                nn.Linear(4 * n_channels, n_channels),
            )
            self.gamma_ffn = nn.Parameter(torch.zeros(1))

        self.residual = residual
        if residual:
            self.gamma = nn.Parameter(torch.zeros(1))

        self._mask_cache = {}

    def _agent_mask(self, A: int, T: int, device):
        key = (A, T, str(device))
        if key not in self._mask_cache:
            idx = torch.arange(A * T, device=device)
            agent_of = idx // T                       # a-major: i // T 就是 agent 编号
            self._mask_cache[key] = agent_of.unsqueeze(0) == agent_of.unsqueeze(1)
        return self._mask_cache[key]

    def forward(self, x, cond=None):
        B, A, C, T = x.shape
        assert A == self.n_agents, f"expected {self.n_agents} agents, got {A}"
        assert C == self.n_channels

        h = x.permute(0, 1, 3, 2).reshape(B, A * T, C)           # (B, A*T, C)

        if self.use_time_enc:
            pe = sinusoidal_pos_emb(T, C, x.device, x.dtype)      # (T, C)
            h = h + pe.view(1, 1, T, C).expand(B, A, T, C).reshape(B, A * T, C)

        if self.use_segment_emb:
            seg = (torch.arange(T, device=x.device) >= self.history_horizon).long()
            h = h + self.seg_emb(seg).view(1, 1, T, C).expand(B, A, T, C).reshape(B, A * T, C)

        s1 = b1 = g1 = None
        if self.cond_mode == "film" and cond is not None:
            c = self.cond_mlp(cond)                               # (B, A, C)
            h = h + c.view(B, A, 1, C).expand(B, A, T, C).reshape(B, A * T, C)
        elif self.cond_mode == "adaln" and cond is not None:
            c = self.adaln(cond)                                  # (B, A, 6C)
            c = c.view(B, A, 1, 6 * C).expand(B, A, T, 6 * C).reshape(B, A * T, 6 * C)
            s1, b1, g1, s2, b2, g2 = c.chunk(6, dim=-1)

        h = self.ln_in(h)
        if s1 is not None:
            h = h * (1 + s1) + b1

        def _heads(t, d):
            return t.view(B, A * T, self.nheads, d).transpose(1, 2)

        qs = _heads(self.q_self(h), self.dqk)
        ks = _heads(self.k_self(h), self.dqk)
        qo = _heads(self.q_other(h), self.dqk)
        ko = _heads(self.k_other(h), self.dqk)
        v = _heads(self.v(h), self.dv)

        scale = 1.0 / math.sqrt(self.dqk)
        logit_self = torch.matmul(qs, ks.transpose(-1, -2)) * scale
        logit_other = torch.matmul(qo, ko.transpose(-1, -2)) * scale

        mask = self._agent_mask(A, T, x.device)                   # (A*T, A*T) bool
        logits = torch.where(mask, logit_self, logit_other)

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                               # (B, H, A*T, dv)

        out = out.transpose(1, 2).reshape(B, A * T, C)
        out = out.reshape(B, A, T, C).permute(0, 1, 3, 2)         # (B, A, C, T)

        if s1 is not None:
            out = g1 * out
        if self.residual:
            out = self.gamma * out
        out = x + out

        if self.use_ffn:
            out = out + self.gamma_ffn * self.ffn(self.ln_ffn(out))
        return out
~~~

### 4.2 修改 /home/lotus/lotus/lhh/madiff/diffuser/models/ma_temporal.py

只改 SharedConvAttentionDeconv，ConvAttentionDeconv（非共享参数版）不动。

**(a) __init__ 增加参数（全部带默认值，保证向后兼容）**

~~~
def __init__(
    self,
    horizon, transition_dim, dim=128, history_horizon=0,
    dim_mults=(1, 2, 4, 8), nhead=4, n_agents=2,
    returns_condition=False, env_ts_condition=False, condition_dropout=0.1,
    kernel_size=5, residual_attn=True, use_layer_norm=False,
    max_path_length=100, use_temporal_attention=True,
    # ---- 新增 ----
    attn_type: str = "legacy",      # "legacy" | "agent_aware"
    attn_heads: int = 4,
    attn_cond_mode: str = "film",
    attn_use_time_enc: bool = True,
    attn_use_ffn: bool = False,
    attn_levels: str = "all",       # "all" | "mid"
):
~~~

**(b) 把构造 self.self_attn 的那段替换为工厂函数**

原代码：

~~~
self.self_attn = [TemporalSelfAttention(in_out[-1][1], in_out[-1][1] // 16, in_out[-1][1] // 4,
                                        residual=residual_attn, embed_dim=self.net.embed_dim)]
for dims in reversed(in_out):
    self.self_attn.append(TemporalSelfAttention(dims[1], dims[1] // 16, dims[1] // 4,
                                                residual=residual_attn, embed_dim=self.net.embed_dim))
self.self_attn = nn.ModuleList(self.self_attn)
~~~

替换为：

~~~
assert attn_type in ("legacy", "agent_aware")
self.attn_type = attn_type
self.attn_levels = attn_levels

def _make_attn(f_ch, level_idx):
    use_aa = (attn_type == "agent_aware") and (attn_levels == "all" or level_idx == 0)
    if use_aa:
        return AgentAwareSpatioTemporalAttention(
            n_channels=f_ch,
            n_agents=n_agents,
            history_horizon=history_horizon,
            nheads=attn_heads,
            cond_dim=self.net.embed_dim,
            cond_mode=attn_cond_mode,
            use_time_enc=attn_use_time_enc,
            use_ffn=attn_use_ffn,
            residual=residual_attn,
        )
    return TemporalSelfAttention(
        f_ch, f_ch // 16, f_ch // 4,
        residual=residual_attn, embed_dim=self.net.embed_dim,
    )

self.self_attn = [_make_attn(in_out[-1][1], 0)]
for _i, _dims in enumerate(reversed(in_out)):
    self.self_attn.append(_make_attn(_dims[1], _i + 1))
self.self_attn = nn.ModuleList(self.self_attn)
~~~

注意：长度与索引映射保持不变（0 = mid，1.. = 每个 upsample 前的 skip）。

**(c) forward 里的调用点**

原代码有三处 if self.use_temporal_attention: ... else: ... 分支，改成统一派发。

在 __init__ 末尾加：

~~~
self._attn_needs_cond = (attn_type != "legacy") or use_temporal_attention
~~~

mid 处：

~~~
if self._attn_needs_cond:
    t_ = t.reshape(bs, t.shape[0] // bs, t.shape[1])
    x = self.self_attn[0](x, t_)
    t = t_.reshape(t_.shape[0] * t_.shape[1], t_.shape[2])
else:
    x = self.self_attn[0](x)
~~~

skip 处同理：

~~~
hiddens = self.self_attn[layer_idx + 1](hiddens, t_)
~~~

保持不变的部分：x.reshape(bs, ...)、layer_norm、h.pop()、通道拼接、上采样，全部照旧。

**(d) 文件顶部加 import**

~~~
from .agent_aware_attention import AgentAwareSpatioTemporalAttention
~~~

### 4.3 修改 /home/lotus/lotus/lhh/madiff/diffuser/models/__init__.py

按本方案（改 SharedConvAttentionDeconv 加开关）无需改动。
仅当选择「新增子类」路线时，才需要在这里导出新类。

### 4.4 修改 /home/lotus/lotus/lhh/madiff/run_scripts/train.py

在 model_config = utils.Config(...) 之前构造条件参数：

~~~
model_extra = {}
if getattr(Config, "attn_type", None):
    model_extra.update(
        attn_type=Config.attn_type,
        attn_heads=getattr(Config, "attn_heads", 4),
        attn_cond_mode=getattr(Config, "attn_cond_mode", "film"),
        attn_use_time_enc=getattr(Config, "attn_use_time_enc", True),
        attn_use_ffn=getattr(Config, "attn_use_ffn", False),
        attn_levels=getattr(Config, "attn_levels", "all"),
    )
~~~

然后：

~~~
model_config = utils.Config(
    Config.model,
    savepath="model_config.pkl",
    ...,
    device=Config.device,
    **model_extra,
)
~~~

为什么用 **model_extra 而不是直接加参数：Config.model 可能指向其它模型类（ConcatenatedTemporalUnet 等），它们不认这些参数。只有在 yaml 里显式写了 attn_type 时才传入。

### 4.5 新增实验配置

复制 /home/lotus/lotus/lhh/madiff/exp_specs/mpe/spread/mad_mpe_spread_attn_exp.yaml
到 /home/lotus/lotus/lhh/madiff/exp_specs/mpe/spread/mad_mpe_spread_aa_attn_exp.yaml，修改：

~~~
meta_data:
  exp_name: "mad_mpe_aa"

constants:
  # ---- 新增 ----
  attn_type: "agent_aware"
  attn_heads: 4
  attn_cond_mode: "film"
  attn_use_time_enc: True
  attn_use_ffn: False
  attn_levels: "all"
~~~

同时把 job_name 加上 aa 前缀，避免和 baseline 日志混淆：

~~~
  job_name: "{dataset}/h_{horizon}-aa-{model}-r_{returns_scale}-guidew_{condition_guidance_w}-ctde_{decentralized_execution}"
~~~

其余保持不变（model: "models.SharedConvAttentionDeconv" 不变）。

### 4.6 新增测试文件：/home/lotus/lotus/lhh/madiff/tests/test_agent_aware_attention.py

~~~
import torch
from diffuser.models.agent_aware_attention import AgentAwareSpatioTemporalAttention


def _make(A=3, C=128, T=24, H=4, **kw):
    return AgentAwareSpatioTemporalAttention(C, A, nheads=H, cond_dim=2 * 128,
                                             cond_mode="none", **kw)


def test_shape_preserved():
    m = _make()
    x = torch.randn(2, 3, 128, 24)
    assert m(x).shape == x.shape


def test_identity_at_init():
    m = _make()
    x = torch.randn(2, 3, 128, 24)
    torch.testing.assert_close(m(x), x)      # gamma = 0 -> 恒等


def test_permutation_equivariance():
    torch.manual_seed(0)
    m = _make()
    x = torch.randn(2, 3, 128, 24)
    perm = torch.tensor([2, 0, 1])
    torch.testing.assert_close(m(x[:, perm]), m(x)[:, perm], atol=1e-6, rtol=1e-5)


def test_single_agent_equals_plain_attention():
    """A=1 时掩码全 True，应等价于标准 self-attention over T。"""
    torch.manual_seed(0)
    A, C, T, H = 1, 16, 5, 2
    m = AgentAwareSpatioTemporalAttention(C, A, nheads=H, cond_dim=None,
                                          cond_mode="none", use_time_enc=False,
                                          use_segment_emb=False, residual=False)
    x = torch.randn(1, A, C, T)
    out = m(x)

    h = m.ln_in(x.permute(0, 1, 3, 2).reshape(1, T, C))
    q = m.q_self(h).view(1, T, H, m.dqk).transpose(1, 2)
    k = m.k_self(h).view(1, T, H, m.dqk).transpose(1, 2)
    v = m.v(h).view(1, T, H, m.dv).transpose(1, 2)
    a = torch.softmax(q @ k.transpose(-1, -2) / (m.dqk ** 0.5), dim=-1)
    ref = (a @ v).transpose(1, 2).reshape(1, T, C)
    ref = ref.reshape(1, A, T, C).permute(0, 1, 3, 2) + x
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)


def test_self_other_split():
    """与 §10 的参考实现逐元素比对。"""
    from tests.ref_attention import reference_agent_aware_attention
    torch.manual_seed(0)
    m = AgentAwareSpatioTemporalAttention(16, 2, nheads=2, cond_dim=None,
                                          cond_mode="none", use_time_enc=False,
                                          use_segment_emb=False, residual=True)
    torch.nn.init.constant_(m.gamma, 0.7)     # 打开门控，避免恒等掩盖错误
    x = torch.randn(1, 2, 16, 3)
    torch.testing.assert_close(m(x), reference_agent_aware_attention(m, x),
                               atol=1e-6, rtol=1e-5)
~~~

---

## 5. 配置项

| 键 | 取值 | 默认 | 说明 |
|---|---|---|---|
| attn_type | legacy / agent_aware | legacy | 总开关。不写该键 = 完全保持原行为 |
| attn_heads | int | 4 | 注意力头数 |
| attn_cond_mode | film / adaln | film | 条件注入方式 |
| attn_use_time_enc | bool | True | 是否加窗口内正弦位置编码 |
| attn_use_ffn | bool | False | 是否在注意力后加 FFN 子层 |
| attn_levels | all / mid | all | 只替换 mid（消融用）还是全部替换 |

---

## 6. 验收标准

### 6.1 单元测试（必须全绿）

~~~
cd /home/lotus/lotus/lhh/madiff && python -m pytest tests/test_agent_aware_attention.py -v
~~~

- [ ] 形状不变
- [ ] 初始化恒等（gamma=0）
- [ ] 置换等变（误差 < 1e-6）
- [ ] A=1 退化为标准自注意力
- [ ] self/other 投影切换正确

### 6.2 回归测试（关键）

~~~
python run_experiment.py -e exp_specs/mpe/spread/mad_mpe_spread_attn_exp.yaml
~~~

判据分两级，**先强后弱**：

- [ ] **强形式（必过，结构等价）**：attn_type 缺省时，模型必须与改动前完全一致
  - `sum(p.numel() for p in model.parameters())` 相同
  - `set(model.state_dict().keys())` 完全相同（一条不多、一条不少）
  - 处理方式：改动前后各 dump 一次参数量与 key 集合，做 diff
- [ ] **弱形式（参考，数值等价）**：同一 seed 下 step-0 loss 差异 < 1e-5

**为什么不能要求 loss 逐比特相同**：即使在 legacy 分支下不创建任何新模块，只要 `__init__` 里多执行了一次 `nn.Linear(...)` 或 `nn.Parameter(...)`（哪怕是死代码），PyTorch 的全局 RNG 状态就会被推进，后续所有参数的初始化值都会跟着变，step-0 loss 必然不同。这不代表改错了。

因此实现上还有一条硬要求：**`attn_type == "legacy"` 时，`_make_attn` 必须完全不实例化 `AgentAwareSpatioTemporalAttention`**（不要“先建好再丢掉”），否则连弱形式都过不了。

### 6.3 冒烟测试

~~~
python run_experiment.py -e exp_specs/mpe/spread/mad_mpe_spread_aa_attn_exp.yaml
~~~

- [ ] 2000 步内 loss 不 NaN / 不爆炸
- [ ] 显存峰值 < baseline × 1.5
- [ ] 单步耗时 < baseline × 2.0

### 6.4 训练测试

- [ ] 单 seed 跑完，评估分数不低于 baseline −5%
- [ ] 注意力统计：跨 agent 注意力占比 > 0（说明模型确实在用）

---

## 7. 实验矩阵

### 7.1 第一阶段（P0，必做）

| 配置 | 环境 | 数据 | seeds | 目的 |
|---|---|---|---|---|
| baseline | Spread | Md-Replay | 100/200/300 | 对照 |
| agent_aware | Spread | Md-Replay | 100/200/300 | 主实验 |
| baseline | Tag | Md-Replay | 100/200/300 | 对照 |
| agent_aware | Tag | Md-Replay | 100/200/300 | 主实验（预期收益最大）|

### 7.2 第二阶段（P1）

- Spread / Tag × {Expert, Medium}
- attn_levels=mid 消融（只换 mid 一处，隔离收益来源）
- attn_cond_mode=adaln vs film
- attn_use_ffn=True vs False

### 7.3 判据

- 主判据：3 seeds 平均 normalized score。Tag 上 agent_aware 应显著优于 baseline（预期 +10% 以上）；Spread Expert 可能持平或微降——这是可接受的，不要为了 Spread Expert 掉点而回退。
- 辅判据：训练曲线收敛速度、注意力熵、跨 agent 注意力占比。

---

## 8. 风险与回退

| 风险 | 现象 | 处置 |
|---|---|---|
| 参数量 +33% 导致过拟合 | 训练 loss 降但评估掉 | 先只替换 mid（attn_levels=mid），或降 attn_heads |
| 显存 OOM（SMAC 8m） | CUDA OOM | 对该层加 torch.utils.checkpoint，或降到 attn_levels=mid |
| 训练不稳定 | loss 尖峰 | attn_cond_mode=film + 梯度裁剪 1.0 + 前 5k 步 lr warmup |
| 注意力退化为只关注自己 | 跨 agent 占比 ≈ 0 | 检查 mask 是否写反（i // T）；检查 gamma 是否在增长 |
| 旧 checkpoint 无法加载 | missing/unexpected keys | 预期行为，不要试图兼容；新实验重新训 |

回退方式：yaml 里删掉 attn_type（或设为 legacy），一切恢复原状。代码路径不删。

---

## 9. 执行顺序（给 Codex 的 checklist）

1. [ ] git checkout -b feat/agent-aware-attention
2. [ ] 跑通 baseline 一个 seed（确认环境、数据、路径都正常），记录 step-0 loss 与参数量
3. [ ] 新建 diffuser/models/agent_aware_attention.py（§4.1）
4. [ ] 写单测 tests/test_agent_aware_attention.py 与 tests/ref_attention.py（§4.6 + §10），跑绿
5. [ ] 改 ma_temporal.py（§4.2），确认 attn_type 缺省时行为不变
6. [ ] 改 train.py（§4.4）
7. [ ] 加 yaml（§4.5）
8. [ ] 冒烟 2000 步（§6.3），记录 loss / 显存 / 耗时
9. [ ] 跑 P0 实验矩阵（§7.1），汇总表格
10. [ ] 输出对比报告 + 注意力统计图

---

## 10. 附：参考实现（供 tests/ref_attention.py 使用）

用 einsum 重写一遍 masked 注意力，与模块输出逐元素比对：

~~~
import torch


def reference_agent_aware_attention(m, x):
    """独立实现，用于交叉验证。x: (B, A, C, T)"""
    B, A, C, T = x.shape
    h = x.permute(0, 1, 3, 2).reshape(B, A * T, C)     # a-major
    h = m.ln_in(h)

    H, dqk, dv = m.nheads, m.dqk, m.dv
    qs = m.q_self(h).view(B, A * T, H, dqk).permute(0, 2, 1, 3)   # B,H,AT,d
    ks = m.k_self(h).view(B, A * T, H, dqk).permute(0, 2, 1, 3)
    qo = m.q_other(h).view(B, A * T, H, dqk).permute(0, 2, 1, 3)
    ko = m.k_other(h).view(B, A * T, H, dqk).permute(0, 2, 1, 3)
    v = m.v(h).view(B, A * T, H, dv).permute(0, 2, 1, 3)

    idx = torch.arange(A * T, device=x.device)
    mask = (idx // T).unsqueeze(0) == (idx // T).unsqueeze(1)
    mask = mask.view(1, 1, A * T, A * T)

    ls = torch.einsum("bhnd,bhmd->bhnm", qs, ks) / dqk ** 0.5
    lo = torch.einsum("bhnd,bhmd->bhnm", qo, ko) / dqk ** 0.5
    logits = torch.where(mask, ls, lo)
    attn = torch.softmax(logits, dim=-1)
    out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
    out = out.permute(0, 2, 1, 3).reshape(B, A * T, C)
    out = out.view(B, A, T, C).permute(0, 1, 3, 2)
    return x + m.gamma * out
~~~

这个实现故意写成与主实现不同的形式（einsum vs matmul、4D mask vs 2D where），用来抓广播/转置错误。

---

## 11. 常见坑

1. nheads * dv == F：v_ratio=4、nheads=4 时 dv=F/4，正好。若改 attn_heads，必须同步改 v_ratio，否则断言失败。
2. 张量顺序：(B, A, F, T) → (B, A, T, F) 的 permute 是 (0,1,3,2)，别写反。
3. mask 的 i // T：只有 a-major 展平才成立。若改成 t-major，必须同步改为 i % A。
4. self.self_attn 长度：保持 1 + len(in_out)，否则 forward 的索引会错位或越界。
5. nn.ModuleList 构造：列表推导里创建模块后必须用 nn.ModuleList 包起来，否则参数不会注册。
6. Config 读取：新键用 getattr(Config, "attn_type", None)，不要直接 Config.attn_type，否则没写该键的旧 yaml 会抛 AttributeError。
7. 不要动 ConvAttentionDeconv（非共享参数版），本次改造只针对 SharedConvAttentionDeconv。
8. EMA：Trainer 的 EMA 会自动包含新参数，无需额外处理；但旧 checkpoint 不能复用。
9. RNG：legacy 分支下不要在 `_make_attn` 里“先实例化新模块再丢弃”——构造时的参数初始化会推进全局 RNG，导致 step-0 loss 无法与改动前对齐（见 §6.2）。要用 if/else 分支，不要用“建了再说”。
