import math
from typing import Tuple

import einops
import torch
import torch.nn as nn
from einops import einsum, rearrange
from einops.layers.torch import Rearrange
from torch.distributions import Bernoulli

from .helpers import Conv1dBlock, Downsample1d, SinusoidalPosEmb, Upsample1d


class Residual(nn.Module): # 残差连接
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module): # 前置归一化
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module): # 线性注意力机制
    def __init__(self, dim, heads=4, dim_head=128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape # [B, C, H, W]
        qkv = self.to_qkv(x) # 生成Q、K、V [B, 3*hidden_dim, H, W]
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", 
            heads=self.heads, 
            qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class TemporalSelfAttention(nn.Module): # 时序自注意力类
    def __init__(
        self,
        n_channels: int,
        qk_n_channels: int,
        v_n_channels: int,
        embed_dim: int,
        nheads: int = 4,
        residual: bool = False,
    ):
        super().__init__()
        self.nheads = nheads

        self.query_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.key_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.value_layer = nn.Conv1d(n_channels, v_n_channels * nheads, kernel_size=1)

        self.query_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, qk_n_channels * nheads),
            Rearrange("batch t -> batch t 1"),
        )
        self.key_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, qk_n_channels * nheads),
            Rearrange("batch t -> batch t 1"),
        )
        self.value_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, v_n_channels * nheads),
            Rearrange("batch t -> batch t 1"),
        )

        self.attend = nn.Softmax(dim=-1)
        self.residual = residual
        if residual:
            self.gamma = nn.Parameter(torch.zeros([1]))

    def forward(self, x, time):
        # x: [B, A, F, H]
        # time: [B, A, embed_dim]
        x_flat = rearrange(x, "b a f t -> (b a) f t") # [B*A, F, H]
        time = rearrange(time, "b a f -> (b a) f") # [B*A, embed_dim]
        query, key, value = (
            self.query_layer(x_flat) + self.query_time_mlp(time),
            self.key_layer(x_flat) + self.key_time_mlp(time),
            self.value_layer(x_flat) + self.value_time_mlp(time),
        ) # 为每个agent生成Q、K、V

        query = rearrange(
            query, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        ) # [heads, B, A, channels*H]
        key = rearrange(
            key, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )
        value = rearrange(
            value, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )

        dots = einsum(query, key, "h b a1 f, h b a2 f -> h b a1 a2") / math.sqrt(
            query.shape[-1]
        ) # QK/sqrt(d_k) [heads, B, A, A]
        attn = self.attend(dots) # softmax
        out = einsum(attn, value, "h b a1 a2, h b a2 f -> h b a1 f")
        # agent会获取其他agents的轨迹信息
        out = rearrange(out, "h b a f -> b a (h f)")
        out = out.reshape(x.shape)
        if self.residual:
            out = x + self.gamma * out
        return out


class TemporalMlpBlock(nn.Module): # 带时间编码的多层感知机块
    '''
    与ResidualTemporalBlock不同的是,ResidualTemporalBlock使用卷积学习trajectory相邻时间变化规律,而TemporalMlpBlock使用全连接层学习trajectory相邻时间变化规律
    '''
    def __init__(self, dim_in, dim_out, embed_dim, act_fn, out_act_fn):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    act_fn,
                ),
                nn.Sequential(
                    nn.Linear(dim_out, dim_out),
                    out_act_fn,
                ),
            ]
        )
        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, dim_out),
        )

    def forward(self, x, t):
        """
        x : [ B, inp_channels, H ]
        t : [ B, embed_dim ]
        return out : [ B, out_channels, H ]
        """

        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out


class ResidualTemporalBlock(nn.Module): # 残差时序块
    '''
    用conv1学习trajectory相邻时间变化规律——局部时序规律
    嵌入diffusion去噪时间步t
    残差保留原始信息
    '''
    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5, mish=True):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
                Conv1dBlock(out_channels, out_channels, kernel_size, mish),
            ]
        )

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        ) # [B, T, 1]

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels # 如果输入通道数不等于输出通道数
            else nn.Identity() # 则使用1x1卷积进行通道匹配
        )

    def forward(self, x, t):
        """
        x : [ B, inp_channels, H ]
        t : [ B, embed_dim ]
        return out : [ B, out_channels, H ]
        """
        # blocks中的两个conv1沿H维度一维卷积,学习trajectory相邻时间变化规律
        # time_mlp将diffusion去噪t编码进out中
        out = self.blocks[0](x) + self.time_mlp(t) # [B, out_channels, H] + [B, out_channels, 1]
        out = self.blocks[1](out) # [B, out_channels, H]

        return out + self.residual_conv(x)


# 核心基础U-net结构
class TemporalUnet(nn.Module):
    '''
    单agent: 将带噪的trajectory windows作为输入,输出去噪后的trajectory windows
    input: [B, H, transition]
    output: [B, H, transition]
    '''
    agent_share_parameters = True # agent共享参数

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        history_horizon: int = 0,
        dim: int = 128,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        returns_condition: bool = False, # 是否使用returns条件
        env_ts_condition: bool = False, # 是否使用环境时间步条件
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        max_path_length: int = 100,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        mish = True
        act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential( # diffusion 时间步编码
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )
        embed_dim = dim # 使模型知道当前处于Diffusion去噪的时间步t

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition
        self.condition_dropout = condition_dropout
        self.history_horizon = history_horizon

        if self.returns_condition: # 嵌入returns条件
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim),
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)
            embed_dim += dim

        if self.env_ts_condition: # 嵌入环境时间步条件
            self.env_ts_mlp = nn.Sequential(
                nn.Embedding(max_path_length + 1, dim),
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            embed_dim += dim

        self.embed_dim = embed_dim

        self.downs = nn.ModuleList([]) # 下采样组合块
        self.ups = nn.ModuleList([]) # 上采样组合块
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append( # 时间维度压缩,特征通道dim增加;将时间维度信息压缩至特征通道处
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock( # 通道数最多,感受野最大,适合进行全局特征提取,学习trajectory全局时序规律
            mid_dim,
            mid_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            mish=mish,
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim,
            mid_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            mish=mish,
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2,
                            dim_in,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        ResidualTemporalBlock(
                            dim_in,
                            dim_in,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(
        self,
        x,
        time,
        returns=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout=True,
        force_dropout=False,
    ):
        """
        x : [ B, H, transition ]
        returns : [B, 1]
        """

        x = einops.rearrange(x, "b t f -> b f t") # [B, transition, H]

        t = self.time_mlp(time) # [B, embed_dim]

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns) # 将returns编码成[ B, embed_dim ]
            if use_dropout: # 按概率mask掉部分returns_embed,使模型在训练时不依赖returns条件,用于训练阶段
                mask = self.mask_dist.sample(
                    sample_shape=(returns_embed.size(0), 1)
                ).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout: # 强制丢弃条件,即无条件
                returns_embed = 0 * returns_embed
            t = torch.cat([t, returns_embed], dim=-1) # 拼接 对应前面的embed_dim += dim

        if self.env_ts_condition:
            assert env_timestep is not None
            env_timestep = env_timestep.to(dtype=torch.int64)
            env_timestep = env_timestep[:, self.history_horizon] # 取当前时间步的环境时间步,因为前history_horizon个时间步是历史轨迹,不需要嵌入环境时间步
            env_ts_embed = self.env_ts_mlp(env_timestep)
            t = torch.cat([t, env_ts_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1) # 取出encoder阶段的特征,与decoder阶段的特征拼接,用于恢复时间维度信息
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b f t -> b t f")
        return x # [B, H, transition]


class TemporalValue(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        horizon,
        transition_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            kernel_size=5,
                            embed_dim=time_dim,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            kernel_size=5,
                            embed_dim=time_dim,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 4
        mid_dim_3 = mid_dim // 16

        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim
        )
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        """
        x : [ batch x horizon x transition ]
        """

        x = einops.rearrange(x, "b h t -> b t h")

        # 屏蔽第一个条件时间步，因为该时间步不是由模型采样得到的。
        # x[:, :, 0] = 0

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out
