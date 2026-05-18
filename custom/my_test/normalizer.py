"""
数据归一化工具 (对齐 MADiff 的 DatasetNormalizer)
==================================================
为 my_test 添加与 MADiff 相同的归一化/反归一化能力。
"""

from typing import Optional, List
import numpy as np


class GaussianNormalizer:
    """
    零均值单位方差归一化。
    与 diffuser/datasets/normalization.py 中的 GaussianNormalizer 一致。
    """
    def __init__(self, X: np.ndarray):
        X = X.astype(np.float32)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class LimitsNormalizer:
    """
    映射到 [-1, 1]。
    与 diffuser/datasets/normalization.py 中的 LimitsNormalizer 一致。
    """
    def __init__(self, X: np.ndarray):
        X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = (x - self.mins) / (self.maxs - self.mins + 1e-8)
        x = 2 * x - 1  # [0,1] → [-1,1]
        return x

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -1.0, 1.0)
        x = (x + 1) / 2.0  # [-1,1] → [0,1]
        return x * (self.maxs - self.mins) + self.mins


class DatasetNormalizer:
    """
    多字段归一化器，支持 per-agent 独立归一化。

    与 diffuser/datasets/normalization.py 的 DatasetNormalizer 接口一致。
    """
    def __init__(
        self,
        dataset: dict,
        normalizer: str = "GaussianNormalizer",
        keys: Optional[List[str]] = None,
        agent_share: bool = True,
    ):
        """
        Args:
            dataset: {'obs': [N, A, D], 'acs': [N, A, D], ...}
            normalizer: "GaussianNormalizer" | "LimitsNormalizer"
            keys: 要归一化的字段名列表 (默认 ['obs', 'acs'])
            agent_share: 是否所有 agent 共享归一化参数
        """
        if keys is None:
            keys = ["obs", "acs"]
        self.keys = keys
        self.agent_share = agent_share

        normalizer_cls = eval(normalizer)

        self.normalizers = {}
        for key in keys:
            data = dataset[key]  # [N, A, D]
            if agent_share:
                # 将所有 agent 的数据展平后拟合一个归一化器
                flat = data.reshape(-1, data.shape[-1])
                self.normalizers[key] = normalizer_cls(flat)
            else:
                # 每个 agent 独立拟合
                self.normalizers[key] = [
                    normalizer_cls(data[:, i]) for i in range(data.shape[1])
                ]

    def normalize(self, x: np.ndarray, key: str) -> np.ndarray:
        if key not in self.normalizers:
            return x
        n = self.normalizers[key]
        if self.agent_share:
            return n.normalize(x)
        else:
            return np.stack(
                [n[i].normalize(x[..., i, :]) for i in range(x.shape[-2])],
                axis=-2,
            )

    def unnormalize(self, x: np.ndarray, key: str) -> np.ndarray:
        if key not in self.normalizers:
            return x
        n = self.normalizers[key]
        if self.agent_share:
            return n.unnormalize(x)
        else:
            return np.stack(
                [n[i].unnormalize(x[..., i, :]) for i in range(x.shape[-2])],
                axis=-2,
            )
