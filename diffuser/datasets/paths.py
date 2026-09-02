import os
from pathlib import Path
from typing import Union


# 获取保存数据集的地址根目录
DEFAULT_OFFLINE_DATA_ROOT = Path("/home/lotus/lotus/lhh/offline_datasets")
DATASET_SOURCES = { # 不同数据集来源
    "mpe": "OMAR",
    "mamujoco": "OG-MARL-MADiff",
    "smac": "OG-MARL-MADiff",
    "smacv2": "OG-MARL-MADiff",
}


def get_data_root() -> Path:
    """Return the shared offline dataset root."""

    return Path(
        os.environ.get(
            "MADIFF_OFFLINE_DATA_ROOT",
            os.environ.get("MADIFF_DATA_ROOT", DEFAULT_OFFLINE_DATA_ROOT),
        )
    ).expanduser()


def get_dataset_path(*parts: Union[str, os.PathLike]) -> str:
    if not parts:
        return os.fspath(get_data_root())

    dataset_type = os.fspath(parts[0])
    source_name = DATASET_SOURCES.get(dataset_type)
    if source_name is None:
        return os.fspath(get_data_root().joinpath(*parts))

    source_env = f"MADIFF_{source_name.replace('-', '_')}_DATA_ROOT"
    source_root = Path(
        os.environ.get(source_env, get_data_root().joinpath(source_name))
    ).expanduser()
    return os.fspath(source_root.joinpath(*parts))
