import os

# 在 NumPy/PyTorch 导入前限制各计算库的默认 CPU 线程数, 外部设置优先。
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "2")

# NOTE: Do NOT delete this. It is used to register environments like `HalfCheetahFullObsEnv` to gym
from . import environments
