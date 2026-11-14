# MADiff 环境迁移

版本快照日期：2026-11-13。目标平台为 Linux x86_64，训练环境为 Python 3.8.20。
本次记录当前已安装的 Python 包及 Git 来源；本机 `pip check` 通过，尚未在另一台干净主机完整复装验证。

已逐项核对版本和 Git 提交，167 项 PyPI/CUDA 包通过当前环境的离线 pip dry-run，未执行安装。
Python 标准库已提供 `argparse`，快照沿用 `pip freeze` 的处理，不重复安装其历史回溯包。

## 安装

在克隆后的项目根目录执行：

```bash
sudo apt-get update
sudo apt-get install build-essential git libssl-dev libcurl4-openssl-dev swig libgl1 ffmpeg
conda env create -f environment-madiff.yml
conda activate madiff
python -m pip install --no-deps -r requirements.txt
python -m pip check
```

`environment-madiff.yml` 固定 Python、pip、setuptools 和 wheel；
`requirements.txt` 记录当前 175 个 Python 包，包括传递依赖。
使用 `--no-deps` 按快照安装，可以避免 D4RL 的 `mjrl@master` 声明
与已固定的 mjrl Git 提交发生 URL 约束冲突。安装结束后由 `pip check` 检查包之间的依赖关系。

快照不包含另一台机器的 Conda 构建目录或绝对 editable 路径。
本地 editable 路径均相对于项目根目录，Git 源码依赖固定到完整提交号。

## 关键版本

| 组件 | 固定版本 |
|---|---|
| Python | 3.8.20 |
| pip / setuptools / wheel | 24.2 / 75.1.0 / 0.44.0 |
| PyTorch / torchvision / torchaudio | 2.0.1+cu118 / 0.15.2+cu118 / 2.0.2+cu118 |
| NumPy / SciPy | 1.22.4 / 1.10.1 |
| Gym | 0.10.8 |
| diffusers / huggingface-hub | 0.24.0 / 0.25.2 |
| protobuf | 3.20.3 |
| Cython | 3.2.8 |
| TensorBoard | 2.0.2 |

PyTorch wheel 从 CUDA 11.8 索引获取。目标主机仍需兼容的 NVIDIA 驱动；
安装 wheel 不要求将主机 CUDA toolkit 升降级到相同版本。
`pip check` 只检查依赖元数据，CUDA 计算是否可用还需在目标 GPU 上执行实际前向和反向验证。

## 源码依赖

以下本地包随克隆源码安装：

```text
-e .
-e ./third_party/multiagent-particle-envs
-e ./third_party/ddpg-agent
-e ./third_party/og-marl
```

以下包保留当前安装的 Git 来源，完整 URL 已写入 `requirements.txt`：

| 包 | 提交 |
|---|---|
| SMAC | `d6aab33f76abc3849c50463a8592a84f59a5ef84` |
| MAMujoco | `b212ddd74b258e7cea006ff1d642b5ffada4b99d` |
| mjrl | `3871d93763d3b49c4741e6daeaebbc605fe140dc` |
| D4RL | `e840e80ec4ed51bbb0e557e466198b5d1ae29a15` |

## SMAC

StarCraft II 游戏和地图不属于 pip 依赖，需要额外安装：

```bash
bash scripts/smac.sh
export SC2PATH="$HOME/StarCraftII"
```

当前主机游戏目录包含 `Versions/Base75689`。迁移时核对游戏版本、地图和 `SC2PATH`，
再验证一次真实环境 reset/step。仅能导入 SMAC 包并不表示游戏已经启动成功。

## MA-MuJoCo

`mujoco==2.3.0` 和 `mujoco-py==2.1.2.14` 是两个 Python 包。
后者还需要独立的 MuJoCo 2.1 原生库；当前主机使用 `~/.mujoco/mujoco210`，
并将其 `bin` 目录加入 `LD_LIBRARY_PATH`。目标机器还需相应的 OpenGL/OSMesa、GLEW、编译工具等系统依赖。

当前 Cython 为 3.2.8，而 mujoco-py 的旧回调源码尚未完成该组合下的 Linux 首次编译验证。
当前已安装的上游 MAMujoco 与本地 `third_party/multiagent_mujoco` 在 Gym 约束及 `step` 返回接口上也存在差异。
因此这里提供的是 Python 包版本快照，MA-MuJoCo 的编译与运行适配需要单独验证。

## 离线数据

训练数据和 Tag/World 的预训练对手权重需另行下载或复制。使用统一的数据根目录，例如：

```bash
export MADIFF_OFFLINE_DATA_ROOT=/path/to/offline_datasets
```

当前训练代码期望的结构为：

```text
offline_datasets/
├── OMAR/mpe/<scenario>/...
└── OG-MARL-MADiff/
    ├── smac/<map>/<Good|Medium|Poor>/*.npy
    └── mamujoco/<map>/<Good|Medium|Poor>/*.npy
```

Tag/World 评估会读取 `OMAR/mpe/<scenario>/pretrained_adv_model.pt`。
还可分别设置 `MADIFF_OMAR_DATA_ROOT` 或 `MADIFF_OG_MARL_MADiff_DATA_ROOT`，
它们分别指向 `OMAR` 和 `OG-MARL-MADiff` 目录本身，变量名按代码中的大小写精确匹配。

当前 Vault 数据转换使用独立的 Python 3.10 环境，参见
[数据转换说明](../my_version/docs/vault_dataset_conversion.md)。
`autoresearch/` 的 GPT 示例也使用自己的环境；这些依赖均未并入本训练快照。

## 既有命令行问题

当前 `run_experiment.py --help` 会触发 ml-logger / params-proto 对日期格式 `%H` 的帮助文本格式化错误。
这是另一个命令行问题，本次未修改；环境版本核对使用包和项目模块导入，未据此宣称训练或新机复装已经通过。
