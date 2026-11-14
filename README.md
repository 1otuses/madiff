# [NeurIPS 2024] MADiff：基于扩散模型的离线多智能体学习

![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2305.17330)

这是 NeurIPS 2024 论文 “MADiff: Offline Multi-agent Learning with Diffusion Models”
的官方实现。

本仓库自写历史方案与候选设计的版本总结见 [a_old version/](<a_old version/README.md>)。

![MADiff](/assets/images/madiff.png)

## 原始 MADiff 基准结果

为简洁起见，下表省略标准差；完整结果见[论文](https://arxiv.org/abs/2305.17330)。

### 多智能体粒子环境（MPE）

以下为 [OMAR](https://arxiv.org/abs/2111.11188) 发布的 MPE 数据集结果，均值来自 5 个随机 seed。

| 数据集 | 任务 | BC | MA-ICQ | MA-TD3+BC | MA-CQL | OMAR | MADiff-D | MADiff-C* |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Expert | Spread | 35.0 | 104.0 | 108.3 | 98.2 | **114.9** | 95.0 | 116.7 |
| Md-Replay | Spread | 10.0 | 13.6 | 15.4 | 20.0 | **37.9** | 30.3 | 42.2 |
| Medium | Spread | 31.6 | 29.3 | 29.3 | 34.1 | 47.9 | **64.9** | 58.2 |
| Random | Spread | -0.5 | 6.3 | 9.8 | 24.0 | **34.4** | 6.9 | 4.3 |
| Expert | Tag | 40.0 | 113.0 | 115.2 | 93.9 | 116.2 | **120.9** | 167.6 |
| Md-Replay | Tag | 0.9 | 34.5 | 28.7 | 24.8 | 47.1 | **62.3** | 95.0 |
| Medium | Tag | 22.5 | 63.3 | 65.1 | 61.7 | 66.7 | **77.2** | 132.9 |
| Random | Tag | 1.2 | 2.2 | 5.7 | 5.0 | **11.1** | 3.2 | 10.7 |
| Expert | World | 33.0 | 109.5 | 110.3 | 71.9 | 110.4 | **122.6** | 174.0 |
| Md-Replay | World | 2.3 | 12.0 | 17.4 | 29.6 | 42.9 | **57.1** | 83.0 |
| Medium | World | 25.3 | 71.9 | 73.4 | 58.6 | 74.6 | **123.5** | 158.2 |
| Random | World | -2.4 | 1.0 | 2.8 | 0.6 | **5.9** | 2.0 | 8.1 |

### 多智能体 MuJoCo（MA-MuJoCo）

以下为 [off-the-grid MARL benchmark](https://arxiv.org/abs/2302.00521) 发布的 MA-MuJoCo 数据集结果，均值来自 5 个随机 seed。

| 数据集 | 任务 | BC | MA-TD3+BC | OMAR | MADiff-D | MADiff-C* |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Good | 2halfcheetah | 6846 | 7025 | 1434 | **8246** | 8514 |
| Medium | 2halfcheetah | 1627 | **2561** | 1892 | 2207 | 2203 |
| Poor | 2halfcheetah | 465 | 736 | 384 | **759** | 760 |
| Good | 2ant | 2697 | 2922 | 464 | **2946** | 3069 |
| Medium | 2ant | 1145 | 744 | 799 | **1211** | 1243 |
| Poor | 2ant | 954 | **1256** | 857 | 946 | 1038 |
| Good | 4ant | 2802 | 2628 | 344 | **3080** | 3068 |
| Medium | 4ant | 1617 | **1843** | 929 | 1649 | 1871 |
| Poor | 4ant | 1033 | 1075 | 518 | **1295** | 1353 |

### 星海争霸多智能体挑战（SMAC）

以下为 [off-the-grid MARL benchmark](https://arxiv.org/abs/2302.00521) 发布的 SMAC 数据集结果，均值来自 5 个随机 seed。

| 数据集 | 任务 | BC | QMIX | MA-ICQ | MA-CQL | MADT | MADiff-D | MADiff-C* |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Good | 3m | 16.0 | 13.8 | 18.8 | **19.6** | 19.1 | 19.3 | 19.9 |
| Medium | 3m | 8.2 | 17.3 | 18.1 | **18.9** | 15.8 | 17.3 | 18.1 | 
| Poor | 3m | 4.4 | 10.0 | **14.4** | 5.8 | 4.4 | 9.6 | 9.5 | 
| Good | 2s3z | 18.2 | 5.9 | **19.6** | 19.0 | 19.3 | **19.6** | 19.7 | 
| Medium | 2s3z | 12.3 | 5.2 | 17.2 | 14.3 | 15.0 | **17.4** | 17.6 | 
| Poor | 2s3z | 6.7 | 3.8 | **12.1** | 10.1 | 7.0 | 9.8 | 10.4 |
| Good | 5m6m | 16.6 | 8.0 | 16.3 | 13.8 | 16.7 | **17.8** | 18.0 | 
| Medium | 5m6m | 12.4 | 12.0 | 15.3 | 17.0 | 16.6 | **17.3** | 18.0 | 
| Poor | 5m6m | 7.5 | **10.7** | 9.4 | 10.4 | 7.8 | 8.9 | 10.3 |
| Good | 8m | 16.7 | 4.6 | **19.6** | 11.3 | 18.4 | 19.2 | 19.8 | 
| Medium | 8m | 10.7 | 13.9 | 18.6 | 16.8 | 18.5 | **18.9** | 19.4 | 
| Poor | 8m | 5.3 | 6.0 | **10.8** | 4.6 | 4.7 | 5.1 | 5.1 |

*\* MADiff-C 不用于和基线方法做公平比较，仅用于检验 MADiff-D 能否在没有全局信息时弥补协调差距。*

## 环境配置

### 安装

`requirements.txt` 按 2026-11-13 的 MADiff 环境锁定 Python 包版本，
`environment-madiff.yml` 固定 Python 3.8.20 和安装工具。以下命令在克隆后的项目根目录执行：

```bash
sudo apt-get update
sudo apt-get install build-essential git libssl-dev libcurl4-openssl-dev swig libgl1 ffmpeg
conda env create -f environment-madiff.yml
conda activate madiff
python -m pip install --no-deps -r requirements.txt
python -m pip check
```

快照使用 PyTorch 2.0.1+cu118、NumPy 1.22.4、Gym 0.10.8，具体版本、
Git 依赖和验证范围见[环境迁移说明](docs/environment.md)。
`--no-deps` 用于按完整快照安装，避免 D4RL 内部的 `mjrl@master`
覆盖已固定的提交；快照已包含当前环境的传递依赖，安装后用 `pip check` 核对。
PyTorch 的 CUDA 11.8 wheel 需要目标主机具有兼容的 NVIDIA 驱动。

### 配置 MPE

MPE 和 DDPG agent 已通过 `requirements.txt` 中的相对路径安装。
本项目使用 [OMAR](https://github.com/ling-pan/OMAR) 的 MPE 数据，
训练数据需另行下载或复制。默认目录为：

```text
/home/lotus/lotus/lhh/offline_datasets/OMAR/mpe
```

其他主机请设置 `MADIFF_OFFLINE_DATA_ROOT`，目录结构见[环境迁移说明](docs/environment.md#离线数据)。
Tag/World 在线评估还需要对应环境的 `pretrained_adv_model.pt`。

### 配置 MA-MuJoCo

`requirements.txt` 将 MAMujoco 固定到当前安装的上游提交。
它与 `third_party/multiagent_mujoco` 的 Gym 约束及环境接口存在差异；
本次锁定的是已安装包版本，MA-MuJoCo 原生编译和在线运行尚未验证。
MuJoCo 2.1 原生库及相关系统库仍需单独准备，详见[环境迁移说明](docs/environment.md#ma-mujoco)。

MA-MuJoCo 使用 [off-the-grid MARL](https://sites.google.com/view/og-marl) 数据，
训练读取转换后的 `.npy` 文件，默认位置为
`/home/lotus/lotus/lhh/offline_datasets/OG-MARL-MADiff/mamujoco`。

### 配置 SMAC

SMAC Python 包已由 `requirements.txt` 固定到当前提交。
StarCraft II 游戏和地图需要另行安装：

```bash
bash scripts/smac.sh
export SC2PATH="$HOME/StarCraftII"
```

当前主机安装的游戏目录包含 `Versions/Base75689`。
SMAC 的转换后数据默认位于
`/home/lotus/lotus/lhh/offline_datasets/OG-MARL-MADiff/smac/<map>/<Good|Medium|Poor>/`，
需要单独下载或复制到目标机器。

### OG-MARL 数据转换

MADiff 训练直接读取已经转换的 NumPy 数据，不依赖 Flashbax。
当前 OG-MARL Vault 转换需要独立的 Python 3.10 环境，流程见
[Vault 数据转换说明](my_version/docs/vault_dataset_conversion.md)。
历史 TFRecord 转换入口为 `scripts/transform_og_marl_dataset.py`，
其 TensorFlow 转换依赖也应在独立环境中准备，以保留训练环境的版本快照。

## 训练与评估

本项目默认从 `/home/lotus/lotus/lhh/offline_datasets` 读取数据。可通过
`MADIFF_OFFLINE_DATA_ROOT` 覆盖根目录，或通过各数据源专用环境变量覆盖。
训练产物、checkpoint、TensorBoard event 和评估结果写入项目的 `logs/` 目录。

训练进度条默认显示 dataset、seed、epoch、global step 和最新 loss；可在实验
`constants` 中设置 `show_progress: false` 关闭。

原始 MADiff 训练命令：

```bash
# 多智能体粒子环境
python run_experiment.py -e exp_specs/mpe/<task>/attn_<dataset>.yaml  # CTCE
python run_experiment.py -e exp_specs/mpe/<task>/ctde_<dataset>.yaml  # CTDE
# MA-MuJoCo
python run_experiment.py -e exp_specs/mamujoco/<task>/attn_<dataset>_history.yaml  # CTCE
python run_experiment.py -e exp_specs/mamujoco/<task>/ctde_<dataset>_history.yaml  # CTDE
# SMAC
python run_experiment.py -e exp_specs/smac/<map>/attn_<dataset>_history.yaml  # CTCE
python run_experiment.py -e exp_specs/smac/<map>/ctde_<dataset>_history.yaml  # CTDE
```

评估前在 `exp_specs/eval_inv.yaml` 中设置待评估的 `log_dir`，然后运行：
```bash
python run_experiment.py -e exp_specs/eval_inv.yaml
```

当前实验方案见 [AgentFormer 时空注意力](agentformer_to_madiff_plan.md) 和 [非注意力信息传递替代方案](MADiff_非注意力消息传递替代方案_v1.md)。历史想法的核心记录保存在 [a_old version](<a_old version/README.md>)。

## 引用

```
@article{zhu2023madiff,
  title={MADiff: Offline Multi-agent Learning with Diffusion Models},
  author={Zhu, Zhengbang and Liu, Minghuan and Mao, Liyuan and Kang, Bingyi and Xu, Minkai and Yu, Yong and Ermon, Stefano and Zhang, Weinan},
  journal={arXiv preprint arXiv:2305.17330},
  year={2023}
}
```

## 致谢

本代码库基于 [decision-diffuser](https://github.com/anuragajay/decision-diffuser) 和 [ILSwiss](https://github.com/Ericonaldo/ILSwiss) 构建。
