# 转换 OG-MARL Vault 数据集

MADiff 使用完整 episode 组成的扁平 NumPy 数组和 `path_lengths.npy`
进行训练，而当前 OG-MARL 数据集采用 Flashbax Vault 格式。推荐将 Vault
保留为权威数据源，再生成供 MADiff 使用的 NumPy 缓存。

## 为什么使用独立环境

- MADiff 的论文复现环境使用 Python 3.8。
- Flashbax 0.1.x 要求 Python 3.9 或更高版本，当前 OG-MARL 要求
  Python 3.10 或更高版本。
- Vault 转换器只在读取 `.vlt` 时依赖 Flashbax；转换完成后，MADiff
  训练仍然只读取 NumPy 文件。
- `third_party/og-marl` 继续服务于历史 TFRecord 数据，不应由当前
  OG-MARL 覆盖。

因此，Vault 转换和 MADiff 训练应使用两个隔离的 Python 环境。

## 创建 Vault 转换环境

仓库提供了固定版本的独立环境定义：

```bash
conda env create -f environment-vault.yml
conda activate madiff-vault
python -c "from flashbax.vault import Vault; print(Vault)"
```

如果机器上已经有安装了 Flashbax 的 Python 3.10 环境，可以直接使用其
Python 解释器，无需重复创建环境。例如当前机器已有：

```bash
/home/lotus/miniconda3/envs/og-marl/bin/python -c \
  "from flashbax.vault import Vault; print(Vault)"
```

## 转换数据

推荐直接指定独立环境中的 Python，这样不需要切换当前 shell：

```bash
/home/lotus/miniconda3/envs/og-marl/bin/python \
  scripts/transform_og_marl_vault.py \
  --vault-root /path/to/vaults \
  --source og_marl \
  --env smac_v1 \
  --scenario 3m \
  --uid Good \
  --output-root /home/lotus/lotus/lhh/offline_datasets/OG-MARL
```

上述命令读取：

```text
/path/to/vaults/og_marl/smac_v1/3m.vlt/Good
```

并输出到：

```text
/home/lotus/lotus/lhh/offline_datasets/OG-MARL/smac/3m/Good
```

环境名称映射如下：

| OG-MARL | MADiff |
| --- | --- |
| `smac_v1` | `smac` |
| `smac_v2` | `smacv2` |
| `mamujoco` | `mamujoco` |
| `gymnasium_mamujoco` | `gymnasium_mamujoco` |

其中 `mamujoco` 表示基于旧版 Multi-Agent MuJoCo 的历史数据，
`gymnasium_mamujoco` 表示基于 Gymnasium 的新版数据。两者使用独立输出目录，
避免同名 scenario 和 UID 相互覆盖。

转换器会：

- 分别处理每个 Vault batch lane；
- 使用 `terminals OR truncations` 切分 episode，并要求所有智能体同步结束；
- 拒绝包含不完整尾部 episode 的数据；
- 为 observation 添加 one-hot 智能体 ID，与 MADiff 评估环境保持一致；
- 将 `infos.legals` 和 `infos.state` 映射为 `legals.npy` 和
  `states.npy`；
- 在 `manifest.json` 中记录源数据形状、输出形状和转换选项。

如果源 observation 已包含 one-hot 智能体 ID，使用
`--no-add-agent-id`。默认不覆盖已有输出；确认需要替换时显式传入
`--overwrite`。

## 验证结果

转换完成后，首先检查输出目录中的 `manifest.json`。对于 SMAC
`3m/Good`，预期为：

```text
n_transitions: 996366
n_episodes: 43559
obs.npy: (996366, 3, 33)
```

然后使用 MADiff Python 3.8 环境加载和训练：

```bash
conda activate madiff
python run_experiment.py \
  -e exp_specs/smac/3m/mad_smac_3m_attn_good_history.yaml
```

不要删除历史 `.npy` 数据。新转换结果应先写入独立目录，通过统计和
smoke test 后再用于正式实验。

## 两条数据链路

```text
历史 TFRecord
  → Python 3.8 + third_party/og-marl
  → scripts/transform_og_marl_dataset.py
  → MADiff NumPy 缓存

当前 Vault
  → Python 3.10 + Flashbax
  → scripts/transform_og_marl_vault.py
  → MADiff NumPy 缓存
```
