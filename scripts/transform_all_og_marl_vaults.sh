#!/usr/bin/env bash

set -euo pipefail

# 无论从哪个目录调用，都先切换到 MADiff 项目根目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 默认使用当前环境的 Python；可通过 PYTHON_BIN 指定独立解释器。
PYTHON_BIN="${PYTHON_BIN:-python}"
VAULT_ROOT="/home/lotus/lotus/lhh/offline_datasets/Vaults"
OUTPUT_ROOT="/home/lotus/lotus/lhh/offline_datasets/OG-MARL-Vault"

SMAC_V1_SCENARIOS=(
  "3m"
  "8m"
  "2s3z"
  "5m_vs_6m"
  "3s5z_vs_3s6z"
)

SMAC_V2_SCENARIOS=(
  "zerg_5_vs_5"
  "terran_5_vs_5"
  "terran_10_vs_10"
)

GYMNASIUM_MAMUJOCO_SCENARIOS=(
  "2ant"
  "2halfcheetah"
  "4ant"
)

MAMUJOCO_SCENARIOS=(
  "2ant"
  "2halfcheetah"
)

# 根据环境名称返回对应的 MADiff 输出目录名称。
madiff_env_name() {
  case "$1" in
    smac_v1) echo "smac" ;;
    smac_v2) echo "smacv2" ;;
    gymnasium_mamujoco) echo "mamujoco" ;;
    mamujoco) echo "mamujoco" ;;
    *)
      echo "不支持的环境：$1" >&2
      return 1
      ;;
  esac
}

# 每个场景的 UID 不完全相同，直接从已下载的 Vault 目录读取。
collect_jobs() {
  local env_name="$1"
  shift
  local scenario
  local vault_dir
  local uid_dir

  for scenario in "$@"; do
    vault_dir="${VAULT_ROOT}/og_marl/${env_name}/${scenario}.vlt"
    if [[ ! -d "${vault_dir}" ]]; then
      echo "找不到 Vault,跳过:${vault_dir}" >&2
      continue
    fi

    for uid_dir in "${vault_dir}"/*/; do
      [[ -d "${uid_dir}" ]] || continue
      JOB_ENVS+=("${env_name}")
      JOB_SCENARIOS+=("${scenario}")
      JOB_UIDS+=("$(basename "${uid_dir}")")
    done
  done
}

JOB_ENVS=()
JOB_SCENARIOS=()
JOB_UIDS=()
collect_jobs "smac_v1" "${SMAC_V1_SCENARIOS[@]}"
collect_jobs "smac_v2" "${SMAC_V2_SCENARIOS[@]}"
collect_jobs "gymnasium_mamujoco" "${GYMNASIUM_MAMUJOCO_SCENARIOS[@]}"
collect_jobs "mamujoco" "${MAMUJOCO_SCENARIOS[@]}"

total=${#JOB_ENVS[@]}
if ((total == 0)); then
  echo "没有找到可转换的 Vault 数据。" >&2
  exit 1
fi

for ((index = 0; index < total; index++)); do
    current=$((index + 1))
    env_name="${JOB_ENVS[index]}"
    scenario="${JOB_SCENARIOS[index]}"
    uid="${JOB_UIDS[index]}"
    output_env_name="$(madiff_env_name "${env_name}")"
    output_dir="${OUTPUT_ROOT}/${output_env_name}/${scenario}/${uid}"

    # manifest 存在表示该组合已经完整转换，允许脚本安全续跑。
    if [[ -f "${output_dir}/manifest.json" ]]; then
      echo "[${current}/${total}] 跳过已完成数据：${env_name}/${scenario}/${uid}"
      continue
    fi

    echo "[${current}/${total}] 开始转换：${env_name}/${scenario}/${uid}"
    "${PYTHON_BIN}" scripts/transform_og_marl_vault.py \
      --vault-root "${VAULT_ROOT}" \
      --env "${env_name}" \
      --scenario "${scenario}" \
      --uid "${uid}" \
      --output-root "${OUTPUT_ROOT}" \
      --drop-incomplete-tail
    echo "[${current}/${total}] 转换完成：${env_name}/${scenario}/${uid}"
done

echo "全部 ${total} 组 OG-MARL Vault 数据处理完成。"
