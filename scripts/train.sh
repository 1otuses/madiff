#!/usr/bin/env bash
# 功能: 每卡并行 3 组; 只显示开始训练、每 1 万步及保存/评估/结束等阶段.
# 默认: legacy + agent_aware, seed=100/200/300, 共 90 组; 原始 YAML 保持不变.
# 用法: bash scripts/train.sh [--dry-run] [--jobs-per-gpu 3] [--attn-type both|legacy|agent_aware]
set -euo pipefail
set +m

attn_type=both
dry_run=false
jobs_per_gpu=3
while (($#)); do
    case "$1" in
        --attn-type)
            attn_type="${2:?--attn-type 需要 both, legacy 或 agent_aware}"
            shift 2
            ;;
        --dry-run) dry_run=true; shift ;;
        --jobs-per-gpu)
            jobs_per_gpu="${2:?--jobs-per-gpu 需要正整数}"
            shift 2
            ;;
        -h|--help)
            echo "用法: bash scripts/train.sh [--dry-run] [--jobs-per-gpu 3] [--attn-type both|legacy|agent_aware]"
            exit 0
            ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done
case "$attn_type" in
    both|legacy|agent_aware) ;;
    *) echo "无效 attn_type: $attn_type" >&2; exit 2 ;;
esac
if ! [[ "$jobs_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
    echo "--jobs-per-gpu 必须为正整数" >&2
    exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
gpus=(0 1 2 3)
slot_count=$((${#gpus[@]} * jobs_per_gpu))
pids=()
labels=()
consoles=()
budgets=()
for ((slot=0; slot<slot_count; slot++)); do
    pids[slot]=""
    labels[slot]=""
    consoles[slot]=""
    budgets[slot]=0
done
mkdir -p exp_specs/generated transh
work_dir="$(mktemp -d exp_specs/generated/train_queue.XXXXXX)"
console_dir="transh/$(basename -- "$work_dir")"

stop_group() {
    local pid="$1"
    kill -TERM -- "-$pid" "$pid" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$pid" "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

cleanup() {
    trap - EXIT
    trap '' INT TERM
    local pid
    for pid in "${pids[@]}"; do
        if [[ -n "$pid" ]]; then
            stop_group "$pid"
        fi
    done
    rm -rf -- "$work_dir"
    rmdir -- "$console_dir" exp_specs/generated 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$console_dir"

# 拆分为单 seed、单方法、单 worker; 关闭连续进度条, 保留阶段日志及训练参数.
python - "$work_dir" "$attn_type" > "$work_dir/tasks.tsv" <<'PY'
import copy
import sys
from pathlib import Path

import yaml

work_dir = Path(sys.argv[1])
methods = ["legacy", "agent_aware"] if sys.argv[2] == "both" else [sys.argv[2]]
sources = [Path("exp_specs/mpe/spread/agent_aware_exp.yaml")]
sources += [Path(f"exp_specs/mpe/{env}/agent_aware_{quality}.yaml")
            for env in ["tag", "world"]
            for quality in ["exp", "md", "mdreplay", "random"]]
sources += [Path(f"exp_specs/smac/{env}/agent_aware_{quality}_history.yaml")
            for env in ["3m", "2s3z"]
            for quality in ["good", "medium", "poor"]]

index = 0
for source in sources:
    original = yaml.safe_load(source.read_text())
    for seed in [100, 200, 300]:
        for method in methods:
            spec = copy.deepcopy(original)
            spec["meta_data"]["num_workers"] = 1
            spec["variables"]["seed"] = [seed]
            spec["variables"]["attn_type"] = [method]
            spec["constants"]["show_progress"] = False
            label = f'{spec["variables"]["dataset"][0]} / {method} / seed={seed}'
            path = work_dir / f"{index:03d}.yaml"
            header = (f"# 功能: 总调度中的单组训练, 来源 {source}.\n"
                      f"# 修改: seed={seed}, attn_type={method}, num_workers=1, show_progress=False; 其余参数沿用源配置.\n")
            path.write_text(header + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
            print(f'{path}\t{label}\t{spec["constants"]["n_train_steps"]}')
            index += 1
PY

mapfile -t tasks < "$work_dir/tasks.tsv"
total=${#tasks[@]}
printf '共 %d 组训练; GPU: %s; 每卡并行 %d 组, 总并发上限 %d.\n' "$total" "${gpus[*]}" "$jobs_per_gpu" "$slot_count"
if "$dry_run"; then
    echo "预览采用轮转分配; 实际运行时由先空闲的 GPU 接续任务."
    for i in "${!tasks[@]}"; do
        IFS=$'\t' read -r spec label budget <<< "${tasks[$i]}"
        printf '[%d/%d] %s\n' "$((i + 1))" "$total" "$label"
        printf '  python run_experiment.py -e %q -g %s\n' "$spec" "${gpus[$((i / jobs_per_gpu % ${#gpus[@]}))]}"
    done
    exit 0
fi

next=0
finished=0
failed_jobs=()
next_scan=0

report_progress() {
    local slot
    {
        for ((slot=0; slot<slot_count; slot++)); do
            if [[ -n "${pids[$slot]}" ]]; then
                printf '%s\t%s\t%s\t%s\t%s\n' "${gpus[$((slot / jobs_per_gpu))]}" "$((slot % jobs_per_gpu + 1))" "${labels[$slot]}" "${consoles[$slot]}" "${budgets[$slot]}"
            fi
        done
    } > "$work_dir/status.tsv"
    python scripts/train_progress.py --state "$work_dir/status.tsv" --completed "$finished" --failed "${#failed_jobs[@]}" --total "$total"
}

while ((finished < total)); do
    for ((slot=0; slot<slot_count; slot++)); do
        gpu="${gpus[$((slot / jobs_per_gpu))]}"
        gpu_slot=$((slot % jobs_per_gpu + 1))
        pid="${pids[$slot]}"
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            # 先读取最后的阶段日志, 再清理成功任务的临时输出.
            report_progress
            if wait "$pid"; then
                printf '[完成 GPU %s/%s] %s\n' "$gpu" "$gpu_slot" "${labels[$slot]}"
                rm -f -- "${consoles[$slot]}"
            else
                stop_group "$pid"
                failed_jobs+=("${labels[$slot]}")
                printf '[失败 GPU %s/%s] %s; 输出: %s\n' "$gpu" "$gpu_slot" "${labels[$slot]}" "${consoles[$slot]}" >&2
                tail -n 20 -- "${consoles[$slot]}" >&2
            fi
            pids[$slot]=""
            finished=$((finished + 1))
            printf '进度: %d/%d, 失败: %d\n' "$finished" "$total" "${#failed_jobs[@]}"
        fi
        if [[ -z "${pids[$slot]}" ]] && ((next < total)); then
            IFS=$'\t' read -r spec label budget <<< "${tasks[$next]}"
            consoles[$slot]="$console_dir/$(printf '%03d' "$next").log"
            printf '[启动 GPU %s/%s] %s\n' "$gpu" "$gpu_slot" "$label"
            printf '  python run_experiment.py -e %q -g %s\n' "$spec" "$gpu"
            # 独立进程组使中断时能一起停止 launcher、训练和评估子进程.
            PYTHONUNBUFFERED=1 setsid python run_experiment.py -e "$spec" -g "$gpu" > "${consoles[$slot]}" 2>&1 &
            pids[$slot]=$!
            labels[$slot]="$label"
            budgets[$slot]="$budget"
            next=$((next + 1))
        fi
    done
    # 后台检查新事件; 没有新阶段时不输出, 不刷新或清空终端.
    if ((SECONDS >= next_scan || finished == total)); then
        report_progress
        next_scan=$((SECONDS + 5))
    fi
    if ((finished < total)); then
        sleep 1
    fi
done

printf '全部任务结束: 成功 %d, 失败 %d.\n' "$((total - ${#failed_jobs[@]}))" "${#failed_jobs[@]}"
if ((${#failed_jobs[@]})); then
    printf '失败任务: %s\n' "${failed_jobs[@]}" >&2
    echo "失败任务的控制台输出保留在 $console_dir" >&2
    exit 1
fi
