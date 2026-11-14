#!/usr/bin/env python3
"""增量读取训练日志, 只在开始、每 1 万步和保存/评估等阶段追加显示."""

import argparse
import csv
import json
import re
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAIN_STEP = re.compile(r"^(\d+):\s*(\S+)\s*\|")


def read_events(log_path, total_steps, state):
    """按字节偏移读取完整新行, 保留半行供下一次读取, 去重训练里程碑."""
    try:
        with Path(log_path).open("rb") as stream:
            stream.seek(0, 2)
            if stream.tell() < state.get("offset", 0):
                state.clear()
            stream.seek(state.get("offset", 0))
            data = stream.read()
    except OSError:
        return []
    complete = data.rfind(b"\n") + 1
    state["offset"] = state.get("offset", 0) + complete
    events = []
    for raw in data[:complete].decode(errors="replace").splitlines():
        line = ANSI_ESCAPE.sub("", raw).strip()
        resumed = re.search(r"Loaded checkpoint .*\(step (\d+)\)", line)
        if resumed:
            state["step"] = int(resumed.group(1))
            events.append(f"[加载 checkpoint] step={state['step']}/{total_steps}")
        if re.match(r"^Epoch 1 / \d+", line) and not state.get("started"):
            state["started"] = True
            events.append(f"[开始训练] step={state.get('step', 0)}/{total_steps}")
        match = TRAIN_STEP.match(line)
        if match:
            step = int(match.group(1))
            if step > state.get("step", 0) and step % 10000 == 0:
                events.append(f"[训练进度] {step}/{total_steps} ({step / total_steps:.1%}) | 日志 loss={match.group(2)}")
                state["step"] = step
        saved = re.search(r"Saved model to .*state_(\d+)\.pt", line)
        if saved:
            events.append(f"[保存模型] step={saved.group(1)}")
        evaluation = re.search(r"Eval requested at step (\d+)", line)
        if evaluation:
            events.append(f"[提交评估] step={evaluation.group(1)}")
        if "Traceback (most recent call last)" in line:
            events.append("[报错] 等待调度清理")
    return events


def render(state_path, completed, failed, total):
    with Path(state_path).open(newline="") as stream:
        rows = [row for row in csv.reader(stream, delimiter="\t") if row]
    cache_path = Path(state_path).with_suffix(".events.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    logs = cache.setdefault("logs", {})
    lines = []
    for gpu, slot, label, log_path, total_steps in rows:
        for event in read_events(log_path, int(total_steps), logs.setdefault(log_path, {})):
            lines.append(f"{event} | GPU {gpu}/{slot} | {label}")
    summary = [completed, failed]
    if cache.get("summary") != summary:
        lines.append(f"训练统计: 完成 {completed}/{total} | 运行 {len(rows)} | 失败 {failed}")
        cache["summary"] = summary
    cache_path.write_text(json.dumps(cache))
    if lines:
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, help="任务状态 TSV 路径")
    parser.add_argument("--completed", required=True, type=int)
    parser.add_argument("--failed", required=True, type=int)
    parser.add_argument("--total", required=True, type=int)
    args = parser.parse_args()
    render(args.state, args.completed, args.failed, args.total)
