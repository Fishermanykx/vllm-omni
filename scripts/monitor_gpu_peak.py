#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Monitor peak GPU memory usage while running a command.

Examples:
  python scripts/monitor_gpu_peak.py -- python run_model.py
  python scripts/monitor_gpu_peak.py --scope total --interval 0.2 -- python serve.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class GpuInfo:
    index: int
    uuid: str
    name: str
    total_mib: int


def _run_nvidia_smi(query_target: str, fields: List[str]) -> List[str]:
    cmd = [
        "nvidia-smi",
        f"--query-{query_target}=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "nvidia-smi command failed")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines


def get_gpu_infos() -> Dict[str, GpuInfo]:
    lines = _run_nvidia_smi("gpu", ["index", "uuid", "name", "memory.total"])
    infos: Dict[str, GpuInfo] = {}
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        index = int(parts[0])
        uuid = parts[1]
        name = parts[2]
        total_mib = int(float(parts[3]))
        infos[uuid] = GpuInfo(index=index, uuid=uuid, name=name, total_mib=total_mib)
    return infos


def get_total_used_by_gpu_uuid() -> Dict[str, int]:
    lines = _run_nvidia_smi("gpu", ["uuid", "memory.used"])
    usage: Dict[str, int] = {}
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        usage[parts[0]] = int(float(parts[1]))
    return usage


def get_process_used_by_gpu_uuid(target_pids: Set[int]) -> Dict[str, int]:
    try:
        lines = _run_nvidia_smi("compute-apps", ["pid", "gpu_uuid", "used_memory"])
    except RuntimeError:
        # Some drivers return non-zero when no compute apps exist.
        return {}

    usage: Dict[str, int] = {}
    for line in lines:
        if "No running processes found" in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            gpu_uuid = parts[1]
            used_mib = int(float(parts[2]))
        except ValueError:
            continue
        if pid in target_pids:
            usage[gpu_uuid] = usage.get(gpu_uuid, 0) + used_mib
    return usage


def get_descendant_pids_linux(root_pid: int) -> Set[int]:
    if not os.path.isdir("/proc"):
        return {root_pid}

    ppid_map: Dict[int, List[int]] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        stat_path = os.path.join("/proc", name, "stat")
        try:
            with open(stat_path, "r", encoding="utf-8") as f:
                content = f.read()
            right = content.rfind(")")
            fields = content[right + 2 :].split()
            ppid = int(fields[1])
        except Exception:
            continue
        ppid_map.setdefault(ppid, []).append(pid)

    pids = {root_pid}
    queue = [root_pid]
    while queue:
        cur = queue.pop()
        for child in ppid_map.get(cur, []):
            if child not in pids:
                pids.add(child)
                queue.append(child)
    return pids


def get_process_tree_pids(root_pid: int) -> Set[int]:
    try:
        import psutil  # type: ignore

        proc = psutil.Process(root_pid)
        pids = {root_pid}
        for child in proc.children(recursive=True):
            pids.add(child.pid)
        return pids
    except Exception:
        return get_descendant_pids_linux(root_pid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor peak GPU memory usage while running a command.")
    parser.add_argument(
        "--scope",
        choices=["process-tree", "total"],
        default="process-tree",
        help="process-tree: only target command and descendants; total: total GPU used memory.",
    )
    parser.add_argument("--interval", type=float, default=0.2, help="Sampling interval in seconds.")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save peak result JSON.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run. Use `--` before the command.")
    args = parser.parse_args()
    if not args.command:
        parser.error("Missing command. Example: python scripts/monitor_gpu_peak.py -- python run_model.py")
    if args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main() -> int:
    args = parse_args()

    try:
        gpu_infos = get_gpu_infos()
    except Exception as e:
        print(f"[ERROR] Failed to query GPU info from nvidia-smi: {e}", file=sys.stderr)
        return 1

    if not gpu_infos:
        print("[ERROR] No NVIDIA GPU detected by nvidia-smi.", file=sys.stderr)
        return 1

    print(f"[monitor] running command: {' '.join(args.command)}")
    proc = subprocess.Popen(args.command)
    root_pid = proc.pid
    print(f"[monitor] root pid: {root_pid}, scope: {args.scope}, interval: {args.interval}s")

    peak_by_uuid: Dict[str, int] = {uuid: 0 for uuid in gpu_infos}
    peak_total_mib = 0
    samples = 0

    while True:
        running = proc.poll() is None
        if args.scope == "total":
            current_by_uuid = get_total_used_by_gpu_uuid()
        else:
            target_pids = get_process_tree_pids(root_pid)
            current_by_uuid = get_process_used_by_gpu_uuid(target_pids)

        current_total = 0
        for uuid in gpu_infos:
            cur = current_by_uuid.get(uuid, 0)
            current_total += cur
            if cur > peak_by_uuid[uuid]:
                peak_by_uuid[uuid] = cur

        if current_total > peak_total_mib:
            peak_total_mib = current_total
        samples += 1

        if not running:
            break
        time.sleep(args.interval)

    return_code = proc.returncode if proc.returncode is not None else 0

    print("\n=== GPU Peak Memory Report (MiB) ===")
    for uuid, info in sorted(gpu_infos.items(), key=lambda kv: kv[1].index):
        peak = peak_by_uuid.get(uuid, 0)
        pct = (peak / info.total_mib * 100.0) if info.total_mib > 0 else 0.0
        print(
            f"GPU {info.index}: peak={peak:>6} MiB  "
            f"total={info.total_mib:>6} MiB  usage={pct:>6.2f}%  name={info.name}"
        )
    print(f"Combined peak (sum over GPUs at sample time): {peak_total_mib} MiB")
    print(f"Samples: {samples}, command exit code: {return_code}")

    if args.output_json:
        result = {
            "scope": args.scope,
            "interval_sec": args.interval,
            "command": args.command,
            "command_exit_code": return_code,
            "samples": samples,
            "combined_peak_mib": peak_total_mib,
            "gpus": [
                {
                    "index": info.index,
                    "uuid": info.uuid,
                    "name": info.name,
                    "total_mib": info.total_mib,
                    "peak_mib": peak_by_uuid.get(uuid, 0),
                }
                for uuid, info in sorted(gpu_infos.items(), key=lambda kv: kv[1].index)
            ],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[monitor] wrote JSON report: {args.output_json}")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
