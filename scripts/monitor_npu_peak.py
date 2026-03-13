#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Monitor peak Ascend NPU memory usage while running a command.

Examples:
  python scripts/monitor_npu_peak.py -- python run_model.py
  python scripts/monitor_npu_peak.py --scope total --interval 0.2 -- python serve.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


_INT_RE = re.compile(r"\d+")
_MEM_PAIR_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


@dataclass
class NpuInfo:
    npu_id: int
    chip_id: int
    name: str
    bus_id: str
    total_mb: int


def _run_npu_smi_info() -> str:
    proc = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or "npu-smi info failed"
        raise RuntimeError(err)
    return proc.stdout


def _parse_bar_line(line: str) -> List[str]:
    line = line.strip()
    if not line.startswith("|"):
        return []
    return [part.strip() for part in line.split("|")[1:-1]]


def _first_int(text: str) -> Optional[int]:
    match = _INT_RE.search(text)
    return int(match.group(0)) if match else None


def _all_ints(text: str) -> List[int]:
    return [int(x) for x in _INT_RE.findall(text)]


def parse_npu_infos_and_total_usage(snapshot: str) -> Tuple[Dict[str, NpuInfo], Dict[str, int]]:
    infos: Dict[str, NpuInfo] = {}
    usage: Dict[str, int] = {}

    pending_npu_id: Optional[int] = None
    pending_name = ""

    for raw_line in snapshot.splitlines():
        parts = _parse_bar_line(raw_line)
        if len(parts) != 3:
            continue

        if "NPU" in parts[0] and "Name" in parts[0]:
            continue
        if "Chip" in parts[0] and "Bus-Id" in parts[1]:
            continue
        if "Process" in parts[1] or "Process" in parts[2]:
            continue

        first = parts[0]
        second = parts[1]
        third = parts[2]

        tokens = first.split()
        if len(tokens) >= 2:
            maybe_npu = _first_int(tokens[0])
            if maybe_npu is not None and any(c.isalpha() for c in " ".join(tokens[1:])):
                pending_npu_id = maybe_npu
                pending_name = " ".join(tokens[1:])
                continue

        if pending_npu_id is None:
            continue

        if not tokens:
            continue

        maybe_chip = _first_int(tokens[0])
        if maybe_chip is None:
            continue

        mem_pairs = _MEM_PAIR_RE.findall(third)
        if not mem_pairs:
            continue

        used_mb, total_mb = (int(mem_pairs[-1][0]), int(mem_pairs[-1][1]))
        key = f"{pending_npu_id}:{maybe_chip}"

        infos[key] = NpuInfo(
            npu_id=pending_npu_id,
            chip_id=maybe_chip,
            name=pending_name,
            bus_id=second,
            total_mb=total_mb,
        )
        usage[key] = used_mb

        pending_npu_id = None
        pending_name = ""

    return infos, usage


def parse_process_usage_by_npu_chip(snapshot: str, target_pids: Set[int]) -> Dict[str, int]:
    usage: Dict[str, int] = {}

    for raw_line in snapshot.splitlines():
        parts = _parse_bar_line(raw_line)
        if len(parts) < 4:
            continue

        if "Process id" in parts[1]:
            continue
        if "No running processes found" in parts[0]:
            continue

        pid = _first_int(parts[1])
        if pid is None or pid not in target_pids:
            continue

        ids = _all_ints(parts[0])
        if len(ids) >= 2:
            npu_id, chip_id = ids[0], ids[1]
        elif len(ids) == 1:
            npu_id, chip_id = ids[0], 0
        else:
            continue

        used_mb = _first_int(parts[-1])
        if used_mb is None:
            continue

        key = f"{npu_id}:{chip_id}"
        usage[key] = usage.get(key, 0) + used_mb

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
    parser = argparse.ArgumentParser(description="Monitor peak NPU memory usage while running a command.")
    parser.add_argument(
        "--scope",
        choices=["process-tree", "total"],
        default="process-tree",
        help="process-tree: only target command and descendants; total: total NPU used memory.",
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
        parser.error("Missing command. Example: python scripts/monitor_npu_peak.py -- python run_model.py")
    if args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main() -> int:
    args = parse_args()

    try:
        first_snapshot = _run_npu_smi_info()
        npu_infos, _ = parse_npu_infos_and_total_usage(first_snapshot)
    except Exception as e:
        print(f"[ERROR] Failed to query NPU info from npu-smi: {e}", file=sys.stderr)
        return 1

    if not npu_infos:
        print("[ERROR] No NPU memory info found in npu-smi output.", file=sys.stderr)
        return 1

    print(f"[monitor] running command: {' '.join(args.command)}")
    proc = subprocess.Popen(args.command)
    root_pid = proc.pid
    print(f"[monitor] root pid: {root_pid}, scope: {args.scope}, interval: {args.interval}s")

    peak_by_key: Dict[str, int] = {key: 0 for key in npu_infos}
    peak_total_mb = 0
    samples = 0

    while True:
        running = proc.poll() is None
        try:
            snapshot = _run_npu_smi_info()
            latest_infos, total_usage = parse_npu_infos_and_total_usage(snapshot)
            for key, info in latest_infos.items():
                if key not in npu_infos:
                    npu_infos[key] = info
                    peak_by_key[key] = 0
        except Exception:
            snapshot = ""
            total_usage = {}

        if args.scope == "total":
            current_by_key = total_usage
        else:
            target_pids = get_process_tree_pids(root_pid)
            current_by_key = parse_process_usage_by_npu_chip(snapshot, target_pids)

        current_total = 0
        for key in npu_infos:
            cur = current_by_key.get(key, 0)
            current_total += cur
            if cur > peak_by_key[key]:
                peak_by_key[key] = cur

        if current_total > peak_total_mb:
            peak_total_mb = current_total
        samples += 1

        if not running:
            break
        time.sleep(args.interval)

    return_code = proc.returncode if proc.returncode is not None else 0

    print("\n=== NPU Peak Memory Report (MB) ===")
    ordered = sorted(npu_infos.items(), key=lambda kv: (kv[1].npu_id, kv[1].chip_id))
    for key, info in ordered:
        peak = peak_by_key.get(key, 0)
        pct = (peak / info.total_mb * 100.0) if info.total_mb > 0 else 0.0
        print(
            f"NPU {info.npu_id} Chip {info.chip_id}: peak={peak:>6} MB  "
            f"total={info.total_mb:>6} MB  usage={pct:>6.2f}%  name={info.name}  bus={info.bus_id}"
        )
    print(f"Combined peak (sum over NPU chips at sample time): {peak_total_mb} MB")
    print(f"Samples: {samples}, command exit code: {return_code}")

    if args.output_json:
        result = {
            "scope": args.scope,
            "interval_sec": args.interval,
            "command": args.command,
            "command_exit_code": return_code,
            "samples": samples,
            "combined_peak_mb": peak_total_mb,
            "npu_chips": [
                {
                    "npu_id": info.npu_id,
                    "chip_id": info.chip_id,
                    "name": info.name,
                    "bus_id": info.bus_id,
                    "total_mb": info.total_mb,
                    "peak_mb": peak_by_key.get(key, 0),
                }
                for key, info in ordered
            ],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[monitor] wrote JSON report: {args.output_json}")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
