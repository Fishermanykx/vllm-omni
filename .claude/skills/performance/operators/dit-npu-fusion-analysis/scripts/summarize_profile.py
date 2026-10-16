#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Summarize explicitly selected trace events or per-invocation CSV rows.

Standard library only. Does not infer hardware lanes, correlate host/device,
compute self time, or claim end-to-end latency/utilization.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}


def number(value, label):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label}: expected a finite nonnegative number, got {value!r}")
    return result


def union_duration(intervals):
    end = None
    total = 0.0
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
        else:
            total += max(0.0, stop - end)
        end = stop if end is None else max(end, stop)
    return total


def summarize(rows, top):
    groups = defaultdict(lambda: {"count": 0, "sum_ms": 0.0, "max_ms": 0.0, "samples": []})
    intervals = []
    for row in rows:
        key = (row["name"], json.dumps(row["shape"], ensure_ascii=False, sort_keys=True))
        group = groups[key]
        group["count"] += 1
        group["sum_ms"] += row["duration_ms"]
        group["max_ms"] = max(group["max_ms"], row["duration_ms"])
        if len(group["samples"]) < 3:
            group["samples"].append(row)
        if "start_ms" in row:
            intervals.append((row["start_ms"], row["start_ms"] + row["duration_ms"]))
    if not rows:
        raise ValueError("No matching events/rows. Inspect inventory and selection; no statistics produced.")
    ranked = []
    for (name, shape), stats in groups.items():
        ranked.append({"name": name, "shape": json.loads(shape), **stats, "mean_ms": stats["sum_ms"] / stats["count"]})
    ranked.sort(key=lambda item: item["sum_ms"], reverse=True)
    result = {
        "selected_count": len(rows),
        "group_count": len(groups),
        "sum_ms": sum(row["duration_ms"] for row in rows),
        "top": ranked[:top],
    }
    if intervals:
        result["selected_intervals_union_ms"] = union_duration(intervals)
        result["observed_span_ms"] = max(end for _, end in intervals) - min(start for start, _ in intervals)
    return result


def trace_report(args):
    opener = gzip.open if args.path.suffix == ".gz" else open
    with opener(args.path, "rt", encoding="utf-8-sig") as source:
        payload = json.load(source)
    events = payload if isinstance(payload, list) else payload["traceEvents"]
    lanes = Counter(
        (str(e.get("cat", "")), str(e.get("pid", "")), str(e.get("tid", "")), str(e.get("ph", ""))) for e in events
    )
    metadata = [e for e in events if e.get("ph") == "M"]
    if args.inventory:
        return {
            "event_count": len(events),
            "metadata": metadata,
            "lanes": [
                {"category": cat, "pid": pid, "tid": tid, "phase": phase, "count": count}
                for (cat, pid, tid, phase), count in lanes.most_common()
            ],
        }
    if not args.category or args.pid is None:
        raise ValueError("Trace analysis requires --category and --pid; inspect --inventory first.")
    rows = []
    unsupported_phases = Counter()
    for index, event in enumerate(events):
        if str(event.get("cat", "")) not in args.category or str(event.get("pid", "")) != args.pid:
            continue
        if args.tid is not None and str(event.get("tid", "")) != args.tid:
            continue
        if event.get("ph") != "X":
            unsupported_phases[str(event.get("ph", ""))] += 1
            continue
        detail = event.get("args") or {}
        shape = next(
            (detail[key] for key in ("Input Shapes", "Input shapes", "input_shapes", "shapes") if key in detail), None
        )
        # Unit comes from confirmed export format, not displayTimeUnit.
        factor = TO_MS[args.unit]
        start = float(event["ts"])
        if not math.isfinite(start):
            raise ValueError(f"event {index}: timestamp must be finite")
        rows.append(
            {
                "name": str(event.get("name", "")),
                "shape": shape,
                "duration_ms": number(event["dur"], f"event {index} dur") * factor,
                "start_ms": start * factor,
                "event_index": index,
                "pid": event.get("pid"),
                "tid": event.get("tid"),
                "args": detail,
            }
        )
    return {
        "selection": {"categories": args.category, "pid": args.pid, "tid": args.tid},
        "ignored_selected_phases": dict(unsupported_phases),
        **summarize(rows, args.top),
    }


def csv_report(args):
    with args.path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        if args.inventory:
            return {"columns": columns, "first_rows": [row for _, row in zip(range(3), reader)]}
        required = [args.name_column, args.duration_column]
        if args.shape_column:
            required.append(args.shape_column)
        if any(not col or col not in columns for col in required):
            raise ValueError(f"Set exact --name-column/--duration-column/--shape-column from: {columns}")
        rows = []
        for record, row in enumerate(reader, 1):
            rows.append(
                {
                    "name": row[args.name_column],
                    "shape": row[args.shape_column] if args.shape_column else None,
                    "duration_ms": number(row[args.duration_column], f"CSV record {record}") * TO_MS[args.unit],
                    "csv_record": record,
                    "csv_line_end": reader.line_num,
                    "fields": row,
                }
            )
    return {
        "selection": {
            "name_column": args.name_column,
            "duration_column": args.duration_column,
            "shape_column": args.shape_column,
        },
        **summarize(rows, args.top),
    }


def main():
    # Windows redirected stdout may otherwise use a locale code page.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="format", required=True)
    for kind in ("trace", "csv"):
        command = commands.add_parser(kind)
        command.add_argument("path", type=Path)
        command.add_argument("--inventory", action="store_true")
        command.add_argument("--unit", choices=TO_MS)
        command.add_argument("--top", type=int, default=30)
        command.add_argument("--output", type=Path)
        if kind == "trace":
            command.add_argument("--category", action="append")
            command.add_argument("--pid")
            command.add_argument("--tid")
        else:
            command.add_argument("--name-column")
            command.add_argument("--duration-column")
            command.add_argument("--shape-column")
    args = parser.parse_args()
    if not args.inventory and not args.unit:
        parser.error("--unit is required for analysis; confirm the export timestamp/duration units.")
    if args.top < 1:
        parser.error("--top must be positive")
    try:
        result = trace_report(args) if args.format == "trace" else csv_report(args)
        result = {
            "source": str(args.path.resolve()),
            "raw_unit": args.unit,
            "warning": (
                "Selected event sums may overlap. No self time, host/device correlation, "
                "request latency, utilization, or speedup is inferred."
            ),
            **result,
        }
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
