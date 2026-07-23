#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Analyze FDWIC merged_swimlane.json for submit exclusive time and lane gaps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

TraceEvent = dict[str, Any]

RUNTIME_PHASES = {
    "alloc",
    "build",
    "claim",
    "drain_won",
    "efdrain",
    "fanin",
    "materialize",
    "prepare_map",
    "register",
    "replay",
    "resolve",
    "resolve_copy",
    "resolve_invalidate",
    "resolve_wait",
    "ringbp",
    "submit",
}


def latest_trace() -> Path:
    traces = list(Path("outputs").glob("TestPagedAttentionUnroll_Case1_*/merged_swimlane.json"))
    if not traces:
        raise FileNotFoundError("no PA Case1 merged_swimlane.json found under outputs/")
    return max(traces, key=lambda path: path.stat().st_mtime)


def load_events(path: Path) -> list[TraceEvent]:
    data = json.loads(path.read_text())
    return [event for event in data.get("traceEvents", []) if event.get("ph") == "X"]


def event_phase(event: TraceEvent) -> str:
    return str(event.get("args", {}).get("phase", ""))


def event_core(event: TraceEvent) -> int:
    return int(event.get("args", {}).get("core", -1))


def event_task(event: TraceEvent) -> int:
    return int(event.get("args", {}).get("task_id", -1))


def event_func(event: TraceEvent) -> int:
    return int(event.get("args", {}).get("func_id", -1))


def event_start(event: TraceEvent) -> float:
    return float(event.get("ts", 0.0))


def event_end(event: TraceEvent) -> float:
    return event_start(event) + float(event.get("dur", 0.0))


def event_lane(event: TraceEvent) -> str:
    pid = int(event.get("pid", -1))
    tid = int(event.get("tid", -1))
    return f"pid{pid}:tid{tid}"


def event_lane_type(event: TraceEvent) -> str:
    tid = int(event.get("tid", -1))
    if tid in (0, 3):
        return "AIC"
    if tid in (1, 4):
        return "AIV0"
    if tid in (2, 5):
        return "AIV1"
    return f"tid{tid}"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return ordered[index]


def interval_union_us(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    total += current_end - current_start
    return total


def phase_table(events: list[TraceEvent]) -> list[tuple[str, int, float, float]]:
    rows: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for event in events:
        phase = event_phase(event)
        dur = float(event.get("dur", 0.0))
        rows[phase][0] += 1
        rows[phase][1] += dur
        rows[phase][2] = max(rows[phase][2], dur)
    return [(phase, int(count), total, max_us) for phase, (count, total, max_us) in rows.items()]


def submit_exclusive(events: list[TraceEvent]) -> list[tuple[float, TraceEvent]]:
    submit_events = [event for event in events if event_phase(event) == "submit"]
    children_by_core_task: dict[tuple[int, int], list[TraceEvent]] = defaultdict(list)
    for event in events:
        if event_phase(event) != "submit":
            children_by_core_task[(event_core(event), event_task(event))].append(event)

    rows: list[tuple[float, TraceEvent]] = []
    for submit in submit_events:
        start = event_start(submit)
        end = event_end(submit)
        intervals = [
            (event_start(child), event_end(child))
            for child in children_by_core_task[(event_core(submit), event_task(submit))]
            if event_start(child) >= start and event_end(child) <= end
        ]
        rows.append((max(0.0, end - start - interval_union_us(intervals)), submit))
    return rows


def runtime_gaps(events: list[TraceEvent]) -> list[tuple[float, TraceEvent, TraceEvent]]:
    by_lane: dict[tuple[int, int], list[TraceEvent]] = defaultdict(list)
    for event in events:
        if event_phase(event) in RUNTIME_PHASES:
            by_lane[(int(event.get("pid", -1)), int(event.get("tid", -1)))].append(event)

    gaps: list[tuple[float, TraceEvent, TraceEvent]] = []
    for lane_events in by_lane.values():
        lane_events.sort(key=lambda event: (event_start(event), event_end(event), event_phase(event)))
        prev: TraceEvent | None = None
        for current in lane_events:
            if prev is not None:
                gap = event_start(current) - event_end(prev)
                if gap > 0:
                    gaps.append((gap, prev, current))
            if prev is None or event_end(current) >= event_end(prev):
                prev = current
    return gaps


def print_phase_totals(events: list[TraceEvent], span_us: float) -> None:
    print("Phase totals:")
    print(f"{'phase':18s} {'count':>8s} {'sum_us':>12s} {'avg_ns':>10s} {'max_us':>10s} {'span%':>8s}")
    for phase, count, total, max_us in sorted(phase_table(events), key=lambda row: row[2], reverse=True):
        avg_ns = total * 1000 / count if count else 0
        span_pct = total * 100 / span_us if span_us > 0 else 0
        print(f"{phase:18s} {count:8d} {total:12.3f} {avg_ns:10.1f} {max_us:10.3f} {span_pct:7.1f}%")


def print_phase_breakdown(events: list[TraceEvent], phases: tuple[str, ...]) -> None:
    for target_phase in phases:
        buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
        for event in events:
            if event_phase(event) == target_phase:
                buckets[(event_lane_type(event), event_task(event) % 5)].append(float(event.get("dur", 0.0)))
        if not buckets:
            continue
        print(f"\n{target_phase} breakdown:")
        print(
            f"{'lane':>5s} {'mod5':>4s} {'count':>7s} {'sum_us':>10s} {'avg_ns':>9s} "
            f"{'p50_ns':>9s} {'p95_ns':>9s} {'max_us':>8s}"
        )
        for (lane, mod), values in sorted(buckets.items(), key=lambda item: sum(item[1]), reverse=True)[:12]:
            total = sum(values)
            print(
                f"{lane:>5s} {mod:4d} {len(values):7d} {total:10.3f} "
                f"{total * 1000 / len(values):9.1f} {percentile(values, 0.50) * 1000:9.1f} "
                f"{percentile(values, 0.95) * 1000:9.1f} {max(values):8.3f}"
            )


def print_submit_exclusive(events: list[TraceEvent]) -> None:
    rows = submit_exclusive(events)
    values = [value for value, _event in rows]
    avg_ns = (sum(values) * 1000 / len(values)) if values else 0
    print("\nSubmitExclusive:")
    print(
        f"count {len(values)} sum_us {sum(values):.3f} avg_ns {avg_ns:.1f} "
        f"p50_ns {percentile(values, 0.50) * 1000:.1f} p95_ns {percentile(values, 0.95) * 1000:.1f} "
        f"max_us {(max(values) if values else 0):.3f}"
    )
    print(f"{'lane':>5s} {'mod5':>4s} {'count':>7s} {'sum_us':>10s} {'avg_ns':>9s} {'p95_ns':>9s} {'max_us':>8s}")
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for value, event in rows:
        buckets[(event_lane_type(event), event_task(event) % 5)].append(value)
    for (lane, mod), values_for_bucket in sorted(buckets.items(), key=lambda item: sum(item[1]), reverse=True)[:15]:
        total = sum(values_for_bucket)
        print(
            f"{lane:>5s} {mod:4d} {len(values_for_bucket):7d} {total:10.3f} "
            f"{total * 1000 / len(values_for_bucket):9.1f} {percentile(values_for_bucket, 0.95) * 1000:9.1f} "
            f"{max(values_for_bucket):8.3f}"
        )
    print("Top SubmitExclusive events:")
    print(f"{'core':>4s} {'lane':>5s} {'task':>6s} {'mod5':>4s} {'func':>5s} {'exclusive_us':>13s} {'submit_us':>10s}")
    for value, event in sorted(rows, reverse=True, key=lambda item: item[0])[:15]:
        print(
            f"{event_core(event):4d} {event_lane_type(event):>5s} {event_task(event):6d} {event_task(event) % 5:4d} "
            f"{event_func(event):5d} {value:13.3f} {float(event.get('dur', 0.0)):10.3f}"
        )


def print_gaps(events: list[TraceEvent]) -> None:
    gaps = runtime_gaps(events)
    values = [gap for gap, _prev, _current in gaps]
    avg_ns = (sum(values) * 1000 / len(values)) if values else 0
    print("\nRuntime lane gaps:")
    print(
        f"count {len(values)} sum_us {sum(values):.3f} avg_ns {avg_ns:.1f} "
        f"p50_ns {percentile(values, 0.50) * 1000:.1f} p95_ns {percentile(values, 0.95) * 1000:.1f} "
        f"max_us {(max(values) if values else 0):.3f}"
    )
    print(f"{'lane':>5s} {'next_mod5':>9s} {'count':>7s} {'sum_us':>10s} {'avg_ns':>9s} {'p95_ns':>9s} {'max_us':>8s}")
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for gap, _prev, current in gaps:
        buckets[(event_lane_type(current), event_task(current) % 5)].append(gap)
    for (lane, mod), values_for_bucket in sorted(buckets.items(), key=lambda item: sum(item[1]), reverse=True)[:15]:
        total = sum(values_for_bucket)
        print(
            f"{lane:>5s} {mod:9d} {len(values_for_bucket):7d} {total:10.3f} "
            f"{total * 1000 / len(values_for_bucket):9.1f} {percentile(values_for_bucket, 0.95) * 1000:9.1f} "
            f"{max(values_for_bucket):8.3f}"
        )
    print("Top runtime gaps:")
    print(
        f"{'lane_id':>12s} {'core':>4s} {'lane':>5s} {'gap_us':>9s} {'prev_phase':>18s} {'prev_task':>9s} "
        f"{'next_phase':>18s} {'next_task':>9s} {'next_mod5':>9s}"
    )
    for gap, prev, current in sorted(gaps, reverse=True, key=lambda item: item[0])[:20]:
        print(
            f"{event_lane(current):>12s} {event_core(current):4d} {event_lane_type(current):>5s} {gap:9.3f} "
            f"{event_phase(prev):>18s} {event_task(prev):9d} {event_phase(current):>18s} "
            f"{event_task(current):9d} {event_task(current) % 5:9d}"
        )
    print("Gap edge breakdown:")
    print(
        f"{'edge':>27s} {'lane':>5s} {'mod5':>4s} {'back':>5s} {'count':>7s} {'sum_us':>10s} "
        f"{'avg_ns':>9s} {'p95_ns':>9s} {'max_us':>8s}"
    )
    edges: dict[tuple[str, str, str, int, bool], list[float]] = defaultdict(list)
    for gap, prev, current in gaps:
        edges[
            (
                event_phase(prev),
                event_phase(current),
                event_lane_type(current),
                event_task(current) % 5,
                event_task(current) < event_task(prev),
            )
        ].append(gap)
    for (prev_phase, next_phase, lane, mod, back), values_for_edge in sorted(
        edges.items(), key=lambda item: sum(item[1]), reverse=True
    )[:20]:
        total = sum(values_for_edge)
        edge = f"{prev_phase}->{next_phase}"
        print(
            f"{edge:>27s} {lane:>5s} {mod:4d} {str(back):>5s} {len(values_for_edge):7d} "
            f"{total:10.3f} {total * 1000 / len(values_for_edge):9.1f} "
            f"{percentile(values_for_edge, 0.95) * 1000:9.1f} {max(values_for_edge):8.3f}"
        )
    forward = [(gap, prev, current) for gap, prev, current in gaps if event_task(current) >= event_task(prev)]
    print("Top forward runtime gaps:")
    for gap, prev, current in sorted(forward, reverse=True, key=lambda item: item[0])[:16]:
        print(
            f"{event_lane_type(current):>5s} core {event_core(current):3d} gap_us {gap:8.3f} "
            f"{event_phase(prev):>12s} t{event_task(prev):4d} -> {event_phase(current):<12s} "
            f"t{event_task(current):4d} mod{event_task(current) % 5} func{event_func(current)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path, help="merged_swimlane.json; defaults to latest PA Case1")
    args = parser.parse_args()

    path = args.trace or latest_trace()
    events = load_events(path)
    if not events:
        print(f"{path}\nevents 0")
        return 0

    start = min(event_start(event) for event in events)
    end = max(event_end(event) for event in events)
    span = end - start
    print(path)
    print(f"events {len(events)} global_span_us {span:.3f} start_us {start:.3f} end_us {end:.3f}")
    print_phase_totals(events, span)
    print_phase_breakdown(events, ("claim", "efdrain", "build", "materialize", "resolve", "resolve_copy"))
    print_submit_exclusive(events)
    print_gaps(events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
