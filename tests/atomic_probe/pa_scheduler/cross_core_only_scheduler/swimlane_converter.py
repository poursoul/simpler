#!/usr/bin/env python3
"""Validate scheduler-only measurements and write a two-track/core Perfetto trace."""
from __future__ import annotations

import argparse
import importlib.util
import json
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

# Resolve beside this file even when imported through importlib by a caller.
_spec = importlib.util.spec_from_file_location(
    "only_scheduler_trace_abi", Path(__file__).resolve().with_name("trace_abi.py")
)
abi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abi)

EXEC_STAGES = {
    "ExecBind": "Acquire payload + bind",
    "ExecFanin": "Fanin check + ready transition",
    "ExecComplete": "Complete + publish DONE",
    "ExecTicketBind": "Take Execute ticket + bind tokens",
    "ExecTokenScan": "TokenScan",
    "ExecDispatch": "Progress owned tokens + Execute ticket dispatch",
    "ExecDrainCheck": "Check local drain + publish/reduce arrivals",
    "ExecAdmission": "Startup admission + throttled fatal check",
    "ExecStartup": "Startup arrival + immutable plan validation",
}
WORKER_STAGES = {"ExecTicketBind", "ExecTokenScan", "ExecDispatch", "ExecDrainCheck",
                 "ExecAdmission", "ExecStartup"}
PHASES = {"Kernel", "Commit", "Atomic", "Dcci", "ClockBaseline", "FinalDrain"} | EXEC_STAGES.keys()
ATOMIC_SITES = {0, 1, 2, 3, 5, 43, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56}
DCCI_SITES = {8, 9, 12, 13}
KINDS = {1: "QK", 2: "SF", 3: "PV", 4: "UP"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def complement(start, end, intervals):
    """Uncovered intervals in source-clock ticks; nested children count once."""
    cursor = start
    for a, b in sorted(intervals):
        a, b = max(start, a), min(end, b)
        if b <= a:
            continue
        if a > cursor:
            yield cursor, a
        cursor = max(cursor, b)
    if cursor < end:
        yield cursor, end


RESIDUAL_LABELS = {
    "ExecBind": "Payload checks / descriptor and context binding",
    "ExecFanin": "Fanin routing / ready-prefix bookkeeping",
    "ExecComplete": "Completion state / owner-token reset",
    "ExecTicketBind": "Ticket routing / idle-slot search / token binding",
    "ExecTokenScan": "TokenScan traversal / token-state checks",
    "ExecDispatch": "Dispatch control / occupied-slot counting / batch gating",
    "ExecDrainCheck": "Drain checks / completion-count reduction",
    "ExecAdmission": "Admission / fatal-poll cadence control",
    "ExecStartup": "Startup bookkeeping / plan-header validation",
    "Scheduler + final drain": "Scheduler loop / watchdog / profiling bookkeeping",
    "Owner initialization + trace attach": "Owner state / token reset / trace attachment",
    "Observer tail (outside performance window)": "Trace export / timing calibration bookkeeping",
}
STAGE_DEPTH = {"ExecStartup": 1, "ExecAdmission": 1, "ExecDispatch": 1,
               "ExecDrainCheck": 1, "ExecTokenScan": 2, "ExecTicketBind": 2,
               "ExecBind": 3, "ExecFanin": 3, "ExecComplete": 3}


def nesting_depth(event):
    if event.get("cat") == "scalar.envelope" or event["name"] == "Scheduler + final drain":
        return 0
    return STAGE_DEPTH.get(event["args"].get("stage"), 4)


def add_business_coverage(events, metadata, drain, stamp, scale):
    """Explain complements of measured stages, never invent instruction timings."""
    residual_count = 0
    for w, worker in enumerate(metadata["workers"]):
        lane = [e for e in events if e.get("pid") == w + 1 and e.get("tid") == 1
                and e.get("ph") == "X" and e["dur"] > 0]
        for name, start, end in (
            ("Owner initialization + trace attach", min(e["args"]["start_tick"] for e in lane),
             worker["startup_tick"]),
            ("Observer tail (outside performance window)", drain[w][7], worker["finish_tick"]),
        ):
            if end <= start:
                continue
            event = dict(ph="X", pid=w + 1, tid=1, ts=stamp(start), dur=(end-start)*scale,
                         cat="scalar.envelope", name=name,
                         args=dict(worker=w, start_tick=start, end_tick=end,
                                   provenance="derived envelope between recorded device boundaries",
                                   outside_performance_window=True))
            events.append(event)
            lane.append(event)

        # Build the nesting tree in integer ticks. Crossing measured stages are
        # a schema/instrumentation error, not something to hide by clipping.
        children, stack = defaultdict(list), []
        ordered = sorted(lane, key=lambda e: (e["args"]["start_tick"],
                                             -e["args"]["end_tick"], nesting_depth(e)))
        for event in ordered:
            a, b = event["args"]["start_tick"], event["args"]["end_tick"]
            while stack and a >= stack[-1]["args"]["end_tick"]:
                stack.pop()
            if stack:
                require(b <= stack[-1]["args"]["end_tick"], "crossing Scalar stages")
                children[id(stack[-1])].append((a, b))
            stage = event["args"].get("stage")
            if stage in ("ExecBind", "ExecFanin", "ExecComplete") or event["cat"] == "scalar.blocked":
                require(stack and stack[-1]["args"].get("stage") == "ExecTokenScan",
                        "task stage outside TokenScan")
            elif stage in ("ExecTokenScan", "ExecTicketBind"):
                require(stack and stack[-1]["args"].get("stage") == "ExecDispatch",
                        "token scan/ticket outside dispatch")
            stack.append(event)

        for parent in ordered:
            stage = parent["args"].get("stage", parent["name"])
            if stage not in RESIDUAL_LABELS:
                continue
            a, b = parent["args"]["start_tick"], parent["args"]["end_tick"]
            for start, end in complement(a, b, children[id(parent)]):
                residual_count += 1
                label = RESIDUAL_LABELS[stage]
                if stage == "ExecBind" and metadata.get("dispatch_binding") == "host_prebound":
                    label = "Claim bookkeeping / attach prebuilt dispatch and task metadata"
                events.append(dict(
                    ph="X", pid=w + 1, tid=1, ts=stamp(start), dur=(end-start)*scale,
                    cat="scalar.residual", name=label + " [residual]",
                    args=dict(worker=w, start_tick=start, end_tick=end, parent_stage=stage,
                              task_id=parent["args"].get("task_id", -1),
                              provenance="derived complement of measured child intervals",
                              interpretation="source-path attribution, not an isolated instruction measurement; "
                                             "includes ordinary memory, control, trace writes and observer effects",
                              pure_scalar_busy=False)))

        calls = [e for e in events if e.get("pid") == w + 1 and e.get("cat") == "task"]
        call_ends = {e["args"]["end_tick"] for e in calls}
        for start, end in complement(drain[w][6], drain[w][7],
                                     [(e["args"]["start_tick"], e["args"]["end_tick"])
                                      for e in calls]):
            # Perfetto binds an outgoing flow at task.end to a new slice at
            # that same timestamp regardless of JSON ordering. Leave one
            # display tick free; preserve the exact measured dependency time
            # and full derived gap in args instead of attaching arrows to gaps.
            display_start = start + int(start in call_ends)
            if display_start >= end:
                continue
            events.append(dict(
                ph="X", pid=w + 1, tid=2, ts=stamp(display_start), dur=(end-display_start)*scale,
                cat="task.not_issued", name="No task issued: see Scalar scheduler",
                args=dict(worker=w, start_tick=display_start, end_tick=end,
                          gap_start_tick=start, full_gap_us=(end-start)*scale,
                          display_endpoint_guard_ticks=display_start-start,
                          provenance="derived gap between synchronous task calls",
                          interpretation="Scalar is handling tokens, dependencies, tickets or drain; "
                                         "not an independent engine-idle/PMU measurement")))
    metadata["residual_spans"] = residual_count
    metadata["gap_semantics"] = "business-path complements are derived, not instruction timings or idle measurements"


def validate(raw):
    m, rows = raw["metadata"], raw["fdwic_events"]
    require(m["schema"] in ("pa-only-scheduler-v1", "pa-only-scheduler-v2"), "wrong schema")
    detailed = m["schema"] == "pa-only-scheduler-v2"
    if detailed:
        require(m.get("atomic_detail") == "per_call", "missing per-call contract")
    require(m["base"] == "cross_core_ordinary", "wrong baseline")
    require(m["task_build"] == "host_before_launch", "device Build is forbidden")
    require(m["aic_workers"] == m["aiv_workers"] == 8, "expected 8 AIC + 8 AIV")
    require(m["batches"] == 256 and m["tasks"] == 1280, "only B256 is supported")
    require(m["backend"] in ("cpu", "ccec"), "unknown backend")
    require(m["clock_source"] == (
        "AICore_SYS_CNT" if m["backend"] == "ccec" else "CPU_steady_clock"
    ), "clock domain/backend mismatch")
    require(m["clock_freq_hz"] == 1_000_000_000, "unexpected counter frequency")
    workers = m["workers"]
    require([c["worker"] for c in workers] == list(range(16)), "missing/duplicate worker")
    records, atomics, dcci = Counter(), Counter(), Counter()
    kernels, commits, drain = {}, {}, {}
    for w, c in enumerate(workers):
        require((c["block"], c["lane"]) == (w % 8, int(w >= 8)), "bad topology")
        require(c["dropped"] == 0, "dropped trace records")
        require(0 < c["startup_tick"] <= c["finish_tick"], "bad worker boundary")
    require(rows, "empty trace")
    for row in rows:
        require(len(row) == 10, "bad record width")
        w, block, lane, task, function, phase, start, end, flags, site = row
        require(type(w) is int and 0 <= w < 16, "bad worker")
        require((block, lane) == (w % 8, int(w >= 8)), "bad row topology")
        require(phase in PHASES, f"Build/unsupported phase: {phase}")
        require(type(start) is int and type(end) is int and 0 < start <= end, "bad time")
        require(0 <= flags <= 0xFFFFFFFF, "invalid flags")
        records[w] += 1
        if phase in ("Kernel", "Commit"):
            require(0 <= task < 1280 and task % 5 in KINDS, "bad executable task")
            require(function == task % 5 - 1, "bad function")
            require((w < 8) == (task % 5 in (1, 3)), "kernel role mismatch")
            target = kernels if phase == "Kernel" else commits
            require(task not in target, f"duplicate {phase} task {task}")
            target[task] = row
        elif phase == "FinalDrain":
            require(task == function == -1 and w not in drain, "bad FinalDrain")
            drain[w] = row
        elif phase == "Atomic":
            require(not detailed or not flags & abi.ATOMIC_POLL_BATCH,
                    "per-call trace contains PollBatch")
            require(site in ATOMIC_SITES, f"Build/unknown Atomic site {site}")
            require(flags & 15 == abi.ATOMIC_SITE_OP_IDS[site], "Atomic operation mismatch")
            count = flags >> 8 if flags & abi.ATOMIC_POLL_BATCH else 1
            require(count > 0, "empty poll batch")
            atomics[w] += count
        elif phase == "Dcci":
            require(site in DCCI_SITES, f"Build/unknown DCCI site {site}")
            calls = (flags >> 3) & 15
            require(calls == (2 if site == 8 else 1), "invalid DCCI call count")
            require(flags >> 8 >= calls, "invalid DCCI line count")
            require(flags & 3 == abi.DCCI_SITE_OP_IDS[site], "invalid DCCI operation")
            dcci[w] += 1
        elif phase in EXEC_STAGES:
            require(detailed, "execution stages need v2")
            if phase == "ExecTokenScan":
                require(task == function == -1 and flags <= 15 and 0 <= site <= 4,
                        "invalid token-scan schema")
            elif phase == "ExecFanin":
                require(flags <= 1 and site == 0, "invalid fanin-stage schema")
            else:
                require(flags == site == 0, "invalid execution-stage schema")
            if phase in WORKER_STAGES - {"ExecTicketBind"}:
                require(task == function == -1, "bad worker stage")
            if phase not in WORKER_STAGES:
                require(task in range(1280) and task % 5 in KINDS, "bad stage task")
                require(function == task % 5 - 1, "bad stage function")
                require((w < 8) == (task % 5 in (1, 3)), "stage role mismatch")
    expected = {t for t in range(1280) if t % 5}
    require(kernels.keys() == commits.keys() == expected, "missing task execution/completion")
    require(set(drain) == set(range(16)), "missing drain")
    if "claim_strategy" in m:
        require(m["claim_strategy"] == "prebuilt_single_cas" and detailed and
                m.get("dispatch_binding") == "host_prebound", "invalid single-CAS contract")
        require(not any(r[5] == "Atomic" and r[9] == 45 for r in rows),
                "single-CAS trace contains a cell state peek")
        claims = [r for r in rows if r[5] == "Atomic" and r[9] == 48]
        require(Counter(r[3] for r in claims) == Counter({t: 1 for t in expected}),
                "single-CAS task claim coverage mismatch")
        for r in claims:
            require(r[0] == kernels[r[3]][0] and r[8] & (1 << 4),
                    "single-CAS claim owner/result contract mismatch")
    if m.get("scheduler_detail") == "business_spans_v1":
        require(detailed, "business spans need v2")
        for phase in ("ExecBind", "ExecComplete"):
            stage_tasks = Counter(r[3] for r in rows if r[5] == phase)
            require(stage_tasks == Counter({t: 1 for t in expected}), f"incomplete {phase} coverage")
        ready_tasks = Counter(r[3] for r in rows if r[5] == "ExecFanin" and r[8] == 1)
        require(ready_tasks == Counter({t: 1 for t in expected}), "incomplete ready-fanin coverage")
        for w in range(16):
            counts = Counter(r[5] for r in rows if r[0] == w)
            require(counts["ExecStartup"] == 1 and all(counts[p] for p in
                    ("ExecDispatch", "ExecAdmission", "ExecDrainCheck")), "missing worker stages")
            require(sum(r[8].bit_count() for r in rows if r[0] == w and r[5] == "ExecTokenScan")
                    == counts["Kernel"], "token-scan completion count mismatch")
    for w, c in enumerate(workers):
        require(c["startup_tick"] <= drain[w][6] <= drain[w][7] <= c["finish_tick"],
                "scheduler outside worker boundary")
        require(records[w] == c["records"], "record count mismatch")
        require(atomics[w] == c["atomic_calls"], "weighted Atomic count mismatch")
        require(dcci[w] == c["dcci_records"], "DCCI count mismatch")
    by_worker = defaultdict(list)
    for task, row in kernels.items():
        commit = commits[task]
        require(commit[0] == row[0] and commit[6] >= row[7], "invalid completion order")
        require(drain[row[0]][6] <= row[6] <= row[7] <= drain[row[0]][7],
                "kernel outside scheduler/drain boundary")
        by_worker[row[0]].append(row)
    for rows_per_worker in by_worker.values():
        ordered = sorted(rows_per_worker, key=lambda r: r[6])
        require(all(a[7] <= b[6] for a, b in zip(ordered, ordered[1:])),
                "synchronous kernels overlap on one core")
    edges = m["task_dependencies"]
    require(len({tuple(e) for e in edges}) == len(edges), "duplicate dependency")
    incoming = defaultdict(set)
    for consumer, producer in edges:
        require(consumer in kernels and 0 <= producer < consumer, "invalid dependency")
        require(consumer // 5 == producer // 5, "unexpected cross-batch dependency")
        incoming[consumer].add(producer)
        if producer % 5:
            require(kernels[producer][7] <= kernels[consumer][6],
                    f"dependency timestamp violation: {producer}->{consumer}")
    # Compare actual payload fanin with ordinary G1: Alloc is UP's in/out
    # buffer producer, not a QK dependency.
    for batch in range(256):
        b = batch * 5
        require(incoming[b + 1] == set(), "unexpected QK dependency")
        require(incoming[b + 2] == {b + 1}, "SF lost QK dependency")
        require(incoming[b + 3] == {b + 2}, "PV lost SF dependency")
        require(incoming[b + 4] == {b, b + 2, b + 3}, "UP fanin mismatch")
    return kernels, commits, drain


def convert(raw):
    kernels, _, drain = validate(raw)
    m = dict(raw["metadata"])
    rows = raw["fdwic_events"]
    origin = min(r[6] for r in rows)
    scale = 1_000_000 / m["clock_freq_hz"]
    events = []
    fanin = {
        w: sorted((r for r in rows if r[0] == w and r[5] == "ExecFanin"), key=lambda r: r[6])
        for w in range(16)
    }
    fanin_starts = {w: [r[6] for r in rs] for w, rs in fanin.items()}
    scan_children = {
        w: sorted((r for r in rows if r[0] == w and r[5] in
                   ("ExecBind", "ExecFanin", "ExecComplete", "Kernel", "Atomic", "Dcci")),
                  key=lambda r: r[6])
        for w in range(16)
    }
    child_starts = {w: [r[6] for r in rs] for w, rs in scan_children.items()}

    def stamp(t):
        return (t - origin) * scale

    for w in range(16):
        core = f"{'AIC' if w < 8 else 'AIV'} {w % 8:02d}"
        for name, args in (
            ("process_name", {"name": core}),
            ("process_sort_index", {"sort_index": w}),
        ):
            events.append({"ph": "M", "name": name, "pid": w + 1, "args": args})
        for tid, name in ((1, "Scalar scheduler"), (2, (
            "Cube task execution" if w < 8 else "Vector task execution"
        ))):
            events.append({"ph": "M", "name": "thread_name", "pid": w + 1,
                           "tid": tid, "args": {"name": name}})
            events.append({"ph": "M", "name": "thread_sort_index", "pid": w + 1,
                           "tid": tid, "args": {"sort_index": tid}})

    for w, block, lane, task, function, phase, start, end, flags, site in rows:
        args = {"worker": w, "task_id": task, "function_id": function,
                "start_tick": start, "end_tick": end, "flags": flags}
        event = {"ph": "X", "pid": w + 1, "tid": 1, "ts": stamp(start),
                 "dur": (end - start) * scale, "cat": "scalar",
                 "name": phase, "args": args}
        if phase == "Kernel":
            event.update(tid=2, cat="task", name=f"{KINDS[task % 5]} task {task}")
            args["boundary"] = "synchronous kernel call including issue, transfer and wait"
            args["pure_engine_busy"] = False
            # Same observed interval, explicitly a blocking wait on Scalar.
            events.append({**event, "tid": 1, "cat": "scalar.blocked",
                           "name": f"issue + wait {KINDS[task % 5]} task {task}",
                           "args": dict(args)})
        elif phase == "FinalDrain":
            event["name"] = "Scheduler + final drain"
        elif phase in EXEC_STAGES:
            event.update(name=EXEC_STAGES[phase], cat="scalar.stage")
            args["stage"] = phase
            args["boundary"] = "measured Scalar stage; includes nested primitives and observer overhead"
            if phase == "ExecFanin" and m.get("scheduler_detail") == "business_spans_v1":
                args["fanin_ready"] = bool(flags)
                event["name"] = ("Fanin ready -> inflight" if flags else "Fanin not ready: poll producer")
            if phase == "ExecTokenScan":
                covered, cursor = 0, start
                for r in scan_children[w][bisect_left(child_starts[w], start):
                                          bisect_right(child_starts[w], end)]:
                    a, b = max(start, r[6]), min(end, r[7])
                    if b > max(cursor, a):
                        covered += b - max(cursor, a)
                    cursor = max(cursor, b)
                args.update(slots_checked=4, completed_slot_mask=flags,
                            completed_tokens=flags.bit_count(), pass_index=site,
                            rescan_after_completion=site > 0,
                            nested_work_us=covered * scale,
                            exclusive_residual_us=(end - start - covered) * scale,
                            residual_semantics="derived: excludes measured children; "
                                               "includes scan control, ordinary memory and trace bookkeeping")
                event["name"] = f"TokenScan: 4 slots, {flags.bit_count()} completed"
        elif phase == "Atomic":
            op = abi.ATOMIC_OP_NAMES[flags & 15]
            label = abi.ATOMIC_SITE_NAMES[site]
            batch = bool(flags & abi.ATOMIC_POLL_BATCH)
            boundary = "poll_window" if batch else (
                "return_ready" if flags & abi.ATOMIC_RETURN_READY else "source_issue"
            )
            args.update(site=label, site_id=site, op=op, boundary=boundary,
                        call_count=flags >> 8 if batch else 1)
            if site == 5:
                args["producer_task_id"] = task
                index = bisect_right(fanin_starts[w], start) - 1
                if index >= 0 and end <= fanin[w][index][7]:
                    args["consumer_task_id"] = fanin[w][index][3]
            event.update(name=f"{label}.{op}.{boundary}", cat="scalar.atomic")
            if batch:
                # Poll envelopes may overlap direct operations. Display as an
                # instant with exact window/call count, not a false busy slice.
                args["poll_window_us"] = event.pop("dur")
                args["not_atomic_latency"] = True
                event.update(ph="i", s="t")
        elif phase == "Dcci":
            label = abi.DCCI_SITE_NAMES[site]
            args.update(site=label, site_id=site, call_count=(flags >> 3) & 15,
                        cache_lines=flags >> 8, trailing_dsb=bool(flags & 4))
            event.update(name=f"dcci.{label}", cat="scalar.dcci")
        events.append(event)

    if m.get("scheduler_detail") == "business_spans_v1":
        add_business_coverage(events, m, drain, stamp, scale)

    flow_count = 0
    for consumer, producer in m["task_dependencies"]:
        if producer % 5 == 0:
            continue  # Alloc happened before launch; do not invent its timestamps.
        a, b = kernels[producer], kernels[consumer]
        flow_count += 1
        for ph, row, tick in (("s", a, a[7]), ("f", b, b[6])):
            e = {"ph": ph, "cat": "dependency", "name": f"{producer}->{consumer}",
                 "id": flow_count, "pid": row[0] + 1, "tid": 2, "ts": stamp(tick)}
            if ph == "f":
                e["bp"] = "e"
            events.append(e)
    m.update(
        origin_tick=origin, lane_count=32, scalar_workers=16, kernel_events=len(kernels),
        dependency_flows=flow_count, no_device_build=True,
        task_lane_semantics="synchronous kernel call, not isolated engine busy time",
        poll_batch_display=(
            "none; every polling Atomic is an individual span"
            if m["schema"] == "pa-only-scheduler-v2" else
            "instant marker; exact window and call count in args"
        ),
        timing_scope="startup to final drain; excludes host preparation and H2D",
        startup_to_final_drain_us=(
            max(r[7] for r in drain.values())
            - min(c["startup_tick"] for c in m["workers"])
        ) * scale,
        startup_to_observer_finish_us=(
            max(c["finish_tick"] for c in m["workers"])
            - min(c["startup_tick"] for c in m["workers"])
        ) * scale,
        profiling=(
            "execution stages + per-call Atomic (including polls) + DCCI; instrumented"
            if m["schema"] == "pa-only-scheduler-v2" else
            "phase + full Atomic/PollBatch + DCCI; instrumented, not performance baseline"
        ),
        validation="PASS",
    )
    events.sort(key=lambda e: (
        0 if e["ph"] == "M" else 1, e.get("ts", 0),
        -e.get("dur", 0),
        nesting_depth(e) if e["ph"] == "X" else 0
    ))
    return {"displayTimeUnit": "ms", "metadata": m, "traceEvents": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), f"refusing to overwrite {args.output}")
    trace = convert(json.loads(args.raw.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive output creation preserves an existing capture.
    with args.output.open("x") as out:
        json.dump(trace, out, ensure_ascii=False, separators=(",", ":"))
        out.write("\n")
    print(f"[PERFETTO] {args.output}: 16 cores / 32 tracks / "
          f"{trace['metadata']['kernel_events']} kernels / validation=PASS")


if __name__ == "__main__":
    main()
