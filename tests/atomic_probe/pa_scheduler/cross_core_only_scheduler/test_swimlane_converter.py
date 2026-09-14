"""Schema/topology/dependency regression tests; generated fixtures are not hardware data."""
import copy
import importlib.util
import unittest
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location("only_scheduler_converter", Path(__file__).with_name("swimlane_converter.py"))
converter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(converter)


def fixture():
    rows, edges = [], []
    for b in range(256):
        base = b * 5
        for kind in range(1, 5):
            w = b % 8 + (8 if kind in (2, 4) else 0)
            start = 100 + b * 20 + kind * 3
            for phase, begin, end in (("Kernel", start, start + 1), ("Commit", start + 1, start + 2)):
                rows.append([w, w % 8, int(w >= 8), base + kind, kind - 1, phase, begin, end, 0, 0])
        edges.extend([[base + c, base + p] for c, p in ((2, 1), (3, 2), (4, 0), (4, 2), (4, 3))])
    for w in range(16):
        rows.append([w, w % 8, int(w >= 8), -1, -1, "FinalDrain", 90, 6000, 0, 0])
    counts = Counter(r[0] for r in rows)
    return {"metadata": {
        "schema": "pa-only-scheduler-v1", "base": "cross_core_ordinary",
        "task_build": "host_before_launch", "aic_workers": 8, "aiv_workers": 8,
        "batches": 256, "tasks": 1280, "backend": "cpu", "clock_source": "CPU_steady_clock",
        "clock_freq_hz": 1_000_000_000, "task_dependencies": edges,
        "workers": [{"worker": w, "block": w % 8, "lane": int(w >= 8),
                     "records": counts[w], "dropped": 0, "atomic_calls": 0,
                     "dcci_records": 0, "startup_tick": 80, "finish_tick": 6000}
                    for w in range(16)]}, "fdwic_events": rows}


def business_fixture():
    raw = fixture()
    raw["metadata"].update(schema="pa-only-scheduler-v2", atomic_detail="per_call",
                           scheduler_detail="business_spans_v1")
    for r in list(raw["fdwic_events"]):
        if r[5] != "Kernel":
            continue
        for phase, begin, end, flags in (
            ("ExecBind", r[6]-2, r[6]-1, 0),
            ("ExecFanin", r[6]-1, r[6], 1),
            ("ExecComplete", r[7], r[7]+1, 0),
            ("ExecTokenScan", r[6]-2, r[7]+1, 1),
            ("ExecDispatch", r[6]-2, r[7]+1, 0),
        ):
            task, function = (-1, -1) if phase in converter.WORKER_STAGES else (r[3], r[4])
            raw["fdwic_events"].append(r[:3] + [task, function, phase, begin, end, flags, 0])
    for w in range(16):
        for phase, begin, end in (("ExecStartup", 80, 90), ("ExecAdmission", 90, 91),
                                  ("ExecDrainCheck", 5998, 5999)):
            raw["fdwic_events"].append([w, w % 8, int(w >= 8), -1, -1, phase, begin, end, 0, 0])
    counts = Counter(r[0] for r in raw["fdwic_events"])
    for w in raw["metadata"]["workers"]:
        w["records"] = counts[w["worker"]]
    return raw


def single_cas_fixture():
    raw = business_fixture()
    raw["metadata"].update(dispatch_binding="host_prebound", claim_strategy="prebuilt_single_cas")
    for r in list(raw["fdwic_events"]):
        if r[5] == "ExecBind":
            raw["fdwic_events"].append(r[:3] + [r[3], -1, "Atomic", r[6], r[7], 0x54, 48])
            worker = raw["metadata"]["workers"][r[0]]
            worker["records"] += 1
            worker["atomic_calls"] += 1
    return raw


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self.raw = fixture()

    def reject(self, mutation, pattern):
        mutation(self.raw)
        with self.assertRaisesRegex(ValueError, pattern):
            converter.convert(self.raw)

    def test_valid_two_tracks(self):
        trace = converter.convert(self.raw)
        names = [e for e in trace["traceEvents"] if e.get("name") == "thread_name"]
        self.assertEqual(len(names), 32)
        self.assertEqual(trace["metadata"]["kernel_events"], 1024)
        self.assertEqual(trace["metadata"]["dependency_flows"], 1024)
        self.assertFalse(any("host" in e["args"]["name"].lower() for e in names))

    def test_build_rejected(self):
        self.reject(lambda d: d["fdwic_events"][0].__setitem__(5, "Build"), "Build/unsupported")

    def test_build_atomic_rejected(self):
        def mutate(d):
            r = d["fdwic_events"][0]
            r[5], r[9] = "Atomic", 42
        self.reject(mutate, "Build/unknown Atomic")

    def test_role_rejected(self):
        def mutate(d):
            d["fdwic_events"][0][:3] = [8, 0, 1]
        self.reject(mutate, "role mismatch")

    def test_duplicate_kernel(self):
        self.reject(lambda d: d["fdwic_events"].append(copy.copy(d["fdwic_events"][0])),
                    "duplicate Kernel")

    def test_dropped_records(self):
        self.reject(lambda d: d["metadata"]["workers"][0].__setitem__("dropped", 1), "dropped")

    def test_clock_domain(self):
        self.reject(lambda d: d["metadata"].__setitem__("backend", "ccec"), "clock domain")

    def test_missing_dependency(self):
        self.reject(lambda d: d["metadata"]["task_dependencies"].pop(0), "lost QK")

    def test_dependency_order(self):
        def mutate(d):
            # SF starts before its QK producer ends; preserve the record shape.
            d["fdwic_events"][2][6:8] = [102, 103]
            d["fdwic_events"][3][6:8] = [103, 104]
        self.reject(mutate, "dependency timestamp violation")

    def test_missing_record(self):
        self.reject(lambda d: d["fdwic_events"].pop(0), "missing task")

    def test_counts(self):
        self.reject(lambda d: d["metadata"]["workers"][0].__setitem__("atomic_calls", 1),
                    "Atomic count mismatch")

    def test_drain_before_startup(self):
        self.reject(lambda d: d["metadata"]["workers"][0].__setitem__("startup_tick", 95),
                    "scheduler outside worker boundary")

    def test_drain_after_finish(self):
        self.reject(lambda d: d["metadata"]["workers"][0].__setitem__("finish_tick", 5999),
                    "scheduler outside worker boundary")

    def test_detailed_per_call(self):
        self.raw["metadata"].update(schema="pa-only-scheduler-v2", atomic_detail="per_call")
        trace = converter.convert(self.raw)
        self.assertEqual(trace["metadata"]["atomic_detail"], "per_call")

    def test_detailed_rejects_poll_batch(self):
        self.raw["metadata"].update(schema="pa-only-scheduler-v2", atomic_detail="per_call")
        flags = converter.abi.ATOMIC_SITE_OP_IDS[5] | converter.abi.ATOMIC_POLL_BATCH | (2 << 8)
        self.raw["fdwic_events"].append([0, 0, 0, 1, -1, "Atomic", 101, 102, flags, 5])
        with self.assertRaisesRegex(ValueError, "per-call trace contains PollBatch"):
            converter.convert(self.raw)

    def test_complement_union_and_clip(self):
        self.assertEqual(list(converter.complement(0, 10, [(-1, 2), (1, 3), (5, 6), (8, 12)])),
                         [(3, 5), (6, 8)])

    def test_business_token_scan_and_residual(self):
        raw = business_fixture()
        # One extra uncompleted pass with no children: the whole span is residual.
        raw["fdwic_events"].append([0, 0, 0, -1, -1, "ExecTokenScan", 92, 96, 0, 1])
        raw["fdwic_events"].append([0, 0, 0, -1, -1, "ExecDispatch", 92, 96, 0, 0])
        raw["metadata"]["workers"][0]["records"] += 2
        trace = converter.convert(raw)
        scan = next(e for e in trace["traceEvents"] if e.get("args", {}).get("pass_index") == 1)
        self.assertEqual(scan["args"]["slots_checked"], 4)
        self.assertEqual(scan["args"]["completed_tokens"], 0)
        self.assertEqual(scan["args"]["exclusive_residual_us"], 0.004)
        self.assertTrue(scan["args"]["rescan_after_completion"])
        completed = [e for e in trace["traceEvents"] if e.get("args", {}).get("completed_tokens") == 1]
        self.assertEqual(len(completed), 1024)
        self.assertTrue(all(e["args"]["exclusive_residual_us"] == 0 for e in completed))
        self.assertGreater(trace["metadata"]["residual_spans"], 0)
        outgoing = [e for e in trace["traceEvents"] if e["ph"] == "s"]
        gaps = [e for e in trace["traceEvents"] if e.get("cat") == "task.not_issued"]
        for edge in outgoing:
            self.assertFalse(any(g["pid"] == edge["pid"] and
                                 g["ts"] <= edge["ts"] < g["ts"] + g["dur"] for g in gaps))

    def test_business_rejects_missing_bind(self):
        raw = business_fixture()
        raw["fdwic_events"] = [r for r in raw["fdwic_events"] if r[5] != "ExecBind"]
        with self.assertRaisesRegex(ValueError, "incomplete ExecBind"):
            converter.convert(raw)

    def test_prebound_bind_label_preserves_measurement(self):
        raw = business_fixture()
        original = converter.convert(raw)
        raw["metadata"]["dispatch_binding"] = "host_prebound"
        prebound = converter.convert(raw)
        old = [e for e in original["traceEvents"]
               if e.get("args", {}).get("parent_stage") == "ExecBind"]
        new = [e for e in prebound["traceEvents"]
               if e.get("args", {}).get("parent_stage") == "ExecBind"]
        self.assertEqual(len(new), 1024)
        self.assertTrue(all(e["name"] ==
                            "Payload checks / descriptor and context binding [residual]" for e in old))
        self.assertTrue(all(e["name"] ==
                            "Claim bookkeeping / attach prebuilt dispatch and task metadata [residual]" for e in new))
        self.assertEqual([{k: v for k, v in e.items() if k != "name"} for e in old],
                         [{k: v for k, v in e.items() if k != "name"} for e in new])

    def test_single_cas_claim_coverage(self):
        trace = converter.convert(single_cas_fixture())
        self.assertEqual(trace["metadata"]["claim_strategy"], "prebuilt_single_cas")
        claims = [e for e in trace["traceEvents"]
                  if e.get("cat") == "scalar.atomic" and e["args"]["site_id"] == 48]
        self.assertEqual(len(claims), 1024)

    def test_single_cas_rejects_state_peek(self):
        raw = single_cas_fixture()
        next(r for r in raw["fdwic_events"] if r[5] == "Atomic")[8:10] = [0x50, 45]
        with self.assertRaisesRegex(ValueError, "single-CAS.*state peek"):
            converter.convert(raw)

    def test_single_cas_rejects_missing_claim(self):
        raw = single_cas_fixture()
        claim = next(r for r in raw["fdwic_events"] if r[5] == "Atomic")
        raw["fdwic_events"].remove(claim)
        with self.assertRaisesRegex(ValueError, "single-CAS.*claim coverage"):
            converter.convert(raw)

    def test_single_cas_rejects_duplicate_task_claim(self):
        raw = single_cas_fixture()
        claims = [r for r in raw["fdwic_events"] if r[5] == "Atomic"]
        claims[1][3] = claims[0][3]
        with self.assertRaisesRegex(ValueError, "single-CAS.*claim coverage"):
            converter.convert(raw)

    def test_business_rejects_scan_count(self):
        raw = business_fixture()
        next(r for r in raw["fdwic_events"] if r[5] == "ExecTokenScan")[8] = 3
        with self.assertRaisesRegex(ValueError, "token-scan completion count"):
            converter.convert(raw)

    def test_business_rejects_scan_mask(self):
        raw = business_fixture()
        next(r for r in raw["fdwic_events"] if r[5] == "ExecTokenScan")[8] = 16
        with self.assertRaisesRegex(ValueError, "invalid token-scan schema"):
            converter.convert(raw)

    def test_business_rejects_crossing_spans(self):
        raw = business_fixture()
        next(r for r in raw["fdwic_events"] if r[5] == "ExecBind")[6:8] = [102, 106]
        with self.assertRaisesRegex(ValueError, "crossing Scalar stages"):
            converter.convert(raw)

    def test_business_rejects_missing_scan_parent(self):
        raw = business_fixture()
        next(r for r in raw["fdwic_events"] if r[5] == "ExecTokenScan")[6] += 1
        with self.assertRaisesRegex(ValueError, "task stage outside TokenScan"):
            converter.convert(raw)


if __name__ == "__main__":
    unittest.main()
