# 2026-09: scheduler-only occupied-count reuse — dropped

Scope: `cross_core_only_scheduler`, based on `cross_core_ordinary` at
`fdwic-swimlane-deps` baseline `6a0378b`. A5, 8 AIC + 8 AIV, B256,
4 owner tokens, Execute ticket batch 2, synchronous issue/wait. No device Build.

## Candidate and decision

Inside one `ProgressCrossCoreExec` call, count occupied tokens after the first
progress pass, then reuse that local count. Add each successfully bound ticket
batch's size and subtract the increase in completed tokens after each progress
call. This removes redundant occupancy-count scans; it does not remove the
four-slot execution-progress passes or change their order.

CPU layout-diagnostic and device golden/protocol checks passed. The candidate
did not show repeatable end-to-end improvement and was reverted. No persistent
occupancy cache or new scheduling policy remains in the delivered source.

## Evidence

Device 0: Ascend950PR_958b, CANN 9.1.0 weekly 20260708, bundled PTO headers.
User-authorized direct execution; no task-submit isolation or frequency lock.
Independent perf-clock binaries, no per-task profiling; sequential AB/BA order,
three observations per variant and workload. Times are startup-to-final-drain
microseconds, not summed phase time.

| Compute counts | Baseline median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| 6/28/4/1 | 2621.741 | 2627.913 | +0.235% |
| 1/1/1/1 | 723.382 | 724.154 | +0.107% |

The small differences do not establish a benefit. Earlier runtime
[drain occupancy snapshots](https://github.com/nalinaly/simpler/blob/fdwic-swimlane-deps/docs/investigations/2026-07-shared-pa-case1-performance-gap.md)
also failed to demonstrate stable improvement, but that was a different path.
Neither result rules out broader changes to owner-token storage or readiness
policy; those need separate hypotheses and measurements.

The subsequent business-span work explains source-path coverage only; inclusive
TokenScan time must not be advertised as removable Scalar scan cost.
See the [scheduler-only record](../test_record/2026-09-11/README.md).
