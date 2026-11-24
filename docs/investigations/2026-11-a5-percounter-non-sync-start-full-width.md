# A5 Percounter continuation for full-width tasks without sync_start

The archived experiment extends the A5 Percounter full-width continuation from
base `5b10887bd`. With one pending task per slot, a synchronous pure-AIC or
pure-AIV full-width task can follow an existing full-width `sync_start` entry
when the interval checks establish a valid continuation. The real entry,
dependency checks, graph boundaries, and the existing terminal pipeline remain
in effect. Ordinary full-width MIX tasks were considered separately in a
Host-only projection.

## Native DSV4.1 results

The [DSV4.1 scheduler benchmark](../../tests/fdwic/dsv41_scheduler/README.md)
used the original workload flags and kernels, Percounter in both arms, and
AICore L0 wall time. Each sample used one warmup and ten measured rounds on
the same card or pair for both arms. The device code sections were identical.

| Case | Reference (µs) | Experiment (µs) | Additional tasks removed |
| ---- | -------------: | --------------: | -----------------------: |
| swa | 140.797 | 142.266 | 0 |
| c2a_full | 191.522 | 190.212 | 0 |
| c1a_reindex | 243.554 | 245.274 | 0 |
| c1a_reuse | 149.173 | 151.259 | 0 |
| engram | 98.503 | 96.161 | 0 |
| moe | 443.258 | 441.875 | 0 |
| layers_19_21 | 1866.215 | 1860.225 | 6 |

The consecutive-layer result is the mean of two matched comparisons in
opposite order. Each rank removed three ordinary 64-block `2v` tasks.
The mean improvement was 5.990 µs (0.321%). The six other rows each
record one comparison with zero change in interval coverage.

Five C++ targets passed three repeated runs. The A5 simulation passed 36
checks. The native hardware suite passed its replay checks and the MoE
execution controls.

## Decision

Keep the existing admission policy on the development branch. The measured
coverage and gain support keeping the experiment for future work on graphs
with more eligible successors.
