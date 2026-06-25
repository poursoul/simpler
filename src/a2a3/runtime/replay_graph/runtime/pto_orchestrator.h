/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * PTO Runtime2 - Orchestrator Interface
 *
 * The Orchestrator is responsible for:
 * 1. Executing the orchestration function (Turing-complete control flow)
 * 2. Allocating intermediate buffers from the heap
 * 3. Submitting tasks via async InCore function calls
 * 4. Building the dependency graph using TensorMap
 * 5. Managing buffer scopes for lifecycle control
 *
 * The Orchestrator can run on either:
 * - Host CPU (lower latency for complex control, easier debugging)
 * - Device AI_CPU (lower latency for task submission)
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#ifndef PTO_ORCHESTRATOR_H
#define PTO_ORCHESTRATOR_H

#include "common/l2_swimlane_profiling.h"
#include "utils/device_arena.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_submit_types.h"
#include "scheduler/pto_scheduler.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"

/**
 * Layout descriptor produced by PTO2OrchestratorState::reserve_layout(). Holds
 * arena offsets for every sub-region the orchestrator owns (per-ring fanin
 * pools, scope arrays, plus the nested PTO2TensorMap layout).
 */
struct PTO2OrchestratorLayout {
    size_t off_fanin_pool;
    size_t off_fanin_seen_epoch;
    // Wiring sub-regions, moved off the scheduler layout: the orchestrator now
    // owns the fanout dep_pool entries, the wiring SPSC buffer, and the
    // initial-ready handoff array (replay_graph stage 1).
    size_t off_dep_pool_entries;
    size_t off_wiring_spsc_buffer;
    size_t off_initial_ready;
    PTO2TensorMapLayout tensor_map;
    int32_t dep_pool_capacity;
    uint64_t spsc_capacity;
    int32_t initial_ready_cap;
    uint64_t scope_stack_capacity;
};

// =============================================================================
// Orchestrator State
// =============================================================================

/**
 * Orchestrator state structure (private to Orchestrator)
 *
 * Contains all state needed for task graph construction and buffer management.
 */
struct PTO2OrchestratorState {
    // === SHARED MEMORY ACCESS ===
    PTO2SharedMemoryHeader *sm_header;

    // === ALLOCATOR RESOURCES (single-shot bump allocators) ===
    PTO2TaskAllocator task_allocator;
    PTO2FaninPool fanin_pool;
    // Fanout dependency-list pool. Owned by the orchestrator: wiring builds the
    // fanout linked list here during the orch phase; the scheduler only reads it
    // (read-only traversal in on_task_complete).
    PTO2DepListPool dep_pool;
    uint32_t *fanin_seen_epoch;
    uint32_t fanin_seen_current_epoch{1};

    // === TENSOR MAP (Private) ===
    PTO2TensorMap tensor_map;  // Producer lookup

    // === SCOPE STACK (Private) ===
    // Depth-only bookkeeping: the single-shot replay model tracks scope nesting
    // for manual-scope semantics, but no longer collects per-scope task lists
    // (slot reclaim was dropped).
    int32_t scope_stack_top;        // Current top of stack (-1 = no scope open)
    uint64_t scope_stack_capacity;  // Max nesting depth (PTO2_MAX_SCOPE_DEPTH)
    int32_t manual_begin_depth{PTO2_MAX_SCOPE_DEPTH};

    // === SCHEDULER REFERENCE ===
    // Note: In simulated mode, orchestrator and scheduler share address space
    // In real mode, they communicate via shared memory only
    PTO2SchedulerState *scheduler;  // For simulated mode only

    // === WIRING (orchestrator-owned) ===
    // replay_graph stage 1: wiring is fully owned by the orchestrator. submit_task
    // only pushes into this queue; run_wiring() drains it after orchestration,
    // builds the fanout linked lists in dep_pool, and produces the
    // initial-ready handoff array consumed by the scheduler. The SPSC queue is
    // kept (not collapsed into submit_task) so submit and wiring can be pipelined
    // once the orchestrator runs multi-threaded.
    struct alignas(64) WiringState {
        static constexpr uint64_t BATCH_SIZE = 30;
        static constexpr int BACKOFF_LIMIT = 32;

        // --- single-thread exclusive: local batch buffer + backoff ---
        int batch_count = 0;
        int batch_index = 0;
        int backoff_counter = 0;
        PTO2TaskSlotState *batch[BATCH_SIZE];

        // --- SPSC queue: submit (push) ↔ drain (pop) ---
        PTO2SpscQueue queue;

        // --- reserved for future submit/drain pipelining ---
        alignas(64) std::atomic<bool> orch_needs_drain{false};
    } wiring;

    static_assert(
        offsetof(WiringState, queue) == 256, "WiringState: batch region must be exactly 4 cache lines before queue"
    );
    static_assert(sizeof(WiringState) == 640, "WiringState must be exactly 10 cache lines (640B)");

    // Initial-ready handoff (orchestrator → scheduler, single direction). wire_task
    // appends every task whose fanin is satisfied at wiring time; the scheduler
    // seeds these into its ready_queues before dispatch. A construction-time
    // pure-function product, reusable across future multi-pass scheduling.
    PTO2TaskSlotState **initial_ready;
    int32_t initial_ready_count;
    int32_t initial_ready_capacity;

    // Total core counts set once at executor init; used for submit-time deadlock detection.
    int32_t total_cluster_count{0};  // AIC cores = MIX clusters
    int32_t total_aiv_count{0};      // AIV cores (= 2 × clusters on standard hardware)
#if PTO2_PROFILING
    // L2 swimlane_level copied from get_l2_swimlane_level().
    L2SwimlaneLevel l2_swimlane_level{L2SwimlaneLevel::DISABLED};
#endif

    // === GM HEAP (for output buffers) ===
    void *gm_heap_base;     // Base address of GM heap
    uint64_t gm_heap_size;  // Total size of GM heap (all rings)

    // === FATAL ERROR ===
    // Fatal error flag (single-thread access by orchestrator, no atomic needed)
    // Cross-thread notification uses shared memory orch_error_code (atomic)
    bool fatal;

    // Hidden alloc tasks complete synchronously inside the orchestrator and
    // therefore bypass the executor's normal worker-completion counter path.
    // The executor adds this count into its completed_tasks_ progress counter
    // after orchestration finishes so shutdown/profiling totals remain closed.
    int64_t inline_completed_tasks{0};

    // === STATISTICS ===
#if PTO2_PROFILING
    int64_t tasks_submitted;
    int64_t buffers_allocated;
    int64_t bytes_allocated;
#endif

    bool in_manual_scope() const { return scope_stack_top >= manual_begin_depth; }

    // === WIRING (orchestrator-owned, replay_graph stage 1) ===

    // Append a task whose fanin is satisfied at wiring time to the initial-ready
    // handoff array (seeded into the scheduler's ready_queues before dispatch).
    void push_initial_ready(PTO2TaskSlotState *ws) {
        if (initial_ready_count >= initial_ready_capacity) {
            report_fatal(
                PTO2_ERROR_DEP_POOL_OVERFLOW, __FUNCTION__, "initial_ready overflow (count=%d cap=%d)",
                initial_ready_count, initial_ready_capacity
            );
            return;
        }
        initial_ready[initial_ready_count++] = ws;
    }

    // Wire one task's fanout edges into the given dep_pool and seed its
    // fanin refcount. Moved off the scheduler (was PTO2SchedulerState::wire_task):
    // builds fanout linked lists the scheduler later traverses read-only. Producers
    // are still PENDING during the orch phase except for inline-completed alloc
    // tasks, so early_finished is normally 0 but the branch is kept for those.
    void wire_task(PTO2DepListPool &dep_pool, PTO2TaskSlotState *ws, int32_t wfanin) {
        PTO2TaskPayload *wp = ws->payload;
        ws->fanin_count = wfanin + 1;

        if (wfanin != 0) {
            int32_t early_finished = 0;
            for_each_fanin_slot_state(*wp, [&](PTO2TaskSlotState *producer) {
                // single-shot: fanout_head is frozen before sched starts, so no
                // lock is needed. MUST restore this lock before any orch/sched
                // time-overlap (stage-3 ping-pong). wire_task is the sole writer
                // and runs single-threaded in the orch phase.
                int32_t pstate = producer->task_state.load(std::memory_order_acquire);
                if (pstate >= PTO2_TASK_COMPLETED) {
                    early_finished++;
                } else {
                    producer->fanout_head = dep_pool.prepend(producer->fanout_head, ws);
                }
            });

            if (early_finished != 0) {
                wp->dispatch_fanin.fetch_add(early_finished, std::memory_order_acq_rel);
            }

            int32_t init_rc = early_finished + 1;
            int32_t new_rc = ws->fanin_refcount.fetch_add(init_rc, std::memory_order_acq_rel) + init_rc;
            if (new_rc >= ws->fanin_count) {
                push_initial_ready(ws);
            }
        } else {
            ws->fanin_refcount.fetch_add(1, std::memory_order_acq_rel);
            push_initial_ready(ws);
        }
    }

    // Drain a batch of submitted tasks from the wiring queue and wire each. No
    // task is CONSUMED during the orch phase, so dep_pool cannot be reclaimed —
    // its capacity must hold every fanout edge of the whole graph. Overflow is
    // fatal rather than a spin (reclamation can never make progress here).
    int drain_wiring_queue(bool force_drain = false) {
        int wired = 0;
        if (wiring.batch_index >= wiring.batch_count) {
            if (!force_drain && wiring.queue.size() < WiringState::BATCH_SIZE) {
                if (!wiring.orch_needs_drain.load(std::memory_order_acquire) &&
                    wiring.backoff_counter < WiringState::BACKOFF_LIMIT) {
                    wiring.backoff_counter++;
                    return 0;
                }
            }
            wiring.backoff_counter = 0;
            wiring.batch_count = wiring.queue.pop_batch(wiring.batch, WiringState::BATCH_SIZE);
            wiring.batch_index = 0;
            if (wiring.batch_count == 0) return 0;
        }

        while (wiring.batch_index < wiring.batch_count) {
            PTO2TaskSlotState *ws = wiring.batch[wiring.batch_index];
            PTO2DepListPool &dep_pool = this->dep_pool;
            int32_t wfanin = ws->payload->fanin_actual_count;
            if (wfanin > 0 && dep_pool.available() < wfanin) {
                report_fatal(
                    PTO2_ERROR_DEP_POOL_OVERFLOW, __FUNCTION__,
                    "dep_pool exhausted during orch wiring (need=%d avail=%d)", wfanin, dep_pool.available()
                );
                return wired;
            }
            wiring.batch_index++;
            wire_task(dep_pool, ws, wfanin);
            wired++;
        }

        return wired;
    }

    // Drain the entire wiring queue after orchestration completes. Single-threaded
    // for now (submit fully precedes wiring); becomes pipelinable when the orch
    // runs multi-threaded.
    void run_wiring() {
        while (drain_wiring_queue(/*force_drain=*/true) > 0) {
            if (fatal) return;
        }
    }

    // === Cold-path API (defined in pto_orchestrator.cpp) ===

    // Phase 1: declare every sub-region (fanin pool, scope arrays,
    // tensor_map sub-layout) on the supplied arena. task_window_size feeds
    // the nested tensor_map layout. Returned layout is consumed by
    // init_from_layout.
    static PTO2OrchestratorLayout
    reserve_layout(DeviceArena &arena, int32_t task_window_size, int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE);

    // Phase 3a: write everything *except* arena-internal pointer fields.
    // sm_dev_base is the SM device address (only stored, never dereferenced);
    // task_window_size feeds the per-ring SM address arithmetic. Safe to call
    // on a host arena that holds the prebuilt image.
    bool init_data_from_layout(
        const PTO2OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap, uint64_t heap_size,
        uint64_t task_window_size
    );

    // Phase 3b: write the arena-internal pointer fields (fanin_pool.base,
    // tensor_map.{buckets,entry_pool,free_entry_list},
    // scheduler reference).
    // Idempotent — host runs once on the image, AICPU runs once after attach.
    void wire_arena_pointers(const PTO2OrchestratorLayout &layout, DeviceArena &arena, PTO2SchedulerState *scheduler);

    // Forget pointers; arena owns the backing buffers.
    void destroy();
    void set_scheduler(PTO2SchedulerState *scheduler);
    void report_fatal(int32_t error_code, const char *func, const char *fmt, ...);
    void begin_scope(PTO2ScopeMode mode = PTO2ScopeMode::AUTO);
    void end_scope();
    TaskOutputTensors submit_task(const MixedKernels &mixed_kernels, const L0TaskArgs &args);
    TaskOutputTensors submit_dummy_task(const L0TaskArgs &args);
    TaskOutputTensors alloc_tensors(const L0TaskArgs &args);
    void mark_done();
};

// =============================================================================
// Orchestrator Profiling Data
// =============================================================================

#if PTO2_ORCH_PROFILING
struct PTO2OrchProfilingData {
    uint64_t alloc_cycle;  // Combined task slot + heap allocation
    uint64_t args_cycle;
    uint64_t lookup_cycle;
    uint64_t insert_cycle;
    uint64_t fanin_cycle;
    uint64_t scope_end_cycle;
    int64_t submit_count;
    // Wait time tracking for blocking phases
    uint64_t fanin_wait_cycle;  // Cycles spent waiting on fanin (was fanout_lock spin)
    // Atomic operation counts per phase
    uint64_t alloc_atomic_count;
};

PTO2OrchProfilingData orchestrator_get_profiling();
#endif

#endif  // PTO_ORCHESTRATOR_H
