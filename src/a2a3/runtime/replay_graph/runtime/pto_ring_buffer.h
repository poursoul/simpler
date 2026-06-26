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
 * PTO Runtime2 - Ring Buffer Data Structures
 *
 * "Ring buffer" here is a historical name: replay_graph is single-shot, so
 * these allocators fill once and never wrap (overflow is fatal, not a
 * wrap-around). The name is kept because renaming the file would churn a large
 * set of includes; the structures below are plain bump allocators.
 *
 * Implements ring buffer designs for zero-overhead memory management:
 *
 * 1. TaskAllocator - Unified task slot + output buffer allocation
 *    - Combines task ring (slot allocation) and heap ring (output buffer allocation)
 *    - Single-shot pure bump: the arena fills once and never wraps; overflow is fatal
 *    - O(1) bump allocation for both task slots and heap buffers
 *
 * 2. DepListPool - Dependency list entry allocation
 *    - Linear bump allocator for linked list entries
 *    - O(1) prepend operation
 *    - Single-shot: filled once, never reclaimed (overflow is fatal)
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#ifndef PTO_RING_BUFFER_H
#define PTO_RING_BUFFER_H

#include <algorithm>
#include <inttypes.h>
#include <type_traits>

#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "common/unified_log.h"

// =============================================================================
// Task Allocator (unified task slot + heap buffer allocation)
// =============================================================================

/**
 * Unified task slot + heap buffer allocator.
 *
 * Since task and heap are always allocated together and the orchestrator is
 * single-threaded, both pointers (task index, heap top) are tracked locally
 * and published to shared memory via plain store — no fetch_add or CAS needed.
 *
 * replay_graph is single-shot: the arena is filled exactly once and never
 * wraps, and the scheduler does not run during the orch phase so nothing is
 * ever reclaimed. alloc() is therefore a pure bump — running out of either the
 * task window or the heap is a fatal sizing error, not a back-pressure spin.
 */
class PTO2TaskAllocator {
public:
    /**
     * Initialize the allocator with task ring and heap ring resources.
     *
     * All pointer arguments are device addresses (live in SM / GM heap); this
     * function only stores them, no dereferences, so it is safe to invoke
     * from host code that constructs a prebuilt arena image.
     *
     * Production callers leave `initial_local_task_id` at 0: the SM ring
     * flow-control counter that task_count_ptr points at starts at zero
     * (PTO2FlowControl::init() runs on the AICPU during SM reset), so we
     * keep local_task_id_ aligned with that without reading the SM. Tests that
     * drive SM state directly may pass a non-zero in-window seed.
     */
    void init(
        int32_t window_size, std::atomic<int32_t> *task_count_ptr, void *heap_base, uint64_t heap_size,
        std::atomic<int32_t> *error_code_ptr, int32_t initial_local_task_id = 0
    ) {
        window_size_ = window_size;
        task_count_ptr_ = task_count_ptr;
        heap_base_ = heap_base;
        heap_size_ = heap_size;
        error_code_ptr_ = error_code_ptr;
        local_task_id_ = initial_local_task_id;
        heap_top_ = 0;
    }

    /**
     * Allocate a task slot and its associated output buffer in one call.
     *
     * replay_graph is single-shot: total tasks <= window and total output
     * bytes <= heap, the arena is filled exactly once and never wraps, and the
     * scheduler does not run during the orch phase so nothing is ever reclaimed.
     * Allocation is therefore a pure bump — running out of either resource is a
     * fatal sizing error, not a back-pressure condition to spin on.
     *
     * @param output_size  Total packed output size in bytes (0 = no heap needed)
     * @return Allocation result; check failed() for errors
     */
    PTO2TaskAllocResult alloc(int32_t output_size) {
        uint64_t aligned = output_size > 0 ? PTO2_ALIGN_UP(static_cast<uint64_t>(output_size), PTO2_ALIGN_SIZE) : 0;
        // single-shot fill, no wrap: out-of-bounds is a fatal config error, not a spin
        if (local_task_id_ + 1 >= window_size_) {
            report_task_overflow();
            return {-1, -1, nullptr, nullptr};
        }
        if (heap_top_ + aligned > heap_size_) {
            report_heap_overflow(output_size);
            return {-1, -1, nullptr, nullptr};
        }
        void *p = static_cast<char *>(heap_base_) + heap_top_;
        heap_top_ += aligned;
        int32_t task_id = local_task_id_++;
        task_count_ptr_->store(local_task_id_, std::memory_order_release);
#if PTO2_ORCH_PROFILING
        extern uint64_t g_orch_alloc_atomic_count;
        g_orch_alloc_atomic_count += 1;
#endif
        return {task_id, task_id, p, static_cast<char *>(p) + aligned};
    }

    // =========================================================================
    // State queries
    // =========================================================================

    // Single-shot: nothing is reclaimed during the orch phase, so every
    // allocated task is still live. active_count == number allocated.
    int32_t active_count() const { return local_task_id_; }

    // Task ring start/end: head = next task id to allocate. Single-shot never
    // reclaims, so the tail stays at the origin.
    int32_t task_tail() const { return 0; }
    int32_t task_head() const { return local_task_id_; }

    int32_t window_size() const { return window_size_; }

    // Pure bump: free space is simply what's left between the top and the cap.
    uint64_t heap_available() const { return heap_size_ - heap_top_; }

    uint64_t heap_top() const { return heap_top_; }
    // Heap ring start: single-shot never reclaims, so the tail stays at 0.
    uint64_t heap_tail() const { return 0; }
    uint64_t heap_capacity() const { return heap_size_; }

private:
    // --- Task Ring ---
    int32_t window_size_ = 0;
    std::atomic<int32_t> *task_count_ptr_ = nullptr;

    // --- Heap ---
    void *heap_base_ = nullptr;
    uint64_t heap_size_ = 0;

    // --- Local state (single-writer, no atomics needed) ---
    int32_t local_task_id_ = 0;  // Next task ID to allocate
    uint64_t heap_top_ = 0;      // Current heap allocation pointer

    // --- Shared ---
    std::atomic<int32_t> *error_code_ptr_ = nullptr;

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /**
     * Report a fatal task-ring overflow: the single-shot fill needs more task
     * slots than the window provides. Sizing error, not back-pressure.
     */
    void report_task_overflow() {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Task Allocator Overflow - Task Window Full!");
        LOG_ERROR("========================================");
        LOG_ERROR(
            "  Task ring:  current=%d, window=%d (single-shot fill exceeds window)", local_task_id_, window_size_
        );
        LOG_ERROR("Solution:");
        LOG_ERROR("  Increase task window size (current: %d, recommended: %d)", window_size_, window_size_ * 2);
        LOG_ERROR("  Compile-time: PTO2_TASK_WINDOW_SIZE in pto_runtime2_types.h");
        LOG_ERROR("  Runtime env:  PTO2_RING_TASK_WINDOW=<power-of-2> (e.g. %d)", window_size_ * 2);
        LOG_ERROR("========================================");
        if (error_code_ptr_) {
            error_code_ptr_->store(PTO2_ERROR_FLOW_CONTROL_DEADLOCK, std::memory_order_release);
        }
    }

    /**
     * Report a fatal heap overflow: the single-shot fill needs more output
     * bytes than the heap provides. Sizing error, not back-pressure.
     */
    void report_heap_overflow(int32_t requested) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Task Allocator Overflow - Heap Exhausted!");
        LOG_ERROR("========================================");
        LOG_ERROR(
            "  Heap ring:  top=%" PRIu64 ", size=%" PRIu64 ", available=%" PRIu64, heap_top_, heap_size_,
            heap_available()
        );
        LOG_ERROR("  Requested:  %d bytes", requested);
        LOG_ERROR("Solution:");
        LOG_ERROR("  Increase heap size (current: %" PRIu64 ", recommended: %" PRIu64 ")", heap_size_, heap_size_ * 2);
        LOG_ERROR("  Compile-time: PTO2_HEAP_SIZE in pto_runtime2_types.h");
        LOG_ERROR("  Runtime env:  PTO2_RING_HEAP=<bytes> (e.g. %" PRIu64 ")", heap_size_ * 2);
        LOG_ERROR("========================================");
        if (error_code_ptr_) {
            error_code_ptr_->store(PTO2_ERROR_HEAP_RING_DEADLOCK, std::memory_order_release);
        }
    }
};

// =============================================================================
// Dependency List Pool
// =============================================================================

/**
 * Dependency list pool structure
 *
 * Single-shot linear bump allocator for fanout linked-list entries. Wiring
 * builds the whole graph's fanout lists here during the orch phase; the
 * scheduler only reads them and never runs during orch, so nothing is ever
 * reclaimed — capacity must hold every fanout edge and overflow is fatal.
 *
 * Linear counters (top, tail) grow monotonically; the physical index
 * is obtained via modulo: base[linear_index % capacity]. tail stays at its
 * initial value (no reclamation) so used() == top - 1.
 */
struct PTO2DepListPool {
    PTO2DepListEntry *base;  // Pool base address
    int32_t capacity;        // Total number of entries
    int32_t top;             // Linear next-allocation counter (starts from 1)
    int32_t tail;            // Linear first-alive counter (fixed; nothing is reclaimed)
    int32_t high_water;      // Peak usage (top - tail)

    // Error code pointer for fatal error reporting (→ sm_header->orch_error_code)
    std::atomic<int32_t> *error_code_ptr = nullptr;

    /**
     *
     * Initialize dependency list pool
     * @param base      Pool base address from shared memory
     * @param capacity  Total number of entries
     */
    void init(PTO2DepListEntry *in_base, int32_t in_capacity, std::atomic<int32_t> *in_error_code_ptr) {
        base = in_base;
        capacity = in_capacity;
        top = 1;   // Start from 1, 0 means NULL/empty
        tail = 1;  // Fixed; nothing is reclaimed in the single-shot model

        high_water = 0;

        // Initialize entry 0 as NULL marker
        base[0].slot_state = nullptr;
        base[0].next = nullptr;

        error_code_ptr = in_error_code_ptr;
    }

    /**
     * Allocate a single entry from the pool (single-thread per pool instance)
     *
     * @return Pointer to allocated entry, or nullptr on fatal error
     */
    PTO2DepListEntry *alloc() {
        int32_t used = top - tail;
        if (used >= capacity) {
            LOG_ERROR("========================================");
            LOG_ERROR("FATAL: Dependency Pool Overflow!");
            LOG_ERROR("========================================");
            LOG_ERROR("DepListPool exhausted: %d entries alive (capacity=%d).", used, capacity);
            LOG_ERROR("  - Pool top:      %d (linear)", top);
            LOG_ERROR("  - Pool tail:     %d (linear)", tail);
            LOG_ERROR("  - High water:    %d", high_water);
            LOG_ERROR("Solution:");
            LOG_ERROR("  Increase dep pool capacity (current: %d, recommended: %d).", capacity, capacity * 2);
            LOG_ERROR("  Compile-time: PTO2_DEP_LIST_POOL_SIZE in pto_runtime2_types.h");
            LOG_ERROR("  Runtime env:  PTO2_RING_DEP_POOL=%d", capacity * 2);
            LOG_ERROR("========================================");
            if (error_code_ptr) {
                error_code_ptr->store(PTO2_ERROR_DEP_POOL_OVERFLOW, std::memory_order_release);
            }
            return nullptr;
        }
        int32_t idx = top % capacity;
        top++;
        used++;
        if (used > high_water) high_water = used;
        return &base[idx];
    }

    /**
     * Prepend a task ID to a dependency list
     *
     * O(1) operation: allocates new entry and links to current head.
     *
     * @param current_head  Current list head offset (0 = empty list)
     * @param task_slot     Task slot to prepend
     * @return New head offset
     */
    PTO2DepListEntry *prepend(PTO2DepListEntry *cur, PTO2TaskSlotState *slot_state) {
        PTO2DepListEntry *new_entry = alloc();
        if (!new_entry) return nullptr;
        new_entry->slot_state = slot_state;
        new_entry->next = cur;
        return new_entry;
    }

    int32_t used() const { return top - tail; }

    int32_t available() const { return capacity - used(); }
};

#endif  // PTO_RING_BUFFER_H
