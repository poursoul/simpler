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
 * @file l2_perf_collector_aicore.h
 * @brief AICore performance data collection interface
 *
 * Provides lightweight performance recording interface for AICore kernels.
 * Uses dcci for efficient cache management instead of memory barriers.
 */

#ifndef PLATFORM_AICORE_L2_PERF_COLLECTOR_AICORE_H_
#define PLATFORM_AICORE_L2_PERF_COLLECTOR_AICORE_H_

#include "common/l2_perf_profiling.h"
#include "aicore/aicore.h"

// Include platform-specific timestamp implementation
// Build system selects the correct inner_kernel.h based on platform:
// - src/a2a3/platform/onboard/aicore/inner_kernel.h (real hardware)
// - src/a2a3/platform/sim/aicore/inner_kernel.h (simulation)
// Both provide unified get_sys_cnt_aicore() interface
#include "inner_kernel.h"

// ============= Public Interface =============

/**
 * Record task execution performance data
 *
 * Writes timing + identity into the per-core staging ring at
 * `dual_issue_slots[reg_task_id % PLATFORM_L2_AICORE_RING_SIZE]`. The ring is
 * stable for the entire run; AICPU does NOT read it in the host-side perf path
 * — host directly walks the ring at run end.
 *
 * AICore writes the FULL PTO2 task_id (passed in by aicore_executor from
 * `exec_payload->local_context.async_ctx.task_token.raw`). `reg_task_id` is
 * stored separately as the dual-issue validation key.
 *
 * @param ring         Per-core staging ring pointer
 * @param reg_task_id  Register dispatch token (low 32 bits) — slot index key
 * @param task_id      Full PTO2 task id (ring_id << 32) | local_id
 * @param start_time   Start timestamp (sys_cnt_aicore)
 * @param end_time     End timestamp (sys_cnt_aicore)
 * @param core_id      AICore block_idx (so host can group slots by core)
 * @param core_type    AIC / AIV — kernel-time property
 */
__aicore__ __attribute__((always_inline)) static inline void l2_perf_aicore_record_task(
    __gm__ L2PerfAicoreRing *ring, uint32_t reg_task_id, uint64_t task_id, uint64_t start_time, uint64_t end_time,
    uint8_t core_id, CoreType core_type
) {
    __gm__ L2PerfRecord *record = &ring->dual_issue_slots[reg_task_id % PLATFORM_L2_AICORE_RING_SIZE];

    record->start_time = start_time;
    record->end_time = end_time;
    record->duration = end_time - start_time;
    record->task_id = task_id;
    record->reg_task_id = reg_task_id;
    record->core_id = core_id;
    record->core_type = core_type;

    // Flush cache to make data visible to host (via DMA mapping on onboard,
    // direct read on sim).
    dcci(record, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
}

#endif  // PLATFORM_AICORE_L2_PERF_COLLECTOR_AICORE_H_
