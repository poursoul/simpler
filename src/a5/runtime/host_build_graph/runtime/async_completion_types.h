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

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

#include "intrinsic.h"

namespace pto2::resident {

inline constexpr uint32_t ASYNC_RECORD_ALIGNMENT = 128;
inline constexpr uint32_t ASYNC_INVALID_INDEX = UINT32_MAX;
enum class AsyncPublicationState : uint32_t { OPEN = 0, SEALED = 1, FAILED = 2 };

// One physical execution owns these lines until its return publishes them.
// Scheduler leases and progress occupy separate cache lines.
struct alignas(ASYNC_RECORD_ALIGNMENT) AsyncGroupData {
    uint64_t run_epoch;
    int64_t task_ref;
    uint32_t group_index;
    uint32_t block_index;
    uint32_t subtask_slot;
    uint16_t capacity;
    uint16_t reserved;
    uint32_t publication_bytes;
    volatile uint32_t count;
    volatile int32_t error_code;
    volatile uint32_t reservation_serial;
    volatile uint32_t reservation_limit;
    volatile uint32_t reservation_begin;
    volatile uint32_t reservation_written;
    volatile AsyncPublicationState state;
};

struct AsyncReservation {
    uint64_t run_epoch{0};
    uint32_t group_index{ASYNC_INVALID_INDEX};
    uint32_t serial{0};
    uint32_t capacity{0};
};

constexpr inline __aicore__ uint32_t async_group_publication_bytes(uint32_t capacity) {
    if (capacity == 0 || capacity > MAX_COMPLETIONS_PER_TASK) return 0;
    const uint32_t bytes = sizeof(AsyncGroupData) + capacity * sizeof(DeferredCompletionEntry);
    return (bytes + ASYNC_RECORD_ALIGNMENT - 1) & ~(ASYNC_RECORD_ALIGNMENT - 1);
}

static_assert(sizeof(AsyncGroupData) == 128 && alignof(AsyncGroupData) == ASYNC_RECORD_ALIGNMENT);
static_assert(offsetof(AsyncGroupData, count) == 36 && offsetof(AsyncGroupData, state) == 60);
static_assert(std::is_standard_layout_v<AsyncGroupData> && std::is_trivially_copyable_v<AsyncGroupData>);
static_assert(sizeof(AsyncReservation) == 24);
static_assert(async_group_publication_bytes(1) == 256 && async_group_publication_bytes(64) == 2176);

}  // namespace pto2::resident
