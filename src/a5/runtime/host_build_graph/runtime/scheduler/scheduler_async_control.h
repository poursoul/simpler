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

#include "async_condition_writer.h"
#include "scheduler_async_directory.h"
#include "scheduler_async_layout.h"
#include "scheduler_graph.h"

struct SchedulerAsyncLocalState {
    __gm__ uint8_t *base{nullptr};
    __gm__ const SchedulerAsyncLayout *layout{nullptr};
    uint64_t bytes{0};
    uint32_t check_cursor{0};
    uint32_t finish_cursor{0};
    int64_t failed_task_ref{-1};
    int32_t error_code{0};
    uint32_t business_returns{0};
    bool finish_first{false};
};

static_assert(sizeof(SchedulerAsyncLocalState) == 56);

inline __aicore__ SchedulerGraphResult scheduler_async_error_result(const SchedulerAsyncLocalState *local) {
    if (local == nullptr) return SchedulerGraphResult::INVALID_ARGUMENTS;
    switch (local->error_code) {
    case SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW:
        return SchedulerGraphResult::ASYNC_WAIT_OVERFLOW;
    case SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED:
        return SchedulerGraphResult::ASYNC_REGISTRATION_FAILED;
    case SIMPLER_ERROR_ASYNC_COMPLETION_INVALID:
        return SchedulerGraphResult::ASYNC_COMPLETION_INVALID;
    default:
        return SchedulerGraphResult::INVALID_ARGUMENTS;
    }
}

namespace scheduler_async_detail {

using namespace pto2::resident;

inline __aicore__ bool
fail(SchedulerAsyncLocalState &local, int64_t task_ref, int32_t code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID) {
    if (local.error_code == 0) {
        local.failed_task_ref = task_ref;
        local.error_code = code == 0 ? SIMPLER_ERROR_ASYNC_COMPLETION_INVALID : code;
    }
    return false;
}

inline __aicore__ bool span(uint64_t bytes, uint64_t offset, uint64_t count, uint64_t stride, uint64_t alignment) {
    return offset % alignment == 0 && offset <= bytes && count <= (bytes - offset) / stride;
}

inline __aicore__ void observe(__gm__ const void *address, uint64_t bytes) {
    if (bytes == 0) return;
    const uint64_t first = reinterpret_cast<uint64_t>(address) & ~UINT64_C(63);
    const uint64_t end = reinterpret_cast<uint64_t>(address) + bytes;
    for (uint64_t line = first; line < end; line += 64)
        scheduler_invalidate_cache_line(reinterpret_cast<__gm__ void *>(line));
    scheduler_cache_barrier();
}

inline __aicore__ void publish(__gm__ void *address, uint64_t bytes) {
    // Publication spans start/end on isolated lines and exclude atomic words.
    for (uint64_t offset = 0; offset < bytes; offset += 64)
        scheduler_writeback_cache_line(reinterpret_cast<__gm__ uint8_t *>(address) + offset);
    scheduler_cache_barrier();
}

inline __aicore__ bool task_index(SchedulerAsyncLocalState &local, int64_t task_ref, uint32_t &index) {
    index = ASYNC_INVALID_INDEX;
    if (local.error_code != 0 || local.layout == nullptr || task_ref < 0 ||
        static_cast<uint64_t>(task_ref) >= local.layout->task_count)
        return fail(local, task_ref);
    auto *map = scheduler_state_at<uint32_t>(local.base, local.layout->task_map_offset);
    observe(&map[task_ref], sizeof(uint32_t));
    index = map[task_ref];
    if (index == ASYNC_INVALID_INDEX) return true;
    if (index >= local.layout->async_task_count) return fail(local, task_ref);
    auto *tasks = scheduler_state_at<SchedulerAsyncTaskData>(local.base, local.layout->task_data_offset);
    observe(&tasks[index], sizeof(SchedulerAsyncTaskData));
    return tasks[index].task_ref == task_ref || fail(local, task_ref);
}

inline __aicore__ bool activate_finish(SchedulerAsyncLocalState &local, uint32_t index) {
    auto *leases = scheduler_state_at<SchedulerAsyncLease>(local.base, local.layout->finish_leases_offset);
    auto *directory =
        scheduler_state_at<SchedulerAsyncDirectoryWord>(local.base, local.layout->finish_directory_offset);
    scheduler_cache_barrier();
    if (scheduler_async_lease_publish(leases[index]).result != SchedulerAsyncLeaseResult::SUCCESS ||
        scheduler_async_directory_publish(directory[index / 64].bits, index % 64) !=
            SchedulerAsyncDirectoryResult::SUCCESS)
        return fail(local, -1);
    return true;
}

inline __aicore__ bool settle_group(SchedulerAsyncLocalState &local, int64_t task_ref) {
    uint32_t index = ASYNC_INVALID_INDEX;
    if (!task_index(local, task_ref, index) || index == ASYNC_INVALID_INDEX) return fail(local, task_ref);
    auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(local.base, local.layout->joins_offset);
    scheduler_cache_barrier();
    const auto settled = scheduler_async_join_settle_group(joins[index]);
    if (settled.result == SchedulerAsyncJoinResult::FINISH_OWNER) return activate_finish(local, index);
    return settled.result == SchedulerAsyncJoinResult::PENDING || fail(local, task_ref);
}

// A directory scan visits one word. Cursors advance past a claimed record so
// a low pending bit cannot starve a later condition in the same word.
inline __aicore__ bool claim(
    SchedulerAsyncLocalState &local, uint64_t directory_offset, uint64_t lease_offset, uint32_t count, uint32_t &cursor,
    uint32_t &index, uint64_t &owned
) {
    index = ASYNC_INVALID_INDEX;
    owned = 0;
    if (count == 0 || cursor >= count) return fail(local, -1);
    auto *words = scheduler_state_at<SchedulerAsyncDirectoryWord>(local.base, directory_offset);
    const uint32_t word = cursor / 64;
    const uint32_t bit = cursor % 64;
    const uint64_t mask = scheduler_gm_query(words[word].bits) & (UINT64_MAX << bit);
    const uint64_t next_word = static_cast<uint64_t>(word + 1) * 64;
    cursor = next_word < count ? static_cast<uint32_t>(next_word) : 0;
    if (mask == 0) return true;
    const uint32_t found = word * 64 + static_cast<uint32_t>(__builtin_ctzll(mask));
    if (found >= count) return fail(local, -1);
    cursor = found + 1 == count ? 0 : found + 1;
    auto *leases = scheduler_state_at<SchedulerAsyncLease>(local.base, lease_offset);
    const auto result = scheduler_async_lease_try_claim(leases[found], scheduler_async_lease_load(leases[found]));
    if (result.result == SchedulerAsyncLeaseResult::NOT_RUNNABLE ||
        result.result == SchedulerAsyncLeaseResult::CONTENDED)
        return true;
    if (result.result != SchedulerAsyncLeaseResult::SUCCESS) return fail(local, -1);
    scheduler_cache_barrier();
    index = found;
    owned = result.word;
    return true;
}

}  // namespace scheduler_async_detail

inline __aicore__ bool scheduler_initialize_async(
    __gm__ void *base, uint64_t bytes, uint64_t offset, uint64_t epoch, uint64_t tasks, SchedulerAsyncLocalState &local
) {
    using namespace scheduler_async_detail;
    local = SchedulerAsyncLocalState{};
    if (offset == 0) return true;
    if (base == nullptr || reinterpret_cast<uintptr_t>(base) % 128 != 0 ||
        bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(base) ||
        !span(bytes, offset, 1, sizeof(SchedulerAsyncLayout), 128))
        return fail(local, -1);
    auto *layout = scheduler_state_at<SchedulerAsyncLayout>(base, offset);
    observe(layout, sizeof(*layout));
    if (epoch == 0 || layout->run_epoch != epoch || layout->image_bytes != bytes || layout->task_count != tasks ||
        layout->async_task_count == 0 || layout->async_task_count > tasks ||
        layout->group_count < layout->async_task_count ||
        (layout->sdma_guard_count != 0 && !span(
                                              bytes, layout->sdma_guards_offset, layout->sdma_guard_count,
                                              sizeof(pto2::resident::AsyncSdmaSubmissionGuard), 128
                                          )) ||
        !span(bytes, layout->task_map_offset, tasks, sizeof(uint32_t), alignof(uint32_t)) ||
        !span(
            bytes, layout->task_data_offset, layout->async_task_count, sizeof(SchedulerAsyncTaskData),
            alignof(SchedulerAsyncTaskData)
        ) ||
        !span(bytes, layout->group_offsets_offset, layout->group_count, sizeof(uint64_t), alignof(uint64_t)) ||
        !span(bytes, layout->group_leases_offset, layout->group_count, sizeof(SchedulerAsyncLease), 128) ||
        !span(bytes, layout->group_progress_offset, layout->group_count, sizeof(SchedulerAsyncGroupProgress), 128) ||
        !span(bytes, layout->joins_offset, layout->async_task_count, sizeof(SchedulerAsyncTaskJoin), 128) ||
        !span(bytes, layout->finish_leases_offset, layout->async_task_count, sizeof(SchedulerAsyncLease), 128) ||
        !span(
            bytes, layout->finish_progress_offset, layout->async_task_count, sizeof(SchedulerAsyncFinishProgress), 128
        ) ||
        !span(
            bytes, layout->check_directory_offset, (static_cast<uint64_t>(layout->group_count) + 63) / 64, 128, 128
        ) ||
        !span(
            bytes, layout->finish_directory_offset, (static_cast<uint64_t>(layout->async_task_count) + 63) / 64, 128,
            128
        ))
        return fail(local, -1);
    local.base = static_cast<__gm__ uint8_t *>(base);
    local.bytes = bytes;
    local.layout = layout;
    return true;
}

inline __aicore__ bool scheduler_async_execution_writer(
    SchedulerAsyncLocalState &local, int64_t task_ref, uint32_t block, uint32_t lane,
    pto2::resident::AsyncWriter &writer, bool observe_header = true
) {
    using namespace scheduler_async_detail;
    writer = AsyncWriter{};
    uint32_t index = ASYNC_INVALID_INDEX;
    if (!task_index(local, task_ref, index)) return false;
    if (index == ASYNC_INVALID_INDEX) return true;
    auto *tasks = scheduler_state_at<SchedulerAsyncTaskData>(local.base, local.layout->task_data_offset);
    const auto &task = tasks[index];
    const uint8_t mask = *reinterpret_cast<__gm__ const uint8_t *>(&task.async_mask);
    if (lane >= 3 || mask == 0 || (mask & ~UINT8_C(7)) != 0 || task.reserved != 0 || block >= task.block_count)
        return fail(local, task_ref);
    if ((mask & (1U << lane)) == 0) return true;
    const uint64_t wide_group_index = static_cast<uint64_t>(task.group_base) +
                                      block * static_cast<uint32_t>(__builtin_popcount(mask)) +
                                      static_cast<uint32_t>(__builtin_popcount(mask & ((1U << lane) - 1)));
    if (wide_group_index >= local.layout->group_count) return fail(local, task_ref);
    const uint32_t group_index = static_cast<uint32_t>(wide_group_index);
    auto *offsets = scheduler_state_at<uint64_t>(local.base, local.layout->group_offsets_offset);
    observe(&offsets[group_index], sizeof(uint64_t));
    const uint64_t offset = offsets[group_index];
    if (!span(local.bytes, offset, 1, sizeof(AsyncGroupData), 128)) return fail(local, task_ref);
    auto *data = scheduler_state_at<AsyncGroupData>(local.base, offset);
    if (observe_header) observe(data, sizeof(*data));
    if (data->run_epoch != local.layout->run_epoch || data->task_ref != task_ref || data->group_index != group_index ||
        data->block_index != block || data->subtask_slot != lane ||
        async_group_publication_bytes(data->capacity) == 0 ||
        data->publication_bytes != async_group_publication_bytes(data->capacity) ||
        !span(local.bytes, offset, data->publication_bytes, 1, 128))
        return fail(local, task_ref);
    writer = {local.base, local.bytes, local.layout->run_epoch, offset, group_index};
    return true;
}

inline __aicore__ bool
scheduler_async_task_is_deferred(SchedulerAsyncLocalState *local, int64_t task_ref, bool &deferred) {
    using namespace scheduler_async_detail;
    deferred = false;
    if (local == nullptr) return true;
    uint32_t index = ASYNC_INVALID_INDEX;
    if (!task_index(*local, task_ref, index)) return false;
    if (index == ASYNC_INVALID_INDEX) return true;
    deferred = true;
    auto *tasks = scheduler_state_at<SchedulerAsyncTaskData>(local->base, local->layout->task_data_offset);
    const auto &task = tasks[index];
    const uint8_t mask = *reinterpret_cast<__gm__ const uint8_t *>(&task.async_mask);
    if (task.block_count != 1 || __builtin_popcount(mask) != 1) return true;
    AsyncWriter writer{};
    if (!scheduler_async_execution_writer(*local, task_ref, 0, static_cast<uint32_t>(__builtin_ctz(mask)), writer))
        return false;
    auto *data = pto2::resident::detail::group(writer);
    // A sealed singleton has no other execution that could still register a
    // condition. Preserve synchronous completion and direct successor refill
    // when this execution registered none; multi-execution tasks retain the join.
    deferred = data->state != AsyncPublicationState::SEALED || data->count != 0 || data->error_code != 0 ||
               data->reserved != 0;
    return true;
}

inline __aicore__ bool scheduler_async_begin_execution(
    SchedulerAsyncLocalState *local, int64_t task_ref, uint32_t block, uint32_t lane, __gm__ GlobalContext &global
) {
    using namespace scheduler_async_detail;
    if (local == nullptr) return true;
    global.async_present = 0;
    AsyncWriter writer{};
    if (!scheduler_async_execution_writer(*local, task_ref, block, lane, writer)) return false;
    if (writer.image == nullptr) return true;
    if (!pto2::resident::detail::valid(writer)) return fail(*local, task_ref, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    global.async_image_address = reinterpret_cast<uint64_t>(writer.image);
    global.async_image_bytes = writer.image_bytes;
    global.async_run_epoch = writer.run_epoch;
    global.async_group_offset = writer.group_offset;
    global.async_group_index = writer.group_index;
    global.async_reserved = 0;
    global.async_present = 1;
    return true;
}

inline __aicore__ bool
scheduler_async_seal_execution(SchedulerAsyncLocalState *local, int64_t task_ref, uint32_t block, uint32_t lane) {
    using namespace scheduler_async_detail;
    if (local == nullptr) return true;
    AsyncWriter writer{};
    // The returning kernel owns dirty count/entry lines until this seal publishes them.
    if (!scheduler_async_execution_writer(*local, task_ref, block, lane, writer, false)) return false;
    if (writer.image == nullptr) return true;
    auto *data = pto2::resident::detail::group(writer);
    auto *leases = scheduler_state_at<SchedulerAsyncLease>(local->base, local->layout->group_leases_offset);
    if (!seal_conditions(writer)) {
        scheduler_async_lease_fail(
            leases[writer.group_index], scheduler_async_lease_encode(1, SchedulerAsyncPhase::DORMANT)
        );
        return fail(*local, task_ref, data->error_code);
    }
    auto *progress = scheduler_state_at<SchedulerAsyncGroupProgress>(local->base, local->layout->group_progress_offset);
    progress[writer.group_index].pending_mask = scheduler_async_condition_mask(data->count);
    progress[writer.group_index].condition_cursor = 0;
    progress[writer.group_index].acquire_cursor = 0;
    publish(data, data->publication_bytes);
    publish(&progress[writer.group_index], sizeof(SchedulerAsyncGroupProgress));
    if (data->count == 0) {
        if (scheduler_async_lease_finish_dormant(leases[writer.group_index]).result !=
            SchedulerAsyncLeaseResult::SUCCESS)
            return fail(*local, task_ref);
        return settle_group(*local, task_ref);
    }
    auto *directory =
        scheduler_state_at<SchedulerAsyncDirectoryWord>(local->base, local->layout->check_directory_offset);
    if (scheduler_async_lease_publish(leases[writer.group_index]).result != SchedulerAsyncLeaseResult::SUCCESS ||
        scheduler_async_directory_publish(directory[writer.group_index / 64].bits, writer.group_index % 64) !=
            SchedulerAsyncDirectoryResult::SUCCESS)
        return fail(*local, task_ref);
    return true;
}

inline __aicore__ bool scheduler_async_release_block(SchedulerAsyncLocalState *local, int64_t task_ref) {
    using namespace scheduler_async_detail;
    if (local == nullptr) return true;
    uint32_t index = ASYNC_INVALID_INDEX;
    if (!task_index(*local, task_ref, index)) return false;
    if (index == ASYNC_INVALID_INDEX) return true;
    auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(local->base, local->layout->joins_offset);
    scheduler_cache_barrier();
    const auto released = scheduler_async_join_release_block(joins[index]);
    if (released.result == SchedulerAsyncJoinResult::FINISH_OWNER) {
        bool deferred = false;
        if (!scheduler_async_task_is_deferred(local, task_ref, deferred)) return false;
        if (deferred) return activate_finish(*local, index);
        auto *leases = scheduler_state_at<SchedulerAsyncLease>(local->base, local->layout->finish_leases_offset);
        return scheduler_async_lease_finish_dormant(leases[index]).result == SchedulerAsyncLeaseResult::SUCCESS ||
               fail(*local, task_ref);
    }
    return released.result == SchedulerAsyncJoinResult::PENDING || fail(*local, task_ref);
}

struct SchedulerAsyncCompletionObserver {
    inline __aicore__ CompletionPollResult operator()(const CompletionToken &token, uint32_t observer_mask) const {
        if (observer_mask == 0 || (observer_mask & ~3U) != 0 || !pto2::resident::detail::token_valid(token))
            return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
        if (token.completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD) {
            auto *record = reinterpret_cast<__gm__ uint64_t *>(token.addr);
#if defined(__CCE_AICORE__)
            const uint64_t completed = __builtin_cce_ld_dev(record, 0);
#else
            const uint64_t completed = __atomic_load_n(record, __ATOMIC_ACQUIRE);
#endif
            return {completed >= token.backend_cookie ? CompletionPollState::READY : CompletionPollState::PENDING, 0};
        }
        auto *counter = reinterpret_cast<__gm__ uint32_t *>(token.addr);
#if defined(__CCE_AICORE__)
        const uint32_t value = static_cast<uint32_t>(__builtin_cce_ld_dev(counter, 0));
#else
        const uint32_t value = __atomic_load_n(counter, __ATOMIC_ACQUIRE);
#endif
        if (value > INT32_MAX) return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
        return {value >= token.expected_value ? CompletionPollState::READY : CompletionPollState::PENDING, 0};
    }
};

template <typename Observer>
inline __aicore__ bool scheduler_async_check_step(
    SchedulerAsyncLocalState &local, uint32_t observer_mask, uint32_t budget, const Observer &observer,
    bool &made_progress
) {
    using namespace scheduler_async_detail;
    made_progress = false;
    if (local.error_code != 0 || local.layout == nullptr || budget == 0 || observer_mask == 0 ||
        (observer_mask & ~3U) != 0)
        return fail(local, -1);
    uint32_t index = ASYNC_INVALID_INDEX;
    uint64_t owned = 0;
    if (!claim(
            local, local.layout->check_directory_offset, local.layout->group_leases_offset, local.layout->group_count,
            local.check_cursor, index, owned
        ))
        return false;
    if (index == ASYNC_INVALID_INDEX) return true;
    auto *leases = scheduler_state_at<SchedulerAsyncLease>(local.base, local.layout->group_leases_offset);
    auto failed = [&](int64_t ref, int32_t code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID) __aicore__ {
        scheduler_async_lease_fail(leases[index], owned);
        return fail(local, ref, code);
    };
    auto *offsets = scheduler_state_at<uint64_t>(local.base, local.layout->group_offsets_offset);
    observe(&offsets[index], sizeof(uint64_t));
    const uint64_t offset = offsets[index];
    if (!span(local.bytes, offset, 1, sizeof(AsyncGroupData), 128)) return failed(-1);
    auto *data = scheduler_state_at<AsyncGroupData>(local.base, offset);
    observe(data, sizeof(*data));
    const int64_t task_ref = data->task_ref;
    AsyncWriter writer{local.base, local.bytes, local.layout->run_epoch, offset, index};
    if (task_ref < 0 || static_cast<uint64_t>(task_ref) >= local.layout->task_count ||
        pto2::resident::detail::group(writer) == nullptr || data->state != AsyncPublicationState::SEALED ||
        data->error_code != 0 || data->reserved != 0 || data->count == 0 || data->count > data->capacity ||
        async_group_publication_bytes(data->capacity) == 0 ||
        data->publication_bytes != async_group_publication_bytes(data->capacity) ||
        !span(local.bytes, offset, async_group_publication_bytes(data->capacity), 1, 128))
        return failed(task_ref);
    auto *progress = scheduler_state_at<SchedulerAsyncGroupProgress>(local.base, local.layout->group_progress_offset);
    auto &current = progress[index];
    observe(&current, sizeof(current));
    uint64_t pending = current.pending_mask;
    uint32_t cursor = current.condition_cursor;
    if (pending == 0 || (pending & ~scheduler_async_condition_mask(data->count)) != 0 || cursor >= 64 ||
        current.acquire_cursor != 0)
        return failed(task_ref);
    auto *entries = scheduler_state_at<DeferredCompletionEntry>(local.base, offset + sizeof(AsyncGroupData));
    uint64_t visited = 0;
    for (uint32_t used = 0; used < budget && (pending & ~visited) != 0; ++used) {
        uint64_t candidates = pending & ~visited;
        const uint64_t later = candidates & (UINT64_MAX << cursor);
        if (later != 0) candidates = later;
        const uint32_t condition = static_cast<uint32_t>(__builtin_ctzll(candidates));
        visited |= UINT64_C(1) << condition;
        cursor = (condition + 1) % 64;
        observe(&entries[condition], sizeof(DeferredCompletionEntry));
        const auto &entry = entries[condition];
        const CompletionToken token{
            entry.addr, entry.expected_value, entry.engine, entry.completion_type, entry.backend_cookie
        };
        if (entry._pad != 0 || !pto2::resident::detail::token_valid(token)) return failed(task_ref);
        const CompletionPollResult result = observer(token, observer_mask);
        if (result.state == CompletionPollState::FAILED || result.error_code != 0)
            return failed(task_ref, result.error_code);
        if (result.state == CompletionPollState::READY) {
            pending &= ~(UINT64_C(1) << condition);
            made_progress = true;
        } else if (result.state != CompletionPollState::PENDING) return failed(task_ref);
    }
    current.pending_mask = pending;
    current.condition_cursor = cursor;
    publish(&current, sizeof(current));
    if (pending != 0)
        return scheduler_async_lease_release(leases[index], owned).result == SchedulerAsyncLeaseResult::SUCCESS ||
               failed(task_ref);
    auto *directory = scheduler_state_at<SchedulerAsyncDirectoryWord>(local.base, local.layout->check_directory_offset);
    if (scheduler_async_lease_finish(leases[index], owned).result != SchedulerAsyncLeaseResult::SUCCESS ||
        scheduler_async_directory_clear(directory[index / 64].bits, index % 64) !=
            SchedulerAsyncDirectoryResult::SUCCESS)
        return fail(local, task_ref);
    return settle_group(local, task_ref);
}
