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

#include "async_completion_plan.h"

#include <cstring>
#include <limits>
#include <unordered_set>
#include <utility>

#include "async_condition_writer.h"

namespace {

using namespace pto2::resident;

bool reject(std::string *error, const char *message) {
    if (error != nullptr) *error = message;
    return false;
}

template <typename T>
bool reserve_array(uint64_t *cursor, uint64_t count, uint64_t *offset) {
    uint64_t bytes = 0;
    return scheduler_layout_checked_mul(count, sizeof(T), &bytes) &&
           scheduler_layout_reserve(cursor, bytes, alignof(T), offset);
}

}  // namespace

bool plan_async_completion(
    uint64_t begin_offset, uint64_t run_epoch, uint32_t task_count, const std::vector<AsyncTaskPlanInput> &inputs,
    AsyncCompletionHostPlan *out, std::string *error, uint32_t sdma_guard_count
) {
    if (out == nullptr || run_epoch == 0 || begin_offset < sizeof(SchedulerRunControl) ||
        begin_offset % ASYNC_RECORD_ALIGNMENT != 0 || task_count > INT32_MAX)
        return reject(error, "invalid async image identity or bounds");
    AsyncCompletionHostPlan next;
    next.begin_offset = begin_offset;
    next.total_bytes = begin_offset;
    auto &layout = next.layout;
    layout.run_epoch = run_epoch;
    layout.task_count = task_count;
    std::unordered_set<int64_t> task_refs;
    for (const auto &input : inputs) {
        if (input.task_ref < 0 || static_cast<uint64_t>(input.task_ref) >= task_count ||
            !task_refs.insert(input.task_ref).second || input.block_count == 0 ||
            input.block_count > SCHEDULER_ASYNC_JOIN_COUNT_MAX || !input.active_mask.is_valid())
            return reject(error, "invalid or duplicate async task shape");
        LaneMask mask{};
        uint32_t lane_count = 0;
        if (input.block_count * static_cast<uint32_t>(input.active_mask.count()) > SCHEDULER_ASYNC_JOIN_COUNT_MAX)
            return reject(error, "task physical execution count exceeds resident capacity");
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (!input.active_mask.has(static_cast<SubtaskSlot>(lane))) continue;
            const uint16_t capacity = input.capacities[lane];
            if (capacity > MAX_COMPLETIONS_PER_TASK) return reject(error, "invalid completion capacity");
            if (capacity == 0) continue;
            mask.activate(static_cast<SubtaskSlot>(lane));
            ++lane_count;
        }
        if (mask.is_empty()) continue;
        const uint32_t groups = input.block_count * lane_count;
        if (groups > SCHEDULER_ASYNC_JOIN_COUNT_MAX || groups > UINT32_MAX - layout.group_count)
            return reject(error, "async group count exceeds static join or index capacity");
        next.tasks.push_back({input.task_ref, layout.group_count, static_cast<uint16_t>(input.block_count), mask, 0});
        layout.group_count += groups;
    }
    if (next.tasks.empty()) {
        *out = std::move(next);
        return true;
    }
    layout.async_task_count = static_cast<uint32_t>(next.tasks.size());
    layout.sdma_guard_count = sdma_guard_count;
    uint64_t cursor = begin_offset;
    if (!reserve_array<SchedulerAsyncLayout>(&cursor, 1, &next.layout_offset) ||
        !reserve_array<uint32_t>(&cursor, task_count, &layout.task_map_offset) ||
        !reserve_array<SchedulerAsyncTaskData>(&cursor, layout.async_task_count, &layout.task_data_offset) ||
        !reserve_array<uint64_t>(&cursor, layout.group_count, &layout.group_offsets_offset) ||
        !reserve_array<SchedulerAsyncLease>(&cursor, layout.group_count, &layout.group_leases_offset) ||
        !reserve_array<SchedulerAsyncGroupProgress>(&cursor, layout.group_count, &layout.group_progress_offset) ||
        !reserve_array<SchedulerAsyncTaskJoin>(&cursor, layout.async_task_count, &layout.joins_offset) ||
        !reserve_array<SchedulerAsyncLease>(&cursor, layout.async_task_count, &layout.finish_leases_offset) ||
        !reserve_array<SchedulerAsyncFinishProgress>(
            &cursor, layout.async_task_count, &layout.finish_progress_offset
        ) ||
        !reserve_array<SchedulerAsyncDirectoryWord>(
            &cursor, (static_cast<uint64_t>(layout.group_count) + 63) / 64, &layout.check_directory_offset
        ) ||
        !reserve_array<SchedulerAsyncDirectoryWord>(
            &cursor, (static_cast<uint64_t>(layout.async_task_count) + 63) / 64, &layout.finish_directory_offset
        ))
        return reject(error, "async control storage overflows");
    if (sdma_guard_count != 0 &&
        !reserve_array<AsyncSdmaSubmissionGuard>(&cursor, sdma_guard_count, &layout.sdma_guards_offset))
        return reject(error, "SDMA submission guards overflow");
    uint32_t task_index = 0;
    for (const auto &input : inputs) {
        if (task_index == next.tasks.size() || input.task_ref != next.tasks[task_index].task_ref) continue;
        const auto &task = next.tasks[task_index++];
        for (uint32_t block = 0; block < input.block_count; ++block) {
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (!task.async_mask.has(static_cast<SubtaskSlot>(lane))) continue;
                AsyncGroupHostPlan group{};
                auto &data = group.data;
                data.run_epoch = run_epoch;
                data.task_ref = input.task_ref;
                data.group_index = static_cast<uint32_t>(next.groups.size());
                data.block_index = block;
                data.subtask_slot = lane;
                data.capacity = input.capacities[lane];
                data.publication_bytes = async_group_publication_bytes(data.capacity);
                data.sdma_guards_offset = layout.sdma_guards_offset;
                data.sdma_guard_count = layout.sdma_guard_count;
                if (!scheduler_layout_reserve(&cursor, data.publication_bytes, ASYNC_RECORD_ALIGNMENT, &group.offset))
                    return reject(error, "async publication storage overflows");
                next.groups.push_back(group);
            }
        }
    }
    if (!scheduler_layout_checked_align(cursor, ASYNC_RECORD_ALIGNMENT, &next.total_bytes) ||
        next.total_bytes > std::numeric_limits<size_t>::max())
        return reject(error, "async image size cannot be represented");
    layout.image_bytes = next.total_bytes;
    *out = std::move(next);
    return true;
}

bool initialize_async_completion(void *image, uint64_t image_bytes, const AsyncCompletionHostPlan &plan) {
    if (plan.layout_offset == 0)
        return plan.tasks.empty() && plan.groups.empty() && plan.total_bytes == plan.begin_offset;
    if (image == nullptr || reinterpret_cast<uintptr_t>(image) % ASYNC_RECORD_ALIGNMENT != 0 ||
        image_bytes < plan.total_bytes || plan.begin_offset >= plan.total_bytes ||
        plan.layout.image_bytes != plan.total_bytes || plan.layout.group_count != plan.groups.size() ||
        plan.layout.async_task_count != plan.tasks.size())
        return false;
    *scheduler_state_at<SchedulerAsyncLayout>(image, plan.layout_offset) = plan.layout;
    if (plan.layout.sdma_guard_count != 0) {
        auto *guards = scheduler_state_at<AsyncSdmaSubmissionGuard>(image, plan.layout.sdma_guards_offset);
        for (uint32_t index = 0; index < plan.layout.sdma_guard_count; ++index)
            guards[index].word = 0;
    }
    auto *map = scheduler_state_at<uint32_t>(image, plan.layout.task_map_offset);
    for (uint32_t ref = 0; ref < plan.layout.task_count; ++ref)
        map[ref] = ASYNC_INVALID_INDEX;
    auto *tasks = scheduler_state_at<SchedulerAsyncTaskData>(image, plan.layout.task_data_offset);
    auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(image, plan.layout.joins_offset);
    auto *finish_leases = scheduler_state_at<SchedulerAsyncLease>(image, plan.layout.finish_leases_offset);
    auto *finish_progress = scheduler_state_at<SchedulerAsyncFinishProgress>(image, plan.layout.finish_progress_offset);
    for (uint32_t index = 0; index < plan.tasks.size(); ++index) {
        const auto &task = plan.tasks[index];
        tasks[index] = task;
        map[task.task_ref] = index;
        uint64_t initial = 0;
        const uint32_t groups = task.block_count * static_cast<uint32_t>(task.async_mask.count());
        if (!scheduler_async_join_initial_word(task.block_count, groups, initial)) return false;
        joins[index].word = initial;
        finish_leases[index].word = scheduler_async_lease_encode(1, SchedulerAsyncPhase::DORMANT);
        finish_progress[index] = SchedulerAsyncFinishProgress{};
    }
    auto *offsets = scheduler_state_at<uint64_t>(image, plan.layout.group_offsets_offset);
    auto *leases = scheduler_state_at<SchedulerAsyncLease>(image, plan.layout.group_leases_offset);
    auto *progress = scheduler_state_at<SchedulerAsyncGroupProgress>(image, plan.layout.group_progress_offset);
    auto *checks = scheduler_state_at<SchedulerAsyncDirectoryWord>(image, plan.layout.check_directory_offset);
    auto *finishes = scheduler_state_at<SchedulerAsyncDirectoryWord>(image, plan.layout.finish_directory_offset);
    for (uint64_t word = 0; word < (static_cast<uint64_t>(plan.layout.group_count) + 63) / 64; ++word)
        checks[word].bits = 0;
    for (uint64_t word = 0; word < (static_cast<uint64_t>(plan.layout.async_task_count) + 63) / 64; ++word)
        finishes[word].bits = 0;
    for (uint32_t index = 0; index < plan.groups.size(); ++index) {
        const auto &group = plan.groups[index];
        offsets[index] = group.offset;
        progress[index] = SchedulerAsyncGroupProgress{};
        leases[index].word = scheduler_async_lease_encode(1, SchedulerAsyncPhase::DORMANT);
        *scheduler_state_at<AsyncGroupData>(image, group.offset) = group.data;
    }
    return true;
}
