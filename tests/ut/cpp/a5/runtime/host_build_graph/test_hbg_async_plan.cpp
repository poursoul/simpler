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

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "async_completion_plan.h"
#include "async_condition_writer.h"
#include "backend/sdma/sdma_submission_guard.h"

namespace {

using namespace pto2::resident;

AsyncTaskPlanInput task(int64_t ref, uint32_t blocks, LaneMask active) {
    AsyncTaskPlanInput input;
    input.task_ref = ref;
    input.block_count = blocks;
    input.active_mask = active;
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (active.has(static_cast<SubtaskSlot>(lane))) input.capacities[lane] = static_cast<uint16_t>(lane + 1);
    return input;
}

struct Image {
    std::vector<uint8_t> backing;
    void *base;

    explicit Image(uint64_t size) :
        backing(size + 127, 0xa5) {
        const uintptr_t aligned = (reinterpret_cast<uintptr_t>(backing.data()) + 127) & ~uintptr_t(127);
        base = reinterpret_cast<void *>(aligned);
    }
};

constexpr uint64_t kPrefix = sizeof(SchedulerRunControl);

TEST(HbgAsyncPlan, SdmaGuardsAreSharedByExecutionsAndIsolatedByCluster) {
    AsyncCompletionHostPlan plan;
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 1, 1, {task(0, 2, LaneMask::both_aiv())}, &plan, &error, 32));
    Image image(plan.total_bytes);
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    EXPECT_EQ(plan.layout.sdma_guard_count, 32u);
    EXPECT_EQ(plan.layout.sdma_guards_offset % 128, 0u);
    AsyncSdmaSubmissionGuard *first = nullptr;
    for (const auto &group : plan.groups) {
        AsyncWriter writer{
            static_cast<uint8_t *>(image.base), plan.total_bytes, 1, group.offset, group.data.group_index
        };
        auto *guard = sdma_submission_guard(writer, 7);
        ASSERT_NE(guard, nullptr);
        if (first == nullptr) first = guard;
        EXPECT_EQ(guard, first);
        EXPECT_NE(sdma_submission_guard(writer, 8), guard);
        EXPECT_EQ(guard->word, 0u);
    }
    const auto &group = plan.groups[0];
    AsyncWriter writer{static_cast<uint8_t *>(image.base), plan.total_bytes, 1, group.offset, 0};
    EXPECT_EQ(sdma_submission_guard(writer, 32), nullptr);
    EXPECT_FALSE(seal_conditions(writer));
}

TEST(HbgAsyncPlan, SdmaGuardSerializesSessionStateAndRetainsFailure) {
    AsyncSdmaSubmissionGuard guard{};
    uint64_t post_id = 0;
    std::atomic<bool> failed{false};
    std::vector<std::thread> submitters;
    for (uint32_t lane = 0; lane < 2; ++lane) {
        submitters.emplace_back([&]() {
            for (uint32_t iteration = 0; iteration < 2000; ++iteration) {
                SdmaGuardResult result;
                do {
                    result = sdma_try_acquire(guard);
                } while (result == SdmaGuardResult::BUSY);
                if (result != SdmaGuardResult::ACQUIRED) {
                    failed = true;
                    return;
                }
                ++post_id;
                if (!sdma_release(guard, true)) failed = true;
            }
        });
    }
    for (auto &thread : submitters)
        thread.join();
    EXPECT_FALSE(failed);
    EXPECT_EQ(post_id, 4000u);
    EXPECT_EQ(sdma_try_acquire(guard), SdmaGuardResult::ACQUIRED);
    EXPECT_EQ(sdma_try_acquire(guard), SdmaGuardResult::BUSY);
    EXPECT_TRUE(sdma_release(guard, false));
    EXPECT_EQ(sdma_try_acquire(guard), SdmaGuardResult::FAILED);
    EXPECT_EQ(guard.word, 2u);
}

TEST(HbgAsyncPlan, SynchronousAndControlOnlyRunsAppendNoStorage) {
    AsyncCompletionHostPlan plan;
    std::string error;
    auto synchronous = task(1, 4, LaneMask::aic());
    synchronous.capacities[0] = 0;
    EXPECT_TRUE(plan_async_completion(kPrefix, 1, 3, {synchronous}, &plan, &error)) << error;
    EXPECT_EQ(plan.total_bytes, kPrefix);
    EXPECT_EQ(plan.layout_offset, 0u);
    EXPECT_TRUE(plan.groups.empty());
    EXPECT_TRUE(initialize_async_completion(nullptr, 0, plan));
    EXPECT_TRUE(plan_async_completion(kPrefix, 2, 3, {}, &plan, &error)) << error;
    EXPECT_EQ(plan.total_bytes, kPrefix);
}

TEST(HbgAsyncPlan, ExpandedTasksBlocksAndLanesHaveIndependentStableRecords) {
    const std::vector<AsyncTaskPlanInput> inputs{
        task(0, 2, LaneMask::aic()), task(2, 3, LaneMask::all()), task(5, 2, LaneMask::both_aiv())
    };
    AsyncCompletionHostPlan plan;
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 19, 7, inputs, &plan, &error)) << error;
    ASSERT_EQ(plan.groups.size(), 15u);
    ASSERT_EQ(plan.tasks.size(), 3u);
    EXPECT_EQ(plan.tasks[0].group_base, 0u);
    EXPECT_EQ(plan.tasks[1].group_base, 2u);
    EXPECT_EQ(plan.tasks[2].group_base, 11u);
    Image image(plan.total_bytes);
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    const auto *map = scheduler_state_at<uint32_t>(image.base, plan.layout.task_map_offset);
    EXPECT_EQ(map[0], 0u);
    EXPECT_EQ(map[1], ASYNC_INVALID_INDEX);
    EXPECT_EQ(map[2], 1u);
    EXPECT_EQ(map[5], 2u);
    EXPECT_EQ(map[6], ASYNC_INVALID_INDEX);
    uint32_t group_index = 0;
    for (const auto &input : inputs) {
        for (uint32_t block = 0; block < input.block_count; ++block) {
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (!input.active_mask.has(static_cast<SubtaskSlot>(lane))) continue;
                const auto &group = plan.groups[group_index];
                EXPECT_EQ(group.data.task_ref, input.task_ref);
                EXPECT_EQ(group.data.block_index, block);
                EXPECT_EQ(group.data.subtask_slot, lane);
                EXPECT_EQ(group.data.group_index, group_index);
                EXPECT_EQ(group.offset % 128, 0u);
                AsyncWriter writer{static_cast<uint8_t *>(image.base), plan.total_bytes, 19, group.offset, group_index};
                EXPECT_TRUE(seal_conditions(writer));
                if (group_index != 0)
                    EXPECT_GE(
                        group.offset,
                        plan.groups[group_index - 1].offset + plan.groups[group_index - 1].data.publication_bytes
                    );
                ++group_index;
            }
        }
    }
    const auto *leases = scheduler_state_at<SchedulerAsyncLease>(image.base, plan.layout.group_leases_offset);
    for (uint32_t index = 0; index < plan.groups.size(); ++index)
        EXPECT_EQ(leases[index].word, scheduler_async_lease_encode(1, SchedulerAsyncPhase::DORMANT));
    const auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(image.base, plan.layout.joins_offset);
    EXPECT_EQ(scheduler_async_join_blocks(joins[1].word), 3u);
    EXPECT_EQ(scheduler_async_join_groups(joins[1].word), 9u);
    EXPECT_EQ(scheduler_async_join_blocks(joins[2].word), 2u);
    EXPECT_EQ(scheduler_async_join_groups(joins[2].word), 4u);
    auto *finishes = scheduler_state_at<SchedulerAsyncFinishProgress>(image.base, plan.layout.finish_progress_offset);
    EXPECT_EQ(finishes[0].resolver.producer_ref, -1);
    EXPECT_EQ(finishes[0].resolver.phase, SchedulerResolvePhase::START);
    for (uint64_t i = 0; i < kPrefix; ++i)
        EXPECT_EQ(static_cast<uint8_t *>(image.base)[i], 0xa5);
}

TEST(HbgAsyncPlan, MixedCapabilityTaskCountsOnlyDeferredLanesAndAllBlocks) {
    auto input = task(0, 5, LaneMask::all());
    input.capacities[1] = 0;
    AsyncCompletionHostPlan plan;
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 2, 1, {input}, &plan, &error)) << error;
    EXPECT_EQ(plan.groups.size(), 10u);
    EXPECT_TRUE(plan.tasks[0].async_mask.has_aic());
    EXPECT_FALSE(plan.tasks[0].async_mask.has_aiv0());
    EXPECT_TRUE(plan.tasks[0].async_mask.has_aiv1());
    Image image(plan.total_bytes);
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    auto *join = scheduler_state_at<SchedulerAsyncTaskJoin>(image.base, plan.layout.joins_offset);
    EXPECT_EQ(scheduler_async_join_blocks(join->word), 5u);
    EXPECT_EQ(scheduler_async_join_groups(join->word), 10u);
}

TEST(HbgAsyncPlan, EveryMixMaskIncludesOnlyItsActiveLanes) {
    for (uint32_t mask : {3u, 5u, 6u, 7u}) {
        const LaneMask lanes((mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0);
        AsyncCompletionHostPlan plan;
        std::string error;
        ASSERT_TRUE(plan_async_completion(kPrefix, 3, 1, {task(0, 4, lanes)}, &plan, &error)) << error;
        EXPECT_EQ(plan.groups.size(), 4u * static_cast<uint32_t>(lanes.count()));
        EXPECT_EQ(plan.tasks[0].async_mask, lanes);
        for (const auto &group : plan.groups)
            EXPECT_TRUE(lanes.has(static_cast<SubtaskSlot>(group.data.subtask_slot)));
    }
}

TEST(HbgAsyncPlan, RelocationPreservesIndependentExecutionRecords) {
    AsyncCompletionHostPlan plan;
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 9, 1, {task(0, 2, LaneMask::aiv0())}, &plan, &error)) << error;
    Image first(plan.total_bytes), relocated(plan.total_bytes);
    ASSERT_TRUE(initialize_async_completion(first.base, plan.total_bytes, plan));
    std::memcpy(relocated.base, first.base, plan.total_bytes);
    AsyncWriter writer{static_cast<uint8_t *>(relocated.base), plan.total_bytes, 9, plan.groups[0].offset, 0};
    CompletionToken token{0x10000, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    ASSERT_TRUE(register_completion_condition(writer, token));
    EXPECT_EQ(scheduler_state_at<AsyncGroupData>(first.base, plan.groups[0].offset)->count, 0u);
    EXPECT_EQ(scheduler_state_at<AsyncGroupData>(relocated.base, plan.groups[1].offset)->count, 0u);
    EXPECT_TRUE(seal_conditions(writer));
}

TEST(HbgAsyncPlan, DirectoriesCoverWordBoundaryAndLeaseInitializationIsNonzero) {
    AsyncCompletionHostPlan plan;
    std::vector<AsyncTaskPlanInput> inputs;
    for (int64_t ref = 0; ref < 65; ++ref)
        inputs.push_back(task(ref, 1, LaneMask::aic()));
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 7, 65, inputs, &plan, &error)) << error;
    Image image(plan.total_bytes);
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    const auto *checks =
        scheduler_state_at<SchedulerAsyncDirectoryWord>(image.base, plan.layout.check_directory_offset);
    const auto *finishes =
        scheduler_state_at<SchedulerAsyncDirectoryWord>(image.base, plan.layout.finish_directory_offset);
    for (uint32_t word = 0; word < 2; ++word) {
        EXPECT_EQ(checks[word].bits, 0u);
        EXPECT_EQ(finishes[word].bits, 0u);
    }
    auto *leases = scheduler_state_at<SchedulerAsyncLease>(image.base, plan.layout.finish_leases_offset);
    for (uint32_t index = 0; index < 65; ++index)
        EXPECT_EQ(leases[index].word, UINT64_C(8));
    EXPECT_EQ(scheduler_async_condition_mask(0), 0u);
    EXPECT_EQ(scheduler_async_condition_mask(1), 1u);
    EXPECT_EQ(scheduler_async_condition_mask(63), UINT64_MAX >> 1);
    EXPECT_EQ(scheduler_async_condition_mask(64), UINT64_MAX);
}

TEST(HbgAsyncPlan, RejectsOverflowsDuplicateRefsAndInvalidStaticJoinWithoutReplacingOutput) {
    AsyncCompletionHostPlan plan;
    plan.total_bytes = 99;
    std::string error;
    const auto input = task(0, 1, LaneMask::aic());
    EXPECT_FALSE(plan_async_completion(UINT64_MAX - 127, 1, 1, {input}, &plan, &error));
    EXPECT_FALSE(plan_async_completion(kPrefix, 0, 1, {input}, &plan, &error));
    EXPECT_FALSE(plan_async_completion(kPrefix, 1, 1, {input, input}, &plan, &error));
    EXPECT_FALSE(plan_async_completion(kPrefix, 1, 0, {input}, &plan, &error));
    EXPECT_FALSE(plan_async_completion(kPrefix, 1, 1, {task(0, 10923, LaneMask::all())}, &plan, &error));
    EXPECT_FALSE(plan_async_completion(kPrefix, 1, 1, {task(0, 0, LaneMask::aic())}, &plan, &error));
    EXPECT_EQ(plan.total_bytes, 99u);
}

TEST(HbgAsyncPlan, InitializationChecksDestinationBeforeWritingAndResetsRetiredStorage) {
    AsyncCompletionHostPlan plan;
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 5, 1, {task(0, 1, LaneMask::aic())}, &plan, &error));
    Image image(plan.total_bytes);
    EXPECT_FALSE(initialize_async_completion(image.base, plan.total_bytes - 1, plan));
    EXPECT_FALSE(initialize_async_completion(static_cast<uint8_t *>(image.base) + 1, plan.total_bytes, plan));
    EXPECT_EQ(static_cast<uint8_t *>(image.base)[plan.begin_offset], 0xa5);
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    auto *data = scheduler_state_at<AsyncGroupData>(image.base, plan.groups[0].offset);
    data->count = 1;
    data->state = AsyncPublicationState::FAILED;
    ASSERT_TRUE(plan_async_completion(kPrefix, 6, 1, {task(0, 1, LaneMask::aic())}, &plan, &error));
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    EXPECT_EQ(data->run_epoch, 6u);
    EXPECT_EQ(data->count, 0u);
    EXPECT_EQ(static_cast<AsyncPublicationState>(data->state), AsyncPublicationState::OPEN);
}

}  // namespace
