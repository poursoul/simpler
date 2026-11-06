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

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "async_completion_plan.h"
#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_completion.h"
#include "scheduler/scheduler_async_finish.h"

namespace {

using namespace pto2::resident;

AsyncTaskPlanInput
async_task(int64_t task_ref, uint32_t blocks = 1, LaneMask mask = LaneMask::aic(), uint8_t capacity = 1) {
    AsyncTaskPlanInput input;
    input.task_ref = task_ref;
    input.block_count = blocks;
    input.active_mask = mask;
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (!mask.has(static_cast<SubtaskSlot>(lane))) continue;
        input.capacities[lane] = capacity;
    }
    return input;
}

struct AsyncFixture {
    alignas(128) std::array<uint32_t, 32> signal{};
    AsyncCompletionHostPlan plan;
    scheduler_test::FixtureStorage storage;
    scheduler_test::BasicGraphBuffer<128> graph;
    std::array<SchedulerAsyncLocalState, 3> workers{};

    explicit AsyncFixture(uint32_t tasks = 2, std::vector<AsyncTaskPlanInput> inputs = {async_task(0)}) :
        storage(
            tasks, 3, 1, {}, false,
            [&](AicoreSchedulerLayout &layout) {
                std::string error;
                EXPECT_TRUE(plan_async_completion(layout.total_size, 1, tasks, inputs, &plan, &error)) << error;
                layout.total_size = plan.total_bytes;
                layout.async_layout_offset = plan.layout_offset;
            }
        ),
        graph(tasks) {
        EXPECT_TRUE(initialize_async_completion(base(), storage.layout.total_size, plan));
        for (uint32_t ref = 0; ref < tasks; ++ref)
            graph.executable(ref, 0);
        for (const auto &input : inputs) {
            storage.metadata[input.task_ref].active_mask = input.active_mask;
            storage.metadata[input.task_ref].logical_block_num = input.block_count;
            for (uint32_t lane = 0; lane < 3; ++lane)
                graph.storage()[input.task_ref].task.kernel_id[lane] =
                    input.active_mask.has(static_cast<SubtaskSlot>(lane)) ? 1 : INVALID_KERNEL_ID;
        }
        for (uint32_t worker = 0; worker < 3; ++worker) {
            for (uint32_t lane = 0; lane < 3; ++lane)
                storage.contexts[worker].cluster_worker_ids[lane] = lane;
            EXPECT_TRUE(
                scheduler_initialize_async(base(), plan.total_bytes, plan.layout_offset, 1, tasks, workers[worker])
            );
            storage.local_context(&storage.contexts[worker])->async = &workers[worker];
        }
    }

    void *base() { return storage.scheduler_state->base(); }
    SchedulerLocalState *local(uint32_t worker = 0) { return storage.local_context(&storage.contexts[worker]); }
    SchedulerTaskControl *task_control(int64_t ref) { return scheduler_task_control_at(base(), local(), ref); }
    SchedulerAsyncTaskJoin &join(uint32_t index = 0) {
        return scheduler_state_at<SchedulerAsyncTaskJoin>(base(), plan.layout.joins_offset)[index];
    }
    SchedulerAsyncLease &group_lease(uint32_t index = 0) {
        return scheduler_state_at<SchedulerAsyncLease>(base(), plan.layout.group_leases_offset)[index];
    }
    SchedulerAsyncLease &finish_lease(uint32_t index = 0) {
        return scheduler_state_at<SchedulerAsyncLease>(base(), plan.layout.finish_leases_offset)[index];
    }

    AsyncWriter start(int64_t task_ref = 0, uint32_t block = 0, uint32_t lane = 0, uint32_t worker = 0) {
        GlobalContext context{};
        EXPECT_TRUE(scheduler_async_begin_execution(&workers[worker], task_ref, block, lane, context));
        EXPECT_EQ(context.async_present, 1u);
        return {
            reinterpret_cast<uint8_t *>(context.async_image_address), context.async_image_bytes,
            context.async_run_epoch, context.async_group_offset, context.async_group_index
        };
    }

    void register_counter(const AsyncWriter &writer, uint32_t expected = 1, uint32_t signal_index = 0) {
        const CompletionToken token{
            reinterpret_cast<uint64_t>(&signal[signal_index]), expected, COMPLETION_ENGINE_SDMA,
            COMPLETION_TYPE_COUNTER, 0
        };
        EXPECT_TRUE(register_completion_condition(writer, token));
    }

    bool check(uint32_t worker = 0, uint32_t budget = 8) {
        bool progress = false;
        EXPECT_TRUE(scheduler_async_check_step(
            workers[worker], worker == 0 ? 1 : 2, budget, SchedulerAsyncCompletionObserver{}, progress
        ));
        return progress;
    }
    bool finish(uint32_t worker = 0, uint32_t budget = 1) {
        bool progress = false;
        EXPECT_TRUE(scheduler_async_finish_step(
            graph.graph(), base(), local(worker), storage.run_control, workers[worker], budget, progress
        ));
        return progress;
    }
};

TEST(HbgAsyncControl, ReturnedExecutionReleasesSlotButCannotReleaseConsumerBeforeSignal) {
    AsyncFixture f;
    f.graph.executable(1, 0, {0});
    ASSERT_EQ(
        scheduler_route_task(f.graph.graph(), f.base(), f.local(), f.storage.run_control, 1, nullptr),
        SchedulerRouteResult::WAITING
    );
    const auto writer = f.start();
    f.register_counter(writer);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0));
    auto &slot = f.local()->slots[0];
    slot.task_ref = 0;
    slot.block_idx = 0;
    slot.subtask_slot = 0;
    slot.state = SchedulerDispatchSlotState::READY;
    slot.generation = 1;
    ASSERT_TRUE(scheduler_service_cluster_completion_slot(
        f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, 0, 1, nullptr, nullptr, nullptr, 0, nullptr,
        nullptr, nullptr
    ));
    EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(scheduler_async_join_blocks(f.join().word), 0u);
    EXPECT_EQ(scheduler_async_join_groups(f.join().word), 1u);
    EXPECT_EQ(f.task_control(0)->state, static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    EXPECT_FALSE(f.check(1));
    EXPECT_FALSE(f.finish(2));
    EXPECT_EQ(f.local()->pending_completed, 0u);
    EXPECT_EQ(scheduler_ready_inbox_at(f.base(), f.local(), 0, 0)->endpoints, SCHEDULER_READY_ENDPOINTS_EMPTY);
    __atomic_store_n(&f.signal[0], 1u, __ATOMIC_RELEASE);
    EXPECT_TRUE(f.check(2));
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::TERMINAL);
    EXPECT_EQ(f.join().word, 0u);
    for (uint32_t step = 0;
         step < 100 && scheduler_async_lease_phase(f.finish_lease().word) != SchedulerAsyncPhase::TERMINAL; ++step)
        f.finish(step % 3);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::TERMINAL);
    EXPECT_EQ(f.task_control(0)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    int64_t ready = -1;
    EXPECT_TRUE(scheduler_ready_pop_from_inbox(
        f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, 0, &ready, nullptr
    ));
    EXPECT_EQ(ready, 1);
    for (uint32_t worker = 0; worker < 3; ++worker)
        scheduler_flush_completions(f.storage.run_control, f.local(worker));
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 1u);
    EXPECT_FALSE(f.finish());
    scheduler_flush_completions(f.storage.run_control, f.local());
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 1u);
}

TEST(HbgAsyncControl, ConditionsMaySettleBeforeTheLastPhysicalBlockReleases) {
    AsyncFixture f(1);
    auto writer = f.start();
    f.register_counter(writer, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0));
    EXPECT_TRUE(f.check(1));
    EXPECT_EQ(scheduler_async_join_groups(f.join().word), 0u);
    EXPECT_EQ(scheduler_async_join_blocks(f.join().word), 1u);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::DORMANT);
    EXPECT_FALSE(f.finish());
    EXPECT_TRUE(scheduler_async_release_block(&f.workers[2], 0));
    EXPECT_TRUE(f.finish(1));
    EXPECT_EQ(f.local(1)->pending_completed, 0u);
    EXPECT_TRUE(f.finish(2));
    EXPECT_EQ(f.local(2)->pending_completed, 1u);
}

TEST(HbgAsyncControl, EmptyAndSkippedExecutionsRemainStaticGroupObligationsForEveryMixMask) {
    for (uint32_t mask : {3u, 5u, 6u, 7u}) {
        const LaneMask lanes((mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0);
        AsyncFixture f(1, {async_task(0, 2, lanes)});
        for (uint32_t block = 0; block < 2; ++block) {
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (!lanes.has(static_cast<SubtaskSlot>(lane))) continue;
                (void)f.start(0, block, lane, lane);
                ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[lane], 0, block, lane));
            }
            ASSERT_TRUE(scheduler_async_release_block(&f.workers[0], 0));
            if (block == 0) EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::DORMANT);
        }
        EXPECT_EQ(f.join().word, 0u);
        EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::RUNNABLE);
        EXPECT_FALSE(f.check());
    }
}

TEST(HbgAsyncControl, OpenReservationFailsSealWithoutSettlingItsGroup) {
    AsyncFixture f(1);
    const auto writer = f.start();
    ASSERT_NE(reserve_conditions(writer, 1).serial, 0u);
    EXPECT_FALSE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0));
    EXPECT_EQ(scheduler_async_join_groups(f.join().word), 1u);
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::FAILED);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::DORMANT);
}

TEST(HbgAsyncControl, DirectoryTailWrapDoesNotStarveEarlierPendingGroups) {
    std::vector<AsyncTaskPlanInput> inputs;
    for (uint32_t task = 0; task < 65; ++task)
        inputs.push_back(async_task(task));
    AsyncFixture f(65, inputs);
    auto first = f.start(0);
    f.register_counter(first, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0));
    f.workers[1].check_cursor = 64;
    EXPECT_FALSE(f.check(1));
    EXPECT_EQ(f.workers[1].check_cursor, 0u);
    EXPECT_TRUE(f.check(1));
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease(0).word), SchedulerAsyncPhase::TERMINAL);
}

TEST(HbgAsyncControl, LowPendingGroupCannotStarveReadyGroupInTheSameDirectoryWord) {
    AsyncFixture f(2, {async_task(0), async_task(1)});
    for (uint32_t task = 0; task < 2; ++task) {
        const auto writer = f.start(task);
        f.register_counter(writer, task == 0 ? 1 : 0);
        ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], task, 0, 0));
    }
    EXPECT_FALSE(f.check());
    EXPECT_TRUE(f.check());
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease(0).word), SchedulerAsyncPhase::RUNNABLE);
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease(1).word), SchedulerAsyncPhase::TERMINAL);
}

TEST(HbgAsyncControl, CheckBudgetAndSatisfiedMaskSurviveObserverMigration) {
    AsyncFixture f(1, {async_task(0, 1, LaneMask::aic(), 64)});
    const auto writer = f.start();
    for (uint32_t condition = 0; condition < 64; ++condition)
        f.register_counter(writer, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0));
    for (uint32_t round = 0; round < 8; ++round) {
        EXPECT_TRUE(f.check(round % 3, 8));
        const auto *state =
            scheduler_state_at<SchedulerAsyncGroupProgress>(f.base(), f.plan.layout.group_progress_offset);
        EXPECT_EQ(__builtin_popcountll(state->pending_mask), static_cast<int>((7 - round) * 8));
        if (round != 7) EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::RUNNABLE);
    }
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::TERMINAL);
    EXPECT_EQ(scheduler_async_join_groups(f.join().word), 0u);
}

TEST(HbgAsyncControl, FinishMigratesWithAnUnpublishedBatchAndCountsOnlyAfterDrain) {
    AsyncFixture f(12);
    for (uint32_t task = 1; task < 12; ++task) {
        f.graph.executable(task, 0, {0});
        ASSERT_EQ(
            scheduler_route_task(f.graph.graph(), f.base(), f.local(), f.storage.run_control, task, nullptr),
            SchedulerRouteResult::WAITING
        );
    }
    (void)f.start();
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0));
    ASSERT_TRUE(scheduler_async_release_block(&f.workers[0], 0));
    bool saw_pending_batch = false;
    for (uint32_t step = 0;
         step < 500 && scheduler_async_lease_phase(f.finish_lease().word) != SchedulerAsyncPhase::TERMINAL; ++step) {
        f.finish(step % 3, 1);
        auto *saved = scheduler_state_at<SchedulerAsyncFinishProgress>(f.base(), f.plan.layout.finish_progress_offset);
        saw_pending_batch = saw_pending_batch || saved->resolver.publication.count != 0;
        for (uint32_t worker = 0; worker < 3; ++worker)
            scheduler_flush_completions(f.storage.run_control, f.local(worker));
        if (saved->resolver.phase != SchedulerResolvePhase::DONE)
            EXPECT_EQ(f.storage.run_control->resolved_task_count, 0u);
    }
    EXPECT_TRUE(saw_pending_batch);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::TERMINAL);
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 1u);
    std::array<bool, 12> seen{};
    for (uint32_t count = 0; count < 11; ++count) {
        int64_t task = -1;
        ASSERT_TRUE(scheduler_ready_pop_from_inbox(
            f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, 0, &task, nullptr
        ));
        ASSERT_GE(task, 1);
        ASSERT_LT(task, 12);
        EXPECT_FALSE(seen[task]);
        seen[task] = true;
    }
}

TEST(HbgAsyncControl, InvalidCounterAndMalformedRecordFailWithoutNormalCompletion) {
    for (uint32_t mode = 0; mode < 2; ++mode) {
        AsyncFixture f(1);
        const auto writer = f.start();
        f.register_counter(writer);
        if (mode == 0) f.signal[0] = UINT32_MAX;
        else
            scheduler_state_at<DeferredCompletionEntry>(f.base(), writer.group_offset + sizeof(AsyncGroupData))
                ->backend_cookie = 1;
        ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0));
        bool progress = false;
        EXPECT_FALSE(scheduler_async_check_step(f.workers[1], 2, 8, SchedulerAsyncCompletionObserver{}, progress));
        EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::FAILED);
        EXPECT_EQ(scheduler_async_join_groups(f.join().word), 1u);
        EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::DORMANT);
    }
}

TEST(HbgAsyncControl, CounterAndFullWidthSdmaRecordsMustBothCompleteAndAreNeverCleared) {
    AsyncFixture f(1, {async_task(0, 1, LaneMask::aic(), 2)});
    alignas(128) uint64_t completed_post = UINT64_C(0xffffffff);
    const auto writer = f.start();
    f.register_counter(writer);
    const CompletionToken dma{
        reinterpret_cast<uint64_t>(&completed_post), 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD,
        UINT64_C(0x100000003)
    };
    ASSERT_TRUE(register_completion_condition(writer, dma));
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0));
    ASSERT_TRUE(scheduler_async_release_block(&f.workers[0], 0));
    EXPECT_FALSE(f.check(1));
    f.signal[0] = 1;
    EXPECT_TRUE(f.check(2));
    EXPECT_EQ(scheduler_async_join_groups(f.join().word), 1u);
    EXPECT_EQ(completed_post, UINT64_C(0xffffffff));
    completed_post = UINT64_C(0x100000003);
    EXPECT_TRUE(f.check(0));
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::TERMINAL);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::RUNNABLE);
    EXPECT_EQ(completed_post, UINT64_C(0x100000003));
}

TEST(HbgAsyncControl, OneSliceClaimsOnlyOneRecordAndAlternatesCheckAndFinish) {
    AsyncFixture f(1);
    const auto writer = f.start();
    f.register_counter(writer, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0));
    ASSERT_TRUE(scheduler_async_release_block(&f.workers[0], 0));
    f.workers[0].business_returns = 4;
    bool progress = false;
    ASSERT_TRUE(
        scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, progress)
    );
    EXPECT_TRUE(progress);
    EXPECT_EQ(f.workers[0].business_returns, 0u);
    EXPECT_TRUE(f.workers[0].finish_first);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::RUNNABLE);
    EXPECT_EQ(f.task_control(0)->state, static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    EXPECT_EQ(f.local()->pending_completed, 0u);
    ASSERT_TRUE(
        scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, progress)
    );
    EXPECT_TRUE(progress);
    EXPECT_FALSE(f.workers[0].finish_first);
    EXPECT_EQ(f.task_control(0)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(f.local()->pending_completed, 1u);
}

TEST(HbgAsyncControl, ObserverFailureLatchesRunErrorAndOtherCoresCannotFinishNormally) {
    AsyncFixture f(1);
    const auto writer = f.start();
    f.register_counter(writer);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0));
    ASSERT_TRUE(scheduler_async_release_block(&f.workers[0], 0));
    f.signal[0] = UINT32_MAX;
    bool progress = false;
    EXPECT_FALSE(
        scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(1), f.storage.run_control, 0, progress)
    );
    EXPECT_EQ(f.storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::ASYNC_CHECK_FAILED));
    EXPECT_NE(scheduler_read_error(f.storage.run_control), 0u);
    EXPECT_FALSE(
        scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(2), f.storage.run_control, 0, progress)
    );
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 0u);
    EXPECT_EQ(scheduler_async_join_groups(f.join().word), 1u);
}

TEST(HbgAsyncControl, ConcurrentObserversAndBlockReleaseGrantOneFinish) {
    AsyncFixture f(1, {async_task(0, 16, LaneMask::all())});
    for (uint32_t block = 0; block < 16; ++block) {
        for (uint32_t lane = 0; lane < 3; ++lane) {
            const auto writer = f.start(0, block, lane);
            f.register_counter(writer);
            ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, block, lane));
        }
    }
    SchedulerAsyncLocalState producer;
    ASSERT_TRUE(scheduler_initialize_async(f.base(), f.plan.total_bytes, f.plan.layout_offset, 1, 1, producer));
    std::atomic<bool> begin{false};
    std::atomic<bool> failed{false};
    std::vector<std::thread> observers;
    for (uint32_t worker = 0; worker < 3; ++worker) {
        auto *local = f.local(worker);
        observers.emplace_back([&, local] {
            while (!begin.load(std::memory_order_acquire)) {}
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!failed.load(std::memory_order_acquire) &&
                   scheduler_async_lease_phase(scheduler_async_lease_load(f.finish_lease())) !=
                       SchedulerAsyncPhase::TERMINAL) {
                bool progress = false;
                if (!scheduler_service_async_slice(
                        f.graph.graph(), f.base(), local, f.storage.run_control, 0, progress
                    ) ||
                    std::chrono::steady_clock::now() >= deadline) {
                    failed.store(true, std::memory_order_release);
                    break;
                }
            }
        });
    }
    begin.store(true, std::memory_order_release);
    for (uint32_t block = 0; block < 16; ++block)
        if (!scheduler_async_release_block(&producer, 0)) failed.store(true, std::memory_order_release);
    __atomic_store_n(&f.signal[0], 1u, __ATOMIC_RELEASE);
    for (auto &observer : observers)
        observer.join();
    EXPECT_FALSE(failed.load());
    EXPECT_EQ(f.join().word, 0u);
    EXPECT_EQ(scheduler_read_error(f.storage.run_control), 0u);
    for (uint32_t worker = 0; worker < 3; ++worker)
        scheduler_flush_completions(f.storage.run_control, f.local(worker));
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 1u);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::TERMINAL);
}

}  // namespace
