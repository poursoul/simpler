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
 * Unit tests for scheduler wiring and completion paths:
 *
 * 1. wire_task()         — fanout wiring, early-finished detection,
 *                          fanin_count initialization, ready push
 * 2. on_task_complete() — COMPLETED transition, fanout traversal,
 *                               consumer fanin release
 *
 * These tests exercise the core scheduling hot-paths that had zero coverage.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "utils/device_arena.h"
#include "pto_orchestrator.h"
#include "pto_shared_memory.h"
#include "scheduler/pto_scheduler.h"

// =============================================================================
// Fixture: sets up a scheduler with shared memory and provides helpers
// =============================================================================

class WiringTest : public ::testing::Test {
protected:
    // Wiring lives in the orchestrator now (replay_graph), so wire_task /
    // drain_wiring_queue / the SPSC queue are exercised through `orch`. The
    // completion-path test (on_task_complete) still drives `sched`, so both are
    // set up here, mirroring test_orchestrator_fanin's fixture.
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    PTO2SharedMemoryHandle *sm_handle = nullptr;
    PTO2OrchestratorState orch{};
    PTO2SchedulerState sched{};
    PTO2OrchestratorLayout orch_layout{};
    PTO2SchedulerLayout sched_layout{};
    std::vector<char> gm_heap;

    // Each init_slot()'d slot gets a distinct zeroed payload from this pool,
    // mirroring orch::prepare_task's bind_buffers: every production slot has a
    // payload, and the scheduler's release/propagate paths dereference it.
    static constexpr int kSlotPayloadPoolSize = 16;
    PTO2TaskPayload slot_payload_pool_[kSlotPayloadPoolSize];
    int slot_payload_pool_idx_ = 0;

    void SetUp() override {
        sm_handle = PTO2SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(4096);

        orch_layout = PTO2OrchestratorState::reserve_layout(runtime_arena, static_cast<int32_t>(PTO2_TASK_WINDOW_SIZE));
        sched_layout = PTO2SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(orch.init_data_from_layout(
            orch_layout, runtime_arena, sm_handle->sm_base, gm_heap.data(), 4096, PTO2_TASK_WINDOW_SIZE
        ));
        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        orch.wire_arena_pointers(orch_layout, runtime_arena, &sched);
    }

    void TearDown() override {
        orch.destroy();
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // Initialize a slot for testing wiring/completion
    void init_slot(PTO2TaskSlotState &slot, PTO2TaskState state, int32_t fanin_count, uint8_t ring_id = 0) {
        memset(&slot, 0, sizeof(slot));
        slot.task_state.store(state);
        slot.fanin_count = fanin_count;
        slot.fanin_refcount.store(0);
        slot.fanout_head = nullptr;
        slot.ring_id = ring_id;
        slot.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIC);
        slot.completed_subtasks.store(0);
        slot.total_required_subtasks = 1;
        slot.logical_block_num = 1;
        PTO2TaskPayload &slot_pl = slot_payload_pool_[slot_payload_pool_idx_++ % kSlotPayloadPoolSize];
        memset(&slot_pl, 0, sizeof(slot_pl));
        slot.payload = &slot_pl;
    }
};

// =============================================================================
// wire_task: no fanin (independent task)
// =============================================================================

TEST_F(WiringTest, WireTaskNoFaninBecomesReady) {
    // A task with 0 actual fanins should immediately be pushed to ready queue
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(task_slot, PTO2_TASK_PENDING, 0);
    payload.fanin_actual_count = 0;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    orch.wire_task(orch.ring.dep_pool, &task_slot, 0);

    // fanin_count set to 0 + 1 = 1 (the wiring "+1" sentinel)
    EXPECT_EQ(task_slot.fanin_count, 1);
    // fanin_refcount should be 1 (the +1 from no-fanin path)
    EXPECT_EQ(task_slot.fanin_refcount.load(), 1);

    // Task should be appended to the orchestrator's initial-ready handoff
    // (wire_task no longer pushes the scheduler's ready_queues directly).
    ASSERT_EQ(orch.initial_ready_count, 1);
    EXPECT_EQ(orch.initial_ready[0], &task_slot);
}

// =============================================================================
// wire_task: with fanin, all producers already completed (early-finished)
// =============================================================================

TEST_F(WiringTest, WireTaskAllProducersEarlyFinished) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskSlotState producer_slots[2];
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    // Set up 2 producers that are already COMPLETED
    for (int i = 0; i < 2; i++) {
        init_slot(producer_slots[i], PTO2_TASK_COMPLETED, 1);
    }

    // Consumer task with 2 fanins
    init_slot(task_slot, PTO2_TASK_PENDING, 0);
    payload.fanin_actual_count = 2;
    payload.fanin_inline_slot_states[0] = &producer_slots[0];
    payload.fanin_inline_slot_states[1] = &producer_slots[1];

    task_slot.payload = &payload;
    task_slot.task = &desc;

    orch.wire_task(orch.ring.dep_pool, &task_slot, 2);

    // fanin_count = 2 + 1 = 3
    EXPECT_EQ(task_slot.fanin_count, 3);
    // early_finished = 2, init_rc = 2 + 1 = 3, so refcount should hit fanin_count
    EXPECT_GE(task_slot.fanin_refcount.load(), task_slot.fanin_count);

    // All producers early-finished -> task appended to initial-ready.
    ASSERT_EQ(orch.initial_ready_count, 1);
    EXPECT_EQ(orch.initial_ready[0], &task_slot);
}

// =============================================================================
// wire_task: with fanin, producers still pending (task NOT ready)
// =============================================================================

TEST_F(WiringTest, WireTaskProducersPendingTaskNotReady) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskSlotState producer_slots[2];
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    // Producers are PENDING (not yet completed)
    for (int i = 0; i < 2; i++) {
        init_slot(producer_slots[i], PTO2_TASK_PENDING, 1);
    }

    init_slot(task_slot, PTO2_TASK_PENDING, 0);
    payload.fanin_actual_count = 2;
    payload.fanin_inline_slot_states[0] = &producer_slots[0];
    payload.fanin_inline_slot_states[1] = &producer_slots[1];
    task_slot.payload = &payload;
    task_slot.task = &desc;

    orch.wire_task(orch.ring.dep_pool, &task_slot, 2);

    // fanin_count = 3 (2 + 1)
    EXPECT_EQ(task_slot.fanin_count, 3);
    // early_finished = 0, init_rc = 1 -> not ready
    EXPECT_EQ(task_slot.fanin_refcount.load(), 1);
    EXPECT_LT(task_slot.fanin_refcount.load(), task_slot.fanin_count);

    // Not ready -> nothing appended to initial-ready.
    EXPECT_EQ(orch.initial_ready_count, 0);

    // Producers should have fanout_head pointing to task_slot
    EXPECT_NE(producer_slots[0].fanout_head, nullptr);
    EXPECT_EQ(producer_slots[0].fanout_head->slot_state, &task_slot);
    EXPECT_NE(producer_slots[1].fanout_head, nullptr);
    EXPECT_EQ(producer_slots[1].fanout_head->slot_state, &task_slot);
}

// =============================================================================
// wire_task: mixed early-finished and pending producers
// =============================================================================

TEST_F(WiringTest, WireTaskMixedProducerStates) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskSlotState producers[3];
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(producers[0], PTO2_TASK_COMPLETED, 1);  // early finished
    init_slot(producers[1], PTO2_TASK_PENDING, 1);    // in flight (< COMPLETED)
    init_slot(producers[2], PTO2_TASK_COMPLETED, 1);  // early finished (>= COMPLETED)

    init_slot(task_slot, PTO2_TASK_PENDING, 0);
    payload.fanin_actual_count = 3;
    for (int i = 0; i < 3; i++) {
        payload.fanin_inline_slot_states[i] = &producers[i];
    }
    task_slot.payload = &payload;
    task_slot.task = &desc;

    orch.wire_task(orch.ring.dep_pool, &task_slot, 3);

    // fanin_count = 4 (3 + 1)
    EXPECT_EQ(task_slot.fanin_count, 4);
    // early_finished = 2 (both COMPLETED producers), init_rc = 3
    // Not yet 4 -> not ready (one producer still running)
    EXPECT_EQ(task_slot.fanin_refcount.load(), 3);

    // Only the running producer should have the consumer in its fanout chain
    EXPECT_EQ(producers[0].fanout_head, nullptr);  // early finished, no dep entry added
    EXPECT_NE(producers[1].fanout_head, nullptr);  // running, dep entry added
    EXPECT_EQ(producers[2].fanout_head, nullptr);  // early finished
}

// =============================================================================
// on_task_complete: notifies consumers via fanout chain
// =============================================================================

TEST_F(WiringTest, OnMixedTaskCompleteNotifiesConsumers) {
    alignas(64) PTO2TaskSlotState producer;
    alignas(64) PTO2TaskSlotState consumer1, consumer2;
    alignas(64) PTO2TaskPayload prod_payload;
    memset(&prod_payload, 0, sizeof(prod_payload));
    PTO2TaskDescriptor desc{};

    // Producer in flight (PENDING, not yet COMPLETED) with 2 consumers in fanout chain
    init_slot(producer, PTO2_TASK_PENDING, 1);
    producer.payload = &prod_payload;
    producer.task = &desc;

    // Consumer1: needs 1 more fanin to become ready
    init_slot(consumer1, PTO2_TASK_PENDING, 2);
    consumer1.fanin_refcount.store(1);  // 1 of 2 satisfied
    consumer1.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIC);

    // Consumer2: this release will make it ready
    init_slot(consumer2, PTO2_TASK_PENDING, 2);
    consumer2.fanin_refcount.store(1);  // 1 of 2 satisfied
    consumer2.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIC);

    // Build fanout chain: producer -> consumer2 -> consumer1
    PTO2DepListEntry dep_entries[2];
    dep_entries[0].slot_state = &consumer1;
    dep_entries[0].next = nullptr;
    dep_entries[1].slot_state = &consumer2;
    dep_entries[1].next = &dep_entries[0];
    producer.fanout_head = &dep_entries[1];

    sched.on_task_complete(producer);

    // Producer should be COMPLETED
    EXPECT_EQ(producer.task_state.load(), PTO2_TASK_COMPLETED);

    // Both consumers should have fanin_refcount incremented
    EXPECT_EQ(consumer1.fanin_refcount.load(), 2);
    EXPECT_EQ(consumer2.fanin_refcount.load(), 2);

    // Both consumers should be ready (fanin_refcount == fanin_count)
    PTO2ResourceShape shape = consumer1.active_mask.to_shape();
    auto *r1 = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    auto *r2 = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    EXPECT_TRUE((r1 == &consumer1 && r2 == &consumer2) || (r1 == &consumer2 && r2 == &consumer1));
}

// =============================================================================
// drain_wiring_queue: pushes tasks through SPSC queue
// =============================================================================

TEST_F(WiringTest, DrainWiringQueueProcessesTasks) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(task_slot, PTO2_TASK_PENDING, 0);
    payload.fanin_actual_count = 0;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    // Push into the orchestrator's wiring SPSC queue (submit side).
    ASSERT_TRUE(orch.wiring.queue.push(&task_slot));

    // Drain it (orchestrator run_wiring side).
    int wired = orch.drain_wiring_queue(true /* force_drain */);
    EXPECT_EQ(wired, 1);

    // Task should be appended to initial-ready.
    ASSERT_EQ(orch.initial_ready_count, 1);
    EXPECT_EQ(orch.initial_ready[0], &task_slot);
}

TEST_F(WiringTest, DrainWiringQueueBackoffDefers) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(task_slot, PTO2_TASK_PENDING, 0);
    payload.fanin_actual_count = 0;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    orch.wiring.queue.push(&task_slot);

    // Without force_drain, single item < BATCH_SIZE → backoff
    orch.wiring.backoff_counter = 0;
    int wired = orch.drain_wiring_queue(false);
    EXPECT_EQ(wired, 0) << "Backoff should defer when queue < BATCH_SIZE";
    EXPECT_EQ(orch.wiring.backoff_counter, 1);
}

TEST_F(WiringTest, DrainWiringQueueBackoffLimitForcesProcess) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(task_slot, PTO2_TASK_PENDING, 0);
    payload.fanin_actual_count = 0;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    orch.wiring.queue.push(&task_slot);

    // Set backoff at limit → should process
    orch.wiring.backoff_counter = PTO2OrchestratorState::WiringState::BACKOFF_LIMIT;
    int wired = orch.drain_wiring_queue(false);
    EXPECT_EQ(wired, 1) << "Backoff limit reached should force processing";
}
