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
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "scheduler/scheduler_async_directory.h"
#include "scheduler/scheduler_async_state.h"

namespace {

using Phase = SchedulerAsyncPhase;
using LeaseResult = SchedulerAsyncLeaseResult;
using JoinResult = SchedulerAsyncJoinResult;

struct CountingAtomics {
    uint64_t word{0};
    uint32_t compares{0};
    uint32_t additions{0};

    uint64_t compare_exchange(uint64_t expected, uint64_t desired) {
        ++compares;
        const uint64_t actual = word;
        if (actual == expected) word = desired;
        return actual;
    }

    uint64_t fetch_add(uint64_t delta) {
        ++additions;
        const uint64_t previous = word;
        word += delta;
        return previous;
    }
};

TEST(SchedulerAsyncStateTest, LeaseEncodingRejectsZeroGenerationAndReservedPhases) {
    const SchedulerAsyncLease initial{};
    EXPECT_EQ(initial.word, scheduler_async_lease_encode(1, Phase::DORMANT));
    EXPECT_TRUE(scheduler_async_lease_word_valid(initial.word));
    EXPECT_FALSE(scheduler_async_lease_word_valid(0));
    EXPECT_FALSE(scheduler_async_lease_word_valid(UINT64_C(16)));
    for (uint64_t phase = 0; phase < 8; ++phase) {
        EXPECT_FALSE(scheduler_async_lease_word_valid(phase));
        EXPECT_EQ(scheduler_async_lease_word_valid(UINT64_C(8) | phase), phase <= 4);
    }
    EXPECT_EQ(scheduler_async_lease_encode(0, Phase::RUNNABLE), 0u);
    EXPECT_EQ(scheduler_async_lease_encode(2, Phase::DORMANT), 0u);
    EXPECT_EQ(scheduler_async_lease_encode(SCHEDULER_ASYNC_LEASE_GENERATION_MAX + 1, Phase::RUNNABLE), 0u);
    const auto last = scheduler_async_lease_encode(SCHEDULER_ASYNC_LEASE_GENERATION_MAX, Phase::OWNED);
    EXPECT_TRUE(scheduler_async_lease_word_valid(last));
    EXPECT_EQ(scheduler_async_lease_generation(last), SCHEDULER_ASYNC_LEASE_GENERATION_MAX);
}

TEST(SchedulerAsyncStateTest, HandoffRejectsAStaleClaimAndAStaleOwner) {
    SchedulerAsyncLease lease{};
    const auto published = scheduler_async_lease_publish(lease);
    ASSERT_EQ(published.result, LeaseResult::SUCCESS);
    const uint64_t original_snapshot = published.word;
    ASSERT_EQ(scheduler_async_lease_publish(lease).result, LeaseResult::OWNERSHIP_LOST);

    const auto first = scheduler_async_lease_try_claim(lease, original_snapshot);
    ASSERT_EQ(first.result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_lease_phase(first.word), Phase::OWNED);
    EXPECT_EQ(scheduler_async_lease_try_claim(lease, original_snapshot).result, LeaseResult::CONTENDED);
    const auto handed_off = scheduler_async_lease_release(lease, first.word);
    ASSERT_EQ(handed_off.result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_lease_generation(handed_off.word), 2u);
    EXPECT_EQ(scheduler_async_lease_phase(handed_off.word), Phase::RUNNABLE);

    EXPECT_EQ(scheduler_async_lease_try_claim(lease, original_snapshot).result, LeaseResult::CONTENDED);
    EXPECT_EQ(scheduler_async_lease_finish(lease, first.word).result, LeaseResult::OWNERSHIP_LOST);
    EXPECT_EQ(scheduler_async_lease_fail(lease, first.word).result, LeaseResult::OWNERSHIP_LOST);
    EXPECT_EQ(scheduler_async_lease_load(lease), handed_off.word);

    const auto second = scheduler_async_lease_try_claim(lease, handed_off.word);
    ASSERT_EQ(second.result, LeaseResult::SUCCESS);
    const auto finished = scheduler_async_lease_finish(lease, second.word);
    ASSERT_EQ(finished.result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_lease_phase(finished.word), Phase::TERMINAL);
    EXPECT_EQ(scheduler_async_lease_try_claim(lease, finished.word).result, LeaseResult::NOT_RUNNABLE);
    EXPECT_EQ(scheduler_async_lease_try_claim(lease, handed_off.word).result, LeaseResult::CONTENDED);
    EXPECT_EQ(scheduler_async_lease_publish(lease).result, LeaseResult::OWNERSHIP_LOST);
    EXPECT_EQ(scheduler_async_lease_finish(lease, second.word).result, LeaseResult::OWNERSHIP_LOST);
}

TEST(SchedulerAsyncStateTest, OwnerGenerationNeverWrapsAndMayStillFinishOrFail) {
    SchedulerAsyncLease lease{};
    lease.word = scheduler_async_lease_encode(SCHEDULER_ASYNC_LEASE_GENERATION_MAX, Phase::RUNNABLE);
    const auto owner = scheduler_async_lease_try_claim(lease, lease.word);
    ASSERT_EQ(owner.result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_lease_release(lease, owner.word).result, LeaseResult::GENERATION_OVERFLOW);
    EXPECT_EQ(lease.word, owner.word);
    EXPECT_EQ(scheduler_async_lease_finish(lease, owner.word).result, LeaseResult::SUCCESS);

    SchedulerAsyncLease failed{};
    failed.word = owner.word;
    const auto outcome = scheduler_async_lease_fail(failed, owner.word);
    EXPECT_EQ(outcome.result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_lease_phase(outcome.word), Phase::FAILED);
    EXPECT_EQ(scheduler_async_lease_generation(outcome.word), SCHEDULER_ASYNC_LEASE_GENERATION_MAX);
}

TEST(SchedulerAsyncStateTest, ConcurrentClaimsOfOneVersionGrantOneOwner) {
    SchedulerAsyncLease lease{};
    const auto publication = scheduler_async_lease_publish(lease);
    ASSERT_EQ(publication.result, LeaseResult::SUCCESS);
    std::mutex mutex;
    std::condition_variable ready;
    bool start = false;
    std::atomic<uint32_t> owners{0};
    std::atomic<uint32_t> contended{0};
    std::atomic<uint32_t> invalid{0};
    std::atomic<uint64_t> owned_word{0};
    std::vector<std::thread> workers;
    for (uint32_t index = 0; index < 8; ++index) {
        workers.emplace_back([&] {
            {
                std::unique_lock<std::mutex> lock(mutex);
                ready.wait(lock, [&] {
                    return start;
                });
            }
            const auto result = scheduler_async_lease_try_claim(lease, publication.word);
            if (result.result == LeaseResult::SUCCESS) {
                ++owners;
                owned_word.store(result.word);
            } else if (result.result == LeaseResult::CONTENDED) ++contended;
            else ++invalid;
        });
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        start = true;
    }
    ready.notify_all();
    for (auto &worker : workers)
        worker.join();
    EXPECT_EQ(owners, 1u);
    EXPECT_EQ(contended, 7u);
    EXPECT_EQ(invalid, 0u);
    EXPECT_EQ(lease.word, owned_word.load());
    EXPECT_EQ(scheduler_async_lease_finish(lease, owned_word.load()).result, LeaseResult::SUCCESS);
}

TEST(SchedulerAsyncStateTest, InvalidClaimsAndOwnerTransitionsDoNotIssueCas) {
    const uint64_t runnable = scheduler_async_lease_encode(1, Phase::RUNNABLE);
    CountingAtomics atomics{runnable};
    EXPECT_EQ(scheduler_async_state_detail::claim(atomics, 0).result, LeaseResult::INVALID_WORD);
    EXPECT_EQ(scheduler_async_state_detail::claim(atomics, UINT64_C(15)).result, LeaseResult::INVALID_WORD);
    EXPECT_EQ(
        scheduler_async_state_detail::claim(atomics, scheduler_async_lease_encode(1, Phase::DORMANT)).result,
        LeaseResult::NOT_RUNNABLE
    );
    EXPECT_EQ(
        scheduler_async_state_detail::owner_transition(atomics, runnable, Phase::TERMINAL).result,
        LeaseResult::INVALID_TRANSITION
    );
    const auto owner = scheduler_async_lease_encode(1, Phase::OWNED);
    EXPECT_EQ(
        scheduler_async_state_detail::owner_transition(atomics, owner, Phase::DORMANT).result,
        LeaseResult::INVALID_TRANSITION
    );
    const auto last = scheduler_async_lease_encode(SCHEDULER_ASYNC_LEASE_GENERATION_MAX, Phase::OWNED);
    EXPECT_EQ(
        scheduler_async_state_detail::owner_transition(atomics, last, Phase::RUNNABLE).result,
        LeaseResult::GENERATION_OVERFLOW
    );
    EXPECT_EQ(atomics.compares, 0u);
    EXPECT_EQ(atomics.word, runnable);
}

TEST(SchedulerAsyncStateTest, ObservedMalformedLeaseCannotBeOverwrittenByAClaimOrRelease) {
    CountingAtomics atomics{UINT64_C(15)};
    const auto runnable = scheduler_async_lease_encode(1, Phase::RUNNABLE);
    const auto owned = scheduler_async_lease_encode(1, Phase::OWNED);
    EXPECT_EQ(scheduler_async_state_detail::claim(atomics, runnable).result, LeaseResult::INVALID_WORD);
    EXPECT_EQ(
        scheduler_async_state_detail::owner_transition(atomics, owned, Phase::TERMINAL).result,
        LeaseResult::INVALID_WORD
    );
    EXPECT_EQ(atomics.word, UINT64_C(15));
    EXPECT_EQ(atomics.compares, 2u);
}

TEST(SchedulerAsyncStateTest, FailureWriterCannotTakeOverAnotherOwner) {
    SchedulerAsyncLease lease{};
    const uint64_t dormant = scheduler_async_lease_load(lease);
    const auto publication = scheduler_async_lease_publish(lease);
    const auto owner = scheduler_async_lease_try_claim(lease, publication.word);
    ASSERT_EQ(owner.result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_lease_fail(lease, dormant).result, LeaseResult::OWNERSHIP_LOST);
    EXPECT_EQ(lease.word, owner.word);
    EXPECT_EQ(scheduler_async_lease_fail(lease, publication.word).result, LeaseResult::INVALID_TRANSITION);
    const auto failed = scheduler_async_lease_fail(lease, owner.word);
    ASSERT_EQ(failed.result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_lease_try_claim(lease, failed.word).result, LeaseResult::NOT_RUNNABLE);
    EXPECT_EQ(scheduler_async_lease_release(lease, owner.word).result, LeaseResult::OWNERSHIP_LOST);

    SchedulerAsyncLease registration_error{};
    const auto error = scheduler_async_lease_fail(registration_error, registration_error.word);
    EXPECT_EQ(error.result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_lease_phase(error.word), Phase::FAILED);
}

TEST(SchedulerAsyncStateTest, JoinInitialCountsAreCheckedBeforeChangingTheOutput) {
    uint64_t word = UINT64_C(0xabcdef);
    for (auto counts : std::vector<std::pair<uint32_t, uint32_t>>{{0, 1}, {1, 0}, {32768, 1}, {1, 32768}}) {
        EXPECT_FALSE(scheduler_async_join_initial_word(counts.first, counts.second, word));
        EXPECT_EQ(word, UINT64_C(0xabcdef));
    }
    ASSERT_TRUE(scheduler_async_join_initial_word(32767, 32767, word));
    EXPECT_EQ(scheduler_async_join_blocks(word), 32767u);
    EXPECT_EQ(scheduler_async_join_groups(word), 32767u);
    EXPECT_TRUE(scheduler_async_join_word_valid(word));
    EXPECT_FALSE(scheduler_async_join_word_valid(UINT64_C(1) << 32));
    EXPECT_FALSE(scheduler_async_join_word_valid(UINT64_C(32768)));
    EXPECT_FALSE(scheduler_async_join_word_valid(UINT64_C(32768) << 16));
}

TEST(SchedulerAsyncStateTest, EveryBlockAndGroupInterleavingHasExactlyOneLastDecrementOwner) {
    std::vector<int> contributions{0, 0, 0, 0, 1, 1};
    uint32_t sequences = 0;
    do {
        SchedulerAsyncTaskJoin join{};
        uint64_t initial = 0;
        ASSERT_TRUE(scheduler_async_join_initial_word(2, 4, initial));
        join.word = initial;
        uint32_t finish_owners = 0;
        for (size_t index = 0; index < contributions.size(); ++index) {
            const auto result = contributions[index] == 0 ? scheduler_async_join_settle_group(join) :
                                                            scheduler_async_join_release_block(join);
            EXPECT_EQ(
                result.result, index + 1 == contributions.size() ? JoinResult::FINISH_OWNER : JoinResult::PENDING
            );
            if (result.result == JoinResult::FINISH_OWNER) ++finish_owners;
        }
        EXPECT_EQ(finish_owners, 1u);
        EXPECT_EQ(join.word, 0u);
        ++sequences;
    } while (std::next_permutation(contributions.begin(), contributions.end()));
    EXPECT_EQ(sequences, 15u);
}

TEST(SchedulerAsyncStateTest, ZeroRegistrationAndSkippedGroupsRemainRealJoinObligations) {
    SchedulerAsyncTaskJoin join{};
    uint64_t initial = 0;
    ASSERT_TRUE(scheduler_async_join_initial_word(1, 3, initial));
    join.word = initial;
    SchedulerAsyncLease deferred{}, empty{}, skipped{};
    volatile uint64_t directory = 0;
    const auto published = scheduler_async_lease_publish(deferred);
    ASSERT_EQ(published.result, LeaseResult::SUCCESS);
    ASSERT_EQ(scheduler_async_directory_publish(directory, 0), SchedulerAsyncDirectoryResult::SUCCESS);
    EXPECT_EQ(scheduler_async_join_release_block(join).result, JoinResult::PENDING);
    const auto owner = scheduler_async_lease_try_claim(deferred, published.word);
    ASSERT_EQ(owner.result, LeaseResult::SUCCESS);
    ASSERT_EQ(scheduler_async_lease_finish(deferred, owner.word).result, LeaseResult::SUCCESS);
    ASSERT_EQ(scheduler_async_directory_clear(directory, 0), SchedulerAsyncDirectoryResult::SUCCESS);
    EXPECT_EQ(scheduler_async_join_settle_group(join).result, JoinResult::PENDING);
    EXPECT_EQ(directory, 0u);
    EXPECT_EQ(scheduler_async_join_groups(join.word), 2u);

    ASSERT_EQ(scheduler_async_lease_finish_dormant(empty).result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_join_settle_group(join).result, JoinResult::PENDING);
    EXPECT_EQ(scheduler_async_lease_finish_dormant(empty).result, LeaseResult::OWNERSHIP_LOST);
    ASSERT_EQ(scheduler_async_lease_finish_dormant(skipped).result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_join_settle_group(join).result, JoinResult::FINISH_OWNER);
    EXPECT_EQ(join.word, 0u);
}

TEST(SchedulerAsyncStateTest, InvalidOldJoinValuesAndUnderflowNeverGrantFinishOwnership) {
    for (uint64_t value : {UINT64_C(0), UINT64_C(1) << 16}) {
        CountingAtomics atomics{value};
        const auto result = scheduler_async_state_detail::contribute<true>(atomics);
        EXPECT_EQ(result.result, JoinResult::UNDERFLOW);
        EXPECT_EQ(result.previous, value);
        EXPECT_EQ(atomics.additions, 1u);
    }
    for (uint64_t value : {UINT64_C(0), UINT64_C(1)}) {
        CountingAtomics atomics{value};
        EXPECT_EQ(scheduler_async_state_detail::contribute<false>(atomics).result, JoinResult::UNDERFLOW);
        EXPECT_EQ(atomics.additions, 1u);
    }
    for (uint64_t value : {UINT64_C(1) << 32, UINT64_C(32768), UINT64_C(32768) << 16, UINT64_MAX}) {
        CountingAtomics group{value};
        CountingAtomics block{value};
        EXPECT_EQ(scheduler_async_state_detail::contribute<true>(group).result, JoinResult::INVALID_WORD);
        EXPECT_EQ(scheduler_async_state_detail::contribute<false>(block).result, JoinResult::INVALID_WORD);
        EXPECT_EQ(group.additions, 1u);
        EXPECT_EQ(block.additions, 1u);
    }
}

TEST(SchedulerAsyncStateTest, FailedGroupDoesNotDischargeItsNormalCompletionObligation) {
    SchedulerAsyncTaskJoin join{};
    uint64_t initial = 0;
    ASSERT_TRUE(scheduler_async_join_initial_word(1, 1, initial));
    join.word = initial;
    SchedulerAsyncLease group{};
    const auto published = scheduler_async_lease_publish(group);
    const auto owner = scheduler_async_lease_try_claim(group, published.word);
    ASSERT_EQ(owner.result, LeaseResult::SUCCESS);
    ASSERT_EQ(scheduler_async_join_release_block(join).result, JoinResult::PENDING);
    ASSERT_EQ(scheduler_async_lease_fail(group, owner.word).result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_join_blocks(join.word), 0u);
    EXPECT_EQ(scheduler_async_join_groups(join.word), 1u);
    EXPECT_EQ(scheduler_async_lease_finish(group, owner.word).result, LeaseResult::OWNERSHIP_LOST);
}

TEST(SchedulerAsyncStateTest, ConcurrentUniqueContributionsProduceOnlyOneFinishOwner) {
    SchedulerAsyncTaskJoin join{};
    uint64_t initial = 0;
    ASSERT_TRUE(scheduler_async_join_initial_word(8, 16, initial));
    join.word = initial;
    std::mutex mutex;
    std::condition_variable ready;
    bool start = false;
    std::atomic<uint32_t> owners{0};
    std::atomic<uint32_t> pending{0};
    std::atomic<uint32_t> invalid{0};
    std::vector<std::thread> workers;
    for (uint32_t index = 0; index < 24; ++index) {
        workers.emplace_back([&, index] {
            {
                std::unique_lock<std::mutex> lock(mutex);
                ready.wait(lock, [&] {
                    return start;
                });
            }
            const auto result =
                index < 16 ? scheduler_async_join_settle_group(join) : scheduler_async_join_release_block(join);
            if (result.result == JoinResult::FINISH_OWNER) ++owners;
            else if (result.result == JoinResult::PENDING) ++pending;
            else ++invalid;
        });
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        start = true;
    }
    ready.notify_all();
    for (auto &worker : workers)
        worker.join();
    EXPECT_EQ(owners, 1u);
    EXPECT_EQ(pending, 23u);
    EXPECT_EQ(invalid, 0u);
    EXPECT_EQ(join.word, 0u);
}

}  // namespace
