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

#include "scheduler_memory.h"

enum class SchedulerAsyncPhase : uint8_t { DORMANT = 0, RUNNABLE = 1, OWNED = 2, TERMINAL = 3, FAILED = 4 };

inline constexpr uint32_t SCHEDULER_ASYNC_LEASE_PHASE_BITS = 3;
inline constexpr uint64_t SCHEDULER_ASYNC_LEASE_GENERATION_MAX = UINT64_MAX >> SCHEDULER_ASYNC_LEASE_PHASE_BITS;
inline constexpr uint32_t SCHEDULER_ASYNC_JOIN_COUNT_MAX = INT16_MAX;

constexpr inline __aicore__ uint64_t scheduler_async_lease_encode(uint64_t generation, SchedulerAsyncPhase phase) {
    if (generation == 0 || generation > SCHEDULER_ASYNC_LEASE_GENERATION_MAX ||
        static_cast<uint8_t>(phase) > static_cast<uint8_t>(SchedulerAsyncPhase::FAILED) ||
        (phase == SchedulerAsyncPhase::DORMANT && generation != 1))
        return 0;
    return (generation << SCHEDULER_ASYNC_LEASE_PHASE_BITS) | static_cast<uint64_t>(phase);
}

constexpr inline __aicore__ SchedulerAsyncPhase scheduler_async_lease_phase(uint64_t word) {
    return static_cast<SchedulerAsyncPhase>(word & UINT64_C(7));
}

constexpr inline __aicore__ uint64_t scheduler_async_lease_generation(uint64_t word) {
    return word >> SCHEDULER_ASYNC_LEASE_PHASE_BITS;
}

constexpr inline __aicore__ bool scheduler_async_lease_word_valid(uint64_t word) {
    return word != 0 &&
           scheduler_async_lease_encode(scheduler_async_lease_generation(word), scheduler_async_lease_phase(word)) ==
               word;
}

struct alignas(128) SchedulerAsyncLease {
    volatile uint64_t word{scheduler_async_lease_encode(1, SchedulerAsyncPhase::DORMANT)};
};

struct alignas(128) SchedulerAsyncTaskJoin {
    volatile uint64_t word{0};
};

static_assert(sizeof(SchedulerAsyncLease) == 128 && alignof(SchedulerAsyncLease) == 128);
static_assert(sizeof(SchedulerAsyncTaskJoin) == 128 && alignof(SchedulerAsyncTaskJoin) == 128);
static_assert(offsetof(SchedulerAsyncLease, word) == 0 && offsetof(SchedulerAsyncTaskJoin, word) == 0);
static_assert(std::is_standard_layout_v<SchedulerAsyncLease> && std::is_trivially_copyable_v<SchedulerAsyncLease>);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncTaskJoin> && std::is_trivially_copyable_v<SchedulerAsyncTaskJoin>
);

enum class SchedulerAsyncLeaseResult : uint8_t {
    SUCCESS,
    NOT_RUNNABLE,
    CONTENDED,
    INVALID_WORD,
    INVALID_TRANSITION,
    GENERATION_OVERFLOW,
    OWNERSHIP_LOST,
};

struct SchedulerAsyncLeaseOutcome {
    SchedulerAsyncLeaseResult result;
    // SUCCESS returns the new word; failure retains the input snapshot or CAS result.
    uint64_t word;
};

enum class SchedulerAsyncJoinResult : uint8_t { PENDING, FINISH_OWNER, INVALID_WORD, UNDERFLOW };

struct SchedulerAsyncJoinOutcome {
    SchedulerAsyncJoinResult result;
    uint64_t previous;
};

constexpr inline __aicore__ uint32_t scheduler_async_join_groups(uint64_t word) {
    return static_cast<uint32_t>(word & UINT64_C(0xffff));
}

constexpr inline __aicore__ uint32_t scheduler_async_join_blocks(uint64_t word) {
    return static_cast<uint32_t>((word >> 16) & UINT64_C(0xffff));
}

constexpr inline __aicore__ bool scheduler_async_join_word_valid(uint64_t word) {
    return (word >> 32) == 0 && scheduler_async_join_groups(word) <= SCHEDULER_ASYNC_JOIN_COUNT_MAX &&
           scheduler_async_join_blocks(word) <= SCHEDULER_ASYNC_JOIN_COUNT_MAX;
}

// Only async tasks own joins: both initial counts are nonzero. Initialization
// precedes device publication and is not a reset operation on a live run.
inline __aicore__ bool scheduler_async_join_initial_word(uint32_t blocks, uint32_t groups, uint64_t &word) {
    if (blocks == 0 || groups == 0 || blocks > SCHEDULER_ASYNC_JOIN_COUNT_MAX ||
        groups > SCHEDULER_ASYNC_JOIN_COUNT_MAX)
        return false;
    word = (static_cast<uint64_t>(blocks) << 16) | groups;
    return true;
}

namespace scheduler_async_state_detail {

struct GmAtomics {
    __gm__ volatile uint64_t &word;

    inline __aicore__ uint64_t load() { return scheduler_gm_fetch_add(word, UINT64_C(0)); }
    inline __aicore__ uint64_t compare_exchange(uint64_t expected, uint64_t desired) {
        return scheduler_gm_compare_exchange(word, expected, desired);
    }
    inline __aicore__ uint64_t fetch_add(uint64_t delta) { return scheduler_gm_fetch_add(word, delta); }
};

template <typename Atomics>
inline __aicore__ SchedulerAsyncLeaseOutcome transition(Atomics &atomics, uint64_t expected, uint64_t desired) {
    const uint64_t actual = atomics.compare_exchange(expected, desired);
    if (actual == expected) return {SchedulerAsyncLeaseResult::SUCCESS, desired};
    return {
        scheduler_async_lease_word_valid(actual) ? SchedulerAsyncLeaseResult::OWNERSHIP_LOST :
                                                   SchedulerAsyncLeaseResult::INVALID_WORD,
        actual
    };
}

template <typename Atomics>
inline __aicore__ SchedulerAsyncLeaseOutcome claim(Atomics &atomics, uint64_t observed) {
    if (!scheduler_async_lease_word_valid(observed)) return {SchedulerAsyncLeaseResult::INVALID_WORD, observed};
    if (scheduler_async_lease_phase(observed) != SchedulerAsyncPhase::RUNNABLE)
        return {SchedulerAsyncLeaseResult::NOT_RUNNABLE, observed};
    const uint64_t desired =
        scheduler_async_lease_encode(scheduler_async_lease_generation(observed), SchedulerAsyncPhase::OWNED);
    const uint64_t actual = atomics.compare_exchange(observed, desired);
    if (actual == observed) return {SchedulerAsyncLeaseResult::SUCCESS, desired};
    return {
        scheduler_async_lease_word_valid(actual) ? SchedulerAsyncLeaseResult::CONTENDED :
                                                   SchedulerAsyncLeaseResult::INVALID_WORD,
        actual
    };
}

template <typename Atomics>
inline __aicore__ SchedulerAsyncLeaseOutcome
owner_transition(Atomics &atomics, uint64_t owned, SchedulerAsyncPhase desired_phase) {
    if (!scheduler_async_lease_word_valid(owned)) return {SchedulerAsyncLeaseResult::INVALID_WORD, owned};
    if (scheduler_async_lease_phase(owned) != SchedulerAsyncPhase::OWNED)
        return {SchedulerAsyncLeaseResult::INVALID_TRANSITION, owned};
    uint64_t generation = scheduler_async_lease_generation(owned);
    if (desired_phase == SchedulerAsyncPhase::RUNNABLE) {
        if (generation == SCHEDULER_ASYNC_LEASE_GENERATION_MAX)
            return {SchedulerAsyncLeaseResult::GENERATION_OVERFLOW, owned};
        ++generation;
    } else if (desired_phase != SchedulerAsyncPhase::TERMINAL && desired_phase != SchedulerAsyncPhase::FAILED) {
        return {SchedulerAsyncLeaseResult::INVALID_TRANSITION, owned};
    }
    return transition(atomics, owned, scheduler_async_lease_encode(generation, desired_phase));
}

template <bool Group, typename Atomics>
inline __aicore__ SchedulerAsyncJoinOutcome contribute(Atomics &atomics) {
    constexpr uint64_t contribution = Group ? UINT64_C(1) : (UINT64_C(1) << 16);
    const uint64_t previous = atomics.fetch_add(UINT64_C(0) - contribution);
    if (!scheduler_async_join_word_valid(previous)) return {SchedulerAsyncJoinResult::INVALID_WORD, previous};
    const uint32_t count = Group ? scheduler_async_join_groups(previous) : scheduler_async_join_blocks(previous);
    if (count == 0) return {SchedulerAsyncJoinResult::UNDERFLOW, previous};
    return {
        previous == contribution ? SchedulerAsyncJoinResult::FINISH_OWNER : SchedulerAsyncJoinResult::PENDING, previous
    };
}

}  // namespace scheduler_async_state_detail

// All live accesses are atomic. No lease is reset while any run participant may
// retain an old word. Run identity, execution ownership and data/progress
// barriers belong to the caller.
inline __aicore__ uint64_t scheduler_async_lease_load(__gm__ SchedulerAsyncLease &lease) {
    return scheduler_gm_fetch_add(lease.word, UINT64_C(0));
}

inline __aicore__ SchedulerAsyncLeaseOutcome scheduler_async_lease_publish(__gm__ SchedulerAsyncLease &lease) {
    scheduler_async_state_detail::GmAtomics atomics{lease.word};
    return scheduler_async_state_detail::transition(
        atomics, scheduler_async_lease_encode(1, SchedulerAsyncPhase::DORMANT),
        scheduler_async_lease_encode(1, SchedulerAsyncPhase::RUNNABLE)
    );
}

inline __aicore__ SchedulerAsyncLeaseOutcome
scheduler_async_lease_try_claim(__gm__ SchedulerAsyncLease &lease, uint64_t observed) {
    scheduler_async_state_detail::GmAtomics atomics{lease.word};
    return scheduler_async_state_detail::claim(atomics, observed);
}

inline __aicore__ SchedulerAsyncLeaseOutcome
scheduler_async_lease_release(__gm__ SchedulerAsyncLease &lease, uint64_t owned) {
    scheduler_async_state_detail::GmAtomics atomics{lease.word};
    return scheduler_async_state_detail::owner_transition(atomics, owned, SchedulerAsyncPhase::RUNNABLE);
}

inline __aicore__ SchedulerAsyncLeaseOutcome
scheduler_async_lease_finish(__gm__ SchedulerAsyncLease &lease, uint64_t owned) {
    scheduler_async_state_detail::GmAtomics atomics{lease.word};
    return scheduler_async_state_detail::owner_transition(atomics, owned, SchedulerAsyncPhase::TERMINAL);
}

// A sealed zero-registration or skipped execution has a group obligation even
// though it never publishes a CHECK directory bit.
inline __aicore__ SchedulerAsyncLeaseOutcome scheduler_async_lease_finish_dormant(__gm__ SchedulerAsyncLease &lease) {
    scheduler_async_state_detail::GmAtomics atomics{lease.word};
    return scheduler_async_state_detail::transition(
        atomics, scheduler_async_lease_encode(1, SchedulerAsyncPhase::DORMANT),
        scheduler_async_lease_encode(1, SchedulerAsyncPhase::TERMINAL)
    );
}

inline __aicore__ SchedulerAsyncLeaseOutcome
scheduler_async_lease_fail(__gm__ SchedulerAsyncLease &lease, uint64_t expected) {
    if (!scheduler_async_lease_word_valid(expected)) return {SchedulerAsyncLeaseResult::INVALID_WORD, expected};
    const auto phase = scheduler_async_lease_phase(expected);
    if (phase != SchedulerAsyncPhase::DORMANT && phase != SchedulerAsyncPhase::OWNED)
        return {SchedulerAsyncLeaseResult::INVALID_TRANSITION, expected};
    scheduler_async_state_detail::GmAtomics atomics{lease.word};
    return scheduler_async_state_detail::transition(
        atomics, expected,
        scheduler_async_lease_encode(scheduler_async_lease_generation(expected), SchedulerAsyncPhase::FAILED)
    );
}

// A unique execution/lease grant authorizes each contribution, including zero
// registration and skip. Error outcomes may follow an invalid RMW and require
// run failure; neither compensation nor reading a zero grants FINISH ownership.
inline __aicore__ SchedulerAsyncJoinOutcome scheduler_async_join_settle_group(__gm__ SchedulerAsyncTaskJoin &join) {
    scheduler_async_state_detail::GmAtomics atomics{join.word};
    return scheduler_async_state_detail::contribute<true>(atomics);
}

inline __aicore__ SchedulerAsyncJoinOutcome scheduler_async_join_release_block(__gm__ SchedulerAsyncTaskJoin &join) {
    scheduler_async_state_detail::GmAtomics atomics{join.word};
    return scheduler_async_state_detail::contribute<false>(atomics);
}
