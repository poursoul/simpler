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

#include "scheduler_ready_publication.h"

// The producer is already DONE. This cursor owns its detached chain until all
// successors are handed off and every ready publication has finished. Returning
// DONE does not itself contribute to the run's completion counter.
inline __aicore__ SchedulerResolveProgress scheduler_resolve_completion_step(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t producer_ref, SchedulerResolveCursor &cursor, uint32_t budget, SchedulerWakeStats *wake_stats = nullptr,
    SchedulerReadyStats *ready_stats = nullptr, SchedulerCompletionStats *completion_stats = nullptr,
    uint64_t profiling_level = 0
) {
    uint32_t used = 0;
    auto fail = [&](SchedulerErrorSite site = SchedulerErrorSite::UNKNOWN) __aicore__ {
        cursor.phase = SchedulerResolvePhase::FAILED;
        if (run != nullptr)
            scheduler_record_error(run, producer_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local, site);
        return SchedulerResolveProgress{SchedulerResolveResult::ERROR, used};
    };
    if (local == nullptr || run == nullptr || !scheduler_task_ref_valid(graph, producer_ref) ||
        (cursor.producer_ref != SCHEDULER_TASK_REF_INVALID && cursor.producer_ref != producer_ref))
        return fail();
    if (cursor.phase != SchedulerResolvePhase::START && cursor.producer_ref != producer_ref) return fail();
    if (cursor.phase == SchedulerResolvePhase::FAILED) return {SchedulerResolveResult::ERROR, 0};
    if (cursor.phase == SchedulerResolvePhase::DONE) return {SchedulerResolveResult::DONE, 0};
    if (scheduler_read_error(run) != 0) {
        cursor.phase = SchedulerResolvePhase::FAILED;
        return {SchedulerResolveResult::ERROR, 0};
    }
    const bool timing = scheduler_phase_timing_enabled(profiling_level);
    if (timing && local->profiling == nullptr) return fail();
    while (used < budget) {
        if (scheduler_read_error(run) != 0) {
            cursor.phase = SchedulerResolvePhase::FAILED;
            return {SchedulerResolveResult::ERROR, used};
        }
        switch (cursor.phase) {
        case SchedulerResolvePhase::START: {
            ++used;
            auto *control = scheduler_task_control_at(base, local, producer_ref);
            if (scheduler_gm_load(control->state) != static_cast<int64_t>(SchedulerTaskState::DONE))
                return fail(SchedulerErrorSite::COMPLETION_TASK_NOT_DONE);
            if (timing) {
                scheduler_observe_cache_line(&control->next_fanin_index);
                control->completion_resolve_start_cycles = scheduler_cycles();
                control->scheduler_worker_id = local->worker_id();
                control->completion_resolve_loop_iter = local->profiling->loop_iter;
                scheduler_publish_cache_line(&control->next_fanin_index);
            }
            cursor.producer_ref = producer_ref;
            cursor.next_waiter = scheduler_gm_exchange(control->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
            if (cursor.next_waiter < SCHEDULER_WAKE_LIST_OPEN)
                return fail(SchedulerErrorSite::COMPLETION_WAKE_ALREADY_CLOSED);
            if (wake_stats != nullptr) ++wake_stats->wake_close_count;
            cursor.phase = SchedulerResolvePhase::SELECT;
            break;
        }
        case SchedulerResolvePhase::SELECT: {
            ++used;
            if (cursor.next_waiter == SCHEDULER_WAKE_LIST_OPEN) {
                if (cursor.publication.count != 0) {
                    cursor.phase = SchedulerResolvePhase::PUBLISH;
                    break;
                }
                if (timing) {
                    auto *control = scheduler_task_control_at(base, local, producer_ref);
                    scheduler_observe_cache_line(&control->next_fanin_index);
                    control->completion_resolve_end_cycles = scheduler_cycles();
                    scheduler_publish_cache_line(&control->next_fanin_index);
                }
                cursor.phase = SchedulerResolvePhase::DONE;
                if (completion_stats != nullptr) ++completion_stats->resolve_count;
                return {SchedulerResolveResult::DONE, used};
            }
            if (!scheduler_task_ref_valid(graph, cursor.next_waiter) || cursor.visited_count >= graph.task_count)
                return fail(SchedulerErrorSite::COMPLETION_INVALID_WAITER);
            cursor.waiter_ref = cursor.next_waiter;
            cursor.saved_next =
                scheduler_observe_next_waiter(scheduler_task_control_at(base, local, cursor.waiter_ref));
            if (cursor.saved_next < SCHEDULER_WAKE_LIST_OPEN || cursor.saved_next == cursor.waiter_ref)
                return fail(SchedulerErrorSite::COMPLETION_INVALID_WAITER);
            ++cursor.visited_count;
            cursor.route = {};
            if (wake_stats != nullptr) ++wake_stats->wake_migrate_count;
            cursor.phase = SchedulerResolvePhase::ROUTE;
            break;
        }
        case SchedulerResolvePhase::ROUTE: {
            const auto progress = scheduler_route_task_step(
                graph, base, local, run, cursor.waiter_ref, cursor.route, budget - used, wake_stats
            );
            used += progress.work_units;
            if (progress.result == SchedulerRouteResult::ERROR) return fail();
            if (progress.result == SchedulerRouteResult::YIELD) return {SchedulerResolveResult::YIELD, used};
            if (progress.result == SchedulerRouteResult::READY_TO_ENQUEUE) {
                cursor.phase = SchedulerResolvePhase::APPEND;
            } else {
                // The old next was captured before routing transferred the waiter.
                cursor.next_waiter = cursor.saved_next;
                cursor.waiter_ref = SCHEDULER_TASK_REF_INVALID;
                cursor.phase = SchedulerResolvePhase::SELECT;
            }
            break;
        }
        case SchedulerResolvePhase::APPEND: {
            ++used;
            const auto appended =
                scheduler_ready_publication_append(graph, base, local, cursor.waiter_ref, cursor.publication);
            if (appended == SchedulerReadyAppendResult::ERROR)
                return fail(SchedulerErrorSite::COMPLETION_READY_APPEND_FAILED);
            if (appended == SchedulerReadyAppendResult::FULL) {
                cursor.phase = SchedulerResolvePhase::PUBLISH;
                break;
            }
            if (ready_stats != nullptr) ++ready_stats->enqueue_count;
            cursor.next_waiter = cursor.saved_next;
            cursor.waiter_ref = SCHEDULER_TASK_REF_INVALID;
            cursor.phase = cursor.publication.count == SCHEDULER_READY_PUBLICATION_CAPACITY ?
                               SchedulerResolvePhase::PUBLISH :
                               SchedulerResolvePhase::SELECT;
            break;
        }
        case SchedulerResolvePhase::PUBLISH: {
            ++used;
            const auto published =
                scheduler_ready_publication_step(graph, base, local, cursor.publication, ready_stats);
            if (published == SchedulerReadyPublishResult::ERROR)
                return fail(SchedulerErrorSite::COMPLETION_READY_PUBLISH_FAILED);
            if (published == SchedulerReadyPublishResult::DONE) {
                cursor.publication = {};
                cursor.phase = cursor.waiter_ref == SCHEDULER_TASK_REF_INVALID ? SchedulerResolvePhase::SELECT :
                                                                                 SchedulerResolvePhase::APPEND;
            }
            break;
        }
        default:
            return fail();
        }
    }
    return {SchedulerResolveResult::YIELD, used};
}
