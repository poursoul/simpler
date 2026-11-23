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

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

namespace simpler::percounter {

inline constexpr uint8_t PERCOUNTER_TYPE_AIC = 1;
inline constexpr uint8_t PERCOUNTER_TYPE_AIV = 2;
inline constexpr uint8_t PERCOUNTER_TYPES_BOTH = PERCOUNTER_TYPE_AIC | PERCOUNTER_TYPE_AIV;

inline uint8_t percounter_task_types(uint32_t mask) {
    return static_cast<uint8_t>((mask & 1U) | ((mask & 6U) ? PERCOUNTER_TYPE_AIV : 0U));
}

template <typename Task>
inline uint8_t percounter_full_width_type(const Task &task, uint32_t clusters) {
    if (!task.sync_start || task.has_predicate || task.graph_invocation >= 0 || task.inline_completed ||
        task.published_off_lane)
        return 0;
    const uint32_t blocks = static_cast<uint32_t>(task.block_num);
    if (task.active_mask == 2 && blocks == 2 * clusters) return PERCOUNTER_TYPE_AIV;
    if (blocks != clusters) return 0;
    if (task.active_mask == 1) return PERCOUNTER_TYPE_AIC;
    if (task.active_mask == 7) return PERCOUNTER_TYPES_BOTH;
    return 0;
}

struct FullWidthSpan {
    uint32_t begin;
    uint32_t end;
    uint32_t graph_segment;
    uint8_t types;  // zero for an ordinary span or a Graph boundary
    bool joined;    // a full-width MIX in the span joins the AIC and AIV chains
};

struct FullWidthOrder {
    std::vector<uint32_t> tasks;
    std::vector<FullWidthSpan> spans;
    std::vector<uint32_t> merge_gaps;
    std::vector<uint32_t> edge_consumers;
};

/**
 * Pick among runnable tasks without crossing the fixed Graph boundaries.
 * The original resource ranks are the priority within each ready queue.
 */
template <typename Task>
inline FullWidthOrder make_full_width_order(
    const Task *tasks, const std::vector<uint32_t> &original, const std::vector<uint8_t> &width,
    const std::vector<uint8_t> &synchronous, const std::vector<uint32_t> &graph_segment
) {
    const size_t count = original.size();
    FullWidthOrder next;
    next.tasks.reserve(count);
    std::vector<uint32_t> rank(count);
    for (uint32_t i = 0; i < count; ++i)
        rank[original[i]] = i;
    std::vector<uint32_t> assigned(count, UINT32_MAX);
    std::vector<uint32_t> pending(count);
    std::vector<std::vector<uint32_t>> successors(count);
    for (const uint32_t id : original) {
        if (tasks[id].fanin_count != 0) next.edge_consumers.push_back(id);
        for (int32_t i = 0; i < tasks[id].fanin_count; ++i) {
            const uint32_t input = static_cast<uint32_t>(tasks[id].fanin_ids[i]);
            if (graph_segment[input] != graph_segment[id]) continue;
            ++pending[id];
            successors[input].push_back(id);
        }
    }
    // The low two bits are the waiting entry's types. An interval's first
    // rank identifies its generation without clearing all rows per interval.
    std::vector<uint64_t> entry_wait(count);
    std::vector<uint64_t> rejected(count);
    for (uint32_t begin = 0; begin < count;) {
        const uint32_t group = graph_segment[original[begin]];
        uint32_t end = begin + 1;
        while (end < count && graph_segment[original[end]] == group)
            ++end;

        using Ready = std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>>;
        std::array<Ready, 4> ready;
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t id = original[i];
            if (pending[id] == 0) ready[width[id]].push(i);
        }
        const auto pop = [&](uint8_t types) {
            const uint32_t id = original[ready[types].top()];
            ready[types].pop();
            return id;
        };
        const auto emit = [&](uint32_t id) {
            assigned[id] = static_cast<uint32_t>(next.tasks.size());
            next.tasks.push_back(id);
            for (const uint32_t consumer : successors[id])
                if (--pending[consumer] == 0) ready[width[consumer]].push(rank[consumer]);
        };

        while (next.tasks.size() < end) {
            const uint32_t first = static_cast<uint32_t>(next.tasks.size());
            if (!ready[0].empty()) {
                do {
                    emit(pop(0));
                } while (!ready[0].empty());
                next.spans.push_back({first, static_cast<uint32_t>(next.tasks.size()), group, 0, false});
                continue;
            }

            uint8_t types = 0;
            bool joined = false;
            const uint64_t generation = (uint64_t{first} + 1) << 2;
            const auto add_entry = [&](uint32_t id) {
                for (int32_t i = 0; i < tasks[id].fanin_count; ++i) {
                    auto &value = entry_wait[tasks[id].fanin_ids[i]];
                    if ((value & ~uint64_t{3}) != generation) value = generation;
                    value |= width[id];
                }
                types |= width[id];
                joined |= width[id] == PERCOUNTER_TYPES_BOTH;
                emit(id);
            };
            if (!ready[PERCOUNTER_TYPE_AIC].empty() && !ready[PERCOUNTER_TYPE_AIV].empty()) {
                const uint32_t aic = pop(PERCOUNTER_TYPE_AIC);
                const uint32_t aiv = pop(PERCOUNTER_TYPE_AIV);
                add_entry(aic);
                add_entry(aiv);
            } else {
                uint8_t entry = 0;
                uint32_t earliest = UINT32_MAX;
                for (uint8_t kind = PERCOUNTER_TYPE_AIC; kind <= PERCOUNTER_TYPES_BOTH; ++kind) {
                    if (ready[kind].empty() || ready[kind].top() >= earliest) continue;
                    earliest = ready[kind].top();
                    entry = kind;
                }
                add_entry(pop(entry));
            }

            const auto can_append = [&](uint8_t active) {
                if ((types & active) != active || ready[active].empty()) return false;
                const uint32_t id = original[ready[active].top()];
                const uint64_t attempt = generation | static_cast<uint64_t>(joined);
                if (rejected[id] == attempt) return false;
                const uint8_t prefix_types = joined ? types : active;
                for (int32_t i = 0; i < tasks[id].fanin_count; ++i) {
                    const uint32_t producer = static_cast<uint32_t>(tasks[id].fanin_ids[i]);
                    if (tasks[producer].inline_completed ||
                        (assigned[producer] >= first && assigned[producer] != UINT32_MAX))
                        continue;
                    const uint64_t waited = entry_wait[producer];
                    if ((waited & ~uint64_t{3}) == generation && (waited & prefix_types) != 0) continue;
                    const uint8_t producer_types = percounter_task_types(tasks[producer].active_mask);
                    if (!synchronous[producer] || assigned[producer] >= first || producer_types == 0 ||
                        (prefix_types & producer_types) != producer_types) {
                        rejected[id] = attempt;
                        return false;
                    }
                }
                return true;
            };
            while (true) {
                // A ready MIX closes the interval when there is no other
                // reachable work to place before it.
                uint8_t active = 0;
                for (uint8_t kind = PERCOUNTER_TYPE_AIC; kind <= PERCOUNTER_TYPES_BOTH; ++kind) {
                    if (!can_append(kind)) continue;
                    active = kind;
                    break;
                }
                if (active == 0) break;
                emit(pop(active));
                joined |= active == PERCOUNTER_TYPES_BOTH;
            }
            const size_t span = next.spans.size();
            if (span >= 2 && next.spans[span - 1].types == 0 && next.spans[span - 2].types != 0 &&
                next.spans[span - 1].graph_segment == group && next.spans[span - 2].graph_segment == group)
                next.merge_gaps.push_back(static_cast<uint32_t>(span - 1));
            next.spans.push_back({first, static_cast<uint32_t>(next.tasks.size()), group, types, joined});
        }
        begin = end;
    }
    return next;
}

template <typename Task>
inline FullWidthOrder merge_full_width_spans(
    const Task *tasks, FullWidthOrder first, const std::vector<uint8_t> &width, const std::vector<uint8_t> &synchronous
) {
    if (first.merge_gaps.empty()) return first;
    struct SpanEdges {
        uint32_t last_full_pred{0};  // rank + 1, or 0
        uint32_t first_outside_succ{UINT32_MAX};
        uint32_t first_mix{UINT32_MAX};
        std::vector<std::array<uint32_t, 2>> incoming;
    };
    std::vector<SpanEdges> edges(first.spans.size());
    std::vector<bool> is_gap(first.spans.size());
    for (const uint32_t gap : first.merge_gaps)
        is_gap[gap] = true;
    std::vector<uint32_t> span_of(first.tasks.size()), rank(first.tasks.size());
    for (uint32_t s = 0; s < first.spans.size(); ++s) {
        const auto &span = first.spans[s];
        for (uint32_t p = span.begin; p < span.end; ++p) {
            const uint32_t id = first.tasks[p];
            rank[id] = p;
            span_of[id] = s;
            if (width[id] == PERCOUNTER_TYPES_BOTH) edges[s].first_mix = std::min(edges[s].first_mix, p);
        }
    }
    for (const uint32_t id : first.edge_consumers) {
        const uint32_t to = span_of[id];
        for (int32_t i = 0; i < tasks[id].fanin_count; ++i) {
            const uint32_t pred = static_cast<uint32_t>(tasks[id].fanin_ids[i]);
            const uint32_t from = span_of[pred];
            if (from == to) continue;
            if (is_gap[to] && first.spans[from].types != 0)
                edges[to].last_full_pred = std::max(edges[to].last_full_pred, rank[pred] + 1);
            if (is_gap[from]) edges[from].first_outside_succ = std::min(edges[from].first_outside_succ, rank[id]);
            if (to != 0 && is_gap[to - 1]) edges[to].incoming.push_back({pred, id});
        }
    }

    std::vector<uint64_t> entry_wait(first.tasks.size());
    std::vector<bool> moved_before(first.spans.size(), false);
    // Record moves as references. Expand each original span once at the end.
    std::vector<std::pair<uint32_t, bool>> ordered;
    std::vector<uint32_t> interval;
    for (uint32_t left = 0; left < first.spans.size();) {
        const auto &head = first.spans[left];
        if (head.types == 0) {
            ordered.emplace_back(left++, false);
            continue;
        }
        bool joined = head.joined;
        const uint64_t generation = (uint64_t{left} + 1) << 2;
        const auto record_wait = [&](uint32_t id) {
            for (int32_t i = 0; i < tasks[id].fanin_count; ++i) {
                auto &value = entry_wait[tasks[id].fanin_ids[i]];
                if ((value & ~uint64_t{3}) != generation) value = generation;
                value |= width[id];
            }
        };
        record_wait(first.tasks[head.begin]);
        if (head.types == PERCOUNTER_TYPES_BOTH && width[first.tasks[head.begin]] != PERCOUNTER_TYPES_BOTH)
            record_wait(first.tasks[head.begin + 1]);

        interval.assign(1, left);
        uint32_t gap = left + 1;
        uint32_t move_after = UINT32_MAX;
        while (gap < first.spans.size() && is_gap[gap]) {
            const uint32_t right = gap + 1;
            const auto &tail = first.spans[right];
            if (first.spans[gap].types != 0 || first.spans[gap].graph_segment != head.graph_segment ||
                tail.graph_segment != head.graph_segment || tail.types == 0 || (tail.types & head.types) != tail.types)
                break;

            const auto can_merge = [&](bool forward) {
                for (const auto &[pred, id] : edges[right].incoming) {
                    const uint32_t from = span_of[pred];
                    if (tasks[pred].inline_completed || (from >= left && from < gap && first.spans[from].types != 0))
                        continue;
                    const uint8_t prefix = joined || edges[right].first_mix < rank[id] ? head.types : width[id];
                    const uint64_t waited = entry_wait[pred];
                    if ((waited & ~uint64_t{3}) == generation && (waited & prefix) != 0) continue;
                    const uint8_t producer_types = percounter_task_types(tasks[pred].active_mask);
                    if (!synchronous[pred] || producer_types == 0 || (prefix & producer_types) != producer_types ||
                        !(from < left || moved_before[from] || (forward && from == gap)))
                        return false;
                }
                return true;
            };
            if (edges[gap].last_full_pred <= head.begin && can_merge(true)) {
                ordered.emplace_back(gap, false);
                moved_before[gap] = true;
            } else if (edges[gap].first_outside_succ >= tail.end && can_merge(false)) {
                move_after = gap;
            } else {
                break;
            }
            interval.push_back(right);
            joined |= tail.joined;
            gap = right + 1;
            if (move_after != UINT32_MAX) break;
        }
        for (uint32_t s : interval)
            ordered.emplace_back(s, s != left);
        if (move_after != UINT32_MAX) ordered.emplace_back(move_after, false);
        left = gap;
    }

    FullWidthOrder next;
    next.tasks.reserve(first.tasks.size());
    for (const auto &[s, extend] : ordered) {
        auto span = first.spans[s];
        const uint32_t begin = static_cast<uint32_t>(next.tasks.size());
        next.tasks.insert(next.tasks.end(), first.tasks.begin() + span.begin, first.tasks.begin() + span.end);
        span.begin = begin;
        span.end = static_cast<uint32_t>(next.tasks.size());
        if (extend) {
            next.spans.back().end = span.end;
            next.spans.back().joined |= span.joined;
        } else {
            next.spans.push_back(span);
        }
    }
    return next;
}

}  // namespace simpler::percounter
