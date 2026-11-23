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

#include <cstdint>
#include <utility>
#include <vector>

#include "pre_dispatch.h"

namespace simpler::percounter {

struct TailPredispatch {
    TailPredispatch(uint32_t id, uint32_t clusters) :
        anchor(id),
        wave(clusters) {}

    uint32_t anchor;
    PredispatchWave wave;
    std::vector<uint32_t> tasks;
};

template <typename Task, typename Plan>
inline std::vector<TailPredispatch> plan_tail_waves(
    const Task *tasks, const Plan &plan, const std::vector<uint8_t> &width, const std::vector<uint8_t> &synchronous,
    const std::vector<uint32_t> &rank, uint32_t clusters
) {
    if (plan.interval_spans.empty()) return {};

    std::vector<TailPredispatch> tails;
    for (const auto &span : plan.interval_spans) {
        const uint32_t anchor = plan.order[span.end - 1];
        if (width[anchor] != PERCOUNTER_TYPES_BOTH) continue;

        TailPredispatch tail(anchor, clusters);
        for (uint32_t p = span.end; p < plan.order.size(); ++p) {
            const uint32_t id = plan.order[p];
            const auto &task = tasks[id];
            if (task.graph_invocation >= 0 || plan.is_terminal(id) || plan.is_interval_entry(id)) break;
            if (task.inline_completed || task.published_off_lane || task.active_mask == 0 ||
                plan.is_predispatched(id) ||
                percounter_full_width_shape(task.active_mask, static_cast<uint32_t>(task.block_num), clusters) != 0)
                continue;

            bool ready = true;
            for (int32_t e = 0; e < task.fanin_count; ++e) {
                const uint32_t producer = static_cast<uint32_t>(task.fanin_ids[e]);
                if (tasks[producer].inline_completed || producer == anchor) continue;
                if (synchronous[producer] && tasks[producer].active_mask != 0 && rank[producer] < rank[anchor])
                    continue;
                ready = false;
                break;
            }
            if (!ready || !tail.wave.place(id, static_cast<uint32_t>(task.block_num), task.active_mask)) continue;
            tail.tasks.push_back(id);
            if (tail.wave.full()) break;
        }
        if (!tail.wave.empty()) tails.push_back(std::move(tail));
    }
    return tails;
}

}  // namespace simpler::percounter
