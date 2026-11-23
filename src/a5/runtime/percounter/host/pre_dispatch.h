/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "full_width_order.h"
#include "scheduler/wire_types.h"

namespace simpler::percounter {

/** One task per physical lane, with a stable block index for each placement. */
class PredispatchWave {
public:
    explicit PredispatchWave(uint32_t clusters) :
        clusters_(clusters),
        lanes_(clusters * PLATFORM_CORES_PER_BLOCKDIM, invalid_entry()) {
        owners_.reserve(clusters * PLATFORM_AIV_CORES_PER_BLOCKDIM);
    }

    bool empty() const { return occupied_ == 0; }
    bool full() const { return occupied_ == lanes_.size(); }
    bool has_mix() const { return has_mix_; }
    bool uses_type_layout() const { return !empty() && full_width_only_; }

    const std::vector<uint64_t> &lanes() const { return lanes_; }
    const std::array<uint64_t, PERCOUNTER_LIST_COUNT> &types() const { return types_; }

    bool place(uint32_t id, uint32_t blocks, uint32_t mask) {
        owners_.clear();
        const bool vectors = mask == 2;
        const bool full_vectors = vectors && blocks == clusters_ * PLATFORM_AIV_CORES_PER_BLOCKDIM;
        if (blocks == 0 || mask == 0 || mask == 4 || mask > 7) return false;
        if (vectors && !full_vectors) {
            // Use AIV1 first so AIV0 can publish ordinary MIX work.
            for (const uint32_t role : {2U, 1U}) {
                for (uint32_t cluster = 0; cluster < clusters_ && owners_.size() < blocks; ++cluster) {
                    const uint32_t lane = cluster * PLATFORM_CORES_PER_BLOCKDIM + role;
                    if (free(lane)) owners_.push_back(lane);
                }
            }
        } else {
            const uint32_t roles = full_vectors ? 6U : mask;
            for (uint32_t cluster = 0; cluster < clusters_ && owners_.size() < blocks; ++cluster) {
                const uint32_t base = cluster * PLATFORM_CORES_PER_BLOCKDIM;
                bool available = true;
                for (uint32_t role = 0; role < PLATFORM_CORES_PER_BLOCKDIM; ++role)
                    if ((roles & (1U << role)) != 0 && !free(base + role)) available = false;
                if (!available) continue;
                if (full_vectors) {
                    owners_.push_back(base + 1);
                    owners_.push_back(base + 2);
                } else {
                    owners_.push_back(base);
                }
            }
        }
        // A task's placement must succeed for all blocks and roles before
        // any of its lanes are committed.
        if (owners_.size() != blocks) return false;
        has_mix_ |= __builtin_popcount(mask) > 1;
        for (uint32_t block = 0; block < blocks; ++block) {
            const uint64_t entry = percounter_order_entry(id, block, blocks);
            if (vectors) {
                lanes_[owners_[block]] = entry;
                ++occupied_;
            } else {
                for (uint32_t role = 0; role < PLATFORM_CORES_PER_BLOCKDIM; ++role) {
                    if ((mask & (1U << role)) == 0) continue;
                    lanes_[owners_[block] + role] = entry;
                    ++occupied_;
                }
            }
        }
        if (full_vectors || (blocks == clusters_ && (mask == 1 || mask == 7))) {
            const uint64_t entry = percounter_order_entry(id, PERCOUNTER_ORDER_BLOCK_IDX_SPMD, blocks);
            if ((mask & 1U) != 0) types_[0] = entry;
            if ((mask & 6U) != 0) types_[1] = entry;
        } else {
            full_width_only_ = false;
        }
        return true;
    }

    static constexpr uint64_t invalid_entry() { return PERCOUNTER_ORDER_TASK_MASK; }

private:
    bool free(uint32_t lane) const { return lanes_[lane] == invalid_entry(); }

    uint32_t clusters_;
    std::vector<uint64_t> lanes_;
    std::vector<uint32_t> owners_;
    std::array<uint64_t, PERCOUNTER_LIST_COUNT> types_{invalid_entry(), invalid_entry()};
    uint32_t occupied_{0};
    bool full_width_only_{true};
    bool has_mix_{false};
};

struct StartupPredispatch {
    explicit StartupPredispatch(uint32_t clusters) :
        wave(clusters) {}

    PredispatchWave wave;
    std::vector<uint32_t> tasks;
};

template <typename Task, typename Plan>
inline std::optional<StartupPredispatch> plan_startup_wave(const Task *tasks, const Plan &plan, uint32_t clusters) {
    StartupPredispatch startup(clusters);
    uint8_t earlier_types = 0;
    for (const uint32_t id : plan.order) {
        const auto &task = tasks[id];
        if (task.graph_invocation >= 0 || plan.is_terminal(id)) break;
        if (task.inline_completed || task.published_off_lane || task.active_mask == 0) continue;
        const uint8_t types = percounter_task_types(task.active_mask);
        const bool prefix_drained = !plan.is_interval_entry(id) || (earlier_types & types) == 0;
        earlier_types |= types;
        if (!prefix_drained || plan.is_interior(id)) continue;

        // A pruned fanin can still require the normal entry prefix to run.
        // Startup is allowed only for tasks already ready in the logical DAG.
        bool ready = true;
        for (int32_t i = 0; i < task.fanin_count; ++i)
            if (!tasks[task.fanin_ids[i]].inline_completed) ready = false;
        if (!ready || !startup.wave.place(id, static_cast<uint32_t>(task.block_num), task.active_mask)) continue;
        startup.tasks.push_back(id);
        if (startup.wave.full()) break;
    }
    if (startup.wave.empty()) return std::nullopt;
    return startup;
}

}  // namespace simpler::percounter
