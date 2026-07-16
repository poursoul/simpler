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

#if PTO_FDWIC_SHARED_TENSORMAP

namespace {

PTO_DEVICE_FUNC uint32_t shared_symbol_hash(int32_t task_id, uint32_t output_slot) {
    uint64_t x = static_cast<uint32_t>(task_id);
    x = (x << 32) ^ output_slot;
    x *= 0x9E3779B97F4A7C15ULL;
    return static_cast<uint32_t>(x >> (64 - kMapBucketShift));
}

PTO_DEVICE_FUNC int32_t shared_map_alloc_entry(__gm__ SharedDistTensorMap &map) {
    const int64_t idx = atomic_fetch_add<int64_t>(map.high_water, 1);
    if (idx < 0 || idx >= kSharedMapCap) {
        set_fatal();
        return -1;
    }
    return static_cast<int32_t>(idx);
}

PTO_DEVICE_FUNC void shared_map_insert_symbol(__gm__ SharedDistTensorMap &map, int32_t task_id, uint32_t output_slot) {
    const int32_t idx = shared_map_alloc_entry(map);
    if (idx < 0) return;
    const uint32_t bucket = shared_symbol_hash(task_id, output_slot);
    __gm__ SharedMapEntry &entry = map.entries[idx];
    entry.owner_task_id = task_id;
    entry.output_slot = static_cast<int32_t>(output_slot);
    entry.next_in_range_bucket = -1;
    entry.next_in_task = map.task_heads[task_id & kTaskWindowMask];
    entry.next_in_symbol_bucket = map.symbol_buckets[bucket];
    map.task_heads[task_id & kTaskWindowMask] = idx;
    store_barrier();
    map.symbol_buckets[bucket] = idx;
}

PTO_DEVICE_FUNC bool shared_map_lookup_symbol(
    __gm__ const SharedDistTensorMap &map, int32_t task_id, uint32_t output_slot, __gm__ const SharedMapEntry *&out
) {
    const uint32_t bucket = shared_symbol_hash(task_id, output_slot);
    for (int32_t cur = map.symbol_buckets[bucket]; cur >= 0; cur = map.entries[cur].next_in_symbol_bucket) {
        __gm__ const SharedMapEntry &entry = map.entries[cur];
        if (entry.owner_task_id == task_id && entry.output_slot == static_cast<int32_t>(output_slot)) {
            out = &entry;
            return true;
        }
    }
    out = nullptr;
    return false;
}

PTO_DEVICE_FUNC void shared_publish_done(int32_t task_id) {
    if (task_id < 0 || task_id >= kFlagCap) return;
    atomic_exchange(g_dist.published[task_id].v, int64_t{1}, __ATOMIC_RELEASE);
    int64_t p = atomic_load(g_dist.producer_publish_cursor.v, __ATOMIC_ACQUIRE);
    while (true) {
        const int64_t next = p + 1;
        if (next < 0 || next >= kFlagCap) break;
        if (atomic_load(g_dist.published[next].v, __ATOMIC_ACQUIRE) == 0) break;
        const int64_t old = atomic_fetch_max<int64_t>(g_dist.producer_publish_cursor.v, next);
        p = old > next ? old : next;
    }
}

PTO_DEVICE_FUNC void shared_wait_published_before(__gm__ DistCore *self, int32_t task_id) {
    const int32_t target = task_id - 1;
    if (target < 0) return;
    while (atomic_load(g_dist.producer_publish_cursor.v, __ATOMIC_ACQUIRE) < target && !fatal_set()) {
        drain_block_won(self);
        if (drain_phase_b(self) == 0) SPIN_WAIT_HINT();
    }
}

}  // namespace

#endif
