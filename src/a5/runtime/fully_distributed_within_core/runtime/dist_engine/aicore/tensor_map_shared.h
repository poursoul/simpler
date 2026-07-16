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

PTO_DEVICE_FUNC uint32_t shared_range_hash(uint64_t addr) {
    addr *= 0x9E3779B97F4A7C15ULL;
    return static_cast<uint32_t>(addr >> (64 - kMapBucketShift));
}

template <typename TensorRef>
PTO_DEVICE_FUNC void shared_map_byte_range(const TensorRef &t, uint64_t &addr, uint64_t &lo, uint64_t &hi) {
    const uint64_t esz = get_element_size(t.dtype);
    addr = t.buffer.addr;
    lo = t.start_offset * esz;
    uint64_t ext;
    if (t.is_contiguous) {
        ext = 1;
        for (uint32_t i = 0; i < t.ndims; i++)
            ext *= t.shapes[i];
    } else {
        ext = t.extent_elem_cache;
    }
    hi = (t.start_offset + ext) * esz;
}

PTO_DEVICE_FUNC bool shared_map_lock(__gm__ SharedDistTensorMap &map) {
    while (atomic_exchange(map.lock, int64_t{1}, __ATOMIC_ACQUIRE) != 0 && !fatal_set()) {
        SPIN_WAIT_HINT();
    }
    return !fatal_set();
}

PTO_DEVICE_FUNC void shared_map_unlock(__gm__ SharedDistTensorMap &map) {
    store_barrier();
    atomic_exchange(map.lock, int64_t{0}, __ATOMIC_RELEASE);
}

PTO_DEVICE_FUNC int32_t shared_map_load_bucket(__gm__ const int32_t &bucket) {
#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(&bucket, sizeof(bucket));
#endif
    return bucket;
}

PTO_DEVICE_FUNC void shared_map_flush_bucket(__gm__ int32_t &bucket) {
#if defined(__CCE_AICORE__)
    dist_aicore_flush_region(&bucket, sizeof(bucket));
#else
    (void)bucket;
#endif
}

PTO_DEVICE_FUNC int64_t shared_load_publish_cursor() {
#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(&g_dist.producer_publish_cursor, sizeof(g_dist.producer_publish_cursor));
#endif
    return atomic_load(g_dist.producer_publish_cursor.v, __ATOMIC_ACQUIRE);
}

PTO_DEVICE_FUNC int64_t shared_load_published(int32_t task_id) {
#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(&g_dist.published[task_id], sizeof(g_dist.published[task_id]));
#endif
    return atomic_load(g_dist.published[task_id].v, __ATOMIC_ACQUIRE);
}

PTO_DEVICE_FUNC void shared_map_invalidate_control(__gm__ SharedDistTensorMap &map) {
#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(const_cast<__gm__ const int64_t *>(&map.high_water), 64);
#else
    (void)map;
#endif
}

PTO_DEVICE_FUNC void shared_map_flush_control(__gm__ SharedDistTensorMap &map) {
#if defined(__CCE_AICORE__)
    dist_aicore_flush_region(const_cast<__gm__ int64_t *>(&map.high_water), 64);
#else
    (void)map;
#endif
}

PTO_DEVICE_FUNC int32_t shared_map_alloc_entry_locked(__gm__ SharedDistTensorMap &map) {
    shared_map_invalidate_control(map);
    if (map.free_head >= 0) {
        const int32_t idx = static_cast<int32_t>(map.free_head);
        __gm__ SharedMapEntry &entry = map.entries[idx];
#if defined(__CCE_AICORE__)
        dist_aicore_invalidate_region(&entry, sizeof(entry));
#endif
        map.free_head = entry.next_in_task;
        return idx;
    }
    const int64_t idx = map.high_water++;
    if (idx < 0 || idx >= kSharedMapCap) {
        set_fatal();
        return -1;
    }
    return static_cast<int32_t>(idx);
}

PTO_DEVICE_FUNC void shared_map_unlink_symbol_locked(__gm__ SharedDistTensorMap &map, int32_t idx) {
    __gm__ SharedMapEntry &target = map.entries[idx];
    if (target.output_slot < 0) return;
    const uint32_t bucket = shared_symbol_hash(target.owner_task_id, static_cast<uint32_t>(target.output_slot));
    int32_t prev = -1;
    int32_t cur = shared_map_load_bucket(map.symbol_buckets[bucket]);
    while (cur >= 0) {
        __gm__ SharedMapEntry &entry = map.entries[cur];
#if defined(__CCE_AICORE__)
        dist_aicore_invalidate_region(&entry, sizeof(entry));
#endif
        const int32_t next = entry.next_in_symbol_bucket;
        if (cur == idx) {
            if (prev < 0) {
                map.symbol_buckets[bucket] = next;
                shared_map_flush_bucket(map.symbol_buckets[bucket]);
            } else {
                map.entries[prev].next_in_symbol_bucket = next;
#if defined(__CCE_AICORE__)
                dist_aicore_flush_region(&map.entries[prev], sizeof(map.entries[prev]));
#endif
            }
            target.next_in_symbol_bucket = -1;
            return;
        }
        prev = cur;
        cur = next;
    }
}

PTO_DEVICE_FUNC void shared_map_unlink_range_locked(__gm__ SharedDistTensorMap &map, int32_t idx) {
    __gm__ SharedMapEntry &target = map.entries[idx];
    if (target.range_bucket < 0) return;
    const uint32_t bucket = static_cast<uint32_t>(target.range_bucket);
    int32_t prev = -1;
    int32_t cur = shared_map_load_bucket(map.range_buckets[bucket]);
    while (cur >= 0) {
        __gm__ SharedMapEntry &entry = map.entries[cur];
#if defined(__CCE_AICORE__)
        dist_aicore_invalidate_region(&entry, sizeof(entry));
#endif
        const int32_t next = entry.next_in_range_bucket;
        if (cur == idx) {
            if (prev < 0) {
                map.range_buckets[bucket] = next;
                shared_map_flush_bucket(map.range_buckets[bucket]);
            } else {
                map.entries[prev].next_in_range_bucket = next;
#if defined(__CCE_AICORE__)
                dist_aicore_flush_region(&map.entries[prev], sizeof(map.entries[prev]));
#endif
            }
            target.next_in_range_bucket = -1;
            return;
        }
        prev = cur;
        cur = next;
    }
}

PTO_DEVICE_FUNC void shared_map_free_entry_locked(__gm__ SharedDistTensorMap &map, int32_t idx) {
    shared_map_unlink_symbol_locked(map, idx);
    shared_map_unlink_range_locked(map, idx);
    __gm__ SharedMapEntry &entry = map.entries[idx];
    entry.owner_task_id = -1;
    entry.output_slot = -1;
    entry.range_bucket = -1;
    entry.next_in_task = static_cast<int32_t>(map.free_head);
    map.free_head = idx;
#if defined(__CCE_AICORE__)
    dist_aicore_flush_region(&entry, sizeof(entry));
#endif
    shared_map_flush_control(map);
}

PTO_DEVICE_FUNC void shared_map_advance_retire(__gm__ SharedDistTensorMap &map, int32_t task_id, int32_t H) {
    (void)task_id;
    const int32_t frontier = static_cast<int32_t>(atomic_load(g_dist.frontier, __ATOMIC_ACQUIRE));
    const int32_t new_floor = frontier - H + 1;
    if (new_floor <= 0) return;
    if (!shared_map_lock(map)) return;
    int32_t cleaned = static_cast<int32_t>(map.cleaned_upto);
    if (new_floor <= cleaned) {
        if (new_floor > map.alive_floor) map.alive_floor = new_floor;
        shared_map_unlock(map);
        return;
    }
    for (int32_t id = cleaned; id < new_floor; id++) {
        const int32_t slot = id & kTaskWindowMask;
#if defined(__CCE_AICORE__)
        dist_aicore_invalidate_region(&map.task_heads[slot], sizeof(map.task_heads[slot]));
#endif
        int32_t prev = -1;
        int32_t cur = map.task_heads[slot];
        while (cur >= 0) {
            __gm__ SharedMapEntry &entry = map.entries[cur];
#if defined(__CCE_AICORE__)
            dist_aicore_invalidate_region(&entry, sizeof(entry));
#endif
            const int32_t next = entry.next_in_task;
            if (entry.owner_task_id == id) {
                if (prev < 0) {
                    map.task_heads[slot] = next;
#if defined(__CCE_AICORE__)
                    dist_aicore_flush_region(&map.task_heads[slot], sizeof(map.task_heads[slot]));
#endif
                } else {
                    map.entries[prev].next_in_task = next;
#if defined(__CCE_AICORE__)
                    dist_aicore_flush_region(&map.entries[prev], sizeof(map.entries[prev]));
#endif
                }
                shared_map_free_entry_locked(map, cur);
            } else {
                prev = cur;
            }
            cur = next;
        }
    }
    map.cleaned_upto = new_floor;
    map.alive_floor = new_floor;
    shared_map_flush_control(map);
    shared_map_unlock(map);
}

template <typename TensorRef>
PTO_DEVICE_FUNC void shared_map_insert_entry(
    __gm__ SharedDistTensorMap &map, int32_t task_id, int32_t output_slot, const TensorRef &tensor, bool publish_symbol,
    bool publish_range
) {
    if (!shared_map_lock(map)) return;
    const int32_t idx = shared_map_alloc_entry_locked(map);
    if (idx < 0) {
        shared_map_unlock(map);
        return;
    }
    __gm__ SharedMapEntry &entry = map.entries[idx];
    Tensor::copy(entry.tensor, tensor);
    uint64_t addr = 0;
    uint64_t lo = 0;
    uint64_t hi = 0;
    shared_map_byte_range(tensor, addr, lo, hi);
    entry.buf_addr = addr;
    entry.lo = lo;
    entry.hi = hi;
    entry.owner_task_id = task_id;
    entry.output_slot = output_slot;
    entry.range_bucket = -1;
    entry.next_in_symbol_bucket = -1;
    entry.next_in_range_bucket = -1;
    entry.next_in_task = map.task_heads[task_id & kTaskWindowMask];
    map.task_heads[task_id & kTaskWindowMask] = idx;
    uint32_t range_bucket = 0;
    uint32_t symbol_bucket = 0;
    if (publish_range) {
        range_bucket = shared_range_hash(entry.buf_addr);
        entry.range_bucket = static_cast<int32_t>(range_bucket);
    }
    if (publish_symbol) {
        symbol_bucket = shared_symbol_hash(task_id, static_cast<uint32_t>(output_slot));
    }
#if defined(__CCE_AICORE__)
    dist_aicore_flush_region(&entry, sizeof(entry));
#endif
    if (publish_range) entry.next_in_range_bucket = atomic_exchange(map.range_buckets[range_bucket], idx);
    if (publish_symbol) entry.next_in_symbol_bucket = atomic_exchange(map.symbol_buckets[symbol_bucket], idx);
    store_barrier();
#if defined(__CCE_AICORE__)
    dist_aicore_flush_region(&entry, sizeof(entry));
#endif
    if (publish_range) shared_map_flush_bucket(map.range_buckets[range_bucket]);
    if (publish_symbol) shared_map_flush_bucket(map.symbol_buckets[symbol_bucket]);
    shared_map_unlock(map);
}

PTO_DEVICE_FUNC void shared_map_insert_symbol(
    __gm__ SharedDistTensorMap &map, int32_t task_id, uint32_t output_slot, __gm__ const Tensor &tensor
) {
    shared_map_insert_entry(map, task_id, static_cast<int32_t>(output_slot), tensor, true, true);
}

template <typename TensorRef>
PTO_DEVICE_FUNC void
shared_map_insert_range(__gm__ SharedDistTensorMap &map, int32_t task_id, const TensorRef &tensor) {
    shared_map_insert_entry(map, task_id, -1, tensor, false, true);
}

PTO_DEVICE_FUNC bool shared_map_lookup_symbol(
    __gm__ SharedDistTensorMap &map, int32_t task_id, uint32_t output_slot, __gm__ const SharedMapEntry *&out
) {
    if (task_id < atomic_load(map.alive_floor, __ATOMIC_ACQUIRE)) {
        out = nullptr;
        return false;
    }
    const uint32_t bucket = shared_symbol_hash(task_id, output_slot);
    for (int32_t cur = shared_map_load_bucket(map.symbol_buckets[bucket]); cur >= 0;
         cur = map.entries[cur].next_in_symbol_bucket) {
        __gm__ const SharedMapEntry &entry = map.entries[cur];
#if defined(__CCE_AICORE__)
        dist_aicore_invalidate_region(&entry, sizeof(entry));
#endif
        if (entry.owner_task_id == task_id && entry.output_slot == static_cast<int32_t>(output_slot)) {
            out = &entry;
            return true;
        }
    }
    out = nullptr;
    return false;
}

template <typename TensorRef>
PTO_DEVICE_FUNC int32_t
shared_map_lookup_range(__gm__ SharedDistTensorMap &map, const TensorRef &tensor, int32_t consumer_task_id) {
    uint64_t addr, lo, hi;
    shared_map_byte_range(tensor, addr, lo, hi);
    const int32_t alive_floor = static_cast<int32_t>(atomic_load(map.alive_floor, __ATOMIC_ACQUIRE));
    int32_t best = -1;
    for (int32_t cur = shared_map_load_bucket(map.range_buckets[shared_range_hash(addr)]); cur >= 0;
         cur = map.entries[cur].next_in_range_bucket) {
        __gm__ const SharedMapEntry &entry = map.entries[cur];
#if defined(__CCE_AICORE__)
        dist_aicore_invalidate_region(&entry, sizeof(entry));
#endif
        if (entry.owner_task_id < alive_floor) continue;
        if (entry.owner_task_id >= consumer_task_id) continue;
        if (entry.buf_addr == addr && lo < entry.hi && entry.lo < hi && entry.owner_task_id > best) {
            best = entry.owner_task_id;
        }
    }
    return best;
}

PTO_DEVICE_FUNC void shared_publish_done(int32_t task_id) {
    if (task_id < 0 || task_id >= kFlagCap) return;
    atomic_exchange(g_dist.published[task_id].v, int64_t{1}, __ATOMIC_RELEASE);
#if defined(__CCE_AICORE__)
    dist_aicore_flush_region(&g_dist.published[task_id], sizeof(g_dist.published[task_id]));
#endif
    int64_t p = shared_load_publish_cursor();
    while (true) {
        const int64_t next = p + 1;
        if (next < 0 || next >= kFlagCap) break;
        if (shared_load_published(static_cast<int32_t>(next)) == 0) break;
        const int64_t old = atomic_fetch_max<int64_t>(g_dist.producer_publish_cursor.v, next);
#if defined(__CCE_AICORE__)
        if (next > old) {
            dist_aicore_flush_region(&g_dist.producer_publish_cursor, sizeof(g_dist.producer_publish_cursor));
        }
#endif
        p = old > next ? old : next;
    }
}

PTO_DEVICE_FUNC void shared_wait_published_before(__gm__ DistCore *self, int32_t task_id) {
    const int32_t target = task_id - 1;
    if (target < 0) return;
    while (shared_load_publish_cursor() < target && !fatal_set()) {
        drain_block_won(self);
        if (drain_phase_b(self) == 0) SPIN_WAIT_HINT();
    }
}

}  // namespace

#endif
