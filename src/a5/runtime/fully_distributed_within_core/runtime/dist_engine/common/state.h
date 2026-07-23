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

#include <cstddef>
#include <cstdint>

#include "dist_engine/common/target.h"

#include "dist_engine/dist_engine.h"
#include "common/core_type.h"
#include "pto2_dispatch_payload.h"
#include "pto_constants.h"
#include "pto_submit_types.h"
#include "pto_types.h"
#include "tensor.h"

struct PTO2Runtime;
struct L2TaskArgs;
class Runtime;

constexpr int32_t kDistRuntimeMaxWorker = 108;
#ifdef RUNTIME_MAX_WORKER
static_assert(kDistRuntimeMaxWorker == RUNTIME_MAX_WORKER, "dist_engine worker ABI must match Runtime");
#endif

#if PTO_FDWIC_SHARED_MAP
constexpr int32_t kPrivateSlots = 14;
#else
constexpr int32_t kPrivateSlots = 4;
#endif
constexpr int32_t kWonReserve = 2;
constexpr int32_t kMaxFanin = 16;
constexpr int32_t kMapCap = 16384;
constexpr int32_t kFlagCap = 1 << 16;
constexpr int32_t kTaskPayloadSlots = 2048;
constexpr int32_t kTaskPayloadMask = kTaskPayloadSlots - 1;
constexpr size_t kCacheLine = 64;
static_assert((kTaskPayloadSlots & kTaskPayloadMask) == 0, "task payload slots must be a power of two");
static_assert(sizeof(Tensor) % kCacheLine == 0, "Tensor descriptor must not share cachelines");

struct DistTaskPayload {
    Tensor tensors[MAX_TENSOR_ARGS];
};
static_assert(sizeof(DistTaskPayload) % kCacheLine == 0, "DistTaskPayload must not share cachelines");
static_assert(offsetof(DistTaskPayload, tensors) % kCacheLine == 0, "payload tensors must be cacheline-aligned");

struct DistOutputLayout {
    int32_t output_indices[MAX_TENSOR_ARGS];
    uint64_t buffer_sizes[MAX_TENSOR_ARGS];
    uint64_t total_output_size;
    int32_t output_count;
};

constexpr int32_t kHDefault = 64;

struct MapEntry {
    uint64_t buf_addr;
    uint64_t lo;
    uint64_t hi;
    int32_t producer;
    int32_t bucket;
    int32_t next_in_bucket;
    int32_t prev_in_bucket;
    int32_t next_in_task;
};

constexpr int32_t kMapBuckets = 1 << 13;
constexpr int32_t kMapBucketShift = 13;
constexpr int32_t kTaskWindow = 1 << 10;
constexpr int32_t kTaskWindowMask = kTaskWindow - 1;
constexpr size_t kDistTensorMapPayloadBytes =
    sizeof(MapEntry) * kMapCap + sizeof(int32_t) * kMapBuckets + sizeof(int32_t) * kTaskWindow + sizeof(int32_t) * 4;
constexpr size_t kDistTensorMapTailPad = 64 - (kDistTensorMapPayloadBytes % 64);

struct DistTensorMap {
    MapEntry entries[kMapCap];
    int32_t buckets[kMapBuckets];
    int32_t task_heads[kTaskWindow];
    int32_t free_head;
    int32_t high_water;
    int32_t alive_floor;
    int32_t cleaned_upto;
    uint8_t tail_pad[kDistTensorMapTailPad];
};
static_assert(sizeof(DistTensorMap) % 64 == 0, "DistTensorMap must not share cachelines");

enum class TracePhase : int32_t {
    Kernel = 0,
    Alloc = 1,
    Build = 2,
    DrainWon = 3,
    Replay = 4,
    RingBp = 5,
    EfDrain = 6,
    Commit = 7,
    Submit = 8,
    Materialize = 9,
    PrepareMap = 10,
    Claim = 11,
    Fanin = 12,
    Register = 13,
    Resolve = 14,
    ResolveWait = 15,
    ResolveInvalidate = 16,
    ResolveCopy = 17,
};

#if PTO_FDWIC_SHARED_MAP
constexpr int32_t kRingSlotTailPad = 96;
constexpr int32_t kBuiltSubtaskTailPad = 112;
#else
constexpr int32_t kRingSlotTailPad = 40;
constexpr int32_t kBuiltSubtaskTailPad = 56;
#endif

struct RingSlot {
    bool occupied;
    bool built;
    int32_t task_id;
    int32_t func_id;
    uint64_t function_bin_addr;

    int32_t tensor_count;
    int32_t scalar_count;
    uint8_t tensors_pad[32];
    Tensor tensors[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];
#if PTO_FDWIC_SHARED_MAP
    uint32_t shared_ref_mask;
    FdwicOutputRef shared_refs[MAX_TENSOR_ARGS];
#endif

    uint64_t args[PTO2_DISPATCH_MAX_ARGS];
    LocalContext local_ctx;
    GlobalContext global_ctx;

    int32_t fanin[kMaxFanin];
    int32_t fanin_count;

    bool is_multicore;
    int32_t won_block;
    int32_t won_slot;
    uint8_t tail_pad[kRingSlotTailPad];
};

struct BuiltSubtask {
    bool present;
    int32_t func_id;
    uint64_t function_bin_addr;
    int32_t tensor_count;
    int32_t scalar_count;
    uint8_t tensors_pad[40];
    Tensor tensors[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];
#if PTO_FDWIC_SHARED_MAP
    uint32_t shared_ref_mask;
    FdwicOutputRef shared_refs[MAX_TENSOR_ARGS];
#endif
    int32_t fanin[kMaxFanin];
    int32_t fanin_count;
    int32_t sub_block_id;
    uint8_t tail_pad[kBuiltSubtaskTailPad];
};

static_assert(offsetof(RingSlot, tensors) % kCacheLine == 0, "RingSlot tensors must be cacheline-aligned");
static_assert(offsetof(BuiltSubtask, tensors) % kCacheLine == 0, "BuiltSubtask tensors must be cacheline-aligned");
static_assert(sizeof(RingSlot) % kCacheLine == 0, "RingSlot must not share cachelines");
static_assert(sizeof(BuiltSubtask) % kCacheLine == 0, "BuiltSubtask must not share cachelines");

struct DrainedCell {
    volatile int64_t v;
    uint8_t pad[64 - sizeof(int64_t)];
};
static_assert(sizeof(DrainedCell) == 64, "DrainedCell must occupy one cacheline");

struct WonAtomicCell {
    volatile int64_t v;
    uint8_t pad[64 - sizeof(int64_t)];
};
static_assert(sizeof(WonAtomicCell) == 64, "WonSlot atomic cell must occupy one cacheline");

struct WonMetaLine {
    int32_t task_id;
    uint8_t pad[64 - sizeof(int32_t)];
};
static_assert(sizeof(WonMetaLine) == 64, "WonSlot metadata must occupy one cacheline");

constexpr int64_t kDrainedFree = 0;
constexpr int64_t kDrainedClaimed = 1;

constexpr int64_t kWonStateFree = 0;
constexpr int64_t kWonStateClaimed = 1;
constexpr int64_t kWonStatePublished = 2;

struct WonSlot {
    WonAtomicCell state;
    WonMetaLine meta;
    WonAtomicCell remaining;
    DrainedCell drained[PTO2_SUBTASK_SLOT_COUNT];
    BuiltSubtask lane[PTO2_SUBTASK_SLOT_COUNT];
};
static_assert(offsetof(WonSlot, state) % 64 == 0, "WonSlot state must be cacheline-aligned");
static_assert(offsetof(WonSlot, meta) % 64 == 0, "WonSlot metadata must be cacheline-aligned");
static_assert(offsetof(WonSlot, remaining) % 64 == 0, "WonSlot remaining must be cacheline-aligned");
static_assert(offsetof(WonSlot, drained) % 64 == 0, "WonSlot drained cells must be cacheline-aligned");
static_assert(offsetof(WonSlot, lane) % 64 == 0, "WonSlot lanes must be cacheline-aligned");
static_assert(sizeof(WonSlot) % 64 == 0, "WonSlot must not share cachelines");

struct BlockWon {
    WonSlot slots[kPrivateSlots];
    uint8_t any_pub_pad[64 - ((sizeof(WonSlot) * kPrivateSlots) % 64)];
    volatile int32_t any_pub;
    uint8_t any_pub_tail_pad[64 - sizeof(int32_t)];
};
static_assert(offsetof(BlockWon, slots) % 64 == 0, "BlockWon slots must be cacheline-aligned");
static_assert(offsetof(BlockWon, any_pub) % 64 == 0, "BlockWon any_pub must be cacheline-aligned");
static_assert(sizeof(BlockWon) % 64 == 0, "BlockWon must not share cachelines");

enum LaneId : int32_t { LANE_AIC = 0, LANE_AIV0 = 1, LANE_AIV1 = 2, LANE_NONE = -1 };
constexpr int32_t kLaneCount = 3;

struct CoreLayout {
    int32_t block_id;
    int32_t lane;
};

struct DistCore {
    int32_t core_idx;
    int32_t block_id;
    int32_t lane;
    int32_t sub_block_id;
    int32_t local_index;
    uint64_t heap_next;

    int32_t occupied_count;
    int32_t owned_total;
    uint64_t swimlane_last_cycle;

    uint8_t hot_prefix_pad[16];
#if !PTO_FDWIC_SHARED_MAP
    DistTensorMap map;
#endif
    RingSlot slots[kPrivateSlots];
    DistTaskPayload task_payloads[kTaskPayloadSlots];
};
#if !PTO_FDWIC_SHARED_MAP
static_assert(offsetof(DistCore, map) % 64 == 0, "DistCore map must be cacheline-aligned");
#endif
static_assert(offsetof(DistCore, slots) % 64 == 0, "DistCore slots must be cacheline-aligned");
static_assert(offsetof(DistCore, task_payloads) % 64 == 0, "DistCore task_payloads must be cacheline-aligned");

static_assert(PTO2_PACKED_OUTPUT_ALIGN >= kCacheLine);
static_assert((PTO2_PACKED_OUTPUT_ALIGN % kCacheLine) == 0);

#if PTO_FDWIC_SHARED_MAP
struct PreparedDeps {
    int32_t task_id;
    int32_t fanin_count;
    int32_t region_fanin_count;
    int32_t fanin[kMaxFanin];
    uint8_t pad[52];
};
static_assert(sizeof(PreparedDeps) == 128, "PreparedDeps must occupy two cachelines");
#endif

constexpr int32_t kCubeCursorShards = 8;
constexpr int32_t kCubeCursorShardMask = kCubeCursorShards - 1;
constexpr int32_t kVectorCursorShards = 16;
constexpr int32_t kVectorCursorShardMask = kVectorCursorShards - 1;
constexpr int32_t kAllocCursorShards = 8;
constexpr int32_t kAllocCursorShardMask = kAllocCursorShards - 1;
static_assert((kCubeCursorShards & kCubeCursorShardMask) == 0, "cube cursor shards must be a power of two");
static_assert((kVectorCursorShards & kVectorCursorShardMask) == 0, "vector cursor shards must be a power of two");
static_assert((kAllocCursorShards & kAllocCursorShardMask) == 0, "alloc cursor shards must be a power of two");

struct PaddedCursor {
    volatile int64_t v;
    uint8_t pad[kCacheLine - sizeof(int64_t)];
};
static_assert(sizeof(PaddedCursor) == kCacheLine, "PaddedCursor must occupy one cacheline");

struct DistTaskCell {
    volatile int64_t flag;
    volatile uint64_t vend;
#if PTO_FDWIC_SHARED_MAP
    volatile int64_t deps_prepared;
    uint8_t pad[kCacheLine - sizeof(int64_t) - sizeof(uint64_t) - sizeof(int64_t)];
#else
    uint8_t pad[kCacheLine - sizeof(int64_t) - sizeof(uint64_t)];
#endif
};
static_assert(sizeof(DistTaskCell) == kCacheLine);

#if PTO_FDWIC_SHARED_MAP
constexpr int32_t kSharedOutputMaxPerTask = 8;
constexpr int32_t kSharedHeapShards = 8;
constexpr int32_t kSharedHeapActiveShards = kSharedHeapShards;
constexpr int32_t kSharedRegionCap = kFlagCap;
constexpr int32_t kSharedRegionBuckets = kMapBuckets;
static_assert(kSharedHeapActiveShards > 0 && kSharedHeapActiveShards <= kSharedHeapShards);

struct SharedOutputCell {
    PaddedCursor published[kSharedOutputMaxPerTask];
    PaddedCursor last_writer[kSharedOutputMaxPerTask];
    Tensor tensors[kSharedOutputMaxPerTask];
};
static_assert(offsetof(SharedOutputCell, published) % 64 == 0, "shared output flags must be cacheline-aligned");
static_assert(sizeof(SharedOutputCell::published[0]) == kCacheLine, "shared output flag must own one cacheline");
static_assert(offsetof(SharedOutputCell, last_writer) % 64 == 0, "shared output writers must be cacheline-aligned");
static_assert(sizeof(SharedOutputCell::last_writer[0]) == kCacheLine, "shared output writer must own one cacheline");
static_assert(offsetof(SharedOutputCell, tensors) % 64 == 0, "shared output tensors must be cacheline-aligned");
static_assert(sizeof(SharedOutputCell) % 64 == 0, "SharedOutputCell must not share cachelines");

struct SharedRegionEntry {
    uint64_t buf_addr;
    uint64_t lo;
    uint64_t hi;
    int32_t producer;
    int32_t next_in_bucket;
    uint8_t pad[kCacheLine - sizeof(uint64_t) * 3 - sizeof(int32_t) * 2];
};
static_assert(sizeof(SharedRegionEntry) == kCacheLine, "SharedRegionEntry must occupy one cacheline");

struct SharedRegionMap {
    PaddedCursor high_water;
    PaddedCursor insert_lock;
    PaddedCursor buckets[kSharedRegionBuckets];
    SharedRegionEntry entries[kSharedRegionCap];
};
static_assert(offsetof(SharedRegionMap, buckets) % 64 == 0, "shared region buckets must be cacheline-aligned");
static_assert(sizeof(SharedRegionMap::buckets[0]) == kCacheLine, "shared region bucket must own one cacheline");
static_assert(offsetof(SharedRegionMap, entries) % 64 == 0, "shared region entries must be cacheline-aligned");
static_assert(sizeof(SharedRegionMap) % 64 == 0, "SharedRegionMap must not share cachelines");
#endif

struct DistGlobal {
    PaddedCursor cube_cursor[kCubeCursorShards];
    PaddedCursor vector_cursor[kVectorCursorShards];
    PaddedCursor alloc_cursor[kLaneCount][kAllocCursorShards];

    volatile int64_t frontier;
    uint8_t frontier_pad[kCacheLine - sizeof(int64_t)];
    int32_t H;
    uint8_t tasks_pad[kCacheLine - sizeof(int32_t)];
    DistTaskCell tasks[kFlagCap];
#if PTO_FDWIC_SHARED_MAP
    PaddedCursor shared_heap_cursor[kSharedHeapShards];
    PaddedCursor shared_heap_vend;
    SharedOutputCell shared_outputs[kFlagCap];
    SharedRegionMap shared_region;
    PaddedCursor joint_launch_expected[kDistRuntimeMaxWorker][kLaneCount];
    PaddedCursor joint_launch_drained[kDistRuntimeMaxWorker][kLaneCount];
#endif

    uint8_t *heap_base;
    size_t heap_size;

    const L2TaskArgs *orch_args;
    PTO2Runtime *rt;
    Runtime *runtime;

    uint8_t fatal_pad[24];
    volatile int32_t fatal;
    uint8_t fatal_tail_pad[kCacheLine - sizeof(int32_t)];

    int32_t num_workers;
    int32_t num_blocks;
    CoreLayout layout[kDistRuntimeMaxWorker];
    uint8_t blocks_pad[24];
    BlockWon blocks[kDistRuntimeMaxWorker];

    volatile int64_t replay_done;
    uint8_t replay_done_pad[kCacheLine - sizeof(int64_t)];

    volatile int64_t started_count;
    uint8_t started_count_pad[kCacheLine - sizeof(int64_t)];

#if PTO_FDWIC_SHARED_MAP
    PreparedDeps prepared_deps[kDistRuntimeMaxWorker];
#endif
    DistCore cores[kDistRuntimeMaxWorker];
};
static_assert(offsetof(DistGlobal, cube_cursor) % 64 == 0, "DistGlobal cube cursor must be cacheline-aligned");
static_assert(offsetof(DistGlobal, frontier) % 64 == 0, "DistGlobal frontier must be cacheline-aligned");
static_assert(offsetof(DistGlobal, tasks) % 64 == 0, "DistGlobal tasks must be cacheline-aligned");
#if PTO_FDWIC_SHARED_MAP
static_assert(offsetof(DistGlobal, shared_outputs) % 64 == 0, "DistGlobal shared_outputs must be cacheline-aligned");
static_assert(offsetof(DistGlobal, shared_region) % 64 == 0, "DistGlobal shared_region must be cacheline-aligned");
static_assert(
    offsetof(DistGlobal, joint_launch_expected) % 64 == 0,
    "DistGlobal joint_launch_expected must be cacheline-aligned"
);
static_assert(
    offsetof(DistGlobal, joint_launch_drained) % 64 == 0,
    "DistGlobal joint_launch_drained must be cacheline-aligned"
);
static_assert(offsetof(DistGlobal, prepared_deps) % 64 == 0, "DistGlobal prepared_deps must be cacheline-aligned");
#endif
static_assert(offsetof(DistGlobal, fatal) % 64 == 0, "DistGlobal fatal must be cacheline-aligned");
static_assert(offsetof(DistGlobal, blocks) % 64 == 0, "DistGlobal blocks must be cacheline-aligned");
static_assert(offsetof(DistGlobal, replay_done) % 64 == 0, "DistGlobal replay_done must be cacheline-aligned");
static_assert(offsetof(DistGlobal, started_count) % 64 == 0, "DistGlobal started_count must be cacheline-aligned");
static_assert(offsetof(DistGlobal, cores) % 64 == 0, "DistGlobal cores must be cacheline-aligned");

static_assert(sizeof(DistGlobal) <= kDistEngineGlobalStateSize, "DistGlobal exceeds the reserved runtime arena size");
static_assert(
    alignof(DistGlobal) <= kDistEngineGlobalStateAlign, "DistGlobal exceeds the reserved runtime arena align"
);
