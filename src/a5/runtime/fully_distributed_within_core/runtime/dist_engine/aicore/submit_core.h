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

PTO_DEVICE_FUNC inline void dist_aicore_call_slot_kernel(__gm__ RingSlot &slot) {
#if defined(__CCE_AICORE__)
    const bool is_aic = g_ccec_core_type == static_cast<int32_t>(CoreType::AIC);
    if (is_aic) {
        if (pto_call_linked_kernel_aic != nullptr) {
            pto_call_linked_kernel_aic(slot.func_id, reinterpret_cast<__gm__ int64_t *>(slot.args));
        }
    } else if (pto_call_linked_kernel_aiv != nullptr) {
        pto_call_linked_kernel_aiv(slot.func_id, reinterpret_cast<__gm__ int64_t *>(slot.args));
    }
#else
    typedef void (*KernelFn)(__gm__ int64_t *);
    if (slot.function_bin_addr != 0) {
        KernelFn fn = reinterpret_cast<KernelFn>(slot.function_bin_addr);
        fn(reinterpret_cast<__gm__ int64_t *>(slot.args));
    }
#endif
}

namespace {

PTO_DEVICE_FUNC void publish_task_flag(int32_t task_id) {
    if (task_id < 0 || task_id >= kFlagCap) return;
    __gm__ DistTaskCell &cell = task_cell(task_id);
    atomic_exchange(cell.flag, int64_t{1}, __ATOMIC_RELEASE);
}

PTO_DEVICE_FUNC bool task_flag_ready(int32_t task_id, int memorder) {
    if (task_id < 0 || task_id >= kFlagCap) return false;
    __gm__ DistTaskCell &cell = task_cell(task_id);
    return atomic_load(cell.flag, memorder) != 0;
}

PTO_DEVICE_FUNC void flush_ring_slot_payload(__gm__ RingSlot &s, int32_t tc, int32_t sc, int32_t arg_count) {
#if defined(__CCE_AICORE__)
    dist_aicore_flush_region(&s, offsetof(RingSlot, tensors));
    dist_aicore_flush_region(s.tensors, static_cast<uint64_t>(tc) * sizeof(Tensor));
    dist_aicore_flush_region(s.scalars, static_cast<uint64_t>(sc) * sizeof(uint64_t));
    dist_aicore_flush_region(s.args, static_cast<uint64_t>(arg_count) * sizeof(uint64_t));
    dist_aicore_flush_region(&s.local_ctx, sizeof(LocalContext));
    dist_aicore_flush_region(&s.global_ctx, sizeof(GlobalContext));
    dist_aicore_flush_region(&s.fanin[0], offsetof(RingSlot, tail_pad) - offsetof(RingSlot, fanin));
#else
    (void)s;
    (void)tc;
    (void)sc;
    (void)arg_count;
#endif
}

PTO_DEVICE_FUNC void publish_ring_slot_built(__gm__ RingSlot &s) {
    store_barrier();
    s.built = true;
#if defined(__CCE_AICORE__)
    dist_aicore_flush_region(&s.built, sizeof(s.built));
#endif
}

#if PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC FdwicOutputRef dist_shared_ref_load(__gm__ const FdwicOutputRef &src) {
    FdwicOutputRef out;
    out.producer_task_id = src.producer_task_id;
    out.output_slot = src.output_slot;
    out.flags = src.flags;
    out.view_ndims = src.view_ndims;
    out.view_shape0 = src.view_shape0;
    out.view_offset0 = src.view_offset0;
    return out;
}

PTO_DEVICE_FUNC void dist_shared_ref_copy(__gm__ FdwicOutputRef &dst, FdwicOutputRef src) {
    dst.producer_task_id = src.producer_task_id;
    dst.output_slot = src.output_slot;
    dst.flags = src.flags;
    dst.view_ndims = src.view_ndims;
    dst.view_shape0 = src.view_shape0;
    dst.view_offset0 = src.view_offset0;
}

PTO_DEVICE_FUNC void dist_shared_ref_copy_from_gm(__gm__ FdwicOutputRef &dst, __gm__ const FdwicOutputRef &src) {
    dist_shared_ref_copy(dst, dist_shared_ref_load(src));
}

PTO_DEVICE_FUNC bool dist_resolve_shared_output_ref(FdwicOutputRef ref, __gm__ DistCore *self, __gm__ Tensor &resolved);
PTO_DEVICE_FUNC bool
dist_resolve_ready_shared_output_ref(FdwicOutputRef ref, __gm__ DistCore *self, __gm__ Tensor &resolved);

PTO_DEVICE_FUNC bool
dist_try_resolve_shared_output_ref(FdwicOutputRef ref, __gm__ DistCore *self, __gm__ Tensor &resolved) {
    if (ref.producer_task_id < 0 || ref.producer_task_id >= kFlagCap) {
        set_fatal();
        return false;
    }
    always_assert(ref.output_slot >= 0 && ref.output_slot < kSharedOutputMaxPerTask);
    __gm__ SharedOutputCell &cell = shared_output_cell(ref.producer_task_id);
    if (atomic_load(cell.published[ref.output_slot].v, __ATOMIC_ACQUIRE) != ref.producer_task_id) return false;
    return dist_resolve_ready_shared_output_ref(ref, self, resolved);
}

PTO_DEVICE_FUNC bool dist_resolve_slot_shared_refs(__gm__ DistCore *self, __gm__ RingSlot &s) {
    if (s.shared_ref_mask == 0) return true;
    TRACE_SPAN_BEGIN(resolve_trace);
    int32_t resolved_count = 0;
    for (int32_t i = 0; i < s.tensor_count; i++) {
        if ((s.shared_ref_mask & (1u << static_cast<uint32_t>(i))) == 0) continue;
        if (!dist_resolve_ready_shared_output_ref(dist_shared_ref_load(s.shared_refs[i]), self, s.tensors[i]))
            return false;
        resolved_count++;
    }
    TRACE_SPAN_END(resolve_trace, self, s.task_id, s.func_id, TracePhase::Resolve, 1u, resolved_count);
    return true;
}
#endif

PTO_DEVICE_FUNC void store_task_vend(int32_t task_id, uint64_t vend) {
    if (task_id < 0 || task_id >= kFlagCap) return;
    __gm__ DistTaskCell &cell = task_cell(task_id);
    atomic_exchange(cell.vend, vend, __ATOMIC_RELAXED);
}

PTO_DEVICE_FUNC void store_won_remaining(__gm__ WonSlot &w, int32_t count) {
    atomic_exchange(w.remaining.v, static_cast<int64_t>(count), __ATOMIC_RELAXED);
}

PTO_DEVICE_FUNC void reset_won_lane(__gm__ WonSlot &w, int32_t lane) {
    atomic_exchange(w.drained[lane].v, kDrainedClaimed);
    w.lane[lane].present = false;
}

PTO_DEVICE_FUNC bool claim_won_lane(__gm__ WonSlot &w, int32_t lane) {
    return atomic_exchange(w.drained[lane].v, kDrainedClaimed) == kDrainedFree;
}

PTO_DEVICE_FUNC void publish_won_slot(__gm__ WonSlot &w) { atomic_exchange(w.state.v, kWonStatePublished); }

PTO_DEVICE_FUNC bool decrement_won_remaining_is_last(__gm__ WonSlot &w) {
    return atomic_fetch_sub<int64_t>(w.remaining.v, 1) == 1;
}

PTO_DEVICE_FUNC void clear_won_slot_state(__gm__ WonSlot &w) { atomic_exchange(w.state.v, kWonStateFree); }

#if PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC void record_joint_launch_drained(int32_t block, int32_t lane) {
    if (block < 0 || block >= kDistRuntimeMaxWorker || lane < 0 || lane >= kLaneCount) return;
    atomic_fetch_add<int64_t>(g_dist.joint_launch_drained[block][lane].v, 1, __ATOMIC_ACQ_REL);
}

PTO_DEVICE_FUNC bool joint_launches_drained_for_lane(__gm__ DistCore *self) {
    if (self->block_id < 0 || self->block_id >= kDistRuntimeMaxWorker || self->lane < 0 || self->lane >= kLaneCount)
        return true;
    const int64_t expected = atomic_load(g_dist.joint_launch_expected[self->block_id][self->lane].v, __ATOMIC_ACQUIRE);
    const int64_t drained = atomic_load(g_dist.joint_launch_drained[self->block_id][self->lane].v, __ATOMIC_ACQUIRE);
    return drained >= expected;
}

PTO_DEVICE_FUNC void publish_joint_followers_when_ready(__gm__ RingSlot &s) {
    if (!s.is_multicore || s.won_block < 0 || s.won_slot < 0) return;
    __gm__ WonSlot &w = g_dist.blocks[s.won_block].slots[s.won_slot];
    if (atomic_load(w.state.v) != kWonStateClaimed) return;
    store_barrier();
    publish_won_slot(w);
    atomic_exchange(g_dist.blocks[s.won_block].any_pub, 1);
}
#endif

PTO_DEVICE_FUNC int64_t load_frontier_for_advance() { return atomic_load(g_dist.frontier); }

PTO_DEVICE_FUNC bool try_advance_frontier_to(int64_t &frontier, int64_t next) {
    const int64_t old = atomic_fetch_max<int64_t>(g_dist.frontier, next);
    frontier = old > next ? old : next;
    return next > old;
}

PTO_DEVICE_FUNC void advance_frontier_until(int32_t target_task_id, int32_t max_steps) {
#if defined(__CCE_AICORE__)
    if (g_dist_ptr == nullptr) return;
#endif
    int64_t f = load_frontier_for_advance();
    int32_t steps = 0;
    while (max_steps < 0 || steps < max_steps) {
        const int64_t next = f + 1;
        if (next >= kFlagCap) break;
        if (target_task_id >= 0 && next > target_task_id) break;
        if (!task_flag_ready(static_cast<int32_t>(next), __ATOMIC_ACQUIRE)) break;
        try_advance_frontier_to(f, next);
        steps++;
    }
}

PTO_DEVICE_FUNC void advance_frontier() { advance_frontier_until(/*target_task_id=*/-1, /*max_steps=*/-1); }

PTO_DEVICE_FUNC void complete_executed_task(__gm__ DistCore *self, int32_t task_id) {
    TRACE_SPAN_BEGIN(commit_trace);
    store_task_vend(task_id, self->heap_next);
    store_barrier();
    publish_task_flag(task_id);
#if !PTO_FDWIC_SHARED_MAP
    advance_frontier();
#endif
    TRACE_SPAN_END(commit_trace, self, task_id, -1, TracePhase::Commit, 0, 0);
}

PTO_DEVICE_FUNC void execute_slot([[maybe_unused]] __gm__ DistCore *self, __gm__ RingSlot &s) {
#if PTO_FDWIC_SHARED_MAP
    if (!dist_resolve_slot_shared_refs(self, s)) return;
#endif
    typedef void (*KernelFn)(__gm__ int64_t *);
#if DIST_SIM_HOST_CLOCK
    const Runtime *rt = g_dist.runtime;
    const int32_t sim_ns =
        (rt != nullptr && rt->use_example_exec_time_ && s.func_id >= 0 && s.func_id < RUNTIME_MAX_FUNC_ID) ?
            rt->example_exec_time_ns_[s.func_id] :
            0;
    if (sim_ns > 0) {
        const uint64_t t0 = now_ns();
        const uint64_t target = t0 + static_cast<uint64_t>(sim_ns);
        TRACE_SPAN_BEGIN(kernel_trace);
        while (now_ns() < target) { /* spin: emulate kernel busy time */
        }
        TRACE_SPAN_END(
            kernel_trace, self, s.task_id, s.func_id, TracePhase::Kernel, static_cast<uint32_t>(s.is_multicore ? 1 : 0),
            0
        );
    } else if (s.function_bin_addr != 0 && !g_skip_exec) {
        KernelFn fn = reinterpret_cast<KernelFn>(s.function_bin_addr);
        TRACE_SPAN_BEGIN(kernel_trace);
        fn(reinterpret_cast<__gm__ int64_t *>(s.args));
        TRACE_SPAN_END(
            kernel_trace, self, s.task_id, s.func_id, TracePhase::Kernel, static_cast<uint32_t>(s.is_multicore ? 1 : 0),
            0
        );
    }
#else
    TRACE_SPAN_BEGIN(kernel_trace);
    dist_aicore_call_slot_kernel(s);
    TRACE_SPAN_END(
        kernel_trace, self, s.task_id, s.func_id, TracePhase::Kernel, static_cast<uint32_t>(s.is_multicore ? 1 : 0), 0
    );
#endif
    store_barrier();
    if (s.is_multicore) {
        __gm__ WonSlot &w = g_dist.blocks[s.won_block].slots[s.won_slot];
        if (decrement_won_remaining_is_last(w)) {
            clear_won_slot_state(w);
            complete_executed_task(self, s.task_id);
        }
    } else {
        complete_executed_task(self, s.task_id);
    }
    TRACE_INSTANT(self, s.task_id, s.func_id, TracePhase::Commit, static_cast<uint32_t>(s.is_multicore ? 1 : 0));
    s.built = false;
    s.occupied = false;
}

PTO_DEVICE_FUNC int32_t drain_phase_b(__gm__ DistCore *self) {
    if (self->occupied_count == 0) return 0;
    int32_t freed = 0;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ RingSlot &s = self->slots[i];
        if (!s.occupied || !s.built) continue;
        bool ready = true;
        for (int32_t f = 0; f < s.fanin_count; f++) {
            if (!task_flag_ready(s.fanin[f], __ATOMIC_ACQUIRE)) {
                ready = false;
                break;
            }
        }
        if (!ready) continue;
#if PTO_FDWIC_SHARED_MAP
        publish_joint_followers_when_ready(s);
#endif
        execute_slot(self, s);
        self->occupied_count--;
        freed++;
    }
    return freed;
}

PTO_DEVICE_FUNC int32_t alloc_ring_slot(__gm__ DistCore *self) {
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        if (!self->slots[i].occupied) return i;
    }
    return -1;
}

PTO_DEVICE_FUNC inline int32_t kernel_id_for_lane(const MixedKernels &mixed, int32_t lane) {
    switch (lane) {
    case LANE_AIC:
        return mixed.aic_kernel_id;
    case LANE_AIV0:
        return mixed.aiv0_kernel_id;
    case LANE_AIV1:
        return mixed.aiv1_kernel_id;
    default:
        return INVALID_KERNEL_ID;
    }
}

PTO_DEVICE_FUNC inline bool lane_active(const ActiveMask &M, int32_t lane) {
    return M.subtask_active(static_cast<PTO2SubtaskSlot>(lane));
}

PTO_DEVICE_FUNC inline bool aic_lane_active(const ActiveMask &M) {
    return (M.core_mask() & PTO2_SUBTASK_MASK_AIC) != 0;
}

PTO_DEVICE_FUNC inline bool aiv_lane_active(const ActiveMask &M) {
    return (M.core_mask() & PTO2_SUBTASK_MASK_AIV) != 0;
}

PTO_DEVICE_FUNC inline bool dist_current_core_is_aic() {
#if defined(__CCE_AICORE__)
    return g_ccec_core_type == static_cast<int32_t>(CoreType::AIC);
#else
    return g_host_core_type == static_cast<int32_t>(CoreType::AIC);
#endif
}

PTO_DEVICE_FUNC inline bool dist_current_core_is_aiv() {
#if defined(__CCE_AICORE__)
    return g_ccec_core_type == static_cast<int32_t>(CoreType::AIV);
#else
    return g_host_core_type == static_cast<int32_t>(CoreType::AIV);
#endif
}

PTO_DEVICE_FUNC void build_ring_slot_from_built_subtask(
    __gm__ RingSlot &s, int32_t task_id, __gm__ const BuiltSubtask &b, int32_t won_block, int32_t won_slot
) {
    s.occupied = true;
    s.task_id = task_id;
    s.func_id = b.func_id;
    s.function_bin_addr = b.function_bin_addr;
    s.built = false;
    s.tensor_count = b.tensor_count;
    s.scalar_count = b.scalar_count;
    for (int32_t i = 0; i < b.tensor_count; i++)
        Tensor::copy(s.tensors[i], b.tensors[i]);
    for (int32_t j = 0; j < b.scalar_count; j++)
        s.scalars[j] = b.scalars[j];
#if PTO_FDWIC_SHARED_MAP
    s.shared_ref_mask = b.shared_ref_mask;
    for (int32_t i = 0; i < b.tensor_count; i++)
        if ((b.shared_ref_mask & (1u << static_cast<uint32_t>(i))) != 0)
            dist_shared_ref_copy_from_gm(s.shared_refs[i], b.shared_refs[i]);
#endif
    int32_t n = 0;
    for (int32_t i = 0; i < b.tensor_count; i++)
        s.args[n++] = reinterpret_cast<uint64_t>(&s.tensors[i]);
    for (int32_t j = 0; j < b.scalar_count; j++)
        s.args[n++] = s.scalars[j];
    s.local_ctx.s_block_idx = 0;
    s.local_ctx.s_block_num = 1;
    s.local_ctx.async_ctx.completion_count = nullptr;
    s.local_ctx.async_ctx.completion_error_code = nullptr;
    s.local_ctx.async_ctx.completion_entries = nullptr;
    s.local_ctx.async_ctx.completion_capacity = 0;
    s.local_ctx.async_ctx.task_token.raw = UINT64_MAX;
    s.global_ctx.sub_block_id = b.sub_block_id;
    s.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&s.local_ctx);
    s.args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&s.global_ctx);
    s.fanin_count = b.fanin_count;
    for (int32_t k = 0; k < b.fanin_count; k++)
        s.fanin[k] = b.fanin[k];
    s.is_multicore = true;
    s.won_block = won_block;
    s.won_slot = won_slot;
    flush_ring_slot_payload(s, b.tensor_count, b.scalar_count, n);
    publish_ring_slot_built(s);
}

PTO_DEVICE_FUNC bool drain_block_won(__gm__ DistCore *self) {
    if (self->lane == LANE_AIC || self->lane == LANE_NONE) return false;
    __gm__ BlockWon &bw = g_dist.blocks[self->block_id];
    if (atomic_load(bw.any_pub) == 0) return false;
    bool drained = false;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ WonSlot &w = bw.slots[i];
        if (atomic_load(w.state.v) != kWonStatePublished) continue;
#if defined(__CCE_AICORE__)
        dist_aicore_invalidate_region(&w.lane[self->lane].present, sizeof(w.lane[self->lane].present));
#endif
        if (!w.lane[self->lane].present) continue;
        if (!claim_won_lane(w, self->lane)) continue;
        int32_t si = alloc_ring_slot(self);
        if (si < 0) {
            atomic_exchange(w.drained[self->lane].v, kDrainedFree);
            return drained;
        }
#if defined(__CCE_AICORE__)
        dist_aicore_invalidate_region(&w.meta, sizeof(w.meta));
        dist_aicore_invalidate_region(&w.lane[self->lane], sizeof(w.lane[self->lane]));
#endif
        const int32_t task_id = w.meta.task_id;
        __gm__ const BuiltSubtask &b = w.lane[self->lane];
        TRACE_SPAN_BEGIN(drain_won_trace);
        build_ring_slot_from_built_subtask(self->slots[si], task_id, b, self->block_id, i);
        TRACE_SPAN_END(
            drain_won_trace, self, task_id, b.func_id, TracePhase::DrainWon, /*flags=*/1u, static_cast<uint32_t>(i)
        );
        self->occupied_count++;
        self->owned_total++;
#if PTO_FDWIC_SHARED_MAP
        record_joint_launch_drained(self->block_id, self->lane);
#endif
        drained = true;
    }
    return drained;
}

PTO_DEVICE_FUNC bool drain_block_won_if_enabled(__gm__ DistCore *self) {
    if (!g_fdwic_block_won_enabled) return false;
    return drain_block_won(self);
}

PTO_DEVICE_FUNC bool has_pending_won(__gm__ DistCore *self) {
    if (self->lane == LANE_AIC || self->lane == LANE_NONE) return false;
    __gm__ BlockWon &bw = g_dist.blocks[self->block_id];
    if (atomic_load(bw.any_pub) == 0) return false;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ WonSlot &w = bw.slots[i];
        if (atomic_load(w.state.v) != kWonStatePublished) continue;
#if defined(__CCE_AICORE__)
        dist_aicore_invalidate_region(&w.lane[self->lane].present, sizeof(w.lane[self->lane].present));
#endif
        if (!w.lane[self->lane].present) continue;
        if (atomic_load(w.drained[self->lane].v) == kDrainedFree) return true;
    }
    return false;
}

#if PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC bool dist_submit_has_drain_work(__gm__ DistCore *self) {
    if (self->occupied_count != 0) return true;
    return has_pending_won(self);
}

#endif

PTO_DEVICE_FUNC void dist_submit_execute_first(__gm__ DistCore *self) {
    TRACE_LAP_RESET(self);
    if (!fatal_set()) {
        drain_block_won_if_enabled(self);
        drain_phase_b(self);
    }
    TRACE_LAP(self, self->local_index, -1, TracePhase::EfDrain);
}

enum class DistSubmitKind : int32_t {
    Kernel = 0,
    Alloc = 1,
};

struct DistSubmitCtx {
    __gm__ DistCore *self;
    __gm__ DistTaskPayload *payload;
    int32_t task_id;
    int32_t tensor_count;
    int32_t scalar_count;
    uint64_t output_bytes;
    TaskOutputTensors result;
#if PTO_FDWIC_SHARED_MAP
    SharedTaskOutputs shared_result;
#endif
    int32_t fanin[kMaxFanin];
    int32_t fanin_count;
    int32_t kernel_id;
    bool won;
    bool joint;
    bool joint_init;
    int32_t joint_block;
    int32_t joint_slot;
    int32_t joint_count;
};

#if PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC bool
dist_populate_built_subtask_shared_refs(__gm__ BuiltSubtask &b, const L0TaskArgs &args, const DistSubmitCtx &ctx) {
    b.shared_ref_mask = 0;
    for (int32_t i = 0; i < ctx.tensor_count; i++) {
        if (!args.tensor(i).tensor_from_shared_output()) continue;
        const FdwicOutputRef ref = args.tensor(i).shared_output_ref();
        if (!dist_resolve_shared_output_ref(ref, ctx.self, b.tensors[i])) return false;
    }
    return true;
}
#else
PTO_DEVICE_FUNC bool
dist_populate_built_subtask_shared_refs(__gm__ BuiltSubtask &, const L0TaskArgs &, const DistSubmitCtx &) {
    return true;
}
#endif

PTO_DEVICE_FUNC void dist_submit_begin(__gm__ DistCore *self, const L0TaskArgs &args, DistSubmitCtx &ctx) {
    ctx.self = self;
    ctx.task_id = ctx.self->local_index++;
    ctx.payload = &ctx.self->task_payloads[ctx.task_id & kTaskPayloadMask];
    ctx.result.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(ctx.task_id)));
#if PTO_FDWIC_SHARED_MAP
    ctx.shared_result.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(ctx.task_id)));
#endif
    ctx.tensor_count = args.tensor_count();
    ctx.scalar_count = args.scalar_count();
    ctx.output_bytes = 0;
    ctx.fanin_count = 0;
    ctx.kernel_id = INVALID_KERNEL_ID;
    ctx.won = false;
    ctx.joint = false;
    ctx.joint_init = false;
    ctx.joint_block = -1;
    ctx.joint_slot = -1;
    ctx.joint_count = 0;
}

PTO_DEVICE_FUNC void dist_submit_begin_presubmit(__gm__ DistCore *self, DistSubmitCtx &ctx) {
    ctx.self = self;
    ctx.task_id = ctx.self->local_index++;
    ctx.payload = &ctx.self->task_payloads[ctx.task_id & kTaskPayloadMask];
    ctx.result.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(ctx.task_id)));
#if PTO_FDWIC_SHARED_MAP
    ctx.shared_result.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(ctx.task_id)));
#endif
    ctx.tensor_count = 0;
    ctx.scalar_count = 0;
    ctx.output_bytes = 0;
    ctx.fanin_count = 0;
    ctx.kernel_id = INVALID_KERNEL_ID;
    ctx.won = false;
    ctx.joint = false;
    ctx.joint_init = false;
    ctx.joint_block = -1;
    ctx.joint_slot = -1;
    ctx.joint_count = 0;
}

PTO_DEVICE_FUNC void dist_submit_begin_from_token(
    __gm__ DistCore *self, const SubmitToken &tok, const L0TaskArgs &args, DistSubmitCtx &ctx
) {
    ctx.self = self;
    ctx.task_id = tok.task_id;
    if (ctx.task_id < 0) {
        ctx.payload = nullptr;
    } else {
        ctx.payload = &ctx.self->task_payloads[ctx.task_id & kTaskPayloadMask];
    }
    ctx.result.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(ctx.task_id)));
#if PTO_FDWIC_SHARED_MAP
    ctx.shared_result.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(ctx.task_id)));
#endif
    ctx.tensor_count = args.tensor_count();
    ctx.scalar_count = args.scalar_count();
    ctx.output_bytes = 0;
    ctx.fanin_count = 0;
    ctx.kernel_id = tok.kernel_id;
    ctx.won = tok.won;
    ctx.joint = tok.joint;
    ctx.joint_init = tok.joint && tok.won;
    ctx.joint_block = tok.joint_block;
    ctx.joint_slot = -1;
    ctx.joint_count = tok.joint_count;
}

PTO_DEVICE_FUNC bool dist_submit_check_task_cap(const DistSubmitCtx &ctx, DistSubmitKind kind) {
    if (ctx.task_id < kFlagCap) return true;
    set_fatal();
    if (kind == DistSubmitKind::Alloc) {
        DIST_ERRF("[dist_engine] alloc task id %d exceeds kFlagCap %d\n", ctx.task_id, kFlagCap);
    } else {
        DIST_ERRF(
            "[dist_engine] task id %d exceeds kFlagCap %d (enlarge or window the flag/vend rings)\n", ctx.task_id,
            kFlagCap
        );
    }
    return false;
}

PTO_DEVICE_FUNC void dist_submit_wait_heap_reuse_window(__gm__ DistCore *self, int32_t task_id) {
    const int32_t target = task_id - g_dist.H;
    if (target < 0) return;
    while (!fatal_set() && static_cast<int32_t>(atomic_load(g_dist.frontier, __ATOMIC_ACQUIRE)) < target) {
        advance_frontier_until(target, /*max_steps=*/64);
        drain_block_won_if_enabled(self);
        if (drain_phase_b(self) == 0) SPIN_WAIT_HINT();
    }
}

PTO_DEVICE_FUNC void calculate_output_layout(const L0TaskArgs &args, DistOutputLayout &layout) {
    layout.total_output_size = 0;
    layout.output_count = 0;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        const int32_t output_ordinal = layout.output_count++;
        layout.output_indices[output_ordinal] = i;
        layout.buffer_sizes[output_ordinal] = TensorCreateInfo::buffer_size_bytes(args.tensor(i).create_info());
        layout.total_output_size += PTO2_ALIGN_UP(layout.buffer_sizes[output_ordinal], PTO2_PACKED_OUTPUT_ALIGN);
    }
}

PTO_DEVICE_FUNC bool
dist_submit_reserve_output_heap(DistSubmitCtx &ctx, uint64_t total, DistSubmitKind kind, uint64_t &task_base) {
    const size_t ring = g_dist.heap_size;
#if PTO_FDWIC_SHARED_MAP
    const uint64_t shard_raw = static_cast<uint64_t>(ring / kSharedHeapActiveShards);
    const uint64_t shard_span = (shard_raw / PTO2_PACKED_OUTPUT_ALIGN) * PTO2_PACKED_OUTPUT_ALIGN;
    if (shard_span == 0 || total > shard_span) {
        set_fatal();
        DIST_ERRF(
            "[dist_engine] shared heap shard too small for task %d outputs %llu B (heap=%zu shards=%d)\n", ctx.task_id,
            (unsigned long long)total, ring, kSharedHeapActiveShards
        );
        return false;
    }
    const int32_t shard = ctx.task_id % kSharedHeapActiveShards;
    const uint64_t reserve = PTO2_ALIGN_UP(total, PTO2_PACKED_OUTPUT_ALIGN);
    uint64_t cursor = 0;
    uint64_t offset = 0;
    for (int32_t retry = 0; retry < 64; retry++) {
        cursor = static_cast<uint64_t>(atomic_fetch_add<int64_t>(g_dist.shared_heap_cursor[shard].v, reserve));
        offset = cursor % shard_span;
        if (cursor >= shard_span) {
            dist_submit_wait_heap_reuse_window(ctx.self, ctx.task_id);
            if (fatal_set()) return false;
        }
        if (offset + total <= shard_span) break;
        const uint64_t after = cursor + reserve;
        const uint64_t after_offset = after % shard_span;
        if (after_offset != 0) {
            atomic_fetch_add<int64_t>(g_dist.shared_heap_cursor[shard].v, shard_span - after_offset);
        }
        offset = shard_span;
    }
    if (offset + total > shard_span) {
        set_fatal();
        DIST_ERRF(
            "[dist_engine] shared heap shard wrap retry exhausted for task %d outputs %llu B at offset %llu/%llu\n",
            ctx.task_id, (unsigned long long)total, (unsigned long long)offset, (unsigned long long)shard_span
        );
        return false;
    }
    const uint64_t global_vend =
        static_cast<uint64_t>(atomic_fetch_add<int64_t>(g_dist.shared_heap_vend.v, reserve)) + reserve;
    ctx.self->heap_next = global_vend;
    task_base = static_cast<uint64_t>(shard) * shard_span + offset;
#else
    task_base = PTO2_ALIGN_UP(ctx.self->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    if ((task_base % ring) + total > ring) {
        task_base = ((task_base / ring) + 1) * ring;
    }
    ctx.self->heap_next = task_base + total;
#endif
    (void)kind;
    return true;
}

PTO_DEVICE_FUNC bool dist_submit_materialize_args(const L0TaskArgs &args, DistSubmitCtx &ctx, DistSubmitKind kind) {
    if (ctx.payload == nullptr) return false;
    ctx.tensor_count = args.tensor_count();
    ctx.scalar_count = args.scalar_count();

    const size_t ring = g_dist.heap_size;
    DistOutputLayout layout;
    calculate_output_layout(args, layout);
    const uint64_t total = layout.total_output_size;
    uint64_t task_base = 0;
    if (total > 0 && g_dist.heap_base != nullptr) {
        if (total > ring) {
            set_fatal();
            if (kind == DistSubmitKind::Alloc) {
                DIST_ERRF(
                    "[dist_engine] alloc task %d outputs %llu B exceed heap ring %zu B\n", ctx.task_id,
                    (unsigned long long)total, ring
                );
            } else {
                DIST_ERRF(
                    "[dist_engine] task %d outputs %llu B exceed heap ring %zu B (enlarge PTO_DIST_HEAP_MB)\n",
                    ctx.task_id, (unsigned long long)total, ring
                );
            }
            return false;
        }
        if (!dist_submit_reserve_output_heap(ctx, total, kind, task_base)) return false;
    }

    uint64_t output_offset = 0;
#if PTO_FDWIC_SHARED_MAP
    __gm__ SharedOutputCell &shared_cell = shared_output_cell(ctx.task_id);
#endif
    for (int32_t output_ordinal = 0; output_ordinal < layout.output_count; output_ordinal++) {
        const int32_t i = layout.output_indices[output_ordinal];
        const auto &ci = args.tensor(i).create_info();
        if (g_dist.heap_base == nullptr) {
            set_fatal();
            if (kind == DistSubmitKind::Alloc) {
                DIST_ERRF("[dist_engine] GM output heap not allocated at alloc %d\n", ctx.task_id);
            } else {
                DIST_ERRF("[dist_engine] GM output heap not allocated at task %d\n", ctx.task_id);
            }
            return false;
        }
        const uint64_t buffer_size = layout.buffer_sizes[output_ordinal];
        const uint64_t phys = (task_base + output_offset) % ring;
        __gm__ Tensor &slot_t = ctx.payload->tensors[i];
        Tensor materialized;
        always_assert(ci.ndims > 0 && ci.ndims <= MAX_TENSOR_DIMS);
        materialized.buffer.addr = reinterpret_cast<uint64_t>(g_dist.heap_base + phys);
        materialized.buffer.size = buffer_size;
        materialized.owner_task_id.raw = UINT64_MAX;
        materialized.start_offset = ci.start_offset;
        materialized.version = ci.version;
        materialized.ndims = ci.ndims;
        materialized.dtype = ci.dtype;
        materialized.manual_dep = ci.manual_dep;
        materialized.is_contiguous = ci.is_contiguous;
        materialized.child_memory = ci.__pad_flags__;
        for (uint32_t dim = 0; dim < MAX_TENSOR_DIMS; dim++) {
            materialized.shapes[dim] = ci.shapes[dim];
        }
        uint32_t stride = 1;
        for (int32_t dim = static_cast<int32_t>(materialized.ndims) - 1; dim >= 0; --dim) {
            materialized.strides[dim] = stride;
            stride *= materialized.shapes[dim];
        }
        materialized.extent_elem_cache = stride;
        materialized.owner_task_id.raw = ctx.result.task_id().raw;
        Tensor::copy(slot_t, materialized);
        if (ci.has_initial_value) {
            fill_tensor_initial_value(slot_t, ci.initial_value);
        }
        ctx.result.materialize_output(slot_t);
#if PTO_FDWIC_SHARED_MAP
        Tensor::copy(shared_cell.tensors[output_ordinal], materialized);
#if defined(__CCE_AICORE__)
        dist_aicore_flush_region(&shared_cell.tensors[output_ordinal], sizeof(Tensor));
#endif
        atomic_fetch_max<int64_t>(
            shared_cell.last_writer[output_ordinal].v, static_cast<int64_t>(ctx.task_id), __ATOMIC_ACQ_REL
        );
        ctx.shared_result.add_output_ref(ctx.task_id, static_cast<int16_t>(output_ordinal));
#endif
        output_offset += PTO2_ALIGN_UP(buffer_size, PTO2_PACKED_OUTPUT_ALIGN);
    }
#if PTO_FDWIC_SHARED_MAP
    store_barrier();
    for (int32_t output_ordinal = 0; output_ordinal < layout.output_count; output_ordinal++)
        atomic_exchange(shared_cell.published[output_ordinal].v, static_cast<int64_t>(ctx.task_id), __ATOMIC_RELEASE);
#endif
#if !PTO_FDWIC_SHARED_MAP
    if (total > 0 && g_dist.heap_base != nullptr) ctx.self->heap_next = task_base + layout.total_output_size;
#endif
    ctx.output_bytes = total;
    return true;
}

#if PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC int32_t dist_shared_output_last_writer(FdwicOutputRef ref) {
    if (ref.producer_task_id < 0 || ref.producer_task_id >= kFlagCap) return -1;
    always_assert(ref.output_slot >= 0 && ref.output_slot < kSharedOutputMaxPerTask);
    __gm__ SharedOutputCell &cell = shared_output_cell(ref.producer_task_id);
    return static_cast<int32_t>(atomic_load(cell.last_writer[ref.output_slot].v, __ATOMIC_ACQUIRE));
}

PTO_DEVICE_FUNC int32_t dist_shared_output_claim_writer(FdwicOutputRef ref, int32_t writer_task_id) {
    if (ref.producer_task_id < 0 || ref.producer_task_id >= kFlagCap || writer_task_id < 0) {
        set_fatal();
        return -1;
    }
    always_assert(ref.output_slot >= 0 && ref.output_slot < kSharedOutputMaxPerTask);
    __gm__ SharedOutputCell &cell = shared_output_cell(ref.producer_task_id);
    return static_cast<int32_t>(
        atomic_exchange(cell.last_writer[ref.output_slot].v, static_cast<int64_t>(writer_task_id), __ATOMIC_ACQ_REL)
    );
}

PTO_DEVICE_FUNC bool
dist_resolve_shared_output_ref(FdwicOutputRef ref, __gm__ DistCore *self, __gm__ Tensor &resolved) {
    if (ref.producer_task_id < 0 || ref.producer_task_id >= kFlagCap) {
        set_fatal();
        return false;
    }
    always_assert(ref.output_slot >= 0 && ref.output_slot < kSharedOutputMaxPerTask);
    __gm__ SharedOutputCell &cell = shared_output_cell(ref.producer_task_id);
    TRACE_SPAN_BEGIN(resolve_wait_trace);
    while (!fatal_set() && atomic_load(cell.published[ref.output_slot].v, __ATOMIC_ACQUIRE) != ref.producer_task_id) {
        drain_block_won_if_enabled(self);
        if (drain_phase_b(self) == 0) SPIN_WAIT_HINT();
    }
    if (fatal_set()) return false;
    TRACE_SPAN_END(resolve_wait_trace, self, ref.producer_task_id, -1, TracePhase::ResolveWait, 0, ref.output_slot);
#if defined(__CCE_AICORE__)
    TRACE_SPAN_BEGIN(resolve_invalidate_trace);
    dist_aicore_invalidate_region(&cell.tensors[ref.output_slot], sizeof(Tensor));
    TRACE_SPAN_END(
        resolve_invalidate_trace, self, ref.producer_task_id, -1, TracePhase::ResolveInvalidate, 0, ref.output_slot
    );
#endif
    TRACE_SPAN_BEGIN(resolve_copy_trace);
    if ((ref.flags & 1u) != 0u) {
        always_assert(ref.view_ndims == 1 && cell.tensors[ref.output_slot].ndims == 1);
        const uint32_t view_shapes[1] = {ref.view_shape0};
        const uint32_t view_offsets[1] = {ref.view_offset0};
        Tensor::copy(resolved, Tensor::view(cell.tensors[ref.output_slot], view_shapes, view_offsets));
    } else {
        Tensor::copy(resolved, cell.tensors[ref.output_slot]);
    }
    TRACE_SPAN_END(
        resolve_copy_trace, self, ref.producer_task_id, -1, TracePhase::ResolveCopy, ref.flags & 1u, ref.output_slot
    );
    return true;
}

PTO_DEVICE_FUNC bool
dist_resolve_ready_shared_output_ref(FdwicOutputRef ref, __gm__ DistCore *self, __gm__ Tensor &resolved) {
    if (ref.producer_task_id < 0 || ref.producer_task_id >= kFlagCap) {
        set_fatal();
        return false;
    }
    always_assert(ref.output_slot >= 0 && ref.output_slot < kSharedOutputMaxPerTask);
    __gm__ SharedOutputCell &cell = shared_output_cell(ref.producer_task_id);
#if defined(__CCE_AICORE__)
    TRACE_SPAN_BEGIN(resolve_invalidate_trace);
    dist_aicore_invalidate_region(&cell.tensors[ref.output_slot], sizeof(Tensor));
    TRACE_SPAN_END(
        resolve_invalidate_trace, self, ref.producer_task_id, -1, TracePhase::ResolveInvalidate, 0, ref.output_slot
    );
#endif
    TRACE_SPAN_BEGIN(resolve_copy_trace);
    if ((ref.flags & 1u) != 0u) {
        always_assert(ref.view_ndims == 1 && cell.tensors[ref.output_slot].ndims == 1);
        const uint32_t view_shapes[1] = {ref.view_shape0};
        const uint32_t view_offsets[1] = {ref.view_offset0};
        Tensor::copy(resolved, Tensor::view(cell.tensors[ref.output_slot], view_shapes, view_offsets));
    } else {
        Tensor::copy(resolved, cell.tensors[ref.output_slot]);
    }
    TRACE_SPAN_END(
        resolve_copy_trace, self, ref.producer_task_id, -1, TracePhase::ResolveCopy, ref.flags & 1u, ref.output_slot
    );
    return true;
}

PTO_DEVICE_FUNC uint32_t dist_shared_region_hash(uint64_t addr) { return dist_tensor_map_hash(addr); }

template <typename TensorRef>
PTO_DEVICE_FUNC void dist_shared_region_byte_range(const TensorRef &t, uint64_t &addr, uint64_t &lo, uint64_t &hi) {
    dist_tensor_map_byte_range(t, addr, lo, hi);
}

PTO_DEVICE_FUNC int32_t dist_shared_region_lookup(const Tensor &tensor, int32_t before_task_id) {
    if (tensor.manual_dep) return -1;
    uint64_t addr, lo, hi;
    dist_shared_region_byte_range(tensor, addr, lo, hi);
    int32_t best = -1;
    const uint32_t bucket = dist_shared_region_hash(addr);
    int32_t cur = static_cast<int32_t>(atomic_load(g_dist.shared_region.buckets[bucket].v, __ATOMIC_ACQUIRE));
    while (cur == -2 && !fatal_set()) {
        SPIN_WAIT_HINT();
        cur = static_cast<int32_t>(atomic_load(g_dist.shared_region.buckets[bucket].v, __ATOMIC_ACQUIRE));
    }
    for (; cur >= 0;) {
        if (cur >= kSharedRegionCap) {
            set_fatal();
            return best;
        }
        __gm__ SharedRegionEntry &entry = g_dist.shared_region.entries[cur];
#if defined(__CCE_AICORE__)
        dist_aicore_invalidate_region(&entry, sizeof(SharedRegionEntry));
#endif
        if (entry.producer < before_task_id && entry.buf_addr == addr && lo < entry.hi && entry.lo < hi &&
            entry.producer > best) {
            best = entry.producer;
        }
        cur = entry.next_in_bucket;
    }
    return best;
}

PTO_DEVICE_FUNC void dist_shared_region_lock() {
    while (atomic_exchange(g_dist.shared_region.insert_lock.v, int64_t{1}, __ATOMIC_ACQUIRE) != 0) {
        SPIN_WAIT_HINT();
    }
}

PTO_DEVICE_FUNC void dist_shared_region_unlock() {
    store_barrier();
    atomic_exchange(g_dist.shared_region.insert_lock.v, int64_t{0}, __ATOMIC_RELEASE);
}

PTO_DEVICE_FUNC void dist_shared_region_insert(const Tensor &tensor, int32_t producer) {
    if (tensor.manual_dep || producer < 0) return;
    const int32_t slot = static_cast<int32_t>(atomic_fetch_add<int64_t>(g_dist.shared_region.high_water.v, int64_t{1}));
    if (slot < 0 || slot >= kSharedRegionCap) {
        set_fatal();
        DIST_ERRF("[dist_engine] shared region map capacity exceeded at task %d\n", producer);
        return;
    }
    uint64_t addr, lo, hi;
    dist_shared_region_byte_range(tensor, addr, lo, hi);
    const uint32_t bucket = dist_shared_region_hash(addr);
    int64_t old_head = 0;
    while (true) {
        old_head = atomic_exchange(g_dist.shared_region.buckets[bucket].v, int64_t{-2}, __ATOMIC_ACQUIRE);
        if (old_head != -2) break;
        SPIN_WAIT_HINT();
    }
    __gm__ SharedRegionEntry &entry = g_dist.shared_region.entries[slot];
    entry.buf_addr = addr;
    entry.lo = lo;
    entry.hi = hi;
    entry.producer = producer;
    entry.next_in_bucket = static_cast<int32_t>(old_head);
    store_barrier();
#if defined(__CCE_AICORE__)
    dist_aicore_flush_region(&entry, sizeof(SharedRegionEntry));
#endif
    atomic_exchange(g_dist.shared_region.buckets[bucket].v, static_cast<int64_t>(slot), __ATOMIC_RELEASE);
}

PTO_DEVICE_FUNC bool dist_shared_region_collect_relevant(TensorArgType tag, const Tensor &tensor) {
    if (tensor.manual_dep) return false;
    if (tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING) return true;
    if (tag == TensorArgType::INPUT && tensor.owner_task_id.raw != UINT64_MAX) return true;
    return false;
}

PTO_DEVICE_FUNC bool dist_shared_region_register_relevant(TensorArgType tag, const Tensor &tensor) {
    if (tensor.manual_dep) return false;
    return tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING;
}

PTO_DEVICE_FUNC void dist_submit_arg_tensor_local(const L0TaskArgs &args, int32_t i, Tensor &tensor) {
#if defined(__CCE_AICORE__)
    if (args.tensor(i).tensor_from_gm()) {
        Tensor::copy(tensor, args.tensor(i).gm_ref());
    } else
#endif
    {
        Tensor::copy(tensor, args.tensor(i).ref());
    }
}

#endif

PTO_DEVICE_FUNC inline void
dist_submit_copy_arg_tensor(__gm__ Tensor &dst, const L0TaskArgs &args, const DistSubmitCtx &ctx, int32_t i) {
    if (args.tag(i) == TensorArgType::OUTPUT) {
        Tensor::copy(dst, ctx.payload->tensors[i]);
        return;
    }
#if PTO_FDWIC_SHARED_MAP
    if (args.tensor(i).tensor_from_shared_output()) return;
#endif
#if defined(__CCE_AICORE__)
    if (args.tensor(i).tensor_from_gm()) {
        Tensor::copy(dst, args.tensor(i).gm_ref());
    } else {
        Tensor::copy(dst, args.tensor(i).ref());
    }
#else
    Tensor::copy(dst, args.tensor(i).ref());
#endif
}

template <typename FaninArrPtr>
PTO_DEVICE_FUNC void build_ring_slot_from_submit(
    __gm__ RingSlot &s, int32_t task_id, int32_t func_id, uint64_t fn_addr, const L0TaskArgs &args,
    const DistSubmitCtx &ctx, FaninArrPtr fanin, int32_t fc, int32_t sub_block_id, bool is_multicore, int32_t won_block,
    int32_t won_slot
) {
    s.occupied = true;
    s.task_id = task_id;
    s.func_id = func_id;
    s.function_bin_addr = fn_addr;
    s.built = false;
    s.tensor_count = ctx.tensor_count;
    s.scalar_count = ctx.scalar_count;
#if PTO_FDWIC_SHARED_MAP
    s.shared_ref_mask = 0;
#endif
    for (int32_t i = 0; i < ctx.tensor_count; i++)
        dist_submit_copy_arg_tensor(s.tensors[i], args, ctx, i);
#if PTO_FDWIC_SHARED_MAP
    for (int32_t i = 0; i < ctx.tensor_count; i++) {
        if (!args.tensor(i).tensor_from_shared_output()) continue;
        const FdwicOutputRef ref = args.tensor(i).shared_output_ref();
        if (dist_try_resolve_shared_output_ref(ref, ctx.self, s.tensors[i])) continue;
        s.shared_ref_mask |= 1u << static_cast<uint32_t>(i);
        dist_shared_ref_copy(s.shared_refs[i], ref);
    }
#endif
    for (int32_t j = 0; j < ctx.scalar_count; j++)
        s.scalars[j] = args.scalar(j);
    int32_t n = 0;
    for (int32_t i = 0; i < ctx.tensor_count; i++)
        s.args[n++] = reinterpret_cast<uint64_t>(&s.tensors[i]);
    for (int32_t j = 0; j < ctx.scalar_count; j++)
        s.args[n++] = s.scalars[j];
    s.local_ctx.s_block_idx = 0;
    s.local_ctx.s_block_num = 1;
    s.local_ctx.async_ctx.completion_count = nullptr;
    s.local_ctx.async_ctx.completion_error_code = nullptr;
    s.local_ctx.async_ctx.completion_entries = nullptr;
    s.local_ctx.async_ctx.completion_capacity = 0;
    s.local_ctx.async_ctx.task_token.raw = UINT64_MAX;
    s.global_ctx.sub_block_id = sub_block_id;
    s.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&s.local_ctx);
    s.args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&s.global_ctx);
    s.fanin_count = fc;
    for (int32_t k = 0; k < fc; k++)
        s.fanin[k] = fanin[k];
    s.is_multicore = is_multicore;
    s.won_block = won_block;
    s.won_slot = won_slot;
    flush_ring_slot_payload(s, ctx.tensor_count, ctx.scalar_count, n);
    publish_ring_slot_built(s);
}

PTO_DEVICE_FUNC void dist_submit_add_fanin(int32_t fanin[], int32_t &fanin_count, int32_t producer) {
    if (producer < 0) return;
    for (int32_t k = 0; k < fanin_count; k++)
        if (fanin[k] == producer) return;
    if (fanin_count < kMaxFanin) fanin[fanin_count++] = producer;
}

#if !PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC void dist_submit_prepare_map(__gm__ DistCore *self, int32_t task_id) {
    dist_tensor_map_advance_retire(self->map, task_id, g_dist.H);
}

PTO_DEVICE_FUNC int32_t dist_submit_collect_fanin(const L0TaskArgs &args, const DistSubmitCtx &ctx, int32_t fanin[]) {
    int32_t fc = 0;
    for (int32_t i = 0; i < ctx.tensor_count; i++) {
        const TensorArgType tag = args.tag(i);
        if (tag == TensorArgType::OUTPUT) continue;
#if defined(__CCE_AICORE__)
        if (args.tensor(i).tensor_from_gm()) {
            const uint64_t owner_raw = args.tensor(i).gm_ref().owner_task_id.raw;
            if (owner_raw != UINT64_MAX) {
                dist_submit_add_fanin(fanin, fc, static_cast<int32_t>(owner_raw & 0xFFFFFFFFu));
            }
            if (tag != TensorArgType::INPUT && tag != TensorArgType::INOUT) continue;
            const int32_t p = dist_tensor_map_lookup(ctx.self->map, args.tensor(i).gm_ref());
            dist_submit_add_fanin(fanin, fc, p);
        } else {
            const Tensor &t = args.tensor(i).ref();
            const uint64_t owner_raw = t.owner_task_id.raw;
            if (owner_raw != UINT64_MAX) {
                dist_submit_add_fanin(fanin, fc, static_cast<int32_t>(owner_raw & 0xFFFFFFFFu));
            }
            if (tag != TensorArgType::INPUT && tag != TensorArgType::INOUT) continue;
            const int32_t p = dist_tensor_map_lookup(ctx.self->map, t);
            dist_submit_add_fanin(fanin, fc, p);
        }
#else
        const Tensor &t = args.tensor(i).ref();
        const uint64_t owner_raw = t.owner_task_id.raw;
        if (owner_raw != UINT64_MAX) dist_submit_add_fanin(fanin, fc, static_cast<int32_t>(owner_raw & 0xFFFFFFFFu));
        if (tag != TensorArgType::INPUT && tag != TensorArgType::INOUT) continue;
        if (t.manual_dep) continue;
        const int32_t p = dist_tensor_map_lookup(ctx.self->map, t);
        dist_submit_add_fanin(fanin, fc, p);
#endif
    }
    return fc;
}

PTO_DEVICE_FUNC void dist_submit_insert_tensor(DistSubmitCtx &ctx, const L0TaskArgs &args, int32_t i) {
    if (args.tag(i) == TensorArgType::OUTPUT) {
        dist_tensor_map_insert(ctx.self->map, ctx.payload->tensors[i], ctx.task_id);
        return;
    }
#if defined(__CCE_AICORE__)
    if (args.tensor(i).tensor_from_gm()) {
        dist_tensor_map_insert(ctx.self->map, args.tensor(i).gm_ref(), ctx.task_id);
    } else {
        dist_tensor_map_insert(ctx.self->map, args.tensor(i).ref(), ctx.task_id);
    }
#else
    dist_tensor_map_insert(ctx.self->map, args.tensor(i).ref(), ctx.task_id);
#endif
}

PTO_DEVICE_FUNC void dist_submit_register_outputs(DistSubmitCtx &ctx, const L0TaskArgs &args, bool include_existing) {
    for (int32_t i = 0; i < ctx.tensor_count; i++) {
        const TensorArgType tag = args.tag(i);
        const bool registers_producer =
            include_existing && (tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING);
        if (registers_producer) dist_submit_insert_tensor(ctx, args, i);
    }
}

PTO_DEVICE_FUNC bool dist_submit_materialize_and_prepare_map(
    __gm__ DistCore *self, const L0TaskArgs &args, DistSubmitCtx &ctx, DistSubmitKind kind
) {
    TRACE_LAP_RESET(self);
    if (!dist_submit_check_task_cap(ctx, kind)) return false;
    TRACE_SPAN_BEGIN(materialize_trace);
    if (!dist_submit_materialize_args(args, ctx, kind)) return false;
    TRACE_SPAN_END(materialize_trace, self, ctx.task_id, -1, TracePhase::Materialize, 0, static_cast<uint32_t>(kind));
#if !defined(__CCE_AICORE__)
    if (fatal_set()) return false;
#endif
    TRACE_SPAN_BEGIN(prepare_map_trace);
    dist_submit_prepare_map(self, ctx.task_id);
    TRACE_SPAN_END(prepare_map_trace, self, ctx.task_id, -1, TracePhase::PrepareMap, 0, static_cast<uint32_t>(kind));
    return true;
}
#endif

}  // namespace
