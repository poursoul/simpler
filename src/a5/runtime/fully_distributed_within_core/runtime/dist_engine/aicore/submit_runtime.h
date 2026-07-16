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

namespace {

PTO_DEVICE_FUNC int32_t anchor_lane_for_mask(const ActiveMask &M) {
    if (lane_active(M, LANE_AIC)) return LANE_AIC;
    if (lane_active(M, LANE_AIV0)) return LANE_AIV0;
    if (lane_active(M, LANE_AIV1)) return LANE_AIV1;
    return LANE_NONE;
}

PTO_DEVICE_FUNC bool dist_submit_self_is_lane(__gm__ DistCore *self, int32_t block, int32_t lane) {
    if (self == nullptr || self->block_id != block || self->lane != lane) return false;
    if (lane == LANE_AIC) return self->role == CoreType::AIC;
    if (lane == LANE_AIV0 || lane == LANE_AIV1) return self->role == CoreType::AIV;
    return false;
}

PTO_DEVICE_FUNC bool dist_submit_claim_kernel(const MixedKernels &mixed, DistSubmitCtx &ctx) {
    ctx.kernel_id = INVALID_KERNEL_ID;
    ctx.won = false;
    if (ctx.self == nullptr || ctx.task_id < 0 || ctx.task_id >= kFlagCap) return false;
    const ActiveMask M = mixed.to_active_mask();
    const uint8_t cmask = M.core_mask();
    const int32_t pc = __builtin_popcount(cmask);
    if (pc >= 2) {
        const int32_t block = ctx.self->block_id;
        if (block < 0 || block >= g_dist.num_blocks) return false;
        ctx.joint = true;
        ctx.joint_block = block;
        ctx.joint_slot = -1;
        ctx.joint_count = pc;
        const int32_t anchor_lane = anchor_lane_for_mask(M);
        if (!dist_submit_self_is_lane(ctx.self, block, anchor_lane)) return false;
        __gm__ PaddedCursor *cursors = anchor_lane == LANE_AIC ? g_dist.cube_cursor : g_dist.vector_cursor;
        ctx.won = claim(cursors[ctx.task_id % kCursorShards].v, ctx.task_id);
        if (!ctx.won) return false;
        ctx.kernel_id = kernel_id_for_lane(mixed, anchor_lane);
        ctx.joint_init = true;
        return true;
    }
    if (lane_active(M, LANE_AIC)) {
        if (ctx.self->role != CoreType::AIC) return false;
        ctx.won = claim(g_dist.cube_cursor[ctx.task_id % kCursorShards].v, ctx.task_id);
        if (!ctx.won) return false;
        ctx.kernel_id = mixed.aic_kernel_id;
        return true;
    }
    if (lane_active(M, LANE_AIV0) || lane_active(M, LANE_AIV1)) {
        if (ctx.self->role != CoreType::AIV) return false;
        ctx.won = claim(g_dist.vector_cursor[ctx.task_id % kCursorShards].v, ctx.task_id);
        if (!ctx.won) return false;
        const int32_t own_lane = lane_active(M, LANE_AIV0) ? LANE_AIV0 : LANE_AIV1;
        ctx.kernel_id = kernel_id_for_lane(mixed, own_lane);
        return true;
    }
    return false;
}

PTO_DEVICE_FUNC bool dist_submit_claim_alloc(DistSubmitCtx &ctx) {
    ctx.kernel_id = INVALID_KERNEL_ID;
    if (ctx.self == nullptr || ctx.task_id < 0 || ctx.task_id >= kFlagCap) return false;
    ctx.won = claim(g_dist.alloc_cursor[ctx.task_id % kCursorShards].v, ctx.task_id);
    return ctx.won;
}

PTO_DEVICE_FUNC bool dist_submit_claim(DistSubmitKind kind, const MixedKernels *mixed, DistSubmitCtx &ctx) {
    if (kind == DistSubmitKind::Alloc) return dist_submit_claim_alloc(ctx);
    if (mixed == nullptr) return false;
    return dist_submit_claim_kernel(*mixed, ctx);
}

PTO_DEVICE_FUNC __gm__ RingSlot *dist_submit_alloc_slot(__gm__ DistCore *self) {
    if (self == nullptr) return nullptr;
    const int32_t si = alloc_ring_slot(self);
    if (si < 0) return nullptr;
    __gm__ RingSlot &slot = self->slots[si];
    slot.occupied = true;
    slot.built = false;
    self->occupied_count++;
    return &slot;
}

PTO_DEVICE_FUNC void dist_submit_wait_slot_capacity(__gm__ DistCore *self, int32_t task_id) {
    if (self == nullptr) return;
    bool waited = false;
    TRACE_SPAN_BEGIN(ring_bp_trace);
    while (self->occupied_count >= kPrivateSlots - kWonReserve) {
        waited = true;
        drain_block_won(self);
        if (drain_phase_b(self) == 0) SPIN_WAIT_HINT();
    }
    if (waited) {
        TRACE_SPAN_END(ring_bp_trace, self, task_id, -1, TracePhase::RingBp, 0, 0);
    }
}

#if !PTO_FDWIC_SHARED_TENSORMAP
PTO_DEVICE_FUNC bool dist_submit_wait_heap_capacity(DistSubmitCtx &ctx, DistSubmitKind kind) {
    if (ctx.self == nullptr || ctx.output_bytes == 0 || g_dist.heap_base == nullptr) return true;
    const size_t ring = g_dist.heap_size;
    bool waited = false;
    TRACE_SPAN_BEGIN(heap_bp_trace);
    while (!fatal_set()) {
        const int32_t f = static_cast<int32_t>(atomic_load(g_dist.frontier));
        const int32_t R = f - g_dist.H;
        const uint64_t vstart_live = load_task_vend(R);
        if (ctx.self->heap_next - vstart_live <= ring) {
            if (waited) {
                TRACE_SPAN_END(heap_bp_trace, ctx.self, ctx.task_id, -1, TracePhase::RingBp, 0, 1);
            }
            return true;
        }
        if (f >= ctx.task_id - 1) {
            set_fatal();
            if (kind == DistSubmitKind::Alloc) {
                DIST_ERRF(
                    "[dist_engine] heap ring %zu B too small for H=%d window at alloc %d (live=%llu B)\n", ring,
                    g_dist.H, ctx.task_id, (unsigned long long)(ctx.self->heap_next - vstart_live)
                );
            } else {
                DIST_ERRF(
                    "[dist_engine] heap ring %zu B too small for H=%d window at task %d (live=%llu B); "
                    "enlarge PTO_DIST_HEAP_MB or reduce PTO_DIST_H\n",
                    ring, g_dist.H, ctx.task_id, (unsigned long long)(ctx.self->heap_next - vstart_live)
                );
            }
            return false;
        }
        waited = true;
        drain_block_won(ctx.self);
        if (drain_phase_b(ctx.self) == 0) SPIN_WAIT_HINT();
    }
    if (waited) {
        TRACE_SPAN_END(heap_bp_trace, ctx.self, ctx.task_id, -1, TracePhase::RingBp, 0, 1);
    }
    return false;
}
#endif

PTO_DEVICE_FUNC void publish_joint_deposits(DistSubmitCtx &ctx, const MixedKernels &mixed, const L0TaskArgs &args) {
    if (!ctx.joint) return;
    __gm__ WonSlot &w = g_dist.blocks[ctx.joint_block].slots[ctx.joint_slot];
    const ActiveMask M = mixed.to_active_mask();
    populate_won_slot_from_submit(
        w, ctx.task_id, M, mixed, ctx.self->lane,
#if defined(__CCE_AICORE__)
        nullptr,
#else
        g_dist.runtime,
#endif
        args, ctx, ctx.fanin, ctx.fanin_count
    );
#if defined(__CCE_AICORE__)
    dist_aicore_flush_region(&w.meta, sizeof(w.meta));
    dist_aicore_flush_region(w.lane, sizeof(w.lane));
#endif
    store_won_remaining(w, ctx.joint_count);
    publish_won_slot(w);
    atomic_exchange(g_dist.blocks[ctx.joint_block].any_pub, 1);
}

PTO_DEVICE_FUNC int32_t wait_alloc_won_slot(__gm__ DistCore *self, int32_t block) {
    int32_t won_slot = alloc_won_slot(block);
    while (won_slot < 0 && !fatal_set()) {
        drain_block_won(self);
        if (drain_phase_b(self) == 0) SPIN_WAIT_HINT();
        won_slot = alloc_won_slot(block);
    }
    return won_slot;
}

PTO_DEVICE_FUNC bool dist_submit_build_winner_slot(DistSubmitCtx &ctx, const L0TaskArgs &args, __gm__ RingSlot *slot) {
    if (slot == nullptr || ctx.payload == nullptr) return false;
    const int32_t sub_block_id = ctx.self != nullptr && ctx.self->lane == LANE_AIV1 ? 1 : 0;
    const uint64_t fn_addr = dist_aicore_slot_function_addr(g_dist.runtime, ctx.kernel_id);
#if PTO_FDWIC_SHARED_TENSORMAP && !defined(__CCE_AICORE__)
    const bool has_sim_exec_time = g_dist.runtime != nullptr && g_dist.runtime->use_example_exec_time_ &&
                                   ctx.kernel_id >= 0 && ctx.kernel_id < RUNTIME_MAX_FUNC_ID &&
                                   g_dist.runtime->example_exec_time_ns_[ctx.kernel_id] > 0;
    if (fn_addr == 0 && !g_skip_exec && !has_sim_exec_time) {
        set_fatal();
        DIST_ERRF("[dist_engine] shared task %d kernel %d has no function address\n", ctx.task_id, ctx.kernel_id);
        return false;
    }
#endif
    build_ring_slot_from_submit(
        *slot, ctx.task_id, ctx.kernel_id, fn_addr, args, ctx, ctx.fanin, ctx.fanin_count, sub_block_id, ctx.joint,
        ctx.joint_block, ctx.joint_slot
    );
    return true;
}

PTO_DEVICE_FUNC void
dist_submit_build_winner_task(DistSubmitCtx &ctx, const MixedKernels &mixed, const L0TaskArgs &args) {
    if (ctx.self == nullptr) return;
    dist_submit_wait_slot_capacity(ctx.self, ctx.task_id);
#if !PTO_FDWIC_SHARED_TENSORMAP
    if (!dist_submit_wait_heap_capacity(ctx, DistSubmitKind::Kernel)) return;
#endif
    if (ctx.joint && ctx.joint_slot < 0) {
        ctx.joint_slot = wait_alloc_won_slot(ctx.self, ctx.joint_block);
        if (ctx.joint_slot < 0) return;
    }
    __gm__ RingSlot *slot = dist_submit_alloc_slot(ctx.self);
    if (slot == nullptr) return;

    if (ctx.joint) publish_joint_deposits(ctx, mixed, args);
    if (!dist_submit_build_winner_slot(ctx, args, slot)) return;
}

#if !PTO_FDWIC_SHARED_TENSORMAP
PTO_DEVICE_FUNC void dist_submit_complete_alloc(DistSubmitCtx &ctx) {
    if (ctx.won) {
        if (!dist_submit_wait_heap_capacity(ctx, DistSubmitKind::Alloc)) return;
        if (ctx.self != nullptr) complete_executed_task(ctx.self, ctx.task_id);
    }
}
#endif

#include "dist_engine/aicore/run_state.h"

PTO_DEVICE_FUNC void dist_submit_drain_to_completion(__gm__ DistCore *self) {
    if (self == nullptr) return;
    atomic_fetch_add<int64_t>(g_dist.replay_done, 1);
    while (true) {
        drain_block_won(self);
        const int32_t freed = drain_phase_b(self);
        const bool all_replayed = atomic_load(g_dist.replay_done) >= g_dist.num_workers;
        const bool ring_empty = self->occupied_count == 0;
        const bool pending = has_pending_won(self);
        if (all_replayed && ring_empty && !pending) break;
        if (freed == 0) SPIN_WAIT_HINT();
    }
}

PTO_DEVICE_FUNC void dist_submit_replay_orch(__gm__ Runtime *runtime) {
    __gm__ DistCore *self = g_self;
    (void)self;
#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(runtime->dist.ccec_orch_tensors, sizeof(runtime->dist.ccec_orch_tensors));
    dist_aicore_invalidate_region(runtime->dist.ccec_orch_scalars, sizeof(runtime->dist.ccec_orch_scalars));
    dist_aicore_invalidate_region(const_cast<__gm__ const int32_t *>(&runtime->dist.ccec_orch_tensor_count), 64);
    if (aicpu_orchestration_entry == nullptr || !ccec_is_valid_worker()) {
        return;
    }
    L2TaskArgs local_args;
    Tensor local_tensors[CHIP_MAX_TENSOR_ARGS];
    const int32_t tensor_count = runtime->dist.ccec_orch_tensor_count;
    const int32_t scalar_count = runtime->dist.ccec_orch_scalar_count;
    for (int32_t i = 0; i < tensor_count && i < CHIP_MAX_TENSOR_ARGS; i++) {
        Tensor::copy(local_tensors[i], runtime->dist.ccec_orch_tensors[i]);
        local_args.add_input(local_tensors[i]);
    }
    for (int32_t i = 0; i < scalar_count && i < CHIP_MAX_SCALAR_ARGS; i++) {
        const uint64_t scalar = runtime->dist.ccec_orch_scalars[i];
        local_args.add_scalar(scalar);
    }
    aicpu_orchestration_entry(local_args);
#else
    (void)runtime;
    if (g_dist.orch_args != nullptr && !fatal_set()) {
        aicpu_orchestration_entry(*g_dist.orch_args);
    }
#endif
}

#include "dist_engine/aicore/onboard_entry.h"

}  // namespace

#if PTO_FDWIC_SHARED_TENSORMAP
PTO_DEVICE_FUNC uint32_t dist_submit_count_fresh_outputs(const L0TaskArgs &args) {
    uint32_t count = 0;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) == TensorArgType::OUTPUT) count++;
    }
    return count;
}

PTO_DEVICE_FUNC bool dist_submit_shared_materialize_outputs(L0TaskArgs &args, DistSubmitCtx &ctx) {
    if (ctx.payload == nullptr) return false;
    ctx.tensor_count = args.tensor_count();
    ctx.scalar_count = args.scalar_count();

    DistOutputLayout layout;
    layout.total_output_size = 0;
    layout.output_count = 0;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        const int32_t output_ordinal = layout.output_count++;
        layout.output_indices[output_ordinal] = i;
        layout.buffer_sizes[output_ordinal] = TensorCreateInfo::buffer_size_bytes(args.tensor(i).create_info());
        layout.total_output_size += PTO2_ALIGN_UP(layout.buffer_sizes[output_ordinal], PTO2_PACKED_OUTPUT_ALIGN);
    }

    if (layout.total_output_size == 0) {
        ctx.output_bytes = 0;
        return true;
    }
    if (g_dist.heap_base == nullptr) {
        set_fatal();
        DIST_ERRF("[dist_engine] shared GM output heap not allocated at task %d\n", ctx.task_id);
        return false;
    }
    if (layout.total_output_size > g_dist.heap_size) {
        set_fatal();
        DIST_ERRF(
            "[dist_engine] shared task %d outputs %llu B exceed heap %zu B\n", ctx.task_id,
            (unsigned long long)layout.total_output_size, g_dist.heap_size
        );
        return false;
    }

    const uint64_t task_base = static_cast<uint64_t>(atomic_fetch_add<int64_t>(
        g_dist.shared_heap_top.v, static_cast<int64_t>(layout.total_output_size), __ATOMIC_ACQ_REL
    ));
    if (task_base + layout.total_output_size > g_dist.heap_size) {
        set_fatal();
        DIST_ERRF(
            "[dist_engine] shared heap exhausted at task %d (need=%llu B, used=%llu B, cap=%zu B)\n", ctx.task_id,
            (unsigned long long)layout.total_output_size, (unsigned long long)task_base, g_dist.heap_size
        );
        return false;
    }

    uint64_t output_offset = 0;
    for (int32_t output_ordinal = 0; output_ordinal < layout.output_count; output_ordinal++) {
        const int32_t i = layout.output_indices[output_ordinal];
        const auto &ci = args.tensor(i).create_info();
        const uint64_t buffer_size = layout.buffer_sizes[output_ordinal];
        __gm__ Tensor &slot_t = ctx.payload->tensors[i];
        init_tensor_from_create_info(slot_t, ci, g_dist.heap_base + task_base + output_offset, buffer_size);
        slot_t.owner_task_id.raw = ctx.result.task_id().raw;
        output_offset += PTO2_ALIGN_UP(buffer_size, PTO2_PACKED_OUTPUT_ALIGN);
    }
    ctx.output_bytes = layout.total_output_size;
    return true;
}

PTO_DEVICE_FUNC void dist_submit_shared_publish_producers(const L0TaskArgs &args, DistSubmitCtx &ctx) {
    uint32_t output_slot = 0;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        const TensorArgType tag = args.tag(i);
        if (tag == TensorArgType::OUTPUT) {
            shared_map_insert_symbol(g_dist.shared_map, ctx.task_id, output_slot, ctx.payload->tensors[i]);
            output_slot++;
            continue;
        }
        if (tag != TensorArgType::INOUT && tag != TensorArgType::OUTPUT_EXISTING) continue;
#if defined(__CCE_AICORE__)
        if (args.tensor(i).tensor_from_gm()) {
            shared_map_insert_range(g_dist.shared_map, ctx.task_id, args.tensor(i).gm_ref());
        } else {
            shared_map_insert_range(g_dist.shared_map, ctx.task_id, args.tensor(i).ref());
        }
#else
        shared_map_insert_range(g_dist.shared_map, ctx.task_id, args.tensor(i).ref());
#endif
    }
}

PTO_DEVICE_FUNC void dist_submit_shared_add_fanin(DistSubmitCtx &ctx, int32_t producer) {
    if (producer < 0 || producer >= ctx.task_id) return;
    for (int32_t k = 0; k < ctx.fanin_count; k++)
        if (ctx.fanin[k] == producer) return;
    if (ctx.fanin_count < kMaxFanin) ctx.fanin[ctx.fanin_count++] = producer;
}

template <typename TensorRef>
PTO_DEVICE_FUNC void
dist_submit_shared_collect_tensor_fanin(DistSubmitCtx &ctx, const TensorRef &tensor, TensorArgType tag) {
    const uint64_t owner_raw = tensor.owner_task_id.raw;
    if (owner_raw != UINT64_MAX) dist_submit_shared_add_fanin(ctx, static_cast<int32_t>(owner_raw & 0xFFFFFFFFu));
    if (tag != TensorArgType::INPUT && tag != TensorArgType::INOUT) return;
    if (tensor.manual_dep) return;
    dist_submit_shared_add_fanin(ctx, shared_map_lookup_range(g_dist.shared_map, tensor, ctx.task_id));
}

PTO_DEVICE_FUNC bool dist_submit_shared_resolve_inputs_and_fanin(L0TaskArgs &args, DistSubmitCtx &ctx) {
    if (ctx.payload == nullptr) return false;
    bool waited = false;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        const TensorArgType tag = args.tag(i);
        if (tag == TensorArgType::OUTPUT) continue;
        if (args.tensor(i).tensor_is_symbolic()) {
            if (!waited) {
                shared_wait_published_before(ctx.self, ctx.task_id);
                waited = true;
            }
            const SymbolicTensor sym = args.tensor(i).symbol();
            __gm__ const SharedMapEntry *entry = nullptr;
            if (!shared_map_lookup_symbol(
                    g_dist.shared_map, static_cast<int32_t>(sym.producer_task_id.raw & 0xFFFFFFFFu), sym.output_slot,
                    entry
                )) {
                set_fatal();
                return false;
            }
            const int32_t producer = static_cast<int32_t>(sym.producer_task_id.raw & 0xFFFFFFFFu);
            dist_submit_shared_add_fanin(ctx, producer);
            Tensor::copy(ctx.payload->tensors[i], entry->tensor);
            args.resolve_symbolic_tensor(i, &ctx.payload->tensors[i]);
            dist_submit_shared_collect_tensor_fanin(ctx, ctx.payload->tensors[i], tag);
            continue;
        }
        if (!waited && (tag == TensorArgType::INPUT || tag == TensorArgType::INOUT)) {
            shared_wait_published_before(ctx.self, ctx.task_id);
            waited = true;
        }
#if defined(__CCE_AICORE__)
        if (args.tensor(i).tensor_from_gm()) {
            dist_submit_shared_collect_tensor_fanin(ctx, args.tensor(i).gm_ref(), tag);
        } else {
            dist_submit_shared_collect_tensor_fanin(ctx, args.tensor(i).ref(), tag);
        }
#else
        dist_submit_shared_collect_tensor_fanin(ctx, args.tensor(i).ref(), tag);
#endif
    }
    return !fatal_set();
}

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors dist_submit_builder_impl(
    PTO2Runtime *, const MixedKernels &mixed, uint32_t output_count, DistSubmitBuildFn build_fn, void *build_ctx
) {
    L0TaskArgs empty_args;
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, empty_args, ctx);
    ctx.result.set_symbolic_output_count(output_count);
    TRACE_SPAN_BEGIN(submit_trace);
    TRACE_LAP_RESET(ctx.self);
    drain_block_won(ctx.self);
    drain_phase_b(ctx.self);
    TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::EfDrain);

    TRACE_SPAN_BEGIN(claim_trace);
    const bool is_winner = dist_submit_claim(DistSubmitKind::Kernel, &mixed, ctx);
    TRACE_SPAN_END(
        claim_trace, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Claim, static_cast<uint32_t>(is_winner), 0
    );

    if (is_winner) {
        atomic_fetch_add<int64_t>(g_dist.shared_winner_count.v, 1);
        L0TaskArgs args;
        atomic_fetch_add<int64_t>(g_dist.shared_builder_count.v, 1);
        if (build_fn != nullptr) build_fn(&args, build_ctx);
        const uint32_t actual_outputs = dist_submit_count_fresh_outputs(args);
        if (args.has_error || actual_outputs != output_count) {
            set_fatal();
        } else if (!dist_submit_shared_materialize_outputs(args, ctx)) {
            set_fatal();
        } else {
            dist_submit_shared_publish_producers(args, ctx);
            shared_publish_done(ctx.task_id);
            if (!dist_submit_shared_resolve_inputs_and_fanin(args, ctx)) {
                set_fatal();
            } else {
                TRACE_LAP(ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Build);
                dist_submit_build_winner_task(ctx, mixed, args);
            }
        }
    } else {
        atomic_fetch_add<int64_t>(g_dist.shared_loser_count.v, 1);
        TRACE_LAP(ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Replay);
        drain_block_won(ctx.self);
    }
    TRACE_SPAN_END(
        submit_trace, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Submit, static_cast<uint32_t>(is_winner), 0
    );
    return ctx.result;
}

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors
dist_submit_impl(PTO2Runtime *, const MixedKernels &mixed, const L0TaskArgs &args) {
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, args, ctx);
    ctx.result.set_symbolic_output_count(dist_submit_count_fresh_outputs(args));
    TRACE_SPAN_BEGIN(submit_trace);
    TRACE_LAP_RESET(ctx.self);
    drain_block_won(ctx.self);
    drain_phase_b(ctx.self);
    TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::EfDrain);

    TRACE_SPAN_BEGIN(claim_trace);
    const bool is_winner = dist_submit_claim(DistSubmitKind::Kernel, &mixed, ctx);
    TRACE_SPAN_END(
        claim_trace, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Claim, static_cast<uint32_t>(is_winner), 0
    );

    if (is_winner) {
        set_fatal();
    } else {
        TRACE_LAP(ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Replay);
        drain_block_won(ctx.self);
    }
    TRACE_SPAN_END(
        submit_trace, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Submit, static_cast<uint32_t>(is_winner), 0
    );
    return ctx.result;
}

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors dist_alloc_tensors(PTO2Runtime *, const L0TaskArgs &args) {
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, args, ctx);
    ctx.result.set_symbolic_output_count(dist_submit_count_fresh_outputs(args));
    TRACE_SPAN_BEGIN(submit_trace);
    TRACE_LAP_RESET(ctx.self);
    drain_block_won(ctx.self);
    drain_phase_b(ctx.self);
    TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::EfDrain);

    TRACE_SPAN_BEGIN(claim_trace);
    const bool is_winner = dist_submit_claim(DistSubmitKind::Alloc, nullptr, ctx);
    TRACE_SPAN_END(claim_trace, ctx.self, ctx.task_id, -1, TracePhase::Claim, static_cast<uint32_t>(is_winner), 1);
    if (is_winner) set_fatal();
    TRACE_SPAN_END(submit_trace, ctx.self, ctx.task_id, -1, TracePhase::Submit, static_cast<uint32_t>(is_winner), 1);
    return ctx.result;
}
#else
DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors
dist_submit_impl(PTO2Runtime *, const MixedKernels &mixed, const L0TaskArgs &args) {
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, args, ctx);
    TRACE_SPAN_BEGIN(submit_trace);
    TRACE_LAP_RESET(ctx.self);
    drain_block_won(ctx.self);
    drain_phase_b(ctx.self);
    TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::EfDrain);
    if (!dist_submit_materialize_and_prepare_map(ctx.self, args, ctx, DistSubmitKind::Kernel)) return ctx.result;
    TRACE_SPAN_BEGIN(claim_trace);
    const bool is_winner = dist_submit_claim(DistSubmitKind::Kernel, &mixed, ctx);
    TRACE_SPAN_END(
        claim_trace, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Claim, static_cast<uint32_t>(is_winner), 0
    );
    if (is_winner) {
        TRACE_SPAN_BEGIN(fanin_trace);
        ctx.fanin_count = dist_submit_collect_fanin(args, ctx, ctx.fanin);
        TRACE_SPAN_END(
            fanin_trace, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Fanin, 0,
            static_cast<uint32_t>(ctx.fanin_count)
        );
    }
    TRACE_SPAN_BEGIN(register_trace);
    dist_submit_register_outputs(ctx, args, /*include_existing=*/true);
    TRACE_SPAN_END(register_trace, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Register, 0, 1);
    if (is_winner) {
        TRACE_LAP(ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Build);
        dist_submit_build_winner_task(ctx, mixed, args);
    } else {
        TRACE_LAP(ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Replay);
        drain_block_won(ctx.self);
    }
    TRACE_SPAN_END(
        submit_trace, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Submit, static_cast<uint32_t>(is_winner), 0
    );
    return ctx.result;
}

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors dist_alloc_tensors(PTO2Runtime *, const L0TaskArgs &args) {
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, args, ctx);
    TRACE_SPAN_BEGIN(submit_trace);
    TRACE_LAP_RESET(ctx.self);
    drain_block_won(ctx.self);
    drain_phase_b(ctx.self);
    TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::EfDrain);
    if (!dist_submit_materialize_and_prepare_map(ctx.self, args, ctx, DistSubmitKind::Alloc)) return ctx.result;
    TRACE_SPAN_BEGIN(register_trace);
    dist_submit_register_outputs(ctx, args, /*include_existing=*/false);
    TRACE_SPAN_END(register_trace, ctx.self, ctx.task_id, -1, TracePhase::Register, 0, 0);
    TRACE_SPAN_BEGIN(claim_trace);
    const bool is_winner = dist_submit_claim(DistSubmitKind::Alloc, nullptr, ctx);
    TRACE_SPAN_END(claim_trace, ctx.self, ctx.task_id, -1, TracePhase::Claim, static_cast<uint32_t>(is_winner), 1);
    if (is_winner) {
        dist_submit_complete_alloc(ctx);
        TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::Alloc);
    } else {
        TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::Replay);
    }
    TRACE_SPAN_END(submit_trace, ctx.self, ctx.task_id, -1, TracePhase::Submit, static_cast<uint32_t>(is_winner), 1);
    return ctx.result;
}
#endif
