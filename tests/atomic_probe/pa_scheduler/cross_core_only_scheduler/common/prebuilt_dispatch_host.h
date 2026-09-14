#pragma once

#include "only_scheduler_host.h"
#include "pa_exec_adapter.h"

namespace pa_scheduler::host {

inline bool PrepareStaticExecBinding(cross_core::SharedExecCell &cell, uint64_t device_payload_address) {
    using namespace cross_core;
    const auto control = DecodeExecState(cell.control.state);
    if (!control.valid || control.phase != ExecPhase::Built ||
        cell.control.state != ExpectedPrebuiltPaControl(control.task_id) || device_payload_address == 0 ||
        device_payload_address % kExecCacheLineBytes != 0) return false;
    ExecutionToken token{};
    token.control.payload_address = reinterpret_cast<uintptr_t>(&cell.payload);
    ExecPayloadHeader header{};
    ExecPayloadLayout layout{};
    if (!ValidateBoundExecPayload(token, control.task_id, control.engine_class, control.payload_lines, header, layout) ||
        header.tensor_reference_mask != 0 || header.function_address != 0 || header.flags != 0 ||
        layout.payload_lines * kExecCacheLineBytes + sizeof(PreparedExecBinding) > sizeof(cell.payload)) return false;
    TaskKind kind = TaskKind::Count;
    PaExecRoute route{};
    if (!PaTaskKindFromExecFunction(header.function_id, kind) ||
        !ResolvePaExecRoute(kind, static_cast<int32_t>(header.function_id), route) ||
        route.engine_class != control.engine_class ||
        !PaExecShapeMatches(kind, header.tensor_count, header.scalar_count, header.fanin_count)) return false;

    CacheValidatedExecPayloadMetadata(token, header, layout);
    PreparedExecBinding prepared{};
    prepared.completion_vend = token.control.completion_vend;
    prepared.function_and_reference = token.control.function_and_reference;
    prepared.shape_and_scalar_offset = token.control.shape_and_scalar_offset;
    prepared.payload_bytes = layout.payload_bytes;
    for (uint32_t tensor = 0; tensor < header.tensor_count; ++tensor)
        prepared.args[tensor] = device_payload_address +
            ExecTensorPayloadWordOffset(tensor, 0) * sizeof(uint64_t);
    for (uint32_t scalar = 0; scalar < header.scalar_count; ++scalar)
        prepared.args[header.tensor_count + scalar] = cell.payload.words[layout.scalar_word_offset + scalar];

    const uint64_t binding_address = device_payload_address + layout.payload_lines * kExecCacheLineBytes;
    prepared.args[kExecDispatchLocalContextIndex] = binding_address + offsetof(PreparedExecBinding, local_context);
    prepared.args[kExecDispatchGlobalContextIndex] = binding_address + offsetof(PreparedExecBinding, global_context);
    // Every active worker has sub_block_id=0 (the second AIV subblock exits).
    // Context is task-private and read-only: this experiment has synchronous
    // kernels and never exposes an asynchronous completion list.
    PaLocalContext local{};
    local.block_count = 1;
    local.async.task_token = kInvalidTaskId;
    const PaGlobalContext global{};
    std::memcpy(prepared.local_context, &local, sizeof(local));
    std::memcpy(prepared.global_context, &global, sizeof(global));
    std::memcpy(const_cast<uint64_t *>(&cell.payload.words[layout.payload_lines * kExecHeaderWords]),
                &prepared, sizeof(prepared));
    return true;
}

inline bool PrepareStaticExecutionBindings(SchedulerState &state, uint64_t device_state_address) {
    const uint32_t count = state.build_dispatch.task_count;
    if (count == 0 || count > kMaxTasks || device_state_address == 0 || device_state_address % 64 != 0) return false;
    for (uint32_t task = 0; task < count; ++task) {
        if (task % 5 == 0) continue;  // Alloc is already complete and has no kernel.
        auto &cell = state.exec_cells[task];
        if (cross_core::DecodeExecState(cell.control.state).task_id != task ||
            !PrepareStaticExecBinding(cell, device_state_address + offsetof(SchedulerState, exec_cells) +
                task * sizeof(cross_core::SharedExecCell) + offsetof(cross_core::SharedExecCell, payload))) return false;
    }
    std::printf("[PREBOUND] %u immutable argument tables; descriptor/scalar/context binding before launch\n",
                state.build_dispatch.executable_task_count);
    return true;
}

}  // namespace pa_scheduler::host
