#include <array>
#include <cassert>
#include <cstring>
#include <iostream>

#include "common/prebuilt_dispatch_host.h"

using namespace pa_scheduler;
using namespace pa_scheduler::cross_core;

struct TestOps {
    static int64_t CompareExchange(volatile int64_t *p, int64_t expected, int64_t desired) {
        const int64_t old = *p;
        if (old == expected) *p = desired;
        return old;
    }
    static void BeforePayloadAcquire(uint32_t) {}
    static void PreloadPayloadSource(const void *, uint64_t) {}
};

struct Observer {
    uint64_t acquired_bytes = 0;
    int claims = 0;
    int64_t LoadCellState(volatile int64_t *, uint32_t) = delete;
    int64_t ClaimCell(volatile int64_t *p, int64_t expected, int64_t desired, uint32_t) {
        ++claims;
        return TestOps::CompareExchange(p, expected, desired);
    }
    int64_t PublishFatal(volatile int64_t *p, int64_t expected, int64_t desired, uint32_t) {
        return TestOps::CompareExchange(p, expected, desired);
    }
    void InvalidateClaimPayload(const void *, uint64_t bytes, uint32_t) { acquired_bytes = bytes; }
};

int main() {
    const std::array<TaskKind, 4> kinds{TaskKind::Qk, TaskKind::Sf, TaskKind::Pv, TaskKind::Up};
    for (const auto kind : kinds) {
        SharedExecCell cell{};
        PaExecShape shape{};
        PaExecRoute route{};
        assert(ResolvePaExecShape(kind, shape));
        assert(ResolvePaExecRoute(kind, static_cast<int32_t>(kind) - 1, route));
        const uint32_t task = 40 + static_cast<uint32_t>(kind);
        ExecPayloadLayout layout{};
        assert(ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout));
        cell.payload.words[0] = task;
        cell.payload.words[2] = 1024;
        cell.payload.words[3] = route.function_id | (uint64_t{layout.payload_bytes} << 32U);
        cell.payload.words[4] = uint64_t{shape.tensor_count} | (uint64_t{shape.scalar_count} << 16U) |
            (uint64_t{shape.fanin_count} << 32U) | (uint64_t{static_cast<uint8_t>(route.engine_class)} << 48U);
        cell.payload.words[5] = uint64_t{1} << 48U;
        for (uint32_t i = 0; i < shape.scalar_count; ++i) cell.payload.words[layout.scalar_word_offset + i] = 100 + i;
        cell.control.state = static_cast<int64_t>(EncodeExecState(
            ExecPhase::Built, kExecMaxOwner, kExecUnboundOwner, route.engine_class, layout.payload_lines, task));
        Observer observer;
        const int64_t expected = PrebuiltPaExecBinding::InitialClaimControl(cell, task, observer);
        assert(expected == cell.control.state);
        constexpr uint64_t device_payload = 0x40000000;
        assert(host::PrepareStaticExecBinding(cell, device_payload));
        const auto &prepared = PreparedBindingForPayload(cell.payload, layout.payload_lines);
        for (uint32_t i = 0; i < shape.tensor_count; ++i)
            assert(prepared.args[i] == device_payload + 64 + i * 128);
        for (uint32_t i = 0; i < shape.scalar_count; ++i)
            assert(prepared.args[shape.tensor_count + i] == 100 + i);
        const uint64_t prepared_base = device_payload + layout.payload_lines * 64;
        assert(prepared.args[48] == prepared_base + offsetof(PreparedExecBinding, local_context));
        assert(prepared.args[49] == prepared_base + offsetof(PreparedExecBinding, global_context));
        PaLocalContext local{};
        PaGlobalContext global{};
        std::memcpy(&local, prepared.local_context, sizeof(local));
        std::memcpy(&global, prepared.global_context, sizeof(global));
        assert(local.block_index == 0 && local.block_count == 1 && global.sub_block_id == 0);
        assert(local.async.completion_count == 0 && local.async.completion_error_code == 0 &&
               local.async.completion_entries == 0 && local.async.completion_capacity == 0 &&
               local.async.alignment_padding == 0 && local.async.task_token == kInvalidTaskId);

        const SharedExecCell before = cell;
        ExecutionToken token{};
        std::memset(token.dispatch.args, 0x5a, sizeof(token.dispatch.args));
        SharedExecFatalControl fatal{};
        const uint32_t owner = route.engine_class == ExecEngineClass::Aic ? 0 : 8;
        assert(PrebuiltPaExecBinding::Claim<TestOps>(cell, expected, task, owner,
            route.engine_class, token, fatal, observer) == ExecClaimResult::Claimed);
        assert(observer.claims == 1 && observer.acquired_bytes == layout.payload_lines * 64 + sizeof(PreparedExecBinding));
        assert(token.control.phase == ExecTokenPhase::WaitingFanin);
        assert(token.control.fanin_ready_prefix == 0 && token.control.completion_vend == 1024);
        assert(ExecutionTokenFunctionId(token) == route.function_id);
        assert(ExecutionTokenDispatchArgs(token) == prepared.args);
        assert(token.dispatch.args[0] == 0x5a5a5a5a5a5a5a5aULL);
        assert(std::memcmp(const_cast<const uint64_t *>(cell.payload.words),
            const_cast<const uint64_t *>(before.payload.words), sizeof(cell.payload)) == 0);
        ResetExecutionToken(token);
        assert(ExecutionTokenDispatchArgs(token) == token.dispatch.args);

        observer = {};
        assert(PrebuiltPaExecBinding::Claim<TestOps>(cell, expected, task, owner,
            route.engine_class, token, fatal, observer) == ExecClaimResult::Lost);
        assert(observer.claims == 1 && observer.acquired_bytes == 0);
        assert(token.control.phase == ExecTokenPhase::Idle && token.dispatch.prepared_args_address == 0);

        cell = before;
        observer = {};
        assert(ClaimAndBindPreparedExecPayload<TestOps>(cell, cell.control.state, task - 1, owner,
            route.engine_class, token, fatal, observer) == ExecClaimResult::InvalidControl);
        assert(observer.claims == 0 && observer.acquired_bytes == 0);
        const auto wrong_engine = route.engine_class == ExecEngineClass::Aic ?
            ExecEngineClass::Aiv : ExecEngineClass::Aic;
        assert(ClaimAndBindPreparedExecPayload<TestOps>(cell, cell.control.state, task, owner,
            wrong_engine, token, fatal, observer) == ExecClaimResult::Incompatible);
        assert(observer.claims == 0 && observer.acquired_bytes == 0);
        cell.control.state = static_cast<int64_t>(EncodeExecState(
            ExecPhase::Built, kExecMaxOwner, kExecUnboundOwner, route.engine_class, kExecMaxPayloadLines, task));
        assert(ClaimAndBindPreparedExecPayload<TestOps>(cell, cell.control.state, task, owner,
            route.engine_class, token, fatal, observer) == ExecClaimResult::InvalidControl);
        assert(observer.claims == 0 && observer.acquired_bytes == 0);

        const std::array<int64_t, 8> rejected_controls{
            0, -1,
            static_cast<int64_t>(EncodeExecState(ExecPhase::Building, kExecMaxOwner,
                kExecUnboundOwner, ExecEngineClass::None, 0, task)),
            static_cast<int64_t>(EncodeExecState(ExecPhase::Built, kExecMaxOwner,
                kExecUnboundOwner, route.engine_class, layout.payload_lines, task + 5)),
            static_cast<int64_t>(EncodeExecState(ExecPhase::Built, 0,
                kExecUnboundOwner, route.engine_class, layout.payload_lines, task)),
            static_cast<int64_t>(EncodeExecState(ExecPhase::Built, kExecMaxOwner,
                kExecUnboundOwner, wrong_engine, layout.payload_lines, task)),
            static_cast<int64_t>(EncodeExecState(ExecPhase::Built, kExecMaxOwner,
                kExecUnboundOwner, route.engine_class, layout.payload_lines + 1, task)),
            static_cast<int64_t>(EncodeExecState(ExecPhase::Done, kExecMaxOwner,
                owner, route.engine_class, layout.payload_lines, task)),
        };
        for (const int64_t wrong : rejected_controls) {
            cell = before;
            cell.control.state = wrong;
            observer = {};
            // The initial word must come from the fixed PA contract, never the
            // mutable cell or its payload; only the CAS may observe that cell.
            assert(PrebuiltPaExecBinding::InitialClaimControl(cell, task, observer) == expected);
            assert(PrebuiltPaExecBinding::Claim<TestOps>(cell, expected, task, owner,
                route.engine_class, token, fatal, observer) == ExecClaimResult::Lost);
            assert(observer.claims == 1 && observer.acquired_bytes == 0 && cell.control.state == wrong);
            assert(token.control.phase == ExecTokenPhase::Idle);
            assert(!host::PrepareStaticExecBinding(cell, device_payload));
        }

        cell = before;
        cell.payload.words[7] = 1;
        assert(!host::PrepareStaticExecBinding(cell, device_payload));
        cell = before;
        cell.payload.words[3] ^= uint64_t{1} << 32U;
        assert(!host::PrepareStaticExecBinding(cell, device_payload));
        cell = before;
        assert(!host::PrepareStaticExecBinding(cell, device_payload + 1));
    }
    assert(ExpectedPrebuiltPaControl(0) == -1 && ExpectedPrebuiltPaControl(kMaxTasks) == -1);
    for (uint32_t task = 0; task < 1280; ++task) {
        const auto decoded = DecodeExecState(ExpectedPrebuiltPaControl(task));
        if (task % 5 == 0) {
            assert(!decoded.valid);
        } else {
            assert(decoded.valid && decoded.phase == ExecPhase::Built && decoded.task_id == task);
            assert(decoded.build_owner == kExecMaxOwner && decoded.execute_owner == kExecUnboundOwner);
            assert(decoded.payload_lines == (task % 5 == 4 ? 16 : 10));
            assert(decoded.engine_class == (task % 5 == 1 || task % 5 == 3 ?
                ExecEngineClass::Aic : ExecEngineClass::Aiv));
        }
    }
    std::cout << "[PASS] prebuilt dispatch relocation, immutable claim, and preparation rejection\n";
}
