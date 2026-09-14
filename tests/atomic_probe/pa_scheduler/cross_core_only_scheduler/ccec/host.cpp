/*
 * Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository.
 */
#include "../common/only_scheduler_host.h"
#include "../common/prebuilt_dispatch_host.h"
#include "../common/winner_workload_host.h"
#include <acl/acl.h>
#include <runtime/rt.h>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>

namespace {
using namespace pa_scheduler;
using namespace pa_scheduler::host;

void Check(int code, const char *operation) {
    if (code != 0) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(code));
}

// No PMU owner, AICPU library or graph runtime is loaded by this runner.
struct DeviceSession {
    bool initialized = false, selected = false, registered_all = true;
    uint32_t device = 0;
    aclrtStream stream = nullptr;
    void *kernel = nullptr;
    std::vector<void *> allocations;

    void *Allocate(size_t bytes) {
        void *p = nullptr;
        Check(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        allocations.push_back(p);
        if (reinterpret_cast<uintptr_t>(p) % 64 != 0)
            throw std::runtime_error("device allocation is not cache-line aligned");
        return p;
    }

    bool Close() {
        bool ok = true;
        const auto cleanup = [&ok](int code, const char *name) {
            if (code != 0) {
                std::fprintf(stderr, "%s failed: %d\n", name, code);
                ok = false;
            }
        };
        for (auto it = allocations.rbegin(); it != allocations.rend(); ++it)
            cleanup(aclrtFree(*it), "aclrtFree");
        allocations.clear();
        if (kernel) {
            cleanup(registered_all ? rtDevBinaryUnRegister(kernel) : rtBinaryUnLoad(kernel), "unload kernel");
            kernel = nullptr;
        }
        if (stream) {
            cleanup(aclrtDestroyStream(stream), "destroy stream");
            stream = nullptr;
        }
        if (selected) {
            cleanup(aclrtResetDevice(device), "reset device");
            selected = false;
        }
        if (initialized) {
            cleanup(aclFinalize(), "aclFinalize");
            initialized = false;
        }
        return ok;
    }
    ~DeviceSession() { Close(); }
};

void H2D(void *destination, const void *source, size_t bytes) {
    Check(aclrtMemcpy(destination, bytes, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
}
void D2H(void *destination, const void *source, size_t bytes) {
    Check(aclrtMemcpy(destination, bytes, source, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
}

int Run(int argc, char **argv) {
    Options options;
    options.runs = 1;
    WinnerWorkloadOptions workload;
    std::vector<char *> common_argv;
    if (!ParseWinnerWorkloadOptions(argc, argv, &workload, &common_argv)) return EXIT_FAILURE;
    const auto parse = ParseOptions(common_argv.size(), common_argv.data(), true, &options);
    if (parse != ParseStatus::Ok) return parse == ParseStatus::Help ? EXIT_SUCCESS : EXIT_FAILURE;
    if (!ValidateOnlySchedulerOptions(options) || !ValidateWinnerWorkloadOptions(workload)) return EXIT_FAILURE;
    std::ifstream input(options.kernel_path, std::ios::binary);
    std::vector<char> binary_data((std::istreambuf_iterator<char>(input)), {});
    if (binary_data.empty()) throw std::runtime_error("empty or missing kernel ELF");

    std::unique_ptr<SchedulerState> state(new SchedulerState);
    InitializeState(state.get(), options);
    SharedHostTaskPlan plan;
    SharedHostHeapAdmission admission;
    std::string admission_error;
    if (!BuildSharedHostTaskPlan(*state, &plan, &admission_error) ||
        !ValidateSharedHostHeapAdmission(plan, state->heap_size, &admission, &admission_error))
        throw std::runtime_error(admission_error);
    PrintSharedHostHeapAdmission(plan, admission);
    if (!PrepareExecutionImage(state.get(), options)) return EXIT_FAILURE;
    const auto immutable_dispatch = state->exec_dispatch;

    const bool real = workload.mode == WinnerWorkloadMode::RealCompute;
    std::vector<float> workload_image, outputs;
    if (real) InitializeWinnerWorkloadBuffers(workload, &workload_image, &outputs);
    PrintBanner("A5 scheduler-only", options);
    PrintWinnerWorkloadConfig(workload, options.nops);

    DeviceSession session;
    session.device = options.device;
    Check(aclInit(nullptr), "aclInit");
    session.initialized = true;
    Check(aclrtSetDevice(options.device), "aclrtSetDevice");
    session.selected = true;
    const char *soc = aclrtGetSocName();
    if (!soc || std::string(soc).find("Ascend950") != 0)
        throw std::runtime_error("scheduler-only CCEC requires verified Ascend950/A5 silicon");
    std::printf("[DEVICE] soc=%s active_aic=8 active_aiv=8\n", soc);
    Check(aclrtCreateStream(&session.stream), "create stream");
    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, binary_data.data(), binary_data.size()};
    int error = rtRegisterAllKernel(&binary, &session.kernel);
    if (error != 0 || !session.kernel) {
        session.registered_all = false;
        error = rtBinaryLoadWithoutTilingKey(binary_data.data(), binary_data.size(), &session.kernel);
    }
    Check(error, "register mixed ELF");
    if (!session.kernel) throw std::runtime_error("null kernel handle");
    auto *device = static_cast<SchedulerState *>(session.Allocate(sizeof(SchedulerState)));
    if (!PrepareStaticExecutionBindings(*state, reinterpret_cast<uintptr_t>(device)))
        throw std::runtime_error("invalid prebuilt dispatch image");
    std::vector<cross_core::ExecPayloadStorage> immutable_payloads;
    for (uint32_t t = 0; t < state->build_dispatch.task_count; ++t)
        immutable_payloads.push_back(state->exec_cells[t].payload);
    void *workspace = real ? session.Allocate(winner_workload::kWorkspaceBytes) : nullptr;
    void *trace = options.trace_enabled ? session.Allocate(kTraceBytes) : nullptr;
    ConfigureWinnerWorkload(state.get(), workload, workspace);
    ConfigureTrace(state.get(), options, trace);
    if (real) H2D(workspace, workload_image.data(), winner_workload::kWorkspaceBytes);
    TraceHeader header{};
    if (trace) {
        InitializeTraceHeader(&header);
        H2D(trace, &header, sizeof(header));
    }
    // Host-built descriptors/fanin and Alloc results are immutable before launch.
    // Preserve the ordinary ABI's arena/offsets, but do not copy unused worker arenas.
    H2D(device, state.get(), StatePrefixBytes());
    H2D(&device->config, &state->config, ControlBytes());
    H2D(device->results, state->results, ResultBytes());
    H2D(&device->shared_map, &state->shared_map, SharedSidecarBytes());
    H2D(&device->claim_tournament, &state->claim_tournament, SharedClaimTournamentBytes());
    H2D(&device->exec_fatal, &state->exec_fatal, CrossCoreExecStateBytes());

    void *kernel_args[] = {device};
    rtArgsEx_t args{};
    args.args = kernel_args;
    args.argsSize = sizeof(kernel_args);
    rtTaskCfgInfo_t task_config{};
    // Hardware mixed metadata creates 1+2 contexts/block. AIV subblock 1 exits
    // before the scheduler, leaving exactly eight AIC and eight AIV workers.
    Check(
        rtKernelLaunchWithHandleV2(session.kernel, 0, kAicWorkers, &args, nullptr, session.stream, &task_config),
        "launch 8 mixed blocks"
    );
    Check(aclrtSynchronizeStreamWithTimeout(session.stream, 30000), "synchronize (30 s)");
    D2H(state.get(), device, StatePrefixBytes());
    D2H(state->results, device->results, ResultBytes());
    D2H(&state->exec_fatal, &device->exec_fatal, CrossCoreExecStateBytes());
    if (trace) D2H(&header, trace, sizeof(header));
    if (real) {
        D2H(outputs.data(),
            static_cast<uint8_t *>(workspace) + winner_workload::kSharedInputTiles * winner_workload::kTileBytes,
            winner_workload::kOutputTiles * winner_workload::kTileBytes);
    }
    bool immutable = true;
    for (uint32_t t = 0; t < state->build_dispatch.task_count; ++t)
        immutable &= std::memcmp(
                         &immutable_payloads[t], &state->exec_cells[t].payload, sizeof(cross_core::ExecPayloadStorage)
                     ) == 0;
    immutable &=
        std::memcmp(
            immutable_dispatch.aic_task_ids, state->exec_dispatch.aic_task_ids, sizeof(immutable_dispatch.aic_task_ids)
        ) == 0;
    immutable &=
        std::memcmp(
            immutable_dispatch.aiv_task_ids, state->exec_dispatch.aiv_task_ids, sizeof(immutable_dispatch.aiv_task_ids)
        ) == 0;
    std::printf("[IMMUTABLE] payloads and execution lists unchanged: %s\n", immutable ? "PASS" : "FAIL");
    auto metrics =
        ValidateOnlyScheduler(*state, 1, 0, trace ? &header : nullptr, RawExecTokenSnapshotAuthority::DiagnosticOnly);
    bool ok = immutable && metrics.passed;
    if (real) ok &= ValidateRealComputeOutputs(*state, workload, outputs, 1);
    if (ok && !options.swimlane_json.empty()) {
        const auto read = [trace](uint32_t w, uint32_t count, TraceStorageRecord *records) {
            D2H(records, static_cast<uint8_t *>(trace) + TraceRecordsOffset(w), count * sizeof(TraceStorageRecord));
            return true;
        };
        const auto unused_submit = [](uint32_t, uint32_t, SharedSubmitClaimTraceRecord *) {
            return false;
        };
        ok &= ExportOnlySchedulerRecords(
            header, *state, options.swimlane_json, workload.mode,
            real ? workload.repeats :
                   WorkloadCounts{options.nops.qk, options.nops.sf, options.nops.pv, options.nops.up},
            real ? RealComputePatternName(workload.pattern) : "none", options.final_barrier_shape,
            options.trace_atomics, "ccec", read, unused_submit
        );
    }
    ok &= session.Close();
    std::printf("[SUMMARY] scheduler_only=%s device_build=0 cores=8+8\n", ok ? "PASS" : "FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
}  // namespace

int main(int argc, char **argv) {
    try {
        return Run(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[ERROR] %s\n", e.what());
        return EXIT_FAILURE;
    }
}
