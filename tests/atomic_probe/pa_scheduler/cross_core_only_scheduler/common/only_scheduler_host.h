#pragma once
#include "host_support.h"

namespace pa_scheduler::host {

// Construction runs on the host. No workload is executed during preparation.
bool PrepareExecutionImage(SchedulerState *state, const Options &options);

inline bool ValidateOnlySchedulerOptions(const Options &options) {
    if (options.batches != 256 || options.runs != 1 || !options.shared_context_lens.empty() || options.profile_phases ||
        options.analyze_swimlane) {
        std::fprintf(
            stderr, "This experiment requires B256, one run, default G1; "
                    "no legacy Submit analyzer, phase-PMU or context-length override.\n"
        );
        return false;
    }
#if PA_BUILD_SWIMLANE
    if (!options.trace_enabled || !options.trace_atomics) return false;
#elif PA_BUILD_PERF_CLOCK
    if (options.trace_enabled || options.trace_atomics || !options.swimlane_json.empty()) return false;
#endif
    return true;
}

inline Metrics ValidateOnlyScheduler(
    const SchedulerState &state, uint32_t run, double, const TraceHeader *trace,
    RawExecTokenSnapshotAuthority token_authority
) {
    Metrics m;
    const uint32_t count = state.build_dispatch.task_count;
    Expect(state.config.workers == 16 && state.started_count.value == 16, "exactly 8 AIC + 8 AIV workers started", &m);
    Expect(state.fatal.value == 0 && state.exec_fatal.state == 0, "no scheduler or execution fatal", &m);
    Expect(state.build_dispatch.next_task.value == 0, "device does not take any Build ticket", &m);
    uint64_t kernels[4]{};
    uint64_t forbidden = 0, occupied = 0;
    uint64_t begin = UINT64_MAX, end = 0;
    for (uint32_t w = 0; w < kWorkers; ++w) {
        const auto &r = state.results[w];
        forbidden += r.submits + r.claim_attempts + r.claim_wins + r.materialized_outputs + r.map_inserts +
                     r.map_lookups + r.fanin_edges + r.tensor_args_added + r.scalar_args_added;
        if (r.submits || r.claim_attempts || r.claim_wins || r.materialized_outputs || r.map_inserts ||
            r.map_lookups || r.fanin_edges || r.tensor_args_added || r.scalar_args_added) {
            std::fprintf(stderr, "[BUILD-COUNTERS] worker=%u submit=%llu claim=%llu/%llu materialize=%llu "
                         "map=%llu/%llu fanin=%llu args=%llu/%llu\n", w,
                         static_cast<unsigned long long>(r.submits),
                         static_cast<unsigned long long>(r.claim_attempts),
                         static_cast<unsigned long long>(r.claim_wins),
                         static_cast<unsigned long long>(r.materialized_outputs),
                         static_cast<unsigned long long>(r.map_inserts),
                         static_cast<unsigned long long>(r.map_lookups),
                         static_cast<unsigned long long>(r.fanin_edges),
                         static_cast<unsigned long long>(r.tensor_args_added),
                         static_cast<unsigned long long>(r.scalar_args_added));
        }
        occupied += r.final_occupied;
        Expect(
            r.worker_id == w && r.role == static_cast<uint32_t>(w < kAicWorkers ? CoreRole::Aic : CoreRole::Aiv),
            "worker result has matching role", &m
        );
        for (uint32_t k = 0; k < 4; ++k)
            kernels[k] += r.kernel_counts[k];
#if PA_BUILD_PERF_CLOCK
        begin = std::min(begin, r.submit_begin);
        end = std::max(end, r.finish_cycle);
#else
        begin = std::min(begin, r.startup_barrier_begin);
        if (r.final_barrier_end < r.startup_barrier_begin) {
            std::fprintf(stderr, "[TIME-BOUNDARY] worker=%u startup=%llu drain=%llu finish=%llu\n", w,
                         static_cast<unsigned long long>(r.startup_barrier_begin),
                         static_cast<unsigned long long>(r.final_barrier_end),
                         static_cast<unsigned long long>(r.finish_cycle));
        }
        // finish_cycle follows trace-buffer export; do not charge that observer
        // tail to the scheduler's startup-to-drain interval.
        end = std::max(end, r.final_barrier_end);
#endif
        if (token_authority == RawExecTokenSnapshotAuthority::Authoritative) {
            for (const auto &token : state.exec_tokens[w]) {
                if (token.control.phase != cross_core::ExecTokenPhase::Idle) ++occupied;
            }
        }
    }
    Expect(forbidden == 0, "zero device Submit/Build/Materialize/Register/Fanin construction", &m);
    Expect(occupied == 0, "all executor tokens drained", &m);
    bool cells_ok = count == 5 * state.config.batches;
    for (uint32_t t = 0; t < count; ++t) {
        const auto decoded = cross_core::DecodeExecState(state.exec_cells[t].control.state);
        if (t % 5 == 0) {
            cells_ok &= state.tasks[t].flag == 1 && state.exec_cells[t].control.state == 0;
        } else {
            cells_ok &=
                decoded.valid && decoded.phase == cross_core::ExecPhase::Done && decoded.task_id == t &&
                decoded.build_owner == cross_core::kExecMaxOwner &&
                (decoded.execute_owner < kWorkers &&
                 ((decoded.engine_class == cross_core::ExecEngineClass::Aic && decoded.execute_owner < kAicWorkers) ||
                  (decoded.engine_class == cross_core::ExecEngineClass::Aiv && decoded.execute_owner >= kAicWorkers))
                ) &&
                state.tasks[t].flag == 1;
        }
    }
    Expect(cells_ok, "Alloc already complete; every kernel task DONE exactly once", &m);
    bool counts_ok = true;
    for (auto n : kernels)
        counts_ok &= n == state.config.batches;
    Expect(counts_ok, "QK/SF/PV/UP each execute B256 times", &m);
    uint64_t completed = 0;
    bool drain_ok = true;
    for (const auto &group : state.exec_drain.arrivals) {
        drain_ok &= cross_core::DecodeExecDrainArrivalCount(group.state) == 2;
        completed += cross_core::DecodeExecDrainCompletionCount(group.state);
    }
    Expect(drain_ok && completed == 4 * state.config.batches, "8 drain groups close with exact kernel count", &m);
    if (trace)
        Expect(ValidateTraceHeader(*trace, "scheduler-only validation"), "complete trace without dropped records", &m);
    Expect(begin != 0 && begin != UINT64_MAX && end >= begin, "valid startup-to-drain time window", &m);
    m.startup_to_final_drain_us = m.lifecycle_span_us =
        begin != UINT64_MAX && end >= begin ? (end - begin) / 1000.0 : 0;
    m.final_drain_span_us = m.startup_to_final_drain_us;
    std::printf(
        "[ONLY_SCHEDULER] run=%u workers=8+8 tasks=%u kernels=%llu "
        "startup_to_final_drain_us=%.3f status=%s\n",
        run, count, static_cast<unsigned long long>(completed), m.startup_to_final_drain_us, m.passed ? "PASS" : "FAIL"
    );
    return m;
}

inline bool IsBuildTracePhase(int32_t phase) {
    return phase == static_cast<int32_t>(TracePhase::Alloc) || phase == static_cast<int32_t>(TracePhase::Build) ||
           phase == static_cast<int32_t>(TracePhase::PrepareMap) ||
           phase == static_cast<int32_t>(TracePhase::OrchestrationReplay) ||
           phase == static_cast<int32_t>(TracePhase::AllocComplete) ||
           (phase >= static_cast<int32_t>(TracePhase::SharedRegisterPublishMetadata) &&
            phase <= static_cast<int32_t>(TracePhase::SharedMaterializePublishTaskOutputsFlush)) ||
           phase == static_cast<int32_t>(TracePhase::Submit) || phase == static_cast<int32_t>(TracePhase::Claim) ||
           phase == static_cast<int32_t>(TracePhase::Materialize) ||
           phase == static_cast<int32_t>(TracePhase::Register) || phase == static_cast<int32_t>(TracePhase::Fanin) ||
           phase == static_cast<int32_t>(TracePhase::WinnerBuild);
}

template <typename ReadRecords, typename ReadSubmitRecords>
inline bool ExportOnlySchedulerRecords(
    const TraceHeader &header, const SchedulerState &state, const std::string &path, WinnerWorkloadMode mode,
    WorkloadCounts counts, const char *pattern, FinalBarrierShape, bool atomic_enabled, const char *backend,
    ReadRecords read, ReadSubmitRecords
) {
    if (!ValidateTraceHeader(header, "scheduler-only export")) return false;
    const std::string temporary = path + ".tmp";
    std::FILE *out = std::fopen(temporary.c_str(), "wb");
    if (!out) return false;
    std::fprintf(
        out,
        "{\"metadata\":{\"schema\":\"pa-only-scheduler-v2\","
        "\"atomic_detail\":\"per_call\","
        "\"scheduler_detail\":\"business_spans_v1\","
        "\"dispatch_binding\":\"host_prebound\","
        "\"claim_strategy\":\"prebuilt_single_cas\","
        "\"base\":\"cross_core_ordinary\",\"aic_workers\":8,\"aiv_workers\":8,"
        "\"task_build\":\"host_before_launch\",\"batches\":%u,\"tasks\":%u,"
        "\"clock_freq_hz\":%llu,\"workload_mode\":%u,\"pattern\":\"%s\","
        "\"counts\":[%u,%u,%u,%u],\"backend\":\"%s\","
        "\"clock_source\":\"%s\",\"task_dependencies\":[",
        state.config.batches, state.build_dispatch.task_count, static_cast<unsigned long long>(header.frequency_hz),
        static_cast<uint32_t>(mode), pattern, counts.qk, counts.sf, counts.pv, counts.up, backend,
        std::strcmp(backend, "ccec") == 0 ? "AICore_SYS_CNT" : "CPU_steady_clock"
    );
    bool first_edge = true;
    for (uint32_t t = 0; t < state.build_dispatch.task_count; ++t) {
        if (t % 5 == 0) continue;
        const auto &payload = state.exec_cells[t].payload;
        const auto h = cross_core::DecodeExecPayloadHeader(payload);
        cross_core::ExecPayloadLayout layout{};
        if (!cross_core::ComputeExecPayloadLayout(
                h.tensor_count, h.scalar_count, h.fanin_count, h.tensor_reference_mask, layout
            )) {
            std::fclose(out);
            std::remove(temporary.c_str());
            return false;
        }
        for (uint32_t e = 0; e < h.fanin_count; ++e) {
            const auto packed = payload.words[layout.fanin_word_offset + e / 2];
            const int32_t producer = static_cast<int32_t>(packed >> (32 * (e % 2)));
            std::fprintf(out, "%s[%u,%d]", first_edge ? "" : ",", t, producer);
            first_edge = false;
        }
    }
    std::fprintf(out, "],\"workers\":[");
    for (uint32_t w = 0; w < kWorkers; ++w) {
        const auto &c = header.cores[w];
        std::fprintf(
            out,
            "%s{\"worker\":%u,\"block\":%d,\"lane\":%d,"
            "\"records\":%u,\"dropped\":%u,\"atomic_calls\":%u,"
            "\"dcci_records\":%u,\"startup_tick\":%llu,\"finish_tick\":%llu}",
            w ? "," : "", w, c.block_id, c.lane, c.count, c.dropped, c.atomic_calls, c.dcci_records,
            static_cast<unsigned long long>(state.results[w].startup_barrier_begin),
            static_cast<unsigned long long>(state.results[w].finish_cycle)
        );
    }
    std::fprintf(out, "]},\"fdwic_events\":[");
    std::vector<TraceStorageRecord> storage(kTraceRecordsPerCore);
    std::vector<TraceRecord> records(kTraceRecordsPerCore);
    bool ok = true, first = true;
    uint32_t kernel_events = 0;
    for (uint32_t w = 0; w < kWorkers && ok; ++w) {
        const auto &c = header.cores[w];
        if (c.block_id != static_cast<int32_t>(w % 8) || c.lane != (w < 8 ? 0 : 1) ||
            c.core_idx != static_cast<int32_t>(w) || !read(w, c.count, storage.data()) ||
            !DecodeTraceStorageRecords(
                storage.data(), c.count, state.results[w].startup_barrier_begin, state.results[w].finish_cycle,
                records.data()
            )) {
            std::fprintf(
                stderr, "Invalid worker trace w=%u block=%d lane=%d core=%d records=%u\n", w, c.block_id, c.lane,
                c.core_idx, c.count
            );
            ok = false;
            break;
        }
        uint64_t atomic_calls = 0, dcci_records = 0;
        for (uint32_t i = 0; i < c.count; ++i) {
            const auto &r = records[i];
            if (r.end_cycle < r.start_cycle || r.phase < 0 || r.phase >= static_cast<int32_t>(TracePhase::Count) ||
                IsBuildTracePhase(r.phase)) {
                std::fprintf(stderr, "Invalid phase/time w=%u record=%u phase=%d\n", w, i, r.phase);
                ok = false;
                break;
            }
            if (r.phase == static_cast<int32_t>(TracePhase::Atomic)) {
                if (!AtomicRecordSchemaValid(r, atomic_enabled) || (r.flags & kAtomicPollBatch) != 0) {
                    std::fprintf(
                        stderr, "Invalid Atomic w=%u record=%u flags=%u site=%u\n", w, i, r.flags, r.auxiliary
                    );
                    ok = false;
                    break;
                }
                atomic_calls += AtomicRecordCallCount(r);
                if (r.auxiliary == static_cast<uint32_t>(AtomicSite::SharedBuildDispatchTicket)) {
                    ok = false;
                    break;
                }
            }
            if (r.phase == static_cast<int32_t>(TracePhase::Dcci)) {
                if (!DcciRecordSchemaValid(r) || (r.auxiliary == static_cast<uint32_t>(DcciSite::ObserverTraceExport) &&
                                                  DcciRecordCallCount(r) != 2)) {
                    std::fprintf(stderr, "Invalid DCCI w=%u record=%u flags=%u site=%u\n", w, i, r.flags, r.auxiliary);
                    ok = false;
                    break;
                }
                ++dcci_records;
            }
            if (r.phase == static_cast<int32_t>(TracePhase::Kernel)) ++kernel_events;
            std::fprintf(
                out, "%s[%u,%d,%d,%d,%d,\"%s\",%llu,%llu,%u,%u]", first ? "" : ",", w, c.block_id, c.lane, r.task_id,
                r.function_id, TracePhaseName(static_cast<uint32_t>(r.phase)),
                static_cast<unsigned long long>(r.start_cycle), static_cast<unsigned long long>(r.end_cycle), r.flags,
                r.auxiliary
            );
            first = false;
        }
        if (atomic_calls != c.atomic_calls || dcci_records != c.dcci_records) {
            std::fprintf(
                stderr, "Trace count mismatch w=%u atomic=%llu/%u dcci=%llu/%u\n", w,
                static_cast<unsigned long long>(atomic_calls), c.atomic_calls,
                static_cast<unsigned long long>(dcci_records), c.dcci_records
            );
            ok = false;
        }
    }
    ok &= kernel_events == state.config.batches * 4;
    if (!ok) std::fprintf(stderr, "Kernel trace count=%u expected=%u\n", kernel_events, state.config.batches * 4);
    std::fprintf(out, "]}\n");
    ok &= std::ferror(out) == 0;
    ok &= std::fclose(out) == 0;
    if (ok && std::rename(temporary.c_str(), path.c_str()) == 0) return true;
    std::remove(temporary.c_str());
    std::fprintf(stderr, "Scheduler-only trace validation/export failed.\n");
    return false;
}
}  // namespace pa_scheduler::host
