#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#endif

#include "aicpu/device_log.h"
#include "aicpu/device_time.h"
#include "spin_hint.h"
#include "runtime.h"
#include "pto2_dispatch_payload.h"

// Runtime headers (full struct definition for create/destroy + PTO2_SCOPE)
#include "pto_runtime2.h"
#include "pto_shared_memory.h"
#include "pto_runtime2_types.h"

// Performance profiling headers
#include "aicpu/performance_collector_aicpu.h"
#include "common/perf_profiling.h"
#include "common/memory_barrier.h"
#include "common/unified_log.h"

// Register-based communication
#include "common/platform_config.h"
#include "aicpu/platform_regs.h"

// Scheduler profiling helper
#ifndef PTO2_PROFILING
#define PTO2_PROFILING 1
#endif

#if PTO2_PROFILING
// Accumulated nanoseconds per sub-step
#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#define CYCLE_COUNT_LAP_AND_MAX(acc, m) \
    do {                                \
        _t1 = get_sys_cnt_aicpu();      \
        if (_t1 - _t0 > m) {            \
            m = _t1 - _t0;              \
        }                               \
        acc += (_t1 - _t0);             \
        _t0 = _t1;                      \
    } while (0)

#define CYCLE_COUNT_START_X(X) uint64_t _t0##X = get_sys_cnt_aicpu(), _t1##X
#define CYCLE_COUNT_LAP_X(acc, X)       \
    do {                              \
        _t1##X = get_sys_cnt_aicpu(); \
        acc += (_t1##X - _t0##X);     \
        _t0##X = _t1##X;              \
    } while (0)

#define CYCLE_COUNT_LAP_AND_MAX_X(acc, m, X) \
    do {                                   \
        _t1##X = get_sys_cnt_aicpu();      \
        if (_t1##X - _t0##X > m) {         \
            m = _t1##X - _t0##X;           \
        }                                  \
        acc += (_t1##X - _t0##X);          \
        _t0##X = _t1##X;                   \
    } while (0)

#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#define CYCLE_COUNT_LAP_AND_MAX(acc, m)
#define CYCLE_COUNT_START_X(X)
#define CYCLE_COUNT_LAP_X(acc, X)
#define CYCLE_COUNT_LAP_AND_MAX_X(acc, m, X)
#endif

// Device orchestration function signature (loaded via dlopen).
// The orchestration .so receives a PTO2Runtime* (with ops table populated)
// instead of a raw shared-memory pointer.
typedef void (*DeviceOrchestrationFunc)(PTO2Runtime* rt, uint64_t* args, int arg_count, int orch_thread_num, int orch_thread_index);

// Config function exported by orchestration .so
typedef PTO2OrchestrationConfig (*DeviceOrchestrationConfigFunc)(uint64_t* args, int arg_count);

constexpr int MAX_AICPU_THREADS = PLATFORM_MAX_AICPU_THREADS;
constexpr int MAX_AIC_PER_THREAD = PLATFORM_MAX_AIC_PER_THREAD;
constexpr int MAX_AIV_PER_THREAD = PLATFORM_MAX_AIV_PER_THREAD;
constexpr int MAX_CORES_PER_THREAD = PLATFORM_MAX_CORES_PER_THREAD;

constexpr int MAX_IDLE_ITERATIONS = 800000;  // ~20s idle then scheduler gives up (avoid long hang)
constexpr int STALL_LOG_INTERVAL = 50000;  // DEV_ALWAYS every N idle iters to debug hang
constexpr int STALL_DUMP_READY_MAX = 8;
constexpr int STALL_DUMP_WAIT_MAX = 4;
constexpr int STALL_DUMP_CORE_MAX = 8;
constexpr int PROGRESS_VERBOSE_THRESHOLD = 10;  // log every completion for the first N tasks
constexpr int PROGRESS_LOG_INTERVAL = 250;     // log every N completions after threshold

// Core information for discovery (with register address for fast dispatch)
struct CoreInfo {
    int worker_id;              // Index in runtime.workers[]
    uint32_t physical_core_id;  // Hardware physical core ID (from AICore)
    uint64_t reg_addr;          // Cached register address for fast access
    CoreType core_type;
};

// Cores donated from a scheduler thread to an orchestrator thread during transition
struct DonatedCoresSlot {
    int idle[MAX_CORES_PER_THREAD];
    int idle_count{0};
    int running[MAX_CORES_PER_THREAD];
    int running_count{0};
};

struct CoreStateTracker {
    int idle[MAX_CORES_PER_THREAD];
    int running[MAX_CORES_PER_THREAD];
    int idle_count;
    int running_count;

    void move_idle_to_running(int idx) {
        running[running_count++] = idle[idx];
        idle[idx] = idle[--idle_count];
    }

    void move_running_to_idle(int idx) {
        idle[idle_count++] = running[idx];
        running[idx] = running[--running_count];
    }
};


static PTO2Runtime *rt{nullptr};

struct AicpuExecutor {
    int orch_thread_num_;
    int sched_thread_num_;

    // ===== Thread management state =====
    std::atomic<int> thread_idx_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<bool> finished_{false};

    int thread_num_{0};
    int cores_total_num_{0};
    int thread_cores_num_{0};  // Cores per scheduler thread (0 for orchestrator when thread_num_==4)
    int32_t core_count_per_thread_[MAX_AICPU_THREADS];  // Actual core count per thread
    int32_t core_assignments_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];

    // Core discovery arrays (with register addresses)
    CoreInfo aic_cores_[MAX_CORES_PER_THREAD];
    CoreInfo aiv_cores_[MAX_CORES_PER_THREAD];
    int aic_count_{0};
    int aiv_count_{0};

    // Fast lookup: core_id -> reg_addr (for register-based dispatch)
    uint64_t core_id_to_reg_addr_[MAX_CORES_PER_THREAD];

    // Platform register base address array (set via get_platform_regs())
    uint64_t regs_{0};

    // Track executing task_id per core (AICPU_TASK_INVALID = idle)
    int executing_task_ids_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];
    CoreStateTracker trackers_[MAX_AICPU_THREADS];

    // ===== Task queue state (managed by scheduler ready queues) =====

    // Task execution tracking
    std::atomic<int> completed_tasks_{0};
    int total_tasks_{0};
    std::atomic<int> finished_count_{0};
    // Device orchestration: set by Thread 3 when graph is built; workers wait for it
    bool orchestrator_done_{false};
    std::atomic<bool> pto2_init_done_{false};
    std::atomic<bool> runtime_init_ready_{false};
    std::atomic<bool> pto2_init_complete_{false};  // init block finished; others wait for this
    std::atomic<int> orch_finished_count_{0};      // Number of orchestrator threads that have finished

    // ===== Dynamic core transition state =====
    std::atomic<bool> transition_requested_{false};
    std::atomic<int32_t> wait_reassign_{0};
    std::atomic<bool> reassigned_{false};
    std::atomic<bool> completed_{false};

    // Orchestration SO handle - defer dlclose until all tasks complete
    void* orch_so_handle_{nullptr};
    char orch_so_path_[256]{};  // Path to orchestration SO file for cleanup

    // Shared orchestration function pointer (loaded by first orch thread, used by all)
    DeviceOrchestrationFunc orch_func_{nullptr};
    uint64_t* orch_args_cached_{nullptr};
    int orch_arg_count_cached_{0};

    // ===== Performance profiling state =====
    uint64_t dispatch_timestamps_[RUNTIME_MAX_WORKER];  // Per-core AICPU dispatch timestamp
    uint32_t core_dispatch_counts_[RUNTIME_MAX_WORKER]; // Per-core total dispatched task counter (for buffer management)

    // ===== Methods =====
    int init(Runtime* runtime);
    int handshake_all_cores(Runtime* runtime);
    void assign_cores_to_threads();
    void reassign_cores_for_all_threads();
    int resolve_and_dispatch_pto2(Runtime* runtime, int thread_idx);
    int shutdown_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores, int core_num);
    int run(Runtime* runtime);
    void deinit();
    void emergency_shutdown();
    void diagnose_stuck_state(Runtime* runtime, int thread_idx, const int* cur_thread_cores,
                              int core_num, Handshake* hank);
};

static AicpuExecutor g_aicpu_executor;

// PTO2 device-mode state (per-core dispatch payloads)
static PTO2DispatchPayload s_pto2_payload_per_core[RUNTIME_MAX_WORKER];

// ===== AicpuExecutor Method Implementations =====

/**
 * Handshake with all cores and discover their types
 * Sets up register addresses for fast dispatch.
 */
int AicpuExecutor::handshake_all_cores(Runtime* runtime) {
    Handshake* all_hanks = (Handshake*)runtime->workers;
    cores_total_num_ = runtime->worker_count;

    // Validate cores_total_num_ before using as array index
    if (cores_total_num_ == 0 || cores_total_num_ > MAX_CORES_PER_THREAD) {
        DEV_ERROR("Invalid cores_total_num %d (expected 1-%d)", cores_total_num_, MAX_CORES_PER_THREAD);
        return -1;
    }

    aic_count_ = 0;
    aiv_count_ = 0;

    DEV_INFO("Handshaking with %d cores", cores_total_num_);

    // Step 1: Write per-core payload addresses and send handshake signal
    // task must be written BEFORE aicpu_ready so AICore sees it after waking up
    for (int i = 0; i < cores_total_num_; i++) {
        all_hanks[i].task = reinterpret_cast<uint64_t>(&s_pto2_payload_per_core[i]);
        all_hanks[i].aicpu_ready = 1;
    }

    // Get platform physical cores count for validation
    uint32_t max_physical_cores_count = platform_get_physical_cores_count();

    // Step 2: Wait for all cores to respond, collect core type and register addresses
    bool handshake_failed = false;
    for (int i = 0; i < cores_total_num_; i++) {
        Handshake* hank = &all_hanks[i];
        while (hank->aicore_done == 0) {
            // Spin wait for core to respond
        }

        CoreType type = hank->core_type;
        uint32_t physical_core_id = hank->physical_core_id;

        // Validate physical_core_id before using as array index
        if (physical_core_id >= max_physical_cores_count) {
            DEV_ERROR("Core %d reported invalid physical_core_id=%u (platform max=%u)",
                      i, physical_core_id, max_physical_cores_count);
            handshake_failed = true;
            continue;
        }

        // Get register address using physical_core_id
        uint64_t* regs = reinterpret_cast<uint64_t*>(regs_);
        uint64_t reg_addr = regs[physical_core_id];

        if (type == CoreType::AIC) {
            aic_cores_[aic_count_].worker_id = i;
            aic_cores_[aic_count_].physical_core_id = physical_core_id;
            aic_cores_[aic_count_].reg_addr = reg_addr;
            aic_cores_[aic_count_].core_type = type;
            aic_count_++;
            DEV_INFO("Core %d: AIC, physical_id=%u, reg_addr=0x%lx", i, physical_core_id, reg_addr);
        } else {
            aiv_cores_[aiv_count_].worker_id = i;
            aiv_cores_[aiv_count_].physical_core_id = physical_core_id;
            aiv_cores_[aiv_count_].reg_addr = reg_addr;
            aiv_cores_[aiv_count_].core_type = type;
            aiv_count_++;
            DEV_INFO("Core %d: AIV, physical_id=%u, reg_addr=0x%lx", i, physical_core_id, reg_addr);
        }

        core_id_to_reg_addr_[i] = reg_addr;

        // Initialize AICore registers (platform-specific)
        if (reg_addr != 0) {
            platform_init_aicore_regs(reg_addr);
        }
    }

    if (handshake_failed) {
        emergency_shutdown();
        return -1;
    }

    DEV_INFO("Core discovery complete: %d AIC, %d AIV", aic_count_, aiv_count_);
    return 0;
}

/**
 * Assign discovered cores to scheduler threads
 * (Aligned with host_build_graph mechanism)
 */
void AicpuExecutor::assign_cores_to_threads() {
    // Determine how many cores each thread gets initially:
    // - Mixed mode: distribute among scheduler threads only
    // - All-orchestrator mode: distribute among all threads (they all transition to schedulers)
    int divisor = (sched_thread_num_ > 0) ? sched_thread_num_ : thread_num_;
    int aic_per_thread = aic_count_ / divisor;
    int aiv_per_thread = aiv_count_ / divisor;

    DEV_INFO("Assigning cores: %d AIC per thread, %d AIV per thread", aic_per_thread, aiv_per_thread);

    for (int32_t i = 0; i < thread_num_; i++) {
        for (int32_t j = 0; j < MAX_CORES_PER_THREAD; j++) {
            executing_task_ids_[i][j] = AICPU_TASK_INVALID;
        }
        trackers_[i].running_count = 0;
        trackers_[i].idle_count = 0;
    }

    for (int t = 0; t < thread_num_; t++) {
        if (sched_thread_num_ > 0 && t >= sched_thread_num_) {
            // Orchestrator thread: no cores
            core_count_per_thread_[t] = 0;
            DEV_INFO("Thread %d: orchestrator (0 cores)", t);
            continue;
        }

        int core_idx = 0;

        // Assign AIC cores
        int aic_start = t * aic_per_thread;
        for (int i = 0; i < aic_per_thread; i++) {
            int worker_id = aic_cores_[aic_start + i].worker_id;
            core_assignments_[t][core_idx++] = worker_id;
            trackers_[t].idle[trackers_[t].idle_count++] = worker_id;
            DEV_INFO("Thread %d: assigned AIC worker_id=%d", t, worker_id);
        }

        // Assign AIV cores
        int aiv_start = t * aiv_per_thread;
        for (int i = 0; i < aiv_per_thread; i++) {
            int worker_id = aiv_cores_[aiv_start + i].worker_id;
            core_assignments_[t][core_idx++] = worker_id;
            trackers_[t].idle[trackers_[t].idle_count++] = worker_id;
            DEV_INFO("Thread %d: assigned AIV worker_id=%d", t, worker_id);
        }

        core_count_per_thread_[t] = core_idx;

        DEV_INFO("Thread %d: total %d cores", t, core_idx);
    }

    thread_cores_num_ = aic_per_thread + aiv_per_thread;
}

/**
 * Reassign all cores evenly across all threads (schedulers + orchestrators).
 * Called by the last orchestrator thread when orchestration completes.
 * Writes into new_core_assignments_ / new_core_count_per_thread_.
 */
void AicpuExecutor::reassign_cores_for_all_threads() {
    // Calculate how many AIC/AIV each thread should have

    DEV_INFO("Reassigning cores for all %d threads: %d AIC, %d AIV", thread_num_, aic_count_, aiv_count_);

    int32_t running_cores[128];
    int32_t running_task_ids[128];
    int32_t idle_cores[128];
    int32_t running_cores_num = 0;
    int32_t idle_cores_num = 0;
    for (int32_t i = 0; i < thread_num_; i++) {
        core_count_per_thread_[i] = 0;
        for (int32_t j = 0; j < trackers_[i].running_count; j++) {
            int32_t core_id = trackers_[i].running[j];
            running_cores[running_cores_num] = core_id;
            running_task_ids[running_cores_num] = executing_task_ids_[i][core_id];
            running_cores_num++;
        }
        for (int32_t j = 0; j < trackers_[i].idle_count; j++) {
            idle_cores[idle_cores_num++] = trackers_[i].idle[j];
        }
        trackers_[i].running_count = 0;
        trackers_[i].idle_count = 0;
        for (int j = 0; j < MAX_CORES_PER_THREAD; j++) {
            executing_task_ids_[i][j] = AICPU_TASK_INVALID;
        }
    }
    for (int32_t i = 0; i < aic_count_; i++) {
        int32_t thread_idx = i % thread_num_;
        int32_t core_id = aic_cores_[i].worker_id;
        core_assignments_[thread_idx][core_count_per_thread_[thread_idx]++] = core_id;
        for (int32_t j = 0; j < running_cores_num; j++) {
            if (core_id == running_cores[j]) {
                trackers_[thread_idx].running[trackers_[thread_idx].running_count++] = core_id;
                executing_task_ids_[thread_idx][core_id] = running_task_ids[j];
                break;
            }
        }
        for (int32_t j = 0; j < idle_cores_num; j++) {
            if (core_id == idle_cores[j]) {
                trackers_[thread_idx].idle[trackers_[thread_idx].idle_count++] = core_id;
                break;
            }
        }
    }
    for (int32_t i = 0; i < aiv_count_; i++) {
        int32_t thread_idx = i % thread_num_;
        int32_t core_id = aiv_cores_[i].worker_id;
        core_assignments_[thread_idx][core_count_per_thread_[thread_idx]++] = core_id;
        for (int32_t j = 0; j < running_cores_num; j++) {
            if (core_id == running_cores[j]) {
                trackers_[thread_idx].running[trackers_[thread_idx].running_count++] = core_id;
                executing_task_ids_[thread_idx][core_id] = running_task_ids[j];
                break;
            }
        }
        for (int32_t j = 0; j < idle_cores_num; j++) {
            if (core_id == idle_cores[j]) {
                trackers_[thread_idx].idle[trackers_[thread_idx].idle_count++] = core_id;
                break;
            }
        }
    }

    // Log final distribution for verification
    DEV_INFO("Core reassignment complete:");
    for (int t = 0; t < thread_num_; t++) {
        DEV_INFO("  Thread %d: %d cores (running=%d, idle=%d)",
                 t, core_count_per_thread_[t],
                 trackers_[t].running_count, trackers_[t].idle_count);
    }
}

int AicpuExecutor::init(Runtime* runtime) {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return 0;
    }

    DEV_INFO("AicpuExecutor: Initializing");

    if (runtime == nullptr) {
        DEV_ERROR("runtime is nullptr");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Read execution parameters from runtime
    thread_num_ = runtime->sche_cpu_num;
    orch_thread_num_ = runtime->orch_thread_num;
    sched_thread_num_ = thread_num_ - orch_thread_num_;
    if (thread_num_ == 0) thread_num_ = 1;

    if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
        DEV_ERROR("Invalid thread_num: %d", thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Initialize core_id_to_reg_addr_ array to 0 before handshake
    for (int i = 0; i < MAX_CORES_PER_THREAD; i++) {
        core_id_to_reg_addr_[i] = 0;
    }

    // Use handshake mechanism to discover cores (aligned with host_build_graph)
    int rc = handshake_all_cores(runtime);
    if (rc != 0) {
        DEV_ERROR("handshake_all_cores failed");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Dynamically assign cores to threads
    assign_cores_to_threads();

    DEV_INFO("Config: threads=%d, cores=%d, cores_per_thread=%d", thread_num_, cores_total_num_, thread_cores_num_);

    // Initialize runtime execution state
    // Task count comes from PTO2 shared memory
    if (runtime->get_pto2_gm_sm_ptr()) {
        auto* header = static_cast<PTO2SharedMemoryHeader*>(runtime->get_pto2_gm_sm_ptr());
        int32_t pto2_count = header->current_task_index.load(std::memory_order_acquire);
        total_tasks_ = pto2_count > 0 ? pto2_count : 0;
    } else {
        total_tasks_ = 0;
    }
    completed_tasks_.store(0, std::memory_order_release);
    // Host orchestration: graph already built, no wait needed. Device orch: Thread 3 will set this.
    bool orch_on_host = runtime->get_orch_built_on_host();
    DEV_INFO("Init: orch_built_on_host=%d", orch_on_host ? 1 : 0);
    orchestrator_done_ = orch_on_host;

    // Initial ready tasks will be populated via scheduler ready queues

    // Reset per-core dispatch timestamps and task counters
    for (int i = 0; i < RUNTIME_MAX_WORKER; i++) {
        dispatch_timestamps_[i] = 0;
        core_dispatch_counts_[i] = 0;
    }

    DEV_INFO("Init: PTO2 mode, task count from shared memory");

    finished_count_.store(0, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    DEV_INFO("AicpuExecutor: Init complete");
    return 0;
}

/**
 * Shutdown AICore - Send exit signal via registers to all AICore kernels
 */
int AicpuExecutor::shutdown_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores, int core_num) {
    (void)runtime;
    if (core_num == 0) return 0;

    DEV_INFO("Thread %d: Shutting down %d cores", thread_idx, core_num);

    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        uint64_t reg_addr = core_id_to_reg_addr_[core_id];
        if (reg_addr != 0) {
            platform_deinit_aicore_regs(reg_addr);
        } else {
            DEV_ERROR("Thread %d: Core %d has invalid register address", thread_idx, core_id);
        }
    }
    DEV_INFO("Thread %d: Shutdown complete", thread_idx);
    return 0;
}

// Build PTO2DispatchPayload from PTO2TaskDescriptor.
static void build_pto2_payload(PTO2DispatchPayload* out, Runtime* runtime,
                               PTO2TaskDescriptor* task, PTO2TaskPayload* task_payload,
                               PTO2TaskDescriptor* task_descriptors,
                               int32_t window_size) {
    (void)task_descriptors;
    (void)window_size;
    out->task_id = task->task_id;
    out->kernel_id = task->kernel_id;
    out->core_type = (task->worker_type == PTO2_WORKER_CUBE) ? CoreType::AIC : CoreType::AIV;
    out->function_bin_addr = runtime->get_function_bin_addr(task->kernel_id);
    int n = 0;

    for (int i = 0; i < task_payload->param_count; i++) {
        if (!task_payload->is_tensor[i]) {
            out->args[n++] = task_payload->scalar_value[i];
        } else {
            out->args[n++] = reinterpret_cast<uint64_t>(&task_payload->tensors[i]);
            task_payload->tensors[i].update_start_offset();
        }
    }

    out->num_args = n;
}

int AicpuExecutor::resolve_and_dispatch_pto2(Runtime* runtime, int thread_idx) {
    const int32_t* cur_thread_cores = core_assignments_[thread_idx];
    int32_t &core_num = core_count_per_thread_[thread_idx];
    int32_t* executing_task_ids = executing_task_ids_[thread_idx];
    CoreStateTracker& tracker = trackers_[thread_idx];
    DEV_INFO("Thread %d: resolve_and_dispatch_pto2 entry", thread_idx);

    void* sm_base = runtime->get_pto2_gm_sm_ptr();
    if (!sm_base) {
        DEV_ERROR("PTO2 dispatch: sm_base is null");
        return -1;
    }
    DEV_INFO("Thread %d: sm_base=%p", thread_idx, sm_base);

    PTO2SharedMemoryHeader* header = static_cast<PTO2SharedMemoryHeader*>(sm_base);
    DEV_INFO("Thread %d: header=%p, task_desc_offset=%d, window_size=%d",
             thread_idx, (void*)header, header->task_descriptors_offset,
             header->task_window_size);

    PTO2TaskDescriptor* task_descriptors = reinterpret_cast<PTO2TaskDescriptor*>(
        static_cast<char*>(sm_base) + header->task_descriptors_offset);
    PTO2TaskPayload* task_payloads = reinterpret_cast<PTO2TaskPayload*>(
        reinterpret_cast<char*>(task_descriptors) +
        PTO2_ALIGN_UP(header->task_window_size * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE));
    DEV_INFO("Thread %d: task_descriptors=%p",
             thread_idx, (void*)task_descriptors);

    int32_t window_size = header->task_window_size;
    if (window_size <= 0 || window_size > PTO2_TASK_WINDOW_SIZE) window_size = PTO2_TASK_WINDOW_SIZE;
    int32_t window_mask = window_size - 1;

    Handshake* hank = static_cast<Handshake*>(runtime->workers);
    DEV_INFO("Thread %d: hank=%p, window_size=%d",
             thread_idx, (void*)hank, window_size);

    // One-time init: assign perf buffers (one thread does it; others wait)
    if (!pto2_init_done_.exchange(true, std::memory_order_acq_rel)) {
        DEV_INFO("Thread %d: doing one-time init", thread_idx);

        // Assign perf buffers to cores early so profiling captures all tasks
        // (total_tasks written to header later when orchestrator completes)
        if (runtime->enable_profiling) {
            perf_aicpu_init_profiling(runtime);
            // Initialize phase profiling for scheduler threads + orchestrator threads
            perf_aicpu_init_phase_profiling(runtime, sched_thread_num_, orch_thread_num_);
            perf_aicpu_set_orch_thread_idx(sched_thread_num_);
        }

        DEV_INFO("Thread %d: one-time init done", thread_idx);
        pto2_init_complete_.store(true, std::memory_order_release);
    } else {
        while (!pto2_init_complete_.load(std::memory_order_acquire)) {
        }
    }

    DEV_INFO("Thread %d: PTO2 dispatch starting with %d cores", thread_idx, core_num);
    int flush_core_num = core_num;  // Preserve original count for perf flush
    int cur_thread_completed = 0;
    int idle_iterations = 0;
    bool profiling_enabled = runtime->enable_profiling;

    // Scheduler profiling counters
#if PTO2_PROFILING
    uint64_t sched_check_cycle = 0;
    uint64_t sched_complete_cycle = 0;
    uint64_t sched_dispatch_cycle = 0;
    uint64_t sched_idle_cycle = 0;
    uint64_t sched_loop_count = 0;
    uint64_t sched_idle_count = 0;
    uint64_t notify_edges_total = 0;
    int32_t  notify_max_degree = 0;
    uint64_t notify_tasks_enqueued = 0;
    uint64_t fanin_edges_total = 0;
    int32_t  fanin_max_degree = 0;
    uint64_t pop_hit = 0;
    uint64_t pop_miss = 0;
    uint32_t phase_complete_count = 0;
    uint32_t phase_dispatch_count = 0;
    uint32_t push_total = 0;
    uint32_t read_reg_count = 0;
    uint64_t total_push_hit_cycle = 0;
    uint64_t read_reg_cycle = 0;
    uint64_t after_complete_cycle = 0;
#endif

    // Deferred release buffer: fanin traversal + CONSUMED check happens here
    // instead of inside on_task_complete, keeping the hot path fast.
    int32_t deferred_release_ids[128];
    int deferred_release_count = 0;

    bool cores_released = false;

    while (true) {
        bool made_progress = false;
#if PTO2_PROFILING
        CYCLE_COUNT_START();
        uint32_t push_hit = 0;
        sched_loop_count++;
        uint64_t _t0_phase = _t0;
#endif
        int32_t task_count = 0;
        if (tracker.running_count == 0) {
            bool orch_done = orchestrator_done_;
            if (orch_done) {
                task_count = total_tasks_;
                if (task_count > 0 && completed_tasks_.load(std::memory_order_relaxed) >= task_count) {
                    completed_.store(true, std::memory_order_release);
                    DEV_INFO("Thread %d: PTO2 completed tasks %d/%d", thread_idx, completed_tasks_.load(std::memory_order_relaxed), task_count);
                    break;
                }
            }
        }

        // Check for core transition request (execute once per thread)
        if (!cores_released && transition_requested_.load(std::memory_order_acquire)) {
            if (!reassigned_.load(std::memory_order_acquire)) {
                wait_reassign_.fetch_add(1, std::memory_order_release);
                while (!reassigned_.load(std::memory_order_acquire)) {
                    if (completed_.load(std::memory_order_acquire)) {
                        break;
                    }
                }
                if (completed_.load(std::memory_order_acquire)) {
                    break;
                }
            }
            cores_released = true;
        }

#if PTO2_PROFILING
        CYCLE_COUNT_LAP(sched_check_cycle);
#endif

        // Process completed and dispatch FIRST to minimize Sched (dispatch→finish) latency.
        // Sched time = finish_ts - dispatch_ts; recording finish_ts here at loop start reduces
        // tail overhead (time from AICore done to AICPU recording finish).

        // Phase 1: Check running cores for completion, process and move to idle
        CYCLE_COUNT_START_X(0);
        int32_t completed_this_turn = 0;
        if (tracker.running_count > 0) {
            for (int i = tracker.running_count - 1; i >= 0; i--) {
                int core_id = tracker.running[i];
                uint64_t reg_addr = core_id_to_reg_addr_[core_id];

                int32_t task_id = executing_task_ids[core_id];
                // Read task_id and state from COND register
                uint64_t reg_val = read_reg(reg_addr, RegId::COND);
                int reg_task_id = EXTRACT_TASK_ID(reg_val);
                int reg_state = EXTRACT_TASK_STATE(reg_val);
                // Only accept FIN state with matching task_id
                bool done = reg_task_id == task_id && reg_state == TASK_FIN_STATE;
                read_reg_count++;
                CYCLE_COUNT_LAP_X(read_reg_cycle, 0);

                if (done) {
                    executing_task_ids[core_id] = AICPU_TASK_INVALID;
                    PTO2CompletionStats cstats = rt->scheduler.on_task_complete(task_id);
                    completed_this_turn++;
                    cur_thread_completed++;
                    made_progress = true;
                    CYCLE_COUNT_LAP_X(total_push_hit_cycle, 0);
#if PTO2_PROFILING
                    notify_edges_total += cstats.fanout_edges;
                    if (cstats.fanout_edges > notify_max_degree) notify_max_degree = cstats.fanout_edges;
                    notify_tasks_enqueued += cstats.tasks_enqueued;
                    phase_complete_count++;
                    push_hit++;
#endif
                    // Queue for deferred fanin release (cold path, after dispatch)
                    if (deferred_release_count < 64) {
                        deferred_release_ids[deferred_release_count++] = task_id;
                    } else {
                        DEV_ALWAYS("Thread %d: release", thread_idx);
                        while (deferred_release_count > 0) {
                            int32_t fe = rt->scheduler.on_task_release(deferred_release_ids[--deferred_release_count]);
#if PTO2_PROFILING
                            fanin_edges_total += fe;
                            if (fe > fanin_max_degree) fanin_max_degree = fe;
#endif
                        }
                    }
                    tracker.move_running_to_idle(i);

                    // Write AICPU dispatch/finish timestamps into the PerfRecord
                    if (profiling_enabled) {
                        PTO2DispatchPayload* payload = &s_pto2_payload_per_core[core_id];
                        Handshake* h = &hank[core_id];
                        uint64_t finish_ts = get_sys_cnt_aicpu();
                        PerfBuffer* perf_buf = (PerfBuffer*)h->perf_records_addr;
                        rmb();
                        uint32_t count = perf_buf->count;
                        if (count > 0) {
                            PerfRecord* record = &perf_buf->records[count - 1];
                            if (record->task_id == static_cast<uint32_t>(payload->task_id)) {
                                perf_aicpu_record_dispatch_and_finish_time(record,
                                                                            dispatch_timestamps_[core_id],
                                                                            finish_ts);
                            }
                        }
                    }

                    DEV_DEBUG("Thread %d: Core %d completed PTO2 task %d", thread_idx, core_id, task_id);
                    // Debug: periodic progress (thread 0 only) to find which task hangs
                    if (thread_idx == 0 && task_count > 0) {
                        int32_t c = completed_tasks_.load(std::memory_order_relaxed);
                        if (c <= PROGRESS_VERBOSE_THRESHOLD || c % PROGRESS_LOG_INTERVAL == 0 || c == task_count) {
                            DEV_ALWAYS("PTO2 progress: completed=%d total=%d last_task_id=%d (%.1f%%)",
                                      c, task_count, task_id, task_count > 0 ? 100.0 * c / task_count : 0.0);
                        }
                    }

                    CYCLE_COUNT_LAP_X(after_complete_cycle, 0);
                } else {
                    CYCLE_COUNT_LAP_X(read_reg_cycle, 0);
                }
            }
            if (completed_this_turn > 0) {
                completed_tasks_.fetch_add(completed_this_turn, std::memory_order_relaxed);
            }
#if PTO2_PROFILING
            CYCLE_COUNT_LAP(sched_complete_cycle);
            push_total += push_hit;
            if (profiling_enabled && phase_complete_count > 0) {
                perf_aicpu_record_phase(
                    thread_idx, AicpuPhaseId::SCHED_COMPLETE, _t0_phase, _t1, sched_loop_count, phase_complete_count);
                _t0_phase = _t1;
                phase_complete_count = 0;
            }

#endif
        } else {
#if PTO2_PROFILING
            CYCLE_COUNT_LAP(sched_idle_cycle);
#endif
        }


        // Phase 2: Dispatch ready tasks to idle cores (register-based dispatch)
        // Pre-check: skip entire dispatch phase when both ready queues are empty.
        // This avoids iterating all idle cores and calling pop() on empty queues.
        // size() is just two relaxed loads — much cheaper than per-core pop attempts.
        if (tracker.idle_count > 0 &&
            (rt->scheduler.ready_queues[PTO2_WORKER_CUBE].size() > 0 ||
             rt->scheduler.ready_queues[PTO2_WORKER_VECTOR].size() > 0)) {
            bool cube_queue_empty = false, vector_queue_empty = false;
            for (int i = tracker.idle_count - 1; i >= 0; i--) {
                // Skip pop if we already know this queue type is empty
                if (cube_queue_empty && vector_queue_empty) {
                    break;
                }
                int core_id = tracker.idle[i];

                Handshake* h = &hank[core_id];
                PTO2WorkerType wt = (h->core_type == CoreType::AIC) ? PTO2_WORKER_CUBE : PTO2_WORKER_VECTOR;
                // Skip cores whose queue type is already known empty
                if ((wt == PTO2_WORKER_CUBE && cube_queue_empty) ||
                    (wt == PTO2_WORKER_VECTOR && vector_queue_empty)) {
#if PTO2_PROFILING
                    pop_miss++;
#endif
                    continue;
                }
                int32_t task_id = rt->scheduler.get_ready_task(wt);
                if (task_id >= 0) {
#if PTO2_PROFILING
                    pop_hit++;
                    phase_dispatch_count++;
#endif
                    PTO2TaskDescriptor* task = &task_descriptors[task_id & window_mask];
                    PTO2TaskPayload* task_pl = &task_payloads[task_id & window_mask];
                    PTO2DispatchPayload* payload = &s_pto2_payload_per_core[core_id];
                    build_pto2_payload(payload, runtime, task, task_pl, task_descriptors, window_size);
                    // Performance profiling: check if buffer needs switching
                    if (profiling_enabled) {
                        dispatch_timestamps_[core_id] = get_sys_cnt_aicpu();
                        if (core_dispatch_counts_[core_id] >= PLATFORM_PROF_BUFFER_SIZE) {
                            perf_aicpu_switch_buffer(runtime, core_id, thread_idx);
                            core_dispatch_counts_[core_id] = 0;
                        }
                        core_dispatch_counts_[core_id]++;
                    }
                    write_reg(core_id_to_reg_addr_[core_id], RegId::DATA_MAIN_BASE, static_cast<uint64_t>(task_id + 1));
                    executing_task_ids[core_id] = task_id;
                    made_progress = true;
                    DEV_DEBUG("Thread %d: Dispatching PTO2 task %d to core %d", thread_idx, task_id, core_id);

                    tracker.move_idle_to_running(i);
                } else {
                    if (wt == PTO2_WORKER_CUBE)
                        cube_queue_empty = true;
                    else
                        vector_queue_empty = true;
#if PTO2_PROFILING
                    pop_miss++;
#endif
                }
            }
#if PTO2_PROFILING
            CYCLE_COUNT_LAP(sched_dispatch_cycle);
            if (profiling_enabled && phase_dispatch_count > 0) {
                perf_aicpu_record_phase(
                    thread_idx, AicpuPhaseId::SCHED_DISPATCH, _t0_phase, _t1, sched_loop_count, phase_dispatch_count);
                _t0_phase = _t1;
                phase_dispatch_count = 0;
            }
#endif
        } else {
#if PTO2_PROFILING
            CYCLE_COUNT_LAP(sched_idle_cycle);
#endif
        }

        if (!made_progress) {
            // Batch deferred fanin releases during idle.
            // Processing all pending releases at once advances the ring faster,
            // freeing heap space for the orchestrator without blocking completion polling.
            while (deferred_release_count > 0) {
                int32_t fe = rt->scheduler.on_task_release(deferred_release_ids[--deferred_release_count]);
#if PTO2_PROFILING
                fanin_edges_total += fe;
                if (fe > fanin_max_degree) fanin_max_degree = fe;
#endif
            }
            idle_iterations++;
            if (false && thread_idx == 0 && task_count > 0 && idle_iterations % STALL_LOG_INTERVAL == 0) {
                int32_t c = completed_tasks_.load(std::memory_order_relaxed);
                DEV_ALWAYS("PTO2 stall: no progress for %d iterations, completed=%d total=%d",
                           idle_iterations, c, task_count);
                // Scan all task slots to find truly stuck tasks using scheduler state
                PTO2SchedulerState* sched = &rt->scheduler;
                int cnt_ready = 0, cnt_waiting = 0, cnt_inflight = 0;
                for (int si = 0; si < task_count; si++) {
                    int32_t slot = si & window_mask;
                    PTO2TaskState st = sched->task_state[slot].load(std::memory_order_relaxed);
                    int32_t rc = sched->fanin_refcount[slot].load(std::memory_order_relaxed);
                    int32_t fi = task_descriptors[slot].fanin_count;
                    int32_t kid = task_descriptors[slot].kernel_id;
                    if (st >= PTO2_TASK_COMPLETED) continue; // Already done
                    if (st == PTO2_TASK_READY || st == PTO2_TASK_RUNNING) { cnt_inflight++; continue; }
                    // PENDING
                    if (rc >= fi) {
                        // Ready (all deps satisfied) but not enqueued — this is the real bug
                        cnt_ready++;
                        if (cnt_ready <= STALL_DUMP_READY_MAX) {
                            DEV_ALWAYS("  STUCK-READY  slot=%d kernel_id=%d refcount=%d fanin=%d state=%d",
                                       slot, kid, rc, fi, (int)st);
                        }
                    } else {
                        cnt_waiting++;
                        if (cnt_waiting <= STALL_DUMP_WAIT_MAX) {
                            DEV_ALWAYS("  STUCK-WAIT   slot=%d kernel_id=%d refcount=%d fanin=%d state=%d",
                                       slot, kid, rc, fi, (int)st);
                        }
                    }
                }
                DEV_ALWAYS("  scan result: stuck_ready=%d stuck_waiting=%d in_flight=%d",
                           cnt_ready, cnt_waiting, cnt_inflight);
                // Log this thread's dispatch state
                DEV_ALWAYS("  thread=%d idle_cores=%d running_cores=%d core_num=%d",
                           thread_idx, tracker.idle_count, tracker.running_count, core_num);
                for (int ci = 0; ci < tracker.running_count && ci < STALL_DUMP_CORE_MAX; ci++) {
                    int cid = tracker.running[ci];
                    Handshake* hh = &hank[cid];
                    int32_t hw_task_id = -1;
                    int32_t hw_kernel = -1;
                    if (hh->task != 0) {
                        const PTO2DispatchPayload* pl = reinterpret_cast<const PTO2DispatchPayload*>((uintptr_t)hh->task);
                        hw_task_id = pl->task_id;
                        hw_kernel  = pl->kernel_id;
                    }
                    uint64_t cond_reg = read_reg(core_id_to_reg_addr_[cid], RegId::COND);
                    DEV_ALWAYS("    core=%d cond=0x%x(state=%d,id=%d) exec_id=%d payload_task=%d kernel=%d",
                               cid, (unsigned)cond_reg,
                               EXTRACT_TASK_STATE(cond_reg), EXTRACT_TASK_ID(cond_reg),
                               executing_task_ids[cid], hw_task_id, hw_kernel);
                }
            }
            if (idle_iterations > MAX_IDLE_ITERATIONS) {
                DEV_ERROR("Thread %d: PTO2 timeout after %d idle iterations", thread_idx, idle_iterations);
                return -1;
            } else {
                SPIN_WAIT_HINT();
            }
#if PTO2_PROFILING
            CYCLE_COUNT_LAP(sched_idle_cycle);
            sched_idle_count++;
#endif
        } else {
            idle_iterations = 0;
        }
    }

    // Drain remaining deferred releases after main loop exits
    for (int dr = 0; dr < deferred_release_count; dr++) {
        int32_t fe = rt->scheduler.on_task_release(deferred_release_ids[dr]);
#if PTO2_PROFILING
        fanin_edges_total += fe;
        if (fe > fanin_max_degree) fanin_max_degree = fe;
#endif
    }
    deferred_release_count = 0;


#if PTO2_PROFILING
    uint64_t sched_total = sched_check_cycle + sched_complete_cycle + sched_dispatch_cycle + sched_idle_cycle;
    if (sched_total == 0) sched_total = 1;  // avoid div-by-zero
    double tasks_per_loop = sched_loop_count > 0 ? (double)cur_thread_completed / sched_loop_count : 0.0;
    DEV_ALWAYS("Thread %d: completed=%d tasks in %.3fus (%llu loops, %.1f tasks/loop)",
        thread_idx,
        cur_thread_completed,
        cycles_to_us(sched_total),
        (unsigned long long)sched_loop_count,
        tasks_per_loop);
    DEV_ALWAYS("Thread %d: --- Phase Breakdown ---", thread_idx);
    double notify_avg = cur_thread_completed > 0 ? (double)notify_edges_total / cur_thread_completed : 0.0;
    double fanin_avg = cur_thread_completed > 0 ? (double)fanin_edges_total / cur_thread_completed : 0.0;
    DEV_ALWAYS("Thread %d:   check:    %.3fus (%.1f%%)",
        thread_idx,
        cycles_to_us(sched_check_cycle),
        sched_check_cycle * 100.0 / sched_total);
    DEV_ALWAYS(
        "Thread %d:   complete:    %.3fus (%.1f%%)  [fanout: edges=%llu, max_degree=%d, avg=%.1f]  [fanin: "
        "edges=%llu, "
        "max_degree=%d, avg=%.1f] [reg count: %u, push_total: %u, avg_push_hit per loop: %.3f] "
        "total push hit cost: %.3fus, read reg cost: %.3fus, after complete cost: %.3fus",
        thread_idx,
        cycles_to_us(sched_complete_cycle),
        sched_complete_cycle * 100.0 / sched_total,
        (unsigned long long)notify_edges_total,
        notify_max_degree,
        notify_avg,
        (unsigned long long)fanin_edges_total,
        fanin_max_degree,
        fanin_avg,
        read_reg_count,
        push_total,
        1.0 * push_total / sched_loop_count,
        cycles_to_us(total_push_hit_cycle),
        cycles_to_us(read_reg_cycle),
        cycles_to_us(after_complete_cycle));
    uint64_t pop_total = pop_hit + pop_miss;
    double pop_hit_rate = pop_total > 0 ? pop_hit * 100.0 / pop_total : 0.0;
    DEV_ALWAYS("Thread %d:   dispatch:    %.3fus (%.1f%%)  [pop: hit=%llu, miss=%llu, hit_rate=%.1f%%]",
        thread_idx,
        cycles_to_us(sched_dispatch_cycle),
        sched_dispatch_cycle * 100.0 / sched_total,
        (unsigned long long)pop_hit,
        (unsigned long long)pop_miss,
        pop_hit_rate);
    DEV_ALWAYS("Thread %d:   idle:        %.3fus (%.1f%%), idle loop count: %llu",
        thread_idx,
        cycles_to_us(sched_idle_cycle),
        sched_idle_cycle * 100.0 / sched_total,
        sched_idle_count);
#endif

    // Flush performance buffers for all cores originally assigned to this thread
    if (profiling_enabled) {
        if (cur_thread_cores != nullptr) {
            perf_aicpu_flush_buffers(runtime, thread_idx, cur_thread_cores, flush_core_num);
        } else {
            // Orchestrator-as-scheduler: build core list from tracker
            int flush_cores[MAX_CORES_PER_THREAD];
            int flush_count = 0;
            for (int i = 0; i < tracker.idle_count; i++) flush_cores[flush_count++] = tracker.idle[i];
            for (int i = 0; i < tracker.running_count; i++) flush_cores[flush_count++] = tracker.running[i];
            perf_aicpu_flush_buffers(runtime, thread_idx, flush_cores, flush_count);
        }
    }

    return cur_thread_completed;
}

int AicpuExecutor::run(Runtime* runtime) {
    int thread_idx = thread_idx_++;

#ifdef __linux__
    {
        // Detect available CPU cores and bind each thread to a distinct core.
        cpu_set_t available;
        CPU_ZERO(&available);
        sched_getaffinity(0, sizeof(available), &available);
        int avail_cores[CPU_SETSIZE];
        int avail_count = 0;
        for (int c = 0; c < CPU_SETSIZE && avail_count < MAX_AICPU_THREADS; c++) {
            if (CPU_ISSET(c, &available)) {
                avail_cores[avail_count++] = c;
            }
        }
        if (thread_idx < avail_count) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(avail_cores[thread_idx], &cpuset);
            sched_setaffinity(0, sizeof(cpuset), &cpuset);
        }
    }
#endif

    DEV_INFO("Thread %d: Start", thread_idx);

    // Orchestrator threads: thread_idx >= sched_thread_num_
    if (thread_idx >= sched_thread_num_) {
        int orch_idx = thread_idx - sched_thread_num_;
        if (runtime->get_orch_built_on_host()) {
            DEV_INFO("Thread %d: Host orchestration mode, no-op (orch_idx=%d)", thread_idx, orch_idx);
        } else {
            // First orchestrator thread (orch_idx == 0): load SO, create runtime
            if (orch_idx == 0) {
                DEV_INFO("Thread %d: Primary orchestrator, loading SO via dlopen", thread_idx);

                const void* so_data = runtime->get_device_orch_so_data();
                size_t so_size = runtime->get_device_orch_so_size();

                if (so_data == nullptr || so_size == 0) {
                    DEV_ERROR("Thread %d: Device orchestration SO not set", thread_idx);
                    return -1;
                }

                // Try multiple paths that may allow execution on AICPU
                char so_path[256];
                bool file_created = false;
                const char* candidate_dirs[] = {
                    "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device",
                    "/usr/lib64",
                    "/lib64",
                    "/var/tmp",
                    "/tmp"
                };
                const int num_candidates = sizeof(candidate_dirs) / sizeof(candidate_dirs[0]);

                for (int i = 0; i < num_candidates && !file_created; i++) {
                    snprintf(so_path, sizeof(so_path), "%s/libdevice_orch_%d.so",
                             candidate_dirs[i], getpid());
                    int fd = open(so_path, O_WRONLY | O_CREAT | O_TRUNC, 0755);
                    if (fd < 0) {
                        DEV_INFO("Thread %d: Cannot create SO at %s (errno=%d), trying next path",
                                 thread_idx, so_path, errno);
                        continue;
                    }
                    ssize_t written = write(fd, so_data, so_size);
                    close(fd);
                    if (written != static_cast<ssize_t>(so_size)) {
                        DEV_INFO("Thread %d: Cannot write SO to %s (errno=%d), trying next path",
                                 thread_idx, so_path, errno);
                        unlink(so_path);
                        continue;
                    }
                    file_created = true;
                    DEV_INFO("Thread %d: Created SO file at %s (%zu bytes)", thread_idx, so_path, so_size);
                }

                if (!file_created) {
                    DEV_ERROR("Thread %d: Failed to create SO file in any candidate path", thread_idx);
                    return -1;
                }

                dlerror();
                void* handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
                const char* dlopen_err = dlerror();
                if (handle == nullptr) {
                    DEV_ERROR("Thread %d: dlopen failed: %s", thread_idx, dlopen_err ? dlopen_err : "unknown");
                    unlink(so_path);
                    return -1;
                }
                DEV_INFO("Thread %d: dlopen succeeded, handle=%p", thread_idx, handle);

                dlerror();
                auto config_func = reinterpret_cast<DeviceOrchestrationConfigFunc>(
                    dlsym(handle, "aicpu_orchestration_config"));

                dlerror();
                DeviceOrchestrationFunc orch_func =
                    reinterpret_cast<DeviceOrchestrationFunc>(dlsym(handle, "aicpu_orchestration_entry"));
                const char* dlsym_error = dlerror();
                if (dlsym_error != nullptr) {
                    DEV_ERROR("Thread %d: dlsym failed: %s", thread_idx, dlsym_error);
                    dlclose(handle);
                    unlink(so_path);
                    return -1;
                }
                if (orch_func == nullptr) {
                    DEV_ERROR("Thread %d: dlsym returned NULL for aicpu_orchestration_entry", thread_idx);
                    dlclose(handle);
                    unlink(so_path);
                    return -1;
                }

                uint64_t* args = runtime->get_orch_args();
                int arg_count = runtime->get_orch_arg_count();
                DEV_INFO("Thread %d: sm_ptr=%p, arg_count=%d", thread_idx, runtime->get_pto2_gm_sm_ptr(), arg_count);
                for (int i = 0; i < arg_count && i < 20; i++) {
                    DEV_INFO("Thread %d: args[%d] = 0x%lx", thread_idx, i, args[i]);
                }

                uint64_t task_window_size = PTO2_TASK_WINDOW_SIZE;
                uint64_t heap_size = PTO2_HEAP_SIZE;
                int expected_arg_count = 0;
                if (config_func) {
                    PTO2OrchestrationConfig cfg = config_func(args, arg_count);
                    expected_arg_count = cfg.expected_arg_count;
                    DEV_INFO("Thread %d: Config: expected_args=%d", thread_idx, expected_arg_count);
                } else {
                    DEV_INFO("Thread %d: No config function, using defaults", thread_idx);
                }

                if (expected_arg_count > 0 && arg_count < expected_arg_count) {
                    DEV_ERROR("Thread %d: arg_count %d < expected %d", thread_idx, arg_count, expected_arg_count);
                    dlclose(handle);
                    unlink(so_path);
                    return -1;
                }

                if (runtime->pto2_task_window_size > 0) {
                    task_window_size = runtime->pto2_task_window_size;
                }
                if (runtime->pto2_heap_size > 0) {
                    heap_size = runtime->pto2_heap_size;
                }
                DEV_INFO("Thread %d: Ring sizes: task_window=%lu, heap=%lu",
                         thread_idx, (unsigned long)task_window_size, (unsigned long)heap_size);

                void* sm_ptr = runtime->get_pto2_gm_sm_ptr();
                void* gm_heap = runtime->get_pto2_gm_heap_ptr();

                uint64_t sm_size = pto2_sm_calculate_size(task_window_size);
                PTO2SharedMemoryHandle* sm_handle =
                    pto2_sm_create_from_buffer(sm_ptr, sm_size, task_window_size,
                                                heap_size);
                if (!sm_handle) {
                    DEV_ERROR("Thread %d: Failed to create shared memory handle", thread_idx);
                    dlclose(handle);
                    unlink(so_path);
                    return -1;
                }

                rt = pto2_runtime_create_from_sm(PTO2_MODE_EXECUTE,
                                                 sm_handle, gm_heap, heap_size, orch_thread_num_);
                if (!rt) {
                    DEV_ERROR("Thread %d: Failed to create PTO2Runtime", thread_idx);
                    pto2_sm_destroy(sm_handle);
                    dlclose(handle);
                    unlink(so_path);
                    return -1;
                }

                // Store shared state for other orchestrator threads
                orch_func_ = orch_func;
                orch_args_cached_ = args;
                orch_arg_count_cached_ = arg_count;
                orch_so_handle_ = handle;
                snprintf(orch_so_path_, sizeof(orch_so_path_), "%s", so_path);

                // All-orchestrator mode: primary orchestrator does one-time init
                if (sched_thread_num_ == 0) {
                    DEV_INFO("Thread %d: All-orchestrator mode, doing one-time init", thread_idx);
                    if (runtime->enable_profiling) {
                        perf_aicpu_init_profiling(runtime);
                        // After transition, all threads become schedulers
                        perf_aicpu_init_phase_profiling(runtime, thread_num_, orch_thread_num_);
                        perf_aicpu_set_orch_thread_idx(0);
                    }
                    pto2_init_done_.store(true, std::memory_order_release);
                    pto2_init_complete_.store(true, std::memory_order_release);
                    DEV_INFO("Thread %d: One-time init done", thread_idx);
                }

                runtime_init_ready_.store(true, std::memory_order_release);
            } else {
                // Non-primary orchestrator: wait for primary to finish setup
                while (!runtime_init_ready_.load(std::memory_order_acquire)) {
                }
            }

            // Wait for scheduler's one-time init to complete
            // (or primary orchestrator's init in all-orchestrator mode)
            while (!pto2_init_complete_.load(std::memory_order_acquire)) {
            }

            // Set thread-local orchestrator index
            pto2_set_orch_thread_idx(orch_idx);

            DEV_ALWAYS("Thread %d: Calling aicpu_orchestration_entry (orch_idx=%d/%d)",
                       thread_idx, orch_idx, orch_thread_num_);
            uint64_t orch_cycle_start = get_sys_cnt_aicpu();
            PTO2_SCOPE(rt) {
                orch_func_(rt, orch_args_cached_, orch_arg_count_cached_,
                           orch_thread_num_, orch_idx);
            }
            uint64_t orch_cycle_end = get_sys_cnt_aicpu();
            DEV_ALWAYS("Thread %d: aicpu_orchestration_entry returned, cost %.3fus (orch_idx=%d)",
                thread_idx, cycles_to_us(orch_cycle_end - orch_cycle_start), orch_idx);

#if PTO2_PROFILING
            if (runtime->enable_profiling) {
                PTO2OrchProfilingData p = pto2_orchestrator_get_profiling();
                uint64_t total = p.sync_cycle + p.alloc_cycle + p.params_cycle +
                                 p.lookup_cycle + p.heap_cycle + p.insert_cycle +
                                 p.fanin_cycle;
                if (total == 0) total = 1;
                DEV_ALWAYS("Thread %d: === Orchestrator Profiling (orch_idx=%d): %lld tasks, total=%.3fus ===",
                         thread_idx, orch_idx, (long long)p.submit_count, cycles_to_us(total));
                DEV_ALWAYS("Thread %d:   sync_tensormap : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.sync_cycle), p.sync_cycle * 100.0 / total);
                DEV_ALWAYS("Thread %d:   task_ring_alloc: %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.alloc_cycle), p.alloc_cycle * 100.0 / total);
                DEV_ALWAYS("Thread %d:   param_copy     : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.params_cycle), p.params_cycle * 100.0 / total);
                DEV_ALWAYS("Thread %d:   lookup+dep     : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.lookup_cycle), p.lookup_cycle * 100.0 / total);
                DEV_ALWAYS("Thread %d:   heap_alloc     : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.heap_cycle), p.heap_cycle * 100.0 / total);
                DEV_ALWAYS("Thread %d:   tensormap_ins  : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.insert_cycle), p.insert_cycle * 100.0 / total);
                DEV_ALWAYS("Thread %d:   fanin+ready    : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.fanin_cycle), p.fanin_cycle * 100.0 / total);
                DEV_ALWAYS("Thread %d:   scope_end      : %.3fus", thread_idx, cycles_to_us(p.scope_end_cycle));
                DEV_ALWAYS("Thread %d:   avg/task       : %.3fus",
                    thread_idx, p.submit_count > 0 ? cycles_to_us(total) / p.submit_count : 0.0);

#if PTO2_TENSORMAP_PROFILING
                PTO2TensorMapProfilingData tp = pto2_tensormap_get_profiling();
                DEV_ALWAYS("Thread %d: === TensorMap Lookup Stats ===", thread_idx);
                DEV_ALWAYS("Thread %d:   lookups        : %llu, inserts: %llu",
                    thread_idx, (unsigned long long)tp.lookup_count, (unsigned long long)tp.insert_count);
                DEV_ALWAYS("Thread %d:   chain walked   : total=%llu, avg=%.1f, max=%d",
                    thread_idx, (unsigned long long)tp.lookup_chain_total,
                    tp.lookup_count > 0 ? (double)tp.lookup_chain_total / tp.lookup_count : 0.0,
                    tp.lookup_chain_max);
                DEV_ALWAYS("Thread %d:   overlap checks : %llu, hits=%llu (%.1f%%)",
                    thread_idx, (unsigned long long)tp.overlap_checks, (unsigned long long)tp.overlap_hits,
                    tp.overlap_checks > 0 ? tp.overlap_hits * 100.0 / tp.overlap_checks : 0.0);
#endif

                if (orch_idx == 0) {
                    AicpuOrchSummary orch_summary = {};
                    orch_summary.start_time = orch_cycle_start;
                    orch_summary.end_time = orch_cycle_end;
                    orch_summary.sync_cycle = p.sync_cycle;
                    orch_summary.alloc_cycle = p.alloc_cycle;
                    orch_summary.params_cycle = p.params_cycle;
                    orch_summary.lookup_cycle = p.lookup_cycle;
                    orch_summary.heap_cycle = p.heap_cycle;
                    orch_summary.insert_cycle = p.insert_cycle;
                    orch_summary.fanin_cycle = p.fanin_cycle;
                    orch_summary.scope_end_cycle = p.scope_end_cycle;
                    orch_summary.submit_count = p.submit_count;
                    perf_aicpu_write_orch_summary(&orch_summary);
                }
            }
#endif

            // Coordinate orchestrator completion
            int finished = orch_finished_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (finished == orch_thread_num_) {
                // Last orchestrator: signal completion and trigger core transition
                pto2_rt_orchestration_done(rt);

                void* sm = runtime->get_pto2_gm_sm_ptr();
                PTO2SharedMemoryHeader* sm_header = static_cast<PTO2SharedMemoryHeader*>(sm);
                int32_t pto2_task_count =
                    sm_header ? sm_header->current_task_index.load(std::memory_order_acquire) : 0;
                DEV_ALWAYS("Thread %d: PTO2 total submitted tasks = %d", thread_idx, pto2_task_count);
                total_tasks_ = pto2_task_count;
                if (runtime->enable_profiling && pto2_task_count > 0) {
                    perf_aicpu_update_total_tasks(runtime, static_cast<uint32_t>(pto2_task_count));
                }
                orchestrator_done_ = true;

                // Compute new core assignments for all threads and initialize donated slots
                DEV_INFO("Thread %d: Set orchestrator_done=true, requesting core transition", thread_idx);
                transition_requested_.store(true, std::memory_order_release);

                // Wait for scheduler threads to acknowledge transition request
                // All-orchestrator mode (sched_thread_num_ == 0): skip the wait
                if (sched_thread_num_ > 0) {
                    while (wait_reassign_.load(std::memory_order_acquire) != sched_thread_num_) {
                        if (completed_.load(std::memory_order_acquire)) {
                            break;
                        }
                    }
                }
                if (!completed_.load(std::memory_order_acquire)) {
                    reassign_cores_for_all_threads();
                    reassigned_.store(true, std::memory_order_release);
                }
            } else {
                // Non-last orchestrator: wait for last orchestrator to finish setup
                while (!transition_requested_.load(std::memory_order_acquire)) {
                }
                while (!reassigned_.load(std::memory_order_acquire)) {
                    if (completed_.load(std::memory_order_acquire)) {
                        break;
                    }
                }
            }
        }
        DEV_INFO("Thread %d: Orchestrator completed (orch_idx=%d)", thread_idx, orch_idx);
    }

    // Scheduler thread
    if (!completed_.load(std::memory_order_acquire)) {
        // Device orchestration: wait for primary orchestrator to initialize SM header
        if (!runtime->get_orch_built_on_host()) {
            while (!runtime_init_ready_.load(std::memory_order_acquire)) {
            }
        }
        DEV_ALWAYS("Thread %d: executed %d tasks", thread_idx, completed_tasks_.load(std::memory_order_acquire));
        always_assert(rt != nullptr);
        DEV_ALWAYS("Thread %d: Starting PTO2 dispatch", thread_idx);
        int completed = resolve_and_dispatch_pto2(runtime, thread_idx);
        DEV_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);

        // After transition, use new core assignments for shutdown
        const int* shutdown_cores = core_assignments_[thread_idx];
        int shutdown_count = core_count_per_thread_[thread_idx];
        auto rc = shutdown_aicore(runtime, thread_idx, shutdown_cores, shutdown_count);
        if (rc != 0) {
            return rc;
        }
    }

    DEV_INFO("Thread %d: Completed", thread_idx);

    // Check if this is the last thread to finish
    int prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        // Destroy PTO2 runtime and close orchestration SO (moved from orchestrator path)
        if (!runtime->get_orch_built_on_host() && orch_so_handle_ != nullptr) {
            pto2_runtime_destroy(rt);
            dlclose(orch_so_handle_);
            unlink(orch_so_path_);
        }
        DEV_INFO("Thread %d: Last thread, marking executor finished", thread_idx);
    }

    return 0;
}

void AicpuExecutor::deinit() {
    // Reset per-core dispatch timestamps and task counters
    for (int i = 0; i < RUNTIME_MAX_WORKER; i++) {
        dispatch_timestamps_[i] = 0;
        core_dispatch_counts_[i] = 0;
    }

    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_ = 0;
    finished_count_.store(0, std::memory_order_release);
    orchestrator_done_ = false;
    pto2_init_done_.store(false, std::memory_order_release);
    pto2_init_complete_.store(false, std::memory_order_release);
    runtime_init_ready_.store(false, std::memory_order_release);

    // Reset core transition state
    transition_requested_.store(false, std::memory_order_release);
    wait_reassign_.store(0, std::memory_order_release);
    reassigned_.store(false, std::memory_order_release);
    completed_.store(false, std::memory_order_release);
    orch_finished_count_.store(0, std::memory_order_release);

    // Reset core discovery state
    aic_count_ = 0;
    aiv_count_ = 0;

    // Reset register-related state
    for (int i = 0; i < MAX_CORES_PER_THREAD; i++) {
        core_id_to_reg_addr_[i] = 0;
    }
    for (int i = 0; i < thread_num_; i++) {
        for (int j = 0; j < MAX_CORES_PER_THREAD; j++) {
            executing_task_ids_[i][j] = AICPU_TASK_INVALID;
        }
    }
    regs_ = 0;

    DEV_INFO("DeInit: Runtime execution state reset");

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    DEV_INFO("DeInit: AicpuExecutor reset complete");
}

void AicpuExecutor::emergency_shutdown() {
    DEV_WARN("Emergency shutdown: sending exit signal to all initialized cores");

    for (int i = 0; i < cores_total_num_; i++) {
        if (core_id_to_reg_addr_[i] != 0) {
            platform_deinit_aicore_regs(core_id_to_reg_addr_[i]);
        }
    }

    DEV_WARN("Emergency shutdown complete");
}

void AicpuExecutor::diagnose_stuck_state(Runtime* runtime, int thread_idx,
                                         const int* cur_thread_cores, int core_num,
                                         Handshake* hank) {
    (void)runtime;
    DEV_ALWAYS("========== DIAGNOSTIC REPORT: Thread %d ==========", thread_idx);

    int completed = completed_tasks_.load(std::memory_order_acquire);
    int total = total_tasks_;
    DEV_ALWAYS("Progress: %d/%d tasks (%.1f%%)",
             completed, total, total > 0 ? completed * 100.0 / total : 0.0);

    uint64_t aic_ready = 0, aiv_ready = 0;
    if (rt) {
        PTO2SchedulerState* sched = &rt->scheduler;
        aic_ready = sched->ready_queues[PTO2_WORKER_CUBE].size();
        aiv_ready = sched->ready_queues[PTO2_WORKER_VECTOR].size();
    }
    DEV_ALWAYS("Ready Queues: AIC=%lu, AIV=%lu", aic_ready, aiv_ready);

    int busy_cores = 0;
    int idle_cores = 0;

    DEV_ALWAYS("Core Status:");
    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* h = &hank[core_id];
        const char* core_type_str = core_type_to_string(h->core_type);

        uint64_t reg_addr = core_id_to_reg_addr_[core_id];
        uint64_t reg_val = read_reg(reg_addr, RegId::COND);
        int reg_task_id = EXTRACT_TASK_ID(reg_val);
        int reg_state = EXTRACT_TASK_STATE(reg_val);
        int task_id = executing_task_ids_[thread_idx][core_id];

        if (reg_state != TASK_FIN_STATE || task_id >= 0) {
            busy_cores++;
            if (task_id >= 0) {
                PTO2DispatchPayload* payload = &s_pto2_payload_per_core[core_id];
                DEV_ALWAYS("  Core %d [%s, BUSY]: COND=0x%lx (reg_task_id=%d, reg_state=%s), executing_task_id=%d, kernel_id=%d",
                        core_id, core_type_str, reg_val, reg_task_id,
                        reg_state == TASK_FIN_STATE ? "FIN" : "ACK",
                        payload->task_id, payload->kernel_id);
            } else {
                DEV_ALWAYS("  Core %d [%s, BUSY]: COND=0x%lx (reg_task_id=%d, reg_state=%s) but task_id not tracked",
                        core_id, core_type_str, reg_val, reg_task_id,
                        reg_state == TASK_FIN_STATE ? "FIN" : "ACK");
            }
        } else {
            idle_cores++;
        }
    }

    DEV_ALWAYS("Summary: %d busy, %d idle", busy_cores, idle_cores);

    // Diagnose deadlock vs livelock
    if (busy_cores == 0 && aic_ready == 0 && aiv_ready == 0 && completed < total) {
        DEV_ALWAYS("*** DEADLOCK DETECTED ***");
        DEV_ALWAYS("All cores idle, no ready tasks, but %d tasks incomplete", total - completed);
        DEV_ALWAYS("Check PTO2 shared memory for task dependency state");
    } else if (busy_cores > 0) {
        DEV_ALWAYS("*** LIVELOCK / HUNG TASK ***");
        DEV_ALWAYS("%d cores executing but no progress", busy_cores);
    }

    DEV_ALWAYS("========== END DIAGNOSTIC ==========");
}

// ===== Public Entry Point =====

/**
 * aicpu_execute - Main AICPU kernel execution entry point
 *
 * This is called by DynTileFwkBackendKernelServer in kernel.cpp.
 * Orchestrates the complete task runtime execution:
 * 1. Initialize executor (thread-safe, first thread only)
 * 2. Wait for initialization to complete
 * 3. Execute tasks on managed cores
 * 4. Cleanup when last thread finishes
 *
 * @param runtime Pointer to Runtime structure
 * @return 0 on success, non-zero on error
 */
extern "C" int aicpu_execute(Runtime* runtime) {
    if (runtime == nullptr) {
        DEV_ERROR("%s", "Invalid argument: null Runtime pointer");
        return -1;
    }

    DEV_INFO("%s", "aicpu_execute: Starting AICPU kernel execution");

    // Get platform register addresses from platform-level global
    g_aicpu_executor.regs_ = get_platform_regs();

    g_aicpu_executor.init(runtime);

    while (!g_aicpu_executor.init_done_.load(std::memory_order_acquire)) {
        if (g_aicpu_executor.init_failed_.load(std::memory_order_acquire)) {
            DEV_ERROR("%s", "aicpu_execute: Initialization failed, aborting execution");
            return -1;
        }
    }

    int rc = g_aicpu_executor.run(runtime);
    if (rc != 0) {
        DEV_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
        return rc;
    }

    // Last thread cleans up
    if (g_aicpu_executor.finished_.load(std::memory_order_acquire)) {
        DEV_INFO("aicpu_execute: Last thread finished, cleaning up");
        g_aicpu_executor.deinit();
    }

    DEV_INFO("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
