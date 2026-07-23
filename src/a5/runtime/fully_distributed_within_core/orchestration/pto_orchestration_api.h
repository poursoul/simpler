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
/**
 * PTO Orchestration API - direct distributed-engine header.
 *
 * fully_distributed_within_core replays orchestration on AICore. The submit
 * helpers therefore call dist_engine_api.h symbols directly on both CPU sim and
 * CCEC onboard; this runtime has no submit function-pointer indirection.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dist_engine/dist_engine_api.h"  // NOLINT(build/include_subdir)
#include "pto_runtime2_types.h"           // PTO2_ERROR_*
#include "pto_submit_types.h"             // MixedKernels, INVALID_KERNEL_ID
#include "pto_types.h"                    // Arg, SharedTaskOutputs, TaskOutputTensors, TensorArgType
#include "task_args.h"                    // ChipStorageTaskArgs, Tensor
#include "tensor.h"                       // Tensor, TensorCreateInfo

struct PTO2Runtime;

#if PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC inline uint32_t rt_count_outputs(const L0TaskArgs &args) {
    uint32_t count = 0;
    for (int32_t i = 0; i < args.tensor_count(); i++)
        if (args.tag(i) == TensorArgType::OUTPUT) count++;
    return count;
}

PTO_DEVICE_FUNC inline SharedTaskOutputs rt_output_refs(int32_t task_id, uint32_t output_count) {
    if (task_id < 0) return SharedTaskOutputs{};
    SharedTaskOutputs outputs;
    outputs.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(task_id)));
    for (uint32_t i = 0; i < output_count; i++)
        outputs.add_output_ref(task_id, static_cast<int16_t>(i));
    return outputs;
}

PTO_DEVICE_FUNC inline SharedTaskOutputs alloc_tensors(const L0TaskArgs &args) {
    if (dist_is_fatal_query()) return SharedTaskOutputs{};
    const int32_t task_id = dist_alloc_outputs_impl(nullptr, args);
    return rt_output_refs(task_id, rt_count_outputs(args));
}
#else
PTO_DEVICE_FUNC inline TaskOutputTensors alloc_tensors(const L0TaskArgs &args) {
    if (dist_is_fatal_query()) return TaskOutputTensors{};
    return dist_alloc_tensors(nullptr, args);
}
#endif

PTO_DEVICE_FUNC inline SubmitToken rt_presubmit_task(const MixedKernels &mixed_kernels) {
    if (dist_is_fatal_query()) return SubmitToken{};
    return dist_presubmit_task_impl(nullptr, mixed_kernels);
}

PTO_DEVICE_FUNC inline SubmitToken rt_presubmit_aic_task(int32_t kernel_id) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt_presubmit_task(mk);
}

PTO_DEVICE_FUNC inline SubmitToken rt_presubmit_aiv_task(int32_t kernel_id) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_presubmit_task(mk);
}

#if PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC inline SubmitToken
rt_presubmit_task_with_region_intent(const MixedKernels &mixed_kernels, const L0TaskArgs &args) {
    if (dist_is_fatal_query()) return SubmitToken{};
    return dist_presubmit_task_with_region_intent_impl(nullptr, mixed_kernels, args);
}

PTO_DEVICE_FUNC inline SubmitToken rt_presubmit_aic_task_with_region_intent(int32_t kernel_id, const L0TaskArgs &args) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt_presubmit_task_with_region_intent(mk, args);
}

PTO_DEVICE_FUNC inline SubmitToken rt_presubmit_aiv_task_with_region_intent(int32_t kernel_id, const L0TaskArgs &args) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_presubmit_task_with_region_intent(mk, args);
}

PTO_DEVICE_FUNC inline SharedTaskOutputs rt_submit_winner(const SubmitToken &tok, const L0TaskArgs &args) {
    if (dist_is_fatal_query()) return SharedTaskOutputs{};
    dist_submit_winner_impl(nullptr, tok, args);
    return rt_output_refs(tok.task_id, rt_count_outputs(args));
}

PTO_DEVICE_FUNC inline SharedTaskOutputs rt_submit_loser(const SubmitToken &tok, uint32_t output_count) {
    if (dist_is_fatal_query()) return SharedTaskOutputs{};
    return rt_output_refs(tok.task_id, output_count);
}
#else
PTO_DEVICE_FUNC inline TaskOutputTensors rt_submit_winner(const SubmitToken &tok, const L0TaskArgs &args) {
    if (dist_is_fatal_query()) return TaskOutputTensors{};
    return dist_submit_winner_impl(nullptr, tok, args);
}

PTO_DEVICE_FUNC inline TaskOutputTensors rt_submit_loser(const SubmitToken &tok, const L0TaskArgs &outputs) {
    if (dist_is_fatal_query()) return TaskOutputTensors{};
    return dist_submit_loser_impl(nullptr, tok, outputs);
}

PTO_DEVICE_FUNC inline TaskOutputTensors rt_submit_task(const MixedKernels &mixed_kernels, const L0TaskArgs &args) {
    if (dist_is_fatal_query()) return TaskOutputTensors{};
    SubmitToken tok = rt_presubmit_task(mixed_kernels);
    if (tok.won) return rt_submit_winner(tok, args);
    return rt_submit_loser(tok, args);
}

PTO_DEVICE_FUNC inline TaskOutputTensors rt_submit_aic_task(int32_t kernel_id, const L0TaskArgs &args) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}

PTO_DEVICE_FUNC inline TaskOutputTensors rt_submit_aiv_task(int32_t kernel_id, const L0TaskArgs &args) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}
#endif

PTO_DEVICE_FUNC inline TaskOutputTensors rt_submit_dummy_task(const L0TaskArgs &args) {
    if (dist_is_fatal_query()) return TaskOutputTensors{};
    return dist_submit_dummy_impl(nullptr, args);
}

PTO_DEVICE_FUNC inline void rt_scope_begin(PTO2ScopeMode /*mode*/ = PTO2ScopeMode::AUTO) {
    if (dist_is_fatal_query()) return;
    dist_scope_begin_impl(nullptr);
}

PTO_DEVICE_FUNC inline void rt_scope_end() {
    if (dist_is_fatal_query()) return;
    dist_scope_end_impl(nullptr);
}

PTO_DEVICE_FUNC inline void rt_orchestration_done() { dist_orchestration_done_impl(nullptr); }

PTO_DEVICE_FUNC inline bool rt_is_fatal() { return dist_is_fatal_query(); }

#define rt_report_fatal(code, fmt, ...)                     \
    do {                                                    \
        dist_report_fatal_msg((code), __FUNCTION__, (fmt)); \
    } while (0)

#define LOG_ERROR(fmt, ...) dist_log_error_msg(__FUNCTION__, (fmt))
#define LOG_WARN(fmt, ...) dist_log_warn_msg(__FUNCTION__, (fmt))
#define LOG_DEBUG(fmt, ...) dist_log_debug_msg(__FUNCTION__, (fmt))
#define LOG_INFO_V0(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 0, (fmt))
#define LOG_INFO_V1(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 1, (fmt))
#define LOG_INFO_V2(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 2, (fmt))
#define LOG_INFO_V3(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 3, (fmt))
#define LOG_INFO_V4(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 4, (fmt))
#define LOG_INFO_V5(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 5, (fmt))
#define LOG_INFO_V6(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 6, (fmt))
#define LOG_INFO_V7(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 7, (fmt))
#define LOG_INFO_V8(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 8, (fmt))
#define LOG_INFO_V9(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 9, (fmt))

template <typename T = uint64_t>
PTO_DEVICE_FUNC inline T get_tensor_data(const Tensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    if (dist_is_fatal_query()) return from_u64<T>(0);
    return from_u64<T>(dist_get_tensor_data_impl(nullptr, tensor, ndims, indices));
}

template <typename T = uint64_t>
PTO_DEVICE_FUNC inline void set_tensor_data(const Tensor &tensor, uint32_t ndims, const uint32_t indices[], T value) {
    if (dist_is_fatal_query()) return;
    dist_set_tensor_data_impl(nullptr, tensor, ndims, indices, to_u64(value));
}

class PTO2ScopeGuard {
public:
    PTO_DEVICE_FUNC explicit PTO2ScopeGuard(
        PTO2ScopeMode mode = PTO2ScopeMode::AUTO, const char *file = __builtin_FILE(), int line = __builtin_LINE()
    ) {
        (void)mode;
        if (dist_is_fatal_query()) return;
        dist_scope_set_site_impl(file, line);
        dist_scope_begin_impl(nullptr);
    }

    PTO_DEVICE_FUNC ~PTO2ScopeGuard() {
        if (dist_is_fatal_query()) return;
        dist_scope_end_impl(nullptr);
    }
};

#define _PTO2_CONCATENATE_IMPL(x, y) x##y
#define _PTO2_CONCATENATE(x, y) _PTO2_CONCATENATE_IMPL(x, y)

#define PTO2_SCOPE_GUARD(...) \
    [[maybe_unused]] PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__) { __VA_ARGS__ }

#define PTO2_SCOPE(...) if (PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__){__VA_ARGS__}; true)

#ifndef PTO2_ORCHESTRATION_CONFIG_DEFINED
#define PTO2_ORCHESTRATION_CONFIG_DEFINED
struct PTO2OrchestrationConfig {
    int expected_arg_count;
};
#endif

#include "pto_arg_with_deps.h"
