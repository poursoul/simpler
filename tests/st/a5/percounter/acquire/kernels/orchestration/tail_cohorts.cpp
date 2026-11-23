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

#include <cstdint>

#include "orchestration_api.h"

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 2};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    constexpr int64_t max_lanes = 320;
    constexpr int64_t stride = 2 * max_lanes;
    const int32_t clusters = rt_available_cluster_count();
    if (clusters < 2 || 3 * clusters > max_lanes) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "invalid topology for tail cohort");
        return;
    }
    for (int64_t i = 0; i < 6; ++i) {
        const bool tail = i % 3 == 2;
        CoreTaskArgs task;
        // Reusing the full tensor gives each wave a true dependency on the previous wave.
        task.add_inout(args.tensor(0).ref());
        task.add_scalar(i * stride);
        task.add_scalar(i * stride + max_lanes);
        task.add_scalar(args.scalar<int64_t>(0));
        task.add_scalar(tail ? int64_t{2} : int64_t{3});
        task.launch_spec.set_block_num(tail ? clusters / 2 : clusters);
        task.launch_spec.set_require_sync_start(true);
        MixedKernels kernels;
        kernels.aic_kernel_id = 0;
        if (!tail) kernels.aiv0_kernel_id = 2;
        kernels.aiv1_kernel_id = tail ? 2 : 3;
        rt_submit_task(kernels, task);
    }
}
}
