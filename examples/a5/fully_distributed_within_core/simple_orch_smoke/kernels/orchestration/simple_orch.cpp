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

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_MARK_AIC 0
#define FUNC_MARK_AIV 1

static PTO_DEVICE_FUNC void submit_mark_task(
    const MixedKernels &mk, const Tensor &input, const Tensor &output, uint64_t n, uint64_t delta, uint64_t i
) {
    SubmitToken tok = rt_presubmit_task(mk);
#if PTO_FDWIC_SHARED_MAP
    if (tok.won) {
        L0TaskArgs args;
        args.add_input(input);
        args.add_inout(output);
        args.add_scalar(n);
        args.add_scalar(delta);
        args.add_scalar(i);
        rt_submit_winner(tok, args);
    } else {
        rt_submit_loser(tok, 0);
    }
#else
    if (tok.won) {
        L0TaskArgs args;
        args.add_input(input);
        args.add_inout(output);
        args.add_scalar(n);
        args.add_scalar(delta);
        args.add_scalar(i);
        rt_submit_winner(tok, args);
    } else {
        L0TaskArgs outputs;
        outputs.add_inout(output);
        rt_submit_loser(tok, outputs);
    }
#endif
}

extern "C" {

__attribute__((visibility("default"), weak)) PTO2OrchestrationConfig
aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default"), weak)) PTO_DEVICE_FUNC void
aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &input = orch_args.tensor(0).ref();
    const Tensor &output = orch_args.tensor(1).ref();
    uint64_t n = orch_args.scalar(0);
    uint64_t delta = orch_args.scalar(1);
    uint64_t mixed = orch_args.scalar(2);

    for (uint64_t i = 0; i < n; i++) {
        MixedKernels mk;
        if (mixed == 2) {
            mk.aiv0_kernel_id = FUNC_MARK_AIV;
            mk.aiv1_kernel_id = FUNC_MARK_AIV;
        } else if (mixed != 0) {
            mk.aic_kernel_id = FUNC_MARK_AIC;
            mk.aiv0_kernel_id = FUNC_MARK_AIV;
        } else if (i % 3 == 0) {
            mk.aic_kernel_id = FUNC_MARK_AIC;
        } else {
            mk.aiv0_kernel_id = FUNC_MARK_AIV;
        }
        submit_mark_task(mk, input, output, n, delta, i);
    }
}

}  // extern "C"
