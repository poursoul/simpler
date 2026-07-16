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

#define FUNC_DUMMY_AIC 0
#define FUNC_DUMMY_AIV 1

extern "C" {

__attribute__((visibility("default"), weak)) PTO2OrchestrationConfig
aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 2,
    };
}

__attribute__((visibility("default"), weak)) PTO_DEVICE_FUNC void
aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &sentinel = orch_args.tensor(0).ref();
    const uint64_t n = orch_args.scalar(0);
    uint32_t shape[1] = {1};

    for (uint64_t i = 0; i < n; i++) {
        TensorCreateInfo ci(shape, 1, DataType::FLOAT32);
        TaskOutputTensors produced = rt_submit_aic_task<1>(FUNC_DUMMY_AIC, [&](SubmitBuilder &builder) {
            builder.add_no_dep([&]() -> const Tensor & {
                return sentinel;
            });
            builder.add_output([&]() -> TensorCreateInfo & {
                return ci;
            });
            builder.add_scalar([&]() -> uint64_t {
                return i;
            });
        });
        const SymbolicTensor produced_symbol = produced.get_symbol(0);

        if ((i & 1) == 0) {
            rt_submit_aic_task<0>(FUNC_DUMMY_AIC, [&](SubmitBuilder &builder) {
                builder.add_input([&]() -> SymbolicTensor {
                    return produced_symbol;
                });
                builder.add_scalar([&]() -> uint64_t {
                    return i;
                });
            });
        } else {
            MixedKernels mixed;
            mixed.aic_kernel_id = FUNC_DUMMY_AIC;
            mixed.aiv0_kernel_id = FUNC_DUMMY_AIV;
            rt_submit_task<0>(mixed, [&](SubmitBuilder &builder) {
                builder.add_input([&]() -> SymbolicTensor {
                    return produced_symbol;
                });
                builder.add_scalar([&]() -> uint64_t {
                    return i;
                });
            });
        }
    }
}

}  // extern "C"
