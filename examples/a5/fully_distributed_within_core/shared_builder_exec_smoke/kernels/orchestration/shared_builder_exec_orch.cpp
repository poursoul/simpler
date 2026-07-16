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

#define FUNC_MAKE_TEMP_AIC 0
#define FUNC_CONSUME_TEMP_AIC 1
#define FUNC_FILL_OUTPUT_AIC 2
#define FUNC_BUMP_OUTPUT_AIC 3
#define FUNC_CONSUME_TEMP_AIV 4

extern "C" {

__attribute__((visibility("default"), weak)) PTO2OrchestrationConfig
aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"), weak)) PTO_DEVICE_FUNC void
aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &input = orch_args.tensor(0).ref();
    const Tensor &output = orch_args.tensor(1).ref();
    const uint64_t n = orch_args.scalar(0);
    const uint64_t mode = orch_args.scalar(1);

    if (mode == 1) {
        rt_submit_aic_task<0>(FUNC_FILL_OUTPUT_AIC, [&](SubmitBuilder &builder) {
            builder.add_input([&]() -> const Tensor & {
                return input;
            });
            builder.add_inout([&]() -> const Tensor & {
                return output;
            });
            builder.add_scalar([&]() -> uint64_t {
                return n;
            });
        });
        rt_submit_aic_task<0>(FUNC_BUMP_OUTPUT_AIC, [&](SubmitBuilder &builder) {
            builder.add_inout([&]() -> const Tensor & {
                return output;
            });
            builder.add_scalar([&]() -> uint64_t {
                return n;
            });
        });
        return;
    }

    if (mode == 2) {
        const uint32_t sub_n = static_cast<uint32_t>(n / 2);
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t sub_offset[1] = {static_cast<uint32_t>(n / 4)};
        Tensor sub_output = Tensor::view(output, sub_shape, sub_offset);
        rt_submit_aic_task<0>(FUNC_FILL_OUTPUT_AIC, [&](SubmitBuilder &builder) {
            builder.add_input([&]() -> const Tensor & {
                return input;
            });
            builder.add_inout([&]() -> const Tensor & {
                return output;
            });
            builder.add_scalar([&]() -> uint64_t {
                return n;
            });
        });
        rt_submit_aic_task<0>(FUNC_BUMP_OUTPUT_AIC, [&](SubmitBuilder &builder) {
            builder.add_inout([&]() -> const Tensor & {
                return sub_output;
            });
            builder.add_scalar([&]() -> uint64_t {
                return sub_n;
            });
        });
        return;
    }

    if (mode == 3) {
        rt_submit_aic_task<0>(FUNC_FILL_OUTPUT_AIC, [&](SubmitBuilder &builder) {
            builder.add_input([&]() -> const Tensor & {
                return input;
            });
            builder.add_output([&]() -> const Tensor & {
                return output;
            });
            builder.add_scalar([&]() -> uint64_t {
                return n;
            });
        });
        rt_submit_aic_task<0>(FUNC_BUMP_OUTPUT_AIC, [&](SubmitBuilder &builder) {
            builder.add_inout([&]() -> const Tensor & {
                return output;
            });
            builder.add_scalar([&]() -> uint64_t {
                return n;
            });
        });
        return;
    }

    if (mode == 6) {
        const uint32_t shape[1] = {static_cast<uint32_t>(n)};
        TensorCreateInfo temp_ci(shape, 1, DataType::FLOAT32);
        for (int32_t i = 0; i < 96; i++) {
            rt_submit_aic_task<1>(FUNC_MAKE_TEMP_AIC, [&](SubmitBuilder &builder) {
                builder.add_input([&]() -> const Tensor & {
                    return input;
                });
                builder.add_output([&]() -> TensorCreateInfo & {
                    return temp_ci;
                });
                builder.add_scalar([&]() -> uint64_t {
                    return n;
                });
            });
        }
        rt_submit_aic_task<0>(FUNC_FILL_OUTPUT_AIC, [&](SubmitBuilder &builder) {
            builder.add_input([&]() -> const Tensor & {
                return input;
            });
            builder.add_output([&]() -> const Tensor & {
                return output;
            });
            builder.add_scalar([&]() -> uint64_t {
                return n;
            });
        });
        return;
    }

    const bool use_mixed = mode == 4 || mode == 5;
    const uint32_t shape[1] = {static_cast<uint32_t>(n)};
    TensorCreateInfo temp_ci(shape, 1, DataType::FLOAT32);

    TaskOutputTensors temp = rt_submit_aic_task<1>(FUNC_MAKE_TEMP_AIC, [&](SubmitBuilder &builder) {
        builder.add_input([&]() -> const Tensor & {
            return input;
        });
        builder.add_output([&]() -> TensorCreateInfo & {
            return temp_ci;
        });
        builder.add_scalar([&]() -> uint64_t {
            return n;
        });
    });
    const SymbolicTensor temp_symbol = temp.get_symbol(0);

    if (use_mixed) {
        MixedKernels mixed;
        if (mode == 4) {
            mixed.aic_kernel_id = FUNC_CONSUME_TEMP_AIC;
            mixed.aiv0_kernel_id = FUNC_CONSUME_TEMP_AIV;
        } else {
            mixed.aiv0_kernel_id = FUNC_CONSUME_TEMP_AIV;
            mixed.aiv1_kernel_id = FUNC_CONSUME_TEMP_AIV;
        }
        rt_submit_task<0>(mixed, [&](SubmitBuilder &builder) {
            builder.add_input([&]() -> SymbolicTensor {
                return temp_symbol;
            });
            builder.add_output([&]() -> const Tensor & {
                return output;
            });
            builder.add_scalar([&]() -> uint64_t {
                return n;
            });
        });
        return;
    }

    rt_submit_aic_task<0>(FUNC_CONSUME_TEMP_AIC, [&](SubmitBuilder &builder) {
        builder.add_input([&]() -> SymbolicTensor {
            return temp_symbol;
        });
        builder.add_output([&]() -> const Tensor & {
            return output;
        });
        builder.add_scalar([&]() -> uint64_t {
            return n;
        });
    });
}

}  // extern "C"
