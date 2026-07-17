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
#define FUNC_MAKE_TEMP_AIV 5

namespace {

PTO_DEVICE_FUNC void submit_fill_output_aic(const Tensor &input, const Tensor &output, uint64_t n, TensorArgType tag) {
    PreparedSubmit submit = rt_prepare_aic_task<0>(FUNC_FILL_OUTPUT_AIC);
    if (!submit.is_winner()) return;
    L0TaskArgs args;
    args.add_input(input);
    if (tag == TensorArgType::OUTPUT) {
        args.add_output(output);
    } else {
        args.add_inout(output);
    }
    args.add_scalar(n);
    submit.commit(args);
}

PTO_DEVICE_FUNC void submit_bump_output_aic(const Tensor &output, uint64_t n) {
    PreparedSubmit submit = rt_prepare_aic_task<0>(FUNC_BUMP_OUTPUT_AIC);
    if (!submit.is_winner()) return;
    L0TaskArgs args;
    args.add_inout(output);
    args.add_scalar(n);
    submit.commit(args);
}

PTO_DEVICE_FUNC TaskOutputTensors submit_make_temp_aic(const Tensor &input, TensorCreateInfo &temp_ci, uint64_t n) {
    PreparedSubmit submit = rt_prepare_aic_task<1>(FUNC_MAKE_TEMP_AIC);
    if (submit.is_winner()) {
        L0TaskArgs args;
        args.add_input(input);
        args.add_output(temp_ci);
        args.add_scalar(n);
        submit.commit(args);
    }
    return submit.outputs();
}

PTO_DEVICE_FUNC TaskOutputTensors submit_make_temp_aiv(const Tensor &input, TensorCreateInfo &temp_ci, uint64_t n) {
    PreparedSubmit submit = rt_prepare_aiv_task<1>(FUNC_MAKE_TEMP_AIV);
    if (submit.is_winner()) {
        L0TaskArgs args;
        args.add_input(input);
        args.add_output(temp_ci);
        args.add_scalar(n);
        submit.commit(args);
    }
    return submit.outputs();
}

PTO_DEVICE_FUNC void submit_consume_temp_aic(SymbolicTensor temp_symbol, const Tensor &output, uint64_t n) {
    PreparedSubmit submit = rt_prepare_aic_task<0>(FUNC_CONSUME_TEMP_AIC);
    if (!submit.is_winner()) return;
    L0TaskArgs args;
    args.add_input(temp_symbol);
    args.add_output(output);
    args.add_scalar(n);
    submit.commit(args);
}

PTO_DEVICE_FUNC void
submit_consume_temp_mixed(const MixedKernels &mixed, SymbolicTensor temp_symbol, const Tensor &output, uint64_t n) {
    PreparedSubmit submit = rt_prepare_task<0>(mixed);
    if (!submit.is_winner()) return;
    L0TaskArgs args;
    args.add_input(temp_symbol);
    args.add_output(output);
    args.add_scalar(n);
    submit.commit(args);
}

}  // namespace

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
        submit_fill_output_aic(input, output, n, TensorArgType::INOUT);
        submit_bump_output_aic(output, n);
        return;
    }

    if (mode == 2) {
        const uint32_t sub_n = static_cast<uint32_t>(n / 2);
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t sub_offset[1] = {static_cast<uint32_t>(n / 4)};
        Tensor sub_output = Tensor::view(output, sub_shape, sub_offset);
        submit_fill_output_aic(input, output, n, TensorArgType::INOUT);
        submit_bump_output_aic(sub_output, sub_n);
        return;
    }

    if (mode == 3) {
        submit_fill_output_aic(input, output, n, TensorArgType::OUTPUT);
        submit_bump_output_aic(output, n);
        return;
    }

    if (mode == 6) {
        const uint32_t shape[1] = {static_cast<uint32_t>(n)};
        TensorCreateInfo temp_ci(shape, 1, DataType::FLOAT32);
        for (int32_t i = 0; i < 96; i++) {
            submit_make_temp_aic(input, temp_ci, n);
        }
        submit_fill_output_aic(input, output, n, TensorArgType::OUTPUT);
        return;
    }

    if (mode == 7) {
        const uint32_t shape[1] = {static_cast<uint32_t>(n)};
        TensorCreateInfo temp_ci(shape, 1, DataType::FLOAT32);
        for (int32_t i = 0; i < 16500; i++) {
            submit_make_temp_aic(input, temp_ci, n);
        }
        submit_fill_output_aic(input, output, n, TensorArgType::OUTPUT);
        return;
    }

    if (mode == 8) {
        const uint32_t shape[1] = {static_cast<uint32_t>(n)};
        TensorCreateInfo temp_ci(shape, 1, DataType::FLOAT32);
        TaskOutputTensors temp = submit_make_temp_aiv(input, temp_ci, n);
        const SymbolicTensor temp_symbol = temp.get_symbol(0);
        submit_consume_temp_aic(temp_symbol, output, n);
        return;
    }

    const bool use_mixed = mode == 4 || mode == 5;
    const uint32_t shape[1] = {static_cast<uint32_t>(n)};
    TensorCreateInfo temp_ci(shape, 1, DataType::FLOAT32);

    TaskOutputTensors temp = submit_make_temp_aic(input, temp_ci, n);
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
        submit_consume_temp_mixed(mixed, temp_symbol, output, n);
        return;
    }

    submit_consume_temp_aic(temp_symbol, output, n);
}

}  // extern "C"
