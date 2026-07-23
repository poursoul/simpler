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

#define FUNC_PRODUCE_AIC 0
#define FUNC_PRODUCE_AIV 1
#define FUNC_CONSUME_AIC 2
#define FUNC_CONSUME_AIV 3
#define FUNC_PRODUCE_PAIR_AIC 4
#define FUNC_PRODUCE_PAIR_AIV 5

#if PTO_FDWIC_SHARED_MAP
using SymbolOutputs = SharedTaskOutputs;
#else
using SymbolOutputs = TaskOutputTensors;
#endif

static PTO_DEVICE_FUNC SymbolOutputs submit_producer(
    const MixedKernels &mk, const Tensor &input, const uint32_t shape[1], uint64_t n, uint32_t output_count
) {
    TensorCreateInfo tmp_ci(shape, 1, DataType::FLOAT32);
    L0TaskArgs args;
    args.add_input(input);
    args.add_output(tmp_ci);
    if (output_count == 2) args.add_output(tmp_ci);
    args.add_scalar(n);
#if PTO_FDWIC_SHARED_MAP
    SubmitToken tok = rt_presubmit_task(mk);
    if (tok.won) return rt_submit_winner(tok, args);
    return rt_submit_loser(tok, output_count);
#else
    return rt_submit_task(mk, args);
#endif
}

static PTO_DEVICE_FUNC void submit_consumer(
    const MixedKernels &mk, const Tensor &input, const SymbolOutputs &producer, uint32_t producer_slot,
    const Tensor &output, uint64_t n
) {
    L0TaskArgs args;
    args.add_input(input);
#if PTO_FDWIC_SHARED_MAP
    args.add_input(producer.output_ref(producer_slot));
#else
    args.add_input(producer.get_ref(producer_slot));
#endif
    args.add_inout(output);
    args.add_scalar(n);
#if PTO_FDWIC_SHARED_MAP
    SubmitToken tok = rt_presubmit_task(mk);
    if (tok.won) {
        rt_submit_winner(tok, args);
    } else {
        rt_submit_loser(tok, 0);
    }
#else
    rt_submit_task(mk, args);
#endif
}

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
    const uint32_t shape[1] = {static_cast<uint32_t>(n)};
    uint64_t effective_mode = mode;

    if (mode == 4) {
        MixedKernels pad_aic;
        pad_aic.aic_kernel_id = FUNC_PRODUCE_AIC;
        (void)submit_producer(pad_aic, input, shape, n, 1);
        MixedKernels pad_aiv;
        pad_aiv.aiv0_kernel_id = FUNC_PRODUCE_AIV;
        (void)submit_producer(pad_aiv, input, shape, n, 1);
        effective_mode = 2;
    }

    MixedKernels producer_mk;
    MixedKernels consumer_mk;
    if (effective_mode == 0) {
        producer_mk.aic_kernel_id = FUNC_PRODUCE_AIC;
        consumer_mk.aiv0_kernel_id = FUNC_CONSUME_AIV;
    } else if (effective_mode == 1) {
        producer_mk.aiv0_kernel_id = FUNC_PRODUCE_AIV;
        consumer_mk.aic_kernel_id = FUNC_CONSUME_AIC;
    } else if (effective_mode == 2) {
        producer_mk.aic_kernel_id = FUNC_PRODUCE_PAIR_AIC;
        consumer_mk.aiv0_kernel_id = FUNC_CONSUME_AIV;
    } else if (effective_mode == 5) {
        producer_mk.aic_kernel_id = FUNC_PRODUCE_PAIR_AIC;
        consumer_mk.aiv0_kernel_id = FUNC_CONSUME_AIV;
        consumer_mk.aiv1_kernel_id = FUNC_CONSUME_AIV;
    } else if (effective_mode == 6) {
        producer_mk.aiv0_kernel_id = FUNC_PRODUCE_PAIR_AIV;
        producer_mk.aiv1_kernel_id = FUNC_PRODUCE_PAIR_AIV;
        consumer_mk.aic_kernel_id = FUNC_CONSUME_AIC;
    } else {
        producer_mk.aiv0_kernel_id = FUNC_PRODUCE_PAIR_AIV;
        consumer_mk.aic_kernel_id = FUNC_CONSUME_AIC;
    }

    const uint32_t output_count = effective_mode >= 2 ? 2 : 1;
    SymbolOutputs produced = submit_producer(producer_mk, input, shape, n, output_count);
    const uint32_t producer_slot = effective_mode >= 2 ? 1 : 0;
    submit_consumer(consumer_mk, input, produced, producer_slot, output, n);
}

}  // extern "C"
