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

#define FUNC_FILL_ALLOC_AIC 0
#define FUNC_MAKE_LEFT_AIC 1
#define FUNC_MAKE_RIGHT_AIV 2
#define FUNC_FANIN_AIV 3
#define FUNC_BUMP_INOUT_AIV 4
#define FUNC_BUMP_OFFSET_AIV 5
#define FUNC_INSPECT_VIEW_AIV 6
#define FUNC_INSPECT_ALLOC_AIC 7
#define FUNC_DCCI_ATOMIC_CLOBBER_AIC 8
#define FUNC_DCCI_ATOMIC_CLOBBER_AIV 9

#if !defined(__CCE_AICORE__) && !defined(dcci)
#define dcci(...) \
    do {          \
    } while (0)
#endif
#if !defined(__CCE_AICORE__) && !defined(SINGLE_CACHE_LINE)
#define SINGLE_CACHE_LINE 0
#endif
#if !defined(__CCE_AICORE__) && !defined(CACHELINE_OUT)
#define CACHELINE_OUT 0
#endif

template <typename TensorT>
PTO_DEVICE_FUNC void
write_descriptor_checks(TensorT &tensor, __gm__ float *out, uint32_t base, uint64_t n, const Tensor &reference) {
    out[base + 0] = tensor.buffer.addr != 0 ? 1.0f : 0.0f;
    out[base + 1] = tensor.buffer.size == n * sizeof(float) ? 1.0f : 0.0f;
    out[base + 2] = tensor.start_offset == 0 ? 1.0f : 0.0f;
    out[base + 3] = tensor.ndims == 1 ? 1.0f : 0.0f;
    out[base + 4] = tensor.dtype == DataType::FLOAT32 ? 1.0f : 0.0f;
    out[base + 5] = tensor.is_contiguous ? 1.0f : 0.0f;
    out[base + 6] = tensor.shapes[0] == n ? 1.0f : 0.0f;
    out[base + 7] = tensor.strides[0] == 1 ? 1.0f : 0.0f;
    out[base + 8] = tensor.extent_elem_cache == n ? 1.0f : 0.0f;
    out[base + 9] = tensor.buffer.addr != reference.buffer.addr ? 1.0f : 0.0f;
}

#if PTO_FDWIC_SHARED_MAP
using SubmitOutputs = SharedTaskOutputs;
using OutputHandle = FdwicOutputRef;
using ViewHandle = FdwicOutputRef;

PTO_DEVICE_FUNC inline OutputHandle output_handle(const SubmitOutputs &outputs, uint32_t slot) {
    return outputs.output_ref(slot);
}

PTO_DEVICE_FUNC inline ViewHandle
output_view(OutputHandle source, const uint32_t view_shapes[], const uint32_t view_offsets[], uint32_t ndims) {
    return source.view(view_shapes, view_offsets, ndims);
}

PTO_DEVICE_FUNC inline bool args_need_region_intent(const L0TaskArgs &args) {
    const int32_t tensor_count = args.tensor_count();
    for (int32_t i = 0; i < tensor_count; i++) {
        const TensorArgType tag = args.tag(i);
        if (tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING) return true;
    }
    return false;
}

PTO_DEVICE_FUNC inline SubmitOutputs
submit_aic_task(int32_t func_id, const L0TaskArgs &args, uint32_t output_count = 0) {
    SubmitToken tok = args_need_region_intent(args) ? rt_presubmit_aic_task_with_region_intent(func_id, args) :
                                                      rt_presubmit_aic_task(func_id);
    if (tok.won) return rt_submit_winner(tok, args);
    return rt_submit_loser(tok, output_count);
}

PTO_DEVICE_FUNC inline SubmitOutputs
submit_aiv_task(int32_t func_id, const L0TaskArgs &args, uint32_t output_count = 0) {
    SubmitToken tok = args_need_region_intent(args) ? rt_presubmit_aiv_task_with_region_intent(func_id, args) :
                                                      rt_presubmit_aiv_task(func_id);
    if (tok.won) return rt_submit_winner(tok, args);
    return rt_submit_loser(tok, output_count);
}

PTO_DEVICE_FUNC inline SubmitOutputs
submit_task(const MixedKernels &mixed, const L0TaskArgs &args, uint32_t output_count = 0) {
    SubmitToken tok =
        args_need_region_intent(args) ? rt_presubmit_task_with_region_intent(mixed, args) : rt_presubmit_task(mixed);
    if (tok.won) return rt_submit_winner(tok, args);
    return rt_submit_loser(tok, output_count);
}

PTO_DEVICE_FUNC inline void spin_before_region_register(uint64_t spins) {
    volatile uint64_t sink = 0;
    for (uint64_t i = 0; i < spins; i++) {
        sink += i;
    }
    if (sink == UINT64_MAX) {
        dcci(nullptr, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
}
#else
using SubmitOutputs = TaskOutputTensors;
using OutputHandle = __gm__ const Tensor &;
using ViewHandle = Tensor;

PTO_DEVICE_FUNC inline OutputHandle output_handle(const SubmitOutputs &outputs, uint32_t slot) {
    return outputs.get_ref(slot);
}

PTO_DEVICE_FUNC inline ViewHandle
output_view(OutputHandle source, const uint32_t view_shapes[], const uint32_t view_offsets[], uint32_t ndims) {
    (void)ndims;
    return Tensor::view(source, view_shapes, view_offsets);
}

PTO_DEVICE_FUNC inline SubmitOutputs
submit_aic_task(int32_t func_id, const L0TaskArgs &args, uint32_t output_count = 0) {
    (void)output_count;
    return rt_submit_aic_task(func_id, args);
}

PTO_DEVICE_FUNC inline SubmitOutputs
submit_aiv_task(int32_t func_id, const L0TaskArgs &args, uint32_t output_count = 0) {
    (void)output_count;
    return rt_submit_aiv_task(func_id, args);
}

PTO_DEVICE_FUNC inline SubmitOutputs
submit_task(const MixedKernels &mixed, const L0TaskArgs &args, uint32_t output_count = 0) {
    (void)output_count;
    return rt_submit_task(mixed, args);
}
#endif

PTO_DEVICE_FUNC inline OutputHandle
alloc_output_handle(const L0TaskArgs &args, SubmitOutputs &outputs, uint32_t slot = 0) {
#if PTO_FDWIC_SHARED_MAP
    const int32_t task_id = dist_alloc_outputs_impl(nullptr, args);
    outputs = rt_output_refs(task_id, rt_count_outputs(args));
    return FdwicOutputRef{task_id, static_cast<int16_t>(slot), 0, 0, 0, 0};
#else
    outputs = alloc_tensors(args);
    return output_handle(outputs, slot);
#endif
}

static PTO_DEVICE_FUNC void run_alloc_fill_runahead(const Tensor &input, uint64_t n) {
    const uint32_t shape[1] = {static_cast<uint32_t>(n)};
    TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
    for (uint64_t i = 0; i < n; i++) {
        (void)i;
        L0TaskArgs alloc_args;
        alloc_args.add_output(scratch_ci);
        SubmitOutputs scratch_out;
        OutputHandle scratch = alloc_output_handle(alloc_args, scratch_out);
        L0TaskArgs fill_args;
        fill_args.add_input(input);
        fill_args.add_output(scratch);
        fill_args.add_scalar(n);
        submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_args);
    }
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
    const Tensor &dump = orch_args.tensor(2).ref();
    const uint64_t n = orch_args.scalar(0);
    const uint64_t mode = orch_args.scalar(1);
    const uint32_t shape[1] = {static_cast<uint32_t>(n)};

    if (mode == 1) {
        const uint32_t sub_n = static_cast<uint32_t>(n / 2);
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t sub_offset[1] = {static_cast<uint32_t>(n / 4)};
        Tensor sub_output = Tensor::view(output, sub_shape, sub_offset);

        L0TaskArgs fill_full_args;
        fill_full_args.add_input(input);
        fill_full_args.add_inout(output);
        fill_full_args.add_scalar(n);
        submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_full_args);

        L0TaskArgs bump_sub_args;
        bump_sub_args.add_inout(sub_output);
        bump_sub_args.add_scalar(sub_n);
        submit_aiv_task(FUNC_BUMP_INOUT_AIV, bump_sub_args);
        return;
    }

    if (mode == 2) {
        L0TaskArgs fill_full_args;
        fill_full_args.add_input(input);
        fill_full_args.add_inout(output);
        fill_full_args.add_scalar(n);
        submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_full_args);

        L0TaskArgs bump_full_args;
        bump_full_args.add_inout(output);
        bump_full_args.add_scalar(n);
        submit_aiv_task(FUNC_BUMP_INOUT_AIV, bump_full_args);
        return;
    }

    if (mode == 3) {
        const uint32_t sub_n = static_cast<uint32_t>(n / 2);
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t sub_offset[1] = {static_cast<uint32_t>(n / 4)};
        Tensor sub_output = Tensor::view(output, sub_shape, sub_offset);

        L0TaskArgs bump_sub_args;
        bump_sub_args.add_inout(sub_output);
        bump_sub_args.add_scalar(sub_n);
        submit_aiv_task(FUNC_BUMP_INOUT_AIV, bump_sub_args);
        return;
    }

    if (mode == 4) {
        const uint32_t sub_n = static_cast<uint32_t>(n / 2);
        const uint32_t sub_offset = static_cast<uint32_t>(n / 4);

        L0TaskArgs bump_offset_args;
        bump_offset_args.add_inout(output);
        bump_offset_args.add_scalar(sub_n);
        bump_offset_args.add_scalar(sub_offset);
        submit_aiv_task(FUNC_BUMP_OFFSET_AIV, bump_offset_args);
        return;
    }

    if (mode == 5) {
        const uint32_t sub_n = static_cast<uint32_t>(n / 2);
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t sub_offset[1] = {static_cast<uint32_t>(n / 4)};
        Tensor sub_output;
        Tensor::init_from_line1(sub_output, output);
        sub_output.start_offset += static_cast<uint64_t>(sub_offset[0]) * output.strides[0];
        sub_output.shapes[0] = sub_shape[0];
        sub_output.strides[0] = output.strides[0];
        sub_output.manual_dep = false;
        sub_output.is_contiguous = true;
        sub_output.extent_elem_cache = sub_shape[0];
        __gm__ float *check_output =
            reinterpret_cast<__gm__ float *>(static_cast<uintptr_t>(output.buffer.addr)) + output.start_offset;
        check_output[80] = sub_output.buffer.addr == output.buffer.addr ? 1.0f : 0.0f;
        check_output[81] = sub_output.start_offset == sub_offset[0] ? 1.0f : 0.0f;
        check_output[82] = sub_output.shapes[0] == sub_n ? 1.0f : 0.0f;
        check_output[83] = sub_output.strides[0] == 1 ? 1.0f : 0.0f;
        check_output[84] = sub_output.ndims == 1 ? 1.0f : 0.0f;
        check_output[85] = sub_output.dtype == DataType::FLOAT32 ? 1.0f : 0.0f;
        check_output[86] = sub_output.is_contiguous ? 1.0f : 0.0f;
        check_output[87] = sub_output.extent_elem_cache == sub_n ? 1.0f : 0.0f;
        check_output[88] = output.start_offset == 0 ? 1.0f : 0.0f;
        check_output[89] = output.shapes[0] == n ? 1.0f : 0.0f;
        check_output[90] = output.strides[0] == 1 ? 1.0f : 0.0f;
        check_output[91] = output.ndims == 1 ? 1.0f : 0.0f;
        check_output[92] = output.dtype == DataType::FLOAT32 ? 1.0f : 0.0f;
        check_output[93] = output.is_contiguous ? 1.0f : 0.0f;
        check_output[94] = output.extent_elem_cache == n ? 1.0f : 0.0f;
        check_output[95] = sub_output.shapes[0] == sub_n ? 1.0f : 0.0f;
        check_output[96] = sub_output.extent_elem_cache == sub_n ? 1.0f : 0.0f;
        check_output[97] = sub_n == n / 2 ? 1.0f : 0.0f;
        check_output[98] = sub_shape[0] == sub_n ? 1.0f : 0.0f;
        check_output[99] = sub_offset[0] == n / 4 ? 1.0f : 0.0f;
        check_output[100] = sub_output.start_offset == sub_offset[0] ? 1.0f : 0.0f;
        check_output[101] = sub_output.strides[0] == 1 ? 1.0f : 0.0f;

        L0TaskArgs inspect_args;
        inspect_args.add_input(sub_output);
        const Tensor &after_input = inspect_args.tensor(0).ref();
        check_output[102] = after_input.buffer.addr == output.buffer.addr ? 1.0f : 0.0f;
        check_output[103] = after_input.start_offset == sub_offset[0] ? 1.0f : 0.0f;
        check_output[104] = after_input.shapes[0] == sub_n ? 1.0f : 0.0f;
        check_output[105] = after_input.strides[0] == 1 ? 1.0f : 0.0f;
        inspect_args.add_inout(output);
        inspect_args.add_inout(dump);
        inspect_args.add_scalar(sub_offset[0]);
        inspect_args.add_scalar(sub_n);
        const Tensor &after_full_args = inspect_args.tensor(0).ref();
        check_output[106] = after_full_args.buffer.addr == output.buffer.addr ? 1.0f : 0.0f;
        check_output[107] = after_full_args.start_offset == sub_offset[0] ? 1.0f : 0.0f;
        check_output[108] = after_full_args.shapes[0] == sub_n ? 1.0f : 0.0f;
        check_output[109] = after_full_args.strides[0] == 1 ? 1.0f : 0.0f;
        submit_aiv_task(FUNC_INSPECT_VIEW_AIV, inspect_args);
        return;
    }

    if (mode == 20) {
        const uint32_t sub_n = static_cast<uint32_t>(n / 2);
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t sub_offset[1] = {static_cast<uint32_t>(n / 4)};
        Tensor sub_output = Tensor::view(output, sub_shape, sub_offset);
        __gm__ float *check_output =
            reinterpret_cast<__gm__ float *>(static_cast<uintptr_t>(output.buffer.addr)) + output.start_offset;
        check_output[0] = sub_output.buffer.addr == output.buffer.addr ? 1.0f : 0.0f;
        check_output[1] = sub_output.start_offset == sub_offset[0] ? 1.0f : 0.0f;
        check_output[2] = sub_output.shapes[0] == sub_n ? 1.0f : 0.0f;
        check_output[3] = sub_output.strides[0] == 1 ? 1.0f : 0.0f;
        check_output[4] = sub_output.ndims == 1 ? 1.0f : 0.0f;
        check_output[5] = sub_output.dtype == DataType::FLOAT32 ? 1.0f : 0.0f;
        check_output[6] = sub_output.is_contiguous ? 1.0f : 0.0f;
        check_output[7] = sub_output.extent_elem_cache == sub_n ? 1.0f : 0.0f;
        dcci(check_output, SINGLE_CACHE_LINE, CACHELINE_OUT);
        return;
    }

    if (mode == 21) {
        const uint32_t sub_n = static_cast<uint32_t>(n / 2);
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t sub_offset[1] = {static_cast<uint32_t>(n / 4)};
        Tensor sub_output = Tensor::view(output, sub_shape, sub_offset);
        L0TaskArgs inspect_args;
        inspect_args.add_input(sub_output);
        const Tensor &after_input = inspect_args.tensor(0).ref();
        __gm__ float *check_output =
            reinterpret_cast<__gm__ float *>(static_cast<uintptr_t>(output.buffer.addr)) + output.start_offset;
        check_output[0] = after_input.buffer.addr == output.buffer.addr ? 1.0f : 0.0f;
        check_output[1] = after_input.start_offset == sub_offset[0] ? 1.0f : 0.0f;
        check_output[2] = after_input.shapes[0] == sub_n ? 1.0f : 0.0f;
        check_output[3] = after_input.strides[0] == 1 ? 1.0f : 0.0f;
        check_output[4] = inspect_args.tensor(0).refers_to(&sub_output) ? 1.0f : 0.0f;
        check_output[5] = inspect_args.tensor(0).refers_to(&input) ? 1.0f : 0.0f;
        check_output[6] = inspect_args.tensor(0).refers_to(&output) ? 1.0f : 0.0f;
        check_output[7] = sub_output.buffer.addr == output.buffer.addr ? 1.0f : 0.0f;
        check_output[8] = sub_output.start_offset == sub_offset[0] ? 1.0f : 0.0f;
        check_output[9] = sub_output.shapes[0] == sub_n ? 1.0f : 0.0f;
        check_output[10] = sub_output.strides[0] == 1 ? 1.0f : 0.0f;
        dcci(check_output, SINGLE_CACHE_LINE, CACHELINE_OUT);
        return;
    }

    if (mode == 14) {
        const uint32_t sub_n = static_cast<uint32_t>(n / 2);
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t sub_offset[1] = {static_cast<uint32_t>(n / 4)};
        Tensor sub_output = Tensor::view(output, sub_shape, sub_offset);
        __gm__ float *check_output =
            reinterpret_cast<__gm__ float *>(static_cast<uintptr_t>(output.buffer.addr)) + output.start_offset;

        L0TaskArgs inspect_args;
        inspect_args.add_input(sub_output);
        check_output[0] = inspect_args.tensor_count() == 1 ? 1.0f : 0.0f;
        check_output[1] = inspect_args.tag(0) == TensorArgType::INPUT ? 1.0f : 0.0f;

        inspect_args.add_inout(output);
        check_output[2] = inspect_args.tensor_count() == 2 ? 1.0f : 0.0f;
        check_output[3] = inspect_args.tag(0) == TensorArgType::INPUT ? 1.0f : 0.0f;
        check_output[4] = inspect_args.tag(1) == TensorArgType::INOUT ? 1.0f : 0.0f;

        inspect_args.add_inout(dump);
        check_output[5] = inspect_args.tensor_count() == 3 ? 1.0f : 0.0f;
        check_output[6] = inspect_args.tag(0) == TensorArgType::INPUT ? 1.0f : 0.0f;
        check_output[7] = inspect_args.tag(1) == TensorArgType::INOUT ? 1.0f : 0.0f;
        check_output[8] = inspect_args.tag(2) == TensorArgType::INOUT ? 1.0f : 0.0f;
        check_output[9] = inspect_args.tensor(0).ref().start_offset == sub_offset[0] ? 1.0f : 0.0f;
        check_output[10] = inspect_args.tensor(0).ref().shapes[0] == sub_n ? 1.0f : 0.0f;
        check_output[11] = inspect_args.tensor(0).ref().strides[0] == 1 ? 1.0f : 0.0f;
        dcci(check_output, SINGLE_CACHE_LINE, CACHELINE_OUT);
        return;
    }

    if (mode == 6) {
        Tensor local_output;
        Tensor::copy(local_output, output);

        L0TaskArgs inspect_args;
        inspect_args.add_input(local_output);
        inspect_args.add_inout(output);
        inspect_args.add_inout(dump);
        inspect_args.add_scalar(0);
        inspect_args.add_scalar(n);
        submit_aiv_task(FUNC_INSPECT_VIEW_AIV, inspect_args);
        return;
    }

    if (mode == 7) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs alloc_args;
        alloc_args.add_output(scratch_ci);
        SubmitOutputs seed_out;
        OutputHandle seed = alloc_output_handle(alloc_args, seed_out);
#if PTO_FDWIC_SHARED_MAP
        const uint32_t check_shape[1] = {10};
        const uint32_t first_offset[1] = {0};
        const uint32_t second_offset[1] = {16};
        Tensor first_check = Tensor::view(output, check_shape, first_offset);
        Tensor second_check = Tensor::view(output, check_shape, second_offset);
        L0TaskArgs inspect_args;
        inspect_args.add_input(seed);
        inspect_args.add_inout(first_check);
        inspect_args.add_inout(first_check);
        inspect_args.add_scalar(n);
        submit_aic_task(FUNC_INSPECT_ALLOC_AIC, inspect_args);
        L0TaskArgs inspect_again_args;
        inspect_again_args.add_input(seed);
        inspect_again_args.add_inout(second_check);
        inspect_again_args.add_inout(second_check);
        inspect_again_args.add_scalar(n);
        submit_aic_task(FUNC_INSPECT_ALLOC_AIC, inspect_again_args);
#else
        __gm__ float *check_output =
            reinterpret_cast<__gm__ float *>(static_cast<uintptr_t>(output.buffer.addr)) + output.start_offset;
        write_descriptor_checks(seed, check_output, 0, n, output);

        L0TaskArgs fill_args;
        fill_args.add_inout(seed);
#if defined(__CCE_AICORE__)
        if (fill_args.tensor(0).tensor_from_gm()) {
            write_descriptor_checks(fill_args.tensor(0).gm_ref(), check_output, 16, n, output);
        } else
#endif
        {
            write_descriptor_checks(fill_args.tensor(0).ref(), check_output, 16, n, output);
        }
        dcci(check_output, SINGLE_CACHE_LINE, CACHELINE_OUT);
        dcci(check_output + 16, SINGLE_CACHE_LINE, CACHELINE_OUT);
#endif
        return;
    }

    if (mode == 8) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs alloc_args;
        alloc_args.add_output(scratch_ci);
        SubmitOutputs seed_out;
        OutputHandle seed = alloc_output_handle(alloc_args, seed_out);

        L0TaskArgs fill_args;
        fill_args.add_input(input);
        fill_args.add_inout(seed);
        fill_args.add_scalar(n);
        submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_args);

        L0TaskArgs fanin_args;
        fanin_args.add_input(seed);
        fanin_args.add_input(seed);
        fanin_args.add_input(seed);
        fanin_args.add_inout(output);
        fanin_args.add_scalar(n);
        submit_aiv_task(FUNC_FANIN_AIV, fanin_args);
        return;
    }

    if (mode == 13) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs alloc_args;
        alloc_args.add_output(scratch_ci);
        SubmitOutputs seed_out;
        OutputHandle seed = alloc_output_handle(alloc_args, seed_out);

        L0TaskArgs fill_args;
        fill_args.add_input(input);
        fill_args.add_inout(seed);
        fill_args.add_scalar(n);
        submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_args);
        return;
    }

    if (mode == 9) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs left_args;
        left_args.add_input(input);
        left_args.add_output(scratch_ci);
        left_args.add_scalar(n);
        SubmitOutputs left_out = submit_aic_task(FUNC_MAKE_LEFT_AIC, left_args, 1);
        OutputHandle left = output_handle(left_out, 0);

        L0TaskArgs fanin_args;
        fanin_args.add_input(left);
        fanin_args.add_input(left);
        fanin_args.add_input(left);
        fanin_args.add_inout(output);
        fanin_args.add_scalar(n);
        submit_aiv_task(FUNC_FANIN_AIV, fanin_args);
        return;
    }

    if (mode == 10) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs right_args;
        right_args.add_input(input);
        right_args.add_output(scratch_ci);
        right_args.add_scalar(n);
        SubmitOutputs right_out = submit_aiv_task(FUNC_MAKE_RIGHT_AIV, right_args, 1);
        OutputHandle right = output_handle(right_out, 0);

        L0TaskArgs fanin_args;
        fanin_args.add_input(right);
        fanin_args.add_input(right);
        fanin_args.add_input(right);
        fanin_args.add_inout(output);
        fanin_args.add_scalar(n);
        submit_aiv_task(FUNC_FANIN_AIV, fanin_args);
        return;
    }

    if (mode == 15) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs left_args;
        left_args.add_input(input);
        left_args.add_output(scratch_ci);
        left_args.add_scalar(n);
        SubmitOutputs left_out = submit_aic_task(FUNC_MAKE_LEFT_AIC, left_args, 1);
        OutputHandle left = output_handle(left_out, 0);

        const uint32_t sub_n = static_cast<uint32_t>(n / 2);
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t sub_offset[1] = {static_cast<uint32_t>(n / 4)};
        ViewHandle left_view = output_view(left, sub_shape, sub_offset, 1);

        L0TaskArgs fanin_args;
        fanin_args.add_input(left_view);
        fanin_args.add_input(left_view);
        fanin_args.add_input(left_view);
        fanin_args.add_inout(output);
        fanin_args.add_scalar(sub_n);
        submit_aiv_task(FUNC_FANIN_AIV, fanin_args);
        return;
    }

    if (mode == 16) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs alloc_args;
        alloc_args.add_output(scratch_ci);
        SubmitOutputs seed_out;
        OutputHandle seed = alloc_output_handle(alloc_args, seed_out);

        L0TaskArgs fill_args;
        fill_args.add_input(input);
        fill_args.add_inout(seed);
        fill_args.add_scalar(n);
        fill_args.add_scalar(static_cast<uint64_t>(2000000));
        submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_args);

        const uint32_t sub_n = static_cast<uint32_t>(n / 8);
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t seed_offset[1] = {0};
        ViewHandle seed_view = output_view(seed, sub_shape, seed_offset, 1);
        for (uint64_t i = 0; i < 6; i++) {
            const uint32_t out_offset[1] = {static_cast<uint32_t>(i * sub_n)};
            Tensor output_view = Tensor::view(output, sub_shape, out_offset);
            L0TaskArgs fanin_args;
            fanin_args.add_input(seed_view);
            fanin_args.add_input(seed_view);
            fanin_args.add_input(seed_view);
            fanin_args.add_inout(output_view);
            fanin_args.add_scalar(sub_n);
            submit_aiv_task(FUNC_FANIN_AIV, fanin_args);
        }
        return;
    }

    if (mode == 17) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs right_args;
        right_args.add_input(input);
        right_args.add_output(scratch_ci);
        right_args.add_scalar(n);
        SubmitOutputs right_out = submit_aiv_task(FUNC_MAKE_RIGHT_AIV, right_args, 1);
        OutputHandle right = output_handle(right_out, 0);

        L0TaskArgs fill_args;
        fill_args.add_input(right);
        fill_args.add_inout(output);
        fill_args.add_scalar(n);
        submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_args);
        return;
    }

    if (mode == 18) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs right_args;
        right_args.add_input(input);
        right_args.add_output(scratch_ci);
        right_args.add_scalar(n);
        right_args.add_scalar(static_cast<uint64_t>(8000000));
        SubmitOutputs right_out = submit_aiv_task(FUNC_MAKE_RIGHT_AIV, right_args, 1);
        OutputHandle right = output_handle(right_out, 0);

        const uint32_t sub_n = 32;
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t right_offset[1] = {0};
        ViewHandle right_view = output_view(right, sub_shape, right_offset, 1);
        for (uint64_t i = 0; i < 6; i++) {
            const uint32_t out_offset[1] = {static_cast<uint32_t>(i * sub_n)};
            Tensor output_view = Tensor::view(output, sub_shape, out_offset);
            L0TaskArgs fill_args;
            fill_args.add_input(right_view);
            fill_args.add_inout(output_view);
            fill_args.add_scalar(sub_n);
            submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_args);
        }
        return;
    }

    if (mode == 28) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs right_args;
        right_args.add_input(input);
        right_args.add_output(scratch_ci);
        right_args.add_scalar(n);
        right_args.add_scalar(static_cast<uint64_t>(8000000));
        SubmitOutputs right_out = submit_aiv_task(FUNC_MAKE_RIGHT_AIV, right_args, 1);
        OutputHandle right = output_handle(right_out, 0);

        const uint32_t sub_n = 32;
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t right_offset[1] = {0};
        ViewHandle right_view = output_view(right, sub_shape, right_offset, 1);
        for (uint64_t i = 0; i < 12; i++) {
            const uint32_t out_offset[1] = {static_cast<uint32_t>(i * sub_n)};
            Tensor output_view = Tensor::view(output, sub_shape, out_offset);
            L0TaskArgs fill_args;
            fill_args.add_input(right_view);
            fill_args.add_inout(output_view);
            fill_args.add_scalar(sub_n);
            submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_args);
        }
        return;
    }

    if (mode == 29) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);

        L0TaskArgs seed_args;
        seed_args.add_input(input);
        seed_args.add_output(scratch_ci);
        seed_args.add_scalar(n);
        SubmitOutputs seed_out = submit_aic_task(FUNC_FILL_ALLOC_AIC, seed_args, 1);
        OutputHandle seed = output_handle(seed_out, 0);

        L0TaskArgs left_args;
        left_args.add_input(input);
        left_args.add_output(scratch_ci);
        left_args.add_scalar(n);
        SubmitOutputs left_out = submit_aic_task(FUNC_MAKE_LEFT_AIC, left_args, 1);
        OutputHandle left = output_handle(left_out, 0);

        L0TaskArgs right_args;
        right_args.add_input(input);
        right_args.add_output(scratch_ci);
        right_args.add_scalar(n);
        right_args.add_scalar(static_cast<uint64_t>(8000000));
        SubmitOutputs right_out = submit_aiv_task(FUNC_MAKE_RIGHT_AIV, right_args, 1);
        OutputHandle right = output_handle(right_out, 0);

        const uint32_t sub_n = 32;
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t src_offset[1] = {0};
        ViewHandle seed_view = output_view(seed, sub_shape, src_offset, 1);
        ViewHandle left_view = output_view(left, sub_shape, src_offset, 1);
        ViewHandle right_view = output_view(right, sub_shape, src_offset, 1);
        for (uint64_t i = 0; i < 12; i++) {
            const uint32_t out_offset[1] = {static_cast<uint32_t>(i * sub_n)};
            Tensor output_view = Tensor::view(output, sub_shape, out_offset);
            L0TaskArgs fanin_args;
            fanin_args.add_input(seed_view);
            fanin_args.add_input(left_view);
            fanin_args.add_input(right_view);
            fanin_args.add_inout(output_view);
            fanin_args.add_scalar(sub_n);
            submit_aiv_task(FUNC_FANIN_AIV, fanin_args);
        }
        return;
    }

    if (mode == 30) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);

        L0TaskArgs seed_args;
        seed_args.add_input(input);
        seed_args.add_output(scratch_ci);
        seed_args.add_scalar(n);
        SubmitOutputs seed_out = submit_aic_task(FUNC_FILL_ALLOC_AIC, seed_args, 1);
        OutputHandle seed = output_handle(seed_out, 0);

        L0TaskArgs left_args;
        left_args.add_input(input);
        left_args.add_output(scratch_ci);
        left_args.add_scalar(n);
        SubmitOutputs left_out = submit_aic_task(FUNC_MAKE_LEFT_AIC, left_args, 1);
        OutputHandle left = output_handle(left_out, 0);

        L0TaskArgs right_args;
        right_args.add_input(input);
        right_args.add_output(scratch_ci);
        right_args.add_scalar(n);
        right_args.add_scalar(static_cast<uint64_t>(8000000));
        SubmitOutputs right_out = submit_aiv_task(FUNC_MAKE_RIGHT_AIV, right_args, 1);
        OutputHandle right = output_handle(right_out, 0);

        const uint32_t sub_n = 32;
        const uint32_t sub_shape[1] = {sub_n};
        const uint32_t src_offset[1] = {0};
        ViewHandle seed_view = output_view(seed, sub_shape, src_offset, 1);
        ViewHandle left_view = output_view(left, sub_shape, src_offset, 1);
        ViewHandle right_view = output_view(right, sub_shape, src_offset, 1);
        for (uint64_t i = 0; i < 96; i++) {
            const uint32_t out_offset[1] = {static_cast<uint32_t>(i * sub_n)};
            Tensor output_view = Tensor::view(output, sub_shape, out_offset);
            L0TaskArgs fanin_args;
            fanin_args.add_input(seed_view);
            fanin_args.add_input(left_view);
            fanin_args.add_input(right_view);
            fanin_args.add_inout(output_view);
            fanin_args.add_scalar(sub_n);
            submit_aiv_task(FUNC_FANIN_AIV, fanin_args);
        }
        return;
    }

    if (mode == 22) {
        run_alloc_fill_runahead(input, n);
        return;
    }

    if (mode == 23) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        for (uint64_t i = 0; i < n; i++) {
            L0TaskArgs alloc_args;
            alloc_args.add_output(scratch_ci);
            SubmitOutputs scratch_out;
            (void)alloc_output_handle(alloc_args, scratch_out);
            (void)scratch_out;
        }
        return;
    }

    if (mode == 24) {
        L0TaskArgs fill_args;
        fill_args.add_input(input);
        fill_args.add_inout(output);
        fill_args.add_scalar(n);
        submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_args);
        return;
    }

    if (mode == 25) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs left_args;
        left_args.add_input(input);
        left_args.add_output(scratch_ci);
        left_args.add_scalar(n);
        submit_aic_task(FUNC_MAKE_LEFT_AIC, left_args);
        return;
    }

    if (mode == 26) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs alloc_args;
        alloc_args.add_output(scratch_ci);
        SubmitOutputs scratch_out;
        OutputHandle scratch = alloc_output_handle(alloc_args, scratch_out);
        L0TaskArgs inspect_args;
        inspect_args.add_input(scratch);
        inspect_args.add_inout(output);
        inspect_args.add_inout(dump);
        inspect_args.add_scalar(0);
        inspect_args.add_scalar(n);
        submit_aiv_task(FUNC_INSPECT_VIEW_AIV, inspect_args);
        return;
    }

    if (mode == 27) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs alloc_args;
        alloc_args.add_output(scratch_ci);
        SubmitOutputs scratch_out;
        OutputHandle scratch = alloc_output_handle(alloc_args, scratch_out);
        L0TaskArgs inspect_args;
        inspect_args.add_input(scratch);
        inspect_args.add_inout(output);
        inspect_args.add_inout(dump);
        inspect_args.add_scalar(n);
        submit_aic_task(FUNC_INSPECT_ALLOC_AIC, inspect_args);
        return;
    }

    if (mode == 19) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs left_args;
        left_args.add_input(input);
        left_args.add_output(scratch_ci);
        left_args.add_scalar(n);
        left_args.add_scalar(static_cast<uint64_t>(12000000));
        SubmitOutputs left_out = submit_aic_task(FUNC_MAKE_LEFT_AIC, left_args, 1);
        OutputHandle left = output_handle(left_out, 0);

        L0TaskArgs fanin_args;
        fanin_args.add_input(left);
        fanin_args.add_input(left);
        fanin_args.add_input(left);
        fanin_args.add_inout(output);
        fanin_args.add_scalar(n);
        fanin_args.add_scalar(static_cast<uint64_t>(12000000));
        submit_aiv_task(FUNC_FANIN_AIV, fanin_args);

        for (uint64_t i = 0; i < 70; i++) {
            L0TaskArgs alloc_args;
            alloc_args.add_output(scratch_ci);
            SubmitOutputs scratch_out;
            OutputHandle scratch = alloc_output_handle(alloc_args, scratch_out);
            L0TaskArgs fill_args;
            fill_args.add_input(input);
            fill_args.add_inout(scratch);
            fill_args.add_scalar(n);
            submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_args);
        }
        return;
    }

    if (mode == 31) {
        L0TaskArgs probe_args;
        probe_args.add_inout(output);
        MixedKernels mk;
        mk.aic_kernel_id = FUNC_DCCI_ATOMIC_CLOBBER_AIC;
        mk.aiv0_kernel_id = FUNC_DCCI_ATOMIC_CLOBBER_AIV;
        submit_task(mk, probe_args);
        return;
    }

    if (mode == 32) {
        L0TaskArgs fill_args;
        fill_args.add_input(input);
        fill_args.add_inout(output);
        fill_args.add_scalar(n);
#if PTO_FDWIC_SHARED_MAP
        SubmitToken fill_tok = rt_presubmit_aic_task_with_region_intent(FUNC_FILL_ALLOC_AIC, fill_args);
        if (fill_tok.won) {
            spin_before_region_register(static_cast<uint64_t>(12000000));
            rt_submit_winner(fill_tok, fill_args);
        } else {
            rt_submit_loser(fill_tok, 0);
        }
#else
        submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_args);
#endif

        L0TaskArgs bump_args;
        bump_args.add_inout(output);
        bump_args.add_scalar(n);
        submit_aiv_task(FUNC_BUMP_INOUT_AIV, bump_args);
        return;
    }

    if (mode == 33) {
        TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);
        L0TaskArgs alloc_args;
        alloc_args.add_output(scratch_ci);
        SubmitOutputs scratch_out;
        OutputHandle scratch = alloc_output_handle(alloc_args, scratch_out);

        L0TaskArgs fill_args;
        fill_args.add_input(input);
        fill_args.add_output(scratch);
        fill_args.add_scalar(n);
#if PTO_FDWIC_SHARED_MAP
        SubmitToken fill_tok = rt_presubmit_aic_task_with_region_intent(FUNC_FILL_ALLOC_AIC, fill_args);
        if (fill_tok.won) {
            spin_before_region_register(static_cast<uint64_t>(12000000));
            rt_submit_winner(fill_tok, fill_args);
        } else {
            rt_submit_loser(fill_tok, 0);
        }
#else
        submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_args);
#endif

        L0TaskArgs fanin_args;
        fanin_args.add_input(scratch);
        fanin_args.add_input(scratch);
        fanin_args.add_input(scratch);
        fanin_args.add_inout(output);
        fanin_args.add_scalar(n);
        submit_aiv_task(FUNC_FANIN_AIV, fanin_args);
        return;
    }

    TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);

    L0TaskArgs alloc_args;
    alloc_args.add_output(scratch_ci);
    SubmitOutputs seed_out;
    OutputHandle seed = alloc_output_handle(alloc_args, seed_out);

    L0TaskArgs fill_args;
    fill_args.add_input(input);
    fill_args.add_inout(seed);
    fill_args.add_scalar(n);
    submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_args);

    L0TaskArgs left_args;
    left_args.add_input(input);
    left_args.add_output(scratch_ci);
    left_args.add_scalar(n);
    SubmitOutputs left_out = submit_aic_task(FUNC_MAKE_LEFT_AIC, left_args, 1);
    OutputHandle left = output_handle(left_out, 0);

    L0TaskArgs right_args;
    right_args.add_input(input);
    right_args.add_output(scratch_ci);
    right_args.add_scalar(n);
    SubmitOutputs right_out = submit_aiv_task(FUNC_MAKE_RIGHT_AIV, right_args, 1);
    OutputHandle right = output_handle(right_out, 0);

    L0TaskArgs fanin_args;
    fanin_args.add_input(seed);
    fanin_args.add_input(left);
    fanin_args.add_input(right);
    fanin_args.add_inout(output);
    fanin_args.add_scalar(n);
    submit_aiv_task(FUNC_FANIN_AIV, fanin_args);
}

}  // extern "C"
