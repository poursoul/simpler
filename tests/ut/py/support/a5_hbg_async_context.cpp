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

#include <pto/pto-inst.hpp>

#include <array>
#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

#include "async_completion_plan.h"
#include "async_kernel_api.h"
#include "dispatch_payload.h"

int main(int argc, char **argv) {
    assert(argc == 2);
    const int scenario = std::atoi(argv[1]);
    AsyncTaskPlanInput input;
    input.task_ref = 0;
    input.block_count = 1;
    input.active_mask = LaneMask::aiv0();
    input.capacities[1] = 2;
    AsyncCompletionHostPlan plan;
    std::string error;
    assert(plan_async_completion(sizeof(SchedulerRunControl), 9, 1, {input}, &plan, &error));
    std::vector<uint8_t> backing(plan.total_bytes + 127);
    auto *image = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(backing.data()) + 127) & ~uintptr_t(127));
    assert(initialize_async_completion(image, plan.total_bytes, plan));
    auto *data = scheduler_state_at<pto2::resident::AsyncGroupData>(image, plan.groups[0].offset);
    DispatchPayload payload{};
    static_assert(sizeof(DispatchPayload) == 512 && offsetof(DispatchPayload, args) == 64);
    static_assert(offsetof(DispatchPayload, global_context) == 464 && sizeof(LocalContext) == 48);
    payload.args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload.local_context);
    payload.args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload.global_context);
    auto &global = payload.global_context;
    global.async_present = 1;
    global.async_image_address = reinterpret_cast<uint64_t>(image);
    global.async_image_bytes = plan.total_bytes;
    global.async_run_epoch = 9;
    global.async_group_offset = plan.groups[0].offset;
    global.async_group_index = 0;
    auto *args = reinterpret_cast<int64_t *>(payload.args);
    const CompletionToken token{0x10000, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    if (scenario == 0) {
        auto ctx = get_async_ctx(args);
        assert(async_ctx_is_deferred(ctx));
        assert(register_completion_condition(ctx, token));
        auto second = get_async_ctx(args);
        save_expected_notification_counter(second, reinterpret_cast<void *>(token.addr), 2);
        assert(data->count == 2);
        assert(pto2::resident::seal_conditions(ctx.resident));
    } else if (scenario == 1) {
        auto ctx = get_async_ctx(args);
        assert(register_completion_condition(ctx, token));
        assert(register_completion_condition(ctx, token));
        assert(!register_completion_condition(ctx, token));
        assert(data->count == 2);
        assert(data->error_code == SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        assert(!pto2::resident::seal_conditions(ctx.resident));
    } else if (scenario == 2) {
        ++global.async_present;
        auto ctx = get_async_ctx(args);
        assert(!async_ctx_is_deferred(ctx));
        assert(data->error_code == SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        assert(!pto2::resident::seal_conditions(ctx.resident));
    } else if (scenario == 3) {
        global.async_reserved = 1;
        auto ctx = get_async_ctx(args);
        assert(!async_ctx_is_deferred(ctx));
        assert(data->error_code == SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    } else if (scenario == 4) {
        ++global.async_run_epoch;
        auto ctx = get_async_ctx(args);
        assert(!async_ctx_is_deferred(ctx));
        assert(!register_completion_condition(ctx, token));
        assert(data->count == 0 && data->error_code == 0);
    } else if (scenario == 5) {
        DeferredCompletionSlab slab{};
        global.async_present = 0;
        payload.local_context.async_ctx = LegacyAsyncCtx::make(TaskId::make_global(1), &slab);
        auto ctx = get_async_ctx(args);
        assert(async_ctx_is_deferred(ctx));
        assert(register_completion_condition(ctx, token));
        assert(slab.count == 1 && slab.entries[0].expected_value == 1);
        assert(data->count == 0 && data->error_code == 0);
    } else if (scenario == 6) {
        auto ctx = get_async_ctx(args);
        auto invalid = token;
        invalid.addr += 1;
        assert(!register_completion_condition(ctx, invalid));
        assert(data->count == 0);
        assert(!pto2::resident::seal_conditions(ctx.resident));
    } else return 2;
    return 0;
}
