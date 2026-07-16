#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared TensorMap dependency smoke coverage for eager submits."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestSharedDependencySmoke(SceneTestCase):
    RTOL = 0
    ATOL = 0
    RUNTIME_COMPILE_DEFINITIONS = {"PTO_FDWIC_SHARED_TENSORMAP": "1"}

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/shared_dependency_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "FILL_INOUT_AIC",
                "source": "kernels/aic/fill_inout.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 1,
                "name": "NO_OP_1_AIV",
                "source": "kernels/aiv/no_op.cpp",
                "core_type": "aiv",
                "signature": [],
            },
            {
                "func_id": 2,
                "name": "NO_OP_2_AIV",
                "source": "kernels/aiv/no_op.cpp",
                "core_type": "aiv",
                "signature": [],
            },
            {
                "func_id": 3,
                "name": "NO_OP_3_AIV",
                "source": "kernels/aiv/no_op.cpp",
                "core_type": "aiv",
                "signature": [],
            },
            {
                "func_id": 4,
                "name": "BUMP_INOUT_AIV",
                "source": "kernels/aiv/bump_inout.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 5,
                "name": "BUMP_OFFSET_AIV",
                "source": "kernels/aiv/bump_offset.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 6,
                "name": "INSPECT_VIEW_AIV",
                "source": "kernels/aiv/inspect_view.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT, D.INOUT],
            },
            {
                "func_id": 7,
                "name": "MAKE_LEFT_AIC",
                "source": "kernels/aic/make_left.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 8,
                "name": "FANIN_AIV",
                "source": "kernels/aiv/fanin.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.IN, D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "A5SimBd36OverlapSubView",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 1},
        },
        {
            "name": "A5OnboardBd36OverlapSubView",
            "manual": True,
            "platforms": ["a5"],
            "config": {"block_dim": 36},
            "params": {"n": 128, "mode": 1},
        },
        {
            "name": "A5SimBd36ExistingInoutChain",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 2},
        },
        {
            "name": "A5OnboardBd36ExistingInoutChain",
            "manual": True,
            "platforms": ["a5"],
            "config": {"block_dim": 36},
            "params": {"n": 128, "mode": 2},
        },
        {
            "name": "A5SimBd36SubViewOnly",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 3},
        },
        {
            "name": "A5OnboardBd36SubViewOnly",
            "manual": True,
            "platforms": ["a5"],
            "config": {"block_dim": 36},
            "params": {"n": 128, "mode": 3},
        },
        {
            "name": "A5SimBd36ScalarOffsetOnly",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 4},
        },
        {
            "name": "A5OnboardBd36ScalarOffsetOnly",
            "manual": True,
            "platforms": ["a5"],
            "config": {"block_dim": 36},
            "params": {"n": 128, "mode": 4},
        },
        {
            "name": "A5SimBd36InspectSubView",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 5},
        },
        {
            "name": "A5OnboardBd36InspectSubView",
            "manual": True,
            "platforms": ["a5"],
            "config": {"block_dim": 36},
            "params": {"n": 128, "mode": 5},
        },
        {
            "name": "A5SimBd36L0TaskArgsTagPersistence",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 14},
        },
        {
            "name": "A5SimBd36EagerFullInout",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 30},
        },
        {
            "name": "A5SimBd36BuilderFullInout",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 31},
        },
        {
            "name": "A5SimBd36AicOutputToAivFanin",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 32},
        },
        {
            "name": "A5SimBd36AllocOutputFanin",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 33},
        },
        {
            "name": "A5OnboardBd36EagerFullInout",
            "manual": True,
            "platforms": ["a5"],
            "config": {"block_dim": 36},
            "params": {"n": 128, "mode": 30},
        },
        {
            "name": "A5OnboardBd36BuilderFullInout",
            "manual": True,
            "platforms": ["a5"],
            "config": {"block_dim": 36},
            "params": {"n": 128, "mode": 31},
        },
        {
            "name": "A5OnboardBd36AicOutputToAivFanin",
            "manual": True,
            "platforms": ["a5"],
            "config": {"block_dim": 36},
            "params": {"n": 128, "mode": 32},
        },
        {
            "name": "A5OnboardBd36AllocOutputFanin",
            "manual": True,
            "platforms": ["a5"],
            "config": {"block_dim": 36},
            "params": {"n": 128, "mode": 33},
        },
        {
            "name": "A5OnboardBd36L0TaskArgsTagPersistence",
            "manual": True,
            "platforms": ["a5"],
            "config": {"block_dim": 36},
            "params": {"n": 128, "mode": 14},
        },
    ]

    def generate_args(self, params):
        n = int(params["n"])
        return TaskArgsBuilder(
            Tensor("input", torch.arange(n, dtype=torch.float32)),
            Tensor("output", torch.full((n,), -1.0, dtype=torch.float32)),
            Tensor("dump", torch.full((n,), -1.0, dtype=torch.float32)),
            Scalar("n", n),
            Scalar("mode", int(params.get("mode", 0))),
        )

    def compute_golden(self, args, params):
        mode = int(params.get("mode", 0))
        if mode == 1:
            n = int(params["n"])
            args.output[:] = args.input + 7.0
            start = n // 4
            end = start + n // 2
            args.output[start:end] += 1.0
            return
        if mode == 2:
            args.output[:] = args.input + 8.0
            return
        if mode in {3, 4}:
            n = int(params["n"])
            start = n // 4
            end = start + n // 2
            args.output[:] = -1.0
            args.output[start:end] += 1.0
            return
        if mode == 5:
            args.output[:] = -1.0
            args.output[:4] = 1.0
            args.output[80:110] = 1.0
            args.dump[:3] = 1.0
            return
        if mode == 14:
            args.output[:] = -1.0
            args.output[:12] = 1.0
            return
        if mode in {30, 31}:
            args.output += 1.0
            return
        if mode == 32:
            args.output[:] = args.input * 6.0 + 17.0
            return
        if mode == 33:
            args.output[:] = args.input * 3.0 + 29.0
            return
        raise AssertionError(f"unsupported mode {mode}")


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
