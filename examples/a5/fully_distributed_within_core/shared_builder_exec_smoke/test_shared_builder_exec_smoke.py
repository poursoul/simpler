#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared TensorMap builder-submit execution smoke test."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestSharedBuilderExecSmoke(SceneTestCase):
    RTOL = 0
    ATOL = 0
    RUNTIME_COMPILE_DEFINITIONS = {"PTO_FDWIC_SHARED_TENSORMAP": "1"}

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/shared_builder_exec_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "MAKE_TEMP_AIC",
                "source": "kernels/aic/make_temp.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "CONSUME_TEMP_AIC",
                "source": "kernels/aic/consume_temp.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 2,
                "name": "FILL_OUTPUT_AIC",
                "source": "kernels/aic/fill_output.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 3,
                "name": "BUMP_OUTPUT_AIC",
                "source": "kernels/aic/bump_output.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "A5SimBd1SymbolicN16",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 16, "mode": 0},
        },
        {
            "name": "A5SimBd36SymbolicN64",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 64, "mode": 0},
        },
        {
            "name": "A5SimBd36InoutRangeN128",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 1},
        },
        {
            "name": "A5SimBd36InoutSubViewN128",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 2},
        },
    ]

    def generate_args(self, params):
        n = int(params["n"])
        return TaskArgsBuilder(
            Tensor("input", torch.arange(n, dtype=torch.float32)),
            Tensor("output", torch.full((n,), -1.0, dtype=torch.float32)),
            Scalar("n", n),
            Scalar("mode", int(params.get("mode", 0))),
        )

    def compute_golden(self, args, params):
        mode = int(params.get("mode", 0))
        if mode == 1:
            args.output[:] = args.input + 4.0
        elif mode == 2:
            n = int(params["n"])
            args.output[:] = args.input + 1.0
            args.output[n // 4 : n // 4 + n // 2] += 3.0
        else:
            args.output[:] = args.input + 3.0


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
