#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""fdwic shared-map symbolic output smoke test."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestSharedSymbolSmoke(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/shared_symbol_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "PRODUCE_AIC",
                "source": "kernels/aic/produce.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "PRODUCE_AIV",
                "source": "kernels/aiv/produce.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "name": "CONSUME_AIC",
                "source": "kernels/aic/consume.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 3,
                "name": "CONSUME_AIV",
                "source": "kernels/aiv/consume.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 4,
                "name": "PRODUCE_PAIR_AIC",
                "source": "kernels/aic/produce_pair.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.OUT, D.OUT],
            },
            {
                "func_id": 5,
                "name": "PRODUCE_PAIR_AIV",
                "source": "kernels/aiv/produce_pair.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "A5SimBd36AicToAivSymbol",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 0},
        },
        {
            "name": "A5SimBd36AivToAicSymbol",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 1},
        },
        {
            "name": "A5SimBd36AicPairSlot1ToAivSymbol",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 2},
        },
        {
            "name": "A5OnboardBd36AicPairSlot1ToAivSymbol",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 2},
        },
        {
            "name": "A5SimBd36AivPairSlot1ToAicSymbol",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 3},
        },
        {
            "name": "A5SimBd36PaddedShardSlot1Symbol",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 4},
        },
        {
            "name": "A5OnboardBd36PaddedShardSlot1Symbol",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 4},
        },
    ]

    def generate_args(self, params):
        n = int(params["n"])
        x = torch.arange(n, dtype=torch.float32)
        mode = int(params["mode"])
        y = torch.full((n,), -1.0, dtype=torch.float32)
        return TaskArgsBuilder(
            Tensor("input", x),
            Tensor("output", y),
            Scalar("n", n),
            Scalar("mode", mode),
        )

    def compute_golden(self, args, params):
        mode = int(params["mode"])
        if mode == 0:
            args.output[:] = args.input * 3.0 + 14.0
        elif mode == 1:
            args.output[:] = args.input * 4.0 + 22.0
        elif mode == 2:
            args.output[:] = args.input * 6.0 + 18.0
        elif mode == 3:
            args.output[:] = args.input * 8.0 + 30.0
        elif mode == 4:
            args.output[:] = args.input * 6.0 + 18.0


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
