#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared TensorMap builder-submit smoke test."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestSharedBuilderNoevalSmoke(SceneTestCase):
    RTOL = 0
    ATOL = 0
    RUNTIME_COMPILE_DEFINITIONS = {"PTO_FDWIC_SHARED_TENSORMAP": "1"}

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/shared_builder_noeval_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "DUMMY_AIC",
                "source": "kernels/aic/dummy.cpp",
                "core_type": "aic",
                "signature": [],
            },
            {
                "func_id": 1,
                "name": "DUMMY_AIV",
                "source": "kernels/aiv/dummy.cpp",
                "core_type": "aiv",
                "signature": [],
            },
        ],
    }

    CASES = [
        {
            "name": "A5SimBd1",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 4},
        },
        {
            "name": "A5SimBd36",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 16},
        },
    ]

    def generate_args(self, params):
        n = int(params["n"])
        return TaskArgsBuilder(
            Tensor("sentinel", torch.arange(n, dtype=torch.float32)),
            Scalar("n", n),
        )

    def compute_golden(self, args, params):
        del args, params


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
