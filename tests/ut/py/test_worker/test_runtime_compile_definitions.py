# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace

from simpler import worker as worker_module
from simpler.worker import Worker

import simpler_setup.runtime_builder as rb_module


class _FakeChipWorker:
    def __init__(self):
        self.init_calls = []
        self.prepare_calls = []

    def init(self, device_id, binaries):
        self.init_calls.append((device_id, binaries))

    def _prepare_callable_at_slot(self, cid, target):
        self.prepare_calls.append((cid, target))


def _patch_runtime_builder(monkeypatch):
    calls = []
    binaries = SimpleNamespace(host_path="host", aicpu_path="aicpu", aicore_path="aicore")

    class FakeRuntimeBuilder:
        def __init__(self, platform):
            self.platform = platform

        def get_binaries(self, runtime, build=False, compile_definitions=None):
            calls.append((self.platform, runtime, build, compile_definitions))
            return binaries

    monkeypatch.setattr(rb_module, "RuntimeBuilder", FakeRuntimeBuilder)
    monkeypatch.setattr(worker_module, "ChipWorker", _FakeChipWorker)
    return calls, binaries


def test_level2_worker_builds_runtime_for_compile_definitions(monkeypatch):
    calls, binaries = _patch_runtime_builder(monkeypatch)

    defs = {"PTO_FDWIC_SHARED_TENSORMAP": "1"}
    w = Worker(
        level=2,
        device_id=7,
        platform="a5sim",
        runtime="fully_distributed_within_core",
        runtime_compile_definitions=defs,
    )
    w.init()

    assert calls == [("a5sim", "fully_distributed_within_core", True, defs)]
    assert isinstance(w._chip_worker, _FakeChipWorker)
    assert w._chip_worker.init_calls == [(7, binaries)]


def test_level2_worker_uses_prebuilt_runtime_without_compile_definitions(monkeypatch):
    calls, binaries = _patch_runtime_builder(monkeypatch)

    w = Worker(level=2, device_id=3, platform="a5sim", runtime="fully_distributed_within_core")
    w.init()

    assert calls == [("a5sim", "fully_distributed_within_core", False, None)]
    assert isinstance(w._chip_worker, _FakeChipWorker)
    assert w._chip_worker.init_calls == [(3, binaries)]
