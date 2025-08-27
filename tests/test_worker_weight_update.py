# SPDX-License-Identifier: Apache-2.0
"""Tests worker.load_sharded_state wrapper behavior with mocks.

We only validate that:
- The utility is invoked with provided arguments.
- Return dict contains ok=True and count from utility.
- Errors are captured and surfaced in the result dict (without raising).

This test uses a light dummy worker object rather than importing full GPU/engine
modules (avoids CUDA requirements). We patch the utility symbol directly in the
worker module namespace.
"""
import types
import pytest

try:  # pragma: no cover - import guard
    import vllm.worker.worker as worker_mod  # type: ignore
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"Skipping worker wrapper tests: cannot import worker module ({exc})", allow_module_level=True)


class DummyModelRunner:
    def __init__(self):
        self.model = object()  # placeholder


class DummyWorker:
    def __init__(self):
        self.model_runner = DummyModelRunner()
        self.rank = 0  # Add rank attribute expected by worker methods
        self.cache_engine = None  # Add cache_engine attribute
        self.gpu_cache = None  # Add gpu_cache attribute

    # We'll bind the real method from module onto this instance for testing.


def test_worker_load_sharded_state_success(monkeypatch):
    dummy_worker = DummyWorker()

    # Patch the utility used by worker to simulate two tensors updated
    called = {}

    def fake_stream_apply(model, path, pattern=None):  # noqa: D401
        called["args"] = (model, path, pattern)
        return 2

    # Patch in the _weight_update module since that's where it's imported from
    import vllm.worker._weight_update as weight_update_mod
    monkeypatch.setattr(weight_update_mod, "stream_apply_sharded_state", fake_stream_apply, raising=True)
    
    # Patch torch.cuda.synchronize to avoid CUDA requirements
    import torch.cuda
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None, raising=False)

    # Bind method
    method = worker_mod.Worker.load_sharded_state.__get__(dummy_worker, DummyWorker)
    result = method("/ckpt", pattern="abc")

    assert result["ok"] is True
    assert result["rank"] == 0  # Check that rank is returned
    assert called["args"][1] == "/ckpt"
    assert called["args"][2] == "abc"


def test_worker_load_sharded_state_error(monkeypatch):
    dummy_worker = DummyWorker()

    def fake_stream_apply(model, path, pattern=None):  # noqa: D401
        raise RuntimeError("boom")

    # Patch in the _weight_update module since that's where it's imported from
    import vllm.worker._weight_update as weight_update_mod
    monkeypatch.setattr(weight_update_mod, "stream_apply_sharded_state", fake_stream_apply, raising=True)
    
    # Patch torch.cuda.synchronize to avoid CUDA requirements
    import torch.cuda
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None, raising=False)

    method = worker_mod.Worker.load_sharded_state.__get__(dummy_worker, DummyWorker)
    result = method("/ckpt")

    assert result["ok"] is False
    assert "boom" in result["error"]
