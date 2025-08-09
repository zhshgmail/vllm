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

    # We'll bind the real method from module onto this instance for testing.


def test_worker_load_sharded_state_success(monkeypatch):
    dummy_worker = DummyWorker()

    # Patch the utility used by worker to simulate two tensors updated
    called = {}

    def fake_stream_apply(model, path, pattern=None):  # noqa: D401
        called["args"] = (model, path, pattern)
        return 2

    monkeypatch.setattr(worker_mod, "stream_apply_sharded_state", fake_stream_apply, raising=True)

    # Bind method
    method = worker_mod.Worker.load_sharded_state.__get__(dummy_worker, DummyWorker)
    result = method("/ckpt", pattern="abc")

    assert result["ok"] is True
    assert result["updated_tensors"] == 2
    assert called["args"][1] == "/ckpt"
    assert called["args"][2] == "abc"


def test_worker_load_sharded_state_error(monkeypatch):
    dummy_worker = DummyWorker()

    def fake_stream_apply(model, path, pattern=None):  # noqa: D401
        raise RuntimeError("boom")

    monkeypatch.setattr(worker_mod, "stream_apply_sharded_state", fake_stream_apply, raising=True)

    method = worker_mod.Worker.load_sharded_state.__get__(dummy_worker, DummyWorker)
    result = method("/ckpt")

    assert result["ok"] is False
    assert "boom" in result["error"]
