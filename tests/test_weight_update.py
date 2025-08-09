# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.worker._weight_update.stream_apply_sharded_state.

These tests are CPU-only and mock all external dependencies so they can run
in minimal environments (no CUDA, no distributed init, no real safetensors).
"""
import types
import pytest
import sys
import types
import os
import importlib.util


class DummyModel:
    def __init__(self):
        self.updates = []  # list of (name, tensor)

    def load_weights(self, weights):  # signature: list[(name, tensor)]
        self.updates.extend(weights)


class DummyLoader:
    DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self, load_cfg):  # load_cfg ignored
        self.pattern = self.DEFAULT_PATTERN
        self.iter_calls = 0

    # Will be monkeypatched per test to return desired tensors
    def iterate_over_files(self, filepaths):  # pragma: no cover - replaced in tests
        yield from ()


class DummyLoadConfig:
    def __init__(self, load_format, model_loader_extra_config):  # noqa: D401
        self.load_format = load_format
        self.model_loader_extra_config = model_loader_extra_config


@pytest.fixture()
def wu(monkeypatch):
    """Provide the loaded weight update module with faked dependencies."""
    captured = {"patterns": [], "files": []}

    # Create minimal fake package hierarchy for vllm.* referenced imports
    vllm_pkg = types.ModuleType("vllm")
    vllm_pkg.__path__ = []  # mark as package
    sys.modules.setdefault("vllm", vllm_pkg)

    def ensure_pkg(name):
        if name in sys.modules:
            return sys.modules[name]
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
        return mod

    ensure_pkg("vllm.model_executor")
    ensure_pkg("vllm.model_executor.model_loader")
    ensure_pkg("vllm.transformers_utils")

    config_mod = types.ModuleType("vllm.config")
    config_mod.LoadConfig = DummyLoadConfig
    sys.modules[config_mod.__name__] = config_mod

    dist_mod = types.ModuleType("vllm.distributed")
    dist_mod.get_tensor_model_parallel_rank = lambda: 0
    sys.modules[dist_mod.__name__] = dist_mod

    sharded_loader_mod = types.ModuleType(
        "vllm.model_executor.model_loader.sharded_state_loader")
    sharded_loader_mod.ShardedStateLoader = DummyLoader
    sys.modules[sharded_loader_mod.__name__] = sharded_loader_mod

    s3_utils_mod = types.ModuleType("vllm.transformers_utils.s3_utils")
    s3_utils_mod.glob = lambda path, allow_pattern: captured["files"]
    sys.modules[s3_utils_mod.__name__] = s3_utils_mod

    utils_mod = types.ModuleType("vllm.transformers_utils.utils")
    utils_mod.is_s3 = lambda _p: False
    sys.modules[utils_mod.__name__] = utils_mod

    # Patch glob.glob
    import glob as real_glob

    orig_glob_fn = real_glob.glob

    def fake_glob(pattern):
        captured["patterns"].append(pattern)
        return captured["files"]

    monkeypatch.setattr(real_glob, "glob", fake_glob, raising=True)

    # Load module after fakes are in place
    mod_path = os.path.join(os.path.dirname(__file__), "..", "vllm", "worker", "_weight_update.py")
    mod_path = os.path.abspath(mod_path)
    spec = importlib.util.spec_from_file_location("weight_update_unit", mod_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore

    # attach helper data for assertions
    module._captured = captured  # type: ignore[attr-defined]
    return module


class FakeTensor:  # minimal stand-in so we don't depend on torch
    def __init__(self, shape):
        self.shape = shape


def test_stream_apply_sharded_state_success(wu, monkeypatch):
    # Arrange: pretend we have one shard file
    wu._captured["files"] = ["/tmp/checkpoint/model-rank-0-part-0.safetensors"]  # type: ignore[attr-defined]

    tensors = [
        ("layer.weight", FakeTensor((2, 3))),
        ("layer.bias", FakeTensor((3,))),
    ]

    def iter_over_files(self, filepaths):  # self is DummyLoader
        assert filepaths == wu._captured["files"]  # type: ignore[attr-defined]
        for k, v in tensors:
            yield k, v

    monkeypatch.setattr(DummyLoader, "iterate_over_files", iter_over_files, raising=True)

    model = DummyModel()
    updated = wu.stream_apply_sharded_state(model, path="/tmp/checkpoint")

    assert updated == len(tensors)
    assert [n for n, _ in model.updates] == [t[0] for t in tensors]
    # Ensure glob pattern built as expected
    glob_patterns = wu._captured["patterns"]  # type: ignore[attr-defined]
    assert any("model-rank-0-part-*" in p for p in glob_patterns)


def test_stream_apply_sharded_state_pattern_override(wu, monkeypatch):
    wu._captured["files"] = ["/tmp/ckpt/custom-r0-p0.safetensors"]  # type: ignore[attr-defined]

    # Capture loader.pattern after override
    seen_patterns = {}

    def custom_init(self, load_cfg):  # override __init__ of DummyLoader
        self.pattern = "IGNORED"  # will be replaced by override logic in function

    monkeypatch.setattr(DummyLoader, "__init__", custom_init, raising=True)

    def iter_over_files(self, filepaths):
        seen_patterns["pattern"] = self.pattern
        yield "w", FakeTensor((1,))

    monkeypatch.setattr(DummyLoader, "iterate_over_files", iter_over_files, raising=True)

    model = DummyModel()
    custom_pattern = "custom-r{rank}-p{part}.safetensors"
    wu.stream_apply_sharded_state(model, path="/tmp/ckpt", pattern=custom_pattern)

    assert seen_patterns["pattern"] == custom_pattern
    assert len(model.updates) == 1


def test_stream_apply_sharded_state_no_files(wu):
    # No files returned by glob => expect ValueError
    model = DummyModel()
    with pytest.raises(ValueError, match="No shards found"):
        wu.stream_apply_sharded_state(model, path="/empty")
    assert model.updates == []
