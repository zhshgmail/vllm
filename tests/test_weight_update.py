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
        self.failing_params = set()  # Parameters that should fail to load
        self._parameters = {
            "layer.weight": FakeTensor((2, 3)),
            "layer.bias": FakeTensor((3,)),
        }

    def named_parameters(self, recurse=True):
        """Mock named_parameters method expected by weight update code."""
        return self._parameters.items()

    def load_weights(self, weights):  # signature: list[(name, tensor)]
        for name, tensor in weights:
            if (hasattr(tensor, 'fail_on_load') and tensor.fail_on_load) or name in self.failing_params:
                raise RuntimeError(f"Simulated load failure for {name}")
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
    def __init__(self, shape, fail_on_load=False):
        self.shape = shape
        self.fail_on_load = fail_on_load
        self._is_contiguous = True
    
    def is_contiguous(self):
        return self._is_contiguous
    
    def contiguous(self):
        self._is_contiguous = True
        return self


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


def test_stream_apply_sharded_state_partial_failures_low_rate(wu, monkeypatch):
    """Test weight loading with some failures but low failure rate (should succeed)."""
    wu._captured["files"] = ["/tmp/checkpoint/model-rank-0-part-0.safetensors"]  # type: ignore[attr-defined]

    # 10 tensors, 1 fails = 10% failure rate (exactly at threshold, should pass)
    tensors = [
        ("good_param_1", FakeTensor((2, 3))),
        ("failing_param", FakeTensor((2, 3), fail_on_load=True)),
        ("good_param_2", FakeTensor((3, 4))),
        ("good_param_3", FakeTensor((4, 5))),
        ("good_param_4", FakeTensor((5, 6))),
        ("good_param_5", FakeTensor((6, 7))),
        ("good_param_6", FakeTensor((7, 8))),
        ("good_param_7", FakeTensor((8, 9))),
        ("good_param_8", FakeTensor((9, 10))),
        ("good_param_9", FakeTensor((10, 11))),
    ]

    def iter_over_files(self, filepaths):
        for k, v in tensors:
            yield k, v

    monkeypatch.setattr(DummyLoader, "iterate_over_files", iter_over_files, raising=True)

    model = DummyModel()
    updated = wu.stream_apply_sharded_state(model, path="/tmp/checkpoint")

    # Should succeed with 9 successful updates (1 failed)
    assert updated == 9
    successful_params = [n for n, _ in model.updates]
    assert "good_param_1" in successful_params
    assert "good_param_2" in successful_params
    assert "failing_param" not in successful_params


def test_stream_apply_sharded_state_high_failure_rate(wu, monkeypatch):
    """Test weight loading with high failure rate (should fail)."""
    wu._captured["files"] = ["/tmp/checkpoint/model-rank-0-part-0.safetensors"]  # type: ignore[attr-defined]

    # 10 tensors, 2 fail = 20% failure rate (above 10% threshold, should fail)
    tensors = [
        ("good_param_1", FakeTensor((2, 3))),
        ("failing_param_1", FakeTensor((2, 3), fail_on_load=True)),
        ("good_param_2", FakeTensor((3, 4))),
        ("failing_param_2", FakeTensor((3, 4), fail_on_load=True)),
        ("good_param_3", FakeTensor((4, 5))),
        ("good_param_4", FakeTensor((5, 6))),
        ("good_param_5", FakeTensor((6, 7))),
        ("good_param_6", FakeTensor((7, 8))),
        ("good_param_7", FakeTensor((8, 9))),
        ("good_param_8", FakeTensor((9, 10))),
    ]

    def iter_over_files(self, filepaths):
        for k, v in tensors:
            yield k, v

    monkeypatch.setattr(DummyLoader, "iterate_over_files", iter_over_files, raising=True)

    model = DummyModel()
    with pytest.raises(RuntimeError, match="Too many parameter loading failures"):
        wu.stream_apply_sharded_state(model, path="/tmp/checkpoint")


def test_stream_apply_sharded_state_all_failures(wu, monkeypatch):
    """Test weight loading where all parameters fail (should fail)."""
    wu._captured["files"] = ["/tmp/checkpoint/model-rank-0-part-0.safetensors"]  # type: ignore[attr-defined]

    tensors = [
        ("failing_param_1", FakeTensor((2, 3), fail_on_load=True)),
        ("failing_param_2", FakeTensor((3, 4), fail_on_load=True)),
    ]

    def iter_over_files(self, filepaths):
        for k, v in tensors:
            yield k, v

    monkeypatch.setattr(DummyLoader, "iterate_over_files", iter_over_files, raising=True)

    model = DummyModel()
    # With all failures, it should fail the 10% threshold check first
    with pytest.raises(RuntimeError, match="Too many parameter loading failures"):
        wu.stream_apply_sharded_state(model, path="/tmp/checkpoint")


def test_stream_apply_sharded_state_non_contiguous_tensors(wu, monkeypatch):
    """Test weight loading with non-contiguous tensors (should make contiguous)."""
    wu._captured["files"] = ["/tmp/checkpoint/model-rank-0-part-0.safetensors"]  # type: ignore[attr-defined]

    # Create non-contiguous tensor
    tensor = FakeTensor((2, 3))
    tensor._is_contiguous = False

    tensors = [("layer.weight", tensor)]

    def iter_over_files(self, filepaths):
        for k, v in tensors:
            yield k, v

    monkeypatch.setattr(DummyLoader, "iterate_over_files", iter_over_files, raising=True)

    model = DummyModel()
    updated = wu.stream_apply_sharded_state(model, path="/tmp/checkpoint")

    assert updated == 1
    # Tensor should have been made contiguous
    assert tensor.is_contiguous() is True


def test_stream_apply_sharded_state_iteration_failure(wu, monkeypatch):
    """Test weight loading when file iteration itself fails."""
    wu._captured["files"] = ["/tmp/checkpoint/model-rank-0-part-0.safetensors"]  # type: ignore[attr-defined]

    def iter_over_files(self, filepaths):
        raise IOError("Failed to read shard file")

    monkeypatch.setattr(DummyLoader, "iterate_over_files", iter_over_files, raising=True)

    model = DummyModel()
    with pytest.raises(RuntimeError, match="Failed to iterate over weight files"):
        wu.stream_apply_sharded_state(model, path="/tmp/checkpoint")
