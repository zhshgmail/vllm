# SPDX-License-Identifier: Apache-2.0
"""Tests for unified weight update extension with both disk and NCCL sources.

This test consolidates and extends functionality from test_weight_update.py and 
test_worker_weight_update.py, following vLLM patterns with light mocks to validate 
the unified extension behavior without requiring CUDA or full engine modules.

Windows compatible - avoids Unix-specific features like signal handling and multiprocessing.
"""
import types
import pytest
from unittest.mock import MagicMock, patch
import torch


class DummyModelRunner:
    def __init__(self):
        self.model = MagicMock()
        self.kv_caches = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        self.requests = {"req1": "data1", "req2": "data2"}


class DummyWorker:
    def __init__(self):
        self.model_runner = DummyModelRunner()
        self.rank = 0
        self.device = torch.device('cpu')


def test_weight_update_extension_initialization():
    """Test that WeightUpdateExtension initializes properly."""
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    
    # Create dummy worker with extension
    worker = DummyWorker()
    extension = WeightUpdateExtension()
    
    # Bind extension attributes to worker (simulating worker extension pattern)
    for attr_name in ['weight_sources', 'update_stats', 'nccl_communicator', 'weight_update_in_progress']:
        setattr(worker, attr_name, getattr(extension, attr_name))
    
    # Bind extension methods to worker
    for attr_name in dir(extension):
        if not attr_name.startswith('_') and callable(getattr(extension, attr_name)):
            setattr(worker, attr_name, getattr(extension, attr_name).__get__(worker))
    
    # Test initialization
    assert hasattr(worker, 'weight_sources')
    assert hasattr(worker, 'update_stats')
    assert 'disk' in worker.weight_sources
    assert 'nccl' not in worker.weight_sources  # Not initialized yet


def test_disk_weight_source_success(monkeypatch):
    """Test disk weight source loading with mocks."""
    worker = DummyWorker()
    
    # Mock the disk weight loading
    called = {}
    def fake_stream_apply(model, path, pattern=None):
        called["args"] = (model, path, pattern)
        return 5  # Return number of updated parameters
    
    # Patch the utility function
    import vllm.worker._weight_update as weight_update_mod
    monkeypatch.setattr(weight_update_mod, "stream_apply_sharded_state", fake_stream_apply)
    
    # Mock torch.cuda.synchronize for Windows compatibility
    import torch.cuda
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None, raising=False)
    
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    extension = WeightUpdateExtension()
    
    # Bind extension methods to worker
    for attr in ['model_runner', 'rank', 'device']:
        setattr(extension, attr, getattr(worker, attr))
    
    # Test disk weight loading
    result = extension.load_sharded_state("/fake/path", "pattern-{rank}.safetensors")
    
    # Verify call was made
    assert called["args"] == (worker.model_runner.model, "/fake/path", "pattern-{rank}.safetensors")
    
    # Verify result
    assert result["ok"] is True
    assert result["rank"] == 0
    assert result["updated_params"] == 5
    assert result["source"] == "disk"
    assert "duration" in result


def test_disk_weight_validation_success(monkeypatch):
    """Test disk weight validation with mocks."""
    worker = DummyWorker()
    
    # Mock the validation function
    def fake_validate(model, path, pattern=None):
        return 10, []  # 10 tensors, no mismatches
    
    import vllm.worker._weight_update as weight_update_mod
    monkeypatch.setattr(weight_update_mod, "validate_sharded_state", fake_validate)
    
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    extension = WeightUpdateExtension()
    
    # Bind extension methods to worker
    for attr in ['model_runner', 'rank', 'device']:
        setattr(extension, attr, getattr(worker, attr))
    
    # Test validation
    result = extension.validate_sharded_state("/fake/path")
    
    # Verify result
    assert result["ok"] is True
    assert result["rank"] == 0
    assert result["tensor_count"] == 10
    assert result["mismatches"] == []


def test_disk_weight_validation_with_mismatches(monkeypatch):
    """Test disk weight validation with mismatches."""
    worker = DummyWorker()
    
    # Mock validation with mismatches
    def fake_validate(model, path, pattern=None):
        mismatches = [
            {"kind": "shape", "name": "layer.weight", "detail": "Shape mismatch"},
            {"kind": "type", "name": "layer.bias", "detail": "Type mismatch"}
        ]
        return 8, mismatches
    
    import vllm.worker._weight_update as weight_update_mod
    monkeypatch.setattr(weight_update_mod, "validate_sharded_state", fake_validate)
    
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    extension = WeightUpdateExtension()
    
    # Bind extension methods to worker
    for attr in ['model_runner', 'rank', 'device']:
        setattr(extension, attr, getattr(worker, attr))
    
    # Test validation
    result = extension.validate_sharded_state("/fake/path")
    
    # Verify result
    assert result["ok"] is True  # Still ok, just has mismatches
    assert result["rank"] == 0
    assert result["tensor_count"] == 8
    assert len(result["mismatches"]) == 2
    assert result["mismatches"][0]["kind"] == "shape"


def test_kv_cache_flushing_v1():
    """Test V1 KV cache flushing logic."""
    worker = DummyWorker()
    
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    extension = WeightUpdateExtension()
    
    # Bind extension methods to worker
    for attr in ['model_runner', 'rank', 'device']:
        setattr(extension, attr, getattr(worker, attr))
    
    # Store original values
    original_kv_values = [kv.clone() for kv in worker.model_runner.kv_caches]
    original_requests = dict(worker.model_runner.requests)
    
    # Flush cache
    extension._flush_kv_cache_unified()
    
    # Verify KV caches are zeroed
    for kv in worker.model_runner.kv_caches:
        assert torch.allclose(kv, torch.zeros_like(kv))
    
    # Verify requests are cleared
    assert len(worker.model_runner.requests) == 0
    
    # Verify original values were actually different
    for orig, kv in zip(original_kv_values, worker.model_runner.kv_caches):
        assert not torch.allclose(orig, kv)  # They should be different now


def test_kv_cache_flushing_v0():
    """Test V0 KV cache flushing logic."""
    # Create V0-style worker with cache_engine
    worker = DummyWorker()
    
    # Mock V0 cache structure
    mock_cache_engine = []
    mock_engine = MagicMock()
    mock_gpu_cache = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    mock_engine.gpu_cache = mock_gpu_cache
    mock_cache_engine.append(mock_engine)
    
    # Remove V1 structure and add V0 structure
    delattr(worker.model_runner, 'kv_caches')
    
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    extension = WeightUpdateExtension()
    
    # Bind extension methods to worker and add V0 cache structure
    for attr in ['model_runner', 'rank', 'device']:
        setattr(extension, attr, getattr(worker, attr))
    extension.cache_engine = mock_cache_engine
    extension._seq_group_metadata_cache = {"meta1": "data1"}
    
    # Store original values
    original_values = [cache.clone() for cache in mock_gpu_cache]
    
    # Flush cache
    extension._flush_kv_cache_unified()
    
    # Verify caches are zeroed
    for cache in mock_gpu_cache:
        assert torch.allclose(cache, torch.zeros_like(cache))
    
    # Verify metadata cache is cleared
    assert len(extension._seq_group_metadata_cache) == 0
    
    # Verify original values were different
    for orig, cache in zip(original_values, mock_gpu_cache):
        assert not torch.allclose(orig, cache)


def test_nccl_weight_source_initialization():
    """Test NCCL weight source initialization."""
    worker = DummyWorker()
    
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    extension = WeightUpdateExtension()
    
    # Bind extension methods to worker
    for attr in ['model_runner', 'rank', 'device']:
        setattr(extension, attr, getattr(worker, attr))
    
    # Mock NCCL initialization components
    with patch('vllm.distributed.parallel_state.get_world_group') as mock_world_group, \
         patch('vllm.distributed.utils.StatelessProcessGroup.create') as mock_pg_create, \
         patch('vllm.distributed.device_communicators.pynccl.PyNcclCommunicator') as mock_nccl, \
         patch('vllm.worker.weight_update_extension.StatelessProcessGroup') as mock_spg, \
         patch('vllm.worker.weight_update_extension.PyNcclCommunicator') as mock_pynccl:
        
        # Setup mocks
        mock_world_group.return_value.rank = 1
        mock_pg = MagicMock()
        mock_pg_create.return_value = mock_pg
        mock_spg.create = mock_pg_create
        mock_nccl_instance = MagicMock()
        mock_nccl.return_value = mock_nccl_instance
        mock_pynccl.return_value = mock_nccl_instance
        
        # Test NCCL initialization
        result = extension.init_nccl_weight_source("localhost", 29500, 1, 2)
        
        # Verify initialization succeeded
        assert result is True
        assert 'nccl' in extension.weight_sources
        assert extension.nccl_communicator is not None
        
        # Verify calls were made correctly
        mock_pg_create.assert_called_once_with(
            host="localhost", 
            port=29500,
            rank=2,  # base_rank (1) + rank_offset (1)
            world_size=2
        )
        mock_pynccl.assert_called_once_with(mock_pg, device=worker.device)


def test_nccl_weight_loading(monkeypatch):
    """Test NCCL weight loading with mocks."""
    worker = DummyWorker()
    
    # Mock torch.cuda operations for Windows compatibility
    import torch.cuda
    mock_stream = MagicMock()
    mock_stream.synchronize = MagicMock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: mock_stream, raising=False)
    
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    extension = WeightUpdateExtension()
    
    # Bind extension methods to worker
    for attr in ['model_runner', 'rank', 'device']:
        setattr(extension, attr, getattr(worker, attr))
    
    # Create mock model with parameters
    mock_model = MagicMock()
    mock_param1 = MagicMock(spec=torch.nn.Parameter)
    mock_param1.requires_grad = True
    mock_param1.data = torch.tensor([1.0, 2.0])
    mock_param2 = MagicMock(spec=torch.nn.Parameter) 
    mock_param2.requires_grad = True
    mock_param2.data = torch.tensor([3.0, 4.0])
    
    mock_model.named_parameters.return_value = [
        ("layer1.weight", mock_param1),
        ("layer2.bias", mock_param2)
    ]
    worker.model_runner.model = mock_model
    
    # Mock NCCL communicator
    mock_communicator = MagicMock()
    mock_communicator.broadcast = MagicMock()
    
    # Add NCCL source manually
    from vllm.worker.weight_update_extension import NCCLWeightSource
    extension.weight_sources["nccl"] = NCCLWeightSource(mock_communicator)
    extension.nccl_communicator = mock_communicator
    
    # Test NCCL weight loading
    result = extension.load_weights_nccl()
    
    # Verify result
    assert result["ok"] is True
    assert result["rank"] == 0
    assert result["updated_params"] == 2
    assert result["source"] == "nccl"
    assert "duration" in result
    
    # Verify NCCL broadcast was called for each parameter
    assert mock_communicator.broadcast.call_count == 2


def test_weight_update_stats():
    """Test weight update statistics collection."""
    worker = DummyWorker()
    
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    extension = WeightUpdateExtension()
    
    # Bind extension methods to worker
    for attr in ['model_runner', 'rank', 'device']:
        setattr(extension, attr, getattr(worker, attr))
    
    # Simulate some updates
    extension.update_stats["total_updates"] = 5
    extension.update_stats["successful_updates"] = 4
    extension.update_stats["last_update_time"] = 1234567890.0
    extension.update_stats["last_update_source"] = "disk"
    extension.weight_update_in_progress = True
    
    # Get stats
    stats = extension.get_weight_update_stats()
    
    # Verify stats
    assert stats["total_updates"] == 5
    assert stats["successful_updates"] == 4
    assert stats["last_update_time"] == 1234567890.0
    assert stats["last_update_source"] == "disk"
    assert stats["weight_update_in_progress"] is True
    assert stats["available_sources"] == ["disk"]
    assert stats["nccl_initialized"] is False
    assert stats["worker_rank"] == 0


def test_error_handling_in_disk_loading(monkeypatch):
    """Test error handling in disk weight loading."""
    worker = DummyWorker()
    
    # Mock the disk weight loading to raise an error
    def fake_stream_apply_error(model, path, pattern=None):
        raise RuntimeError("Disk loading failed")
    
    import vllm.worker._weight_update as weight_update_mod
    monkeypatch.setattr(weight_update_mod, "stream_apply_sharded_state", fake_stream_apply_error)
    
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    extension = WeightUpdateExtension()
    
    # Bind extension methods to worker
    for attr in ['model_runner', 'rank', 'device']:
        setattr(extension, attr, getattr(worker, attr))
    
    # Test disk weight loading with error
    result = extension.load_sharded_state("/fake/path")
    
    # Verify error is handled
    assert result["ok"] is False
    assert result["rank"] == 0
    assert "Disk loading failed" in result["error"]


def test_error_handling_in_validation(monkeypatch):
    """Test error handling in weight validation."""
    worker = DummyWorker()
    
    # Mock validation to raise an error
    def fake_validate_error(model, path, pattern=None):
        raise ValueError("Validation error")
    
    import vllm.worker._weight_update as weight_update_mod
    monkeypatch.setattr(weight_update_mod, "validate_sharded_state", fake_validate_error)
    
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    extension = WeightUpdateExtension()
    
    # Bind extension methods to worker
    for attr in ['model_runner', 'rank', 'device']:
        setattr(extension, attr, getattr(worker, attr))
    
    # Test validation with error
    result = extension.validate_sharded_state("/fake/path")
    
    # Verify error is handled
    assert result["ok"] is False
    assert result["rank"] == 0
    assert "Validation error" in result["error"]
    assert result["tensor_count"] == 0
    assert len(result["mismatches"]) == 1
    assert result["mismatches"][0]["kind"] == "error"


def test_weight_update_in_progress_protection():
    """Test that concurrent weight updates are prevented."""
    worker = DummyWorker()
    
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    extension = WeightUpdateExtension()
    
    # Bind extension methods to worker
    for attr in ['model_runner', 'rank', 'device']:
        setattr(extension, attr, getattr(worker, attr))
    
    # Set update in progress
    extension.weight_update_in_progress = True
    
    # Try to start another update
    result = extension.load_sharded_state("/fake/path")
    
    # Verify it's rejected
    assert result["ok"] is False
    assert "Update already in progress" in result["error"]
    
    # Test same for NCCL - need to initialize NCCL source first or expect different error
    result = extension.load_weights_nccl()
    assert result["ok"] is False
    # Could be either "NCCL weight source not initialized" or "Update already in progress"
    assert ("NCCL weight source not initialized" in result["error"] or 
            "Update already in progress" in result["error"])


def test_cleanup_resources():
    """Test resource cleanup."""
    worker = DummyWorker()
    
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    extension = WeightUpdateExtension()
    
    # Bind extension methods to worker
    for attr in ['model_runner', 'rank', 'device']:
        setattr(extension, attr, getattr(worker, attr))
    
    # Add mock NCCL communicator
    mock_communicator = MagicMock()
    extension.nccl_communicator = mock_communicator
    extension.weight_sources["nccl"] = MagicMock()
    
    # Test cleanup
    extension.cleanup_weight_sources()
    
    # Verify cleanup
    assert extension.nccl_communicator is None
    assert "nccl" not in extension.weight_sources


# Additional tests consolidated from test_weight_update.py and test_worker_weight_update.py

def test_basic_weight_loading_consolidated():
    """Test basic weight loading (consolidated from test_weight_update.py)."""
    from vllm.worker.weight_update_extension import DiskWeightSource
    
    class MockModel:
        def __init__(self):
            self.loaded_weights = []
            
    class MockTensor:
        def __init__(self, shape):
            self.shape = shape
            
    # Mock the stream_apply_sharded_state function
    import vllm.worker._weight_update as weight_update_mod
    original_func = getattr(weight_update_mod, 'stream_apply_sharded_state', None)
    
    def mock_stream_apply(model, path, pattern=None):
        return 2  # Return count of updated parameters
    
    weight_update_mod.stream_apply_sharded_state = mock_stream_apply
    
    try:
        source = DiskWeightSource()
        mock_model = MockModel()
        
        updated_params, metadata = source.load_weights(mock_model, path="/fake/path")
        
        assert updated_params == 2
        assert metadata["source"] == "disk"
        assert metadata["path"] == "/fake/path"
    finally:
        # Restore original function if it existed
        if original_func is not None:
            weight_update_mod.stream_apply_sharded_state = original_func
        elif hasattr(weight_update_mod, 'stream_apply_sharded_state'):
            delattr(weight_update_mod, 'stream_apply_sharded_state')


def test_worker_wrapper_behavior(monkeypatch):
    """Test worker wrapper behavior (consolidated from test_worker_weight_update.py)."""
    worker = DummyWorker()
    
    from vllm.worker.weight_update_extension import WeightUpdateExtension
    extension = WeightUpdateExtension()
    
    # Bind extension methods to worker
    for attr in ['model_runner', 'rank', 'device']:
        setattr(extension, attr, getattr(worker, attr))
    
    # Mock the underlying weight loading
    import vllm.worker._weight_update as weight_update_mod
    def mock_stream_apply(model, path, pattern=None):
        return 3  # Return count of updated parameters
    
    monkeypatch.setattr(weight_update_mod, 'stream_apply_sharded_state', mock_stream_apply)
    
    # Mock torch.cuda.synchronize for Windows compatibility  
    import torch.cuda
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None, raising=False)
    
    result = extension.load_sharded_state("/ckpt", pattern="abc")
    
    assert result["ok"] is True
    assert result["rank"] == 0
    assert result["updated_params"] == 3
    assert result["source"] == "disk"
    assert "duration" in result


if __name__ == "__main__":
    pytest.main([__file__])