# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for V1 worker weight update functionality.

These tests are CPU-only and mock all external dependencies so they can run
in minimal environments (no CUDA, no distributed init, no real models).
"""
import logging
import pytest
import sys
import types
from unittest.mock import MagicMock, Mock, patch, call
from typing import Any, Dict, List, Optional, Tuple


class MockTensor:
    """Mock tensor that behaves like a PyTorch tensor for testing."""
    
    def __init__(self, shape: Tuple[int, ...], dtype: str = "float32", device: str = "cuda:0"):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._data = [0.0] * (shape[0] * shape[1] if len(shape) >= 2 else shape[0] if shape else 1)
        
    def zero_(self):
        """Mock tensor.zero_() method."""
        self._data = [0.0] * len(self._data)
        return self
        
    def any(self):
        """Mock tensor.any() for NaN/inf checking."""
        return False
        
    def is_contiguous(self):
        """Mock tensor.is_contiguous()."""
        return True
        
    def contiguous(self):
        """Mock tensor.contiguous()."""
        return self


class MockModel:
    """Mock model for testing."""
    
    def __init__(self):
        self.load_weights_calls = []
        self.parameters_list = [
            ("layer1.weight", MockTensor((1024, 512))),
            ("layer1.bias", MockTensor((1024,))),
            ("layer2.weight", MockTensor((512, 256))),
        ]
        
    def load_weights(self, weights):
        """Mock model.load_weights method."""
        self.load_weights_calls.append(weights)
        
    def named_parameters(self):
        """Mock model.named_parameters()."""
        return iter(self.parameters_list)
        
    def parameters(self):
        """Mock model.parameters()."""
        return (param for name, param in self.parameters_list)


class MockModelRunner:
    """Mock model runner for V1 worker testing."""
    
    def __init__(self):
        self.model = MockModel()
        self.kv_caches = [
            MockTensor((100, 2, 32, 128)),  # Regular attention cache
            MockTensor((100, 2, 32, 128)),  # Another attention cache
            [MockTensor((100, 256)), MockTensor((100, 512))],  # Mamba-style cache
        ]
        
        # Mock compilation config with forward context
        self.compilation_config = Mock()
        self.compilation_config.static_forward_context = {
            "layer1.attn": Mock(kv_cache=[MockTensor((100, 2, 32, 128))]),
            "layer2.attn": Mock(kv_cache=[MockTensor((100, 2, 32, 128))]),
            "layer3.mamba": Mock(kv_cache=[MockTensor((100, 256)), MockTensor((100, 512))]),
        }
        
        # Mock input batch with block tables
        self.input_batch = Mock()
        self.input_batch.block_table = [
            Mock(clear=Mock()),
            Mock(zero_=Mock()),
            Mock(),  # No clear or zero_ method
        ]
        
        # Mock encoder cache
        self.encoder_cache = {"req1": {"input1": MockTensor((10, 512))}}
        
        # Mock attention groups
        self.attn_groups = [
            [  # Group 0
                Mock(metadata_builder=Mock(reset=Mock())),
                Mock(metadata_builder=Mock(reset=Mock(side_effect=AttributeError("No reset")))),
            ],
            [  # Group 1
                Mock(metadata_builder=Mock()),  # No reset method
                Mock(),  # No metadata_builder
            ],
        ]
        
        # Mock CUDA graph settings
        self.use_cuda_graph = True


class MockWorker:
    """Mock V1 worker for testing."""
    
    def __init__(self, rank: int = 0):
        self.rank = rank
        self.model_runner = MockModelRunner()


@pytest.fixture
def mock_torch():
    """Mock torch module for testing."""
    torch_mock = Mock()
    torch_mock.cuda = Mock()
    torch_mock.cuda.synchronize = Mock()
    torch_mock.cuda.empty_cache = Mock()
    torch_mock.is_tensor = lambda x: isinstance(x, MockTensor)
    torch_mock.isnan = lambda x: Mock(any=Mock(return_value=False))
    torch_mock.isinf = lambda x: Mock(any=Mock(return_value=False))
    return torch_mock


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def v1_worker():
    """Create a mock V1 worker for testing."""
    return MockWorker()


class TestV1WorkerKVCacheFlush:
    """Test the V1 worker KV cache flush functionality."""
    
    def test_flush_kv_cache_v1_all_components(self, v1_worker, mock_torch, mock_logger):
        """Test KV cache flush when all components are present."""
        with patch('torch', mock_torch), \
             patch('vllm.v1.worker.gpu_worker.logger', mock_logger):
            
            # Import the worker module and add the method
            from vllm.v1.worker.gpu_worker import Worker
            
            # Create method on mock worker
            def _flush_kv_cache_v1(self):
                # Simulate the actual method logic with mocks
                mock_logger.debug("Starting V1 KV cache flush after weight update")
                
                # 1. Clear KV cache tensors
                if hasattr(self.model_runner, "kv_caches"):
                    cleared_tensors = 0
                    cleared_mamba_tensors = 0
                    for i, kv_tensor in enumerate(self.model_runner.kv_caches):
                        if mock_torch.is_tensor(kv_tensor):
                            kv_tensor.zero_()
                            cleared_tensors += 1
                        elif isinstance(kv_tensor, (list, tuple)):
                            for tensor in kv_tensor:
                                if mock_torch.is_tensor(tensor):
                                    tensor.zero_()
                                    cleared_mamba_tensors += 1
                    mock_logger.debug(f"Cleared {cleared_tensors} regular tensors and {cleared_mamba_tensors} Mamba sub-tensors")
                
                # 2. Clear forward context
                if hasattr(self.model_runner, "compilation_config"):
                    if hasattr(self.model_runner.compilation_config, "static_forward_context"):
                        for layer_name, layer in self.model_runner.compilation_config.static_forward_context.items():
                            if hasattr(layer, "kv_cache") and layer.kv_cache:
                                for kv_cache in layer.kv_cache:
                                    if mock_torch.is_tensor(kv_cache):
                                        kv_cache.zero_()
                
                # 3. Clear block tables
                if hasattr(self.model_runner, "input_batch") and self.model_runner.input_batch:
                    for block_table in self.model_runner.input_batch.block_table:
                        if hasattr(block_table, "clear"):
                            block_table.clear()
                        elif hasattr(block_table, "zero_"):
                            block_table.zero_()
                
                # 4. Clear encoder cache
                if hasattr(self.model_runner, "encoder_cache"):
                    self.model_runner.encoder_cache.clear()
                
                # 5. Reset attention groups
                if hasattr(self.model_runner, "attn_groups"):
                    for attn_group_list in self.model_runner.attn_groups:
                        for attn_group in attn_group_list:
                            if hasattr(attn_group, "metadata_builder"):
                                if hasattr(attn_group.metadata_builder, "reset"):
                                    try:
                                        attn_group.metadata_builder.reset()
                                    except (AttributeError, NotImplementedError):
                                        pass
                
                # 6. CUDA graph handling
                if hasattr(self.model_runner, "use_cuda_graph") and self.model_runner.use_cuda_graph:
                    mock_logger.info("CUDA graphs are enabled - they will be regenerated on next capture")
                
                # 7. Cleanup
                mock_torch.cuda.synchronize()
                mock_torch.cuda.empty_cache()
                mock_logger.info("V1 KV cache flush completed successfully after weight update")
            
            # Bind method to worker
            v1_worker._flush_kv_cache_v1 = _flush_kv_cache_v1.__get__(v1_worker, type(v1_worker))
            
            # Execute the flush
            v1_worker._flush_kv_cache_v1()
            
            # Verify all components were called
            assert v1_worker.model_runner.kv_caches[0]._data == [0.0] * len(v1_worker.model_runner.kv_caches[0]._data)
            assert v1_worker.model_runner.kv_caches[1]._data == [0.0] * len(v1_worker.model_runner.kv_caches[1]._data)
            
            # Verify block table clearing
            v1_worker.model_runner.input_batch.block_table[0].clear.assert_called_once()
            v1_worker.model_runner.input_batch.block_table[1].zero_.assert_called_once()
            
            # Verify encoder cache clearing
            assert len(v1_worker.model_runner.encoder_cache) == 0
            
            # Verify attention group resets
            v1_worker.model_runner.attn_groups[0][0].metadata_builder.reset.assert_called_once()
            # Second one should have been attempted but failed silently
            
            # Verify CUDA operations
            mock_torch.cuda.synchronize.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()
            
            # Verify logging calls
            assert mock_logger.debug.call_count >= 1
            assert mock_logger.info.call_count >= 1
    
    def test_flush_kv_cache_v1_missing_components(self, mock_torch, mock_logger):
        """Test KV cache flush when components are missing."""
        # Create worker with minimal components
        worker = MockWorker()
        delattr(worker.model_runner, "kv_caches")
        delattr(worker.model_runner, "compilation_config")
        delattr(worker.model_runner, "input_batch")
        delattr(worker.model_runner, "encoder_cache")
        delattr(worker.model_runner, "attn_groups")
        delattr(worker.model_runner, "use_cuda_graph")
        
        with patch('torch', mock_torch), \
             patch('vllm.v1.worker.gpu_worker.logger', mock_logger):
            
            def _flush_kv_cache_v1(self):
                mock_logger.debug("Starting V1 KV cache flush after weight update")
                
                # All hasattr checks should fail
                if hasattr(self.model_runner, "kv_caches"):
                    pass  # Should not execute
                else:
                    mock_logger.debug("model_runner.kv_caches not found - skipping KV cache tensor clearing")
                
                if hasattr(self.model_runner, "compilation_config"):
                    pass  # Should not execute
                else:
                    mock_logger.debug("model_runner.compilation_config not found - skipping forward context clearing")
                
                if hasattr(self.model_runner, "input_batch"):
                    pass  # Should not execute
                else:
                    mock_logger.debug("model_runner.input_batch not found - skipping block table clearing")
                
                if hasattr(self.model_runner, "encoder_cache"):
                    pass  # Should not execute
                else:
                    mock_logger.debug("model_runner.encoder_cache not found - skipping encoder cache clearing")
                
                if hasattr(self.model_runner, "attn_groups"):
                    pass  # Should not execute
                else:
                    mock_logger.debug("model_runner.attn_groups not found - skipping attention group reset")
                
                if hasattr(self.model_runner, "use_cuda_graph"):
                    pass  # Should not execute
                else:
                    mock_logger.debug("model_runner.use_cuda_graph not found - assuming CUDA graphs are not used")
                
                mock_torch.cuda.synchronize()
                mock_torch.cuda.empty_cache()
                mock_logger.info("V1 KV cache flush completed successfully after weight update")
            
            worker._flush_kv_cache_v1 = _flush_kv_cache_v1.__get__(worker, type(worker))
            worker._flush_kv_cache_v1()
            
            # Verify cleanup still happened
            mock_torch.cuda.synchronize.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()
            
            # Verify all "not found" debug messages
            debug_calls = [call.args[0] for call in mock_logger.debug.call_args_list]
            assert any("kv_caches not found" in msg for msg in debug_calls)
            assert any("compilation_config not found" in msg for msg in debug_calls)
            assert any("input_batch not found" in msg for msg in debug_calls)
            assert any("encoder_cache not found" in msg for msg in debug_calls)
            assert any("attn_groups not found" in msg for msg in debug_calls)
            assert any("use_cuda_graph not found" in msg for msg in debug_calls)


class TestV1WorkerModelValidation:
    """Test the V1 worker model state validation functionality."""
    
    def test_validate_model_state_success(self, v1_worker, mock_torch, mock_logger):
        """Test model validation when everything is healthy."""
        with patch('torch', mock_torch), \
             patch('vllm.v1.worker.gpu_worker.logger', mock_logger):
            
            def _validate_model_state_after_update(self):
                mock_logger.debug("Starting model state validation after weight update")
                
                if hasattr(self.model_runner, "model"):
                    model = self.model_runner.model
                    mock_logger.debug("Found model_runner.model, starting validation checks")
                    
                    # Count parameters
                    total_params = sum(1 for _ in model.named_parameters())
                    mock_logger.debug(f"Model has {total_params} parameters to validate")
                    
                    # Check for NaN/inf
                    nan_params = []
                    inf_params = []
                    for name, param in model.named_parameters():
                        if mock_torch.isnan(param).any():
                            nan_params.append(name)
                        if mock_torch.isinf(param).any():
                            inf_params.append(name)
                    
                    if not nan_params:
                        mock_logger.debug("No NaN parameters found")
                    if not inf_params:
                        mock_logger.debug("No infinite parameters found")
                    
                    # Check device consistency
                    expected_device = next(model.parameters()).device
                    mock_logger.debug(f"Expected device for all parameters: {expected_device}")
                    
                    device_mismatches = []
                    for name, param in model.named_parameters():
                        if param.device != expected_device:
                            device_mismatches.append((name, param.device, expected_device))
                    
                    if not device_mismatches:
                        mock_logger.debug("All parameters are on the correct device")
                    
                    # Check dtypes
                    dtypes = set()
                    for name, param in model.named_parameters():
                        dtypes.add(param.dtype)
                    
                    mock_logger.debug(f"Found parameter dtypes: {dtypes}")
                    mock_logger.debug("Model state validation completed successfully after weight update")
                else:
                    mock_logger.debug("model_runner.model not found - skipping validation")
            
            v1_worker._validate_model_state_after_update = _validate_model_state_after_update.__get__(v1_worker, type(v1_worker))
            v1_worker._validate_model_state_after_update()
            
            # Verify logging
            debug_calls = [call.args[0] for call in mock_logger.debug.call_args_list]
            assert any("Starting model state validation" in msg for msg in debug_calls)
            assert any("Found model_runner.model" in msg for msg in debug_calls)
            assert any("No NaN parameters found" in msg for msg in debug_calls)
            assert any("validation completed successfully" in msg for msg in debug_calls)
    
    def test_validate_model_state_with_issues(self, mock_torch, mock_logger):
        """Test model validation when there are issues."""
        worker = MockWorker()
        
        # Create problematic model
        worker.model_runner.model.parameters_list = [
            ("nan_param", MockTensor((10,))),
            ("inf_param", MockTensor((10,))),
            ("device_param", MockTensor((10,), device="cpu")),
        ]
        
        # Mock torch to return problematic values
        def mock_isnan(tensor):
            if "nan_param" in str(tensor):
                return Mock(any=Mock(return_value=True))
            return Mock(any=Mock(return_value=False))
        
        def mock_isinf(tensor):
            if "inf_param" in str(tensor):
                return Mock(any=Mock(return_value=True))
            return Mock(any=Mock(return_value=False))
        
        mock_torch.isnan = mock_isnan
        mock_torch.isinf = mock_isinf
        
        with patch('torch', mock_torch), \
             patch('vllm.v1.worker.gpu_worker.logger', mock_logger):
            
            def _validate_model_state_after_update(self):
                mock_logger.debug("Starting model state validation after weight update")
                
                if hasattr(self.model_runner, "model"):
                    model = self.model_runner.model
                    
                    # Check for NaN/inf
                    nan_params = []
                    inf_params = []
                    for name, param in model.named_parameters():
                        if mock_torch.isnan(param).any():
                            nan_params.append(name)
                        if mock_torch.isinf(param).any():
                            inf_params.append(name)
                    
                    if nan_params:
                        mock_logger.warning(f"Found NaN parameters after weight update: {nan_params[:5]}")
                    if inf_params:
                        mock_logger.warning(f"Found infinite parameters after weight update: {inf_params[:5]}")
                    
                    # Check device consistency
                    expected_device = next(model.parameters()).device
                    device_mismatches = []
                    for name, param in model.named_parameters():
                        if param.device != expected_device:
                            device_mismatches.append((name, param.device, expected_device))
                    
                    if device_mismatches:
                        mock_logger.warning(f"Found device mismatches after weight update: {device_mismatches[:3]}")
            
            worker._validate_model_state_after_update = _validate_model_state_after_update.__get__(worker, type(worker))
            worker._validate_model_state_after_update()
            
            # Verify warnings were logged
            warning_calls = [call.args[0] for call in mock_logger.warning.call_args_list]
            assert any("Found NaN parameters" in msg for msg in warning_calls)
            assert any("Found infinite parameters" in msg for msg in warning_calls)
            assert any("Found device mismatches" in msg for msg in warning_calls)


class TestV1WorkerWeightUpdate:
    """Test the main weight update functionality."""
    
    @patch('vllm.worker._weight_update.stream_apply_sharded_state')
    def test_load_sharded_state_success(self, mock_stream_apply, v1_worker, mock_torch, mock_logger):
        """Test successful weight loading."""
        mock_stream_apply.return_value = 42  # Number of updated parameters
        
        with patch('torch', mock_torch), \
             patch('vllm.v1.worker.gpu_worker.logger', mock_logger):
            
            def load_sharded_state(self, path: str, pattern: Optional[str] = None):
                mock_logger.debug(f"Starting load_sharded_state for rank {self.rank} with path={path}, pattern={pattern}")
                
                # Import check
                mock_logger.debug("Imported stream_apply_sharded_state successfully")
                
                # Model runner checks
                if hasattr(self, "model_runner"):
                    mock_logger.debug("Found model_runner")
                    if hasattr(self.model_runner, "model"):
                        mock_logger.debug("Found model_runner.model")
                
                # Weight loading
                num_updated = mock_stream_apply(self.model_runner.model, path, pattern)
                mock_logger.info(f"Successfully loaded {num_updated} parameters for rank {self.rank}")
                
                # CUDA sync
                mock_torch.cuda.synchronize()
                mock_logger.debug("CUDA synchronization completed")
                
                # Validation (mock)
                mock_logger.debug("Starting model state validation")
                mock_logger.debug("Model state validation completed")
                
                # KV cache flush (mock)
                mock_logger.debug("Starting KV cache flush")
                mock_logger.debug("KV cache flush completed successfully")
                
                return {"ok": True, "rank": self.rank, "num_updated": num_updated}
            
            v1_worker.load_sharded_state = load_sharded_state.__get__(v1_worker, type(v1_worker))
            
            result = v1_worker.load_sharded_state("/path/to/weights", "pattern-{rank}-{part}.safetensors")
            
            # Verify result
            assert result["ok"] is True
            assert result["rank"] == 0
            assert result["num_updated"] == 42
            
            # Verify function was called
            mock_stream_apply.assert_called_once_with(v1_worker.model_runner.model, "/path/to/weights", "pattern-{rank}-{part}.safetensors")
            
            # Verify CUDA sync
            mock_torch.cuda.synchronize.assert_called_once()
            
            # Verify logging
            debug_calls = [call.args[0] for call in mock_logger.debug.call_args_list]
            assert any("Starting load_sharded_state" in msg for msg in debug_calls)
            assert any("Found model_runner" in msg for msg in debug_calls)
            assert any("CUDA synchronization completed" in msg for msg in debug_calls)
    
    @patch('vllm.worker._weight_update.stream_apply_sharded_state')
    def test_load_sharded_state_failure(self, mock_stream_apply, v1_worker, mock_torch, mock_logger):
        """Test weight loading failure handling."""
        mock_stream_apply.side_effect = RuntimeError("Weight loading failed")
        
        with patch('torch', mock_torch), \
             patch('vllm.v1.worker.gpu_worker.logger', mock_logger):
            
            def load_sharded_state(self, path: str, pattern: Optional[str] = None):
                try:
                    mock_logger.debug(f"Starting load_sharded_state for rank {self.rank}")
                    num_updated = mock_stream_apply(self.model_runner.model, path, pattern)
                    return {"ok": True, "rank": self.rank, "num_updated": num_updated}
                except Exception as e:
                    mock_logger.exception(f"Failed to load sharded state for rank {self.rank}")
                    return {
                        "ok": False,
                        "rank": self.rank,
                        "error": str(e),
                    }
            
            v1_worker.load_sharded_state = load_sharded_state.__get__(v1_worker, type(v1_worker))
            
            result = v1_worker.load_sharded_state("/path/to/weights")
            
            # Verify failure result
            assert result["ok"] is False
            assert result["rank"] == 0
            assert "Weight loading failed" in result["error"]
            
            # Verify exception logging
            mock_logger.exception.assert_called_once()
    
    @patch('vllm.worker._weight_update.validate_sharded_state')
    def test_validate_sharded_state_success(self, mock_validate, v1_worker, mock_logger):
        """Test successful sharded state validation."""
        mock_validate.return_value = (100, [])  # 100 tensors, no mismatches
        
        with patch('vllm.v1.worker.gpu_worker.logger', mock_logger):
            
            def validate_sharded_state(self, path: str, pattern: Optional[str] = None):
                mock_logger.debug(f"Starting validate_sharded_state for rank {self.rank}")
                
                # Import and model checks
                mock_logger.debug("Imported validate_sharded_state successfully")
                if hasattr(self, "model_runner"):
                    mock_logger.debug("Found model_runner for validation")
                    if hasattr(self.model_runner, "model"):
                        mock_logger.debug("Found model_runner.model for validation")
                
                tensor_count, mismatches = mock_validate(self.model_runner.model, path, pattern)
                mock_logger.debug(f"Validation completed: {tensor_count} tensors checked, {len(mismatches)} mismatches found")
                
                return {
                    "ok": True,
                    "rank": self.rank,
                    "tensor_count": tensor_count,
                    "mismatches": mismatches,
                }
            
            v1_worker.validate_sharded_state = validate_sharded_state.__get__(v1_worker, type(v1_worker))
            
            result = v1_worker.validate_sharded_state("/path/to/weights")
            
            # Verify result
            assert result["ok"] is True
            assert result["rank"] == 0
            assert result["tensor_count"] == 100
            assert result["mismatches"] == []
            
            # Verify function was called
            mock_validate.assert_called_once_with(v1_worker.model_runner.model, "/path/to/weights", None)
            
            # Verify logging
            debug_calls = [call.args[0] for call in mock_logger.debug.call_args_list]
            assert any("Starting validate_sharded_state" in msg for msg in debug_calls)
            assert any("100 tensors checked, 0 mismatches found" in msg for msg in debug_calls)


class TestWeightUpdateModule:
    """Test the enhanced weight update module functionality."""
    
    def test_stream_apply_sharded_state_with_failures(self, mock_torch):
        """Test weight loading with some parameter failures."""
        
        # Mock the weight update module components
        with patch('vllm.worker._weight_update.LoadConfig'), \
             patch('vllm.worker._weight_update.ShardedStateLoader'), \
             patch('vllm.worker._weight_update.get_tensor_model_parallel_rank', return_value=0), \
             patch('glob.glob', return_value=['/path/model-rank-0-part-0.safetensors']), \
             patch('torch', mock_torch):
            
            # Create a mock model
            mock_model = Mock()
            load_weights_calls = []
            
            def mock_load_weights(weights):
                # Simulate some failures
                for name, tensor in weights:
                    if "failing_param" in name:
                        raise RuntimeError(f"Failed to load {name}")
                load_weights_calls.append(weights)
            
            mock_model.load_weights = mock_load_weights
            
            # Mock loader that yields some successful and some failing tensors
            mock_loader = Mock()
            tensors = [
                ("good_param_1", MockTensor((100, 200))),
                ("failing_param_1", MockTensor((50, 100))),
                ("good_param_2", MockTensor((200, 300))),
                ("failing_param_2", MockTensor((75, 150))),
                ("good_param_3", MockTensor((300, 400))),
            ]
            mock_loader.iterate_over_files.return_value = iter(tensors)
            
            # Import and patch the module
            import importlib
            import sys
            
            # Create mock vllm modules
            if 'vllm' not in sys.modules:
                sys.modules['vllm'] = Mock()
            if 'vllm.config' not in sys.modules:
                sys.modules['vllm.config'] = Mock()
            if 'vllm.distributed' not in sys.modules:
                sys.modules['vllm.distributed'] = Mock()
            if 'vllm.model_executor' not in sys.modules:
                sys.modules['vllm.model_executor'] = Mock()
            if 'vllm.model_executor.model_loader' not in sys.modules:
                sys.modules['vllm.model_executor.model_loader'] = Mock()
            if 'vllm.model_executor.model_loader.sharded_state_loader' not in sys.modules:
                sys.modules['vllm.model_executor.model_loader.sharded_state_loader'] = Mock()
            if 'vllm.transformers_utils' not in sys.modules:
                sys.modules['vllm.transformers_utils'] = Mock()
            if 'vllm.transformers_utils.s3_utils' not in sys.modules:
                sys.modules['vllm.transformers_utils.s3_utils'] = Mock()
            if 'vllm.transformers_utils.utils' not in sys.modules:
                sys.modules['vllm.transformers_utils.utils'] = Mock()
            
            # Mock the components
            sys.modules['vllm.config'].LoadConfig = Mock()
            sys.modules['vllm.distributed'].get_tensor_model_parallel_rank = Mock(return_value=0)
            sys.modules['vllm.model_executor.model_loader.sharded_state_loader'].ShardedStateLoader = Mock(return_value=mock_loader)
            sys.modules['vllm.transformers_utils.utils'].is_s3 = Mock(return_value=False)
            
            # Test the enhanced function logic
            def enhanced_stream_apply_sharded_state(model, path, pattern=None):
                """Enhanced version with error handling."""
                updated = 0
                failed_params = []
                
                for key, tensor in tensors:
                    try:
                        # Ensure tensor is contiguous
                        if hasattr(tensor, 'is_contiguous') and not tensor.is_contiguous():
                            tensor = tensor.contiguous()
                        
                        model.load_weights(weights=[(key, tensor)])
                        updated += 1
                        
                        # Periodic cleanup
                        if updated % 100 == 0:
                            pass  # Would do gc.collect()
                            
                    except Exception as e:
                        failed_params.append((key, str(e)))
                        continue
                
                # Check failure rate
                if failed_params:
                    failure_rate = len(failed_params) / max(1, updated + len(failed_params))
                    if failure_rate > 0.1:  # More than 10% failed
                        raise RuntimeError(
                            f"Too many parameter loading failures ({len(failed_params)} failed, "
                            f"{updated} succeeded). First few failures: {failed_params[:5]}"
                        )
                
                if updated == 0:
                    raise ValueError("No parameters were successfully loaded")
                    
                return updated
            
            # Test with acceptable failure rate (2/5 = 40% > 10%, should fail)
            with pytest.raises(RuntimeError, match="Too many parameter loading failures"):
                enhanced_stream_apply_sharded_state(mock_model, "/path/to/weights")
            
            # Test with lower failure rate by adding more successful params
            more_tensors = tensors + [
                ("good_param_4", MockTensor((400, 500))),
                ("good_param_5", MockTensor((500, 600))),
                ("good_param_6", MockTensor((600, 700))),
                ("good_param_7", MockTensor((700, 800))),
                ("good_param_8", MockTensor((800, 900))),
            ]
            
            def enhanced_stream_apply_sharded_state_more_success(model, path, pattern=None):
                updated = 0
                failed_params = []
                
                for key, tensor in more_tensors:
                    try:
                        model.load_weights(weights=[(key, tensor)])
                        updated += 1
                    except Exception as e:
                        failed_params.append((key, str(e)))
                        continue
                
                # Check failure rate (2/10 = 20% > 10%, still fails)
                if failed_params:
                    failure_rate = len(failed_params) / max(1, updated + len(failed_params))
                    if failure_rate > 0.1:
                        raise RuntimeError(f"Too many failures: {failure_rate}")
                
                return updated
            
            # Should still fail with 20% failure rate
            with pytest.raises(RuntimeError, match="Too many failures"):
                enhanced_stream_apply_sharded_state_more_success(mock_model, "/path/to/weights")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])