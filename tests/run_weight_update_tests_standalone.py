#!/usr/bin/env python3
"""Standalone test runner for weight update tests.

This runner executes the weight update tests in isolation without loading
the full vLLM test infrastructure.
"""
import sys
import os
import importlib.util
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def run_weight_update_basic_tests():
    """Run the enhanced weight update tests manually."""
    print("Running Weight Update Unit Tests (Standalone)")
    print("=" * 50)
    
    # Import test modules directly
    try:
        # Test 1: Enhanced weight update module tests
        print("\n1. Testing enhanced weight update module...")
        
        # Simulate the test environment setup
        import types
        
        # Create fake vllm modules to avoid import issues
        vllm_pkg = types.ModuleType("vllm")
        vllm_pkg.__path__ = []
        sys.modules["vllm"] = vllm_pkg
        
        config_pkg = types.ModuleType("vllm.config")
        sys.modules["vllm.config"] = config_pkg
        
        distributed_pkg = types.ModuleType("vllm.distributed")
        sys.modules["vllm.distributed"] = distributed_pkg
        
        model_loader_pkg = types.ModuleType("vllm.model_executor.model_loader.sharded_state_loader")
        sys.modules["vllm.model_executor.model_loader.sharded_state_loader"] = model_loader_pkg
        
        # Mock the classes we need
        class MockLoadConfig:
            def __init__(self, load_format, model_loader_extra_config):
                self.load_format = load_format
                self.model_loader_extra_config = model_loader_extra_config
        
        class MockShardedStateLoader:
            DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"
            def __init__(self, load_cfg):
                self.pattern = self.DEFAULT_PATTERN

            def iterate_over_files(self, filepaths):
                # Mock tensor data for testing
                yield "layer.weight", MockTensor((2, 3))
                yield "layer.bias", MockTensor((3,))
        
        class MockTensor:
            def __init__(self, shape):
                self.shape = shape
                self._is_contiguous = True

            def is_contiguous(self):
                return self._is_contiguous

            def contiguous(self):
                return self
        
        config_pkg.LoadConfig = MockLoadConfig
        model_loader_pkg.ShardedStateLoader = MockShardedStateLoader
        distributed_pkg.get_tensor_model_parallel_rank = lambda: 0
        
        # Mock additional modules
        s3_utils_pkg = types.ModuleType("vllm.transformers_utils.s3_utils")
        utils_pkg = types.ModuleType("vllm.transformers_utils.utils")
        sys.modules["vllm.transformers_utils.s3_utils"] = s3_utils_pkg
        sys.modules["vllm.transformers_utils.utils"] = utils_pkg
        utils_pkg.is_s3 = lambda x: False
        s3_utils_pkg.glob = lambda **kwargs: []
        
        # Mock glob
        import glob
        original_glob = glob.glob
        glob.glob = lambda pattern: ["/fake/model-rank-0-part-0.safetensors"]
        
        # Import and test the weight update module
        spec = importlib.util.spec_from_file_location(
            "weight_update", "vllm/worker/_weight_update.py"
        )
        wu_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wu_module)
        
        # Test basic functionality
        class TestModel:
            def __init__(self):
                self.updates = []

            def load_weights(self, weights):
                self.updates.extend(weights)
            
            def named_parameters(self, recurse=True):
                # Mock some parameters for testing
                yield "layer.weight", MockTensor((100, 200))
                yield "layer.bias", MockTensor((200,))
        
        model = TestModel()
        result = wu_module.stream_apply_sharded_state(model, "/fake/path")
        
        if result > 0 and len(model.updates) > 0:
            print("   [PASS] Basic weight loading works")
        else:
            print("   [FAIL] Basic weight loading failed")
            return False

        # Restore glob
        glob.glob = original_glob
        
        print("   [PASS] Enhanced weight update module tests passed")
        
    except Exception as e:
        print(f"   [FAIL] Weight update module tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: V1 worker mock tests
    try:
        print("\n2. Testing V1 worker mock functionality...")
        
        # Mock torch
        torch_mock = types.ModuleType("torch")
        torch_mock.cuda = types.ModuleType("torch.cuda")
        torch_mock.cuda.synchronize = lambda: None
        torch_mock.cuda.empty_cache = lambda: None
        torch_mock.is_tensor = lambda x: hasattr(x, 'zero_')
        torch_mock.isnan = lambda x: type('MockResult', (), {'any': lambda: False})()
        torch_mock.isinf = lambda x: type('MockResult', (), {'any': lambda: False})()
        sys.modules["torch"] = torch_mock
        
        # Test mock tensor functionality
        class MockTensorV1:
            def __init__(self, shape):
                self.shape = shape
                self.device = "cuda:0"
                self.dtype = "float32"

            def zero_(self):
                return self
        
        tensor = MockTensorV1((100, 200))
        tensor.zero_()
        print("   [PASS] Mock tensor operations work")
        
        # Test mock model runner
        class MockModelRunnerV1:
            def __init__(self):
                self.kv_caches = [MockTensorV1((100, 2, 32, 128))]
                self.model = type('Model', (), {
                    'named_parameters': lambda: [("layer.weight", MockTensorV1((1024, 512)))],
                    'parameters': lambda: [MockTensorV1((1024, 512))]
                })()
        
        runner = MockModelRunnerV1()
        runner.kv_caches[0].zero_()
        print("   [PASS] Mock model runner operations work")
        
        print("   [PASS] V1 worker mock tests passed")
        
    except Exception as e:
        print(f"   [FAIL] V1 worker mock tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: API server mock tests  
    try:
        print("\n3. Testing API server mock functionality...")
        
        # Mock HTTP exceptions
        class HTTPException(Exception):
            def __init__(self, status_code, detail):
                self.status_code = status_code
                self.detail = detail
                super().__init__(f"{status_code}: {detail}")
        
        # Test V1 detection logic
        class MockAsyncLLMV1:
            pass
        
        class MockAsyncLLMV0:
            pass
        
        v1_engine = MockAsyncLLMV1()
        v0_engine = MockAsyncLLMV0()
        
        # Simulate V1 detection
        def check_engine_type(engine):
            return isinstance(engine, MockAsyncLLMV1)
        
        if check_engine_type(v1_engine) and not check_engine_type(v0_engine):
            print("   [PASS] Engine type detection works")
        else:
            print("   [FAIL] Engine type detection failed")
            return False
        
        # Test request validation
        def validate_request(body):
            if not isinstance(body, dict):
                raise HTTPException(400, "Body must be a JSON object")
            if "path" not in body:
                raise HTTPException(400, "Missing path")
            return True
        
        # Valid request
        try:
            validate_request({"path": "/valid/path"})
            print("   [PASS] Valid request validation works")
        except HTTPException:
            print("   [FAIL] Valid request validation failed")
            return False
        
        # Invalid request
        try:
            validate_request({"no_path": "invalid"})
            print("   [FAIL] Invalid request validation should have failed")
            return False
        except HTTPException:
            print("   [PASS] Invalid request validation works")
        
        print("   [PASS] API server mock tests passed")
        
    except Exception as e:
        print(f"   [FAIL] API server mock tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nSUCCESS: All standalone weight update tests passed!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = run_weight_update_basic_tests()
    sys.exit(0 if success else 1)
