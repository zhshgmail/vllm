# Weight Update Unit Tests

This directory contains comprehensive unit tests for the vLLM weight update functionality. These tests are designed to run in isolation without requiring a full vLLM engine startup, GPU/CUDA dependencies, or external model files.

## Test Files

### 1. `test_weight_update.py` (Enhanced)
Tests the core weight loading logic in `vllm.worker._weight_update`:
- Basic weight streaming functionality
- Enhanced error handling with failure rate thresholds
- Non-contiguous tensor handling
- Partial failure scenarios
- Iterator failure handling

### 2. `test_v1_worker_weight_update.py` (New)
Tests the V1 worker weight update functionality:
- KV cache flush logic with all conditional branches
- Model state validation after weight updates
- Complete weight loading workflow
- Component availability checking with debug logging

### 3. `test_api_server_weight_update.py` (New)
Tests the API server weight update endpoint:
- V1 engine detection logic
- Request validation
- Weight update in-progress conflict detection
- Path validation and preflight checks
- V0 worker restriction enforcement

## Key Testing Features

### Full Mocking Strategy
- **No External Dependencies**: All vLLM components, PyTorch tensors, CUDA operations, and file I/O are mocked
- **Isolated Unit Tests**: Each test focuses on a specific piece of logic without side effects
- **CI/CD Friendly**: Tests run in any Python environment without GPU or special setup

### Comprehensive Coverage
- **All Conditional Branches**: Tests cover every `hasattr()` check and conditional path
- **Error Scenarios**: Tests both success and failure paths
- **Edge Cases**: Handles boundary conditions like 10% failure rates, missing components, etc.
- **Debug Logging**: Validates that debug logging works correctly for troubleshooting

### Mock Components

#### MockTensor
- Simulates PyTorch tensor behavior
- Supports `.zero_()`, `.is_contiguous()`, `.contiguous()` methods
- Can be configured to fail on load for testing error handling

#### MockModel
- Simulates model with `load_weights()` and parameter iteration
- Configurable parameter failures for testing resilience

#### MockModelRunner
- Complete V1 model runner mock with all components:
  - KV caches (regular and Mamba-style)
  - Compilation config with forward context
  - Input batch with block tables
  - Encoder cache for multimodal models
  - Attention groups with metadata builders

## Running the Tests

### Prerequisites
```bash
pip install pytest
```

### Run Individual Test Files
```bash
# Enhanced core weight update tests
python -m pytest tests/test_weight_update.py -v

# V1 worker functionality tests  
python -m pytest tests/test_v1_worker_weight_update.py -v

# API server endpoint tests
python -m pytest tests/test_api_server_weight_update.py -v
```

### Run All Weight Update Tests
```bash
# Using the test runner
python tests/run_weight_update_tests.py

# Or with pytest
python -m pytest tests/test_*weight_update*.py -v
```

### Run Specific Test Cases
```bash
# Test KV cache flush with all components
python -m pytest tests/test_v1_worker_weight_update.py::TestV1WorkerKVCacheFlush::test_flush_kv_cache_v1_all_components -v

# Test high failure rate handling
python -m pytest tests/test_weight_update.py::test_stream_apply_sharded_state_high_failure_rate -v

# Test V1 engine detection
python -m pytest tests/test_api_server_weight_update.py::TestAPIServerV1Detection::test_v1_engine_detection_success -v
```

## Test Categories

### 1. Happy Path Tests
- All components present and working
- Successful weight loading
- Proper KV cache clearing
- Valid API requests

### 2. Error Handling Tests
- Missing components (graceful degradation)
- Weight loading failures (with acceptable failure rates)
- Invalid API requests
- V0 engine usage attempts

### 3. Edge Case Tests
- Non-contiguous tensors
- Boundary failure rates (exactly 10%)
- Empty component lists
- Mixed success/failure scenarios

### 4. Debug Logging Tests
- All conditional branches log appropriate messages
- Error details are captured
- Component discovery is logged
- Performance metrics are tracked

## Integration with CI/CD

These tests are designed to run in GitHub Actions and other CI/CD environments:

```yaml
- name: Run Weight Update Unit Tests
  run: |
    pip install pytest
    python -m pytest tests/test_*weight_update*.py -v --tb=short
```

The tests will pass/fail based on the logic correctness without requiring:
- GPU hardware
- CUDA installation  
- Large model files
- Network access
- vLLM engine initialization

## Debugging Test Failures

### Enable Debug Logging in Tests
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Run with More Verbose Output
```bash
python -m pytest tests/test_v1_worker_weight_update.py -v -s --tb=long
```

### Check Mock Configurations
Tests include assertions on mock call counts and arguments to verify the mocking strategy is working correctly.

## Adding New Tests

When adding new weight update functionality:

1. **Add Unit Tests**: Create focused tests that mock all dependencies
2. **Test All Branches**: Ensure every conditional path is tested
3. **Include Debug Logging**: Verify debug messages for troubleshooting
4. **Test Error Scenarios**: Don't just test the happy path
5. **Mock Everything**: Keep tests isolated and fast

Example test structure:
```python
def test_new_functionality(mock_torch, mock_logger):
    with patch('torch', mock_torch), \
         patch('vllm.some.module.logger', mock_logger):
        
        # Test setup with mocks
        worker = create_mock_worker()
        
        # Execute functionality
        result = worker.new_method()
        
        # Assert behavior and logging
        assert result["success"] is True
        mock_logger.debug.assert_called()
```

This ensures the tests remain fast, reliable, and CI/CD friendly while providing comprehensive coverage of the weight update functionality.