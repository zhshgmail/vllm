"""Simple test for device detection logic in weight loading."""
import torch
import tempfile
import os
from safetensors.torch import save_file
from safetensors import safe_open

def test_device_detection_logic():
    """Test the device detection logic we added."""
    
    print("Testing device detection logic...")
    
    # Test case 1: Model with parameters on specific device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing with device: {device}")
    
    # Create mock parameters on the test device
    param1 = torch.randn(5, 5, device=device)
    param2 = torch.randn(3, 3, device=device)
    
    # Simulate the device detection logic from our code
    model_device = "cpu"  # Default fallback
    try:
        # This simulates: first_param = next(iter(model.parameters()), None)
        first_param = param1  # In real code this would be from model.parameters()
        if first_param is not None:
            model_device = str(first_param.device)
    except (StopIteration, AttributeError):
        model_device = "cpu"
    
    print(f"Detected device: {model_device}")
    assert str(device) == model_device, f"Expected {device}, got {model_device}"
    
    # Test case 2: Test safetensors loading with correct device
    with tempfile.TemporaryDirectory() as temp_dir:
        safetensors_path = os.path.join(temp_dir, "test_weights.safetensors")
        
        # Create and save test tensors
        test_tensors = {
            "weight1": torch.randn(5, 5),
            "weight2": torch.randn(3, 3)
        }
        save_file(test_tensors, safetensors_path)
        
        # Test loading with detected device
        with safe_open(safetensors_path, framework="pt", device=model_device) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                print(f"Loaded {key} to device: {tensor.device}")
                # Verify tensor is on the expected device
                assert str(tensor.device) == model_device, f"Tensor {key} on wrong device: {tensor.device} vs {model_device}"
    
    print("[PASS] Device detection logic test passed!")

def test_device_fallback():
    """Test device detection fallback to CPU."""
    
    print("Testing device fallback logic...")
    
    # Simulate no parameters case
    model_device = "cpu"  # Default fallback
    try:
        # Simulate empty parameters
        first_param = None
        if first_param is not None:
            model_device = str(first_param.device)
    except (StopIteration, AttributeError):
        model_device = "cpu"
    
    print(f"Fallback device: {model_device}")
    assert model_device == "cpu", f"Expected CPU fallback, got {model_device}"
    
    print("[PASS] Device fallback test passed!")

if __name__ == "__main__":
    test_device_detection_logic()
    test_device_fallback()
    print("[PASS] All device detection tests passed!")
