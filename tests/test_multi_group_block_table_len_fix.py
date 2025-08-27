"""
Test case for MultiGroupBlockTable __len__ method fix.
This test ensures the fix for the KV cache flush error is working.
"""
import torch
import pytest

from vllm.v1.worker.block_table import MultiGroupBlockTable


def test_multi_group_block_table_len_fix():
    """
    Test that MultiGroupBlockTable supports len() operation.
    
    This test specifically addresses the error:
    TypeError: object of type 'MultiGroupBlockTable' has no len()
    
    That was occurring in gpu_worker.py line: len(self.model_runner.input_batch.block_table)
    """
    # Test with various block sizes similar to real usage
    test_cases = [
        [16],           # Single block size
        [16, 32],       # Two block sizes  
        [8, 16, 32],    # Three block sizes
        [4, 8, 16, 32, 64]  # Multiple block sizes
    ]
    
    for block_sizes in test_cases:
        table = MultiGroupBlockTable(
            max_num_reqs=10,
            max_model_len=2048,
            max_num_batched_tokens=128,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=block_sizes
        )
        
        # This should not raise TypeError anymore
        table_length = len(table)
        
        # Verify the length matches the number of block sizes
        assert table_length == len(block_sizes), \
            f"Expected len(table) == {len(block_sizes)}, got {table_length}"
        
        # Verify we can also call len() multiple times
        assert len(table) == len(table) == table_length
        
        print(f"✅ len() works for {len(block_sizes)} block sizes: {table_length}")


def test_multi_group_block_table_len_in_context():
    """
    Test len() in a context similar to the original error location.
    Simulates the gpu_worker.py usage pattern.
    """
    # Create table similar to model_runner.input_batch.block_table
    block_sizes = [16, 32]  # Common configuration
    block_table = MultiGroupBlockTable(
        max_num_reqs=8,
        max_model_len=4096,
        max_num_batched_tokens=512,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=block_sizes
    )
    
    # Simulate the operation that was failing:
    # len(self.model_runner.input_batch.block_table)
    try:
        block_table_length = len(block_table)
        print(f"✅ KV cache flush operation len() successful: {block_table_length}")
        assert block_table_length == 2
    except TypeError as e:
        pytest.fail(f"len() operation failed: {e}")


if __name__ == "__main__":
    test_multi_group_block_table_len_fix()
    test_multi_group_block_table_len_in_context()
    print("🎉 All __len__ fix tests passed!")
