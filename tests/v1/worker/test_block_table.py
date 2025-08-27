"""Tests for vLLM v1 worker block table functionality."""

import pytest
import torch

from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable


class TestBlockTable:
    """Tests for the BlockTable class."""

    def test_block_table_initialization(self):
        """Test basic BlockTable initialization."""
        block_table = BlockTable(
            max_num_reqs=4,
            max_num_blocks_per_req=8,
            max_num_batched_tokens=32,
            pin_memory=False,
            device=torch.device("cpu")
        )
        
        assert block_table.max_num_reqs == 4
        assert block_table.max_num_blocks_per_req == 8
        assert block_table.max_num_batched_tokens == 32
        assert block_table.block_table.shape == (4, 8)


class TestMultiGroupBlockTable:
    """Tests for the MultiGroupBlockTable class."""

    def test_multi_group_block_table_initialization(self):
        """Test MultiGroupBlockTable initialization."""
        block_sizes = [16, 32, 64]
        table = MultiGroupBlockTable(
            max_num_reqs=4,
            max_model_len=1024,
            max_num_batched_tokens=32,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=block_sizes
        )
        
        assert len(table.block_tables) == 3
        assert len(table) == 3  # Test __len__ method
        
        # Verify each block table is properly configured
        for i, block_size in enumerate(block_sizes):
            expected_blocks = (1024 + block_size - 1) // block_size  # cdiv(1024, block_size)
            assert table[i].max_num_blocks_per_req == expected_blocks

    def test_multi_group_block_table_len_method(self):
        """Test the __len__ method specifically."""
        # Test with different numbers of block sizes
        test_cases = [
            [16],
            [16, 32],
            [8, 16, 32, 64],
            [4, 8, 16, 32, 64, 128]
        ]
        
        for block_sizes in test_cases:
            table = MultiGroupBlockTable(
                max_num_reqs=2,
                max_model_len=512,
                max_num_batched_tokens=16,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=block_sizes
            )
            
            assert len(table) == len(block_sizes), \
                f"Expected len(table) == {len(block_sizes)}, got {len(table)}"

    def test_multi_group_block_table_indexing(self):
        """Test the __getitem__ method."""
        block_sizes = [16, 32, 64]
        table = MultiGroupBlockTable(
            max_num_reqs=2,
            max_model_len=512,
            max_num_batched_tokens=16,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=block_sizes
        )
        
        # Test valid indexing
        for i in range(len(block_sizes)):
            block_table = table[i]
            assert isinstance(block_table, BlockTable)
        
        # Test invalid indexing
        with pytest.raises(IndexError):
            _ = table[len(block_sizes)]

    def test_multi_group_block_table_operations(self):
        """Test basic operations on MultiGroupBlockTable."""
        block_sizes = [16, 32]
        table = MultiGroupBlockTable(
            max_num_reqs=4,
            max_model_len=512,
            max_num_batched_tokens=32,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=block_sizes
        )
        
        # Test add_row
        block_ids = ([1, 2, 3], [4, 5])  # Two groups with different block IDs
        table.add_row(block_ids, row_idx=0)
        
        # Test append_row
        more_block_ids = ([6, 7], [8])
        table.append_row(more_block_ids, row_idx=0)
        
        # Test commit
        table.commit(num_reqs=1)
        
        # Test clear
        table.clear()
        
        # All operations should complete without error
        assert len(table) == 2  # Still have 2 block tables


if __name__ == "__main__":
    pytest.main([__file__])
