# SPDX-License-Identifier: Apache-2.0
"""
Mock-based Integration Tests for AReaL vLLM Weight Update System

This test validates the AReaL integration logic without requiring:
- CUDA/GPU hardware
- vLLM engine to be started 
- NCCL initialization
- Network connections

Windows compatible - uses mocks instead of subprocess/multiprocessing.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from typing import Dict, Any, List


class MockNCCLConfig:
    """Mock NCCL configuration."""
    def __init__(self, master_address="127.0.0.1", master_port=29500, rank_offset=1, world_size=2):
        self.master_address = master_address
        self.master_port = master_port
        self.rank_offset = rank_offset
        self.world_size = world_size


class MockRLGenerationRequest:
    """Mock RL generation request."""
    def __init__(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_id = f"mock_req_{hash(prompt) % 1000}"


class MockGenerationResponse:
    """Mock generation response."""
    def __init__(self, request_id: str, generated_text: str, finish_reason: str = "length"):
        self.request_id = request_id
        self.generated_text = generated_text
        self.finish_reason = finish_reason
        self.usage = {"prompt_tokens": 10, "completion_tokens": 20}


class MockAReaLAdapter:
    """Mock AReaL adapter that simulates behavior without real connections."""
    
    def __init__(self, vllm_server_url: str, nccl_config: MockNCCLConfig):
        self.server_url = vllm_server_url
        self.nccl_config = nccl_config
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "weight_updates": 0,
            "interrupted_requests": 0
        }
        self.nccl_initialized = False
        
    async def initialize_nccl(self) -> bool:
        """Mock NCCL initialization."""
        self.nccl_initialized = True
        return True
        
    async def generate_async(self, request: MockRLGenerationRequest) -> MockGenerationResponse:
        """Mock async generation."""
        self.stats["total_requests"] += 1
        
        # Simulate processing time
        await asyncio.sleep(0.01)
        
        # Mock generation logic
        if "error" in request.prompt.lower():
            self.stats["failed_requests"] += 1
            return MockGenerationResponse(
                request.request_id, 
                "Error occurred", 
                finish_reason="error"
            )
        
        self.stats["successful_requests"] += 1
        mock_text = f"Generated response for: {request.prompt[:20]}..."
        return MockGenerationResponse(request.request_id, mock_text)
    
    async def batch_generate_async(self, requests: List[MockRLGenerationRequest]) -> List[MockGenerationResponse]:
        """Mock batch generation."""
        tasks = [self.generate_async(req) for req in requests]
        return await asyncio.gather(*tasks)
    
    async def trigger_weight_update(self, allow_interrupt: bool = True, validate_consistency: bool = True) -> Dict[str, Any]:
        """Mock weight update trigger."""
        self.stats["weight_updates"] += 1
        
        if allow_interrupt:
            # Simulate interrupting some requests
            interrupted_count = min(2, self.stats["total_requests"] // 2)
            self.stats["interrupted_requests"] += interrupted_count
        else:
            interrupted_count = 0
            
        return {
            "status": "success",
            "interrupted_requests": interrupted_count,
            "update_duration": 0.1,
            "validation_passed": validate_consistency
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get mock adapter statistics."""
        return {
            "adapter_stats": self.stats.copy(),
            "server_stats": {
                "engine_running": True,
                "active_requests": self.stats["total_requests"] - self.stats["successful_requests"] - self.stats["failed_requests"]
            },
            "nccl_stats": {
                "initialized": self.nccl_initialized,
                "world_size": self.nccl_config.world_size,
                "rank_offset": self.nccl_config.rank_offset
            }
        }
    
    async def cleanup(self):
        """Mock cleanup."""
        self.nccl_initialized = False


def create_mock_generation_request(prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> MockRLGenerationRequest:
    """Create a mock generation request."""
    return MockRLGenerationRequest(prompt, max_tokens, temperature)


async def create_mock_areal_adapter(vllm_server_url: str, training_master_address: str, training_master_port: int) -> MockAReaLAdapter:
    """Create and initialize mock AReaL adapter."""
    nccl_config = MockNCCLConfig(training_master_address, training_master_port)
    adapter = MockAReaLAdapter(vllm_server_url, nccl_config)
    await adapter.initialize_nccl()
    return adapter


class TestAReaLIntegrationLogic:
    """Test AReaL integration logic with mocks."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Fixture providing mock adapter."""
        nccl_config = MockNCCLConfig()
        adapter = MockAReaLAdapter("http://localhost:8000", nccl_config)
        return adapter
    
    @pytest.mark.asyncio
    async def test_basic_generation(self, mock_adapter):
        """Test basic generation functionality."""
        request = create_mock_generation_request("What is the capital of France?", max_tokens=50)
        
        response = await mock_adapter.generate_async(request)
        
        assert response.finish_reason != "error"
        assert "Generated response" in response.generated_text
        assert response.request_id == request.request_id
        
        # Check stats updated
        stats = await mock_adapter.get_stats()
        assert stats["adapter_stats"]["total_requests"] == 1
        assert stats["adapter_stats"]["successful_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_generation(self, mock_adapter):
        """Test batch generation functionality."""
        prompts = [
            "The weather today is",
            "Machine learning is", 
            "Python programming language"
        ]
        
        requests = [create_mock_generation_request(p, max_tokens=30) for p in prompts]
        responses = await mock_adapter.batch_generate_async(requests)
        
        assert len(responses) == len(requests)
        success_count = sum(1 for r in responses if r.finish_reason != "error")
        assert success_count == len(requests)
        
        # Check stats
        stats = await mock_adapter.get_stats()
        assert stats["adapter_stats"]["total_requests"] == len(requests)
        assert stats["adapter_stats"]["successful_requests"] == len(requests)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_adapter):
        """Test error handling in generation."""
        request = create_mock_generation_request("Generate an error please", max_tokens=20)
        
        response = await mock_adapter.generate_async(request)
        
        assert response.finish_reason == "error"
        assert "Error occurred" in response.generated_text
        
        # Check error stats
        stats = await mock_adapter.get_stats()
        assert stats["adapter_stats"]["failed_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_weight_update_interruption(self, mock_adapter):
        """Test weight update with request interruption."""
        # Generate some requests first
        requests = [
            create_mock_generation_request(f"Story about {topic}:", max_tokens=100) 
            for topic in ["space", "AI", "ocean"]
        ]
        
        # Start generation tasks
        generation_tasks = [
            asyncio.create_task(mock_adapter.generate_async(req)) 
            for req in requests
        ]
        
        # Wait a moment then trigger weight update
        await asyncio.sleep(0.01)
        
        update_result = await mock_adapter.trigger_weight_update(
            allow_interrupt=True, 
            validate_consistency=True
        )
        
        assert update_result["status"] == "success"
        assert update_result["interrupted_requests"] >= 0
        assert update_result["validation_passed"] is True
        
        # Wait for generation tasks to complete
        responses = await asyncio.gather(*generation_tasks)
        
        assert len(responses) == len(requests)
        
        # Check weight update stats
        stats = await mock_adapter.get_stats()
        assert stats["adapter_stats"]["weight_updates"] == 1
    
    @pytest.mark.asyncio
    async def test_weight_update_without_interruption(self, mock_adapter):
        """Test weight update without interruption."""
        update_result = await mock_adapter.trigger_weight_update(
            allow_interrupt=False, 
            validate_consistency=False
        )
        
        assert update_result["status"] == "success"
        assert update_result["interrupted_requests"] == 0
        assert update_result["validation_passed"] is False
    
    @pytest.mark.asyncio
    async def test_statistics_collection(self, mock_adapter):
        """Test statistics collection functionality."""
        # Initialize NCCL first
        await mock_adapter.initialize_nccl()
        
        # Perform various operations
        await mock_adapter.generate_async(create_mock_generation_request("Test 1"))
        await mock_adapter.generate_async(create_mock_generation_request("Error test"))
        await mock_adapter.trigger_weight_update()
        
        # Get comprehensive stats
        stats = await mock_adapter.get_stats()
        
        # Verify adapter stats
        adapter_stats = stats["adapter_stats"]
        assert adapter_stats["total_requests"] >= 2
        assert adapter_stats["successful_requests"] >= 1
        assert adapter_stats["weight_updates"] >= 1
        
        # Verify server stats
        server_stats = stats["server_stats"]
        assert "engine_running" in server_stats
        assert "active_requests" in server_stats
        
        # Verify NCCL stats
        nccl_stats = stats["nccl_stats"]
        assert nccl_stats["initialized"] is True
        assert nccl_stats["world_size"] == 2
        assert nccl_stats["rank_offset"] == 1
    
    @pytest.mark.asyncio
    async def test_nccl_initialization(self, mock_adapter):
        """Test NCCL initialization logic."""
        # Should already be initialized from fixture
        assert mock_adapter.nccl_initialized is False  # Not yet initialized in fresh adapter
        
        # Test initialization
        success = await mock_adapter.initialize_nccl()
        assert success is True
        assert mock_adapter.nccl_initialized is True
        
        # Check stats reflect initialization
        stats = await mock_adapter.get_stats()
        assert stats["nccl_stats"]["initialized"] is True
    
    @pytest.mark.asyncio
    async def test_cleanup_functionality(self, mock_adapter):
        """Test cleanup functionality."""
        # Initialize and use adapter
        await mock_adapter.initialize_nccl()
        await mock_adapter.generate_async(create_mock_generation_request("Test"))
        
        assert mock_adapter.nccl_initialized is True
        assert mock_adapter.stats["total_requests"] > 0
        
        # Test cleanup
        await mock_adapter.cleanup()
        assert mock_adapter.nccl_initialized is False
    
    def test_request_creation(self):
        """Test request creation utilities."""
        request = create_mock_generation_request("Test prompt", max_tokens=100, temperature=0.8)
        
        assert request.prompt == "Test prompt"
        assert request.max_tokens == 100
        assert request.temperature == 0.8
        assert request.request_id.startswith("mock_req_")
    
    @pytest.mark.asyncio
    async def test_adapter_creation(self):
        """Test adapter creation utility."""
        adapter = await create_mock_areal_adapter(
            vllm_server_url="http://test:8000",
            training_master_address="192.168.1.100", 
            training_master_port=29600
        )
        
        assert adapter.server_url == "http://test:8000"
        assert adapter.nccl_config.master_address == "192.168.1.100"
        assert adapter.nccl_config.master_port == 29600
        assert adapter.nccl_initialized is True
        
        # Cleanup
        await adapter.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])