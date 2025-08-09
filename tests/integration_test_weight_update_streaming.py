# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for weight update with live request interruption.

These tests verify that streaming requests receive proper abort signals
and partial responses when weight updates occur.
"""

import asyncio
import json
import tempfile
import os
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm import SamplingParams
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


class MockStreamingEngine:
    """Mock AsyncLLM that simulates streaming with interruption."""
    
    def __init__(self):
        self.active_streams = {}
        self.interrupted = False
        self.output_processor = MagicMock()
        self.engine_core = AsyncMock()
        self.log_requests = True
        
    async def generate(self, request_id: str, prompt: str, sampling_params: SamplingParams) -> AsyncGenerator[RequestOutput, None]:
        """Simulate streaming generation with potential interruption."""
        self.active_streams[request_id] = {"tokens": 0}
        
        try:
            # Simulate generating tokens over time
            for i in range(10):  # Would generate 10 tokens normally
                if self.interrupted:
                    # Simulate abort: create final output with partial content
                    final_output = RequestOutput(
                        request_id=request_id,
                        prompt=prompt,
                        prompt_token_ids=[1, 2, 3],
                        outputs=[MagicMock(
                            text=f"Partial response with {i} tokens",
                            token_ids=list(range(i)),
                            finish_reason="abort",
                            stop_reason=None
                        )],
                        finished=True
                    )
                    yield final_output
                    return
                    
                # Normal streaming output
                output = RequestOutput(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=[1, 2, 3],
                    outputs=[MagicMock(
                        text=f"Token {i}",
                        token_ids=[i],
                        finish_reason=None if i < 9 else "stop",
                        stop_reason=None
                    )],
                    finished=i >= 9
                )
                self.active_streams[request_id]["tokens"] = i + 1
                yield output
                await asyncio.sleep(0.1)  # Simulate processing time
                
        finally:
            self.active_streams.pop(request_id, None)
    
    async def abort_all_active(self) -> int:
        """Mock implementation of abort_all_active."""
        active_count = len(self.active_streams)
        self.interrupted = True
        # In real implementation, this would trigger finalize_and_abort_all
        return active_count
    
    async def collective_rpc(self, method: str, **kwargs):
        """Mock collective RPC for weight validation/loading."""
        if method == "validate_sharded_state":
            return [{"tensor_count": 100, "mismatches": []}]
        elif method == "load_sharded_state":
            return [{"ok": True, "rank": 0}]
        return []


class TestStreamingWithWeightUpdate:
    """Test streaming requests during weight updates."""
    
    @pytest.fixture
    def mock_engine(self):
        return MockStreamingEngine()
    
    @pytest.fixture
    def temp_model_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock safetensors files
            filename = "model-rank-0-part-0.safetensors"
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, "wb") as f:
                f.write(b"mock_data")
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_streaming_interrupted_by_weight_update(self, mock_engine):
        """Test that streaming requests are properly interrupted during weight update."""
        # Start a streaming request
        request_task = asyncio.create_task(
            self._collect_stream_outputs(mock_engine, "req_1", "Hello world", max_tokens=20)
        )
        
        # Let it generate a few tokens
        await asyncio.sleep(0.25)  # Should generate ~2-3 tokens
        
        # Simulate weight update interruption
        aborted_count = await mock_engine.abort_all_active()
        
        # Wait for stream to complete
        outputs = await request_task
        
        # Verify behavior
        assert aborted_count == 1  # One active request was aborted
        assert len(outputs) > 0  # Should have received some outputs
        
        # Last output should be the abort signal
        final_output = outputs[-1]
        assert final_output.finished is True
        assert final_output.outputs[0].finish_reason == "abort"
        assert "Partial response" in final_output.outputs[0].text
    
    @pytest.mark.asyncio
    async def test_multiple_streams_interrupted(self, mock_engine):
        """Test multiple concurrent streams interrupted by weight update."""
        # Start multiple streaming requests
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                self._collect_stream_outputs(mock_engine, f"req_{i}", f"Prompt {i}", max_tokens=15)
            )
            tasks.append(task)
        
        # Let them generate some tokens
        await asyncio.sleep(0.3)
        
        # Interrupt all
        aborted_count = await mock_engine.abort_all_active()
        
        # Wait for all streams to complete
        all_outputs = await asyncio.gather(*tasks)
        
        # Verify
        assert aborted_count == 3  # Three active requests
        
        # Each stream should have received partial content + abort
        for outputs in all_outputs:
            assert len(outputs) > 0
            final_output = outputs[-1]
            assert final_output.finished is True
            assert final_output.outputs[0].finish_reason == "abort"
    
    async def _collect_stream_outputs(self, engine, request_id: str, prompt: str, max_tokens: int):
        """Helper to collect all outputs from a stream."""
        outputs = []
        sampling_params = SamplingParams(
            max_tokens=max_tokens, 
            temperature=0.5,
            output_kind=RequestOutputKind.DELTA
        )
        
        async for output in engine.generate(request_id, prompt, sampling_params):
            outputs.append(output)
            if output.finished:
                break
        
        return outputs


class TestWeightUpdateEndpointIntegration:
    """Integration tests for the complete weight update endpoint with interruption."""
    
    @pytest.fixture
    def temp_model_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = "model-rank-0-part-0.safetensors"
            with open(os.path.join(temp_dir, filename), "wb") as f:
                f.write(b"mock_data")
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_complete_weight_update_flow(self, temp_model_dir):
        """Test complete flow: start streams -> weight update -> verify interruption."""
        
        with patch('vllm.entrypoints.openai.api_server.engine_client') as mock_engine_client:
            # Setup mock engine with streaming capability
            mock_engine = MockStreamingEngine()
            mock_engine_client.return_value = mock_engine
            
            # Mock additional required attributes for endpoint
            hasattr_orig = hasattr
            def mock_hasattr(obj, attr):
                if attr == "abort_all_active":
                    return True
                return hasattr_orig(obj, attr)
            
            with patch('builtins.hasattr', side_effect=mock_hasattr):
                with patch('vllm.entrypoints.openai.api_server.getattr') as mock_getattr:
                    mock_getattr.side_effect = lambda obj, attr, default=None: getattr(mock_engine, attr, default)
                    
                    # Start some background "requests" 
                    mock_engine.active_streams["bg_req_1"] = {"tokens": 5}
                    mock_engine.active_streams["bg_req_2"] = {"tokens": 3}
                    
                    # Create request for weight update
                    mock_request = MagicMock()
                    mock_request.json = AsyncMock(return_value={
                        "path": temp_model_dir,
                        "interrupt": True,
                        "dry_run": True  # Skip actual loading for test
                    })
                    mock_request.app.state.weight_update_in_progress = False
                    mock_request.app.state.server_load_metrics = 0
                    
                    # Import endpoint and execute
                    from vllm.entrypoints.openai.api_server import update_weights_from_disk
                    
                    with patch('vllm.entrypoints.openai.api_server.setattr'):
                        with patch('vllm.entrypoints.openai.api_server.os.path.isdir', return_value=True):
                            with patch('vllm.entrypoints.openai.api_server.glob.glob', return_value=[f"{temp_model_dir}/model-rank-0-part-0.safetensors"]):
                                response = await update_weights_from_disk(mock_request)
                    
                    # Verify results
                    response_data = json.loads(response.body.decode())
                    assert response_data["ok"] is True
                    assert response_data["num_interrupted_requests"] == 2  # bg_req_1 and bg_req_2
                    assert "validated_tensors" in response_data
                    assert mock_engine.interrupted is True
    
    @pytest.mark.asyncio
    async def test_weight_update_with_validation_failure(self, temp_model_dir):
        """Test weight update when validation fails - should still interrupt first."""
        
        with patch('vllm.entrypoints.openai.api_server.engine_client') as mock_engine_client:
            mock_engine = MockStreamingEngine()
            
            # Make validation fail
            async def failing_rpc(method, **kwargs):
                if method == "validate_sharded_state":
                    return [{"tensor_count": 0, "mismatches": [
                        {"kind": "shape_mismatch", "name": "layer.weight", "expected": [100, 50], "actual": [100, 60]}
                    ]}]
                return []
            
            mock_engine.collective_rpc = failing_rpc
            mock_engine_client.return_value = mock_engine
            
            # Setup active requests
            mock_engine.active_streams["active_1"] = {"tokens": 10}
            
            hasattr_orig = hasattr
            def mock_hasattr(obj, attr):
                if attr == "abort_all_active":
                    return True
                return hasattr_orig(obj, attr)
            
            with patch('builtins.hasattr', side_effect=mock_hasattr):
                with patch('vllm.entrypoints.openai.api_server.getattr') as mock_getattr:
                    mock_getattr.side_effect = lambda obj, attr, default=None: getattr(mock_engine, attr, default)
                    
                    mock_request = MagicMock()
                    mock_request.json = AsyncMock(return_value={
                        "path": temp_model_dir,
                        "interrupt": True,
                    })
                    mock_request.app.state.weight_update_in_progress = False
                    mock_request.app.state.server_load_metrics = 0
                    
                    from vllm.entrypoints.openai.api_server import update_weights_from_disk
                    
                    with patch('vllm.entrypoints.openai.api_server.setattr'):
                        with patch('vllm.entrypoints.openai.api_server.os.path.isdir', return_value=True):
                            with patch('vllm.entrypoints.openai.api_server.glob.glob', return_value=[f"{temp_model_dir}/model-rank-0-part-0.safetensors"]):
                                response = await update_weights_from_disk(mock_request)
                    
                    # Verify: requests were interrupted even though validation failed
                    assert mock_engine.interrupted is True
                    
                    # Response should indicate validation failure
                    response_data = json.loads(response.body.decode())
                    assert response_data["ok"] is False
                    assert response_data["validation_failed"] is True
                    assert len(response_data["mismatches"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
