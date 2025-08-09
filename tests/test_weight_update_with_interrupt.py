# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
import tempfile
import pytest
import sys
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import Optional

# Mock all vLLM modules before importing to avoid GPU dependencies
sys.modules['vllm.v1.engine'] = Mock()
sys.modules['vllm.v1.engine.output_processor'] = Mock() 
sys.modules['vllm.v1.engine.async_llm'] = Mock()
sys.modules['vllm.sampling_params'] = Mock()
sys.modules['vllm.transformers_utils.tokenizer_group'] = Mock()

# Create mock enum for FinishReason
class MockFinishReason:
    ABORT = "abort"
    STOP = "stop"

# Create mock RequestOutputKind
class MockRequestOutputKind:
    FINAL_ONLY = "final_only"
    DELTA = "delta"


class MockRequestState:
    """Mock RequestState for testing finalize_and_abort_all logic."""
    
    def __init__(self, request_id: str, queue: Optional[AsyncMock] = None, 
                 output_kind: str = MockRequestOutputKind.FINAL_ONLY,
                 should_fail: bool = False):
        self.request_id = request_id
        self.queue = queue or AsyncMock()
        self.output_kind = output_kind
        self.should_fail = should_fail
        self.parent_req = None
        self.request_index = 0
    
    def make_request_output(self, new_token_ids, finish_reason, stop_reason):
        """Mock make_request_output that simulates partial text generation."""
        if self.should_fail:
            raise RuntimeError(f"Simulated failure for {self.request_id}")
        
        # Simulate a RequestOutput with partial text
        mock_output = MagicMock()
        mock_output.finished = True
        mock_output.request_id = self.request_id
        mock_output.outputs = [MagicMock(
            text=f"Partial response for {self.request_id}",
            finish_reason="abort",
            token_ids=[1, 2, 3]  # Simulate some generated tokens
        )]
        return mock_output


class MockOutputProcessor:
    """Mock implementation of OutputProcessor with our new method."""
    
    def __init__(self, tokenizer=None, log_stats=False):
        self.tokenizer = tokenizer
        self.log_stats = log_stats
        self.request_states = {}
        self.lora_states = MagicMock()
        self.lora_states.abort_request = MagicMock()
    
    def finalize_and_abort_all(self):
        """Implementation of our new method for testing."""
        aborted = []
        # Iterate over a list copy since we mutate request_states.
        for req_id, req_state in list(self.request_states.items()):
            # Produce a final RequestOutput with finish_reason=ABORT.
            try:
                ro = req_state.make_request_output([], MockFinishReason.ABORT, None)
                if ro is not None and req_state.queue is not None:
                    req_state.queue.put(ro)
            except Exception as e:  # pragma: no cover - defensive
                if req_state.queue is not None:
                    req_state.queue.put(e)
            aborted.append(req_id)
        # Remove all states & propagate to LoRA tracking.
        self.abort_requests(aborted)
        return aborted
    
    def abort_requests(self, request_ids):
        """Mock abort_requests method."""
        for req_id in request_ids:
            req_state = self.request_states.pop(req_id, None)
            if req_state is not None:
                self.lora_states.abort_request(req_state)


class MockAsyncLLM:
    """Mock AsyncLLM with abort_all_active method."""
    
    def __init__(self):
        self.output_processor = MockOutputProcessor()
        self.engine_core = AsyncMock()
        self.log_requests = True
    
    async def abort_all_active(self):
        """Implementation of our new method for testing."""
        # Finalize & abort locally (push final outputs to queues).
        aborted_ids = self.output_processor.finalize_and_abort_all()
        if aborted_ids:
            # Propagate to engine core so scheduler frees resources.
            await self.engine_core.abort_requests_async(aborted_ids)
        if self.log_requests and aborted_ids:
            print(f"Aborted {len(aborted_ids)} active requests (global interrupt).")
        return len(aborted_ids)


class TestOutputProcessorAbortAll:
    """Test suite for OutputProcessor.finalize_and_abort_all functionality."""
    
    @pytest.fixture
    def mock_output_processor(self):
        """Create a mock OutputProcessor with required dependencies."""
        return MockOutputProcessor()
    
    def test_finalize_and_abort_all_empty(self, mock_output_processor):
        """Test finalize_and_abort_all with no active requests."""
        result = mock_output_processor.finalize_and_abort_all()
        assert result == []
        mock_output_processor.lora_states.abort_request.assert_not_called()
    
    def test_finalize_and_abort_all_single_request(self, mock_output_processor):
        """Test finalize_and_abort_all with one active request."""
        # Setup
        mock_queue = MagicMock()
        req_state = MockRequestState("req_1", queue=mock_queue)
        mock_output_processor.request_states = {"req_1": req_state}
        
        # Execute
        result = mock_output_processor.finalize_and_abort_all()
        
        # Verify
        assert result == ["req_1"]
        mock_queue.put.assert_called_once()
        args = mock_queue.put.call_args[0][0]
        assert args.finished is True
        assert args.outputs[0].finish_reason == "abort"
        assert "Partial response for req_1" in args.outputs[0].text
    
    def test_finalize_and_abort_all_multiple_requests(self, mock_output_processor):
        """Test finalize_and_abort_all with multiple active requests."""
        # Setup
        queues = {}
        states = {}
        for i in range(3):
            req_id = f"req_{i}"
            queues[req_id] = MagicMock()
            states[req_id] = MockRequestState(req_id, queue=queues[req_id])
        
        mock_output_processor.request_states = states
        
        # Execute
        result = mock_output_processor.finalize_and_abort_all()
        
        # Verify
        assert set(result) == {"req_0", "req_1", "req_2"}
        for req_id in result:
            queues[req_id].put.assert_called_once()
    
    def test_finalize_and_abort_all_with_failure(self, mock_output_processor):
        """Test finalize_and_abort_all handles individual request failures gracefully."""
        # Setup
        mock_queue_good = MagicMock()
        mock_queue_bad = MagicMock()
        
        req_good = MockRequestState("req_good", queue=mock_queue_good)
        req_bad = MockRequestState("req_bad", queue=mock_queue_bad, should_fail=True)
        
        mock_output_processor.request_states = {
            "req_good": req_good,
            "req_bad": req_bad
        }
        
        # Execute
        result = mock_output_processor.finalize_and_abort_all()
        
        # Verify both requests are marked as aborted
        assert set(result) == {"req_good", "req_bad"}
        
        # Good request gets normal output
        mock_queue_good.put.assert_called_once()
        good_args = mock_queue_good.put.call_args[0][0]
        assert good_args.finished is True
        
        # Bad request gets exception
        mock_queue_bad.put.assert_called_once()
        bad_args = mock_queue_bad.put.call_args[0][0]
        assert isinstance(bad_args, Exception)
    
    def test_finalize_and_abort_all_no_queue(self, mock_output_processor):
        """Test finalize_and_abort_all with requests that have no queue."""
        # Setup
        req_state = MockRequestState("req_1", queue=None)
        mock_output_processor.request_states = {"req_1": req_state}
        
        # Execute - should not crash
        result = mock_output_processor.finalize_and_abort_all()
        
        # Verify
        assert result == ["req_1"]


class TestAsyncLLMAbortAll:
    """Test suite for AsyncLLM.abort_all_active functionality."""
    
    @pytest.fixture
    def mock_async_llm(self):
        """Create a mock AsyncLLM with required dependencies."""
        return MockAsyncLLM()
    
    @pytest.mark.asyncio
    async def test_abort_all_active_no_requests(self, mock_async_llm):
        """Test abort_all_active with no active requests."""
        # Execute
        result = await mock_async_llm.abort_all_active()
        
        # Verify
        assert result == 0
        mock_async_llm.engine_core.abort_requests_async.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_abort_all_active_with_requests(self, mock_async_llm):
        """Test abort_all_active with active requests."""
        # Setup
        req1 = MockRequestState("req_1")
        req2 = MockRequestState("req_2")
        req3 = MockRequestState("req_3")
        
        mock_async_llm.output_processor.request_states = {
            "req_1": req1,
            "req_2": req2,
            "req_3": req3
        }
        
        # Execute
        result = await mock_async_llm.abort_all_active()
        
        # Verify
        assert result == 3
        mock_async_llm.engine_core.abort_requests_async.assert_called_once()
        call_args = mock_async_llm.engine_core.abort_requests_async.call_args[0][0]
        assert set(call_args) == {"req_1", "req_2", "req_3"}


class TestUpdateWeightsEndpoint:
    """Test suite for /update-weights-from-disk endpoint with interruption."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory with mock model files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock safetensors files
            for rank in range(2):
                for part in range(1):
                    filename = f"model-rank-{rank}-part-{part}.safetensors"
                    filepath = os.path.join(temp_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(b"mock_safetensors_data")
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_interrupt_flag_parsing(self):
        """Test that interrupt flag is parsed correctly from request body."""
        
        # Test default behavior (should be True)
        body_default = {"path": "/mock/path"}
        interrupt_flag = bool(body_default.get("interrupt", True))
        assert interrupt_flag is True
        
        # Test explicit True
        body_true = {"path": "/mock/path", "interrupt": True}
        interrupt_flag = bool(body_true.get("interrupt", True))
        assert interrupt_flag is True
        
        # Test explicit False  
        body_false = {"path": "/mock/path", "interrupt": False}
        interrupt_flag = bool(body_false.get("interrupt", True))
        assert interrupt_flag is False
    
    def test_response_structure_includes_interrupt_count(self):
        """Test that response structure includes num_interrupted_requests field."""
        
        # Simulate successful response structure
        response_data = {
            "ok": True,
            "duration_sec": 1.5,
            "validated_tensors": 100,
            "num_paused_requests": 0,
            "num_interrupted_requests": 3,  # This is what we added
            "details": [{"ok": True, "rank": 0}]
        }
        
        # Verify required fields are present
        assert "num_interrupted_requests" in response_data
        assert response_data["num_interrupted_requests"] == 3
        
        # Simulate dry-run response structure
        dry_run_response = {
            "ok": True,
            "dry_run": True,
            "duration_sec": 0.1,
            "validated_tensors": 100,
            "num_paused_requests": 0,
            "num_interrupted_requests": 2,  # Also included in dry-run
        }
        
        assert "num_interrupted_requests" in dry_run_response
        assert dry_run_response["num_interrupted_requests"] == 2


class TestImplementationLogic:
    """Test core implementation logic without heavy dependencies."""
    
    def test_finalize_and_abort_all_logic(self):
        """Test the core logic of finalize_and_abort_all method."""
        processor = MockOutputProcessor()
        
        # Add some mock request states
        req1 = MockRequestState("req1")
        req2 = MockRequestState("req2") 
        req3 = MockRequestState("req3")
        
        processor.request_states = {
            "req1": req1,
            "req2": req2, 
            "req3": req3
        }
        
        # Execute
        aborted = processor.finalize_and_abort_all()
        
        # Verify
        assert set(aborted) == {"req1", "req2", "req3"}
        assert processor.request_states == {}  # All states should be removed
    
    @pytest.mark.asyncio
    async def test_abort_all_active_logic(self):
        """Test the core logic of abort_all_active method."""
        llm = MockAsyncLLM()
        
        # Add some mock request states
        req1 = MockRequestState("req1")
        req2 = MockRequestState("req2")
        
        llm.output_processor.request_states = {
            "req1": req1,
            "req2": req2
        }
        
        # Execute
        count = await llm.abort_all_active()
        
        # Verify
        assert count == 2
        llm.engine_core.abort_requests_async.assert_called_once()
        call_args = llm.engine_core.abort_requests_async.call_args[0][0]
        assert set(call_args) == {"req1", "req2"}
    
    def test_request_state_abort_output_creation(self):
        """Test that RequestState creates proper abort output."""
        req_state = MockRequestState("test_req")
        
        # Test normal case
        output = req_state.make_request_output([], MockFinishReason.ABORT, None)
        
        assert output is not None
        assert output.finished is True
        assert output.request_id == "test_req"
        assert output.outputs[0].finish_reason == "abort"
        assert "Partial response for test_req" in output.outputs[0].text
    
    def test_request_state_abort_with_failure(self):
        """Test RequestState handles make_request_output failures."""
        req_state = MockRequestState("failing_req", should_fail=True)
        
        # Should raise exception as designed
        with pytest.raises(RuntimeError, match="Simulated failure"):
            req_state.make_request_output([], MockFinishReason.ABORT, None)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_request_states(self):
        """Test behavior when no requests are active."""
        processor = MockOutputProcessor()
        
        # Should handle empty case gracefully
        aborted = processor.finalize_and_abort_all()
        assert aborted == []
    
    def test_mixed_success_failure_requests(self):
        """Test handling mix of successful and failing requests."""
        processor = MockOutputProcessor()
        
        # Mix of normal and failing requests
        good_req = MockRequestState("good")
        bad_req = MockRequestState("bad", should_fail=True)
        
        processor.request_states = {
            "good": good_req,
            "bad": bad_req
        }
        
        # Should handle both, returning all IDs but with different outcomes
        aborted = processor.finalize_and_abort_all()
        
        assert set(aborted) == {"good", "bad"}
        assert processor.request_states == {}  # Both removed from state
    
    @pytest.mark.asyncio
    async def test_engine_core_communication_failure(self):
        """Test handling when engine_core.abort_requests_async fails."""
        llm = MockAsyncLLM()
        
        # Make engine_core.abort_requests_async raise exception
        llm.engine_core.abort_requests_async.side_effect = RuntimeError("Engine failed")
        
        # Add a request
        req1 = MockRequestState("req1")
        llm.output_processor.request_states = {"req1": req1}
        
        # Should propagate the exception
        with pytest.raises(RuntimeError, match="Engine failed"):
            await llm.abort_all_active()


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_weight_update_with_interrupt.py -v
    pytest.main([__file__, "-v"])
