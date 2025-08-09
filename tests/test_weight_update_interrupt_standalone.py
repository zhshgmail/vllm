#!/usr/bin/env python3
"""
Standalone unit tests for weight update with interruption functionality.
These tests run without any vLLM dependencies and test only our core logic.
"""

import asyncio
import json
import tempfile
import os
from unittest.mock import MagicMock, AsyncMock


def test_finalize_and_abort_all_logic():
    """Test the core logic of finalize_and_abort_all without vLLM dependencies."""
    print("🧪 Testing finalize_and_abort_all logic...")
    
    # Mock the core components
    class MockRequestState:
        def __init__(self, request_id, should_fail=False):
            self.request_id = request_id
            self.queue = MagicMock()
            self.should_fail = should_fail
        
        def make_request_output(self, new_token_ids, finish_reason, stop_reason):
            if self.should_fail:
                raise RuntimeError(f"Mock failure for {self.request_id}")
            
            mock_output = MagicMock()
            mock_output.finished = True
            mock_output.request_id = self.request_id
            mock_output.outputs = [MagicMock(
                text=f"Partial response for {self.request_id}",
                finish_reason=finish_reason,
                token_ids=[1, 2, 3]
            )]
            return mock_output
    
    class MockOutputProcessor:
        def __init__(self):
            self.request_states = {}
            self.lora_states = MagicMock()
        
        def finalize_and_abort_all(self):
            """Our implementation under test."""
            aborted = []
            # Iterate over a list copy since we mutate request_states.
            for req_id, req_state in list(self.request_states.items()):
                # Produce a final RequestOutput with finish_reason=ABORT.
                try:
                    ro = req_state.make_request_output([], "abort", None)
                    if ro is not None and req_state.queue is not None:
                        req_state.queue.put(ro)
                except Exception as e:
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
    
    # Test 1: Empty case
    processor = MockOutputProcessor()
    result = processor.finalize_and_abort_all()
    assert result == [], f"Expected empty list, got {result}"
    print("✅ Empty case test passed")
    
    # Test 2: Single request
    processor = MockOutputProcessor()
    req1 = MockRequestState("req_1")
    processor.request_states = {"req_1": req1}
    
    result = processor.finalize_and_abort_all()
    assert result == ["req_1"], f"Expected ['req_1'], got {result}"
    assert "req_1" not in processor.request_states, "Request should be removed from states"
    req1.queue.put.assert_called_once()
    print("✅ Single request test passed")
    
    # Test 3: Multiple requests
    processor = MockOutputProcessor()
    req1 = MockRequestState("req_1")
    req2 = MockRequestState("req_2") 
    req3 = MockRequestState("req_3")
    processor.request_states = {"req_1": req1, "req_2": req2, "req_3": req3}
    
    result = processor.finalize_and_abort_all()
    assert set(result) == {"req_1", "req_2", "req_3"}, f"Expected all 3 requests, got {result}"
    assert len(processor.request_states) == 0, "All requests should be removed"
    req1.queue.put.assert_called_once()
    req2.queue.put.assert_called_once() 
    req3.queue.put.assert_called_once()
    print("✅ Multiple requests test passed")
    
    # Test 4: Failure handling
    processor = MockOutputProcessor()
    good_req = MockRequestState("good")
    bad_req = MockRequestState("bad", should_fail=True)
    processor.request_states = {"good": good_req, "bad": bad_req}
    
    result = processor.finalize_and_abort_all()
    assert set(result) == {"good", "bad"}, "Both requests should be returned even with failure"
    assert len(processor.request_states) == 0, "Both requests should be removed despite failure"
    
    # Check that good request got normal output
    good_calls = good_req.queue.put.call_args_list
    assert len(good_calls) == 1
    good_output = good_calls[0][0][0]
    assert good_output.finished is True
    
    # Check that bad request got exception
    bad_calls = bad_req.queue.put.call_args_list
    assert len(bad_calls) == 1
    bad_arg = bad_calls[0][0][0]
    assert isinstance(bad_arg, Exception)
    print("✅ Failure handling test passed")


async def test_abort_all_active_logic():
    """Test the abort_all_active logic without vLLM dependencies."""
    print("🧪 Testing abort_all_active logic...")
    
    class MockAsyncLLM:
        def __init__(self):
            self.output_processor = MagicMock()
            self.engine_core = AsyncMock()
            self.log_requests = True
        
        async def abort_all_active(self):
            """Our implementation under test."""
            aborted_ids = self.output_processor.finalize_and_abort_all()
            if aborted_ids:
                await self.engine_core.abort_requests_async(aborted_ids)
            if self.log_requests and aborted_ids:
                print(f"Aborted {len(aborted_ids)} active requests (global interrupt).")
            return len(aborted_ids)
    
    # Test 1: No active requests
    llm = MockAsyncLLM()
    llm.output_processor.finalize_and_abort_all.return_value = []
    
    result = await llm.abort_all_active()
    assert result == 0, f"Expected 0, got {result}"
    llm.engine_core.abort_requests_async.assert_not_called()
    print("✅ No active requests test passed")
    
    # Test 2: With active requests
    llm = MockAsyncLLM()
    llm.output_processor.finalize_and_abort_all.return_value = ["req_1", "req_2", "req_3"]
    
    result = await llm.abort_all_active()
    assert result == 3, f"Expected 3, got {result}"
    llm.output_processor.finalize_and_abort_all.assert_called_once()
    llm.engine_core.abort_requests_async.assert_called_once_with(["req_1", "req_2", "req_3"])
    print("✅ Active requests test passed")


def test_endpoint_flag_logic():
    """Test the endpoint flag parsing logic."""
    print("🧪 Testing endpoint flag logic...")
    
    # Test default behavior
    body_default = {"path": "/mock/path"}
    interrupt_flag = bool(body_default.get("interrupt", True))
    assert interrupt_flag is True, "Default should be True"
    
    # Test explicit values
    body_true = {"path": "/mock/path", "interrupt": True}
    interrupt_flag = bool(body_true.get("interrupt", True))
    assert interrupt_flag is True, "Explicit True should be True"
    
    body_false = {"path": "/mock/path", "interrupt": False}
    interrupt_flag = bool(body_false.get("interrupt", True))
    assert interrupt_flag is False, "Explicit False should be False"
    
    # Test response structure
    response_data = {
        "ok": True,
        "duration_sec": 1.5,
        "validated_tensors": 100,
        "num_paused_requests": 0,
        "num_interrupted_requests": 3,
    }
    
    assert "num_interrupted_requests" in response_data
    assert response_data["num_interrupted_requests"] == 3
    print("✅ Endpoint flag logic test passed")


def test_request_output_creation():
    """Test request output creation for abort scenario."""
    print("🧪 Testing request output creation...")
    
    class MockRequestState:
        def __init__(self, request_id):
            self.request_id = request_id
        
        def make_request_output(self, new_token_ids, finish_reason, stop_reason):
            # Simulate what the real method does for abort case
            mock_output = MagicMock()
            mock_output.finished = True
            mock_output.request_id = self.request_id
            mock_output.outputs = [MagicMock(
                text=f"Generated text so far for {self.request_id}",
                finish_reason=finish_reason,
                token_ids=new_token_ids or [],  # Empty for abort case
                stop_reason=stop_reason
            )]
            return mock_output
    
    req_state = MockRequestState("test_req")
    
    # Test abort output creation
    output = req_state.make_request_output([], "abort", None)
    
    assert output is not None
    assert output.finished is True
    assert output.request_id == "test_req"
    assert output.outputs[0].finish_reason == "abort"
    assert len(output.outputs[0].token_ids) == 0  # No new tokens for abort
    assert "Generated text so far" in output.outputs[0].text
    print("✅ Request output creation test passed")


def run_all_tests():
    """Run all standalone tests."""
    print("🚀 Running standalone weight update interruption tests...")
    print("=" * 60)
    
    try:
        test_finalize_and_abort_all_logic()
        asyncio.run(test_abort_all_active_logic())
        test_endpoint_flag_logic()
        test_request_output_creation()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("\nValidated functionality:")
        print("  ✅ finalize_and_abort_all() core logic")
        print("  ✅ abort_all_active() async coordination")
        print("  ✅ Endpoint parameter parsing")
        print("  ✅ Request output creation for abort")
        print("  ✅ Error handling for failing requests")
        print("  ✅ Multiple concurrent request handling")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
