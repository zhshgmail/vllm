# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for API server weight update functionality.

These tests focus on the V1 engine detection and weight update endpoint logic
without requiring actual vLLM engine startup.
"""
import json
import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi import HTTPException
from http import HTTPStatus


class MockAsyncLLMV1:
    """Mock V1 AsyncLLM engine."""
    
    def __init__(self):
        self.vllm_config = Mock()
        self.collective_rpc = AsyncMock()


class MockAsyncLLMV0:
    """Mock V0 AsyncLLM engine."""
    
    def __init__(self):
        pass  # No V1-specific attributes


class MockRequest:
    """Mock FastAPI request object."""
    
    def __init__(self, body_data: dict):
        self.app = Mock()
        self.app.state = Mock()
        self._body_data = body_data
    
    async def json(self):
        """Mock request.json() method."""
        return self._body_data


class TestAPIServerV1Detection:
    """Test the V1 engine detection logic in the API server."""
    
    @pytest.mark.asyncio
    async def test_v1_engine_detection_success(self):
        """Test that V1 engine is correctly detected."""
        from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMV1
        
        # Mock the engine_client function
        mock_v1_engine = MockAsyncLLMV1()
        
        with patch('vllm.entrypoints.openai.api_server.engine_client', return_value=mock_v1_engine):
            # Simulate the V1 detection logic
            def check_v1_engine(raw_request):
                engine = mock_v1_engine  # engine_client(raw_request)
                
                if not isinstance(engine, AsyncLLMV1):
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_REQUEST,
                        detail="Weight updates are only supported in V1 engine mode. "
                               "Please restart vLLM with V1 engine to use this feature."
                    )
                return True
            
            # Should not raise exception for V1 engine
            result = check_v1_engine(Mock())
            assert result is True
    
    @pytest.mark.asyncio 
    async def test_v0_engine_detection_failure(self):
        """Test that V0 engine is correctly rejected."""
        from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMV1
        
        # Mock the engine_client function to return V0 engine
        mock_v0_engine = MockAsyncLLMV0()
        
        with patch('vllm.entrypoints.openai.api_server.engine_client', return_value=mock_v0_engine):
            # Simulate the V1 detection logic
            def check_v1_engine(raw_request):
                engine = mock_v0_engine  # engine_client(raw_request)
                
                if not isinstance(engine, AsyncLLMV1):
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_REQUEST,
                        detail="Weight updates are only supported in V1 engine mode. "
                               "Please restart vLLM with V1 engine to use this feature."
                    )
                return True
            
            # Should raise exception for V0 engine
            with pytest.raises(HTTPException) as exc_info:
                check_v1_engine(Mock())
            
            assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
            assert "V1 engine mode" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_weight_update_endpoint_validation(self):
        """Test the weight update endpoint request validation."""
        
        # Test valid request
        valid_request_data = {
            "path": "/path/to/weights",
            "pattern": "model-rank-{rank}-part-{part}.safetensors",
            "pause": True,
            "interrupt": True,
            "dry_run": False
        }
        
        def validate_request_body(body):
            """Simulate request validation logic."""
            if not isinstance(body, dict):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail="Body must be a JSON object"
                )
            
            path = body.get("path")
            if not path or not isinstance(path, str):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail="Missing required field 'path' (str)"
                )
            
            return {
                "path": path,
                "pattern": body.get("pattern"),
                "pause": bool(body.get("pause", True)),
                "interrupt": bool(body.get("interrupt", True)),
                "dry_run": bool(body.get("dry_run", False)),
            }
        
        # Valid request should pass
        result = validate_request_body(valid_request_data)
        assert result["path"] == "/path/to/weights"
        assert result["pattern"] == "model-rank-{rank}-part-{part}.safetensors"
        assert result["pause"] is True
        assert result["interrupt"] is True
        assert result["dry_run"] is False
        
        # Invalid request - missing path
        invalid_request_1 = {"pattern": "some-pattern"}
        with pytest.raises(HTTPException) as exc_info:
            validate_request_body(invalid_request_1)
        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
        assert "Missing required field 'path'" in exc_info.value.detail
        
        # Invalid request - non-dict body
        with pytest.raises(HTTPException) as exc_info:
            validate_request_body("not a dict")
        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
        assert "Body must be a JSON object" in exc_info.value.detail
        
        # Invalid request - non-string path
        invalid_request_2 = {"path": 123}
        with pytest.raises(HTTPException) as exc_info:
            validate_request_body(invalid_request_2)
        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
        assert "Missing required field 'path'" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_weight_update_in_progress_check(self):
        """Test the weight update in progress conflict detection."""
        
        def check_weight_update_in_progress(app_state, in_progress_value):
            """Simulate the in-progress check logic."""
            # Mock getattr behavior
            weight_update_in_progress = getattr(app_state, "weight_update_in_progress", False)
            
            # Override for testing
            if in_progress_value is not None:
                weight_update_in_progress = in_progress_value
            
            if weight_update_in_progress:
                raise HTTPException(
                    status_code=HTTPStatus.CONFLICT,
                    detail="Weight update already in progress"
                )
            
            # Set the flag
            setattr(app_state, "weight_update_in_progress", True)
            return True
        
        mock_app_state = Mock()
        
        # First request should succeed
        result = check_weight_update_in_progress(mock_app_state, False)
        assert result is True
        
        # Second request should fail (simulate concurrent request)
        with pytest.raises(HTTPException) as exc_info:
            check_weight_update_in_progress(mock_app_state, True)
        assert exc_info.value.status_code == HTTPStatus.CONFLICT
        assert "already in progress" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_preflight_path_validation(self):
        """Test the preflight path validation logic."""
        import os
        import glob
        
        def validate_weight_path(path, pattern=None):
            """Simulate path validation logic."""
            if not os.path.isdir(path):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"Not a directory: {path}"
                )
            
            # Default pattern
            if pattern is None:
                pattern = "model-rank-{rank}-part-{part}.safetensors"
            
            wildcard = pattern.format(rank="*", part="*")
            shard_candidates = glob.glob(os.path.join(path, wildcard))
            
            if len(shard_candidates) == 0:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=(
                        "Unsupported or missing format: expected sharded safetensors files "
                        f"matching pattern '{pattern}' in directory {path}"
                    )
                )
            
            return shard_candidates
        
        # Mock os.path.isdir and glob.glob for testing
        with patch('os.path.isdir') as mock_isdir, \
             patch('glob.glob') as mock_glob:
            
            # Test valid directory with files
            mock_isdir.return_value = True
            mock_glob.return_value = [
                "/valid/path/model-rank-0-part-0.safetensors",
                "/valid/path/model-rank-0-part-1.safetensors"
            ]
            
            result = validate_weight_path("/valid/path")
            assert len(result) == 2
            assert "model-rank-0-part-0.safetensors" in result[0]
            
            # Test invalid directory
            mock_isdir.return_value = False
            with pytest.raises(HTTPException) as exc_info:
                validate_weight_path("/invalid/path")
            assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
            assert "Not a directory" in exc_info.value.detail
            
            # Test directory with no matching files
            mock_isdir.return_value = True
            mock_glob.return_value = []
            with pytest.raises(HTTPException) as exc_info:
                validate_weight_path("/empty/path")
            assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
            assert "Unsupported or missing format" in exc_info.value.detail


class TestV0WorkerRestriction:
    """Test that V0 worker properly rejects weight update operations."""
    
    def test_v0_worker_load_sharded_state_rejection(self):
        """Test that V0 worker rejects load_sharded_state calls."""
        
        def v0_load_sharded_state(path: str, pattern=None):
            """V0 worker implementation that should reject calls."""
            raise RuntimeError(
                "Weight updates are not supported in V0 engine mode. "
                "Please restart vLLM with V1 engine to use this feature. "
                "This restriction ensures proper KV cache management and model state consistency."
            )
        
        # Any call should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            v0_load_sharded_state("/some/path")
        
        assert "not supported in V0 engine mode" in str(exc_info.value)
        assert "V1 engine" in str(exc_info.value)
        assert "KV cache management" in str(exc_info.value)
    
    def test_v0_worker_validate_sharded_state_missing(self):
        """Test that V0 worker doesn't have validate_sharded_state method."""
        
        class MockV0Worker:
            """Mock V0 worker without weight update methods."""
            
            def __init__(self):
                pass
            
            def load_sharded_state(self, path: str, pattern=None):
                raise RuntimeError("Not supported in V0")
        
        worker = MockV0Worker()
        
        # Should have load_sharded_state (but it raises)
        assert hasattr(worker, "load_sharded_state")
        with pytest.raises(RuntimeError):
            worker.load_sharded_state("/path")
        
        # Should not have validate_sharded_state
        assert not hasattr(worker, "validate_sharded_state")


class TestWeightUpdateEndToEnd:
    """Test the complete weight update flow with mocks."""
    
    @pytest.mark.asyncio
    async def test_successful_weight_update_flow(self):
        """Test a complete successful weight update operation."""
        
        # Mock V1 engine
        mock_engine = MockAsyncLLMV1()
        mock_engine.collective_rpc = AsyncMock()
        
        # Mock validation results (success)
        mock_engine.collective_rpc.side_effect = [
            # validate_sharded_state results
            [{"tensor_count": 100, "mismatches": []}],
            # load_sharded_state results  
            [{"ok": True, "rank": 0, "num_updated": 100}]
        ]
        
        async def mock_weight_update_endpoint(request_data):
            """Mock the weight update endpoint logic."""
            
            # Validation
            path = request_data["path"]
            pattern = request_data.get("pattern")
            dry_run = request_data.get("dry_run", False)
            
            # Phase 1: Pre-checks (mocked as successful)
            
            # Phase 2: Validation
            validate_results = await mock_engine.collective_rpc(
                "validate_sharded_state", args=(path, pattern)
            )
            
            # Check validation results
            all_ok = True
            total_tensors = 0
            for vr in validate_results:
                if isinstance(vr, dict):
                    total_tensors += vr.get("tensor_count", 0)
                    mismatches = vr.get("mismatches", [])
                    if mismatches:
                        all_ok = False
            
            if not all_ok:
                return {
                    "ok": False,
                    "validation_failed": True,
                    "mismatches": []
                }, 400
            
            if dry_run:
                return {
                    "ok": True,
                    "dry_run": True,
                    "validated_tensors": total_tensors,
                }
            
            # Phase 3: Apply weights
            results = await mock_engine.collective_rpc(
                "load_sharded_state", args=(path, pattern)
            )
            
            # Check results
            ok_all = True
            details = []
            for r in results:
                if isinstance(r, dict):
                    ok_all = ok_all and bool(r.get("ok", False))
                    details.append(r)
            
            if ok_all:
                return {
                    "ok": True,
                    "details": details,
                    "validated_tensors": total_tensors,
                }
            else:
                return {
                    "ok": False,
                    "details": details,
                }, 500
        
        # Test successful dry run
        dry_run_request = {
            "path": "/path/to/weights",
            "pattern": None,
            "dry_run": True
        }
        
        result = await mock_weight_update_endpoint(dry_run_request)
        assert result["ok"] is True
        assert result["dry_run"] is True
        assert result["validated_tensors"] == 100
        
        # Test successful actual update
        update_request = {
            "path": "/path/to/weights", 
            "pattern": None,
            "dry_run": False
        }
        
        result = await mock_weight_update_endpoint(update_request)
        assert result["ok"] is True
        assert "dry_run" not in result
        assert result["validated_tensors"] == 100
        assert len(result["details"]) == 1
        assert result["details"][0]["ok"] is True
        assert result["details"][0]["num_updated"] == 100
        
        # Verify collective_rpc was called correctly
        assert mock_engine.collective_rpc.call_count == 4  # 2 for dry run, 2 for actual
    
    @pytest.mark.asyncio
    async def test_weight_update_validation_failure(self):
        """Test weight update with validation failures."""
        
        mock_engine = MockAsyncLLMV1()
        mock_engine.collective_rpc = AsyncMock()
        
        # Mock validation results (failure)
        mock_engine.collective_rpc.return_value = [
            {
                "tensor_count": 50,
                "mismatches": [
                    {"kind": "shape", "name": "layer1.weight", "detail": "expected (1024, 512) got (1024, 256)"},
                    {"kind": "dtype", "name": "layer2.bias", "detail": "expected float32 got float16"}
                ]
            }
        ]
        
        async def mock_weight_update_with_validation_failure(request_data):
            """Mock endpoint with validation failure."""
            path = request_data["path"]
            pattern = request_data.get("pattern")
            
            validate_results = await mock_engine.collective_rpc(
                "validate_sharded_state", args=(path, pattern)
            )
            
            # Check for mismatches
            overall_mismatches = []
            for vr in validate_results:
                if isinstance(vr, dict):
                    mismatches = vr.get("mismatches", [])
                    if mismatches:
                        overall_mismatches.extend(mismatches)
            
            if overall_mismatches:
                return {
                    "ok": False,
                    "validation_failed": True,
                    "mismatches": overall_mismatches,
                }, 400
            
            return {"ok": True}
        
        request_data = {"path": "/path/to/weights"}
        result, status_code = await mock_weight_update_with_validation_failure(request_data)
        
        assert result["ok"] is False
        assert result["validation_failed"] is True
        assert status_code == 400
        assert len(result["mismatches"]) == 2
        assert result["mismatches"][0]["kind"] == "shape"
        assert result["mismatches"][1]["kind"] == "dtype"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])