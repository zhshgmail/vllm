# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Weight Update API endpoints for vLLM

This module provides API endpoints for weight updates, including NCCL-based
weight updates and request interruption capabilities for async training.
"""

import asyncio
import glob
import os
import time
from typing import Any, Dict, List, Optional
from http import HTTPStatus
from pydantic import BaseModel, Field

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.logger import init_logger
from vllm.entrypoints.openai.api_server import engine_client

logger = init_logger(__name__)


# Request/Response models
class NCCLGroupConfig(BaseModel):
    """Configuration for NCCL process group."""
    master_address: str = Field(description="Master process address for NCCL communication")
    master_port: int = Field(description="Master process port for NCCL communication")
    rank_offset: int = Field(default=0, description="Rank offset for training process group")
    world_size: int = Field(description="Total number of processes in training group")


class InitNCCLGroupRequest(BaseModel):
    """Request to initialize NCCL group."""
    nccl_config: NCCLGroupConfig
    timeout: float = Field(default=30.0, description="Initialization timeout in seconds")


class DiskWeightUpdateRequest(BaseModel):
    """Request for disk-based weight update."""
    path: str = Field(description="Local directory containing sharded safetensors files")
    pattern: Optional[str] = Field(default=None, description="Override filename pattern (e.g., 'model-{rank}-{part}.safetensors')")
    pause: bool = Field(default=True, description="Quiesce scheduler before swap (v1 only)")
    interrupt: bool = Field(default=True, description="Abort all ongoing requests immediately")
    dry_run: bool = Field(default=False, description="Validate only; do not mutate weights")


class NCCLWeightUpdateRequest(BaseModel):
    """Request for NCCL-based weight update."""
    allow_interrupt: bool = Field(default=True, description="Whether to interrupt running requests")
    timeout: float = Field(default=60.0, description="Weight update timeout in seconds")
    validate_consistency: bool = Field(default=False, description="Whether to validate weight consistency")
    sample_param_names: Optional[List[str]] = Field(default=None, description="Parameters to check for consistency")


class InterruptedRequestInfo(BaseModel):
    """Information about interrupted request."""
    request_id: str
    prompt: str
    generated_tokens: List[int]
    num_generated_tokens: int
    interrupt_time: float
    sampling_params: Optional[Dict[str, Any]] = None


class DiskWeightUpdateResponse(BaseModel):
    """Response from disk-based weight update."""
    ok: bool
    dry_run: Optional[bool] = None
    validation_failed: Optional[bool] = None
    duration_sec: Optional[float] = None
    validated_tensors: Optional[int] = None
    num_paused_requests: Optional[int] = None
    num_interrupted_requests: Optional[int] = None
    details: Optional[List[Dict[str, Any]]] = None
    mismatches: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class NCCLWeightUpdateResponse(BaseModel):
    """Response from NCCL weight update."""
    status: str
    updated_parameters: Optional[int] = None
    total_parameters: Optional[int] = None
    interrupted_requests: List[InterruptedRequestInfo] = []
    update_duration: Optional[float] = None
    update_count: Optional[int] = None
    worker_rank: Optional[int] = None
    timestamp: Optional[float] = None
    message: Optional[str] = None
    consistency_valid: Optional[bool] = None


class WeightUpdateStatsResponse(BaseModel):
    """Response with weight update statistics."""
    weight_update_count: int
    last_weight_update_time: float
    weight_update_in_progress: bool
    nccl_group_initialized: bool
    interrupted_requests_count: int
    worker_device: Optional[str] = None
    all_workers_stats: List[Dict[str, Any]] = []


# Utility functions
def validate_weight_update_support(engine) -> bool:
    """Validate that engine supports weight update extensions."""
    if not hasattr(engine, 'collective_rpc'):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Engine doesn't support collective RPC operations required for weight updates"
        )
    
    # Check if WeightUpdateExtension is loaded
    try:
        stats = engine.collective_rpc("get_weight_update_stats")
        return True
    except Exception:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Weight update extension not loaded. Set worker_extension_cls='vllm.worker.weight_update_extension.WeightUpdateExtension'"
        )


# API endpoints
async def init_nccl_group(request: InitNCCLGroupRequest, raw_request: Request) -> JSONResponse:
    """
    Initialize NCCL process group for distributed training communication.
    
    This must be called before any NCCL weight updates can be performed.
    The training process should be the master (rank 0) in this group.
    """
    engine = engine_client(raw_request)
    validate_weight_update_support(engine)
    
    logger.info(f"Initializing NCCL group: {request.nccl_config}")
    
    try:
        # Initialize NCCL group on all workers
        results = await engine.collective_rpc(
            "init_nccl_weight_source",
            args=(
                request.nccl_config.master_address,
                request.nccl_config.master_port,
                request.nccl_config.rank_offset,
                request.nccl_config.world_size
            ),
            timeout=request.timeout
        )
        
        # Check if all workers initialized successfully
        success_count = sum(1 for result in results if result)
        total_workers = len(results)
        
        if success_count != total_workers:
            logger.error(f"NCCL group initialization failed on {total_workers - success_count} workers")
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={
                    "status": "partial_failure",
                    "successful_workers": success_count,
                    "total_workers": total_workers,
                    "message": f"Failed to initialize on {total_workers - success_count} workers"
                }
            )
        
        logger.info(f"Successfully initialized NCCL group on {total_workers} workers")
        return JSONResponse(content={
            "status": "success",
            "initialized_workers": total_workers,
            "nccl_config": request.nccl_config.model_dump()
        })
        
    except Exception as e:
        logger.error(f"NCCL group initialization failed: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize NCCL group: {str(e)}"
        )


async def update_weights_nccl(request: NCCLWeightUpdateRequest, raw_request: Request) -> NCCLWeightUpdateResponse:
    """
    Update model weights via NCCL broadcast from training process.
    
    This endpoint:
    1. Optionally interrupts running requests for consistency
    2. Flushes KV caches to ensure consistency
    3. Broadcasts new weights from training process (rank 0)
    4. Validates weight consistency across workers (optional)
    """
    engine = engine_client(raw_request)
    validate_weight_update_support(engine)
    
    logger.info(f"Starting NCCL weight update: interrupt={request.allow_interrupt}")
    
    try:
        # Check if any worker has update in progress
        stats = await engine.collective_rpc("get_weight_update_stats")
        if any(stat.get("weight_update_in_progress", False) for stat in stats):
            raise HTTPException(
                status_code=HTTPStatus.CONFLICT,
                detail="Weight update already in progress on one or more workers"
            )
        
        # Perform weight update on all workers
        results = await engine.collective_rpc(
            "load_weights_nccl",
            args=(),
            kwargs={
                "timeout": request.timeout
            },
            timeout=request.timeout + 10  # Add buffer for RPC overhead
        )
        
        # Aggregate results  
        successful_workers = [r for r in results if r.get("ok", False)]
        failed_workers = [r for r in results if not r.get("ok", False)]
        
        if not successful_workers:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=f"Weight update failed on all workers. Errors: {[r.get('message', 'Unknown') for r in failed_workers]}"
            )
        
        # Aggregate interrupted requests from all workers
        all_interrupted_requests = []
        for result in successful_workers:
            if "interrupted_requests" in result:
                all_interrupted_requests.extend(result["interrupted_requests"])
        
        # Validate consistency if requested
        consistency_valid = None
        if request.validate_consistency:
            try:
                consistency_results = await engine.collective_rpc(
                    "validate_weights_nccl"
                )
                consistency_valid = all(r.get("ok", False) for r in consistency_results)
                if not consistency_valid:
                    logger.warning("Weight consistency validation failed on some workers")
            except Exception as e:
                logger.error(f"Weight consistency validation error: {e}")
                consistency_valid = False
        
        # Create response from first successful result
        primary_result = successful_workers[0]
        
        response = NCCLWeightUpdateResponse(
            status="success" if len(failed_workers) == 0 else "partial_success",
            updated_parameters=primary_result.get("updated_params"),
            total_parameters=None,  # Not tracked in unified response
            interrupted_requests=[],  # Handled at server level
            update_duration=primary_result.get("duration"),
            update_count=None,  # Available in stats
            worker_rank=primary_result.get("rank"),
            timestamp=time.time(),
            consistency_valid=consistency_valid
        )
        
        if failed_workers:
            response.message = f"Weight update failed on {len(failed_workers)} workers"
            logger.warning(response.message)
        
        logger.info(f"NCCL weight update completed: {len(successful_workers)} successful, "
                   f"{len(failed_workers)} failed, {len(all_interrupted_requests)} requests interrupted")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NCCL weight update failed: {e}")
        return NCCLWeightUpdateResponse(
            status="error",
            message=str(e),
            updated_parameters=0,
            update_duration=0.0,
            worker_rank=None,
            timestamp=time.time()
        )


async def get_weight_update_stats(raw_request: Request) -> WeightUpdateStatsResponse:
    """Get weight update statistics from all workers."""
    engine = engine_client(raw_request)
    validate_weight_update_support(engine)
    
    try:
        stats = await engine.collective_rpc("get_weight_update_stats")
        
        # Aggregate stats from all workers
        primary_stats = stats[0] if stats else {}
        
        return WeightUpdateStatsResponse(
            weight_update_count=primary_stats.get("total_updates", 0),
            last_weight_update_time=primary_stats.get("last_update_time", 0.0),
            weight_update_in_progress=any(s.get("weight_update_in_progress", False) for s in stats),
            nccl_group_initialized=all(s.get("nccl_initialized", False) for s in stats),
            interrupted_requests_count=primary_stats.get("interrupted_requests_count", 0),
            worker_device=primary_stats.get("worker_device"),
            all_workers_stats=stats
        )
        
    except Exception as e:
        logger.error(f"Failed to get weight update stats: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Failed to get weight update stats: {str(e)}"
        )


async def cleanup_nccl_group(raw_request: Request) -> JSONResponse:
    """Cleanup NCCL group resources."""
    engine = engine_client(raw_request)
    validate_weight_update_support(engine)
    
    try:
        await engine.collective_rpc("cleanup_weight_sources")
        
        logger.info("Successfully cleaned up NCCL group")
        return JSONResponse(content={"status": "success", "message": "NCCL group cleaned up"})
        
    except Exception as e:
        logger.error(f"Failed to cleanup NCCL group: {e}")
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": f"Failed to cleanup: {str(e)}"}
        )


async def update_weights_from_disk(request: DiskWeightUpdateRequest, raw_request: Request) -> DiskWeightUpdateResponse:
    """
    Update model weights in-place from a local HF-style sharded checkpoint.

    This endpoint:
    1. Validates the checkpoint directory and files
    2. Optionally pauses/interrupts running requests
    3. Validates weights before loading (collective across all workers)
    4. Loads weights from disk if validation passes
    """
    engine = engine_client(raw_request)
    
    # Preflight: verify local dir and presence of at least one shard file.
    try:
        if not os.path.isdir(request.path):
            logger.error("Update weights failed: path is not a directory: %s", request.path)
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"Not a directory: {request.path}"
            )
        
        # Use default sharded pattern structure if none provided
        pattern = request.pattern
        if pattern is None:
            from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader
            pattern = ShardedStateLoader.DEFAULT_PATTERN
        
        wildcard = pattern.format(rank="*", part="*")
        shard_candidates = glob.glob(os.path.join(request.path, wildcard))
        if len(shard_candidates) == 0:
            logger.error(
                "Update weights failed: no shard files matching pattern %s in %s",
                pattern, request.path)
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=(
                    "Unsupported or missing format: expected sharded safetensors files "
                    f"matching pattern '{pattern}' in directory {request.path}"
                ))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Update weights preflight error for path %s", request.path)
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))

    # Check engine compatibility
    is_v1 = (getattr(engine, "vllm_config", None) is not None and 
             getattr(engine.vllm_config.model_config, "runner_type", None) is not None and 
             hasattr(engine, "collective_rpc") and 
             "v1" in type(engine).__module__)
    if not is_v1:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="/weights/update-from-disk supported only in V1 mode for now"
        )

    # Check for concurrent weight updates
    app = raw_request.app
    if getattr(app.state, "weight_update_in_progress", False):
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail="Weight update already in progress"
        )

    setattr(app.state, "weight_update_in_progress", True)
    logger.info(
        "Weight update start path=%s pattern=%s dry_run=%s pause=%s interrupt=%s", 
        request.path, pattern, request.dry_run, request.pause, request.interrupt
    )

    start = time.time()
    try:
        # Phase 1 (optional): pause & drain.
        num_paused_requests = 0
        num_interrupted_requests = 0
        
        if request.interrupt:
            # Immediately abort all active requests (returns partial outputs to clients).
            try:
                if hasattr(engine, "abort_all_active"):
                    num_interrupted_requests = await engine.abort_all_active()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                logger.exception("Failed aborting active requests prior to weight update")
                
        if request.pause:
            # Best-effort: ask engine to report zero running requests by polling stats logger snapshot.
            # (Simplified: rely on absence of active server_load_metrics usage.)
            # TODO: integrate with formal scheduler pause API when available.
            wait_loops = 0
            while getattr(app.state, 'server_load_metrics', 0) > 0 and wait_loops < 100:
                await asyncio.sleep(0.05)
                wait_loops += 1
            num_paused_requests = getattr(app.state, 'server_load_metrics', 0)

        # Phase 2: validation (collective per-rank) before mutation.
        validate_results = await engine.collective_rpc("validate_sharded_state", args=(request.path, pattern))
        overall_mismatches = []
        total_tensors = 0
        all_ok = True
        
        for vr in validate_results:
            if isinstance(vr, dict):
                total_tensors += vr.get("tensor_count", 0)
                mismatches = vr.get("mismatches", [])
                if mismatches:
                    all_ok = False
                    overall_mismatches.extend([{**m, "rank": vr.get("rank")} for m in mismatches])
            else:
                all_ok = False
                overall_mismatches.append({"kind": "error", "name": "*", "detail": repr(vr)})

        if not all_ok:
            logger.error("Weight update validation failed; aborting. mismatches=%s", overall_mismatches[:5])
            return DiskWeightUpdateResponse(
                ok=False,
                dry_run=request.dry_run,
                validation_failed=True,
                mismatches=overall_mismatches,
            )

        if request.dry_run:
            duration = time.time() - start
            logger.info("Dry-run weight validation succeeded in %.2fs", duration)
            return DiskWeightUpdateResponse(
                ok=True,
                dry_run=True,
                duration_sec=round(duration, 3),
                validated_tensors=total_tensors,
                num_paused_requests=num_paused_requests,
                num_interrupted_requests=num_interrupted_requests,
            )

        # Phase 3: apply (collective) now that validation passed.
        results = await engine.collective_rpc("load_sharded_state", args=(request.path, pattern))

        ok_all = True
        details: list[dict[str, Any]] = []
        if isinstance(results, (list, tuple)):
            for r in results:
                if isinstance(r, dict):
                    ok_all = ok_all and bool(r.get("ok", False))
                    details.append(r)
                else:
                    ok_all = False
                    details.append({"ok": False, "error": repr(r)})
        else:
            ok_all = False
            details.append({"ok": False, "error": "Unexpected result"})

        duration = time.time() - start
        if ok_all:
            logger.info("Weight update successful in %.2fs tensors=%s", duration, total_tensors)
            return DiskWeightUpdateResponse(
                ok=True,
                duration_sec=round(duration, 3),
                details=details,
                validated_tensors=total_tensors,
                num_paused_requests=num_paused_requests,
                num_interrupted_requests=num_interrupted_requests,
            )
        else:
            logger.error("Weight update failed: %s", details)
            return DiskWeightUpdateResponse(
                ok=False,
                duration_sec=round(duration, 3),
                details=details,
            )
            
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Weight update failed with exception")
        return DiskWeightUpdateResponse(
            ok=False,
            error=str(e),
        )
    finally:
        # Re-enable health checks.
        setattr(app.state, "weight_update_in_progress", False)