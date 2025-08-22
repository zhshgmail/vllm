"""Utilities for hot weight updates (shared by V0 and V1 workers).

This consolidates the runtime sharded weight loading logic used by
`Worker.load_sharded_state` in both legacy (v0) and v1 worker stacks. It
streams tensors from safetensors shards and applies them via the model's
`load_weights` method one tensor at a time to preserve parameter storage.
"""
from __future__ import annotations
from typing import Optional, Iterable, Tuple, List, Dict

def stream_apply_sharded_state(model, path: str, pattern: Optional[str] = None) -> int:
    """Stream sharded state tensors into an existing model in-place.

    Args:
        model: The model object exposing `load_weights(weights=[(name, tensor)])`.
        path: Directory (or S3 URI) containing safetensors shards named like
              model-rank-{rank}-part-{part}.safetensors.
        pattern: Optional override of the filename pattern.

    Returns:
        int: Number of tensors updated.

    Raises:
        ValueError: If no shard files are found for this rank.
        Other exceptions propagate.
    """
    import glob as _glob
    import os
    from vllm.config import LoadConfig
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.model_loader.sharded_state_loader import (
        ShardedStateLoader,
    )
    from vllm.transformers_utils.s3_utils import glob as s3_glob
    from vllm.transformers_utils.utils import is_s3

    load_cfg = LoadConfig(load_format="sharded_state", model_loader_extra_config={})
    loader = ShardedStateLoader(load_cfg)
    if pattern is not None:
        loader.pattern = pattern

    rank = get_tensor_model_parallel_rank()
    local_model_path = path
    file_glob = os.path.join(
        local_model_path,
        loader.pattern.format(rank=rank, part="*"),
    )
    if is_s3(local_model_path):
        file_pattern = f"*{loader.pattern.format(rank=rank, part=' * ')}"
        filepaths = s3_glob(path=local_model_path, allow_pattern=[file_pattern])
    else:
        filepaths = _glob.glob(file_glob)
    if not filepaths:
        raise ValueError(f"No shards found for rank {rank} with pattern {file_glob}")

    updated = 0
    failed_params = []
    
    try:
        for key, tensor in loader.iterate_over_files(filepaths):
            try:
                # Ensure tensor is on the correct device and contiguous
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                
                # Load weights one at a time for better error isolation
                model.load_weights(weights=[(key, tensor)])  # type: ignore[attr-defined]
                updated += 1
                
                # Periodic memory cleanup for large models
                if updated % 100 == 0:
                    import gc
                    gc.collect()
                    
            except Exception as e:
                failed_params.append((key, str(e)))
                # Continue loading other parameters even if one fails
                continue
                
    except Exception as e:
        # Critical failure in iteration
        raise RuntimeError(f"Failed to iterate over weight files: {str(e)}") from e
    
    if failed_params:
        # Log failed parameters but don't fail the entire operation
        # unless too many parameters failed
        failure_rate = len(failed_params) / max(1, updated + len(failed_params))
        if failure_rate > 0.1:  # More than 10% failed
            raise RuntimeError(
                f"Too many parameter loading failures ({len(failed_params)} failed, "
                f"{updated} succeeded). First few failures: {failed_params[:5]}"
            )
        else:
            # Log warnings for failed parameters
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Some parameters failed to load ({len(failed_params)} failed, "
                f"{updated} succeeded): {failed_params[:3]}"
            )
    
    if updated == 0:
        raise ValueError("No parameters were successfully loaded from sharded state")
        
    return updated


def validate_sharded_state(model, path: str, pattern: Optional[str] = None) -> Tuple[int, List[Dict[str, str]]]:
    """Validate that a sharded state at `path` matches current model shapes/dtypes.

    Iterates all tensors in the shard set for this TP rank and checks that
    each exists in the current model parameters/buffers with identical shape
    and dtype. Also records any missing parameters that were expected but not
    seen in the shard set.

    Returns:
        (tensor_count, mismatches)
        tensor_count: number of tensors encountered in shards.
        mismatches: list of mismatch dicts with keys: kind, name, detail.
    """
    import glob as _glob
    import os
    from vllm.config import LoadConfig
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.model_loader.sharded_state_loader import (
        ShardedStateLoader,
    )
    from vllm.transformers_utils.s3_utils import glob as s3_glob
    from vllm.transformers_utils.utils import is_s3

    # Build expected map
    expected: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    for n, p in model.named_parameters(recurse=True):  # type: ignore[attr-defined]
        expected[n] = (tuple(p.shape), str(p.dtype))
    for n, b in model.named_buffers(recurse=True):  # type: ignore[attr-defined]
        expected[n] = (tuple(b.shape), str(b.dtype))

    load_cfg = LoadConfig(load_format="sharded_state", model_loader_extra_config={})
    loader = ShardedStateLoader(load_cfg)
    if pattern is not None:
        loader.pattern = pattern

    rank = get_tensor_model_parallel_rank()
    local_model_path = path
    file_glob = os.path.join(
        local_model_path,
        loader.pattern.format(rank=rank, part="*"),
    )
    if is_s3(local_model_path):
        file_pattern = f"*{loader.pattern.format(rank=rank, part=' * ')}"
        filepaths = s3_glob(path=local_model_path, allow_pattern=[file_pattern])
    else:
        filepaths = _glob.glob(file_glob)
    if not filepaths:
        raise ValueError(f"No shards found for rank {rank} with pattern {file_glob}")

    seen: set[str] = set()
    mismatches: List[Dict[str, str]] = []
    count = 0
    for key, tensor in loader.iterate_over_files(filepaths):
        count += 1
        seen.add(key)
        if key not in expected:
            mismatches.append({
                "kind": "unexpected",
                "name": key,
                "detail": "tensor not present in current model",
            })
            continue
        exp_shape, exp_dtype = expected[key]
        if tuple(tensor.shape) != exp_shape:
            mismatches.append({
                "kind": "shape",
                "name": key,
                "detail": f"expected {exp_shape} got {tuple(tensor.shape)}",
            })
        if str(tensor.dtype) != exp_dtype:
            mismatches.append({
                "kind": "dtype",
                "name": key,
                "detail": f"expected {exp_dtype} got {tensor.dtype}",
            })

    # Missing parameters (only flag those that look like weights: skip buffers?)
    missing = [n for n in expected.keys() if n not in seen]
    # To avoid huge payloads, truncate missing list if large.
    MAX_MISSING_REPORT = 50
    if missing:
        truncated = missing[:MAX_MISSING_REPORT]
        mismatches.append({
            "kind": "missing",
            "name": "*multiple*" if len(truncated) > 1 else truncated[0],
            "detail": (
                f"{len(missing)} tensors missing from shard set; first {len(truncated)}: "
                + ",".join(truncated)
            ),
        })

    return count, mismatches
