"""Utilities for hot weight updates (shared by V0 and V1 workers).

This consolidates the runtime sharded weight loading logic used by
`Worker.load_sharded_state` in both legacy (v0) and v1 worker stacks. It
uses the existing DefaultModelLoader infrastructure to ensure consistency
with regular model loading and proper handling of all edge cases.
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Dict

def stream_apply_sharded_state(model, path: str, pattern: Optional[str] = None) -> int:
    """Load sharded state tensors into an existing model using DefaultModelLoader.

    Uses the existing DefaultModelLoader infrastructure to ensure consistency
    with regular model loading and proper handling of all edge cases including
    device placement, file formats, memory efficiency, and stacked parameters.
    
    Args:
        model: The model instance to load weights into
        path: Path to model weights directory
        pattern: Optional pattern for weight files (e.g., "model.safetensors")
    
    Returns:
        Number of parameters successfully loaded
    """
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    import os
    
    # Determine load format based on pattern or available files
    load_format = "auto"
    if pattern == "model.safetensors":
        load_format = "safetensors"
    elif pattern and pattern.endswith(".bin"):
        load_format = "pt"

    # Create loader configuration
    load_config = LoadConfig(load_format=load_format)

    # Create the DefaultModelLoader
    loader = DefaultModelLoader(load_config)

    # Create a Source using the loader's Source class
    Source = loader.Source
    source = Source(
        model_or_path=path,
        revision=None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
        prefix=""
    )

    try:
        # Get the weights iterator from the loader
        weights_iterator = loader._get_weights_iterator(source)

        # Call model.load_weights once with all weights
        # This ensures proper handling of stacked parameters and state tracking
        loaded_params = model.load_weights(weights_iterator)

        # Return the number of loaded parameters
        if loaded_params is not None:
            return len(loaded_params)
        else:
            # Fallback: count weights manually if load_weights doesn't return loaded params
            count = sum(1 for _ in loader._get_weights_iterator(source))
            return count

    except Exception as e:
        # Enhanced error handling with more context
        raise RuntimeError(
            f"Failed to load weights from {path} with pattern {pattern}: {str(e)}"
        ) from e


def validate_sharded_state(model, path: str, pattern: Optional[str] = None) -> Tuple[int, List[Dict[str, str]]]:
    """Validate shard tensors using DefaultModelLoader for consistency.
    
    Args:
        model: The model instance to validate against
        path: Path to model weights directory
        pattern: Optional pattern for weight files
    
    Returns:
        Tuple of (count, mismatches) where count is number of tensors found
        and mismatches is a list of validation errors
    """
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    import torch
    
    # Build expected parameter map
    expected: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    for n, p in model.named_parameters(recurse=True):
        expected[n] = (tuple(p.shape), str(p.dtype))
    for n, b in model.named_buffers(recurse=True):
        expected[n] = (tuple(b.shape), str(b.dtype))
    
    # Create loader configuration
    load_format = "auto"
    if pattern == "model.safetensors":
        load_format = "safetensors"
    elif pattern and pattern.endswith(".bin"):
        load_format = "pt"
    
    load_config = LoadConfig(load_format=load_format)
    loader = DefaultModelLoader(load_config)
    
    # Create source using the loader's Source class
    Source = loader.Source
    source = Source(
        model_or_path=path,
        revision=None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
        prefix=""
    )
    
    try:
        weights_iterator = loader._get_weights_iterator(source)
        
        seen: set[str] = set()
        mismatches: List[Dict[str, str]] = []
        count = 0
        
        for name, tensor in weights_iterator:
            count += 1
            seen.add(name)
            
            if name not in expected:
                mismatches.append({
                    "kind": "unexpected",
                    "name": name,
                    "detail": "not in model"
                })
                continue
            
            exp_shape, exp_dtype = expected[name]
            if tuple(tensor.shape) != exp_shape:
                mismatches.append({
                    "kind": "shape",
                    "name": name,
                    "detail": f"expected {exp_shape} got {tuple(tensor.shape)}"
                })
            
            if str(tensor.dtype) != exp_dtype:
                mismatches.append({
                    "kind": "dtype",
                    "name": name,
                    "detail": f"expected {exp_dtype} got {tensor.dtype}"
                })
        
        # Check for missing parameters
        # Ignore runtime buffers not expected in checkpoints
        ignore_missing_substrings = ["rotary_emb.cos_sin_cache"]
        missing = [
            n for n in expected.keys() 
            if n not in seen and not any(s in n for s in ignore_missing_substrings)
        ]
        
        if missing:
            # Limit reporting to prevent huge payloads
            MAX_MISSING_REPORT = 50
            truncated = missing[:MAX_MISSING_REPORT]
            mismatches.append({
                "kind": "missing",
                "name": "*multiple*" if len(truncated) > 1 else truncated[0],
                "detail": (
                    f"{len(missing)} tensors missing from checkpoint; "
                    f"first {len(truncated)}: " + ",".join(truncated)
                ),
            })
        
        return count, mismatches
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to validate weights from {path} with pattern {pattern}: {str(e)}"
        ) from e
