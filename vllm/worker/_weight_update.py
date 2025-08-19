"""Utilities for hot weight updates (shared by V0 and V1 workers).

This consolidates the runtime sharded weight loading logic used by
`Worker.load_sharded_state` in both legacy (v0) and v1 worker stacks. It
streams tensors from safetensors shards and applies them via the model's
`load_weights` method one tensor at a time to preserve parameter storage.
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Dict


def stream_apply_sharded_state(model, path: str, pattern: Optional[str] = None) -> int:
    """Stream sharded state tensors into an existing model.

    Minimal logic: allow checkpoints that still store split q/k/v and gate/up
    shards while the runtime model exposes fused qkv_proj / gate_up_proj.
    We no longer concatenate; we just forward the original shard names so the
    model's own load_weights stacking path handles them. Split biases are
    skipped if the fused bias param does not exist. Certain nested path forms
    (".gate.gate_proj") are normalized to prevent doubled prefixes.
    Returns number of tensor loads invoked.
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

    # Collect fused param shapes (include fused biases if model defines them) for quick checks.
    fused_shapes: Dict[str, Tuple[int, ...]] = {}
    for n, p in model.named_parameters(recurse=True):  # type: ignore[attr-defined]
        if any(n.endswith(suf) for suf in ("qkv_proj.weight", "qkv_proj.bias", "gate_up_proj.weight", "gate_up_proj.bias")):
            fused_shapes[n] = tuple(p.shape)  # type: ignore[attr-defined]

    updated = 0

    def _normalize(key: str) -> str:
        # Collapse nested gate paths to avoid duplicate gate_ in fused names downstream.
        if '.gate.gate_proj' in key:
            key = key.replace('.gate.gate_proj', '.gate_proj')
        if '.gate.up_proj' in key:
            key = key.replace('.gate.up_proj', '.up_proj')
        if '.gate.gate_up_proj' in key:  # defensive
            key = key.replace('.gate.gate_up_proj', '.gate_up_proj')
        return key

    import re
    qkv_pat = re.compile(r"^(.*)\.(q|k|v)_proj\.(weight|bias)$")
    gate_pat = re.compile(r"^(.*)\.(gate|up)_proj\.(weight|bias)$")

    for key, tensor in loader.iterate_over_files(filepaths):
        key = _normalize(key)
        # Direct hit (already fused or unrelated param not a split weight/bias)
        if key in fused_shapes or (not key.endswith("_proj.weight") and not key.endswith("_proj.bias")):
            model.load_weights(weights=[(key, tensor)])  # type: ignore[attr-defined]
            updated += 1
            continue
        m = qkv_pat.match(key)
        if m:
            prefix, which, kind = m.group(1), m.group(2), m.group(3)
            fused_name = f"{prefix}.qkv_proj.{kind}"
            if fused_name in fused_shapes:
                # Model exposes fused qkv_proj; let its internal loader stack split shards.
                if kind == "bias" and fused_name not in fused_shapes:
                    # No fused bias param present.
                    continue
                model.load_weights(weights=[(key, tensor)])  # type: ignore[attr-defined]
                updated += 1
                continue
            elif kind == "bias":
                # No fused bias expected; skip split bias.
                continue
        m2 = gate_pat.match(key)
        if m2:
            prefix, which, kind = m2.group(1), m2.group(2), m2.group(3)
            norm_prefix = prefix[:-5] if prefix.endswith('.gate') else prefix
            fused_name = f"{norm_prefix}.gate_up_proj.{kind}"
            if fused_name in fused_shapes:
                # Let model loader stack gate/up shards; don't pre-concatenate to avoid substring replacement.
                if kind == "bias" and fused_name not in fused_shapes:
                    continue
                model.load_weights(weights=[(key, tensor)])  # type: ignore[attr-defined]
                updated += 1
                continue
            elif kind == "bias":
                continue
    # Fallback (unfused architecture or unexpected name): always load as-is.
        model.load_weights(weights=[(key, tensor)])  # type: ignore[attr-defined]
        updated += 1
    return updated


def validate_sharded_state(model, path: str, pattern: Optional[str] = None) -> Tuple[int, List[Dict[str, str]]]:
    """Validate shard tensors (accepting split q/k/v & gate/up for fused models)."""
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
    import re
    import torch
    qkv_pat = re.compile(r"^(.*)\.(q|k|v)_proj\.(weight|bias)$")
    gate_pat = re.compile(r"^(.*)\.(gate|up)_proj\.(weight|bias)$")
    # prefix -> kind(weight/bias) -> part -> tensor
    qkv_parts: Dict[str, Dict[str, Dict[str, object]]] = {}
    gate_parts: Dict[str, Dict[str, Dict[str, object]]] = {}

    def _normalize(key: str) -> str:
        if '.gate.gate_proj' in key:
            key = key.replace('.gate.gate_proj', '.gate_proj')
        if '.gate.up_proj' in key:
            key = key.replace('.gate.up_proj', '.up_proj')
        if '.gate.gate_up_proj' in key:
            key = key.replace('.gate.gate_up_proj', '.gate_up_proj')
        return key

    for key, tensor in loader.iterate_over_files(filepaths):
        key = _normalize(key)
        count += 1
        m = qkv_pat.match(key)
        if m:
            prefix, which, kind = m.group(1), m.group(2), m.group(3)
            fused_name = f"{prefix}.qkv_proj.{kind}"
            if fused_name in expected:
                bucket = qkv_parts.setdefault(prefix, {}).setdefault(kind, {})
                bucket[which] = tensor
                continue
            elif kind == "bias":
                # Ignore split bias for models without fused bias
                continue
        m2 = gate_pat.match(key)
        if m2:
            prefix, which, kind = m2.group(1), m2.group(2), m2.group(3)
            norm_prefix = prefix[:-5] if prefix.endswith('.gate') else prefix
            fused_name = f"{norm_prefix}.gate_up_proj.{kind}"
            if fused_name in expected:
                bucket = gate_parts.setdefault(prefix, {}).setdefault(kind, {})
                bucket[which] = tensor
                continue
            elif kind == "bias":
                continue
        # Normal param path
        seen.add(key)
        if key not in expected:
            mismatches.append({"kind": "unexpected", "name": key, "detail": "not in model"})
            continue
        exp_shape, exp_dtype = expected[key]
        if tuple(tensor.shape) != exp_shape:
            mismatches.append({"kind": "shape", "name": key, "detail": f"expected {exp_shape} got {tuple(tensor.shape)}"})
        if str(tensor.dtype) != exp_dtype:
            mismatches.append({"kind": "dtype", "name": key, "detail": f"expected {exp_dtype} got {tensor.dtype}"})

    # Synthesize and check fused groups
    def validate_qkv(prefix: str, kind: str, parts: Dict[str, object]):
        fused_name = f"{prefix}.qkv_proj.{kind}"
        needed = ("q", "k", "v")
        if not all(x in parts for x in needed):
            mismatches.append({"kind": "incomplete", "name": fused_name, "detail": f"have {sorted(parts.keys())}"})
            return
        exp_shape, exp_dtype = expected.get(fused_name, ((), "?"))
        q, k, v = parts["q"], parts["k"], parts["v"]
        ok = False
        if kind == "bias":
            try:
                cand = torch.cat([q, k, v], dim=0)  # type: ignore[arg-type]
                if tuple(cand.shape) == exp_shape and str(cand.dtype) == exp_dtype:
                    ok = True
            except Exception:
                pass
        else:
            for dim in (0, 1):
                try:
                    cand = torch.cat([q, k, v], dim=dim)  # type: ignore[arg-type]
                except Exception:
                    continue
                if tuple(cand.shape) == exp_shape and str(cand.dtype) == exp_dtype:  # type: ignore[attr-defined]
                    ok = True
                    break
        if not ok:
            mismatches.append({"kind": "shape", "name": fused_name, "detail": f"split qkv {kind} mismatch"})
        else:
            seen.add(fused_name)

    def validate_gate(prefix: str, kind: str, parts: Dict[str, object]):
        norm_prefix = prefix[:-5] if prefix.endswith('.gate') else prefix
        fused_name = f"{norm_prefix}.gate_up_proj.{kind}"
        needed = ("gate", "up")
        if not all(x in parts for x in needed):
            mismatches.append({"kind": "incomplete", "name": fused_name, "detail": f"have {sorted(parts.keys())}"})
            return
        exp_shape, exp_dtype = expected.get(fused_name, ((), "?"))
        gate, up = parts["gate"], parts["up"]
        ok = False
        if kind == "bias":
            try:
                cand = torch.cat([gate, up], dim=0)  # type: ignore[arg-type]
                if tuple(cand.shape) == exp_shape and str(cand.dtype) == exp_dtype:
                    ok = True
            except Exception:
                pass
        else:
            for dim in (0, 1):
                try:
                    cand = torch.cat([gate, up], dim=dim)  # type: ignore[arg-type]
                except Exception:
                    continue
                if tuple(cand.shape) == exp_shape and str(cand.dtype) == exp_dtype:  # type: ignore[attr-defined]
                    ok = True
                    break
        if not ok:
            mismatches.append({"kind": "shape", "name": fused_name, "detail": f"split gate_up {kind} mismatch"})
        else:
            seen.add(fused_name)

    for prefix, kinds in qkv_parts.items():
        for kind, parts in kinds.items():
            validate_qkv(prefix, kind, parts)
    for prefix, kinds in gate_parts.items():
        for kind, parts in kinds.items():
            validate_gate(prefix, kind, parts)

    # Missing parameters (only flag those that look like weights: skip buffers?)
    # Ignore runtime buffers not expected to appear in sharded state (e.g., rotary cache)
    ignore_missing_substrings = ["rotary_emb.cos_sin_cache"]
    missing = [n for n in expected.keys() if n not in seen and not any(s in n for s in ignore_missing_substrings)]
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
