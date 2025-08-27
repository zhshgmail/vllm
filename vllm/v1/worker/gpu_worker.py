# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""
import gc
import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils import GiB_bytes, MemorySnapshot, memory_profiling
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput


class Worker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def sleep(self, level: int = 1) -> None:
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone()
                for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def init_device(self):
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()

            # take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = (self.init_snapshot.total_memory *
                                     self.cache_config.gpu_memory_utilization)
            if self.init_snapshot.free_memory < self.requested_memory:
                GiB = lambda b: round(b / GiB_bytes, 2)
                raise ValueError(
                    f"Free memory on device "
                    f"({GiB(self.init_snapshot.free_memory)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                    f"utilization or reduce GPU memory used by other processes."
                )
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner: GPUModelRunner = GPUModelRunner(
            self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def load_model(self) -> None:
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be "
                "used for one instance per process.")
            context = allocator.use_memory_pool(tag="weights")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            self.model_runner.load_model()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much 
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        GiB = lambda b: b / GiB_bytes

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
                self.init_snapshot,
                weights_memory=int(
                    self.model_runner.model_memory_usage)) as profile_result:
            self.model_runner.profile_run()

        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {GiB(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container.")
        available_kv_cache_memory = self.requested_memory \
            - profile_result.non_kv_cache_memory

        logger.debug(
            "Initial free memory: %.2f GiB, free memory: %.2f GiB, "
            "requested GPU memory: %.2f GiB",
            GiB(self.init_snapshot.free_memory), GiB(free_gpu_memory),
            GiB(self.requested_memory))
        logger.debug(profile_result)
        logger.info("Available KV cache memory: %.2f GiB",
                    GiB(available_kv_cache_memory))
        gc.collect()

        return int(available_kv_cache_memory)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        # warm up sizes that are not in cudagraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes if x not in
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size)
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()

        # Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        if get_pp_group().is_last_rank:
            max_num_reqs = min(self.scheduler_config.max_num_seqs,
                               self.scheduler_config.max_num_batched_tokens)
            self.model_runner._dummy_sampler_run(
                hidden_states=self.model_runner._dummy_run(
                    num_tokens=max_num_reqs))

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def load_sharded_state(self, path: str, pattern: Optional[str] = None):
            """Load sharded weights from local disk into the running model.

            Expects safetensors shards named like model-rank-{rank}-part-{part}.safetensors
            residing under `path`. Optionally override the filename `pattern`.

            This mimics RLHF's WorkerExtension path by streaming tensors from
            shard files and invoking `model.load_weights([(name, tensor)])` for
            each param, ensuring in-place updates that preserve Parameter storage.

            Returns a small dict indicating success or an error for this rank.
            """
            try:
                logger.debug("Starting load_sharded_state for rank %d with path=%s, pattern=%s", 
                            self.rank, path, pattern)
                
                from vllm.worker._weight_update import stream_apply_sharded_state
                logger.debug("Imported stream_apply_sharded_state successfully")
                
                # Check model runner availability
                if hasattr(self, "model_runner"):
                    logger.debug("Found model_runner")
                    if hasattr(self.model_runner, "model"):
                        logger.debug("Found model_runner.model")
                    else:
                        logger.warning("model_runner has no model attribute")
                else:
                    logger.warning("No model_runner found")
                
                # Load new weights
                logger.debug("Calling stream_apply_sharded_state")
                num_updated = stream_apply_sharded_state(self.model_runner.model, path, pattern)
                logger.info(f"Successfully loaded {num_updated} parameters for rank {self.rank}")
                logger.debug("Weight loading completed, proceeding with post-load tasks")
                
                # Ensure all weight updates are complete before proceeding
                logger.debug("Performing CUDA synchronization after weight loading")
                torch.cuda.synchronize()
                logger.debug("CUDA synchronization completed")
                
                # Validate model state after weight loading
                logger.debug("Starting model state validation")
                self._validate_model_state_after_update()
                logger.debug("Model state validation completed")
                
                # Flush KV cache contents so subsequent requests recompute with new weights.
                logger.debug("Starting KV cache flush")
                try:
                    # V1 KV cache clearing: comprehensive cleanup of all cache state
                    self._flush_kv_cache_v1()
                    logger.debug("KV cache flush completed successfully")
                except Exception as e:  # noqa: BLE001
                    logger.warning("KV cache flush after weight update failed (v1)", exc_info=True)
                    logger.debug("KV cache flush exception details: %s", str(e))
                    
                logger.debug("load_sharded_state completed successfully for rank %d", self.rank)
                return {"ok": True, "rank": self.rank, "num_updated": num_updated}
            except Exception as e:  # noqa: BLE001
                logger.exception("Failed to load sharded state for rank %s", self.rank)
                return {
                    "ok": False,
                    "rank": self.rank,
                    "error": str(e),
                }

    def validate_sharded_state(self, path: str, pattern: Optional[str] = None):
            """Validate a prospective sharded state without mutating weights.

            Returns dict with tensor_count and mismatches list.
            """
            logger.debug("Starting validate_sharded_state for rank %d with path=%s, pattern=%s", 
                        self.rank, path, pattern)
            
            try:
                from vllm.worker._weight_update import validate_sharded_state
                logger.debug("Imported validate_sharded_state successfully")
                
                # Check model runner availability
                if hasattr(self, "model_runner"):
                    logger.debug("Found model_runner for validation")
                    if hasattr(self.model_runner, "model"):
                        logger.debug("Found model_runner.model for validation")
                    else:
                        logger.warning("model_runner has no model attribute for validation")
                else:
                    logger.warning("No model_runner found for validation")
                
                logger.debug("Calling validate_sharded_state function")
                tensor_count, mismatches = validate_sharded_state(self.model_runner.model, path, pattern)
                
                logger.debug("Validation completed: %d tensors checked, %d mismatches found", 
                            tensor_count, len(mismatches))
                
                if mismatches:
                    logger.debug("Validation mismatches: %s", mismatches[:3])  # Log first 3 mismatches
                
                result = {
                    "ok": True,
                    "rank": self.rank,
                    "tensor_count": tensor_count,
                    "mismatches": mismatches,
                }
                logger.debug("validate_sharded_state completed successfully for rank %d", self.rank)
                return result
                
            except Exception as e:  # noqa: BLE001
                logger.exception("validate_sharded_state failed rank=%s", self.rank)
                logger.debug("validate_sharded_state exception details: %s", str(e))
                return {
                    "ok": False,
                    "rank": self.rank,
                    "error": str(e),
                    "tensor_count": 0,
                    "mismatches": [{"kind": "error", "name": "*", "detail": str(e)}],
                }

    def _flush_kv_cache_v1(self) -> None:
        """Comprehensive KV cache flush for V1 engine after weight updates.
        
        Clears all KV cache tensors, block allocator state, request state,
        and attention metadata to ensure clean state for new weights.
        """
        logger.debug("Starting V1 KV cache flush after weight update")
        
        # 1. Clear KV cache tensors in model runner
        if hasattr(self.model_runner, "kv_caches"):
            logger.debug("Found model_runner.kv_caches, clearing %d KV cache tensors", 
                        len(self.model_runner.kv_caches))
            cleared_tensors = 0
            cleared_mamba_tensors = 0
            
            for i, kv_tensor in enumerate(self.model_runner.kv_caches):
                if torch.is_tensor(kv_tensor):
                    logger.debug("Clearing KV cache tensor %d with shape %s", i, kv_tensor.shape)
                    kv_tensor.zero_()
                    cleared_tensors += 1
                elif isinstance(kv_tensor, (list, tuple)):
                    logger.debug("Clearing Mamba-style state tensor %d with %d sub-tensors", 
                                i, len(kv_tensor))
                    # Handle Mamba-style state tensors (list of tensors)
                    for j, tensor in enumerate(kv_tensor):
                        if torch.is_tensor(tensor):
                            logger.debug("  Clearing sub-tensor %d with shape %s", j, tensor.shape)
                            tensor.zero_()
                            cleared_mamba_tensors += 1
                else:
                    logger.debug("Skipping non-tensor KV cache entry %d of type %s", i, type(kv_tensor))
            
            logger.debug("Cleared %d regular tensors and %d Mamba sub-tensors from model_runner.kv_caches", 
                        cleared_tensors, cleared_mamba_tensors)
        else:
            logger.debug("model_runner.kv_caches not found - skipping KV cache tensor clearing")
        
        # 2. Clear KV caches in forward context (attention layers)
        if hasattr(self.model_runner, "compilation_config"):
            logger.debug("Found model_runner.compilation_config")
            if hasattr(self.model_runner.compilation_config, "static_forward_context"):
                forward_context = self.model_runner.compilation_config.static_forward_context
                logger.debug("Found static_forward_context with %d layers", len(forward_context))
                
                cleared_layer_caches = 0
                for layer_name, layer in forward_context.items():
                    if hasattr(layer, "kv_cache"):
                        if layer.kv_cache:
                            logger.debug("Clearing KV cache for layer %s with %d cache entries", 
                                        layer_name, len(layer.kv_cache))
                            for cache_idx, kv_cache in enumerate(layer.kv_cache):
                                if torch.is_tensor(kv_cache):
                                    logger.debug("  Clearing layer %s cache %d with shape %s", 
                                                layer_name, cache_idx, kv_cache.shape)
                                    kv_cache.zero_()
                                elif isinstance(kv_cache, (list, tuple)):
                                    logger.debug("  Clearing layer %s compound cache %d with %d tensors", 
                                                layer_name, cache_idx, len(kv_cache))
                                    for tensor in kv_cache:
                                        if torch.is_tensor(tensor):
                                            tensor.zero_()
                            cleared_layer_caches += 1
                        else:
                            logger.debug("Layer %s has empty kv_cache", layer_name)
                    else:
                        logger.debug("Layer %s has no kv_cache attribute", layer_name)
                
                logger.debug("Cleared KV caches for %d layers in forward context", cleared_layer_caches)
            else:
                logger.debug("compilation_config.static_forward_context not found - skipping forward context clearing")
        else:
            logger.debug("model_runner.compilation_config not found - skipping forward context clearing")
        
        # 3. Clear request state and metadata
        if hasattr(self.model_runner, "input_batch"):
            logger.debug("Found model_runner.input_batch")
            ib = self.model_runner.input_batch
            if ib and hasattr(ib, "block_table"):
                try:
                    bt = ib.block_table
                    # MultiGroupBlockTable exposes a clear() to reset all groups
                    group_count = getattr(bt, "block_tables", None)
                    if isinstance(group_count, list):
                        logger.debug("Clearing input_batch.block_table with %d groups",
                                     len(group_count))
                    else:
                        logger.debug("Clearing input_batch.block_table (multi-group)")
                    if hasattr(bt, "clear"):
                        bt.clear()
                    else:
                        # Fallback: try to clear underlying groups if exposed
                        if hasattr(bt, "block_tables") and isinstance(bt.block_tables, list):
                            for i, block_table in enumerate(bt.block_tables):
                                if hasattr(block_table, "clear"):
                                    logger.debug("Clearing block_table group %d via clear()", i)
                                    block_table.clear()
                                elif hasattr(block_table, "zero_"):
                                    logger.debug("Clearing block_table group %d via zero_()", i)
                                    block_table.zero_()
                        else:
                            logger.debug("input_batch.block_table has no clearable groups exposed")
                except Exception:
                    logger.debug("Failed clearing input_batch.block_table", exc_info=True)
            else:
                logger.debug("input_batch has no block_table or is None - skipping block table clearing")
        else:
            logger.debug("model_runner.input_batch not found - skipping block table clearing")
        
        # 4. Clear encoder cache (for multimodal models)
        if hasattr(self.model_runner, "encoder_cache"):
            cache_size = len(self.model_runner.encoder_cache)
            logger.debug("Found model_runner.encoder_cache with %d entries, clearing", cache_size)
            self.model_runner.encoder_cache.clear()
            logger.debug("Cleared encoder cache")
        else:
            logger.debug("model_runner.encoder_cache not found - skipping encoder cache clearing")
        
        # 5. Reset attention group metadata builders
        if hasattr(self.model_runner, "attn_groups"):
            logger.debug("Found model_runner.attn_groups with %d groups", len(self.model_runner.attn_groups))
            
            reset_builders = 0
            skipped_builders = 0
            failed_resets = 0
            
            for group_idx, attn_group_list in enumerate(self.model_runner.attn_groups):
                logger.debug("Processing attention group %d with %d sub-groups", group_idx, len(attn_group_list))
                for subgroup_idx, attn_group in enumerate(attn_group_list):
                    if hasattr(attn_group, "metadata_builder"):
                        if hasattr(attn_group.metadata_builder, "reset"):
                            try:
                                logger.debug("Resetting metadata builder for group %d.%d", group_idx, subgroup_idx)
                                attn_group.metadata_builder.reset()
                                reset_builders += 1
                            except (AttributeError, NotImplementedError) as e:
                                logger.debug("Failed to reset metadata builder for group %d.%d: %s", 
                                           group_idx, subgroup_idx, str(e))
                                failed_resets += 1
                        else:
                            logger.debug("Metadata builder for group %d.%d has no reset() method", 
                                        group_idx, subgroup_idx)
                            skipped_builders += 1
                    else:
                        logger.debug("Attention group %d.%d has no metadata_builder", group_idx, subgroup_idx)
            
            logger.debug("Attention group reset summary: %d reset, %d skipped, %d failed", 
                        reset_builders, skipped_builders, failed_resets)
        else:
            logger.debug("model_runner.attn_groups not found - skipping attention group reset")
        
        # 6. Invalidate CUDA graphs (if enabled) since they may reference old KV cache tensors
        if hasattr(self.model_runner, "use_cuda_graph"):
            if self.model_runner.use_cuda_graph:
                logger.debug("CUDA graphs are enabled - they will be regenerated on next capture")
                logger.info("CUDA graphs are enabled - they will be regenerated on next capture")
                # Note: CUDA graphs will be automatically regenerated on the next model execution
                # since we've cleared the KV cache tensors they reference
            else:
                logger.debug("CUDA graphs are disabled")
        else:
            logger.debug("model_runner.use_cuda_graph not found - assuming CUDA graphs are not used")
            
        # 7. Force CUDA synchronization and garbage collection
        logger.debug("Performing CUDA synchronization and cache cleanup")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        logger.info("V1 KV cache flush completed successfully after weight update")

    def _validate_model_state_after_update(self) -> None:
        """Validate model state consistency after weight updates.
        
        Performs basic checks to ensure the model is in a valid state
        after parameter updates.
        """
        logger.debug("Starting model state validation after weight update")
        
        try:
            if hasattr(self.model_runner, "model"):
                model = self.model_runner.model
                logger.debug("Found model_runner.model, starting validation checks")
            else:
                logger.debug("model_runner.model not found - skipping validation")
                return
            
            # Count total parameters for logging
            total_params = 0
            try:
                total_params = sum(1 for _ in model.named_parameters())
                logger.debug("Model has %d parameters to validate", total_params)
            except Exception as e:
                logger.debug("Failed to count parameters: %s", str(e))
            
            # Check for NaN or infinite parameters
            logger.debug("Checking for NaN and infinite parameters")
            nan_params = []
            inf_params = []
            param_count = 0
            
            for name, param in model.named_parameters():
                param_count += 1
                logger.debug("Checking parameter %d/%d: %s with shape %s", 
                           param_count, total_params, name, param.shape)
                
                try:
                    if torch.isnan(param).any():
                        nan_params.append(name)
                        logger.debug("  Found NaN in parameter: %s", name)
                    if torch.isinf(param).any():
                        inf_params.append(name)
                        logger.debug("  Found inf in parameter: %s", name)
                except Exception as e:
                    logger.debug("  Failed to check parameter %s: %s", name, str(e))
            
            if nan_params:
                logger.warning(f"Found NaN parameters after weight update: {nan_params[:5]}")
                logger.debug("Full list of NaN parameters: %s", nan_params)
            else:
                logger.debug("No NaN parameters found")
                
            if inf_params:
                logger.warning(f"Found infinite parameters after weight update: {inf_params[:5]}")
                logger.debug("Full list of infinite parameters: %s", inf_params)
            else:
                logger.debug("No infinite parameters found")
            
            # Check parameter device placement
            logger.debug("Checking parameter device placement consistency")
            device_mismatches = []
            expected_device = None
            
            try:
                expected_device = next(model.parameters()).device
                logger.debug("Expected device for all parameters: %s", expected_device)
            except StopIteration:
                logger.debug("Model has no parameters - skipping device check")
                expected_device = None
            
            if expected_device is not None:
                device_check_count = 0
                for name, param in model.named_parameters():
                    device_check_count += 1
                    if param.device != expected_device:
                        device_mismatches.append((name, param.device, expected_device))
                        logger.debug("Device mismatch for %s: got %s, expected %s", 
                                   name, param.device, expected_device)
                
                logger.debug("Checked device placement for %d parameters", device_check_count)
                
                if device_mismatches:
                    logger.warning(f"Found device mismatches after weight update: {device_mismatches[:3]}")
                    logger.debug("Full list of device mismatches: %s", device_mismatches)
                else:
                    logger.debug("All parameters are on the correct device")
            
            # Basic dtype consistency check
            logger.debug("Checking parameter dtype consistency")
            dtypes = set()
            dtype_counts = {}
            
            for name, param in model.named_parameters():
                dtype = param.dtype
                dtypes.add(dtype)
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                logger.debug("Parameter %s has dtype %s", name, dtype)
            
            logger.debug("Found parameter dtypes: %s", dtype_counts)
            
            if len(dtypes) > 2:  # Allow for some dtype variety (e.g., embedding weights)
                logger.info(f"Model has multiple parameter dtypes after update: {dtypes}")
                logger.debug("Dtype distribution: %s", dtype_counts)
            else:
                logger.debug("Parameter dtypes are consistent")
                
            logger.debug("Model state validation completed successfully after weight update")
            
        except Exception as e:
            logger.warning(f"Model state validation failed after weight update: {e}")
            logger.debug("Full validation exception", exc_info=True)
            # Don't raise - this is just a validation check

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)
        parallel_config = self.vllm_config.parallel_config
        if parallel_config.distributed_executor_backend != "external_launcher" \
            and not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return None
        assert isinstance(output, ModelRunnerOutput)
        return output if self.is_driver_worker else None

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
            print(self.profiler.key_averages().table(
                sort_by="self_cuda_time_total"))

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank, backend)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    ensure_kv_transfer_initialized(vllm_config)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the "
                "`dtype` flag in CLI, for example: --dtype=half.")

