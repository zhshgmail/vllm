# vLLM Weight Update Examples

This directory contains examples demonstrating how to use vLLM's dynamic weight update capabilities for training workflows, particularly async RL training.

## Overview

vLLM supports two types of weight updates during runtime:

1. **Disk-based updates**: Load new model weights from disk files
2. **NCCL-based updates**: Receive weights via NCCL broadcast from training processes

These examples show how to integrate both approaches into your training pipeline.

## Files

- `weight_update_client.py` - Complete client library with examples for both update modes
- `async_rl_training.py` - Simulation of async RL training workflow (like AReaL)
- `README.md` - This documentation

## Prerequisites

1. **Start vLLM server with weight update API enabled:**
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model /path/to/model \
       --enable-weight-update-api \
       --worker-extension-cls vllm.worker.weight_update_extension.WeightUpdateExtension
   ```

2. **Install dependencies:**
   ```bash
   pip install aiohttp  # For HTTP client functionality
   ```

## Usage Examples

### Disk-based Weight Updates

Update model weights from a directory containing safetensors files:

```bash
# Dry run (validation only)
python weight_update_client.py --mode disk --path /path/to/weights --dry-run

# Actual update
python weight_update_client.py --mode disk --path /path/to/weights
```

**Python API:**
```python
from weight_update_client import WeightUpdateClient

async def update_from_disk():
    async with WeightUpdateClient("http://localhost:8000") as client:
        result = await client.update_weights_from_disk(
            path="/path/to/weights",
            dry_run=False,
            interrupt=True  # Interrupt running requests
        )
        print(f"Update result: {result}")
```

### NCCL-based Weight Updates

Coordinate weight updates with a training process via NCCL:

```bash
# Setup NCCL group and wait for training process
python weight_update_client.py --mode nccl --master-addr 127.0.0.1 --master-port 29500
```

**Python API:**
```python
from weight_update_client import WeightUpdateClient

async def update_via_nccl():
    async with WeightUpdateClient("http://localhost:8000") as client:
        # Initialize NCCL group
        await client.setup_nccl_group("127.0.0.1", 29500, rank_offset=1)
        
        # Update weights from training process
        result = await client.update_weights_from_nccl(allow_interrupt=True)
        print(f"Update result: {result}")
```

### Async RL Training Simulation

Run a complete async RL training simulation:

```bash
python async_rl_training.py --server-url http://localhost:8000 --num-steps 10
```

This example demonstrates:
- Generating rollouts with the current policy
- Training the policy on collected data
- Updating vLLM server with new weights
- Handling request interruption for on-policy training

### Monitoring Weight Updates

Monitor the weight update server status:

```bash
python weight_update_client.py --mode monitor
```

## Integration Patterns

### Pattern 1: Disk-based Training

```python
# Training loop with disk-based updates
async def training_loop():
    client = WeightUpdateClient("http://localhost:8000")
    await client.initialize()
    
    for epoch in range(num_epochs):
        # Train model and save checkpoint
        train_model_and_save(f"/tmp/checkpoint_epoch_{epoch}")
        
        # Update vLLM server
        await client.update_weights_from_disk(
            path=f"/tmp/checkpoint_epoch_{epoch}",
            interrupt=True
        )
```

### Pattern 2: NCCL-based Async RL

```python
# Async RL training with NCCL coordination
async def async_rl_training():
    client = WeightUpdateClient("http://localhost:8000")
    await client.initialize()
    await client.setup_nccl_group("127.0.0.1", 29500)
    
    for step in range(training_steps):
        # Generate rollouts
        rollouts = await generate_rollouts(client)
        
        # Train policy (in separate training process)
        train_policy_async(rollouts)
        
        # Coordinate weight update via NCCL
        await client.update_weights_from_nccl(allow_interrupt=True)
```

## Server Configuration

The vLLM server must be started with specific arguments to enable weight updates:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/DialoGPT-medium \
    --enable-weight-update-api \
    --worker-extension-cls vllm.worker.weight_update_extension.WeightUpdateExtension \
    --max-model-len 512 \
    --port 8000
```

**Required arguments:**
- `--enable-weight-update-api`: Enables weight update endpoints
- `--worker-extension-cls`: Loads weight update extension

**Recommended for testing:**
- `--max-model-len 512`: Smaller context for faster startup
- `--enforce-eager`: Disable CUDA graphs for simpler debugging

## API Reference

### WeightUpdateClient

**Methods:**
- `update_weights_from_disk(path, pattern=None, dry_run=False, interrupt=True, pause=True)`
- `setup_nccl_group(master_address, master_port, rank_offset=0, world_size=2)`
- `update_weights_from_nccl(allow_interrupt=True, validate_consistency=False)`
- `get_stats()` - Get client and server statistics

**Context manager:**
```python
async with WeightUpdateClient(server_url) as client:
    # Client is automatically initialized and cleaned up
    result = await client.update_weights_from_disk("/path/to/weights")
```

## Troubleshooting

### Common Issues

1. **"Weight update API not enabled"**
   - Start server with `--enable-weight-update-api`
   - Include `--worker-extension-cls vllm.worker.weight_update_extension.WeightUpdateExtension`

2. **"NCCL initialization failed"**
   - Ensure training process is running with matching NCCL configuration
   - Check firewall settings for NCCL ports
   - Verify `world_size` matches between training and inference processes

3. **"Weight update already in progress"**
   - Wait for current update to complete
   - Check server status with `get_stats()`

4. **Connection errors**
   - Verify vLLM server is running and accessible
   - Check server URL and port

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Server Health Check

```bash
curl http://localhost:8000/health
curl http://localhost:8000/weights/stats
```

## Performance Considerations

- **Disk updates**: I/O bound, speed depends on storage and model size
- **NCCL updates**: Network bound, requires coordination between processes
- **Request interruption**: Brief service interruption during weight updates
- **Memory usage**: Temporary memory increase during weight loading

## Related Examples

- `examples/offline_inference/rlhf.py` - RLHF training patterns
- `examples/offline_inference/save_sharded_state.py` - Saving model checkpoints
- `examples/offline_inference/load_sharded_state.py` - Loading model checkpoints