#!/usr/bin/env python3
"""
vLLM Weight Update Client Examples

This module demonstrates how to use vLLM's weight update API for dynamic model updates
during training. It provides examples for both disk-based and NCCL-based weight updates.

Usage:
    # For disk-based weight updates
    python weight_update_client.py --mode disk --path /path/to/weights

    # For NCCL-based weight updates (async RL training)
    python weight_update_client.py --mode nccl --master-addr 127.0.0.1 --master-port 29500
"""

import asyncio
import argparse
import json
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp

# For standalone usage without installing vLLM
try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


@dataclass
class NCCLConfig:
    """NCCL configuration for distributed training communication."""
    master_address: str
    master_port: int
    rank_offset: int = 0
    world_size: int = 2


@dataclass
class WeightUpdateRequest:
    """Request for weight update operation."""
    request_id: str
    mode: str  # 'disk' or 'nccl'
    
    # Disk-specific parameters
    path: Optional[str] = None
    pattern: Optional[str] = None
    dry_run: bool = False
    interrupt: bool = True
    pause: bool = True
    
    # NCCL-specific parameters
    allow_interrupt: bool = True
    validate_consistency: bool = False
    timeout: float = 60.0


class WeightUpdateClient:
    """
    Complete client for vLLM weight update operations.
    
    Supports both disk-based and NCCL-based weight updates with examples
    for integration into training workflows.
    """
    
    def __init__(self, vllm_server_url: str):
        """Initialize weight update client."""
        self.server_url = vllm_server_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        
        # State tracking
        self.nccl_initialized = False
        self.stats = {
            'disk_updates': 0,
            'nccl_updates': 0,
            'total_requests': 0,
            'failed_requests': 0,
            'last_update_time': 0.0,
        }
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        
    async def initialize(self):
        """Initialize client and validate server."""
        self.session = aiohttp.ClientSession()
        await self._validate_server()
        
    async def cleanup(self):
        """Cleanup client resources."""
        if self.nccl_initialized:
            await self._cleanup_nccl()
        if self.session:
            await self.session.close()
            
    async def _validate_server(self):
        """Validate vLLM server and weight update API availability."""
        try:
            # Check server health
            async with self.session.get(f"{self.server_url}/health") as response:
                if response.status != 200:
                    raise RuntimeError(f"vLLM server unhealthy: {response.status}")
            
            # Check weight update API
            async with self.session.get(f"{self.server_url}/weights/stats") as response:
                if response.status == 404:
                    raise RuntimeError(
                        "Weight update API not enabled. Start server with:\n"
                        "  --enable-weight-update-api \\\n"  
                        "  --worker-extension-cls vllm.worker.weight_update_extension.WeightUpdateExtension"
                    )
                elif response.status != 200:
                    raise RuntimeError(f"Weight update API error: {response.status}")
                    
            logger.info("Successfully connected to vLLM server with weight update API")
            
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Cannot connect to vLLM server: {e}")

    async def update_weights_from_disk(self, 
                                     path: str, 
                                     pattern: Optional[str] = None,
                                     dry_run: bool = False,
                                     interrupt: bool = True,
                                     pause: bool = True) -> Dict[str, Any]:
        """
        Update model weights from disk-based checkpoint.
        
        Args:
            path: Directory containing sharded safetensors files
            pattern: Filename pattern (e.g., "model-{rank}-{part}.safetensors")
            dry_run: Only validate, don't update weights
            interrupt: Interrupt running requests during update
            pause: Pause scheduler during update
            
        Returns:
            Dictionary with update results
        """
        request_data = {
            "path": path,
            "dry_run": dry_run,
            "interrupt": interrupt,
            "pause": pause
        }
        
        if pattern:
            request_data["pattern"] = pattern
            
        try:
            self.stats['total_requests'] += 1
            start_time = time.time()
            
            async with self.session.post(
                f"{self.server_url}/weights/update-from-disk",
                json=request_data
            ) as response:
                result = await response.json()
                
                if result.get("ok"):
                    self.stats['disk_updates'] += 1
                    self.stats['last_update_time'] = time.time()
                    duration = time.time() - start_time
                    
                    logger.info(f"Disk weight update successful in {duration:.2f}s")
                    if dry_run:
                        logger.info(f"Dry run validated {result.get('validated_tensors', 0)} tensors")
                    else:
                        logger.info(f"Updated {result.get('validated_tensors', 0)} tensors")
                        
                    return {
                        "status": "success",
                        "mode": "disk",
                        "duration": duration,
                        **result
                    }
                else:
                    self.stats['failed_requests'] += 1
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Disk weight update failed: {error_msg}")
                    return {
                        "status": "error", 
                        "mode": "disk",
                        "error": error_msg,
                        **result
                    }
                    
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Disk weight update exception: {e}")
            return {"status": "error", "mode": "disk", "error": str(e)}

    async def setup_nccl_group(self, 
                             master_address: str,
                             master_port: int, 
                             rank_offset: int = 0,
                             world_size: int = 2,
                             timeout: float = 30.0) -> bool:
        """
        Initialize NCCL group for distributed weight updates.
        
        Args:
            master_address: Training process master address
            master_port: Training process master port  
            rank_offset: Rank offset for vLLM workers
            world_size: Total processes in NCCL group
            timeout: Initialization timeout
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.session.post(
                f"{self.server_url}/weights/init-nccl-group", 
                json={
                    "nccl_config": {
                        "master_address": master_address,
                        "master_port": master_port,
                        "rank_offset": rank_offset,
                        "world_size": world_size
                    },
                    "timeout": timeout
                }
            ) as response:
                result = await response.json()
                
                if result.get("status") == "success":
                    self.nccl_initialized = True
                    logger.info(f"NCCL group initialized: {world_size} processes")
                    return True
                else:
                    logger.error(f"NCCL initialization failed: {result}")
                    return False
                    
        except Exception as e:
            logger.error(f"NCCL group setup failed: {e}")
            return False

    async def update_weights_from_nccl(self,
                                     allow_interrupt: bool = True,
                                     validate_consistency: bool = False,
                                     timeout: float = 60.0) -> Dict[str, Any]:
        """
        Update model weights via NCCL broadcast from training process.
        
        Args:
            allow_interrupt: Whether to interrupt running requests  
            validate_consistency: Validate weight consistency across workers
            timeout: Update timeout in seconds
            
        Returns:
            Dictionary with update results
        """
        if not self.nccl_initialized:
            return {
                "status": "error", 
                "mode": "nccl",
                "error": "NCCL group not initialized. Call setup_nccl_group() first."
            }
            
        try:
            self.stats['total_requests'] += 1
            start_time = time.time()
            
            async with self.session.post(
                f"{self.server_url}/weights/update-from-nccl",
                json={
                    "allow_interrupt": allow_interrupt,
                    "validate_consistency": validate_consistency,
                    "timeout": timeout
                }
            ) as response:
                result = await response.json()
                
                if result.get("status") == "success":
                    self.stats['nccl_updates'] += 1
                    self.stats['last_update_time'] = time.time()
                    duration = time.time() - start_time
                    
                    interrupted_count = len(result.get("interrupted_requests", []))
                    logger.info(f"NCCL weight update successful in {duration:.2f}s, "
                              f"{interrupted_count} requests interrupted")
                              
                    return {
                        "status": "success",
                        "mode": "nccl", 
                        "duration": duration,
                        **result
                    }
                else:
                    self.stats['failed_requests'] += 1
                    error_msg = result.get("message", "Unknown error")
                    logger.error(f"NCCL weight update failed: {error_msg}")
                    return {
                        "status": "error",
                        "mode": "nccl", 
                        "error": error_msg,
                        **result
                    }
                    
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"NCCL weight update exception: {e}")
            return {"status": "error", "mode": "nccl", "error": str(e)}

    async def _cleanup_nccl(self):
        """Cleanup NCCL group resources."""
        if not self.nccl_initialized:
            return
            
        try:
            async with self.session.post(f"{self.server_url}/weights/cleanup-nccl-group") as response:
                result = await response.json()
                if result.get("status") == "success":
                    self.nccl_initialized = False
                    logger.info("NCCL group cleaned up")
        except Exception as e:
            logger.error(f"NCCL cleanup failed: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get client and server statistics."""
        try:
            async with self.session.get(f"{self.server_url}/weights/stats") as response:
                server_stats = await response.json()
                
            return {
                "client_stats": self.stats.copy(),
                "server_stats": server_stats,
                "nccl_initialized": self.nccl_initialized
            }
        except Exception as e:
            return {"error": f"Failed to get stats: {e}"}


# Example usage functions
async def example_disk_update(server_url: str, weights_path: str, dry_run: bool = False):
    """Example: Update weights from disk."""
    print(f"\n=== Disk Weight Update Example ===")
    print(f"Server: {server_url}")
    print(f"Weights path: {weights_path}")
    print(f"Dry run: {dry_run}")
    
    async with WeightUpdateClient(server_url) as client:
        # Get initial stats
        stats = await client.get_stats()
        print(f"Initial stats: {stats.get('client_stats', {})}")
        
        # Update weights from disk
        result = await client.update_weights_from_disk(
            path=weights_path,
            dry_run=dry_run,
            interrupt=True
        )
        
        print(f"Update result: {result}")
        
        # Get final stats
        stats = await client.get_stats()
        print(f"Final stats: {stats.get('client_stats', {})}")


async def example_nccl_update(server_url: str, master_addr: str, master_port: int):
    """Example: Update weights via NCCL."""
    print(f"\n=== NCCL Weight Update Example ===")
    print(f"Server: {server_url}")
    print(f"Training master: {master_addr}:{master_port}")
    
    async with WeightUpdateClient(server_url) as client:
        # Setup NCCL group
        print("Setting up NCCL group...")
        success = await client.setup_nccl_group(
            master_address=master_addr,
            master_port=master_port,
            rank_offset=1,  # Training process is rank 0, vLLM workers start from 1
            world_size=2    # Training process + vLLM server
        )
        
        if not success:
            print("NCCL setup failed!")
            return
            
        print("NCCL group initialized successfully")
        
        # Simulate waiting for training process to be ready
        print("Waiting for training process to be ready...")
        await asyncio.sleep(2)
        
        # Update weights from training
        print("Updating weights via NCCL...")
        result = await client.update_weights_from_nccl(
            allow_interrupt=True,
            validate_consistency=False
        )
        
        print(f"Update result: {result}")


async def example_monitoring_loop(server_url: str):
    """Example: Monitor weight update server."""
    print(f"\n=== Weight Update Monitoring Example ===")
    print(f"Server: {server_url}")
    print("Monitoring for 30 seconds...")
    
    async with WeightUpdateClient(server_url) as client:
        start_time = time.time()
        
        while time.time() - start_time < 30:
            stats = await client.get_stats()
            server_stats = stats.get('server_stats', {})
            
            print(f"[{time.time() - start_time:.1f}s] "
                  f"Updates: {server_stats.get('weight_update_count', 0)}, "
                  f"In progress: {server_stats.get('weight_update_in_progress', False)}")
            
            await asyncio.sleep(5)


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="vLLM Weight Update Client Examples")
    parser.add_argument("--server-url", default="http://localhost:8000", 
                       help="vLLM server URL")
    parser.add_argument("--mode", choices=["disk", "nccl", "monitor"], required=True,
                       help="Weight update mode")
    
    # Disk-specific arguments
    parser.add_argument("--path", help="Path to weights directory (for disk mode)")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't update")
    
    # NCCL-specific arguments
    parser.add_argument("--master-addr", default="127.0.0.1", help="Training master address")
    parser.add_argument("--master-port", type=int, default=29500, help="Training master port")
    
    args = parser.parse_args()
    
    if args.mode == "disk":
        if not args.path:
            parser.error("--path is required for disk mode")
        asyncio.run(example_disk_update(args.server_url, args.path, args.dry_run))
        
    elif args.mode == "nccl":
        asyncio.run(example_nccl_update(args.server_url, args.master_addr, args.master_port))
        
    elif args.mode == "monitor":
        asyncio.run(example_monitoring_loop(args.server_url))


if __name__ == "__main__":
    main()