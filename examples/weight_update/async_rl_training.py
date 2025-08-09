#!/usr/bin/env python3
"""
Async RL Training Example with vLLM Weight Updates

This example demonstrates how to integrate vLLM with async RL training frameworks 
like AReaL, showing the complete workflow for dynamic weight updates during training.

Usage:
    python async_rl_training.py --server-url http://localhost:8000
"""

import asyncio
import argparse
import time
import uuid
from typing import List, Dict, Any
from weight_update_client import WeightUpdateClient

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class AsyncRLTrainingSimulator:
    """
    Simulates async RL training workflow with vLLM weight updates.
    
    This demonstrates the typical pattern used in frameworks like AReaL:
    1. Generate rollouts using current policy
    2. Train policy on collected data
    3. Update vLLM server with new policy weights
    4. Repeat
    """
    
    def __init__(self, vllm_server_url: str, nccl_master_port: int = 29500):
        self.server_url = vllm_server_url
        self.nccl_port = nccl_master_port
        self.client = None
        
        # Training simulation state
        self.training_step = 0
        self.total_generations = 0
        self.interrupted_generations = 0
        
    async def initialize(self):
        """Initialize the RL training simulator."""
        logger.info("Initializing async RL training simulator...")
        
        self.client = WeightUpdateClient(self.server_url)
        await self.client.initialize()
        
        # Setup NCCL group for weight updates
        success = await self.client.setup_nccl_group(
            master_address="127.0.0.1",
            master_port=self.nccl_port,
            rank_offset=1,  # Training process is rank 0
            world_size=2    # Training + vLLM server
        )
        
        if not success:
            raise RuntimeError("Failed to initialize NCCL group")
            
        logger.info("RL training simulator initialized successfully")
        
    async def cleanup(self):
        """Cleanup training simulator resources."""
        if self.client:
            await self.client.cleanup()
            
    async def simulate_rollout_generation(self, num_prompts: int = 10) -> List[Dict[str, Any]]:
        """
        Simulate generating rollouts for RL training.
        
        In a real implementation, this would:
        1. Send prompts to vLLM server
        2. Collect generated responses
        3. Compute rewards/values
        4. Return training data
        """
        logger.info(f"Simulating rollout generation with {num_prompts} prompts")
        
        # Simulate generation requests
        generations = []
        for i in range(num_prompts):
            generation = {
                "request_id": str(uuid.uuid4()),
                "prompt": f"Example prompt {i} for training step {self.training_step}",
                "generated_text": f"Generated response {i}",
                "reward": 0.5 + (i % 3) * 0.2,  # Simulated reward
                "value": 1.0 + (i % 2) * 0.5    # Simulated value
            }
            generations.append(generation)
            
        self.total_generations += len(generations)
        
        # Simulate generation time
        await asyncio.sleep(2.0)
        
        logger.info(f"Generated {len(generations)} rollouts")
        return generations
        
    async def simulate_policy_training(self, rollout_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulate training the policy on collected rollout data.
        
        In a real implementation, this would:
        1. Process rollout data (compute advantages, etc.)
        2. Run policy gradient updates
        3. Update model weights
        4. Return training metrics
        """
        logger.info(f"Simulating policy training on {len(rollout_data)} rollouts")
        
        # Simulate training time
        await asyncio.sleep(3.0)
        
        # Simulate training metrics
        metrics = {
            "training_step": self.training_step,
            "policy_loss": 0.1 + (self.training_step % 5) * 0.02,
            "value_loss": 0.05 + (self.training_step % 3) * 0.01,
            "entropy": 2.5 - self.training_step * 0.1,
            "learning_rate": 1e-4 * (0.95 ** self.training_step),
            "num_rollouts": len(rollout_data)
        }
        
        logger.info(f"Training metrics: {metrics}")
        return metrics
        
    async def update_vllm_weights(self, allow_interrupt: bool = True) -> Dict[str, Any]:
        """
        Update vLLM server with new policy weights via NCCL.
        
        In a real implementation, this would broadcast the updated
        model weights from the training process to the vLLM server.
        """
        logger.info("Updating vLLM server weights via NCCL...")
        
        result = await self.client.update_weights_from_nccl(
            allow_interrupt=allow_interrupt,
            validate_consistency=False,
            timeout=30.0
        )
        
        if result.get("status") == "success":
            interrupted_count = len(result.get("interrupted_requests", []))
            self.interrupted_generations += interrupted_count
            
            logger.info(f"Weight update successful, {interrupted_count} requests interrupted")
        else:
            logger.error(f"Weight update failed: {result.get('error', 'Unknown error')}")
            
        return result
        
    async def run_training_loop(self, num_steps: int = 5):
        """
        Run the complete async RL training loop.
        
        This demonstrates the typical async RL workflow:
        1. Generate rollouts with current policy
        2. Train policy on collected data  
        3. Update vLLM server with new weights
        4. Repeat
        """
        logger.info(f"Starting async RL training loop for {num_steps} steps")
        
        try:
            for step in range(num_steps):
                self.training_step = step + 1
                logger.info(f"\n=== Training Step {self.training_step}/{num_steps} ===")
                
                # Step 1: Generate rollouts using current policy
                rollouts = await self.simulate_rollout_generation(num_prompts=8)
                
                # Step 2: Train policy on collected data
                training_metrics = await self.simulate_policy_training(rollouts)
                
                # Step 3: Update vLLM server with new policy weights
                update_result = await self.update_vllm_weights(allow_interrupt=True)
                
                # Log step summary
                logger.info(f"Step {self.training_step} complete:")
                logger.info(f"  - Generated {len(rollouts)} rollouts")
                logger.info(f"  - Policy loss: {training_metrics['policy_loss']:.4f}")
                logger.info(f"  - Weight update: {update_result.get('status', 'unknown')}")
                
                # Brief pause between steps
                if step < num_steps - 1:
                    await asyncio.sleep(1.0)
                    
        except Exception as e:
            logger.error(f"Training loop failed: {e}")
            raise
            
        # Final statistics
        stats = await self.client.get_stats()
        logger.info(f"\n=== Training Complete ===")
        logger.info(f"Total training steps: {self.training_step}")
        logger.info(f"Total generations: {self.total_generations}")
        logger.info(f"Interrupted generations: {self.interrupted_generations}")
        logger.info(f"Client stats: {stats.get('client_stats', {})}")


async def main():
    """Main async RL training example."""
    parser = argparse.ArgumentParser(description="Async RL Training with vLLM")
    parser.add_argument("--server-url", default="http://localhost:8000",
                       help="vLLM server URL")
    parser.add_argument("--nccl-port", type=int, default=29500,
                       help="NCCL master port for training coordination")
    parser.add_argument("--num-steps", type=int, default=5,
                       help="Number of training steps to simulate")
    
    args = parser.parse_args()
    
    # Create and run training simulator
    simulator = AsyncRLTrainingSimulator(args.server_url, args.nccl_port)
    
    try:
        await simulator.initialize()
        await simulator.run_training_loop(args.num_steps)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        await simulator.cleanup()
        

if __name__ == "__main__":
    asyncio.run(main())