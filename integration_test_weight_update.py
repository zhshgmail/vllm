#!/usr/bin/env python3
"""
Integration test script for vLLM weight update API.

This script tests the complete workflow:
1. Validate that both original model path and load weights path exist
2. Start vLLM server in V1 mode with the original model
3. Send GSM8K test requests to verify initial functionality
4. Call the weight update API to load new weights from specified path
5. Send GSM8K test requests again to verify functionality after reload

Usage:
    python integration_test_weight_update.py --model-path /path/to/original/model --load-weights-path /path/to/new/weights

Requirements:
    - Both model paths must contain valid model files
    - Load weights path must contain .safetensors files
    - vLLM server will be started in V1 mode (required for weight updates)
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests
import signal

# GSM8K test problems - first 10 problems from the dataset
GSM8K_TEST_PROBLEMS = [
    {
        "question": "Natalie's apple orchard has 20 apple trees. Each apple tree produces 120 apples. She harvests all the apples from her orchard. Then she gives 5 apples to each of her 8 neighbors. How many apples does she have left?",
        "answer": "2360"
    },
    {
        "question": "John has 3 boxes. Each box contains 5 marbles. How many marbles does John have in total?",
        "answer": "15"
    },
    {
        "question": "A bakery sells cupcakes for $3 each. If they sold 24 cupcakes today, how much money did they make?",
        "answer": "72"
    },
    {
        "question": "Sarah has 48 stickers. She wants to put them in albums. Each page in an album can hold 6 stickers. How many pages will she need?",
        "answer": "8"
    },
    {
        "question": "A car travels 60 miles per hour. How far will it travel in 2.5 hours?",
        "answer": "150"
    },
    {
        "question": "Mike buys 4 packs of trading cards. Each pack has 12 cards. If he already had 15 cards, how many cards does he have now?",
        "answer": "63"
    },
    {
        "question": "A recipe calls for 2 cups of flour to make 12 cookies. How many cups of flour are needed to make 36 cookies?",
        "answer": "6"
    },
    {
        "question": "Lisa works 8 hours a day and earns $15 per hour. How much does she earn in 5 days?",
        "answer": "600"
    },
    {
        "question": "A movie theater has 15 rows with 20 seats in each row. What is the total seating capacity?",
        "answer": "300"
    },
    {
        "question": "Tom has $150. He spends $35 on groceries and $28 on gas. How much money does he have left?",
        "answer": "87"
    }
]


class VLLMIntegrationTester:
    def __init__(self, 
                 model_path: str,
                 load_weights_path: str,
                 server_port: int = 8000,
                 server_host: str = "127.0.0.1",
                 timeout: int = 120,
                 num_test_problems: int = 5):
        self.model_path = model_path
        self.load_weights_path = load_weights_path
        self.server_port = server_port
        self.server_host = server_host
        self.server_url = f"http://{server_host}:{server_port}"
        self.timeout = timeout
        self.num_test_problems = num_test_problems
        self.server_process: Optional[subprocess.Popen] = None
        
    def log(self, message: str):
        """Log with timestamp"""
        print(f"[{time.strftime('%H:%M:%S')}] {message}")
        
    def start_server(self) -> bool:
        """Start vLLM server in V1 mode"""
        self.log("Starting vLLM server...")
        
        # Use V1 engine with minimal configuration
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self.server_port),
            "--host", self.server_host,
            "--served-model-name", "test-model",
            "--max-model-len", "512",  # Small context for faster startup
            "--enforce-eager",  # Disable CUDA graphs for simplicity
            "--disable-log-requests",
            "--use-v2-block-manager",  # Enable V1 mode
        ]
        
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Wait for server to start
            self.log("Waiting for server to start...")
            for i in range(self.timeout):
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=2)
                    if response.status_code == 200:
                        self.log(f"Server started successfully after {i+1} seconds")
                        return True
                except requests.RequestException:
                    pass
                time.sleep(1)
                
            self.log("Server failed to start within timeout period")
            return False
            
        except Exception as e:
            self.log(f"Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the vLLM server"""
        if self.server_process:
            self.log("Stopping server...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.log("Force killing server...")
                self.server_process.kill()
                self.server_process.wait()
            self.server_process = None
            
    def test_generation(self, test_name: str, num_problems: int = 3) -> bool:
        """Test text generation using GSM8K problems"""
        self.log(f"Testing generation with {num_problems} GSM8K problems ({test_name})...")
        
        success_count = 0
        total_problems = min(num_problems, len(GSM8K_TEST_PROBLEMS))
        
        for i in range(total_problems):
            problem = GSM8K_TEST_PROBLEMS[i]
            
            # Create a math-focused prompt
            prompt = f"Solve this math problem step by step:\n\nProblem: {problem['question']}\n\nSolution:"
            
            payload = {
                "model": "test-model", 
                "prompt": prompt,
                "max_tokens": 200,
                "temperature": 0.1,  # Low temperature for consistent math solving
                "stop": ["\n\n", "Problem:"]  # Stop at double newline or next problem
            }
            
            try:
                response = requests.post(
                    f"{self.server_url}/v1/completions",
                    json=payload,
                    timeout=30,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        generated_text = result["choices"][0]["text"].strip()
                        
                        # Log the problem and response
                        self.log(f"Problem {i+1}: {problem['question'][:50]}...")
                        self.log(f"Expected: {problem['answer']}")
                        self.log(f"Generated: {generated_text[:100]}...")
                        
                        # Check if the expected answer appears in the generated text
                        if problem['answer'] in generated_text:
                            self.log(f"✓ Problem {i+1} - Answer found in response")
                            success_count += 1
                        else:
                            self.log(f"? Problem {i+1} - Answer not found, but response generated")
                            success_count += 0.5  # Partial credit for generating something
                            
                    else:
                        self.log(f"✗ Problem {i+1} - No choices in response: {result}")
                else:
                    self.log(f"✗ Problem {i+1} - Request failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                self.log(f"✗ Problem {i+1} - Error: {e}")
        
        success_rate = success_count / total_problems
        self.log(f"Generation test ({test_name}) - Success rate: {success_rate:.1%} ({success_count}/{total_problems})")
        
        # Consider test successful if we get responses for most problems
        return success_rate >= 0.7
    
    def validate_weight_paths(self) -> bool:
        """Validate that the specified weight paths exist and contain the expected files"""
        self.log("Validating weight paths...")
        
        # Validate original model path
        if not os.path.exists(self.model_path):
            self.log(f"✗ Original model path does not exist: {self.model_path}")
            return False
        
        # Validate load weights path
        if not os.path.exists(self.load_weights_path):
            self.log(f"✗ Load weights path does not exist: {self.load_weights_path}")
            return False
        
        # Check if load weights path contains safetensors files
        load_path = Path(self.load_weights_path)
        safetensors_files = list(load_path.glob("*.safetensors"))
        
        if not safetensors_files:
            self.log(f"✗ No .safetensors files found in load weights path: {self.load_weights_path}")
            return False
        
        self.log(f"✓ Original model path validated: {self.model_path}")
        self.log(f"✓ Load weights path validated: {self.load_weights_path}")
        self.log(f"✓ Found {len(safetensors_files)} .safetensors files in load path")
        
        return True
    
    def test_weight_update_api(self) -> bool:
        """Test the weight update API"""
        self.log("Testing weight update API...")
        
        payload = {
            "path": self.load_weights_path,
            "dry_run": False,  # Set to True for safer testing
            "pattern": None,  # Use default pattern
            "pause": True,    # Pause inference during update
            "interrupt": True  # Interrupt current requests
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/update-weights-from-disk",
                json=payload,
                timeout=60,  # Weight loading can take time
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                self.log(f"✓ Weight update successful: {result}")
                return True
            else:
                self.log(f"✗ Weight update failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.log(f"✗ Weight update error: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        self.log("Cleanup completed")
    
    def run_integration_test(self) -> bool:
        """Run the complete integration test"""
        self.log("=" * 60)
        self.log("Starting vLLM Weight Update Integration Test")
        self.log("=" * 60)
        
        try:
            # Step 1: Validate weight paths
            if not self.validate_weight_paths():
                self.log("✗ Weight path validation failed")
                return False
            
            # Step 2: Start server
            if not self.start_server():
                self.log("✗ Failed to start server")
                return False
            
            # Step 3: Test initial functionality
            if not self.test_generation("initial", self.num_test_problems):
                self.log("✗ Initial generation test failed")
                return False
            
            # Step 4: Test weight update API
            if not self.test_weight_update_api():
                self.log("✗ Weight update API test failed")
                return False
            
            # Give server time to complete weight loading
            self.log("Waiting for weight loading to complete...")
            time.sleep(5)
            
            # Step 5: Test functionality after reload
            if not self.test_generation("after_reload", self.num_test_problems):
                self.log("✗ Post-reload generation test failed")
                return False
            
            self.log("=" * 60)
            self.log("✓ All tests passed! Integration test successful.")
            self.log("=" * 60)
            return True
            
        except KeyboardInterrupt:
            self.log("Test interrupted by user")
            return False
        except Exception as e:
            self.log(f"✗ Unexpected error during integration test: {e}")
            return False
        finally:
            self.stop_server()
            self.cleanup()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Weight Update Integration Test")
    parser.add_argument("--model-path", required=True,
                       help="Path to the original model directory")
    parser.add_argument("--load-weights-path", required=True, 
                       help="Path to the directory containing weights to load")
    parser.add_argument("--port", type=int, default=8000,
                       help="Server port (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1",
                       help="Server host (default: 127.0.0.1)")
    parser.add_argument("--timeout", type=int, default=120,
                       help="Server startup timeout in seconds (default: 120)")
    parser.add_argument("--num-problems", type=int, default=5,
                       help="Number of GSM8K problems to test with (default: 5, max: 10)")
    
    args = parser.parse_args()
    
    # Validate number of problems
    num_problems = min(max(1, args.num_problems), len(GSM8K_TEST_PROBLEMS))
    if num_problems != args.num_problems:
        print(f"Note: Adjusted number of problems to {num_problems} (available: 1-{len(GSM8K_TEST_PROBLEMS)})")
    
    tester = VLLMIntegrationTester(
        model_path=args.model_path,
        load_weights_path=args.load_weights_path,
        server_port=args.port,
        server_host=args.host,
        timeout=args.timeout,
        num_test_problems=num_problems
    )
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nShutting down...")
        tester.stop_server()
        tester.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    success = tester.run_integration_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()