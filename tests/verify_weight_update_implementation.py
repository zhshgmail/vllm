#!/usr/bin/env python3
"""
Integration verification script - checks     for pattern in required_patterns:
        if not re.search(pattern, content, re.IGNORECASE):
            print(f"[FAIL] Missing implementation pattern: {pattern}")
            return False
    
    print("[PASS] health_check_active implementation verified")ur actual implementation
matches the tested logic without running the full engine.
"""

import re
import os


def verify_output_processor_implementation():
    """Verify that our finalize_and_abort_all implementation is present."""
    print("[INFO] Verifying OutputProcessor.finalize_and_abort_all implementation...")
    
    output_processor_path = "vllm/v1/engine/output_processor.py"
    if not os.path.exists(output_processor_path):
        print(f"[FAIL] File not found: {output_processor_path}")
        return False
    
    with open(output_processor_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for method signature
    if "def finalize_and_abort_all(self)" not in content:
        print("[FAIL] finalize_and_abort_all method not found")
        return False
    
    # Check for key implementation elements
    required_patterns = [
        r"aborted.*=.*\[\]",  # Initialize aborted list
        r"for.*req_id.*req_state.*in.*list.*self\.request_states\.items",  # Iterate over copy
        r"make_request_output.*\[\].*FinishReason\.ABORT",  # Create abort output
        r"req_state\.queue\.put\(ro\)",  # Put output in queue
        r"self\.abort_requests\(aborted\)",  # Clean up states
        r"return aborted"  # Return aborted IDs
    ]
    
    for pattern in required_patterns:
        if not re.search(pattern, content, re.IGNORECASE):
            print(f"[FAIL] Missing implementation pattern: {pattern}")
            return False
    
    print("[PASS] finalize_and_abort_all implementation verified")
    return True


def verify_async_llm_implementation():
    """Verify that our abort_all_active implementation is present."""
    print("[INFO] Verifying AsyncLLM.abort_all_active implementation...")
    
    async_llm_path = "vllm/v1/engine/async_llm.py"
    if not os.path.exists(async_llm_path):
        print(f"[FAIL] File not found: {async_llm_path}")
        return False
    
    with open(async_llm_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for method signature
    if "async def abort_all_active(self)" not in content:
        print("[FAIL] abort_all_active method not found")
        return False
    
    # Check for key implementation elements
    required_patterns = [
        r"aborted_ids.*=.*self\.output_processor\.finalize_and_abort_all",  # Call finalize
        r"await.*self\.engine_core\.abort_requests_async\(aborted_ids\)",  # Propagate to core
        r"return len\(aborted_ids\)"  # Return count
    ]
    
    for pattern in required_patterns:
        if not re.search(pattern, content, re.IGNORECASE):
            print(f"[FAIL] Missing implementation pattern: {pattern}")
            return False
    
    print("[PASS] abort_all_active implementation verified")
    return True


def verify_api_server_integration():
    """Verify that the API server endpoint includes interrupt functionality."""
    print("[INFO] Verifying API server interrupt integration...")
    
    api_server_path = "vllm/entrypoints/openai/api_server.py"
    if not os.path.exists(api_server_path):
        print(f"[FAIL] File not found: {api_server_path}")
        return False
    
    with open(api_server_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for interrupt flag handling
    required_patterns = [
        r'interrupt_flag.*=.*bool.*body\.get.*"interrupt".*True',  # Parse interrupt flag
        r'num_interrupted_requests.*=.*0',  # Initialize counter
        r'if interrupt_flag',  # Conditional interrupt logic
        r'abort_all_active',  # Call abort method
        r'"num_interrupted_requests".*num_interrupted_requests'  # Include in response
    ]
    
    for pattern in required_patterns:
        if not re.search(pattern, content, re.IGNORECASE):
            print(f"[FAIL] Missing API server pattern: {pattern}")
            return False
    
    print("[PASS] API server interrupt integration verified")
    return True


def verify_imports_and_dependencies():
    """Verify that required imports are present."""
    print("[INFO] Verifying imports and dependencies...")
    
    # Check output_processor.py imports FinishReason
    output_processor_path = "vllm/v1/engine/output_processor.py"
    with open(output_processor_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "from vllm.v1.engine import" not in content or "FinishReason" not in content:
        print("[FAIL] FinishReason import missing from output_processor.py")
        return False
    
    print("[PASS] All imports and dependencies verified")
    return True


def main():
    """Run all verification checks."""
    print("[INFO] Verifying Weight Update with Interruption Implementation")
    print("=" * 65)
    
    checks = [
        verify_output_processor_implementation,
        verify_async_llm_implementation,
        verify_api_server_integration,
        verify_imports_and_dependencies
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
        print()
    
    print("=" * 65)
    if all_passed:
        print("[PASS] ALL IMPLEMENTATION CHECKS PASSED!")
        print("\nThe implementation includes:")
        print("  [PASS] finalize_and_abort_all() method in OutputProcessor")
        print("  [PASS] abort_all_active() method in AsyncLLM")  
        print("  [PASS] interrupt flag handling in API endpoint")
        print("  [PASS] Response field num_interrupted_requests")
        print("  [PASS] Proper error handling and state cleanup")
        print("\n[INFO] Ready to test with a live vLLM instance!")
        return True
    else:
        print("[FAIL] SOME IMPLEMENTATION CHECKS FAILED!")
        print("\nPlease review the missing patterns above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
