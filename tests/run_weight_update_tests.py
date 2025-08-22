#!/usr/bin/env python3
"""Test runner for weight update functionality.

This script runs all weight update related tests to ensure they work
in isolation without vLLM engine dependencies.
"""
import sys
import subprocess
import os


def run_test_file(test_file: str) -> bool:
    """Run a single test file and return success status."""
    print(f"\n{'='*50}")
    print(f"Running tests in {test_file}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
        ], cwd=os.path.dirname(os.path.abspath(__file__)), 
           capture_output=False, text=True)
        
        success = result.returncode == 0
        if success:
            print(f"✅ {test_file} - ALL TESTS PASSED")
        else:
            print(f"❌ {test_file} - SOME TESTS FAILED")
        
        return success
        
    except Exception as e:
        print(f"❌ {test_file} - ERROR RUNNING TESTS: {e}")
        return False


def main():
    """Run all weight update tests."""
    test_files = [
        "test_weight_update.py",
        "test_v1_worker_weight_update.py", 
        "test_api_server_weight_update.py"
    ]
    
    print("Running Weight Update Test Suite")
    print("=" * 50)
    print("These tests verify weight update functionality without requiring")
    print("a full vLLM engine startup or GPU/CUDA dependencies.")
    print()
    
    results = {}
    all_passed = True
    
    for test_file in test_files:
        if os.path.exists(test_file):
            success = run_test_file(test_file)
            results[test_file] = success
            if not success:
                all_passed = False
        else:
            print(f"⚠️  {test_file} - FILE NOT FOUND")
            results[test_file] = False
            all_passed = False
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    for test_file, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_file}")
    
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())