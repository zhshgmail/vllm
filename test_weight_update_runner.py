#!/usr/bin/env python3
"""
Test runner for weight update with interruption functionality.
Run this to verify the new functionality works correctly.
"""

import subprocess
import sys
import os


def run_test_file(test_file: str) -> bool:
    """Run a single test file and return True if all tests pass."""
    print(f"\n{'='*60}")
    print(f"Running tests in {test_file}")
    print('='*60)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_file, 
            "-v",  # verbose output
            "--tb=short",  # shorter traceback format
            "--no-header",  # skip pytest header
        ], capture_output=False, check=True)
        
        print(f"✅ All tests in {test_file} PASSED")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests in {test_file} FAILED (exit code: {e.returncode})")
        return False
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False


def main():
    """Run all weight update tests."""
    print("🧪 Running Weight Update with Interruption Tests")
    print("=" * 60)
    
    test_files = [
        "tests/test_weight_update_with_interrupt.py",
        "tests/integration_test_weight_update_streaming.py"
    ]
    
    # Check if test files exist
    missing_files = []
    for test_file in test_files:
        if not os.path.exists(test_file):
            missing_files.append(test_file)
    
    if missing_files:
        print("❌ Missing test files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nMake sure you're running this from the vLLM root directory.")
        return 1
    
    # Run tests
    all_passed = True
    for test_file in test_files:
        passed = run_test_file(test_file)
        all_passed &= passed
    
    # Summary
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe weight update with interruption functionality is working correctly:")
        print("  ✅ OutputProcessor.finalize_and_abort_all() creates proper abort outputs")
        print("  ✅ AsyncLLM.abort_all_active() handles active request interruption")
        print("  ✅ /update-weights-from-disk endpoint supports interrupt flag")
        print("  ✅ Streaming requests receive partial content before abort")
        print("  ✅ Multiple concurrent streams are handled correctly")
        print("  ✅ Error cases are handled gracefully")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("\nPlease review the test output above to identify issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
