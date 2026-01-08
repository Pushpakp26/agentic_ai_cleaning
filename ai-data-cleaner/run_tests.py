#!/usr/bin/env python3
"""
Test runner script for AI Data Cleaner

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --verbose          # Run with verbose output
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode == 0

def main():
    """Main test runner"""
    args = sys.argv[1:]
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Parse arguments
    if "--unit" in args:
        cmd.extend(["tests/unit/", "-m", "unit"])
    elif "--integration" in args:
        cmd.extend(["tests/integration/", "-m", "integration"])
    else:
        cmd.append("tests/")
    
    # Add coverage if requested
    if "--coverage" in args:
        cmd.extend([
            "--cov=server",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add verbose if requested
    if "--verbose" in args or "-v" in args:
        cmd.append("-v")
    
    # Add other common options
    cmd.extend([
        "--tb=short"
    ])
    
    # Run the tests
    success = run_command(cmd)
    
    if success:
        print("\n✅ All tests passed!")
        if "--coverage" in args:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()