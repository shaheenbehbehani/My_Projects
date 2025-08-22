#!/usr/bin/env python3
"""
Test Runner for Premier League Data Quality Tests

Simple script to run all data quality and schema validation tests.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the test suite."""
    project_root = Path(__file__).parent
    test_file = project_root / "tests" / "test_schema.py"
    
    print("🏈 Running Premier League Data Quality Tests")
    print("=" * 50)
    
    # Run pytest with nice formatting
    cmd = [
        sys.executable, "-m", "pytest", 
        str(test_file),
        "-v",           # Verbose output
        "--tb=short",   # Short traceback format
        "--color=yes"   # Colored output
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 