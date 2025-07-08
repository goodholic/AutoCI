#!/usr/bin/env python3
"""
AutoCI Windows Entry Point
This ensures proper execution on Windows systems
"""

import sys
import os
import subprocess

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the main autoci script
    autoci_script = os.path.join(script_dir, 'autoci')
    
    # Check if autoci exists
    if not os.path.exists(autoci_script):
        print(f"Error: Cannot find autoci script at {autoci_script}")
        return 1
    
    # Prepare arguments
    args = [sys.executable, autoci_script] + sys.argv[1:]
    
    # Debug info
    if os.environ.get('DEBUG'):
        print(f"Running: {' '.join(args)}")
    
    # Run the autoci script
    try:
        result = subprocess.run(args, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running autoci: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())