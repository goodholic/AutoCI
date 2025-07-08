#!/usr/bin/env python3
# This is a wrapper to ensure proper Python execution on Windows

import sys
import os

# Add the script directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import and run the main autoci script
if __name__ == "__main__":
    # Read and execute the autoci file
    autoci_path = os.path.join(script_dir, 'autoci')
    with open(autoci_path, 'r', encoding='utf-8') as f:
        autoci_code = f.read()
    
    # Execute the code
    exec(autoci_code)