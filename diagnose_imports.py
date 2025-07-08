#!/usr/bin/env python3
"""
Diagnose import issues for AutoCI create and fix commands
"""

import sys
import importlib
import subprocess
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'core_system'))
sys.path.insert(0, str(PROJECT_ROOT / 'modules'))

print("AutoCI Import Diagnostics")
print("=" * 50)

# Critical modules for create/fix commands
critical_imports = [
    ('numpy', 'numpy'),
    ('PIL', 'pillow'),
    ('torch', 'torch'),
    ('transformers', 'transformers'),
    ('flask', 'flask'),
    ('flask_socketio', 'flask-socketio'),
    ('pandas', 'pandas'),
    ('aiohttp', 'aiohttp'),
    ('aiofiles', 'aiofiles'),
    ('psutil', 'psutil'),
    ('colorama', 'colorama'),
    ('rich', 'rich'),
    ('tqdm', 'tqdm'),
    ('yaml', 'pyyaml'),
    ('dotenv', 'python-dotenv'),
    ('screeninfo', 'screeninfo'),
    ('pynput', 'pynput'),
    ('cv2', 'opencv-python'),
    ('pythonnet', 'pythonnet'),
]

missing_packages = []

print("\nChecking critical imports:")
for module_name, package_name in critical_imports:
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} - OK")
    except ImportError:
        print(f"✗ {module_name} - MISSING (install with: pip install {package_name})")
        missing_packages.append(package_name)

print("\n" + "=" * 50)

# Try to import the main modules
print("\nChecking AutoCI modules:")

try:
    from modules.ai_model_integration import get_ai_integration
    print("✓ ai_model_integration - OK")
except ImportError as e:
    print(f"✗ ai_model_integration - ERROR: {e}")

try:
    from modules.panda3d_automation_controller import Panda3DAutomationController
    print("✓ panda3d_automation_controller - OK")
except ImportError as e:
    print(f"✗ panda3d_automation_controller - ERROR: {e}")

try:
    from core_system.autoci_panda3d_main import AutoCIPanda3DMain
    print("✓ autoci_panda3d_main - OK")
except ImportError as e:
    print(f"✗ autoci_panda3d_main - ERROR: {e}")

try:
    from core_system.ai_engine_updater import AIEngineUpdater
    print("✓ ai_engine_updater - OK")
except ImportError as e:
    print(f"✗ ai_engine_updater - ERROR: {e}")

print("\n" + "=" * 50)

if missing_packages:
    print("\nMISSING PACKAGES DETECTED!")
    print("\nTo fix the import errors, run ONE of these commands:")
    print("\n1. Install all requirements (recommended):")
    print(f"   py -m pip install -r requirements.txt")
    print("\n2. Install only missing packages:")
    print(f"   py -m pip install {' '.join(missing_packages)}")
    print("\n3. Install minimal set for create/fix:")
    print(f"   py -m pip install numpy pillow torch transformers flask flask-socketio aiohttp aiofiles psutil pyyaml python-dotenv screeninfo pynput opencv-python")
else:
    print("\nAll critical packages are installed!")
    print("\nIf you're still having issues, try:")
    print("1. Check if you're in the correct directory")
    print("2. Make sure Python can find the modules directory")
    print("3. Check for syntax errors in the Python files")

print("\nPython executable:", sys.executable)
print("Python version:", sys.version)
print("Current directory:", Path.cwd())
print("Project root:", PROJECT_ROOT)