#!/usr/bin/env python3
"""
AutoCI - Cross-Platform AI Game Development System
Supports both WSL and Windows environments
"""

import sys
import os
import asyncio
import platform
import subprocess
from pathlib import Path

# Detect platform and set appropriate paths
def get_project_root():
    """Get project root path based on platform"""
    if platform.system() == "Windows":
        # Windows native path
        return Path(os.path.dirname(os.path.abspath(__file__)))
    else:
        # WSL/Linux path
        return Path("/mnt/d/AutoCI/AutoCI")

# Set project root
PROJECT_ROOT = get_project_root()

# Add paths to sys.path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'core_system'))
sys.path.insert(0, str(PROJECT_ROOT / 'modules'))
sys.path.insert(0, str(PROJECT_ROOT / 'modules_active'))

# Platform-specific imports
if platform.system() != "Windows":
    try:
        from core.xlib_suppressor import suppress_all_xlib_warnings
        suppress_all_xlib_warnings()
    except ImportError:
        pass  # Xlib not needed on Windows

def check_virtualenv():
    """Check if virtual environment is activated"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def get_python_executable():
    """Get the appropriate Python executable"""
    return sys.executable

def run_python_script(script_path, args=""):
    """Run Python script with cross-platform support"""
    python_exe = get_python_executable()
    cmd = f'"{python_exe}" "{script_path}" {args}'
    
    if platform.system() == "Windows":
        return subprocess.run(cmd, shell=True, capture_output=False)
    else:
        return os.system(cmd)

def main():
    """Main execution function"""
    # Virtual environment check
    if not check_virtualenv():
        print("⚠️  Virtual environment is not activated.")
        print("Please activate the virtual environment:")
        if platform.system() == "Windows":
            print("  autoci_env\\Scripts\\activate.bat  # Windows Command Prompt")
            print("  autoci_env\\Scripts\\Activate.ps1  # Windows PowerShell")
        else:
            print("  source autoci_env/bin/activate  # Linux/WSL")
        
        print("\nContinue without virtual environment? (y/N): ", end='')
        if input().lower() != 'y':
            sys.exit(1)
    
    # Process command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'learn':
            # AI learning mode
            script_path = PROJECT_ROOT / "core_system" / "continuous_learning_system.py"
            if len(sys.argv) > 2 and sys.argv[2] == 'low':
                print("🧠 Starting memory-optimized learning mode...")
                run_python_script(script_path, "--low-memory")
            else:
                print("🧠 Starting integrated learning mode...")
                run_python_script(script_path)
        
        elif command == 'monitor':
            # Monitoring dashboard
            print("📊 Starting monitoring dashboard...")
            script_path = PROJECT_ROOT / "modules" / "realtime_monitoring_system.py"
            run_python_script(script_path)
        
        elif command == 'fix':
            # Learning-based engine improvement
            print("🔧 Starting learning-based engine improvement...")
            script_path = PROJECT_ROOT / "core_system" / "ai_engine_updater.py"
            run_python_script(script_path)
        
        elif command == 'create':
            # Game creation mode
            if len(sys.argv) > 2:
                game_type = sys.argv[2]
                print(f"🎮 Creating {game_type} game...")
                asyncio.run(create_game(game_type))
            else:
                print("❌ Please specify game type: platformer, racing, rpg, puzzle")
        
        elif command == 'chat':
            # Korean conversation mode
            print("💬 Starting Korean conversation mode...")
            asyncio.run(chat_mode())
        
        elif command == '--help' or command == '-h':
            show_help()
        
        else:
            print(f"❌ Unknown command: {command}")
            show_help()
    
    else:
        # Default execution (interactive mode)
        try:
            asyncio.run(interactive_mode())
        except ImportError as e:
            print(f"❌ Module import error: {e}")
            print("\nPlease install required packages:")
            print("  pip install -r requirements.txt")
            sys.exit(1)

async def interactive_mode():
    """Interactive mode"""
    try:
        from core_system.autoci_panda3d_main import AutoCIPanda3DMain
        autoci = AutoCIPanda3DMain()
        await autoci.start()
    except ImportError as e:
        print(f"❌ Failed to load interactive mode: {e}")
        print("Running in command-line mode instead.")
        show_help()

async def create_game(game_type):
    """Create game with specified type"""
    try:
        from core_system.autoci_panda3d_main import AutoCIPanda3DMain
        autoci = AutoCIPanda3DMain()
        await autoci.create_game(game_type)
    except Exception as e:
        print(f"❌ Failed to create game: {e}")

async def chat_mode():
    """Korean conversation mode"""
    try:
        from modules.korean_conversation_interface import KoreanConversationInterface
        from modules.ai_model_integration import AIModelIntegration
    except ImportError as e:
        print(f"❌ Module import error: {e}")
        return
    
    ai_model = AIModelIntegration()
    chat_interface = KoreanConversationInterface(ai_model)
    
    print("\n💬 AutoCI Korean Conversation Mode")
    print("=" * 50)
    print("Chat in natural Korean. Type '종료' to exit.")
    print("=" * 50)
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['종료', 'exit', 'quit']:
            print("Ending conversation.")
            break
        
        response = await chat_interface.process_input(user_input)
        print(f"\nAutoCI: {response}")

def show_help():
    """Show help information"""
    print(f"""
AutoCI - Cross-Platform AI Game Development System v5.0
Platform: {platform.system()}

Usage:
  autoci                    Start interactive mode
  autoci learn              Start AI integrated learning
  autoci learn low          Memory-optimized learning (8GB VRAM)
  autoci monitor            Real-time monitoring dashboard
  autoci fix                Learning-based engine improvement
  autoci create [type]      Create game with specified type
  autoci chat               Korean conversation mode
  autoci --help             Show help

Game Types:
  platformer - Platform game
  racing     - Racing game
  rpg        - RPG game
  puzzle     - Puzzle game

Interactive Mode Commands:
  create [type] game        Create game
  add feature [name]        Add feature
  modify [aspect]           Modify game
  open_panda3d             Open Panda3D editor
  status                    System status
  help                      Help
  exit/quit/종료           Exit

Examples:
  autoci
  > create platformer game  # Create platform game
  > add feature double_jump # Add double jump feature
  > status                  # Check development status

Platform-Specific Notes:
""")
    
    if platform.system() == "Windows":
        print("  - Running on Windows")
        print("  - Use backslashes for paths: C:\\AutoCI\\AutoCI")
    else:
        print("  - Running on WSL/Linux")
        print("  - Use forward slashes for paths: /mnt/d/AutoCI/AutoCI")

if __name__ == "__main__":
    main()