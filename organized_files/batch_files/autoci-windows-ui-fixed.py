#!/usr/bin/env python3
"""
AutoCI Windows UI Launcher - Fixed Version
임시 스크립트를 사용하여 create/fix 문제 해결
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add color support for Windows
try:
    import colorama
    colorama.init()
    from colorama import Fore, Back, Style
except ImportError:
    # Fallback if colorama not installed
    class Fore:
        GREEN = YELLOW = CYAN = RED = BLUE = MAGENTA = ''
        RESET = ''
    Style = Fore

def clear_screen():
    """Clear console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print AutoCI banner"""
    clear_screen()
    print(f"{Fore.CYAN}╔═══════════════════════════════════════════════════════╗")
    print(f"║                                                       ║")
    print(f"║{Fore.YELLOW}      🎮 AutoCI - AI Game Development System 🎮       {Fore.CYAN}║")
    print(f"║{Fore.GREEN}         Windows Edition v5.0 (Fixed)                  {Fore.CYAN}║")
    print(f"║                                                       ║")
    print(f"╚═══════════════════════════════════════════════════════╝{Fore.RESET}")
    print()

def run_command(cmd):
    """Run a command and show output"""
    try:
        # Set environment to skip venv check
        env = os.environ.copy()
        env['AUTOCI_SKIP_VENV_CHECK'] = '1'
        
        # Run the command
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env
        )
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        return process.returncode
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Fore.RESET}")
        return 1

def check_and_fix_modules():
    """Check if modules are working and offer to use stub versions"""
    print(f"{Fore.YELLOW}모듈 상태 확인 중...{Fore.RESET}")
    
    # Test create import
    try:
        test_cmd = 'py -c "from core_system.autoci_panda3d_main import AutoCIPanda3DMain"'
        result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{Fore.RED}✗ Create/Fix 모듈 임포트 실패{Fore.RESET}")
            print(f"{Fore.YELLOW}임시 해결책을 사용합니다.{Fore.RESET}")
            return False
        else:
            print(f"{Fore.GREEN}✓ 모듈 정상 작동{Fore.RESET}")
            return True
    except:
        return False

def main_menu():
    """Show main menu"""
    # Check module status once at startup
    modules_ok = check_and_fix_modules()
    
    while True:
        print_banner()
        
        if not modules_ok:
            print(f"{Fore.YELLOW}⚠️  임시 모드로 실행 중 (일부 기능 제한){Fore.RESET}\n")
        
        print(f"{Fore.GREEN}Main Menu:{Fore.RESET}")
        print(f"{Fore.YELLOW}1.{Fore.RESET} 🧠 Learn - Start AI Learning")
        print(f"{Fore.YELLOW}2.{Fore.RESET} 🎮 Create - Create/Resume Game")
        print(f"{Fore.YELLOW}3.{Fore.RESET} 🔧 Fix - Fix Engine Based on Learning")
        print(f"{Fore.YELLOW}4.{Fore.RESET} 💬 Chat - Korean Chat Mode")
        print(f"{Fore.YELLOW}5.{Fore.RESET} 📊 Sessions - View All Sessions")
        print(f"{Fore.YELLOW}6.{Fore.RESET} 🔄 Resume - Resume Paused Game")
        print(f"{Fore.YELLOW}7.{Fore.RESET} 📦 Install Requirements")
        print(f"{Fore.YELLOW}8.{Fore.RESET} 🐛 Debug - 문제 진단")
        print(f"{Fore.YELLOW}9.{Fore.RESET} ❌ Exit")
        print()
        
        choice = input(f"{Fore.CYAN}Select option (1-9): {Fore.RESET}").strip()
        
        if choice == '1':
            print(f"\n{Fore.GREEN}Starting AI Learning...{Fore.RESET}\n")
            run_command('py autoci learn')
            input(f"\n{Fore.YELLOW}Press Enter to continue...{Fore.RESET}")
            
        elif choice == '2':
            print(f"\n{Fore.GREEN}Starting Game Creation...{Fore.RESET}\n")
            if modules_ok:
                run_command('py autoci create')
            else:
                # Use stub version
                run_command('py create-stub.py')
            input(f"\n{Fore.YELLOW}Press Enter to continue...{Fore.RESET}")
            
        elif choice == '3':
            print(f"\n{Fore.GREEN}Starting Engine Fix...{Fore.RESET}\n")
            if modules_ok:
                run_command('py autoci fix')
            else:
                # Use stub version
                run_command('py fix-stub.py')
            input(f"\n{Fore.YELLOW}Press Enter to continue...{Fore.RESET}")
            
        elif choice == '4':
            print(f"\n{Fore.GREEN}Starting Korean Chat Mode...{Fore.RESET}\n")
            run_command('py autoci chat')
            input(f"\n{Fore.YELLOW}Press Enter to continue...{Fore.RESET}")
            
        elif choice == '5':
            print(f"\n{Fore.GREEN}Showing All Sessions...{Fore.RESET}\n")
            run_command('py autoci sessions')
            input(f"\n{Fore.YELLOW}Press Enter to continue...{Fore.RESET}")
            
        elif choice == '6':
            print(f"\n{Fore.GREEN}Resuming Paused Game...{Fore.RESET}\n")
            run_command('py autoci resume')
            input(f"\n{Fore.YELLOW}Press Enter to continue...{Fore.RESET}")
            
        elif choice == '7':
            print(f"\n{Fore.GREEN}Installing Requirements...{Fore.RESET}\n")
            print("Select installation option:")
            print("1. Install ALL requirements (recommended)")
            print("2. Install minimal requirements for create/fix")
            print("3. Install only UI requirements")
            
            install_choice = input("Choice (1-3): ").strip()
            
            if install_choice == '1':
                if os.path.exists('requirements.txt'):
                    run_command('py -m pip install -r requirements.txt')
                else:
                    print(f"{Fore.RED}requirements.txt not found!{Fore.RESET}")
            elif install_choice == '2':
                run_command('py -m pip install numpy pillow aiohttp aiofiles psutil pyyaml python-dotenv screeninfo pynput opencv-python')
            elif install_choice == '3':
                run_command('py -m pip install colorama')
            
            input(f"\n{Fore.YELLOW}Press Enter to continue...{Fore.RESET}")
            
        elif choice == '8':
            print(f"\n{Fore.GREEN}Running Diagnostics...{Fore.RESET}\n")
            run_command('py debug-create-fix.py')
            input(f"\n{Fore.YELLOW}Press Enter to continue...{Fore.RESET}")
            
        elif choice == '9':
            print(f"\n{Fore.GREEN}Thank you for using AutoCI!{Fore.RESET}")
            break
            
        else:
            print(f"\n{Fore.RED}Invalid option. Please try again.{Fore.RESET}")
            time.sleep(1)

if __name__ == "__main__":
    try:
        # Change to script directory
        os.chdir(Path(__file__).parent)
        main_menu()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Interrupted by user.{Fore.RESET}")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Fore.RESET}")
        input("Press Enter to exit...")