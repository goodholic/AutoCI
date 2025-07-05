#!/usr/bin/env python3
"""
AutoCI Terminal Interface - Panda3D로 리다이렉트
"""

import os
import sys
import subprocess
from pathlib import Path

# Panda3D 터미널로 리다이렉트
def main():
    """Panda3D 터미널로 리다이렉트"""
    panda3d_terminal = Path(__file__).parent / "panda3d_terminal.py"
    
    if panda3d_terminal.exists():
        # Panda3D 터미널 실행
        print("🎮 Panda3D 게임 개발 모드로 전환합니다...")
        subprocess.run([sys.executable, str(panda3d_terminal)] + sys.argv[1:])
    else:
        print("❌ Panda3D 터미널을 찾을 수 없습니다.")
        print(f"   경로: {panda3d_terminal}")
        sys.exit(1)

if __name__ == "__main__":
    main()