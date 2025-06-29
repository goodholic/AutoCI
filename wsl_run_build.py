#!/usr/bin/env python3
"""
WSL에서 Windows 배치 파일 실행
"""
import os
import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 WSL에서 간단한 AI Godot 설정 시작...")
    print("=" * 50)
    
    # 현재 디렉토리
    current_dir = Path(__file__).parent
    bat_file = current_dir / "RUN_SIMPLE_BUILD.bat"
    
    if not bat_file.exists():
        print(f"❌ 배치 파일을 찾을 수 없습니다: {bat_file}")
        return 1
    
    # WSL 경로를 Windows 경로로 변환
    wsl_path = str(bat_file.absolute())
    
    # /mnt/d/ -> D:\ 변환
    if wsl_path.startswith('/mnt/'):
        drive = wsl_path[5]
        win_path = f"{drive.upper()}:{wsl_path[6:].replace('/', '\\')}"
    else:
        # wslpath 명령어 사용
        result = subprocess.run(['wslpath', '-w', wsl_path], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            win_path = result.stdout.strip()
        else:
            print("❌ Windows 경로 변환 실패")
            return 1
    
    print(f"Windows 경로: {win_path}")
    print("")
    print("Windows에서 빌드 스크립트 실행 중...")
    print("새 창이 열립니다. 완료될 때까지 기다려주세요...")
    print("")
    
    # cmd.exe를 통해 실행
    try:
        # 새 창에서 실행
        subprocess.run(['cmd.exe', '/c', 'start', win_path])
        
        print("✅ 빌드 스크립트가 실행되었습니다!")
        print("")
        print("빌드가 완료되면:")
        print("1. 이 터미널로 돌아와서")
        print("2. 'autoci' 명령어를 실행하세요")
        
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())