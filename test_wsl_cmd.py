#!/usr/bin/env python3
"""
WSL에서 Windows 명령어 실행 테스트
"""
import subprocess
import os

def test_cmd():
    print("🧪 WSL → Windows 명령어 테스트")
    print("-" * 40)
    
    # 1. 간단한 echo 테스트
    print("1. Echo 테스트:")
    result = subprocess.run(['cmd.exe', '/c', 'echo Hello from Windows!'], 
                          capture_output=True, text=True)
    print(f"   결과: {result.stdout.strip()}")
    
    # 2. 현재 디렉토리 확인
    print("\n2. Windows 디렉토리:")
    result = subprocess.run(['cmd.exe', '/c', 'cd'], 
                          capture_output=True, text=True)
    print(f"   결과: {result.stdout.strip()}")
    
    # 3. Python 버전 확인
    print("\n3. Windows Python:")
    result = subprocess.run(['cmd.exe', '/c', 'python --version'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   결과: {result.stdout.strip()}")
    else:
        print("   결과: Python이 설치되지 않음")
    
    # 4. 배치 파일 실행 가능 여부
    print("\n4. 배치 파일 테스트:")
    test_bat = "echo Test successful!"
    result = subprocess.run(['cmd.exe', '/c', test_bat], 
                          capture_output=True, text=True)
    print(f"   결과: {result.stdout.strip()}")
    
    print("\n✅ 테스트 완료!")
    print("\nWSL에서 Windows 명령어를 실행할 수 있습니다.")
    print("'python3 wsl_run_build.py' 또는 'build-godot' 명령어를 사용하세요.")

if __name__ == "__main__":
    test_cmd()