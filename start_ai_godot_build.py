#!/usr/bin/env python3
"""
AI Godot 빌드 시작 스크립트
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 AI Godot 빌드를 시작합니다...")
    
    # 현재 디렉토리 확인
    current_dir = Path.cwd()
    print(f"현재 디렉토리: {current_dir}")
    
    # build_ai_godot.py 찾기
    build_script = current_dir / "build_ai_godot.py"
    if not build_script.exists():
        build_script = Path(__file__).parent / "build_ai_godot.py"
    
    if not build_script.exists():
        print("❌ build_ai_godot.py를 찾을 수 없습니다.")
        return 1
    
    print(f"✅ 빌드 스크립트 발견: {build_script}")
    
    # Python으로 빌드 스크립트 실행
    try:
        result = subprocess.run([sys.executable, str(build_script)], 
                               cwd=build_script.parent)
        return result.returncode
    except Exception as e:
        print(f"❌ 빌드 실행 중 오류: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())