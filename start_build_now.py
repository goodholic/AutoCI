#!/usr/bin/env python3
"""
AI Godot 즉시 빌드 시작
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 AI Godot 빌드를 시작합니다...")
    print("=" * 50)
    
    # 현재 디렉토리 확인
    current_dir = Path.cwd()
    build_script = current_dir / "build_ai_godot.py"
    
    if not build_script.exists():
        build_script = Path(__file__).parent / "build_ai_godot.py"
    
    print(f"빌드 스크립트: {build_script}")
    
    # 필요한 디렉토리 생성
    dirs_to_create = [
        "godot_ai_build",
        "godot_ai_build/output",
        "godot_ai_patches"
    ]
    
    for dir_name in dirs_to_create:
        dir_path = build_script.parent / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 디렉토리 생성: {dir_path}")
    
    # Python으로 직접 실행
    print("\n빌드 스크립트를 실행합니다...")
    cmd = [sys.executable, str(build_script)]
    
    try:
        # 빌드 실행
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 text=True,
                                 bufsize=1,
                                 cwd=str(build_script.parent))
        
        # 실시간 출력
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ 빌드 완료!")
        else:
            print(f"\n❌ 빌드 실패 (코드: {process.returncode})")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return 1
    
    return process.returncode

if __name__ == "__main__":
    sys.exit(main())