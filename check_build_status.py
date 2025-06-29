#!/usr/bin/env python3
"""
AI Godot 빌드 상태 확인 스크립트
"""
import os
import sys
import subprocess
from pathlib import Path
import shutil

def check_tool(tool_name, install_cmd=None):
    """도구 설치 확인"""
    if shutil.which(tool_name):
        print(f"✅ {tool_name} 설치됨")
        return True
    else:
        print(f"❌ {tool_name} 없음")
        if install_cmd:
            print(f"   설치: {install_cmd}")
        return False

def main():
    print("=" * 50)
    print("    AI Godot 빌드 환경 확인")
    print("=" * 50)
    print()
    
    # 필수 도구 확인
    print("🔍 필수 도구 확인:")
    tools_ok = True
    
    tools_ok &= check_tool("python3", "sudo apt install python3")
    tools_ok &= check_tool("pip3", "sudo apt install python3-pip")
    tools_ok &= check_tool("git", "sudo apt install git")
    
    # SCons 확인
    try:
        import SCons
        print("✅ SCons 설치됨")
    except:
        print("❌ SCons 없음")
        print("   설치: pip3 install scons")
        tools_ok = False
    
    print()
    
    # 빌드 파일 확인
    print("📁 빌드 파일 확인:")
    current_dir = Path.cwd()
    
    files_to_check = [
        "build_ai_godot.py",
        "BUILD_AI_GODOT.bat",
        "godot_ai_patches/README.md"
    ]
    
    for file in files_to_check:
        file_path = current_dir / file
        if file_path.exists():
            print(f"✅ {file} 존재")
        else:
            print(f"❌ {file} 없음")
    
    print()
    
    # 빌드 디렉토리 확인
    build_dir = current_dir / "godot_ai_build"
    if build_dir.exists():
        print(f"📦 빌드 디렉토리 존재: {build_dir}")
        
        # 빌드된 실행 파일 확인
        exe_path = build_dir / "output" / "godot.windows.editor.x86_64.exe"
        if exe_path.exists():
            print(f"✅ AI Godot 실행 파일 발견: {exe_path}")
        else:
            print("⏳ AI Godot 아직 빌드되지 않음")
    else:
        print("⏳ 빌드 디렉토리 없음 (빌드 전)")
    
    print()
    
    # 결과
    if tools_ok:
        print("✅ 빌드 환경 준비 완료!")
        print()
        print("다음 단계:")
        print("1. Windows에서: BUILD_AI_GODOT.bat 실행")
        print("2. WSL/Linux에서: python3 build_ai_godot.py 실행")
    else:
        print("❌ 필수 도구를 먼저 설치하세요")
    
    print()
    print("=" * 50)

if __name__ == "__main__":
    main()