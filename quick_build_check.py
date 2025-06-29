#!/usr/bin/env python3
"""
빠른 빌드 환경 체크
"""
import sys
import subprocess
import shutil
from pathlib import Path

def check_python():
    """Python 버전 확인"""
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    return version.major >= 3 and version.minor >= 8

def check_tool(name):
    """도구 설치 확인"""
    return shutil.which(name) is not None

def main():
    print("🔍 빠른 환경 체크...")
    print("-" * 30)
    
    checks = {
        "Python 3.8+": check_python(),
        "Git": check_tool("git"),
        "pip": check_tool("pip") or check_tool("pip3")
    }
    
    # SCons 체크
    try:
        subprocess.run([sys.executable, "-c", "import SCons"], 
                      capture_output=True, check=True)
        checks["SCons"] = True
    except:
        checks["SCons"] = False
    
    # 결과 출력
    all_ok = True
    for item, status in checks.items():
        emoji = "✅" if status else "❌"
        print(f"{emoji} {item}")
        all_ok &= status
    
    print("-" * 30)
    
    if all_ok:
        print("✅ 모든 체크 통과!")
        print("\n빌드를 시작하려면:")
        print("  python3 build_ai_godot.py")
    else:
        print("❌ 일부 요구사항 누락")
        if not checks["SCons"]:
            print("\nSCons 설치:")
            print("  pip3 install scons")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())