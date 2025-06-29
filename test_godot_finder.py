#!/usr/bin/env python3
"""Godot 실행 파일 찾기 테스트"""

import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from modules.godot_realtime_dashboard import GodotRealtimeDashboard

def test_godot_finder():
    dashboard = GodotRealtimeDashboard()
    
    print("🔍 Godot 실행 파일을 찾는 중...")
    exe = dashboard.find_godot_executable()
    
    if exe:
        print(f"✅ Godot을 찾았습니다: {exe}")
        # 파일이 실제로 존재하는지 확인
        if Path(exe).exists():
            print("✅ 파일이 실제로 존재합니다.")
        else:
            print("❌ 경로는 반환되었지만 파일이 존재하지 않습니다.")
    else:
        print("❌ Godot을 찾을 수 없습니다.")
        
    # 현재 환경 정보 출력
    print("\n📊 환경 정보:")
    print(f"  - 현재 사용자: {os.environ.get('USER', 'unknown')}")
    print(f"  - WSL 여부: {'WSL_DISTRO_NAME' in os.environ}")
    print(f"  - 프로젝트 경로: {dashboard.project_root}")
    
    # godot_bin 디렉토리 확인
    godot_bin = dashboard.project_root / "godot_bin"
    if godot_bin.exists():
        print(f"\n📁 godot_bin 디렉토리 내용:")
        for file in godot_bin.iterdir():
            print(f"  - {file.name}")

if __name__ == "__main__":
    test_godot_finder()