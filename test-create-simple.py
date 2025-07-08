#!/usr/bin/env python3
"""간단한 create 명령 테스트"""

import sys
import os

print("=== AutoCI Create Test ===")
print(f"Python: {sys.version}")
print(f"작업 디렉토리: {os.getcwd()}")

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'core_system'))

print(f"\nsys.path에 추가된 경로:")
for p in sys.path[:3]:
    print(f"  - {p}")

# 필수 파일 확인
print(f"\n필수 파일 확인:")
autoci_panda3d_path = os.path.join(script_dir, 'core_system', 'autoci_panda3d_main.py')
print(f"autoci_panda3d_main.py 존재: {os.path.exists(autoci_panda3d_path)}")

# 임포트 시도
print(f"\n임포트 시도:")
try:
    import asyncio
    print("✓ asyncio 임포트 성공")
except Exception as e:
    print(f"✗ asyncio 임포트 실패: {e}")

try:
    from core_system.autoci_panda3d_main import AutoCIPanda3DMain
    print("✓ AutoCIPanda3DMain 임포트 성공!")
    print("\nCREATE 명령이 작동할 것입니다!")
except ImportError as e:
    print(f"✗ AutoCIPanda3DMain 임포트 실패: {e}")
    print("\n필요한 패키지가 누락되었을 수 있습니다:")
    print("  py -m pip install numpy pillow aiohttp aiofiles")
except Exception as e:
    print(f"✗ 예상치 못한 오류: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

input("\n엔터를 눌러 종료...")