#!/usr/bin/env python3
"""
AutoCI Create/Fix 명령 디버그 도구
"""

import sys
import os
import traceback
import importlib.util
from pathlib import Path

print("=" * 60)
print("AutoCI Create/Fix 디버그 도구")
print("=" * 60)

# Python 정보
print(f"\n[Python 환경]")
print(f"Python 버전: {sys.version}")
print(f"Python 실행 파일: {sys.executable}")
print(f"현재 작업 디렉토리: {os.getcwd()}")
print(f"sys.path:")
for p in sys.path[:5]:  # 처음 5개만
    print(f"  - {p}")

# 프로젝트 루트 확인
script_dir = Path(__file__).parent
print(f"\n[프로젝트 경로]")
print(f"스크립트 디렉토리: {script_dir}")
print(f"autoci 파일 존재: {(script_dir / 'autoci').exists()}")
print(f"autoci.py 파일 존재: {(script_dir / 'autoci.py').exists()}")

# 필수 디렉토리 확인
print(f"\n[필수 디렉토리 확인]")
dirs_to_check = ['core_system', 'modules', 'modules_active']
for dir_name in dirs_to_check:
    dir_path = script_dir / dir_name
    print(f"{dir_name}: {'존재 ✓' if dir_path.exists() else '없음 ✗'}")
    if dir_path.exists():
        py_files = list(dir_path.glob('*.py'))
        print(f"  └─ Python 파일 수: {len(py_files)}")

# 필수 모듈 임포트 테스트
print(f"\n[모듈 임포트 테스트]")

# sys.path에 경로 추가
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / 'core_system'))
sys.path.insert(0, str(script_dir / 'modules'))

modules_to_test = [
    ('numpy', None),
    ('PIL', None),
    ('aiohttp', None),
    ('asyncio', None),
    ('core_system.autoci_panda3d_main', 'AutoCIPanda3DMain'),
    ('modules.game_session_manager', 'GameSessionManager'),
]

for module_name, class_name in modules_to_test:
    try:
        if '.' in module_name:
            # 로컬 모듈
            parts = module_name.split('.')
            file_path = script_dir / '/'.join(parts[:-1]) / f"{parts[-1]}.py"
            if file_path.exists():
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if class_name and hasattr(module, class_name):
                    print(f"✓ {module_name}.{class_name} - 성공")
                else:
                    print(f"✓ {module_name} - 성공")
            else:
                print(f"✗ {module_name} - 파일 없음: {file_path}")
        else:
            # 외부 패키지
            __import__(module_name)
            print(f"✓ {module_name} - 성공")
    except ImportError as e:
        print(f"✗ {module_name} - 실패: {str(e)}")
    except Exception as e:
        print(f"✗ {module_name} - 오류: {type(e).__name__}: {str(e)}")

# create 명령 시뮬레이션
print(f"\n[Create 명령 시뮬레이션]")
try:
    # autoci 파일 직접 실행
    autoci_path = script_dir / 'autoci'
    if autoci_path.exists():
        # sys.argv 설정
        original_argv = sys.argv
        sys.argv = ['autoci', 'create', 'platformer']
        
        with open(autoci_path, 'r', encoding='utf-8') as f:
            autoci_code = f.read()
        
        # create 명령 부분만 테스트
        print("create 명령 코드 실행 테스트...")
        # main 함수 찾기
        if 'def main():' in autoci_code:
            print("✓ main 함수 발견")
            # create_new_game 함수 테스트
            if 'async def create_new_game' in autoci_code:
                print("✓ create_new_game 함수 발견")
                # 임포트 테스트
                try:
                    exec("from core_system.autoci_panda3d_main import AutoCIPanda3DMain", globals())
                    print("✓ AutoCIPanda3DMain 임포트 성공")
                except Exception as e:
                    print(f"✗ AutoCIPanda3DMain 임포트 실패: {e}")
                    traceback.print_exc()
            else:
                print("✗ create_new_game 함수 없음")
        
        sys.argv = original_argv
    else:
        print("✗ autoci 파일이 없습니다")
except Exception as e:
    print(f"✗ 오류 발생: {type(e).__name__}: {str(e)}")
    traceback.print_exc()

# fix 명령 시뮬레이션
print(f"\n[Fix 명령 시뮬레이션]")
try:
    ai_engine_updater_path = script_dir / 'core_system' / 'ai_engine_updater.py'
    if ai_engine_updater_path.exists():
        print("✓ ai_engine_updater.py 파일 존재")
        # 임포트 테스트
        try:
            spec = importlib.util.spec_from_file_location("ai_engine_updater", ai_engine_updater_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("✓ ai_engine_updater 모듈 로드 성공")
        except Exception as e:
            print(f"✗ ai_engine_updater 모듈 로드 실패: {e}")
            traceback.print_exc()
    else:
        print("✗ ai_engine_updater.py 파일이 없습니다")
except Exception as e:
    print(f"✗ 오류 발생: {type(e).__name__}: {str(e)}")

print("\n" + "=" * 60)
print("진단 완료!")
print("위의 오류 메시지를 확인하여 문제를 해결하세요.")
print("=" * 60)