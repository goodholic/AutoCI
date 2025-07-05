#!/usr/bin/env python3
"""
가상 입력 시스템 검증 도구
설치 상태와 기능을 체계적으로 확인합니다.
"""

import sys
import os
import asyncio
import time
from datetime import datetime
import platform
import json
from pathlib import Path

# 색상 코드 (터미널 출력용)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_status(message: str, status: str = "info"):
    """상태 메시지 출력"""
    if status == "success":
        print(f"{Colors.GREEN}✓{Colors.END} {message}")
    elif status == "error":
        print(f"{Colors.RED}✗{Colors.END} {message}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠{Colors.END} {message}")
    elif status == "info":
        print(f"{Colors.BLUE}ℹ{Colors.END} {message}")
    else:
        print(f"  {message}")


async def check_dependencies():
    """의존성 라이브러리 확인"""
    print(f"\n{Colors.BOLD}1. 의존성 라이브러리 확인{Colors.END}")
    print("-" * 50)
    
    dependencies = {
        "pyautogui": False,
        "pynput": False,
        "numpy": False,
        "pillow": False,
        "screeninfo": False
    }
    
    # 각 라이브러리 import 시도
    for lib in dependencies.keys():
        try:
            __import__(lib)
            dependencies[lib] = True
            print_status(f"{lib} 설치됨", "success")
        except ImportError:
            print_status(f"{lib} 설치 안됨", "error")
    
    # 플랫폼별 추가 확인
    if platform.system() == "Linux":
        try:
            import Xlib
            print_status("python-xlib 설치됨 (Linux)", "success")
        except ImportError:
            print_status("python-xlib 설치 안됨 (Linux)", "warning")
    
    elif platform.system() == "Windows":
        try:
            import win32api
            print_status("pywin32 설치됨 (Windows)", "success")
        except ImportError:
            print_status("pywin32 설치 안됨 (Windows)", "warning")
    
    # 설치 명령 제안
    missing = [lib for lib, installed in dependencies.items() if not installed]
    if missing:
        print_status(f"\n누락된 라이브러리 설치:", "info")
        print(f"  pip install {' '.join(missing)}")
    
    return all(dependencies.values())


async def test_virtual_input_import():
    """가상 입력 모듈 import 테스트"""
    print(f"\n{Colors.BOLD}2. 가상 입력 모듈 로드 테스트{Colors.END}")
    print("-" * 50)
    
    try:
        from modules.virtual_input_controller import (
            VirtualInputController, 
            get_virtual_input, 
            InputMode,
            VirtualScreen
        )
        print_status("virtual_input_controller 모듈 로드 성공", "success")
        
        # 클래스 확인
        print_status("VirtualInputController 클래스 확인", "success")
        print_status("InputMode 열거형 확인", "success")
        print_status("VirtualScreen 데이터클래스 확인", "success")
        
        return True
    except Exception as e:
        print_status(f"모듈 로드 실패: {e}", "error")
        return False


async def test_virtual_input_initialization():
    """가상 입력 초기화 테스트"""
    print(f"\n{Colors.BOLD}3. 가상 입력 초기화 테스트{Colors.END}")
    print("-" * 50)
    
    try:
        from modules.virtual_input_controller import get_virtual_input
        
        # 싱글톤 인스턴스 생성
        virtual_input = get_virtual_input()
        print_status("가상 입력 인스턴스 생성 성공", "success")
        
        # 속성 확인
        print_status(f"현재 모드: {virtual_input.mode.value}", "info")
        print_status(f"가상 스크린: {virtual_input.virtual_screen.width}x{virtual_input.virtual_screen.height}", "info")
        print_status(f"활성 상태: {'활성' if virtual_input.is_active else '비활성'}", "info")
        
        # 매크로 확인
        print_status(f"사전 정의 매크로: {len(virtual_input.macro_library)}개", "info")
        for macro_name in list(virtual_input.macro_library.keys())[:5]:
            print(f"    - {macro_name}")
        
        return virtual_input
    except Exception as e:
        print_status(f"초기화 실패: {e}", "error")
        return None


async def test_basic_functions(virtual_input):
    """기본 기능 테스트"""
    print(f"\n{Colors.BOLD}4. 기본 기능 테스트{Colors.END}")
    print("-" * 50)
    
    if not virtual_input:
        print_status("가상 입력 인스턴스가 없습니다", "error")
        return False
    
    # 활성화 테스트
    try:
        await virtual_input.activate()
        print_status("가상 입력 활성화 성공", "success")
    except Exception as e:
        print_status(f"활성화 실패: {e}", "error")
        return False
    
    # 기능별 테스트
    tests = [
        ("마우스 이동", lambda: virtual_input.move_mouse(100, 100, 0.1)),
        ("텍스트 입력", lambda: virtual_input.type_text("test", 0.01)),
        ("키 입력", lambda: virtual_input.press_key("a")),
        ("단축키", lambda: virtual_input.hotkey("ctrl", "a")),
        ("매크로 실행", lambda: virtual_input.execute_macro("godot_save")),
    ]
    
    success_count = 0
    for test_name, test_func in tests:
        try:
            print(f"  테스트: {test_name}...", end="")
            await test_func()
            print(f" {Colors.GREEN}✓{Colors.END}")
            success_count += 1
        except Exception as e:
            print(f" {Colors.RED}✗{Colors.END} ({e})")
    
    # 비활성화
    await virtual_input.deactivate()
    print_status("가상 입력 비활성화 성공", "success")
    
    return success_count == len(tests)


async def test_godot_integration():
    """Godot 통합 기능 테스트"""
    print(f"\n{Colors.BOLD}5. Godot 통합 기능 테스트{Colors.END}")
    print("-" * 50)
    
    try:
        from modules.virtual_input_controller import get_virtual_input, InputMode
        virtual_input = get_virtual_input()
        
        # Godot 모드 설정
        virtual_input.set_mode(InputMode.GODOT_EDITOR)
        print_status("Godot 에디터 모드 설정", "success")
        
        # Godot 전용 메서드 확인
        godot_methods = [
            "godot_create_node",
            "godot_add_script",
            "execute_macro"
        ]
        
        for method in godot_methods:
            if hasattr(virtual_input, method):
                print_status(f"{method} 메서드 확인", "success")
            else:
                print_status(f"{method} 메서드 없음", "error")
        
        return True
    except Exception as e:
        print_status(f"Godot 통합 테스트 실패: {e}", "error")
        return False


async def test_action_recording():
    """액션 기록 기능 테스트"""
    print(f"\n{Colors.BOLD}6. 액션 기록/재생 테스트{Colors.END}")
    print("-" * 50)
    
    try:
        from modules.virtual_input_controller import get_virtual_input
        virtual_input = get_virtual_input()
        
        await virtual_input.activate()
        
        # 몇 가지 액션 수행
        print_status("테스트 액션 수행 중...", "info")
        await virtual_input.move_mouse(200, 200, 0.1)
        await virtual_input.type_text("hello", 0.01)
        await virtual_input.press_key("enter")
        
        # 액션 히스토리 확인
        history = virtual_input.get_action_history()
        print_status(f"기록된 액션: {len(history)}개", "info")
        
        # 최근 액션 표시
        for action in history[-3:]:
            print(f"    - {action['type']}: {action.get('data', {})}")
        
        # 재생 테스트
        print_status("액션 재생 테스트...", "info")
        await virtual_input.replay_actions(history[-3:])
        print_status("액션 재생 완료", "success")
        
        await virtual_input.deactivate()
        return True
    except Exception as e:
        print_status(f"액션 기록 테스트 실패: {e}", "error")
        return False


async def test_complex_learning_integration():
    """복합 학습 통합 테스트"""
    print(f"\n{Colors.BOLD}7. 복합 학습 시스템 통합 테스트{Colors.END}")
    print("-" * 50)
    
    try:
        from modules.complex_learning_integration import get_complex_learning
        
        complex_learning = get_complex_learning()
        print_status("복합 학습 시스템 로드 성공", "success")
        
        # 컴포넌트 확인
        components = [
            ("godot_learning", "Godot 조작 학습"),
            ("virtual_input", "가상 입력 컨트롤러"),
            ("continuous_learning", "연속 학습 시스템"),
            ("game_pipeline", "게임 개발 파이프라인")
        ]
        
        for attr, name in components:
            if hasattr(complex_learning, attr):
                print_status(f"{name} 연결됨", "success")
            else:
                print_status(f"{name} 연결 안됨", "error")
        
        return True
    except Exception as e:
        print_status(f"복합 학습 통합 테스트 실패: {e}", "error")
        return False


async def generate_diagnostic_report():
    """진단 보고서 생성"""
    print(f"\n{Colors.BOLD}8. 진단 보고서 생성{Colors.END}")
    print("-" * 50)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version()
        },
        "test_results": {},
        "recommendations": []
    }
    
    # 테스트 결과 수집
    print_status("진단 정보 수집 중...", "info")
    
    # 환경 변수 확인
    env_vars = ["DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE"]
    report["environment"] = {}
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        report["environment"][var] = value
        if value != "Not set":
            print(f"    {var}: {value}")
    
    # 권한 확인
    if platform.system() == "Linux":
        try:
            import subprocess
            result = subprocess.run(["groups"], capture_output=True, text=True)
            groups = result.stdout.strip()
            report["linux_groups"] = groups
            print(f"    사용자 그룹: {groups}")
            
            if "input" not in groups:
                report["recommendations"].append(
                    "Linux에서 입력 장치 접근을 위해 'input' 그룹에 추가 필요: sudo usermod -a -G input $USER"
                )
        except:
            pass
    
    # 보고서 저장
    report_path = Path("virtual_input_diagnostic_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print_status(f"진단 보고서 생성: {report_path}", "success")
    
    # 권장 사항 출력
    if report["recommendations"]:
        print(f"\n{Colors.YELLOW}권장 사항:{Colors.END}")
        for rec in report["recommendations"]:
            print(f"  • {rec}")


async def main():
    """메인 검증 프로세스"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("=" * 60)
    print("       AutoCI 가상 입력 시스템 검증 도구")
    print("=" * 60)
    print(f"{Colors.END}")
    
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"플랫폼: {platform.system()} {platform.release()}")
    print(f"Python 버전: {platform.python_version()}")
    
    # 검증 단계 실행
    all_passed = True
    
    # 1. 의존성 확인
    deps_ok = await check_dependencies()
    all_passed &= deps_ok
    
    # 2. 모듈 import
    import_ok = await test_virtual_input_import()
    all_passed &= import_ok
    
    if import_ok:
        # 3. 초기화 테스트
        virtual_input = await test_virtual_input_initialization()
        all_passed &= (virtual_input is not None)
        
        if virtual_input:
            # 4. 기본 기능 테스트
            basic_ok = await test_basic_functions(virtual_input)
            all_passed &= basic_ok
            
            # 5. Godot 통합 테스트
            godot_ok = await test_godot_integration()
            all_passed &= godot_ok
            
            # 6. 액션 기록 테스트
            recording_ok = await test_action_recording()
            all_passed &= recording_ok
        
        # 7. 복합 학습 통합 테스트
        complex_ok = await test_complex_learning_integration()
        all_passed &= complex_ok
    
    # 8. 진단 보고서 생성
    await generate_diagnostic_report()
    
    # 최종 결과
    print(f"\n{Colors.BOLD}최종 검증 결과{Colors.END}")
    print("=" * 60)
    
    if all_passed:
        print(f"{Colors.GREEN}✅ 모든 테스트 통과! 가상 입력 시스템이 정상 작동합니다.{Colors.END}")
    else:
        print(f"{Colors.RED}❌ 일부 테스트 실패. 위의 오류를 확인하세요.{Colors.END}")
    
    print(f"\n검증 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    print("⚠️  주의: 이 검증 도구는 마우스와 키보드를 제어할 수 있습니다.")
    print("   검증 중에는 마우스/키보드를 건드리지 마세요.")
    response = input("\n계속하시겠습니까? (y/N): ")
    
    if response.lower() == 'y':
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}검증이 중단되었습니다.{Colors.END}")
    else:
        print("검증을 취소했습니다.")