#!/usr/bin/env python3
"""
Godot 자동화 시스템 테스트 스크립트
AutoCI가 Godot을 제어하는 기능을 테스트
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
import subprocess

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "modules"))

# 색상 출력을 위한 ANSI 코드
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """헤더 출력"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_step(text):
    """단계 출력"""
    print(f"{Colors.OKCYAN}▶ {text}{Colors.ENDC}")

def print_success(text):
    """성공 메시지"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    """에러 메시지"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_warning(text):
    """경고 메시지"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def check_requirements():
    """필요한 라이브러리 확인"""
    print_header("요구사항 확인")
    
    requirements = {
        "opencv-python": "cv2",
        "pyautogui": "pyautogui",
        "pillow": "PIL",
        "pytesseract": "pytesseract",
        "torch": "torch",
        "mss": "mss"
    }
    
    missing = []
    
    for package, module in requirements.items():
        try:
            __import__(module)
            print_success(f"{package} 설치됨")
        except ImportError:
            print_error(f"{package} 미설치")
            missing.append(package)
    
    # Windows 전용 모듈 확인
    if sys.platform == "win32":
        try:
            import win32api
            print_success("pywin32 설치됨")
        except ImportError:
            print_error("pywin32 미설치")
            missing.append("pywin32")
    
    if missing:
        print(f"\n{Colors.WARNING}다음 명령어로 필요한 패키지를 설치하세요:{Colors.ENDC}")
        print(f"{Colors.BOLD}pip install {' '.join(missing)}{Colors.ENDC}")
        
        # Tesseract 설치 안내
        if "pytesseract" in [m.split('-')[0] for m in missing]:
            print(f"\n{Colors.WARNING}Tesseract OCR도 별도로 설치해야 합니다:{Colors.ENDC}")
            print("Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            print("설치 후 환경변수 PATH에 추가하거나 pytesseract.pytesseract.tesseract_cmd 설정 필요")
        
        return False
    
    return True

def test_screen_capture():
    """화면 캡처 테스트"""
    print_header("화면 캡처 테스트")
    
    try:
        from modules.godot_automation_system import GodotScreenRecognizer
        
        recognizer = GodotScreenRecognizer()
        print_step("화면 캡처 시도 중...")
        
        screenshot = recognizer.capture_screen()
        if screenshot is not None:
            print_success(f"화면 캡처 성공! 크기: {screenshot.shape}")
            
            # 테스트 이미지 저장
            import cv2
            test_path = Path("test_screenshot.png")
            cv2.imwrite(str(test_path), screenshot)
            print_success(f"테스트 스크린샷 저장됨: {test_path}")
            
            return True
        else:
            print_error("화면 캡처 실패")
            return False
            
    except Exception as e:
        print_error(f"오류 발생: {e}")
        return False

def test_virtual_input():
    """가상 입력 테스트"""
    print_header("가상 입력 테스트")
    
    try:
        from modules.godot_automation_system import VirtualInputController
        
        controller = VirtualInputController()
        
        print_warning("5초 후 마우스가 자동으로 움직입니다. 테스트를 중단하려면 마우스를 화면 왼쪽 상단으로 이동하세요.")
        time.sleep(5)
        
        print_step("마우스 이동 테스트...")
        import pyautogui
        current_x, current_y = pyautogui.position()
        controller.move_mouse(current_x + 100, current_y + 100, duration=1)
        print_success("마우스 이동 완료")
        
        time.sleep(1)
        
        print_step("키보드 입력 테스트 준비...")
        print_warning("메모장이나 텍스트 에디터를 열고 포커스를 맞춰주세요. 5초 후 시작합니다.")
        time.sleep(5)
        
        test_text = "AutoCI Godot Test"
        controller.type_text(test_text, interval=0.1)
        print_success("키보드 입력 완료")
        
        return True
        
    except Exception as e:
        print_error(f"오류 발생: {e}")
        return False

def test_godot_detection():
    """Godot 창 감지 테스트"""
    print_header("Godot 창 감지 테스트")
    
    if sys.platform != "win32":
        print_warning("이 테스트는 Windows에서만 작동합니다.")
        return False
    
    try:
        from modules.advanced_godot_controller import GodotAutomationController
        
        controller = GodotAutomationController()
        
        if controller.godot_hwnd:
            print_success("Godot 창을 찾았습니다!")
            controller.focus_godot_window()
            print_success("Godot 창에 포커스를 맞췄습니다.")
            return True
        else:
            print_warning("Godot 창을 찾을 수 없습니다. Godot이 실행 중인지 확인하세요.")
            return False
            
    except Exception as e:
        print_error(f"오류 발생: {e}")
        return False

def test_simple_automation():
    """간단한 자동화 테스트"""
    print_header("간단한 자동화 시나리오")
    
    try:
        from modules.realtime_godot_automation import AutoCICreateController
        
        print_step("AutoCI Create 컨트롤러 초기화...")
        controller = AutoCICreateController()
        
        print_warning("이 테스트는 실제로 Godot을 제어합니다.")
        print_warning("Godot이 실행 중이고 빈 프로젝트가 열려있는지 확인하세요.")
        
        response = input("\n계속하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("테스트 취소됨")
            return False
        
        print_step("자동화 시스템 시작...")
        controller.start()
        
        print_step("간단한 2D 씬 생성 시도...")
        
        # 더미 작업 생성 (실제 실행은 Godot이 열려있을 때만)
        from modules.realtime_godot_automation import AutomationTask
        
        simple_task = AutomationTask(
            task_id="test_task_1",
            task_type="create_simple_scene",
            description="간단한 테스트 씬 생성",
            steps=[
                {
                    "type": "create_node",
                    "node_type": "Node2D"
                }
            ]
        )
        
        controller.executor.add_task(simple_task)
        print_success("작업이 큐에 추가되었습니다.")
        
        print("\n10초간 실행을 기다립니다...")
        time.sleep(10)
        
        controller.stop()
        print_success("자동화 시스템 중지됨")
        
        return True
        
    except Exception as e:
        print_error(f"오류 발생: {e}")
        return False

def create_demo_script():
    """데모 스크립트 생성"""
    print_header("데모 스크립트 생성")
    
    demo_path = Path("demo_godot_automation.py")
    
    demo_content = '''#!/usr/bin/env python3
"""
Godot 자동화 데모
이 스크립트는 AutoCI가 Godot을 제어하는 방법을 보여줍니다.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from modules.realtime_godot_automation import AutoCICreateController

def main():
    print("🎮 AutoCI Godot 자동화 데모")
    print("=" * 50)
    
    # 컨트롤러 생성
    controller = AutoCICreateController()
    
    print("\\n사용 가능한 기능:")
    print("1. 2D 플랫포머 플레이어 생성")
    print("2. UI 메뉴 생성")
    print("3. 커스텀 작업")
    
    choice = input("\\n선택하세요 (1-3): ")
    
    try:
        controller.start()
        
        if choice == "1":
            print("\\n2D 플랫포머 플레이어를 생성합니다...")
            task_id = controller.create_2d_platformer_player()
            print(f"작업 ID: {task_id}")
            
        elif choice == "2":
            print("\\nUI 메뉴를 생성합니다...")
            task_id = controller.create_ui_menu()
            print(f"작업 ID: {task_id}")
            
        elif choice == "3":
            node_type = input("생성할 노드 타입: ")
            
            from modules.realtime_godot_automation import AutomationTask
            
            custom_task = AutomationTask(
                task_id=f"custom_{int(time.time())}",
                task_type="custom",
                description=f"{node_type} 노드 생성",
                steps=[
                    {
                        "type": "create_node",
                        "node_type": node_type
                    }
                ]
            )
            
            controller.executor.add_task(custom_task)
            print(f"커스텀 작업이 추가되었습니다: {custom_task.task_id}")
        
        print("\\n작업이 실행되는 동안 30초간 대기합니다...")
        print("(Godot 창을 확인하세요)")
        
        time.sleep(30)
        
    finally:
        controller.stop()
        print("\\n자동화 시스템이 중지되었습니다.")

if __name__ == "__main__":
    main()
'''
    
    demo_path.write_text(demo_content, encoding='utf-8')
    print_success(f"데모 스크립트 생성됨: {demo_path}")
    
    return True

def create_test_templates():
    """테스트용 템플릿 이미지 생성 안내"""
    print_header("템플릿 이미지 설정")
    
    template_dir = Path("modules/templates")
    template_dir.mkdir(exist_ok=True)
    
    print(f"템플릿 디렉토리: {template_dir.absolute()}")
    
    needed_templates = [
        "file_menu.png - File 메뉴 스크린샷",
        "scene_panel.png - Scene 패널 스크린샷",
        "inspector.png - Inspector 패널 스크린샷",
        "node_button.png - Add Node 버튼 스크린샷",
        "script_editor.png - 스크립트 에디터 스크린샷",
        "play_button.png - 실행 버튼 스크린샷",
        "save_button.png - 저장 버튼 스크린샷"
    ]
    
    print("\n다음 템플릿 이미지가 필요합니다:")
    for template in needed_templates:
        print(f"  • {template}")
    
    print(f"\n{Colors.WARNING}Godot을 열고 각 UI 요소의 스크린샷을 찍어서")
    print(f"{template_dir.absolute()} 폴더에 저장하세요.{Colors.ENDC}")
    
    # 템플릿 README 생성
    readme_path = template_dir / "README.md"
    readme_content = """# Godot UI 템플릿 이미지

이 폴더에는 화면 인식을 위한 템플릿 이미지가 필요합니다.

## 필요한 템플릿

1. **file_menu.png** - File 메뉴 텍스트
2. **scene_panel.png** - Scene 도크 헤더
3. **inspector.png** - Inspector 도크 헤더
4. **node_button.png** - Add Node 버튼 (+아이콘)
5. **script_editor.png** - 스크립트 에디터 탭
6. **play_button.png** - 재생 버튼 (▶)
7. **save_button.png** - 저장 버튼 아이콘

## 템플릿 만들기

1. Godot 에디터를 엽니다
2. 각 UI 요소를 Windows 캡처 도구로 캡처합니다
3. 배경을 포함하지 않고 아이콘/텍스트만 정확히 잘라냅니다
4. PNG 형식으로 이 폴더에 저장합니다

## 팁

- 고해상도로 캡처하되, 파일 크기는 작게 유지하세요
- 가능한 한 깨끗하고 선명한 이미지를 사용하세요
- Godot의 테마가 바뀌면 템플릿도 업데이트해야 합니다
"""
    
    readme_path.write_text(readme_content, encoding='utf-8')
    print_success(f"템플릿 가이드 생성됨: {readme_path}")
    
    return True

def main():
    """메인 테스트 함수"""
    print_header("AutoCI Godot 자동화 테스트")
    
    print("이 스크립트는 AutoCI가 Godot을 제어하는 기능을 테스트합니다.")
    print(f"{Colors.WARNING}주의: 일부 테스트는 실제로 마우스와 키보드를 제어합니다.{Colors.ENDC}")
    
    # 1. 요구사항 확인
    if not check_requirements():
        print(f"\n{Colors.FAIL}필요한 패키지를 먼저 설치하세요.{Colors.ENDC}")
        return
    
    # 2. 테스트 메뉴
    while True:
        print("\n" + "="*60)
        print("테스트 메뉴:")
        print("1. 화면 캡처 테스트")
        print("2. 가상 입력 테스트 (마우스/키보드)")
        print("3. Godot 창 감지 테스트 (Windows)")
        print("4. 간단한 자동화 시나리오")
        print("5. 데모 스크립트 생성")
        print("6. 템플릿 설정 가이드")
        print("7. 전체 테스트 실행")
        print("0. 종료")
        
        choice = input("\n선택하세요 (0-7): ")
        
        if choice == "0":
            print("테스트를 종료합니다.")
            break
            
        elif choice == "1":
            test_screen_capture()
            
        elif choice == "2":
            test_virtual_input()
            
        elif choice == "3":
            test_godot_detection()
            
        elif choice == "4":
            test_simple_automation()
            
        elif choice == "5":
            create_demo_script()
            
        elif choice == "6":
            create_test_templates()
            
        elif choice == "7":
            print_header("전체 테스트 실행")
            results = []
            
            results.append(("화면 캡처", test_screen_capture()))
            time.sleep(1)
            
            results.append(("가상 입력", test_virtual_input()))
            time.sleep(1)
            
            if sys.platform == "win32":
                results.append(("Godot 감지", test_godot_detection()))
                time.sleep(1)
            
            # 결과 요약
            print_header("테스트 결과 요약")
            for name, result in results:
                if result:
                    print_success(f"{name}: 성공")
                else:
                    print_error(f"{name}: 실패")
        
        else:
            print_warning("잘못된 선택입니다.")

if __name__ == "__main__":
    main()