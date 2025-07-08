#!/usr/bin/env python3
"""
간단한 Godot 자동화 테스트
최소한의 기능으로 작동 확인
"""

import time
import sys
from pathlib import Path

def test_basic_imports():
    """기본 임포트 테스트"""
    print("1. 기본 라이브러리 확인...")
    
    try:
        import cv2
        print("✓ OpenCV 설치됨")
    except:
        print("✗ OpenCV 미설치 - pip install opencv-python")
        return False
        
    try:
        import pyautogui
        print("✓ PyAutoGUI 설치됨")
    except:
        print("✗ PyAutoGUI 미설치 - pip install pyautogui")
        return False
        
    try:
        import torch
        print("✓ PyTorch 설치됨")
    except:
        print("✗ PyTorch 미설치 - pip install torch")
        return False
        
    return True

def test_screen_capture():
    """화면 캡처 간단 테스트"""
    print("\n2. 화면 캡처 테스트...")
    
    try:
        import pyautogui
        
        # 스크린샷 찍기
        screenshot = pyautogui.screenshot()
        print(f"✓ 화면 캡처 성공: {screenshot.size}")
        
        # 파일로 저장
        screenshot.save("test_capture.png")
        print("✓ test_capture.png 저장됨")
        
        return True
        
    except Exception as e:
        print(f"✗ 오류: {e}")
        return False

def test_mouse_movement():
    """마우스 이동 테스트"""
    print("\n3. 마우스 제어 테스트...")
    print("⚠️  5초 후 마우스가 움직입니다. 중단하려면 Ctrl+C")
    
    try:
        import pyautogui
        
        # 안전 설정
        pyautogui.FAILSAFE = True
        
        for i in range(5, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        # 현재 위치 저장
        original_x, original_y = pyautogui.position()
        print(f"현재 마우스 위치: ({original_x}, {original_y})")
        
        # 사각형 그리기
        moves = [
            (100, 0),   # 오른쪽
            (0, 100),   # 아래
            (-100, 0),  # 왼쪽
            (0, -100)   # 위
        ]
        
        for dx, dy in moves:
            pyautogui.moveRel(dx, dy, duration=0.5)
        
        # 원래 위치로
        pyautogui.moveTo(original_x, original_y, duration=0.5)
        print("✓ 마우스 이동 완료")
        
        return True
        
    except Exception as e:
        print(f"✗ 오류: {e}")
        return False

def test_keyboard_input():
    """키보드 입력 테스트"""
    print("\n4. 키보드 입력 테스트...")
    print("⚠️  메모장을 열고 클릭하세요. 5초 후 시작합니다.")
    
    try:
        import pyautogui
        
        for i in range(5, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        # 텍스트 입력
        pyautogui.write("Hello from AutoCI!\n", interval=0.1)
        pyautogui.write("Godot Automation Test", interval=0.05)
        
        print("✓ 키보드 입력 완료")
        return True
        
    except Exception as e:
        print(f"✗ 오류: {e}")
        return False

def test_godot_window_search():
    """Godot 창 찾기 테스트"""
    print("\n5. Godot 창 검색...")
    
    if sys.platform == "win32":
        try:
            import win32gui
            
            godot_windows = []
            
            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if "Godot" in title:
                        windows.append(title)
                return True
            
            win32gui.EnumWindows(callback, godot_windows)
            
            if godot_windows:
                print(f"✓ Godot 창 발견: {godot_windows}")
                return True
            else:
                print("✗ Godot 창을 찾을 수 없습니다")
                return False
                
        except ImportError:
            print("✗ pywin32 미설치 - pip install pywin32")
            return False
    else:
        print("⚠️  Windows가 아닌 환경에서는 이 테스트를 건너뜁니다")
        return True

def main():
    """메인 테스트"""
    print("=" * 50)
    print("AutoCI Godot 자동화 간단 테스트")
    print("=" * 50)
    
    results = []
    
    # 1. 임포트 테스트
    if test_basic_imports():
        results.append(("임포트", True))
    else:
        print("\n필수 패키지를 설치하고 다시 실행하세요.")
        return
    
    # 2. 화면 캡처
    results.append(("화면 캡처", test_screen_capture()))
    
    # 3. 사용자 확인
    print("\n다음 테스트는 마우스와 키보드를 제어합니다.")
    response = input("계속하시겠습니까? (y/n): ")
    
    if response.lower() == 'y':
        # 4. 마우스 테스트
        results.append(("마우스 제어", test_mouse_movement()))
        
        # 5. 키보드 테스트
        results.append(("키보드 제어", test_keyboard_input()))
    
    # 6. Godot 창 검색
    results.append(("Godot 검색", test_godot_window_search()))
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("테스트 결과 요약:")
    print("=" * 50)
    
    for name, success in results:
        status = "✓ 성공" if success else "✗ 실패"
        print(f"{name}: {status}")
    
    # 다음 단계 안내
    print("\n다음 단계:")
    if all(success for _, success in results):
        print("✓ 모든 테스트 통과! demo_godot_automation.py를 실행해보세요.")
    else:
        print("✗ 일부 테스트 실패. 필요한 설정을 확인하세요.")

if __name__ == "__main__":
    main()