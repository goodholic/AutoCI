#!/usr/bin/env python3
"""
Godot Visual Controller - AI가 Godot을 시각적으로 조종하는 모습을 보여줍니다.
실제 마우스 움직임, 클릭, 타이핑 등을 사람이 볼 수 있게 구현합니다.
"""

import os
import sys
import time
import asyncio
import subprocess
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import threading
import queue

# Windows 자동화 라이브러리
try:
    import pyautogui
    import win32gui
    import win32con
    import win32api
    import win32process
    import psutil
    AUTOMATION_AVAILABLE = True
except ImportError:
    AUTOMATION_AVAILABLE = False
    print("⚠️ Windows 자동화 라이브러리가 없습니다. 시뮬레이션 모드로 실행됩니다.")

class GodotVisualController:
    """AI가 Godot을 시각적으로 조종하는 컨트롤러"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.godot_window = None
        self.godot_process = None
        self.current_action = None
        self.action_queue = queue.Queue()
        self.mouse_speed = 0.5  # 마우스 이동 속도
        self.typing_speed = 0.1  # 타이핑 속도
        
        # PyAutoGUI 설정
        if AUTOMATION_AVAILABLE:
            pyautogui.FAILSAFE = True  # 화면 모서리로 마우스 이동시 중단
            pyautogui.PAUSE = 0.3  # 각 동작 사이 일시정지
    
    def find_godot_window(self) -> Optional[int]:
        """Godot 창 찾기"""
        if not AUTOMATION_AVAILABLE:
            return None
            
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if "Godot" in window_text:
                    windows.append((hwnd, window_text))
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        for hwnd, title in windows:
            if hwnd:
                self.godot_window = hwnd
                return hwnd
        
        return None
    
    def bring_godot_to_front(self):
        """Godot 창을 앞으로 가져오기"""
        if self.godot_window and AUTOMATION_AVAILABLE:
            try:
                win32gui.ShowWindow(self.godot_window, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(self.godot_window)
                time.sleep(0.5)
                return True
            except:
                return False
        return False
    
    async def show_ai_control_start(self):
        """AI 제어 시작 알림"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🤖 AI가 Godot을 직접 제어합니다                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

👀 지금부터 AI가 마우스와 키보드를 사용하여 Godot을 조작하는 모습을 보실 수 있습니다.
🖱️ 마우스가 자동으로 움직이며 클릭합니다.
⌨️ 키보드로 코드와 설정을 입력합니다.
⏸️ 언제든지 ESC 키를 눌러 중단할 수 있습니다.
""")
        await asyncio.sleep(2)
    
    async def move_mouse_smoothly(self, x: int, y: int, duration: float = 1.0):
        """마우스를 부드럽게 이동"""
        if not AUTOMATION_AVAILABLE:
            print(f"🖱️ [시뮬레이션] 마우스 이동: ({x}, {y})")
            return
        
        print(f"🖱️ 마우스를 ({x}, {y})로 이동합니다...")
        
        # 현재 마우스 위치
        start_x, start_y = pyautogui.position()
        
        # 베지어 곡선으로 부드러운 이동
        pyautogui.moveTo(x, y, duration=duration, tween=pyautogui.easeInOutQuad)
        
        # 살짝 흔들기 (인간적인 움직임)
        for _ in range(2):
            offset_x = random.randint(-2, 2)
            offset_y = random.randint(-2, 2)
            pyautogui.moveRel(offset_x, offset_y, duration=0.1)
        
        await asyncio.sleep(0.2)
    
    async def click_at(self, x: int, y: int, button: str = 'left', clicks: int = 1):
        """특정 위치 클릭"""
        if not AUTOMATION_AVAILABLE:
            print(f"🖱️ [시뮬레이션] {button} 클릭: ({x}, {y})")
            return
        
        await self.move_mouse_smoothly(x, y, 0.5)
        print(f"🖱️ {button} 클릭!")
        
        # 클릭 전 잠시 대기 (인간적인 동작)
        await asyncio.sleep(0.2)
        pyautogui.click(x, y, button=button, clicks=clicks)
        await asyncio.sleep(0.3)
    
    async def type_text(self, text: str, interval: float = 0.1):
        """텍스트 타이핑"""
        if not AUTOMATION_AVAILABLE:
            print(f"⌨️ [시뮬레이션] 타이핑: {text}")
            return
        
        print(f"⌨️ 타이핑: {text}")
        
        # 한 글자씩 타이핑 (인간적인 속도)
        for char in text:
            pyautogui.write(char, interval=interval)
            # 가끔 타이핑 속도 변화
            if random.random() < 0.1:
                await asyncio.sleep(random.uniform(0.1, 0.3))
        
        await asyncio.sleep(0.3)
    
    async def press_key(self, key: str, presses: int = 1):
        """키 누르기"""
        if not AUTOMATION_AVAILABLE:
            print(f"⌨️ [시뮬레이션] 키 누르기: {key}")
            return
        
        print(f"⌨️ {key} 키를 누릅니다")
        pyautogui.press(key, presses=presses, interval=0.1)
        await asyncio.sleep(0.2)
    
    async def hotkey(self, *keys):
        """단축키 조합"""
        if not AUTOMATION_AVAILABLE:
            print(f"⌨️ [시뮬레이션] 단축키: {'+'.join(keys)}")
            return
        
        print(f"⌨️ 단축키: {'+'.join(keys)}")
        pyautogui.hotkey(*keys)
        await asyncio.sleep(0.3)
    
    async def create_new_project(self, project_name: str, project_path: str):
        """새 프로젝트 생성 시연"""
        print("\n📁 새 프로젝트를 생성합니다...")
        
        # Godot 창 활성화
        self.bring_godot_to_front()
        
        # Project 메뉴 클릭
        await self.click_at(100, 50)  # Project 메뉴 위치
        await asyncio.sleep(0.5)
        
        # New Project 클릭
        await self.click_at(120, 100)  # New Project 메뉴 항목
        await asyncio.sleep(1)
        
        # 프로젝트 이름 입력
        print("📝 프로젝트 이름을 입력합니다...")
        await self.click_at(400, 200)  # 프로젝트 이름 필드
        await self.hotkey('ctrl', 'a')  # 전체 선택
        await self.type_text(project_name, 0.1)
        
        # 프로젝트 경로 설정
        await self.click_at(400, 250)  # 경로 필드
        await self.hotkey('ctrl', 'a')
        await self.type_text(project_path, 0.05)
        
        # Create 버튼 클릭
        await self.click_at(600, 400)  # Create 버튼
        print("✅ 프로젝트 생성 완료!")
        await asyncio.sleep(2)
    
    async def create_scene(self, scene_name: str):
        """씬 생성 시연"""
        print(f"\n🎬 {scene_name} 씬을 생성합니다...")
        
        # 2D Scene 버튼 클릭
        await self.click_at(200, 300)  # 2D Scene 버튼 위치
        await asyncio.sleep(1)
        
        # 씬 저장
        await self.hotkey('ctrl', 's')
        await asyncio.sleep(0.5)
        
        # 파일명 입력
        await self.type_text(f"scenes/{scene_name}.tscn", 0.1)
        await self.press_key('enter')
        
        print(f"✅ {scene_name} 씬 생성 완료!")
    
    async def add_node(self, node_type: str, node_name: str):
        """노드 추가 시연"""
        print(f"\n🔧 {node_type} 노드를 추가합니다...")
        
        # Add Node 버튼 클릭
        await self.click_at(50, 150)  # + 버튼
        await asyncio.sleep(0.5)
        
        # 노드 타입 검색
        await self.type_text(node_type, 0.1)
        await asyncio.sleep(0.5)
        
        # 첫 번째 결과 선택
        await self.press_key('enter')
        await asyncio.sleep(0.5)
        
        # 노드 이름 변경
        await self.press_key('f2')
        await self.type_text(node_name, 0.1)
        await self.press_key('enter')
        
        print(f"✅ {node_name} 노드 추가 완료!")
    
    async def write_script(self, script_content: str):
        """스크립트 작성 시연"""
        print("\n📝 스크립트를 작성합니다...")
        
        # Script 에디터로 전환
        await self.click_at(600, 100)  # Script 탭
        await asyncio.sleep(0.5)
        
        # 코드 작성
        lines = script_content.split('\n')
        for line in lines:
            await self.type_text(line, 0.05)
            await self.press_key('enter')
            
            # 들여쓰기 처리
            if line.strip().endswith(':'):
                await self.press_key('tab')
        
        # 저장
        await self.hotkey('ctrl', 's')
        print("✅ 스크립트 작성 완료!")
    
    async def demonstrate_game_creation(self, game_name: str, game_type: str):
        """게임 제작 과정 시연"""
        await self.show_ai_control_start()
        
        # Godot 창 찾기
        if not self.find_godot_window():
            print("❌ Godot 창을 찾을 수 없습니다!")
            return
        
        self.bring_godot_to_front()
        
        # 1. 프로젝트 생성
        project_path = str(self.project_root / "ai_demos" / game_name)
        await self.create_new_project(game_name, project_path)
        
        # 2. 메인 씬 생성
        await self.create_scene("Main")
        
        # 3. 플레이어 추가
        if game_type == "platformer":
            await self.add_node("CharacterBody2D", "Player")
            await self.add_node("Sprite2D", "PlayerSprite")
            await self.add_node("CollisionShape2D", "PlayerCollision")
            
            # 플레이어 스크립트
            script = """extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

func _physics_process(delta):
    if not is_on_floor():
        velocity.y += 980 * delta
    
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = JUMP_VELOCITY
    
    var direction = Input.get_axis("ui_left", "ui_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = 0
    
    move_and_slide()"""
            
            await self.write_script(script)
        
        # 4. 플랫폼 추가
        await self.add_node("StaticBody2D", "Platform")
        await self.add_node("Sprite2D", "PlatformSprite")
        await self.add_node("CollisionShape2D", "PlatformCollision")
        
        # 5. 게임 실행
        print("\n▶️ 게임을 실행합니다...")
        await self.press_key('f5')
        
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ✅ AI 제어 시연 완료!                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎮 기본적인 게임 구조가 생성되었습니다.
👀 AI가 Godot을 직접 조작하는 모습을 보셨습니다.
🚀 이제 24시간 게임 제작이 계속됩니다...
""")
    
    async def show_continuous_work(self):
        """지속적인 작업 모습 보여주기"""
        actions = [
            ("🎨 스프라이트 추가", self.add_sprite_animation),
            ("🔊 사운드 효과 추가", self.add_sound_effects),
            ("💡 라이팅 설정", self.setup_lighting),
            ("🎮 게임 로직 개선", self.improve_game_logic),
            ("🧪 테스트 실행", self.run_tests),
        ]
        
        for action_name, action_func in actions:
            print(f"\n{action_name}...")
            await action_func()
            await asyncio.sleep(random.uniform(2, 5))
    
    async def add_sprite_animation(self):
        """스프라이트 애니메이션 추가"""
        # AnimationPlayer 노드 추가
        await self.add_node("AnimationPlayer", "PlayerAnimator")
        
        # 애니메이션 생성
        await self.click_at(300, 400)  # Animation 패널
        await self.click_at(350, 420)  # New Animation
        await self.type_text("idle", 0.1)
        await self.press_key('enter')
        
        print("✅ 애니메이션 추가 완료!")
    
    async def add_sound_effects(self):
        """사운드 효과 추가"""
        await self.add_node("AudioStreamPlayer2D", "JumpSound")
        await self.add_node("AudioStreamPlayer2D", "LandSound")
        
        print("✅ 사운드 효과 추가 완료!")
    
    async def setup_lighting(self):
        """라이팅 설정"""
        await self.add_node("DirectionalLight2D", "SunLight")
        
        # 속성 설정
        await self.click_at(800, 300)  # Inspector
        await self.click_at(850, 350)  # Energy 속성
        await self.type_text("0.8", 0.1)
        
        print("✅ 라이팅 설정 완료!")
    
    async def improve_game_logic(self):
        """게임 로직 개선"""
        # 스크립트 에디터로 이동
        await self.click_at(600, 100)
        
        # 코드 추가
        await self.hotkey('ctrl', 'end')  # 끝으로 이동
        await self.press_key('enter', 2)
        
        code = """
func _ready():
    print("Game Started!")
    
func game_over():
    get_tree().reload_current_scene()
"""
        
        for line in code.strip().split('\n'):
            await self.type_text(line, 0.05)
            await self.press_key('enter')
        
        await self.hotkey('ctrl', 's')
        print("✅ 게임 로직 개선 완료!")
    
    async def run_tests(self):
        """테스트 실행"""
        print("🧪 게임 테스트 중...")
        await self.press_key('f6')  # 현재 씬 실행
        await asyncio.sleep(3)
        
        # 테스트 동작
        await self.press_key('right', 5)  # 오른쪽 이동
        await self.press_key('space', 2)  # 점프
        await self.press_key('left', 3)   # 왼쪽 이동
        
        await self.press_key('escape')  # 종료
        print("✅ 테스트 완료!")


# 싱글톤 인스턴스
_visual_controller = None

def get_visual_controller() -> GodotVisualController:
    """비주얼 컨트롤러 싱글톤 인스턴스 반환"""
    global _visual_controller
    if _visual_controller is None:
        _visual_controller = GodotVisualController()
    return _visual_controller