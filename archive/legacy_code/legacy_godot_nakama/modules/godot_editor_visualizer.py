#!/usr/bin/env python3
"""
Godot 에디터 실시간 제어 및 게임 제작 시각화
실제 Godot 에디터를 열고 게임이 만들어지는 과정을 직접 보여줍니다
"""

import os
import sys
import time
import asyncio
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pyautogui
import win32gui
import win32con
import ctypes
from ctypes import wintypes

# PyAutoGUI 안전 설정
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

class GodotEditorVisualizer:
    """Godot 에디터를 실제로 제어하여 게임 제작 과정을 보여주는 시스템"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.godot_exe = None
        self.godot_process = None
        self.godot_window = None
        self.is_visualizing = False
        
        # 윈도우 핸들 찾기를 위한 설정
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32
        
    def find_godot_executable(self) -> Optional[str]:
        """Godot 실행 파일 찾기"""
        # AI 수정된 Godot 우선
        godot_paths = [
            self.project_root / "godot_ai_build" / "output" / "godot.windows.editor.x86_64.exe",
            self.project_root / "godot_ai_build" / "Godot_v4.3-stable_win64.exe",
            Path("/mnt/c/Program Files/Godot/Godot.exe"),
            Path("/mnt/d/Godot/Godot.exe"),
        ]
        
        for path in godot_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def wsl_to_windows_path(self, wsl_path: str) -> str:
        """WSL 경로를 Windows 경로로 변환"""
        if wsl_path.startswith("/mnt/"):
            # /mnt/c/ -> C:\
            drive = wsl_path[5].upper()
            path = wsl_path[7:].replace("/", "\\")
            return f"{drive}:\\{path}"
        return wsl_path
    
    async def open_godot_editor(self, project_path: Optional[str] = None) -> bool:
        """Godot 에디터 열기"""
        self.godot_exe = self.find_godot_executable()
        if not self.godot_exe:
            print("❌ Godot 실행 파일을 찾을 수 없습니다.")
            return False
        
        # WSL에서 Windows 실행 파일 실행
        cmd = ["cmd.exe", "/c", self.wsl_to_windows_path(self.godot_exe)]
        
        if project_path:
            windows_project_path = self.wsl_to_windows_path(str(project_path))
            cmd.extend(["--path", windows_project_path])
        else:
            # 새 프로젝트를 위한 에디터 열기
            cmd.append("--editor")
        
        print(f"🚀 Godot 에디터를 시작합니다...")
        print(f"   명령어: {' '.join(cmd)}")
        
        try:
            self.godot_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 에디터가 완전히 로드될 때까지 대기
            await asyncio.sleep(5)
            
            # Godot 창 찾기
            self.find_godot_window()
            
            if self.godot_window:
                print("✅ Godot 에디터가 성공적으로 열렸습니다!")
                # 창을 전면으로 가져오기
                self.bring_window_to_front()
                return True
            else:
                print("⚠️ Godot 창을 찾을 수 없습니다.")
                return False
                
        except Exception as e:
            print(f"❌ Godot 에디터 실행 중 오류: {e}")
            return False
    
    def find_godot_window(self):
        """Godot 에디터 창 찾기"""
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if "Godot" in window_text:
                    windows.append((hwnd, window_text))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        for hwnd, title in windows:
            if "Godot" in title:
                self.godot_window = hwnd
                print(f"🎯 Godot 창을 찾았습니다: {title}")
                return True
        
        return False
    
    def bring_window_to_front(self):
        """Godot 창을 전면으로 가져오기"""
        if self.godot_window:
            win32gui.ShowWindow(self.godot_window, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(self.godot_window)
            time.sleep(0.5)
    
    async def create_new_project(self, project_name: str, project_type: str) -> bool:
        """새 프로젝트 생성 과정 시각화"""
        print(f"\n🎮 '{project_name}' {project_type} 게임 프로젝트를 만듭니다...")
        
        # 프로젝트 매니저 대신 바로 에디터 열기
        project_path = self.project_root / "game_projects" / project_name
        project_path.mkdir(parents=True, exist_ok=True)
        
        # project.godot 파일 생성
        project_file = project_path / "project.godot"
        project_config = f"""
; Engine configuration file.
; It's best edited using the editor UI and not directly,
; since the properties are organized in sections here.

[application]

config/name="{project_name}"
config/features=PackedStringArray("4.3", "GL Compatibility")
config/icon="res://icon.svg"

[rendering]

renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"
"""
        project_file.write_text(project_config.strip())
        
        # 에디터 열기
        if not await self.open_godot_editor(project_path):
            return False
        
        await asyncio.sleep(2)
        return True
    
    async def show_scene_creation(self, scene_type: str):
        """씬 생성 과정 시각화"""
        print(f"\n🎬 {scene_type} 씬을 생성합니다...")
        
        if not self.godot_window:
            return
        
        self.bring_window_to_front()
        
        # Ctrl+N으로 새 씬 생성
        print("  📝 새 씬 생성 (Ctrl+N)...")
        pyautogui.hotkey('ctrl', 'n')
        await asyncio.sleep(1)
        
        # 2D Scene 선택
        if scene_type == "2D":
            print("  🎨 2D 씬 선택...")
            # 2D Scene 버튼 클릭 위치 (대략적인 위치)
            pyautogui.click(x=400, y=300)
            await asyncio.sleep(1)
    
    async def add_node(self, node_type: str, node_name: str):
        """노드 추가 과정 시각화"""
        print(f"\n➕ {node_type} 노드 '{node_name}'를 추가합니다...")
        
        if not self.godot_window:
            return
        
        self.bring_window_to_front()
        
        # 씬 트리에서 루트 노드 선택
        print("  🎯 루트 노드 선택...")
        pyautogui.click(x=200, y=200)  # 씬 트리 영역
        await asyncio.sleep(0.5)
        
        # Ctrl+A로 노드 추가
        print(f"  ➕ {node_type} 노드 추가 (Ctrl+A)...")
        pyautogui.hotkey('ctrl', 'a')
        await asyncio.sleep(1)
        
        # 노드 타입 검색
        print(f"  🔍 {node_type} 검색...")
        pyautogui.write(node_type)
        await asyncio.sleep(1)
        
        # Enter로 선택
        pyautogui.press('enter')
        await asyncio.sleep(1)
    
    async def show_script_creation(self, node_name: str, script_content: str):
        """스크립트 작성 과정 시각화"""
        print(f"\n📝 '{node_name}' 노드에 스크립트를 추가합니다...")
        
        if not self.godot_window:
            return
        
        self.bring_window_to_front()
        
        # 노드 선택 후 스크립트 추가
        print("  📎 스크립트 첨부...")
        pyautogui.rightClick(x=200, y=250)  # 노드 위치
        await asyncio.sleep(0.5)
        
        # "Attach Script" 메뉴 선택
        pyautogui.click(x=250, y=280)  # 메뉴 위치
        await asyncio.sleep(1)
        
        # 스크립트 생성 다이얼로그에서 Create 클릭
        pyautogui.click(x=600, y=500)  # Create 버튼
        await asyncio.sleep(2)
        
        # 스크립트 내용 작성 (시뮬레이션)
        print("  ✍️ 스크립트 작성 중...")
        # 실제로는 더 복잡한 코드 작성 과정을 보여줄 수 있음
    
    async def visualize_game_creation(self, game_type: str, game_name: str):
        """게임 제작 전체 과정 시각화"""
        self.is_visualizing = True
        
        try:
            # 1. 프로젝트 생성
            if not await self.create_new_project(game_name, game_type):
                print("❌ 프로젝트 생성 실패")
                return
            
            print("\n" + "="*60)
            print("🎬 이제 실제 게임 제작 과정을 보여드립니다!")
            print("="*60 + "\n")
            
            # 2. 메인 씬 생성
            await self.show_scene_creation("2D")
            
            # 3. 게임 타입별 노드 추가
            if game_type == "platformer":
                await self.show_platformer_creation()
            elif game_type == "racing":
                await self.show_racing_creation()
            else:
                await self.show_basic_game_creation()
            
            print("\n✅ 게임 제작 시연이 완료되었습니다!")
            print("💬 이제 사용자가 직접 수정하고 발전시킬 수 있습니다.")
            
        except Exception as e:
            print(f"❌ 시각화 중 오류 발생: {e}")
        finally:
            self.is_visualizing = False
    
    async def show_platformer_creation(self):
        """플랫포머 게임 제작 과정"""
        print("\n🎮 플랫포머 게임 제작을 시작합니다...")
        
        # 플레이어 캐릭터 추가
        await self.add_node("CharacterBody2D", "Player")
        await asyncio.sleep(1)
        
        # 스프라이트 추가
        await self.add_node("Sprite2D", "PlayerSprite")
        await asyncio.sleep(1)
        
        # 충돌 모양 추가
        await self.add_node("CollisionShape2D", "PlayerCollision")
        await asyncio.sleep(1)
        
        # 플레이어 스크립트 추가
        player_script = """
extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

func _physics_process(delta):
    # 중력 추가
    if not is_on_floor():
        velocity.y += gravity * delta
    
    # 점프
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = JUMP_VELOCITY
    
    # 좌우 이동
    var direction = Input.get_axis("ui_left", "ui_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = move_toward(velocity.x, 0, SPEED)
    
    move_and_slide()
"""
        await self.show_script_creation("Player", player_script)
        
        # 플랫폼 추가
        print("\n🏗️ 플랫폼을 추가합니다...")
        await self.add_node("StaticBody2D", "Platform")
        
    async def show_racing_creation(self):
        """레이싱 게임 제작 과정"""
        print("\n🏎️ 레이싱 게임 제작을 시작합니다...")
        
        # 차량 추가
        await self.add_node("RigidBody2D", "Car")
        await asyncio.sleep(1)
        
        # 트랙 추가
        await self.add_node("Path2D", "RaceTrack")
        
    async def show_basic_game_creation(self):
        """기본 게임 제작 과정"""
        print("\n🎮 기본 게임 구조를 만듭니다...")
        
        # 기본 노드들 추가
        await self.add_node("Node2D", "GameWorld")
        await self.add_node("Camera2D", "MainCamera")
    
    def close_editor(self):
        """에디터 닫기"""
        if self.godot_process:
            self.godot_process.terminate()
            self.godot_process = None
            print("🔚 Godot 에디터를 닫았습니다.")


# 전역 인스턴스
_visualizer = None

def get_godot_visualizer() -> GodotEditorVisualizer:
    """Godot 에디터 시각화 싱글톤 인스턴스 반환"""
    global _visualizer
    if _visualizer is None:
        _visualizer = GodotEditorVisualizer()
    return _visualizer