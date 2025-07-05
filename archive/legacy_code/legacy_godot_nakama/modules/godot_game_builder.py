#!/usr/bin/env python3
"""
Godot 게임 빌더 - 실제 게임 파일을 생성하고 Godot에서 열어서 보여주는 시스템
로딩창 대신 실제 게임이 만들어지는 과정을 보여줍니다
"""

import os
import sys
import json
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

class GodotGameBuilder:
    """실제 Godot 게임 프로젝트를 빌드하고 에디터에서 보여주는 시스템"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.current_project_path = None
        self.godot_exe = None
        self.godot_process = None
        
    def find_godot_executable(self) -> Optional[str]:
        """Godot 실행 파일 찾기"""
        godot_paths = [
            self.project_root / "godot_ai_build" / "output" / "godot.ai.editor.windows.x86_64.exe",
            self.project_root / "godot_ai_build" / "godot-source" / "bin" / "godot.windows.editor.x86_64.exe",
            self.project_root / "godot_ai_build" / "Godot_v4.3-stable_win64.exe",
            Path("/mnt/c/Program Files/Godot/Godot.exe"),
            Path("/mnt/d/Godot/Godot.exe"),
        ]
        
        for path in godot_paths:
            if path.exists():
                print(f"✅ Godot 찾음: {path}")
                return str(path)
        
        return None
    
    async def create_game_project(self, game_name: str, game_type: str) -> Path:
        """실제 게임 프로젝트 생성"""
        project_path = self.project_root / "game_projects" / game_name
        project_path.mkdir(parents=True, exist_ok=True)
        self.current_project_path = project_path
        
        print(f"\n🎮 {game_name} 프로젝트를 생성합니다...")
        
        # 1. project.godot 파일 생성
        await self.create_project_file(project_path, game_name, game_type)
        
        # 2. 메인 씬 생성
        await self.create_main_scene(project_path, game_type)
        
        # 3. 플레이어 씬 생성
        if game_type in ["platformer", "racing"]:
            await self.create_player_scene(project_path, game_type)
        
        # 4. 게임 스크립트 생성
        await self.create_game_scripts(project_path, game_type)
        
        # 5. 리소스 폴더 구조 생성
        await self.create_folder_structure(project_path)
        
        return project_path
    
    async def create_project_file(self, project_path: Path, game_name: str, game_type: str):
        """project.godot 파일 생성"""
        print("  📄 project.godot 파일 생성 중...")
        
        config = f"""
[application]

config/name="{game_name}"
config/description="AI와 함께 만드는 {game_type} 게임"
run/main_scene="res://scenes/Main.tscn"
config/features=PackedStringArray("4.3", "GL Compatibility")
config/icon="res://icon.svg"

[display]

window/size/viewport_width=1280
window/size/viewport_height=720

[input]

move_left={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":65,"key_label":0,"unicode":97,"location":0,"echo":false,"script":null)
, Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":4194319,"key_label":0,"unicode":0,"location":0,"echo":false,"script":null)
]
}}
move_right={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":68,"key_label":0,"unicode":100,"location":0,"echo":false,"script":null)
, Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":4194321,"key_label":0,"unicode":0,"location":0,"echo":false,"script":null)
]
}}
jump={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":32,"key_label":0,"unicode":32,"location":0,"echo":false,"script":null)
]
}}

[physics]

2d/default_gravity=980.0

[rendering]

renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"
"""
        
        (project_path / "project.godot").write_text(config.strip())
        await asyncio.sleep(0.5)
        print("  ✅ project.godot 생성 완료!")
    
    async def create_main_scene(self, project_path: Path, game_type: str):
        """메인 씬 파일 생성"""
        print("\n  🎬 메인 씬 생성 중...")
        
        scenes_dir = project_path / "scenes"
        scenes_dir.mkdir(exist_ok=True)
        
        # Main.tscn 생성
        main_scene = """[gd_scene load_steps=3 format=3 uid="uid://main_scene"]

[ext_resource type="Script" path="res://scripts/Main.gd" id="1"]
[ext_resource type="PackedScene" uid="uid://player_scene" path="res://scenes/Player.tscn" id="2"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

[node name="World" type="Node2D" parent="."]

[node name="Player" parent="World" instance=ExtResource("2")]
position = Vector2(640, 360)

[node name="Camera2D" type="Camera2D" parent="World/Player"]
enabled = true
zoom = Vector2(1, 1)
"""
        
        if game_type == "platformer":
            # 플랫폼 추가
            main_scene += """
[node name="Platforms" type="Node2D" parent="World"]

[node name="Ground" type="StaticBody2D" parent="World/Platforms"]
position = Vector2(640, 600)

[node name="CollisionShape2D" type="CollisionShape2D" parent="World/Platforms/Ground"]
shape = SubResource("RectangleShape2D_1")

[node name="ColorRect" type="ColorRect" parent="World/Platforms/Ground"]
offset_left = -640.0
offset_top = -20.0
offset_right = 640.0
offset_bottom = 20.0
color = Color(0.4, 0.2, 0.1, 1)
"""
        
        (scenes_dir / "Main.tscn").write_text(main_scene)
        await asyncio.sleep(0.5)
        print("  ✅ 메인 씬 생성 완료!")
    
    async def create_player_scene(self, project_path: Path, game_type: str):
        """플레이어 씬 생성"""
        print("\n  🎮 플레이어 캐릭터 생성 중...")
        
        scenes_dir = project_path / "scenes"
        
        if game_type == "platformer":
            player_scene = """[gd_scene load_steps=4 format=3 uid="uid://player_scene"]

[ext_resource type="Script" path="res://scripts/Player.gd" id="1"]

[sub_resource type="RectangleShape2D" id="RectangleShape2D_1"]
size = Vector2(32, 64)

[sub_resource type="PlaceholderTexture2D" id="PlaceholderTexture2D_1"]
size = Vector2(32, 64)

[node name="Player" type="CharacterBody2D"]
script = ExtResource("1")

[node name="CollisionShape2D" type="CollisionShape2D" parent="."]
shape = SubResource("RectangleShape2D_1")

[node name="Sprite2D" type="Sprite2D" parent="."]
texture = SubResource("PlaceholderTexture2D_1")
"""
        elif game_type == "racing":
            player_scene = """[gd_scene load_steps=4 format=3 uid="uid://player_scene"]

[ext_resource type="Script" path="res://scripts/Car.gd" id="1"]

[sub_resource type="RectangleShape2D" id="RectangleShape2D_1"]
size = Vector2(40, 80)

[sub_resource type="PlaceholderTexture2D" id="PlaceholderTexture2D_1"]
size = Vector2(40, 80)

[node name="Car" type="RigidBody2D"]
script = ExtResource("1")

[node name="CollisionShape2D" type="CollisionShape2D" parent="."]
shape = SubResource("RectangleShape2D_1")

[node name="Sprite2D" type="Sprite2D" parent="."]
texture = SubResource("PlaceholderTexture2D_1")
"""
        else:
            return
        
        (scenes_dir / "Player.tscn").write_text(player_scene)
        await asyncio.sleep(0.5)
        print("  ✅ 플레이어 씬 생성 완료!")
    
    async def create_game_scripts(self, project_path: Path, game_type: str):
        """게임 스크립트 생성"""
        print("\n  📝 게임 스크립트 작성 중...")
        
        scripts_dir = project_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Main.gd
        main_script = """extends Node2D

# 게임 메인 스크립트
func _ready():
\tprint("게임이 시작되었습니다!")
\tprint("AI와 함께 만든 %s 게임입니다." % ["%s"])
\t
\t# 게임 초기화
\t_initialize_game()

func _initialize_game():
\t# 게임 설정 초기화
\tpass

func _input(event):
\tif event.is_action_pressed("ui_cancel"):
\t\tget_tree().quit()
""" % (game_type, game_type)
        
        (scripts_dir / "Main.gd").write_text(main_script)
        
        # Player.gd (플랫포머용)
        if game_type == "platformer":
            player_script = """extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

# Get the gravity from the project settings to be synced with RigidBody nodes.
var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

func _physics_process(delta):
\t# Add the gravity.
\tif not is_on_floor():
\t\tvelocity.y += gravity * delta

\t# Handle jump.
\tif Input.is_action_just_pressed("jump") and is_on_floor():
\t\tvelocity.y = JUMP_VELOCITY

\t# Get the input direction and handle the movement/deceleration.
\tvar direction = Input.get_axis("move_left", "move_right")
\tif direction:
\t\tvelocity.x = direction * SPEED
\telse:
\t\tvelocity.x = move_toward(velocity.x, 0, SPEED)

\tmove_and_slide()
"""
            (scripts_dir / "Player.gd").write_text(player_script)
        
        # Car.gd (레이싱용)
        elif game_type == "racing":
            car_script = """extends RigidBody2D

const ENGINE_POWER = 800
const STEERING_POWER = 3.0

var velocity = Vector2.ZERO
var steering_input = 0.0

func _physics_process(delta):
\t# 입력 처리
\tvar throttle = Input.get_axis("move_down", "move_up")
\tsteering_input = Input.get_axis("move_left", "move_right")
\t
\t# 엔진 힘 적용
\tif throttle != 0:
\t\tapply_central_force(transform.y * throttle * ENGINE_POWER)
\t
\t# 조향
\tif abs(linear_velocity.length()) > 10:
\t\tangular_velocity = steering_input * STEERING_POWER
"""
            (scripts_dir / "Car.gd").write_text(car_script)
        
        await asyncio.sleep(0.5)
        print("  ✅ 스크립트 작성 완료!")
    
    async def create_folder_structure(self, project_path: Path):
        """프로젝트 폴더 구조 생성"""
        print("\n  📁 프로젝트 폴더 구조 생성 중...")
        
        folders = ["assets", "assets/sprites", "assets/sounds", "assets/music", "assets/fonts"]
        for folder in folders:
            (project_path / folder).mkdir(parents=True, exist_ok=True)
        
        # 기본 아이콘 생성
        icon_svg = """<svg height="128" width="128" xmlns="http://www.w3.org/2000/svg">
<rect x="0" y="0" width="128" height="128" fill="#4A90E2"/>
<text x="64" y="64" font-family="Arial" font-size="40" fill="white" text-anchor="middle" alignment-baseline="middle">AI</text>
</svg>"""
        (project_path / "icon.svg").write_text(icon_svg)
        
        await asyncio.sleep(0.5)
        print("  ✅ 폴더 구조 생성 완료!")
    
    async def open_in_godot(self, project_path: Path) -> bool:
        """Godot 에디터에서 프로젝트 열기"""
        self.godot_exe = self.find_godot_executable()
        if not self.godot_exe:
            print("❌ Godot을 찾을 수 없습니다.")
            return False
        
        print(f"\n🚀 Godot 에디터에서 프로젝트를 엽니다...")
        
        # WSL 경로를 Windows 경로로 변환
        windows_path = str(project_path).replace("/mnt/c", "C:").replace("/mnt/d", "D:").replace("/", "\\")
        
        # Godot 에디터 실행
        cmd = ["cmd.exe", "/c", "start", "", self.godot_exe.replace("/mnt/c", "C:").replace("/mnt/d", "D:").replace("/", "\\"), "--path", windows_path, "--editor"]
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ Godot 에디터가 열렸습니다!")
            print("\n💡 이제 실제 게임 프로젝트를 볼 수 있습니다:")
            print("  - 왼쪽: 파일 시스템 (생성된 파일들)")
            print("  - 중앙: 2D/3D 뷰 (게임 씬)")
            print("  - 오른쪽: 씬 트리 (게임 구조)")
            print("  - 하단: 스크립트 에디터 (게임 코드)")
            return True
        except Exception as e:
            print(f"❌ Godot 실행 중 오류: {e}")
            return False
    
    async def build_and_show_game(self, game_name: str, game_type: str):
        """게임을 빌드하고 Godot에서 보여주기"""
        print("\n" + "="*60)
        print("🎮 실제 게임 제작을 시작합니다!")
        print("💡 로딩창이 아닌 실제 게임 파일이 만들어집니다.")
        print("="*60)
        
        # 시각화 모듈 임포트
        try:
            from modules.game_creation_visualizer import get_game_creation_visualizer
            visualizer = get_game_creation_visualizer()
            
            # 시각화 시작 (비동기 태스크로 실행)
            visualization_task = asyncio.create_task(visualizer.start_visualization(game_type, game_name))
        except:
            visualizer = None
            visualization_task = None
        
        # 1. 게임 프로젝트 생성
        project_path = await self.create_game_project(game_name, game_type)
        
        # 2. Godot 에디터에서 열기
        success = await self.open_in_godot(project_path)
        
        # 시각화 완료 대기
        if visualization_task:
            try:
                await visualization_task
            except:
                pass
        
        if success:
            print("\n" + "="*60)
            print("✅ 게임 제작이 완료되었습니다!")
            print("🎮 Godot 에디터에서 게임을 확인하세요:")
            print("  - F5: 게임 실행")
            print("  - F6: 현재 씬 실행")
            print("  - Ctrl+S: 저장")
            print("💬 이제 게임을 자유롭게 수정하고 발전시킬 수 있습니다!")
            print("="*60)
            
            # 시각화 요약 표시
            if visualizer:
                visualizer.show_creation_summary()
        
        return success


# 전역 인스턴스
_game_builder = None

def get_game_builder() -> GodotGameBuilder:
    """게임 빌더 싱글톤 인스턴스 반환"""
    global _game_builder
    if _game_builder is None:
        _game_builder = GodotGameBuilder()
    return _game_builder