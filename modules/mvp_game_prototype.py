#!/usr/bin/env python3
"""
MVP Game Prototype System - 최소 기능 제품으로 실제 작동하는 게임 제작
간단하지만 확실하게 작동하는 프로토타입 게임을 만듭니다.
"""

import os
import sys
import json
import shutil
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

class MVPGamePrototype:
    """MVP 게임 프로토타입 제작 시스템"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.games_dir = self.project_root / "mvp_games"
        self.games_dir.mkdir(exist_ok=True)
        self.current_project = None
        self.errors = []
        
    async def create_mvp_game(self, game_name: str = "SimpleGame") -> Dict[str, Any]:
        """MVP 게임 생성 - 간단하지만 완전히 작동하는 게임"""
        print(f"\n🎮 MVP 게임 프로토타입 제작 시작: {game_name}")
        print("=" * 60)
        
        # 1. 프로젝트 디렉토리 생성
        project_path = self.games_dir / game_name
        if project_path.exists():
            i = 1
            while project_path.exists():
                project_path = self.games_dir / f"{game_name}_{i}"
                i += 1
        project_path.mkdir(parents=True)
        self.current_project = project_path
        
        # 2. 기본 디렉토리 구조
        for folder in ["scenes", "scripts", "assets"]:
            (project_path / folder).mkdir()
        
        # 3. project.godot 생성 (Godot 4.3 호환)
        await self._create_project_godot(project_path, game_name)
        
        # 4. 메인 씬 생성 (Main.tscn)
        await self._create_main_scene(project_path)
        
        # 5. 플레이어 씬과 스크립트 생성
        await self._create_player(project_path)
        
        # 6. 게임 월드 생성
        await self._create_game_world(project_path)
        
        # 7. UI 생성
        await self._create_ui(project_path)
        
        # 8. 게임 매니저 스크립트
        await self._create_game_manager(project_path)
        
        # 9. 검증
        validation_result = await self._validate_project(project_path)
        
        # 10. 오류 복구
        if not validation_result["valid"]:
            print("\n🔧 오류가 발견되어 자동 복구를 시도합니다...")
            try:
                from modules.game_error_recovery import get_error_recovery
                recovery = get_error_recovery()
                recovery_result = await recovery.check_and_fix_project(project_path)
                
                if recovery_result["success"]:
                    print("✅ 오류 복구 성공!")
                    # 재검증
                    validation_result = await self._validate_project(project_path)
                else:
                    print(f"⚠️ 일부 오류를 복구하지 못했습니다: {recovery_result['unfixed_errors']}")
            except Exception as e:
                print(f"⚠️ 오류 복구 중 문제 발생: {e}")
        
        return {
            "success": validation_result["valid"],
            "project_path": str(project_path),
            "errors": self.errors,
            "validation": validation_result
        }
    
    async def _create_project_godot(self, project_path: Path, game_name: str):
        """project.godot 파일 생성"""
        print("📄 project.godot 생성 중...")
        
        config = f"""[application]

config/name="{game_name}"
config/description="MVP Game Prototype - Simple but fully functional"
run/main_scene="res://scenes/Main.tscn"
config/features=PackedStringArray("4.3")
config/icon="res://icon.svg"

[display]

window/size/viewport_width=1024
window/size/viewport_height=600

[input]

move_left={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":0,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":65,"physical_keycode":0,"key_label":0,"unicode":0,"echo":false,"script":null)
, Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":0,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":4194319,"physical_keycode":0,"key_label":0,"unicode":0,"echo":false,"script":null)
]
}}
move_right={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":0,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":68,"physical_keycode":0,"key_label":0,"unicode":0,"echo":false,"script":null)
, Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":0,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":4194321,"physical_keycode":0,"key_label":0,"unicode":0,"echo":false,"script":null)
]
}}
jump={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":0,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":32,"physical_keycode":0,"key_label":0,"unicode":0,"echo":false,"script":null)
]
}}

[rendering]

renderer/rendering_method="mobile"
"""
        
        with open(project_path / "project.godot", "w", encoding="utf-8") as f:
            f.write(config)
        
        # 아이콘 생성
        await self._create_icon(project_path)
    
    async def _create_icon(self, project_path: Path):
        """간단한 SVG 아이콘 생성"""
        icon_svg = """<svg height="128" width="128" xmlns="http://www.w3.org/2000/svg">
<rect x="0" y="0" width="128" height="128" fill="#4287f5"/>
<circle cx="64" cy="64" r="40" fill="#ffffff"/>
<text x="64" y="75" font-family="Arial" font-size="30" fill="#4287f5" text-anchor="middle">MVP</text>
</svg>"""
        
        with open(project_path / "icon.svg", "w") as f:
            f.write(icon_svg)
    
    async def _create_main_scene(self, project_path: Path):
        """메인 씬 생성"""
        print("🎬 메인 씬 생성 중...")
        
        # Main.tscn - 게임의 루트 씬
        main_scene = """[gd_scene load_steps=5 format=3 uid="uid://main_scene"]

[ext_resource type="Script" path="res://scripts/GameManager.gd" id="1"]
[ext_resource type="PackedScene" uid="uid://player_scene" path="res://scenes/Player.tscn" id="2"]
[ext_resource type="PackedScene" uid="uid://world_scene" path="res://scenes/World.tscn" id="3"]
[ext_resource type="PackedScene" uid="uid://ui_scene" path="res://scenes/UI.tscn" id="4"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

[node name="World" parent="." instance=ExtResource("3")]

[node name="Player" parent="." instance=ExtResource("2")]
position = Vector2(100, 300)

[node name="UI" parent="." instance=ExtResource("4")]
"""
        
        with open(project_path / "scenes" / "Main.tscn", "w") as f:
            f.write(main_scene)
    
    async def _create_player(self, project_path: Path):
        """플레이어 씬과 스크립트 생성"""
        print("🏃 플레이어 생성 중...")
        
        # Player.tscn
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
        
        with open(project_path / "scenes" / "Player.tscn", "w") as f:
            f.write(player_scene)
        
        # Player.gd - 플레이어 컨트롤 스크립트
        player_script = """extends CharacterBody2D

# 플레이어 설정
const SPEED = 300.0
const JUMP_VELOCITY = -400.0

# 중력 설정
var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

func _physics_process(delta):
    # 중력 적용
    if not is_on_floor():
        velocity.y += gravity * delta
    
    # 점프 처리
    if Input.is_action_just_pressed("jump") and is_on_floor():
        velocity.y = JUMP_VELOCITY
    
    # 좌우 이동 처리
    var direction = Input.get_axis("move_left", "move_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = move_toward(velocity.x, 0, SPEED)
    
    # 물리 이동 적용
    move_and_slide()
    
    # 화면 밖으로 나가지 않도록
    position.x = clamp(position.x, 32, 992)
    
    # 떨어지면 리스폰
    if position.y > 700:
        position = Vector2(100, 300)
        velocity = Vector2.ZERO

func _ready():
    print("Player ready!")
"""
        
        with open(project_path / "scripts" / "Player.gd", "w") as f:
            f.write(player_script)
    
    async def _create_game_world(self, project_path: Path):
        """게임 월드 생성"""
        print("🌍 게임 월드 생성 중...")
        
        # World.tscn - 간단한 플랫폼 레벨
        world_scene = """[gd_scene load_steps=3 format=3 uid="uid://world_scene"]

[sub_resource type="RectangleShape2D" id="RectangleShape2D_ground"]
size = Vector2(1024, 40)

[sub_resource type="PlaceholderTexture2D" id="PlaceholderTexture2D_ground"]
size = Vector2(1024, 40)

[node name="World" type="Node2D"]

[node name="Ground" type="StaticBody2D" parent="."]
position = Vector2(512, 580)

[node name="CollisionShape2D" type="CollisionShape2D" parent="Ground"]
shape = SubResource("RectangleShape2D_ground")

[node name="Sprite2D" type="Sprite2D" parent="Ground"]
texture = SubResource("PlaceholderTexture2D_ground")
modulate = Color(0.4, 0.2, 0.1, 1.0)

[node name="Platform1" type="StaticBody2D" parent="."]
position = Vector2(300, 450)

[node name="CollisionShape2D" type="CollisionShape2D" parent="Platform1"]
shape = SubResource("RectangleShape2D_ground")
scale = Vector2(0.2, 1.0)

[node name="Sprite2D" type="Sprite2D" parent="Platform1"]
texture = SubResource("PlaceholderTexture2D_ground")
scale = Vector2(0.2, 1.0)
modulate = Color(0.6, 0.3, 0.1, 1.0)

[node name="Platform2" type="StaticBody2D" parent="."]
position = Vector2(700, 350)

[node name="CollisionShape2D" type="CollisionShape2D" parent="Platform2"]
shape = SubResource("RectangleShape2D_ground")
scale = Vector2(0.3, 1.0)

[node name="Sprite2D" type="Sprite2D" parent="Platform2"]
texture = SubResource("PlaceholderTexture2D_ground")
scale = Vector2(0.3, 1.0)
modulate = Color(0.6, 0.3, 0.1, 1.0)
"""
        
        with open(project_path / "scenes" / "World.tscn", "w") as f:
            f.write(world_scene)
    
    async def _create_ui(self, project_path: Path):
        """UI 생성"""
        print("🎨 UI 생성 중...")
        
        # UI.tscn
        ui_scene = """[gd_scene load_steps=2 format=3 uid="uid://ui_scene"]

[ext_resource type="Script" path="res://scripts/UI.gd" id="1"]

[node name="UI" type="CanvasLayer"]
script = ExtResource("1")

[node name="ScoreLabel" type="Label" parent="."]
offset_left = 20.0
offset_top = 20.0
offset_right = 200.0
offset_bottom = 50.0
text = "Score: 0"
theme_override_font_sizes/font_size = 24

[node name="InstructionsLabel" type="Label" parent="."]
anchors_preset = 3
anchor_left = 1.0
anchor_top = 1.0
anchor_right = 1.0
anchor_bottom = 1.0
offset_left = -250.0
offset_top = -100.0
offset_right = -20.0
offset_bottom = -20.0
text = "Controls:
A/D or Arrow Keys - Move
Space - Jump"
theme_override_font_sizes/font_size = 16
"""
        
        with open(project_path / "scenes" / "UI.tscn", "w") as f:
            f.write(ui_scene)
        
        # UI.gd
        ui_script = """extends CanvasLayer

var score = 0
var start_time = 0

func _ready():
    start_time = Time.get_ticks_msec()
    print("UI ready!")

func _process(_delta):
    # 시간 기반 점수 시스템
    var elapsed_time = (Time.get_ticks_msec() - start_time) / 1000
    score = int(elapsed_time * 10)
    $ScoreLabel.text = "Score: " + str(score)
"""
        
        with open(project_path / "scripts" / "UI.gd", "w") as f:
            f.write(ui_script)
    
    async def _create_game_manager(self, project_path: Path):
        """게임 매니저 스크립트 생성"""
        print("🎮 게임 매니저 생성 중...")
        
        game_manager = """extends Node2D

# 게임 상태
var game_started = false
var game_paused = false

func _ready():
    print("===== MVP Game Started =====")
    print("Game Manager initialized")
    game_started = true
    
    # 게임 정보 출력
    print("Game: Simple Platform Game")
    print("Controls: A/D to move, Space to jump")
    print("Objective: Survive and collect score!")

func _process(_delta):
    # ESC로 종료
    if Input.is_action_just_pressed("ui_cancel"):
        get_tree().quit()
    
    # P로 일시정지 (옵션)
    if Input.is_key_pressed(KEY_P):
        game_paused = !game_paused
        get_tree().paused = game_paused

func _notification(what):
    if what == NOTIFICATION_WM_CLOSE_REQUEST:
        print("Game closed. Final score: ", get_node("UI").score)
        get_tree().quit()
"""
        
        with open(project_path / "scripts" / "GameManager.gd", "w") as f:
            f.write(game_manager)
    
    async def _validate_project(self, project_path: Path) -> Dict[str, Any]:
        """프로젝트 검증"""
        print("\n🔍 프로젝트 검증 중...")
        
        validation = {
            "valid": True,
            "checks": [],
            "missing_files": []
        }
        
        # 필수 파일 체크
        required_files = [
            "project.godot",
            "icon.svg",
            "scenes/Main.tscn",
            "scenes/Player.tscn",
            "scenes/World.tscn",
            "scenes/UI.tscn",
            "scripts/GameManager.gd",
            "scripts/Player.gd",
            "scripts/UI.gd"
        ]
        
        for file in required_files:
            file_path = project_path / file
            if file_path.exists():
                validation["checks"].append(f"✅ {file}")
            else:
                validation["valid"] = False
                validation["missing_files"].append(file)
                validation["checks"].append(f"❌ {file} - MISSING!")
        
        # 파일 크기 체크
        if (project_path / "project.godot").exists():
            size = (project_path / "project.godot").stat().st_size
            if size > 100:
                validation["checks"].append(f"✅ project.godot 크기: {size} bytes")
            else:
                validation["checks"].append(f"⚠️ project.godot가 너무 작습니다: {size} bytes")
        
        # 결과 출력
        print("\n검증 결과:")
        for check in validation["checks"]:
            print(f"  {check}")
        
        if validation["valid"]:
            print("\n✅ 프로젝트 검증 성공! 게임을 실행할 준비가 되었습니다.")
        else:
            print(f"\n❌ 프로젝트 검증 실패! 누락된 파일: {validation['missing_files']}")
        
        return validation
    
    def get_run_instructions(self, project_path: Path) -> str:
        """게임 실행 방법"""
        return f"""
🎮 게임 실행 방법:
==================
1. Godot 4.3 에디터 열기
2. Project > Import 선택
3. 다음 경로의 project.godot 선택:
   {project_path}/project.godot
4. Import & Edit 클릭
5. F5 또는 Play 버튼으로 게임 실행

또는 명령줄에서:
godot --path "{project_path}"
"""

# 싱글톤 인스턴스
_mvp_instance = None

def get_mvp_prototype():
    """MVP 프로토타입 인스턴스 반환"""
    global _mvp_instance
    if _mvp_instance is None:
        _mvp_instance = MVPGamePrototype()
    return _mvp_instance

async def create_simple_game():
    """간단한 게임 생성 헬퍼 함수"""
    mvp = get_mvp_prototype()
    result = await mvp.create_mvp_game("SimplePlatformer")
    
    if result["success"]:
        print(mvp.get_run_instructions(Path(result["project_path"])))
    
    return result

if __name__ == "__main__":
    # 테스트 실행
    import asyncio
    asyncio.run(create_simple_game())