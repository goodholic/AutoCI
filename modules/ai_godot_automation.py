#!/usr/bin/env python3
"""
AutoCI AI Godot 완전 자동화 시스템
AI가 Godot Editor의 모든 기능을 제어할 수 있는 고급 자동화 시스템
"""

import os
import sys
import json
import asyncio
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import shutil
import time
import re

@dataclass
class GodotProject:
    """Godot 프로젝트 정보"""
    name: str
    path: Path
    project_type: str
    scenes: List[str]
    scripts: List[str]
    resources: List[str]
    settings: Dict[str, Any]
    version: str = "4.3"

@dataclass
class SceneNode:
    """씬 노드 정보"""
    name: str
    type: str
    position: Optional[Tuple[float, float]] = None
    scale: Optional[Tuple[float, float]] = None
    rotation: Optional[float] = None
    properties: Dict[str, Any] = None
    children: List['SceneNode'] = None
    script_path: Optional[str] = None

class AIGodotAutomation:
    """AI Godot 완전 자동화 시스템"""
    
    def __init__(self, godot_executable: str = None):
        self.godot_path = godot_executable or self._find_godot()
        self.logger = logging.getLogger("AIGodotAutomation")
        
        # 게임 타입별 템플릿
        self.game_templates = {
            "platformer": {
                "main_scene": "Main.tscn",
                "player_scene": "Player.tscn", 
                "level_scene": "Level.tscn",
                "ui_scene": "UI.tscn",
                "required_scripts": ["Player.gd", "GameManager.gd", "Enemy.gd"],
                "physics_layers": ["Player", "Enemy", "Platform", "Collectible"],
                "input_map": ["move_left", "move_right", "jump", "dash"]
            },
            "racing": {
                "main_scene": "Main.tscn",
                "car_scene": "Car.tscn",
                "track_scene": "Track.tscn", 
                "ui_scene": "RaceUI.tscn",
                "required_scripts": ["Car.gd", "RaceManager.gd", "AIRacer.gd"],
                "physics_layers": ["Car", "Track", "Obstacle", "Checkpoint"],
                "input_map": ["accelerate", "brake", "steer_left", "steer_right", "handbrake"]
            },
            "puzzle": {
                "main_scene": "Main.tscn",
                "puzzle_scene": "Puzzle.tscn",
                "piece_scene": "PuzzlePiece.tscn",
                "ui_scene": "PuzzleUI.tscn", 
                "required_scripts": ["PuzzleManager.gd", "PuzzlePiece.gd", "LevelLoader.gd"],
                "physics_layers": ["Piece", "Target", "UI"],
                "input_map": ["select", "drag", "rotate", "reset"]
            },
            "rpg": {
                "main_scene": "Main.tscn",
                "player_scene": "Player.tscn",
                "world_scene": "World.tscn",
                "ui_scene": "GameUI.tscn",
                "required_scripts": ["Player.gd", "GameManager.gd", "NPC.gd", "InventorySystem.gd"],
                "physics_layers": ["Player", "NPC", "Enemy", "Environment", "Item"],
                "input_map": ["move_up", "move_down", "move_left", "move_right", "interact", "menu"]
            }
        }
        
        # 스크립트 템플릿
        self.script_templates = self._load_script_templates()
        
        # 씬 구조 템플릿
        self.scene_templates = self._load_scene_templates()
        
    def _find_godot(self) -> str:
        """시스템에서 Godot 실행 파일 찾기"""
        possible_paths = [
            "/usr/local/bin/godot",
            "/usr/bin/godot", 
            "/opt/godot/godot",
            "godot",
            "Godot_v4.3-stable_linux.x86_64"
        ]
        
        for path in possible_paths:
            if shutil.which(path):
                return path
                
        # WSL 환경에서 Windows Godot 찾기
        windows_paths = [
            "/mnt/c/Program Files/Godot/Godot.exe",
            "/mnt/c/Godot/Godot.exe"
        ]
        
        for path in windows_paths:
            if Path(path).exists():
                return path
                
        self.logger.warning("Godot 실행 파일을 찾을 수 없습니다. 수동으로 설정하세요.")
        return "godot"
        
    async def create_complete_game_project(self, project_name: str, project_path: str, 
                                         game_type: str = "platformer") -> GodotProject:
        """완전한 게임 프로젝트 자동 생성"""
        self.logger.info(f"AI가 {game_type} 게임 프로젝트 '{project_name}' 생성 시작")
        
        project_dir = Path(project_path)
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 프로젝트 설정 파일 생성
        await self._create_project_settings(project_dir, project_name, game_type)
        
        # 2. 디렉토리 구조 생성
        await self._create_directory_structure(project_dir, game_type)
        
        # 3. 입력 맵 설정
        await self._setup_input_map(project_dir, game_type)
        
        # 4. 물리 레이어 설정
        await self._setup_physics_layers(project_dir, game_type)
        
        # 5. 씬 파일들 자동 생성
        scenes = await self._create_all_scenes(project_dir, game_type)
        
        # 6. 스크립트 파일들 자동 생성
        scripts = await self._create_all_scripts(project_dir, game_type)
        
        # 7. 리소스 파일들 생성
        resources = await self._create_game_resources(project_dir, game_type)
        
        # 8. 프로젝트 임포트 및 최적화
        await self._import_and_optimize_project(project_dir)
        
        # 9. 자동 테스트 씬 생성
        await self._create_test_scenes(project_dir, game_type)
        
        project = GodotProject(
            name=project_name,
            path=project_dir,
            project_type=game_type,
            scenes=scenes,
            scripts=scripts,
            resources=resources,
            settings=self.game_templates[game_type]
        )
        
        self.logger.info(f"게임 프로젝트 '{project_name}' 생성 완료")
        return project
        
    async def _create_project_settings(self, project_dir: Path, project_name: str, game_type: str):
        """프로젝트 설정 파일 생성"""
        settings = {
            "application/config/name": project_name,
            "application/config/description": f"AI generated {game_type} game",
            "application/run/main_scene": f"res://scenes/{self.game_templates[game_type]['main_scene']}",
            "application/config/icon": "res://assets/icon.png",
            "rendering/renderer/rendering_method": "forward_plus",
            "physics/common/enable_pause_aware_picking": True,
            "debug/file_logging/enable_file_logging": True,
            "gui/common/drop_mouse_on_gui_input_disabled": True
        }
        
        # 게임 타입별 특별 설정
        if game_type == "racing":
            settings.update({
                "physics/3d/default_gravity": 9.8,
                "physics/3d/default_linear_damp": 0.1,
                "physics/3d/default_angular_damp": 0.1
            })
        elif game_type == "platformer":
            settings.update({
                "physics/2d/default_gravity": 980,
                "layer_names/2d_physics/layer_1": "Player",
                "layer_names/2d_physics/layer_2": "Enemy", 
                "layer_names/2d_physics/layer_3": "Platform",
                "layer_names/2d_physics/layer_4": "Collectible"
            })
        elif game_type == "puzzle":
            settings.update({
                "physics/2d/default_gravity": 0,
                "rendering/2d/snap_2d_transforms_to_pixel": True
            })
        elif game_type == "rpg":
            settings.update({
                "physics/2d/default_gravity": 0,
                "layer_names/2d_physics/layer_1": "Player",
                "layer_names/2d_physics/layer_2": "NPC",
                "layer_names/2d_physics/layer_3": "Enemy",
                "layer_names/2d_physics/layer_4": "Environment",
                "layer_names/2d_physics/layer_5": "Item"
            })
            
        # project.godot 파일 생성
        project_file = project_dir / "project.godot"
        with open(project_file, 'w') as f:
            f.write("; Engine configuration file.\n")
            f.write("; It's best edited using the editor UI and not directly,\n")
            f.write("; since the parameters that go here are not all obvious.\n\n")
            
            for key, value in settings.items():
                section = key.split('/')[0]
                f.write(f"[{section}]\n\n")
                if isinstance(value, str):
                    f.write(f'{key}="{value}"\n')
                elif isinstance(value, bool):
                    f.write(f'{key}={str(value).lower()}\n')
                else:
                    f.write(f'{key}={value}\n')
                f.write('\n')
                
        self.logger.info("프로젝트 설정 파일 생성 완료")
        
    async def _create_directory_structure(self, project_dir: Path, game_type: str):
        """프로젝트 디렉토리 구조 생성"""
        directories = [
            "scenes", "scripts", "assets", "assets/textures", "assets/sounds", 
            "assets/music", "assets/fonts", "assets/models", "assets/animations",
            "data", "autoloads", "ui", "effects", "shaders", "tests"
        ]
        
        # 게임 타입별 추가 디렉토리
        if game_type == "racing":
            directories.extend(["tracks", "cars", "environments"])
        elif game_type == "platformer": 
            directories.extend(["levels", "enemies", "items"])
        elif game_type == "puzzle":
            directories.extend(["puzzles", "pieces", "solutions"])
        elif game_type == "rpg":
            directories.extend(["world", "characters", "items", "quests", "dialogue"])
            
        for directory in directories:
            (project_dir / directory).mkdir(parents=True, exist_ok=True)
            
        # .gitkeep 파일로 빈 디렉토리 유지
        for directory in directories:
            gitkeep = project_dir / directory / ".gitkeep"
            gitkeep.touch()
            
        self.logger.info("디렉토리 구조 생성 완료")
        
    async def _setup_input_map(self, project_dir: Path, game_type: str):
        """입력 맵 자동 설정"""
        input_map = self.game_templates[game_type]["input_map"]
        
        # 키 바인딩 설정
        key_bindings = {
            "move_left": ["KEY_A", "KEY_LEFT"],
            "move_right": ["KEY_D", "KEY_RIGHT"], 
            "move_up": ["KEY_W", "KEY_UP"],
            "move_down": ["KEY_S", "KEY_DOWN"],
            "jump": ["KEY_SPACE", "KEY_UP"],
            "dash": ["KEY_SHIFT"],
            "accelerate": ["KEY_W", "KEY_UP"],
            "brake": ["KEY_S", "KEY_DOWN"],
            "steer_left": ["KEY_A", "KEY_LEFT"],
            "steer_right": ["KEY_D", "KEY_RIGHT"],
            "handbrake": ["KEY_SPACE"],
            "select": ["MOUSE_BUTTON_LEFT"],
            "drag": ["MOUSE_BUTTON_LEFT"],
            "rotate": ["KEY_R"],
            "reset": ["KEY_R"],
            "interact": ["KEY_E"],
            "menu": ["KEY_ESCAPE"]
        }
        
        # input_map.tres 파일 생성 (Godot에서 자동으로 project.godot에 통합됨)
        input_config = project_dir / "input_map.tres"
        with open(input_config, 'w') as f:
            f.write("[gd_resource type=\"InputMap\" format=3]\n\n")
            for action in input_map:
                if action in key_bindings:
                    f.write(f"[resource]\n")
                    f.write(f"action_name = \"{action}\"\n")
                    for key in key_bindings[action]:
                        f.write(f"events = [\"{key}\"]\n")
                    f.write("\n")
                    
        self.logger.info("입력 맵 설정 완료")
        
    async def _setup_physics_layers(self, project_dir: Path, game_type: str):
        """물리 레이어 자동 설정"""
        layers = self.game_templates[game_type]["physics_layers"]
        
        # 레이어 설정을 project.godot에 추가하는 것은 _create_project_settings에서 처리
        self.logger.info(f"물리 레이어 설정 완료: {', '.join(layers)}")
        
    async def _create_all_scenes(self, project_dir: Path, game_type: str) -> List[str]:
        """모든 씬 파일 자동 생성"""
        template = self.game_templates[game_type]
        scenes = []
        
        # 메인 씬 생성
        main_scene = await self._create_main_scene(project_dir, game_type)
        scenes.append(main_scene)
        
        # 플레이어/자동차 씬 생성
        if "player_scene" in template:
            player_scene = await self._create_player_scene(project_dir, game_type)
            scenes.append(player_scene)
        elif "car_scene" in template:
            car_scene = await self._create_car_scene(project_dir, game_type)
            scenes.append(car_scene)
            
        # 레벨/트랙/퍼즐 씬 생성
        if "level_scene" in template:
            level_scene = await self._create_level_scene(project_dir, game_type)
            scenes.append(level_scene)
        elif "track_scene" in template:
            track_scene = await self._create_track_scene(project_dir, game_type)
            scenes.append(track_scene)
        elif "puzzle_scene" in template:
            puzzle_scene = await self._create_puzzle_scene(project_dir, game_type)
            scenes.append(puzzle_scene)
        elif "world_scene" in template:
            world_scene = await self._create_world_scene(project_dir, game_type)
            scenes.append(world_scene)
            
        # UI 씬 생성
        ui_scene = await self._create_ui_scene(project_dir, game_type)
        scenes.append(ui_scene)
        
        # 추가 씬들 생성
        additional_scenes = await self._create_additional_scenes(project_dir, game_type)
        scenes.extend(additional_scenes)
        
        return scenes
        
    async def _create_main_scene(self, project_dir: Path, game_type: str) -> str:
        """메인 씬 자동 생성"""
        scene_path = project_dir / "scenes" / "Main.tscn"
        
        # 게임 타입별 메인 씬 구조 생성
        if game_type == "platformer":
            scene_content = self._generate_platformer_main_scene()
        elif game_type == "racing":
            scene_content = self._generate_racing_main_scene()
        elif game_type == "puzzle":
            scene_content = self._generate_puzzle_main_scene()
        elif game_type == "rpg":
            scene_content = self._generate_rpg_main_scene()
        else:
            scene_content = self._generate_default_main_scene()
            
        with open(scene_path, 'w') as f:
            f.write(scene_content)
            
        self.logger.info("메인 씬 생성 완료")
        return "Main.tscn"
        
    def _generate_platformer_main_scene(self) -> str:
        """플랫포머 메인 씬 내용 생성"""
        return '''[gd_scene load_steps=4 format=3 uid="uid://bvxjq8q8q8q8q"]

[ext_resource type="Script" path="res://scripts/GameManager.gd" id="1"]
[ext_resource type="PackedScene" path="res://scenes/Player.tscn" id="2"]
[ext_resource type="PackedScene" path="res://scenes/Level.tscn" id="3"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

[node name="GameManager" type="Node" parent="."]

[node name="Camera2D" type="Camera2D" parent="."]
enabled = true
zoom = Vector2(2, 2)
process_callback = 1
limit_smoothed = true

[node name="Level" parent="." instance=ExtResource("3")]

[node name="Player" parent="." instance=ExtResource("2")]
position = Vector2(100, 400)

[node name="UI" type="CanvasLayer" parent="."]

[node name="DebugLabel" type="Label" parent="UI"]
anchors_preset = 2
anchor_top = 1.0
anchor_bottom = 1.0
offset_left = 10.0
offset_top = -30.0
offset_right = 200.0
offset_bottom = -10.0
text = "FPS: 60"
'''

    def _generate_racing_main_scene(self) -> str:
        """레이싱 메인 씬 내용 생성"""
        return '''[gd_scene load_steps=4 format=3 uid="uid://racing_main"]

[ext_resource type="Script" path="res://scripts/RaceManager.gd" id="1"]
[ext_resource type="PackedScene" path="res://scenes/Track.tscn" id="2"]
[ext_resource type="PackedScene" path="res://scenes/Car.tscn" id="3"]

[node name="Main" type="Node3D"]
script = ExtResource("1")

[node name="Track" parent="." instance=ExtResource("2")]

[node name="PlayerCar" parent="." instance=ExtResource("3")]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5, 0)

[node name="Camera3D" type="Camera3D" parent="PlayerCar"]
transform = Transform3D(1, 0, 0, 0, 0.8, 0.6, 0, -0.6, 0.8, 0, 2, 5)

[node name="UI" type="CanvasLayer" parent="."]

[node name="SpeedLabel" type="Label" parent="UI"]
anchors_preset = 3
anchor_left = 1.0
anchor_top = 1.0
anchor_right = 1.0
anchor_bottom = 1.0
offset_left = -150.0
offset_top = -50.0
offset_right = -10.0
offset_bottom = -10.0
text = "Speed: 0 km/h"
'''

    def _generate_puzzle_main_scene(self) -> str:
        """퍼즐 메인 씬 내용 생성"""
        return '''[gd_scene load_steps=3 format=3 uid="uid://puzzle_main"]

[ext_resource type="Script" path="res://scripts/PuzzleManager.gd" id="1"]
[ext_resource type="PackedScene" path="res://scenes/Puzzle.tscn" id="2"]

[node name="Main" type="Control"]
layout_mode = 3
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0
script = ExtResource("1")

[node name="Background" type="ColorRect" parent="."]
layout_mode = 1
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0
color = Color(0.2, 0.3, 0.4, 1)

[node name="PuzzleArea" type="Control" parent="."]
layout_mode = 1
anchors_preset = 8
anchor_left = 0.5
anchor_top = 0.5
anchor_right = 0.5
anchor_bottom = 0.5
offset_left = -300.0
offset_top = -200.0
offset_right = 300.0
offset_bottom = 200.0

[node name="Puzzle" parent="PuzzleArea" instance=ExtResource("2")]

[node name="UI" type="Control" parent="."]
layout_mode = 1
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0
'''

    def _generate_rpg_main_scene(self) -> str:
        """RPG 메인 씬 내용 생성"""
        return '''[gd_scene load_steps=4 format=3 uid="uid://rpg_main"]

[ext_resource type="Script" path="res://scripts/GameManager.gd" id="1"]
[ext_resource type="PackedScene" path="res://scenes/World.tscn" id="2"]
[ext_resource type="PackedScene" path="res://scenes/Player.tscn" id="3"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

[node name="World" parent="." instance=ExtResource("2")]

[node name="Player" parent="." instance=ExtResource("3")]
position = Vector2(500, 300)

[node name="Camera2D" type="Camera2D" parent="Player"]
zoom = Vector2(1.5, 1.5)
limit_smoothed = true

[node name="UI" type="CanvasLayer" parent="."]

[node name="HealthBar" type="ProgressBar" parent="UI"]
anchors_preset = 2
anchor_top = 1.0
anchor_bottom = 1.0
offset_left = 20.0
offset_top = -80.0
offset_right = 220.0
offset_bottom = -50.0
max_value = 100.0
value = 100.0

[node name="Inventory" type="Control" parent="UI"]
visible = false
layout_mode = 3
anchors_preset = 8
anchor_left = 0.5
anchor_top = 0.5
anchor_right = 0.5
anchor_bottom = 0.5
'''

    def _generate_default_main_scene(self) -> str:
        """기본 메인 씬 내용 생성"""
        return '''[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://scripts/GameManager.gd" id="1"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

[node name="Camera2D" type="Camera2D" parent="."]

[node name="UI" type="CanvasLayer" parent="."]
'''

    async def _create_player_scene(self, project_dir: Path, game_type: str) -> str:
        """플레이어 씬 자동 생성"""
        scene_path = project_dir / "scenes" / "Player.tscn"
        
        if game_type == "platformer":
            content = self._generate_platformer_player_scene()
        elif game_type == "rpg":
            content = self._generate_rpg_player_scene()
        else:
            content = self._generate_default_player_scene()
            
        with open(scene_path, 'w') as f:
            f.write(content)
            
        return "Player.tscn"
        
    def _generate_platformer_player_scene(self) -> str:
        """플랫포머 플레이어 씬 생성"""
        return '''[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://scripts/Player.gd" id="1"]

[sub_resource type="RectangleShape2D" id="RectangleShape2D_1"]
size = Vector2(32, 48)

[sub_resource type="Animation" id="Animation_idle"]
resource_name = "idle"
length = 1.0
loop_mode = 1

[sub_resource type="AnimationLibrary" id="AnimationLibrary_1"]
_data = {
"idle": SubResource("Animation_idle")
}

[node name="Player" type="CharacterBody2D"]
script = ExtResource("1")

[node name="CollisionShape2D" type="CollisionShape2D" parent="."]
shape = SubResource("RectangleShape2D_1")

[node name="Sprite2D" type="Sprite2D" parent="."]
modulate = Color(0.5, 0.8, 1, 1)
texture = preload("res://icon.svg")
region_enabled = true
region_rect = Rect2(0, 0, 32, 48)

[node name="AnimationPlayer" type="AnimationPlayer" parent="."]
libraries = {
"": SubResource("AnimationLibrary_1")
}

[node name="JumpSound" type="AudioStreamPlayer2D" parent="."]

[node name="GroundDetector" type="RayCast2D" parent="."]
target_position = Vector2(0, 25)
collision_mask = 2
'''

    def _generate_rpg_player_scene(self) -> str:
        """RPG 플레이어 씬 생성"""
        return '''[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://scripts/Player.gd" id="1"]

[sub_resource type="CapsuleShape2D" id="CapsuleShape2D_1"]
radius = 16.0
height = 48.0

[sub_resource type="Animation" id="Animation_walk"]
resource_name = "walk"
length = 0.8
loop_mode = 1

[sub_resource type="AnimationLibrary" id="AnimationLibrary_1"]
_data = {
"walk": SubResource("Animation_walk")
}

[node name="Player" type="CharacterBody2D"]
collision_layer = 1
collision_mask = 6
script = ExtResource("1")

[node name="CollisionShape2D" type="CollisionShape2D" parent="."]
shape = SubResource("CapsuleShape2D_1")

[node name="Sprite2D" type="Sprite2D" parent="."]
modulate = Color(1, 0.8, 0.6, 1)
texture = preload("res://icon.svg")

[node name="AnimationPlayer" type="AnimationPlayer" parent="."]
libraries = {
"": SubResource("AnimationLibrary_1")
}

[node name="InteractionArea" type="Area2D" parent="."]
collision_layer = 0
collision_mask = 16

[node name="InteractionShape" type="CollisionShape2D" parent="InteractionArea"]
shape = SubResource("CapsuleShape2D_1")

[node name="HealthComponent" type="Node" parent="."]

[node name="InventoryComponent" type="Node" parent="."]
'''

    def _generate_default_player_scene(self) -> str:
        """기본 플레이어 씬 생성"""
        return '''[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://scripts/Player.gd" id="1"]

[sub_resource type="RectangleShape2D" id="RectangleShape2D_1"]

[node name="Player" type="CharacterBody2D"]
script = ExtResource("1")

[node name="CollisionShape2D" type="CollisionShape2D" parent="."]
shape = SubResource("RectangleShape2D_1")

[node name="Sprite2D" type="Sprite2D" parent="."]
texture = preload("res://icon.svg")
'''

    async def _create_all_scripts(self, project_dir: Path, game_type: str) -> List[str]:
        """모든 스크립트 파일 자동 생성"""
        template = self.game_templates[game_type]
        scripts = []
        
        for script_name in template["required_scripts"]:
            script_path = await self._create_script_file(project_dir, script_name, game_type)
            scripts.append(script_path)
            
        # 추가 유틸리티 스크립트들
        utility_scripts = [
            "Utils.gd", "Constants.gd", "Events.gd", "SaveSystem.gd"
        ]
        
        for script_name in utility_scripts:
            script_path = await self._create_utility_script(project_dir, script_name)
            scripts.append(script_path)
            
        return scripts
        
    async def _create_script_file(self, project_dir: Path, script_name: str, game_type: str) -> str:
        """개별 스크립트 파일 생성"""
        script_path = project_dir / "scripts" / script_name
        
        # 스크립트 내용 생성
        if script_name == "Player.gd":
            content = self._generate_player_script(game_type)
        elif script_name == "GameManager.gd":
            content = self._generate_game_manager_script(game_type)
        elif script_name == "RaceManager.gd":
            content = self._generate_race_manager_script()
        elif script_name == "PuzzleManager.gd":
            content = self._generate_puzzle_manager_script()
        elif script_name == "Enemy.gd":
            content = self._generate_enemy_script(game_type)
        elif script_name == "Car.gd":
            content = self._generate_car_script()
        elif script_name == "AIRacer.gd":
            content = self._generate_ai_racer_script()
        elif script_name == "PuzzlePiece.gd":
            content = self._generate_puzzle_piece_script()
        elif script_name == "LevelLoader.gd":
            content = self._generate_level_loader_script()
        elif script_name == "NPC.gd":
            content = self._generate_npc_script()
        elif script_name == "InventorySystem.gd":
            content = self._generate_inventory_script()
        else:
            content = self._generate_default_script(script_name)
            
        with open(script_path, 'w') as f:
            f.write(content)
            
        self.logger.info(f"스크립트 생성 완료: {script_name}")
        return script_name
        
    def _generate_player_script(self, game_type: str) -> str:
        """플레이어 스크립트 생성"""
        if game_type == "platformer":
            return '''extends CharacterBody2D
class_name Player

# 플레이어 설정
@export var speed: float = 300.0
@export var jump_velocity: float = -400.0
@export var acceleration: float = 2000.0
@export var friction: float = 1500.0
@export var air_resistance: float = 200.0

# 점프 설정
@export var max_jumps: int = 2
var jumps_remaining: int
var coyote_time: float = 0.1
var coyote_timer: float = 0.0
var jump_buffer_time: float = 0.1
var jump_buffer_timer: float = 0.0

# 대시 설정
@export var dash_speed: float = 600.0
@export var dash_duration: float = 0.2
var dash_timer: float = 0.0
var dash_cooldown: float = 1.0
var dash_cooldown_timer: float = 0.0

# 컴포넌트
@onready var sprite = $Sprite2D
@onready var animation_player = $AnimationPlayer
@onready var jump_sound = $JumpSound
@onready var ground_detector = $GroundDetector

# 상태
var was_on_ground: bool = false
var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

signal health_changed(new_health: int)
signal died()

var health: int = 100:
    set(value):
        health = clamp(value, 0, 100)
        health_changed.emit(health)
        if health <= 0:
            die()

func _ready():
    jumps_remaining = max_jumps

func _physics_process(delta):
    handle_gravity(delta)
    handle_ground_detection()
    handle_jump(delta)
    handle_dash(delta)
    handle_movement(delta)
    update_animation()
    move_and_slide()

func handle_gravity(delta):
    if not is_on_floor():
        velocity.y += gravity * delta

func handle_ground_detection():
    var is_on_ground = is_on_floor() or ground_detector.is_colliding()
    
    if is_on_ground and not was_on_ground:
        jumps_remaining = max_jumps
        coyote_timer = coyote_time
    elif not is_on_ground and was_on_ground:
        coyote_timer = coyote_time
    
    was_on_ground = is_on_ground
    
    if coyote_timer > 0:
        coyote_timer -= get_physics_process_delta_time()

func handle_jump(delta):
    if jump_buffer_timer > 0:
        jump_buffer_timer -= delta
    
    if Input.is_action_just_pressed("jump"):
        jump_buffer_timer = jump_buffer_time
    
    if jump_buffer_timer > 0 and can_jump():
        jump()
        jump_buffer_timer = 0

func can_jump() -> bool:
    return (coyote_timer > 0 or jumps_remaining > 0) and jumps_remaining > 0

func jump():
    velocity.y = jump_velocity
    jumps_remaining -= 1
    coyote_timer = 0
    jump_sound.play()

func handle_dash(delta):
    if dash_cooldown_timer > 0:
        dash_cooldown_timer -= delta
    
    if dash_timer > 0:
        dash_timer -= delta
        return
    
    if Input.is_action_just_pressed("dash") and dash_cooldown_timer <= 0:
        start_dash()

func start_dash():
    dash_timer = dash_duration
    dash_cooldown_timer = dash_cooldown
    var dash_direction = Vector2.RIGHT if sprite.scale.x > 0 else Vector2.LEFT
    velocity.x = dash_direction.x * dash_speed

func handle_movement(delta):
    if dash_timer > 0:
        return
    
    var direction = Input.get_axis("move_left", "move_right")
    
    if direction != 0:
        velocity.x = move_toward(velocity.x, direction * speed, acceleration * delta)
        sprite.scale.x = direction
    else:
        var friction_force = friction if is_on_floor() else air_resistance
        velocity.x = move_toward(velocity.x, 0, friction_force * delta)

func update_animation():
    if dash_timer > 0:
        animation_player.play("dash")
    elif not is_on_floor():
        if velocity.y < 0:
            animation_player.play("jump")
        else:
            animation_player.play("fall")
    elif abs(velocity.x) > 50:
        animation_player.play("run")
    else:
        animation_player.play("idle")

func take_damage(amount: int):
    health -= amount

func die():
    died.emit()
    # 죽음 애니메이션 및 처리
    set_physics_process(false)
    animation_player.play("death")
'''
        elif game_type == "rpg":
            return '''extends CharacterBody2D
class_name Player

# 이동 설정
@export var speed: float = 200.0
@export var acceleration: float = 1500.0
@export var friction: float = 1200.0

# 스탯
@export var max_health: int = 100
@export var level: int = 1
@export var experience: int = 0
@export var experience_to_next_level: int = 100

# 컴포넌트
@onready var sprite = $Sprite2D
@onready var animation_player = $AnimationPlayer
@onready var interaction_area = $InteractionArea
@onready var health_component = $HealthComponent
@onready var inventory_component = $InventoryComponent

# 상태
var health: int
var facing_direction: Vector2 = Vector2.DOWN
var interactable_objects: Array = []

signal health_changed(new_health: int, max_health: int)
signal level_up(new_level: int)
signal experience_gained(amount: int)

func _ready():
    health = max_health
    interaction_area.body_entered.connect(_on_interactable_entered)
    interaction_area.body_exited.connect(_on_interactable_exited)

func _physics_process(delta):
    handle_movement(delta)
    handle_interaction()
    update_animation()
    move_and_slide()

func handle_movement(delta):
    var input_vector = Vector2.ZERO
    input_vector.x = Input.get_axis("move_left", "move_right")
    input_vector.y = Input.get_axis("move_up", "move_down")
    
    if input_vector != Vector2.ZERO:
        input_vector = input_vector.normalized()
        facing_direction = input_vector
        velocity = velocity.move_toward(input_vector * speed, acceleration * delta)
    else:
        velocity = velocity.move_toward(Vector2.ZERO, friction * delta)

func handle_interaction():
    if Input.is_action_just_pressed("interact") and interactable_objects.size() > 0:
        var closest_object = get_closest_interactable()
        if closest_object and closest_object.has_method("interact"):
            closest_object.interact(self)

func get_closest_interactable():
    if interactable_objects.is_empty():
        return null
    
    var closest = interactable_objects[0]
    var closest_distance = global_position.distance_to(closest.global_position)
    
    for obj in interactable_objects:
        var distance = global_position.distance_to(obj.global_position)
        if distance < closest_distance:
            closest = obj
            closest_distance = distance
    
    return closest

func update_animation():
    if velocity.length() > 50:
        animation_player.play("walk")
        # 스프라이트 방향 설정
        if facing_direction.x < 0:
            sprite.scale.x = -1
        elif facing_direction.x > 0:
            sprite.scale.x = 1
    else:
        animation_player.play("idle")

func take_damage(amount: int):
    health = max(0, health - amount)
    health_changed.emit(health, max_health)
    
    if health <= 0:
        die()

func heal(amount: int):
    health = min(max_health, health + amount)
    health_changed.emit(health, max_health)

func gain_experience(amount: int):
    experience += amount
    experience_gained.emit(amount)
    
    while experience >= experience_to_next_level:
        level_up_character()

func level_up_character():
    experience -= experience_to_next_level
    level += 1
    experience_to_next_level = int(experience_to_next_level * 1.2)
    max_health += 10
    health = max_health
    level_up.emit(level)

func die():
    # 죽음 처리
    set_physics_process(false)
    modulate = Color.RED

func _on_interactable_entered(body):
    if body.has_method("interact"):
        interactable_objects.append(body)

func _on_interactable_exited(body):
    interactable_objects.erase(body)
'''
        else:
            return '''extends CharacterBody2D
class_name Player

@export var speed: float = 200.0

func _physics_process(delta):
    var direction = Input.get_vector("move_left", "move_right", "move_up", "move_down")
    velocity = direction * speed
    move_and_slide()
'''

    def _generate_game_manager_script(self, game_type: str) -> str:
        """게임 매니저 스크립트 생성"""
        return f'''extends Node
class_name GameManager

# 게임 상태
enum GameState {{
    MENU,
    PLAYING,
    PAUSED,
    GAME_OVER,
    VICTORY
}}

@export var game_type: String = "{game_type}"
var current_state: GameState = GameState.MENU
var score: int = 0
var time_elapsed: float = 0.0
var is_game_active: bool = false

# 신호
signal state_changed(new_state: GameState)
signal score_changed(new_score: int)
signal game_started()
signal game_ended()

func _ready():
    start_game()

func _process(delta):
    if is_game_active:
        time_elapsed += delta
        update_game_logic(delta)

func start_game():
    current_state = GameState.PLAYING
    is_game_active = true
    time_elapsed = 0.0
    score = 0
    game_started.emit()
    state_changed.emit(current_state)

func pause_game():
    if current_state == GameState.PLAYING:
        current_state = GameState.PAUSED
        is_game_active = false
        get_tree().paused = true
        state_changed.emit(current_state)

func resume_game():
    if current_state == GameState.PAUSED:
        current_state = GameState.PLAYING
        is_game_active = true
        get_tree().paused = false
        state_changed.emit(current_state)

func end_game(victory: bool = false):
    is_game_active = false
    current_state = GameState.VICTORY if victory else GameState.GAME_OVER
    game_ended.emit()
    state_changed.emit(current_state)

func add_score(points: int):
    score += points
    score_changed.emit(score)

func update_game_logic(delta: float):
    # 게임 타입별 로직
    match game_type:
        "{game_type}":
            update_{game_type}_logic(delta)

func update_{game_type}_logic(delta: float):
    # {game_type} 특화 게임 로직
    pass

func _input(event):
    if event.is_action_pressed("menu"):
        if current_state == GameState.PLAYING:
            pause_game()
        elif current_state == GameState.PAUSED:
            resume_game()
'''

    async def _create_game_resources(self, project_dir: Path, game_type: str) -> List[str]:
        """게임 리소스 파일들 생성"""
        resources = []
        
        # 기본 아이콘 생성 (간단한 색상 사각형)
        await self._create_default_icon(project_dir)
        resources.append("icon.png")
        
        # 게임 타입별 리소스 생성
        if game_type == "platformer":
            await self._create_platformer_resources(project_dir)
            resources.extend(["player_texture.png", "platform_texture.png", "enemy_texture.png"])
        elif game_type == "racing":
            await self._create_racing_resources(project_dir)
            resources.extend(["car_texture.png", "track_texture.png", "environment_texture.png"])
        elif game_type == "puzzle":
            await self._create_puzzle_resources(project_dir)
            resources.extend(["piece_texture.png", "background_texture.png"])
        elif game_type == "rpg":
            await self._create_rpg_resources(project_dir)
            resources.extend(["character_texture.png", "world_texture.png", "item_texture.png"])
            
        # 오디오 리소스 생성
        await self._create_audio_resources(project_dir, game_type)
        resources.extend(["jump_sound.ogg", "background_music.ogg"])
        
        return resources
        
    async def _import_and_optimize_project(self, project_dir: Path):
        """프로젝트 임포트 및 최적화"""
        try:
            # Godot 프로젝트 임포트
            cmd = [self.godot_path, "--headless", "--path", str(project_dir), "--import"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info("프로젝트 임포트 완료")
            else:
                self.logger.warning(f"프로젝트 임포트 경고: {stderr.decode()}")
                
        except Exception as e:
            self.logger.error(f"프로젝트 임포트 실패: {e}")
            
    async def modify_scene_with_ai(self, project_path: Path, scene_name: str, 
                                 modifications: Dict[str, Any]) -> bool:
        """AI를 통한 씬 수정"""
        scene_path = project_path / "scenes" / scene_name
        
        if not scene_path.exists():
            self.logger.error(f"씬 파일이 존재하지 않습니다: {scene_path}")
            return False
            
        try:
            # 씬 파일 파싱
            with open(scene_path, 'r') as f:
                scene_content = f.read()
                
            # 수정사항 적용
            modified_content = self._apply_scene_modifications(scene_content, modifications)
            
            # 백업 생성
            backup_path = scene_path.with_suffix('.tscn.backup')
            shutil.copy2(scene_path, backup_path)
            
            # 수정된 내용 저장
            with open(scene_path, 'w') as f:
                f.write(modified_content)
                
            self.logger.info(f"씬 수정 완료: {scene_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"씬 수정 실패: {e}")
            return False
            
    def _apply_scene_modifications(self, content: str, modifications: Dict[str, Any]) -> str:
        """씬 수정사항 적용"""
        modified_content = content
        
        for mod_type, mod_data in modifications.items():
            if mod_type == "add_node":
                modified_content = self._add_node_to_scene(modified_content, mod_data)
            elif mod_type == "modify_property":
                modified_content = self._modify_node_property(modified_content, mod_data)
            elif mod_type == "remove_node":
                modified_content = self._remove_node_from_scene(modified_content, mod_data)
                
        return modified_content
        
    async def generate_adaptive_gameplay(self, project_path: Path, game_type: str, 
                                       difficulty_settings: Dict[str, Any]) -> bool:
        """적응형 게임플레이 생성"""
        self.logger.info(f"적응형 게임플레이 생성 시작: {game_type}")
        
        try:
            # 난이도 기반 설정 생성
            adaptive_script = self._generate_adaptive_script(game_type, difficulty_settings)
            
            # 적응형 매니저 스크립트 생성
            script_path = project_path / "scripts" / "AdaptiveManager.gd"
            with open(script_path, 'w') as f:
                f.write(adaptive_script)
                
            # 게임 타입별 적응형 로직 구현
            await self._implement_adaptive_logic(project_path, game_type, difficulty_settings)
            
            self.logger.info("적응형 게임플레이 생성 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"적응형 게임플레이 생성 실패: {e}")
            return False
            
    def _generate_adaptive_script(self, game_type: str, difficulty_settings: Dict[str, Any]) -> str:
        """적응형 매니저 스크립트 생성"""
        return f'''extends Node
class_name AdaptiveManager

# 적응형 게임플레이 매니저
@export var game_type: String = "{game_type}"
@export var base_difficulty: float = 1.0
@export var adaptation_rate: float = 0.1

var current_difficulty: float = 1.0
var player_performance: float = 0.5
var performance_history: Array[float] = []
var adaptation_timer: float = 0.0
var adaptation_interval: float = 30.0

# 난이도 설정
var difficulty_settings = {{
    "enemy_spawn_rate": {difficulty_settings.get("enemy_spawn_rate", 1.0)},
    "enemy_speed": {difficulty_settings.get("enemy_speed", 1.0)},
    "player_damage": {difficulty_settings.get("player_damage", 1.0)},
    "collectible_frequency": {difficulty_settings.get("collectible_frequency", 1.0)}
}}

signal difficulty_changed(new_difficulty: float)

func _ready():
    current_difficulty = base_difficulty

func _process(delta):
    adaptation_timer += delta
    
    if adaptation_timer >= adaptation_interval:
        adapt_difficulty()
        adaptation_timer = 0.0

func record_player_performance(performance: float):
    player_performance = performance
    performance_history.append(performance)
    
    # 히스토리 크기 제한
    if performance_history.size() > 10:
        performance_history.pop_front()

func adapt_difficulty():
    var avg_performance = calculate_average_performance()
    
    # 성능 기반 난이도 조정
    if avg_performance > 0.7:  # 너무 쉬움
        current_difficulty += adaptation_rate
    elif avg_performance < 0.3:  # 너무 어려움
        current_difficulty -= adaptation_rate
    
    current_difficulty = clamp(current_difficulty, 0.5, 2.0)
    apply_difficulty_changes()
    difficulty_changed.emit(current_difficulty)

func calculate_average_performance() -> float:
    if performance_history.is_empty():
        return 0.5
    
    var sum = 0.0
    for performance in performance_history:
        sum += performance
    
    return sum / performance_history.size()

func apply_difficulty_changes():
    # 게임 타입별 난이도 적용
    match game_type:
        "platformer":
            apply_platformer_difficulty()
        "racing":
            apply_racing_difficulty()
        "puzzle":
            apply_puzzle_difficulty()
        "rpg":
            apply_rpg_difficulty()

func apply_platformer_difficulty():
    # 플랫포머 난이도 조정
    var enemies = get_tree().get_nodes_in_group("enemies")
    for enemy in enemies:
        if enemy.has_method("set_difficulty"):
            enemy.set_difficulty(current_difficulty)

func apply_racing_difficulty():
    # 레이싱 난이도 조정
    var ai_racers = get_tree().get_nodes_in_group("ai_racers")
    for racer in ai_racers:
        if racer.has_method("set_ai_difficulty"):
            racer.set_ai_difficulty(current_difficulty)

func apply_puzzle_difficulty():
    # 퍼즐 난이도 조정
    var puzzle_manager = get_tree().get_first_node_in_group("puzzle_manager")
    if puzzle_manager and puzzle_manager.has_method("set_difficulty"):
        puzzle_manager.set_difficulty(current_difficulty)

func apply_rpg_difficulty():
    # RPG 난이도 조정
    var enemies = get_tree().get_nodes_in_group("enemies")
    for enemy in enemies:
        if enemy.has_method("adjust_stats"):
            enemy.adjust_stats(current_difficulty)
'''

    async def create_procedural_content(self, project_path: Path, content_type: str, 
                                      parameters: Dict[str, Any]) -> bool:
        """절차적 콘텐츠 생성"""
        self.logger.info(f"절차적 콘텐츠 생성: {content_type}")
        
        try:
            if content_type == "level":
                return await self._create_procedural_level(project_path, parameters)
            elif content_type == "enemy":
                return await self._create_procedural_enemy(project_path, parameters)
            elif content_type == "item":
                return await self._create_procedural_item(project_path, parameters)
            elif content_type == "quest":
                return await self._create_procedural_quest(project_path, parameters)
            else:
                self.logger.warning(f"지원하지 않는 콘텐츠 타입: {content_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"절차적 콘텐츠 생성 실패: {e}")
            return False
            
    def _load_script_templates(self) -> Dict[str, str]:
        """스크립트 템플릿 로드"""
        # 실제로는 외부 파일에서 로드하거나 더 복잡한 템플릿 시스템 사용
        return {
            "basic_enemy": """extends CharacterBody2D
class_name Enemy

@export var speed: float = 100.0
@export var health: int = 50
@export var damage: int = 10

func _ready():
    add_to_group("enemies")

func take_damage(amount: int):
    health -= amount
    if health <= 0:
        die()

func die():
    queue_free()
""",
            "collectible": """extends Area2D
class_name Collectible

@export var value: int = 10
@export var auto_collect: bool = true

signal collected(collector)

func _ready():
    if auto_collect:
        body_entered.connect(_on_body_entered)

func _on_body_entered(body):
    if body.has_method("collect_item"):
        body.collect_item(self)
        collected.emit(body)
        queue_free()
""",
            "interactive_object": """extends StaticBody2D
class_name InteractiveObject

@export var interaction_text: String = "Press E to interact"
@export var single_use: bool = false

var is_used: bool = false

signal interacted(interactor)

func interact(interactor):
    if single_use and is_used:
        return
    
    is_used = true
    interacted.emit(interactor)
    perform_interaction(interactor)

func perform_interaction(interactor):
    # 상속받은 클래스에서 구현
    pass
"""
        }
        
    def _load_scene_templates(self) -> Dict[str, Any]:
        """씬 템플릿 로드"""
        return {
            "enemy_spawner": {
                "type": "Node2D",
                "script": "res://scripts/EnemySpawner.gd",
                "children": [
                    {"type": "Timer", "name": "SpawnTimer"},
                    {"type": "Marker2D", "name": "SpawnPoint"}
                ]
            },
            "collectible_item": {
                "type": "Area2D",
                "script": "res://scripts/Collectible.gd",
                "children": [
                    {"type": "Sprite2D", "name": "Sprite"},
                    {"type": "CollisionShape2D", "name": "CollisionShape"},
                    {"type": "AnimationPlayer", "name": "AnimationPlayer"}
                ]
            }
        }

def main():
    """메인 실행 함수"""
    print("🤖 AI Godot 완전 자동화 시스템")
    print("=" * 60)
    
    # 사용 예제
    automation = AIGodotAutomation()
    
    # 비동기 테스트 함수
    async def test_automation():
        project = await automation.create_complete_game_project(
            "AI_Generated_Game",
            "/tmp/test_game",
            "platformer"
        )
        print(f"프로젝트 생성 완료: {project.name}")
        
        # 적응형 게임플레이 추가
        await automation.generate_adaptive_gameplay(
            project.path,
            "platformer",
            {"enemy_spawn_rate": 1.2, "enemy_speed": 1.1}
        )
        
        # 절차적 레벨 생성
        await automation.create_procedural_content(
            project.path,
            "level",
            {"width": 1000, "height": 600, "platform_density": 0.3}
        )
        
    # 비동기 실행
    asyncio.run(test_automation())

if __name__ == "__main__":
    main()