#!/usr/bin/env python3
"""
자동 리소스 생성기 - 누락된 리소스를 자동으로 생성합니다.
오류로 인해 필요한 파일이 없다면, 즉시 만들어냅니다.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime

class AutoResourceGenerator:
    """자동 리소스 생성기"""
    
    def __init__(self):
        self.generated_resources = []
        self.resource_templates = {
            ".png": self._generate_image,
            ".jpg": self._generate_image,
            ".svg": self._generate_svg,
            ".ogg": self._generate_audio,
            ".wav": self._generate_audio,
            ".mp3": self._generate_audio,
            ".tscn": self._generate_scene,
            ".gd": self._generate_script,
            ".tres": self._generate_resource,
            ".material": self._generate_material,
            ".shader": self._generate_shader,
            ".json": self._generate_json,
            ".cfg": self._generate_config,
            ".txt": self._generate_text
        }
    
    async def generate_missing_resource(self, resource_path: str, project_path: Path) -> bool:
        """누락된 리소스 자동 생성"""
        # res:// 경로를 실제 경로로 변환
        if resource_path.startswith("res://"):
            relative_path = resource_path.replace("res://", "")
        else:
            relative_path = resource_path
        
        full_path = project_path / relative_path
        
        # 이미 존재하면 스킵
        if full_path.exists():
            return True
        
        print(f"🔨 누락된 리소스 생성: {relative_path}")
        
        # 디렉토리 생성
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 확장자에 따라 적절한 리소스 생성
        extension = full_path.suffix.lower()
        
        if extension in self.resource_templates:
            generator = self.resource_templates[extension]
            success = await generator(full_path)
            
            if success:
                self.generated_resources.append(str(full_path))
                print(f"  ✅ 생성 완료: {full_path.name}")
                return True
            else:
                print(f"  ❌ 생성 실패: {full_path.name}")
                return False
        else:
            # 알 수 없는 확장자는 빈 파일 생성
            full_path.touch()
            print(f"  ✅ 빈 파일 생성: {full_path.name}")
            return True
    
    async def _generate_image(self, path: Path) -> bool:
        """이미지 파일 생성"""
        if path.suffix == ".svg":
            return await self._generate_svg(path)
        
        # PNG/JPG는 작은 이미지 데이터 생성
        # 실제로는 PIL/Pillow 사용
        # 여기서는 최소한의 PNG 헤더
        png_header = b'\x89PNG\r\n\x1a\n'
        
        try:
            with open(path, 'wb') as f:
                f.write(png_header)
            return True
        except:
            return False
    
    async def _generate_svg(self, path: Path) -> bool:
        """SVG 이미지 생성"""
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F"]
        color = random.choice(colors)
        
        svg_content = f"""<svg width="128" height="128" xmlns="http://www.w3.org/2000/svg">
  <rect width="128" height="128" fill="{color}"/>
  <circle cx="64" cy="64" r="40" fill="white" opacity="0.8"/>
  <text x="64" y="70" font-family="Arial" font-size="24" fill="{color}" text-anchor="middle">{path.stem[:3].upper()}</text>
</svg>"""
        
        try:
            path.write_text(svg_content)
            return True
        except:
            return False
    
    async def _generate_audio(self, path: Path) -> bool:
        """오디오 파일 생성 (무음)"""
        # 최소한의 WAV 헤더 (무음)
        wav_header = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
        
        try:
            with open(path, 'wb') as f:
                f.write(wav_header)
            return True
        except:
            return False
    
    async def _generate_scene(self, path: Path) -> bool:
        """씬 파일 생성"""
        scene_name = path.stem
        
        # 씬 타입 추론
        if "player" in scene_name.lower():
            scene_content = self._get_player_scene_template(scene_name)
        elif "enemy" in scene_name.lower():
            scene_content = self._get_enemy_scene_template(scene_name)
        elif "item" in scene_name.lower():
            scene_content = self._get_item_scene_template(scene_name)
        elif "ui" in scene_name.lower():
            scene_content = self._get_ui_scene_template(scene_name)
        else:
            scene_content = self._get_basic_scene_template(scene_name)
        
        try:
            path.write_text(scene_content)
            return True
        except:
            return False
    
    async def _generate_script(self, path: Path) -> bool:
        """스크립트 파일 생성"""
        script_name = path.stem
        
        # 스크립트 타입 추론
        if "player" in script_name.lower():
            script_content = self._get_player_script_template()
        elif "enemy" in script_name.lower():
            script_content = self._get_enemy_script_template()
        elif "manager" in script_name.lower():
            script_content = self._get_manager_script_template()
        elif "ui" in script_name.lower():
            script_content = self._get_ui_script_template()
        else:
            script_content = self._get_basic_script_template()
        
        try:
            path.write_text(script_content)
            return True
        except:
            return False
    
    async def _generate_resource(self, path: Path) -> bool:
        """리소스 파일 생성"""
        resource_content = """[gd_resource type="Resource" format=3]

[resource]
"""
        
        try:
            path.write_text(resource_content)
            return True
        except:
            return False
    
    async def _generate_material(self, path: Path) -> bool:
        """머티리얼 파일 생성"""
        material_content = """[gd_resource type="StandardMaterial3D" format=3]

[resource]
albedo_color = Color(1, 1, 1, 1)
"""
        
        try:
            path.write_text(material_content)
            return True
        except:
            return False
    
    async def _generate_shader(self, path: Path) -> bool:
        """셰이더 파일 생성"""
        shader_content = """shader_type canvas_item;

void fragment() {
    COLOR = texture(TEXTURE, UV);
}
"""
        
        try:
            path.write_text(shader_content)
            return True
        except:
            return False
    
    async def _generate_json(self, path: Path) -> bool:
        """JSON 파일 생성"""
        json_content = {
            "generated": True,
            "timestamp": str(datetime.now()),
            "name": path.stem
        }
        
        try:
            path.write_text(json.dumps(json_content, indent=2))
            return True
        except:
            return False
    
    async def _generate_config(self, path: Path) -> bool:
        """설정 파일 생성"""
        config_content = f"""# Auto-generated config
[general]
name = "{path.stem}"
version = "1.0"
"""
        
        try:
            path.write_text(config_content)
            return True
        except:
            return False
    
    async def _generate_text(self, path: Path) -> bool:
        """텍스트 파일 생성"""
        text_content = f"""Auto-generated file: {path.name}
Created by AutoCI Resource Generator
Timestamp: {datetime.now()}
"""
        
        try:
            path.write_text(text_content)
            return True
        except:
            return False
    
    def _get_player_scene_template(self, name: str) -> str:
        """플레이어 씬 템플릿"""
        return f"""[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://scripts/{name}.gd" id="1"]

[sub_resource type="RectangleShape2D" id="1"]
size = Vector2(32, 64)

[node name="{name}" type="CharacterBody2D"]
script = ExtResource("1")

[node name="CollisionShape2D" type="CollisionShape2D" parent="."]
shape = SubResource("1")

[node name="Sprite2D" type="Sprite2D" parent="."]
"""
    
    def _get_enemy_scene_template(self, name: str) -> str:
        """적 씬 템플릿"""
        return f"""[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://scripts/{name}.gd" id="1"]

[sub_resource type="CircleShape2D" id="1"]
radius = 16.0

[node name="{name}" type="CharacterBody2D"]
script = ExtResource("1")

[node name="CollisionShape2D" type="CollisionShape2D" parent="."]
shape = SubResource("1")

[node name="Sprite2D" type="Sprite2D" parent="."]
modulate = Color(1, 0.2, 0.2, 1)
"""
    
    def _get_item_scene_template(self, name: str) -> str:
        """아이템 씬 템플릿"""
        return f"""[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://scripts/{name}.gd" id="1"]

[sub_resource type="RectangleShape2D" id="1"]
size = Vector2(16, 16)

[node name="{name}" type="Area2D"]
script = ExtResource("1")

[node name="CollisionShape2D" type="CollisionShape2D" parent="."]
shape = SubResource("1")

[node name="Sprite2D" type="Sprite2D" parent="."]
modulate = Color(1, 1, 0, 1)
"""
    
    def _get_ui_scene_template(self, name: str) -> str:
        """UI 씬 템플릿"""
        return f"""[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://scripts/{name}.gd" id="1"]

[node name="{name}" type="Control"]
anchor_right = 1.0
anchor_bottom = 1.0
script = ExtResource("1")

[node name="Label" type="Label" parent="."]
offset_right = 200.0
offset_bottom = 50.0
text = "{name}"
"""
    
    def _get_basic_scene_template(self, name: str) -> str:
        """기본 씬 템플릿"""
        return f"""[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://scripts/{name}.gd" id="1"]

[node name="{name}" type="Node2D"]
script = ExtResource("1")
"""
    
    def _get_player_script_template(self) -> str:
        """플레이어 스크립트 템플릿"""
        return """extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

func _ready():
    print("Player ready!")

func _physics_process(delta):
    if not is_on_floor():
        velocity.y += gravity * delta
    
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = JUMP_VELOCITY
    
    var direction = Input.get_axis("ui_left", "ui_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = move_toward(velocity.x, 0, SPEED)
    
    move_and_slide()
"""
    
    def _get_enemy_script_template(self) -> str:
        """적 스크립트 템플릿"""
        return """extends CharacterBody2D

const SPEED = 100.0
var direction = 1

func _ready():
    print("Enemy ready!")

func _physics_process(delta):
    velocity.x = direction * SPEED
    
    if is_on_wall():
        direction *= -1
    
    move_and_slide()
"""
    
    def _get_manager_script_template(self) -> str:
        """매니저 스크립트 템플릿"""
        return """extends Node

signal game_started
signal game_ended

var score = 0
var game_active = false

func _ready():
    print("Manager ready!")

func start_game():
    game_active = true
    score = 0
    emit_signal("game_started")

func end_game():
    game_active = false
    emit_signal("game_ended")

func add_score(points: int):
    score += points
"""
    
    def _get_ui_script_template(self) -> str:
        """UI 스크립트 템플릿"""
        return """extends Control

func _ready():
    print("UI ready!")

func update_display(text: String):
    if has_node("Label"):
        $Label.text = text

func show():
    visible = true

func hide():
    visible = false
"""
    
    def _get_basic_script_template(self) -> str:
        """기본 스크립트 템플릿"""
        return """extends Node

func _ready():
    print("Node ready!")

func _process(delta):
    pass
"""

# 싱글톤 인스턴스
_generator_instance = None

def get_resource_generator():
    """리소스 생성기 인스턴스 반환"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = AutoResourceGenerator()
    return _generator_instance