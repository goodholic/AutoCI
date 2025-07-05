"""
Godot 엔진 인터페이스
Panda3D 코드를 Godot으로 변환하고 조작하는 시스템
"""

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class GodotEngineInterface:
    """Godot 엔진 조작 및 변환 인터페이스"""
    
    def __init__(self):
        self.godot_path = self._find_godot_executable()
        self.project_path: Optional[Path] = None
        self.conversion_rules = self._initialize_conversion_rules()
        
    def _find_godot_executable(self) -> Optional[str]:
        """Godot 실행 파일 찾기"""
        # 일반적인 Godot 설치 경로들
        possible_paths = [
            "godot",  # PATH에 있는 경우
            "/usr/local/bin/godot",
            "/usr/bin/godot",
            "C:\\Program Files\\Godot\\godot.exe",
            "C:\\Program Files (x86)\\Godot\\godot.exe",
            os.path.expanduser("~/Applications/Godot.app/Contents/MacOS/Godot")
        ]
        
        for path in possible_paths:
            if os.path.exists(path) or self._check_command_exists(path):
                logger.info(f"Godot 실행 파일 발견: {path}")
                return path
        
        logger.warning("Godot 실행 파일을 찾을 수 없습니다")
        return None
    
    def _check_command_exists(self, command: str) -> bool:
        """명령어 존재 확인"""
        try:
            subprocess.run([command, "--version"], capture_output=True, timeout=5)
            return True
        except:
            return False
    
    def _initialize_conversion_rules(self) -> Dict[str, Dict[str, Any]]:
        """Panda3D to Godot 변환 규칙"""
        return {
            "imports": {
                "from direct.showbase.ShowBase import ShowBase": "extends Node3D",
                "from panda3d.core import *": "# Godot uses built-in classes",
                "from direct.task import Task": "# Use _process or _physics_process",
                "from direct.actor.Actor import Actor": "# Use AnimationPlayer",
                "from direct.gui": "# Use Control nodes"
            },
            "classes": {
                "ShowBase": "Node3D",
                "NodePath": "Node3D",
                "Actor": "CharacterBody3D",
                "CollisionNode": "CollisionShape3D",
                "DirectionalLight": "DirectionalLight3D",
                "AmbientLight": "Environment",
                "PandaNode": "Node3D"
            },
            "methods": {
                "loadModel": "load",
                "reparentTo": "add_child",
                "setPos": "position =",
                "setHpr": "rotation =",
                "setScale": "scale =",
                "setColor": "modulate =",
                "setTexture": "texture =",
                "taskMgr.add": "_process or _physics_process",
                "accept": "_input or connect signal"
            },
            "properties": {
                "render": "get_tree().root",
                "camera": "get_viewport().get_camera_3d()",
                "globalClock": "delta in _process",
                "base": "self"
            }
        }
    
    async def create_godot_project(self, project_name: str, game_type: str) -> bool:
        """새 Godot 프로젝트 생성"""
        try:
            project_path = Path(f"godot_projects/{project_name}")
            project_path.mkdir(parents=True, exist_ok=True)
            self.project_path = project_path
            
            # project.godot 파일 생성
            project_config = f"""[application]

config/name="{project_name}"
config/description="Auto-generated {game_type} game by AutoCI"
run/main_scene="res://scenes/main.tscn"
config/features=PackedStringArray("4.2", "GL Compatibility")
config/icon="res://icon.svg"

[rendering]

renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"
"""
            
            (project_path / "project.godot").write_text(project_config)
            
            # 기본 디렉토리 구조 생성
            for dir_name in ["scenes", "scripts", "assets", "sounds", "textures"]:
                (project_path / dir_name).mkdir(exist_ok=True)
            
            # 메인 씬 생성
            await self._create_main_scene(game_type)
            
            logger.info(f"Godot 프로젝트 생성 완료: {project_path}")
            return True
            
        except Exception as e:
            logger.error(f"Godot 프로젝트 생성 실패: {e}")
            return False
    
    async def _create_main_scene(self, game_type: str):
        """메인 씬 생성"""
        scene_content = """[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://scripts/main.gd" id="1"]

[node name="Main" type="Node3D"]
script = ExtResource("1")

[node name="Camera3D" type="Camera3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 0.866025, 0.5, 0, -0.5, 0.866025, 0, 5, 10)

[node name="DirectionalLight3D" type="DirectionalLight3D" parent="."]
transform = Transform3D(0.707107, -0.5, 0.5, 0, 0.707107, 0.707107, -0.707107, -0.5, 0.5, 0, 0, 0)
"""
        
        # 게임 타입별 추가 노드
        if game_type == "platformer":
            scene_content += """
[node name="Player" type="CharacterBody3D" parent="."]

[node name="Platform" type="StaticBody3D" parent="."]
transform = Transform3D(10, 0, 0, 0, 1, 0, 0, 0, 10, 0, -2, 0)
"""
        elif game_type == "rpg":
            scene_content += """
[node name="Player" type="CharacterBody3D" parent="."]

[node name="World" type="Node3D" parent="."]
"""
        
        (self.project_path / "scenes" / "main.tscn").write_text(scene_content)
        
        # 메인 스크립트 생성
        script_content = self._generate_main_script(game_type)
        (self.project_path / "scripts" / "main.gd").write_text(script_content)
    
    def _generate_main_script(self, game_type: str) -> str:
        """게임 타입별 메인 스크립트 생성"""
        base_script = """extends Node3D

# 게임 상태
var game_started = false
var score = 0

func _ready():
\tprint("Game initialized: %s")
\t_initialize_game()

func _initialize_game():
\t# 게임 초기화 로직
\tpass

func _process(delta):
\tif game_started:
\t\t_update_game(delta)

func _update_game(delta):
\t# 게임 업데이트 로직
\tpass
""" % game_type
        
        # 게임 타입별 추가 코드
        if game_type == "platformer":
            base_script += """
func _physics_process(delta):
\t# 플랫폼 게임 물리 처리
\tpass
"""
        elif game_type == "rpg":
            base_script += """
var player_stats = {
\t"level": 1,
\t"hp": 100,
\t"mp": 50
}

func _on_player_level_up():
\tplayer_stats.level += 1
\tprint("Level up! Now level ", player_stats.level)
"""
        
        return base_script
    
    async def convert_panda3d_to_godot(self, panda3d_code: str) -> str:
        """Panda3D 코드를 Godot GDScript로 변환"""
        godot_code = panda3d_code
        
        # Import 문 변환
        for panda_import, godot_equivalent in self.conversion_rules["imports"].items():
            godot_code = godot_code.replace(panda_import, godot_equivalent)
        
        # 클래스명 변환
        for panda_class, godot_class in self.conversion_rules["classes"].items():
            godot_code = godot_code.replace(panda_class, godot_class)
        
        # 메서드 변환
        for panda_method, godot_method in self.conversion_rules["methods"].items():
            godot_code = godot_code.replace(panda_method, godot_method)
        
        # 속성 변환
        for panda_prop, godot_prop in self.conversion_rules["properties"].items():
            godot_code = godot_code.replace(panda_prop, godot_prop)
        
        # Python to GDScript 문법 변환
        godot_code = self._convert_python_to_gdscript(godot_code)
        
        return godot_code
    
    def _convert_python_to_gdscript(self, code: str) -> str:
        """Python 문법을 GDScript로 변환"""
        # self를 제거하거나 변환
        code = code.replace("self.", "")
        
        # __init__를 _ready로 변환
        code = code.replace("def __init__(self):", "func _ready():")
        
        # def를 func로 변환
        code = code.replace("def ", "func ")
        
        # True/False를 true/false로
        code = code.replace("True", "true")
        code = code.replace("False", "false")
        code = code.replace("None", "null")
        
        # 들여쓰기를 탭으로 변환
        lines = code.split('\n')
        converted_lines = []
        for line in lines:
            # 공백 4개를 탭으로
            space_count = len(line) - len(line.lstrip(' '))
            tabs = '\t' * (space_count // 4)
            converted_lines.append(tabs + line.lstrip())
        
        return '\n'.join(converted_lines)
    
    async def run_godot_project(self) -> bool:
        """Godot 프로젝트 실행"""
        if not self.godot_path or not self.project_path:
            logger.error("Godot 경로 또는 프로젝트 경로가 설정되지 않음")
            return False
        
        try:
            cmd = [self.godot_path, "--path", str(self.project_path)]
            process = subprocess.Popen(cmd)
            logger.info(f"Godot 프로젝트 실행: {self.project_path}")
            return True
        except Exception as e:
            logger.error(f"Godot 프로젝트 실행 실패: {e}")
            return False
    
    def get_conversion_suggestion(self, error_message: str) -> Optional[str]:
        """에러 메시지 기반 변환 제안"""
        suggestions = {
            "ShowBase": "Godot에서는 Node3D를 상속받아 사용하세요",
            "loadModel": "Godot에서는 load() 함수와 instantiate()를 사용하세요",
            "taskMgr": "Godot에서는 _process(delta) 함수를 사용하세요",
            "CollisionNode": "Godot에서는 CollisionShape3D를 사용하세요",
            "Actor": "Godot에서는 AnimationPlayer와 CharacterBody3D를 사용하세요"
        }
        
        for key, suggestion in suggestions.items():
            if key in error_message:
                return suggestion
        
        return None