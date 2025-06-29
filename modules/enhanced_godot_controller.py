#!/usr/bin/env python3
"""
Enhanced Godot Controller - 상용화 수준의 Godot Editor 제어
"""

import os
import asyncio
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import shutil
import tempfile
import re

class EnhancedGodotController:
    """향상된 Godot 제어 시스템"""
    
    def __init__(self, godot_path: Optional[str] = None):
        self.logger = logging.getLogger("GodotController")
        
        # Godot 경로 설정
        self.godot_path = godot_path or self._find_godot_executable()
        if not self.godot_path:
            self.logger.warning("Godot 실행 파일을 찾을 수 없습니다")
            self.godot_path = "godot"  # 시스템 PATH에서 찾기
        
        # 프로젝트 관리
        self.active_projects: Dict[str, Dict[str, Any]] = {}
        
        # 템플릿 경로
        self.templates_dir = Path(__file__).parent.parent / "templates" / "godot"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # 설정
        self.config = {
            "default_resolution": (1280, 720),
            "default_fps": 60,
            "physics_fps": 60,
            "render_thread_mode": 1,  # Single-safe
            "audio_driver": "Dummy",  # Headless용
            "display_driver": "headless"
        }
        
        # 프로젝트 템플릿
        self._create_templates()
    
    def _find_godot_executable(self) -> Optional[str]:
        """Godot 실행 파일 찾기"""
        possible_paths = [
            "/usr/local/bin/godot",
            "/usr/bin/godot",
            "/opt/godot/godot",
            "~/godot/godot",
            "./godot_engine/godot"
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path) and os.access(expanded_path, os.X_OK):
                self.logger.info(f"Godot 찾음: {expanded_path}")
                return expanded_path
        
        # PATH에서 찾기
        result = subprocess.run(["which", "godot"], capture_output=True, text=True)
        if result.returncode == 0:
            path = result.stdout.strip()
            self.logger.info(f"Godot 찾음 (PATH): {path}")
            return path
        
        return None
    
    def _create_templates(self):
        """프로젝트 템플릿 생성"""
        # 기본 프로젝트 설정 템플릿
        project_template = """[application]
config/name="{project_name}"
config/features=PackedStringArray("4.2", "GL Compatibility")
run/main_scene="res://scenes/Main.tscn"
config/icon="res://icon.svg"

[display]
window/size/viewport_width={width}
window/size/viewport_height={height}

[physics]
common/physics_fps={physics_fps}

[rendering]
renderer/rendering_method="gl_compatibility"
environment/defaults/default_clear_color=Color(0.1, 0.1, 0.2, 1)
"""
        
        template_path = self.templates_dir / "project.godot.template"
        template_path.write_text(project_template)
        
        # 게임 타입별 템플릿
        self._create_game_templates()
    
    def _create_game_templates(self):
        """게임 타입별 템플릿 생성"""
        # Racing 게임 템플릿
        racing_template = {
            "player_controller": '''extends CharacterBody2D

const ACCELERATION = 500.0
const MAX_SPEED = 400.0
const ROTATION_SPEED = 2.0
const FRICTION = 200.0

func _physics_process(delta):
    var input_vector = Vector2.ZERO
    input_vector.y = Input.get_axis("ui_up", "ui_down")
    var rotation_dir = Input.get_axis("ui_left", "ui_right")
    
    if input_vector.y < 0:  # 전진
        velocity += transform.y * input_vector.y * ACCELERATION * delta
    else:  # 후진
        velocity += transform.y * input_vector.y * ACCELERATION * 0.5 * delta
    
    # 속도 제한
    velocity = velocity.limit_length(MAX_SPEED)
    
    # 회전
    if velocity.length() > 10:
        rotation += rotation_dir * ROTATION_SPEED * delta
    
    # 마찰
    velocity = velocity.move_toward(Vector2.ZERO, FRICTION * delta)
    
    move_and_slide()
''',
            "track_generator": '''extends Node2D

@export var checkpoint_scene: PackedScene
@export var track_width: float = 200.0
@export var track_points: Array[Vector2] = []

func _ready():
    generate_track()

func generate_track():
    # 트랙 포인트 생성
    var angle = 0.0
    var radius = 500.0
    
    for i in range(16):
        angle = i * TAU / 16
        var point = Vector2(
            cos(angle) * radius + randf_range(-50, 50),
            sin(angle) * radius + randf_range(-50, 50)
        )
        track_points.append(point)
    
    # 체크포인트 배치
    for i in range(track_points.size()):
        var checkpoint = checkpoint_scene.instantiate()
        checkpoint.position = track_points[i]
        checkpoint.rotation = angle_to_point(track_points[i], track_points[(i + 1) % track_points.size()])
        add_child(checkpoint)

func angle_to_point(from: Vector2, to: Vector2) -> float:
    return (to - from).angle()
'''
        }
        
        # Platformer 게임 템플릿
        platformer_template = {
            "player_controller": '''extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0
const GRAVITY = 980.0
const COYOTE_TIME = 0.1
const JUMP_BUFFER_TIME = 0.1

var coyote_timer = 0.0
var jump_buffer_timer = 0.0
var jump_count = 0
const MAX_JUMPS = 2

func _physics_process(delta):
    # 중력
    if not is_on_floor():
        velocity.y += GRAVITY * delta
        if coyote_timer > 0:
            coyote_timer -= delta
    else:
        jump_count = 0
        coyote_timer = COYOTE_TIME
    
    # 점프
    if Input.is_action_just_pressed("ui_accept"):
        jump_buffer_timer = JUMP_BUFFER_TIME
    
    if jump_buffer_timer > 0:
        jump_buffer_timer -= delta
        
        if is_on_floor() or (coyote_timer > 0 and jump_count == 0):
            velocity.y = JUMP_VELOCITY
            jump_count = 1
            jump_buffer_timer = 0
        elif jump_count < MAX_JUMPS:
            velocity.y = JUMP_VELOCITY * 0.8  # 더블 점프는 약간 약하게
            jump_count += 1
            jump_buffer_timer = 0
    
    # 좌우 이동
    var direction = Input.get_axis("ui_left", "ui_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = move_toward(velocity.x, 0, SPEED * delta)
    
    move_and_slide()
''',
            "level_generator": '''extends Node2D

@export var platform_scene: PackedScene
@export var enemy_scene: PackedScene
@export var collectible_scene: PackedScene

func _ready():
    generate_level()

func generate_level():
    # 플랫폼 생성
    var platforms = [
        {"pos": Vector2(0, 0), "size": Vector2(800, 40)},
        {"pos": Vector2(300, -150), "size": Vector2(200, 40)},
        {"pos": Vector2(600, -300), "size": Vector2(200, 40)},
        {"pos": Vector2(-300, -200), "size": Vector2(150, 40)},
        {"pos": Vector2(900, -400), "size": Vector2(300, 40)}
    ]
    
    for platform_data in platforms:
        create_platform(platform_data.pos, platform_data.size)
    
    # 적 배치
    var enemy_positions = [
        Vector2(400, -50),
        Vector2(700, -350),
        Vector2(-200, -250)
    ]
    
    for pos in enemy_positions:
        create_enemy(pos)
    
    # 수집품 배치
    var collectible_positions = [
        Vector2(150, -100),
        Vector2(450, -250),
        Vector2(750, -450)
    ]
    
    for pos in collectible_positions:
        create_collectible(pos)

func create_platform(pos: Vector2, size: Vector2):
    var platform = platform_scene.instantiate()
    platform.position = pos
    platform.scale = size / Vector2(100, 20)  # 기본 크기 대비 스케일
    add_child(platform)

func create_enemy(pos: Vector2):
    var enemy = enemy_scene.instantiate()
    enemy.position = pos
    add_child(enemy)

func create_collectible(pos: Vector2):
    var collectible = collectible_scene.instantiate()
    collectible.position = pos
    add_child(collectible)
'''
        }
        
        # 템플릿 저장
        for game_type, templates in [("racing", racing_template), ("platformer", platformer_template)]:
            type_dir = self.templates_dir / game_type
            type_dir.mkdir(exist_ok=True)
            
            for name, content in templates.items():
                file_path = type_dir / f"{name}.gd"
                file_path.write_text(content)
    
    async def create_project(self, project_name: str, project_path: str, 
                           game_type: str = "platformer") -> bool:
        """프로젝트 생성"""
        try:
            project_dir = Path(project_path)
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # project.godot 생성
            template_path = self.templates_dir / "project.godot.template"
            if template_path.exists():
                content = template_path.read_text()
                content = content.format(
                    project_name=project_name,
                    width=self.config["default_resolution"][0],
                    height=self.config["default_resolution"][1],
                    physics_fps=self.config["physics_fps"]
                )
                
                project_file = project_dir / "project.godot"
                project_file.write_text(content)
            else:
                # 기본 project.godot 생성
                await self._create_default_project_file(project_dir, project_name)
            
            # 디렉토리 구조 생성
            dirs = ["scenes", "scripts", "assets", "assets/sprites", 
                   "assets/sounds", "assets/fonts", "resources"]
            for dir_name in dirs:
                (project_dir / dir_name).mkdir(exist_ok=True)
            
            # 기본 씬 생성
            await self._create_main_scene(project_dir, game_type)
            
            # 게임 타입별 스크립트 복사
            await self._copy_game_templates(project_dir, game_type)
            
            # 프로젝트 정보 저장
            self.active_projects[project_name] = {
                "path": str(project_dir),
                "type": game_type,
                "created": datetime.now().isoformat(),
                "scenes": ["Main.tscn"],
                "scripts": []
            }
            
            self.logger.info(f"프로젝트 생성 완료: {project_name} ({game_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"프로젝트 생성 실패: {e}")
            return False
    
    async def _create_default_project_file(self, project_dir: Path, project_name: str):
        """기본 project.godot 파일 생성"""
        content = f'''[application]
config/name="{project_name}"
config/features=PackedStringArray("4.2", "GL Compatibility")
run/main_scene="res://scenes/Main.tscn"

[rendering]
renderer/rendering_method="gl_compatibility"
'''
        (project_dir / "project.godot").write_text(content)
    
    async def _create_main_scene(self, project_dir: Path, game_type: str):
        """메인 씬 생성"""
        scene_content = '''[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://scripts/Main.gd" id="1"]
[ext_resource type="Script" path="res://scripts/Player.gd" id="2"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

[node name="Player" type="CharacterBody2D" parent="."]
position = Vector2(640, 360)
script = ExtResource("2")

[node name="CollisionShape2D" type="CollisionShape2D" parent="Player"]

[node name="Sprite2D" type="Sprite2D" parent="Player"]
'''
        
        scene_path = project_dir / "scenes" / "Main.tscn"
        scene_path.write_text(scene_content)
        
        # 메인 스크립트 생성
        main_script = '''extends Node2D

func _ready():
    print("Game started: %s" % get_tree().current_scene.name)
    setup_game()

func setup_game():
    # 게임 초기화
    pass
'''
        
        script_path = project_dir / "scripts" / "Main.gd"
        script_path.write_text(main_script)
    
    async def _copy_game_templates(self, project_dir: Path, game_type: str):
        """게임 템플릿 복사"""
        template_dir = self.templates_dir / game_type
        
        if template_dir.exists():
            for template_file in template_dir.glob("*.gd"):
                dest_path = project_dir / "scripts" / template_file.name
                shutil.copy2(template_file, dest_path)
                
                # 프로젝트 정보에 스크립트 추가
                if hasattr(self, 'active_projects'):
                    project = next((p for p in self.active_projects.values() 
                                  if p["path"] == str(project_dir)), None)
                    if project:
                        project["scripts"].append(template_file.name)
    
    async def run_editor(self, project_path: str) -> subprocess.Popen:
        """에디터 실행"""
        try:
            cmd = [self.godot_path, "--editor", "--path", project_path]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.logger.info(f"Godot 에디터 실행: {project_path}")
            return process
            
        except Exception as e:
            self.logger.error(f"에디터 실행 실패: {e}")
            return None
    
    async def run_headless(self, project_path: str, script: str = None) -> Dict[str, Any]:
        """헤드리스 모드 실행"""
        try:
            cmd = [
                self.godot_path,
                "--headless",
                "--path", project_path
            ]
            
            if script:
                cmd.extend(["--script", script])
            
            # 환경 변수 설정
            env = os.environ.copy()
            env["GODOT_DISPLAY_DRIVER"] = "headless"
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await result.communicate()
            
            return {
                "success": result.returncode == 0,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "returncode": result.returncode
            }
            
        except Exception as e:
            self.logger.error(f"헤드리스 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_scene(self, project_path: str, scene_name: str, 
                         scene_type: str = "2D") -> bool:
        """씬 생성"""
        try:
            project_dir = Path(project_path)
            scene_path = project_dir / "scenes" / f"{scene_name}.tscn"
            
            # 씬 타입별 기본 구조
            if scene_type == "2D":
                content = f'''[gd_scene format=3]

[node name="{scene_name}" type="Node2D"]
'''
            elif scene_type == "3D":
                content = f'''[gd_scene format=3]

[node name="{scene_name}" type="Node3D"]

[node name="Camera3D" type="Camera3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 0.866025, 0.5, 0, -0.5, 0.866025, 0, 5, 10)

[node name="DirectionalLight3D" type="DirectionalLight3D" parent="."]
transform = Transform3D(0.707107, -0.5, 0.5, 0, 0.707107, 0.707107, -0.707107, -0.5, 0.5, 0, 10, 0)
'''
            else:
                content = f'''[gd_scene format=3]

[node name="{scene_name}" type="Control"]
anchor_right = 1.0
anchor_bottom = 1.0
'''
            
            scene_path.write_text(content)
            
            # 프로젝트 정보 업데이트
            for project in self.active_projects.values():
                if project["path"] == str(project_dir):
                    project["scenes"].append(f"{scene_name}.tscn")
                    break
            
            self.logger.info(f"씬 생성 완료: {scene_name} ({scene_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"씬 생성 실패: {e}")
            return False
    
    async def add_node_to_scene(self, project_path: str, scene_name: str,
                              node_name: str, node_type: str,
                              parent_path: str = ".") -> bool:
        """씬에 노드 추가"""
        try:
            project_dir = Path(project_path)
            scene_path = project_dir / "scenes" / f"{scene_name}.tscn"
            
            if not scene_path.exists():
                self.logger.error(f"씬 파일 없음: {scene_path}")
                return False
            
            # 씬 파일 파싱 (간단한 구현)
            content = scene_path.read_text()
            lines = content.split('\n')
            
            # 노드 추가
            node_entry = f'\n[node name="{node_name}" type="{node_type}" parent="{parent_path}"]'
            
            # 적절한 위치 찾기 (파일 끝 직전)
            insert_index = len(lines) - 1
            lines.insert(insert_index, node_entry)
            
            # 파일 저장
            scene_path.write_text('\n'.join(lines))
            
            self.logger.info(f"노드 추가 완료: {node_name} ({node_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"노드 추가 실패: {e}")
            return False
    
    async def create_script(self, project_path: str, script_name: str,
                          base_class: str = "Node", content: str = None) -> bool:
        """스크립트 생성"""
        try:
            project_dir = Path(project_path)
            script_path = project_dir / "scripts" / f"{script_name}.gd"
            
            if content is None:
                content = f'''extends {base_class}

# Called when the node enters the scene tree for the first time.
func _ready():
    pass

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
    pass
'''
            
            script_path.write_text(content)
            
            # 프로젝트 정보 업데이트
            for project in self.active_projects.values():
                if project["path"] == str(project_dir):
                    project["scripts"].append(f"{script_name}.gd")
                    break
            
            self.logger.info(f"스크립트 생성 완료: {script_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"스크립트 생성 실패: {e}")
            return False
    
    async def export_project(self, project_path: str, export_path: str,
                           platform: str = "linux") -> bool:
        """프로젝트 내보내기"""
        try:
            # 내보내기 프리셋 생성
            preset_path = Path(project_path) / "export_presets.cfg"
            
            preset_content = f'''[preset.0]
name="{platform}"
platform="{platform}"
runnable=true
custom_features=""
export_filter="all_resources"
include_filter=""
exclude_filter=""
patch_list=PoolStringArray()
'''
            
            preset_path.write_text(preset_content)
            
            # 내보내기 실행
            cmd = [
                self.godot_path,
                "--headless",
                "--path", project_path,
                "--export", platform,
                export_path
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                self.logger.info(f"프로젝트 내보내기 완료: {export_path}")
                return True
            else:
                self.logger.error(f"내보내기 실패: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"프로젝트 내보내기 실패: {e}")
            return False
    
    async def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """프로젝트 분석"""
        try:
            project_dir = Path(project_path)
            
            analysis = {
                "project_name": project_dir.name,
                "scenes": [],
                "scripts": [],
                "assets": {
                    "sprites": 0,
                    "sounds": 0,
                    "fonts": 0
                },
                "total_size": 0,
                "file_count": 0
            }
            
            # 씬 파일 분석
            for scene_file in (project_dir / "scenes").glob("*.tscn"):
                analysis["scenes"].append({
                    "name": scene_file.name,
                    "size": scene_file.stat().st_size,
                    "modified": datetime.fromtimestamp(scene_file.stat().st_mtime).isoformat()
                })
            
            # 스크립트 분석
            for script_file in (project_dir / "scripts").glob("*.gd"):
                content = script_file.read_text()
                lines = content.split('\n')
                
                analysis["scripts"].append({
                    "name": script_file.name,
                    "lines": len(lines),
                    "size": script_file.stat().st_size,
                    "base_class": self._extract_base_class(content)
                })
            
            # 에셋 카운트
            if (project_dir / "assets").exists():
                analysis["assets"]["sprites"] = len(list((project_dir / "assets" / "sprites").glob("*")))
                analysis["assets"]["sounds"] = len(list((project_dir / "assets" / "sounds").glob("*")))
                analysis["assets"]["fonts"] = len(list((project_dir / "assets" / "fonts").glob("*")))
            
            # 전체 크기 계산
            for file_path in project_dir.rglob("*"):
                if file_path.is_file():
                    analysis["total_size"] += file_path.stat().st_size
                    analysis["file_count"] += 1
            
            # 크기를 MB로 변환
            analysis["total_size_mb"] = analysis["total_size"] / (1024 * 1024)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"프로젝트 분석 실패: {e}")
            return {}
    
    def _extract_base_class(self, script_content: str) -> str:
        """스크립트에서 베이스 클래스 추출"""
        match = re.search(r'extends\s+(\w+)', script_content)
        return match.group(1) if match else "Unknown"
    
    async def optimize_project(self, project_path: str) -> Dict[str, Any]:
        """프로젝트 최적화"""
        try:
            optimizations = {
                "textures_compressed": 0,
                "unused_resources_removed": 0,
                "scripts_optimized": 0,
                "total_size_saved": 0
            }
            
            project_dir = Path(project_path)
            
            # 사용하지 않는 리소스 찾기
            used_resources = set()
            
            # 씬 파일에서 사용된 리소스 수집
            for scene_file in (project_dir / "scenes").glob("*.tscn"):
                content = scene_file.read_text()
                # 리소스 경로 추출 (간단한 구현)
                resources = re.findall(r'path="res://([^"]+)"', content)
                used_resources.update(resources)
            
            # 스크립트에서 사용된 리소스 수집
            for script_file in (project_dir / "scripts").glob("*.gd"):
                content = script_file.read_text()
                resources = re.findall(r'load\("res://([^"]+)"\)', content)
                used_resources.update(resources)
            
            # 사용하지 않는 파일 제거 (실제로는 백업 후 제거해야 함)
            for asset_file in project_dir.rglob("*"):
                if asset_file.is_file():
                    rel_path = asset_file.relative_to(project_dir)
                    if str(rel_path) not in used_resources and asset_file.suffix in ['.png', '.jpg', '.ogg', '.mp3']:
                        # 실제 환경에서는 백업 후 제거
                        self.logger.info(f"미사용 리소스 발견: {rel_path}")
                        optimizations["unused_resources_removed"] += 1
            
            return optimizations
            
        except Exception as e:
            self.logger.error(f"프로젝트 최적화 실패: {e}")
            return {}
    
    def get_project_info(self, project_name: str) -> Optional[Dict[str, Any]]:
        """프로젝트 정보 조회"""
        return self.active_projects.get(project_name)
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """프로젝트 목록"""
        return [
            {"name": name, **info}
            for name, info in self.active_projects.items()
        ]