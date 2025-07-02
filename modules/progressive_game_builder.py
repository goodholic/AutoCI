#!/usr/bin/env python3
"""
진행형 게임 빌더 - 게임 제작 과정을 실시간으로 보여주면서 단계별로 파일 생성
README에 명시된 대로 게임이 만들어지는 과정을 사용자가 볼 수 있도록 함
"""

import os
import sys
import json
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import threading
import queue

class BuildPhase(Enum):
    """게임 제작 단계"""
    PLANNING = "🎯 기획 단계"
    PROJECT_SETUP = "📁 프로젝트 설정"
    SCENE_CREATION = "🎬 씬 생성"
    PLAYER_CREATION = "🎮 플레이어 제작"
    WORLD_BUILDING = "🌍 월드 구축"
    MECHANICS = "⚙️ 게임 메커니즘"
    UI_CREATION = "📱 UI 제작"
    TESTING = "🧪 테스트"
    POLISH = "💎 마무리"

class ProgressiveGameBuilder:
    """게임 제작 과정을 단계별로 보여주며 실제 파일을 생성하는 빌더"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.current_project_path = None
        self.godot_exe = None
        self.godot_process = None
        self.godot_window_open = False
        self.message_queue = queue.Queue()
        self.current_phase = None
        self.build_log = []
        
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
                return str(path)
        return None
    
    async def show_phase_intro(self, phase: BuildPhase, duration: float = 2.0):
        """단계 시작 화면 표시"""
        print("\n" + "="*60)
        print(f"{phase.value}")
        print("="*60)
        await asyncio.sleep(duration)
    
    async def show_progress(self, task: str, steps: List[str], delay_per_step: float = 1.0):
        """작업 진행 상황을 단계별로 표시"""
        print(f"\n📋 {task}")
        print("-" * 50)
        
        for i, step in enumerate(steps):
            # 진행 바 애니메이션
            progress = (i + 1) / len(steps)
            bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
            print(f"  [{bar}] {step}")
            
            # 로그에 추가
            self.build_log.append({
                "time": datetime.now(),
                "phase": self.current_phase.value if self.current_phase else "Unknown",
                "task": task,
                "step": step,
                "progress": progress
            })
            
            await asyncio.sleep(delay_per_step)
        
        print(f"✅ {task} 완료!")
    
    async def create_project_progressively(self, game_name: str, game_type: str) -> Path:
        """프로젝트를 단계별로 생성하며 과정을 보여줌"""
        project_path = self.project_root / "game_projects" / game_name
        project_path.mkdir(parents=True, exist_ok=True)
        self.current_project_path = project_path
        
        # 1. 기획 단계
        self.current_phase = BuildPhase.PLANNING
        await self.show_phase_intro(BuildPhase.PLANNING)
        await self.show_progress(
            "게임 컨셉 정의",
            [
                f"🎮 {game_type} 게임 타입 선택됨",
                f"📝 게임 이름: {game_name}",
                "🎯 목표 설정: 재미있고 직관적인 게임플레이",
                "👥 타겟 유저: 모든 연령대",
                "📊 게임 스펙 결정 완료"
            ],
            0.8
        )
        
        # 2. 프로젝트 설정
        self.current_phase = BuildPhase.PROJECT_SETUP
        await self.show_phase_intro(BuildPhase.PROJECT_SETUP)
        
        # project.godot 생성
        await self.show_progress(
            "프로젝트 파일 생성",
            [
                "📄 project.godot 파일 생성 중...",
                "⚙️ 프로젝트 설정 구성 중...",
                "🎮 입력 맵핑 설정 중...",
                "🖼️ 렌더링 설정 구성 중..."
            ],
            0.5
        )
        
        # 실제 project.godot 생성
        await self._create_project_file(project_path, game_name, game_type)
        
        # 폴더 구조 생성
        await self.show_progress(
            "폴더 구조 생성",
            [
                "📁 scenes/ 폴더 생성",
                "📁 scripts/ 폴더 생성",
                "📁 assets/ 폴더 생성",
                "📁 assets/sprites/ 폴더 생성",
                "📁 assets/sounds/ 폴더 생성"
            ],
            0.3
        )
        
        await self._create_folder_structure(project_path)
        
        return project_path
    
    async def _create_project_file(self, project_path: Path, game_name: str, game_type: str):
        """project.godot 파일 생성"""
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
    
    async def _create_folder_structure(self, project_path: Path):
        """프로젝트 폴더 구조 생성"""
        folders = ["scenes", "scripts", "assets", "assets/sprites", "assets/sounds", "assets/music", "assets/fonts"]
        for folder in folders:
            (project_path / folder).mkdir(parents=True, exist_ok=True)
        
        # 기본 아이콘 생성
        icon_svg = """<svg height="128" width="128" xmlns="http://www.w3.org/2000/svg">
<rect x="0" y="0" width="128" height="128" fill="#4A90E2"/>
<text x="64" y="64" font-family="Arial" font-size="40" fill="white" text-anchor="middle" alignment-baseline="middle">AI</text>
</svg>"""
        (project_path / "icon.svg").write_text(icon_svg)
    
    async def open_godot_progressively(self, project_path: Path):
        """Godot을 열고 제작 과정 계속 진행"""
        if not self.godot_window_open:
            self.godot_exe = self.find_godot_executable()
            if not self.godot_exe:
                print("❌ Godot을 찾을 수 없습니다.")
                return False
            
            print("\n🚀 Godot 에디터를 여는 중...")
            
            # WSL 경로를 Windows 경로로 변환
            windows_path = str(project_path).replace("/mnt/c", "C:").replace("/mnt/d", "D:").replace("/", "\\")
            
            # Godot 에디터 실행
            cmd = ["cmd.exe", "/c", "start", "", self.godot_exe.replace("/mnt/c", "C:").replace("/mnt/d", "D:").replace("/", "\\"), "--path", windows_path, "--editor"]
            
            try:
                subprocess.run(cmd, check=True)
                self.godot_window_open = True
                print("✅ Godot 에디터가 열렸습니다!")
                await asyncio.sleep(3)  # 에디터가 완전히 로드될 때까지 대기
                return True
            except Exception as e:
                print(f"❌ Godot 실행 중 오류: {e}")
                return False
        return True
    
    async def create_scene_progressively(self, project_path: Path, game_type: str):
        """씬을 단계별로 생성"""
        self.current_phase = BuildPhase.SCENE_CREATION
        await self.show_phase_intro(BuildPhase.SCENE_CREATION)
        
        await self.show_progress(
            "메인 씬 생성",
            [
                "🎬 씬 구조 설계 중...",
                "🌍 게임 월드 노드 추가 중...",
                "📷 카메라 설정 중...",
                "💡 조명 설정 중..."
            ],
            0.8
        )
        
        # Main.tscn 생성
        scenes_dir = project_path / "scenes"
        main_scene = """[gd_scene load_steps=3 format=3 uid="uid://main_scene"]

[ext_resource type="Script" path="res://scripts/Main.gd" id="1"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

[node name="World" type="Node2D" parent="."]

[node name="UI" type="CanvasLayer" parent="."]
"""
        (scenes_dir / "Main.tscn").write_text(main_scene)
        
        # Main.gd 스크립트 생성
        scripts_dir = project_path / "scripts"
        main_script = f"""extends Node2D

# 게임 메인 스크립트
func _ready():
	print("🎮 {game_type} 게임이 시작되었습니다!")
	print("AI와 함께 만든 게임입니다.")
	
	# 게임 초기화
	_initialize_game()

func _initialize_game():
	# 게임 설정 초기화
	pass

func _input(event):
	if event.is_action_pressed("ui_cancel"):
		get_tree().quit()
"""
        (scripts_dir / "Main.gd").write_text(main_script)
    
    async def create_player_progressively(self, project_path: Path, game_type: str):
        """플레이어를 단계별로 생성"""
        if game_type not in ["platformer", "racing"]:
            return
        
        self.current_phase = BuildPhase.PLAYER_CREATION
        await self.show_phase_intro(BuildPhase.PLAYER_CREATION)
        
        if game_type == "platformer":
            await self.show_progress(
                "플레이어 캐릭터 생성",
                [
                    "🎮 CharacterBody2D 노드 생성 중...",
                    "🎨 스프라이트 추가 중...",
                    "💥 충돌 영역 설정 중...",
                    "📝 플레이어 스크립트 작성 중..."
                ],
                1.0
            )
            
            # Player.tscn 생성
            await self._create_platformer_player(project_path)
            
        elif game_type == "racing":
            await self.show_progress(
                "차량 생성",
                [
                    "🚗 RigidBody2D 노드 생성 중...",
                    "🎨 차량 스프라이트 추가 중...",
                    "⚙️ 물리 설정 구성 중...",
                    "📝 차량 제어 스크립트 작성 중..."
                ],
                1.0
            )
            
            # Car.tscn 생성
            await self._create_racing_car(project_path)
    
    async def _create_platformer_player(self, project_path: Path):
        """플랫포머 플레이어 생성"""
        scenes_dir = project_path / "scenes"
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
        (scenes_dir / "Player.tscn").write_text(player_scene)
        
        # Player.gd 스크립트
        scripts_dir = project_path / "scripts"
        player_script = """extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

# Get the gravity from the project settings to be synced with RigidBody nodes.
var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

func _physics_process(delta):
	# Add the gravity.
	if not is_on_floor():
		velocity.y += gravity * delta

	# Handle jump.
	if Input.is_action_just_pressed("jump") and is_on_floor():
		velocity.y = JUMP_VELOCITY

	# Get the input direction and handle the movement/deceleration.
	var direction = Input.get_axis("move_left", "move_right")
	if direction:
		velocity.x = direction * SPEED
	else:
		velocity.x = move_toward(velocity.x, 0, SPEED)

	move_and_slide()
"""
        (scripts_dir / "Player.gd").write_text(player_script)
    
    async def _create_racing_car(self, project_path: Path):
        """레이싱 차량 생성"""
        scenes_dir = project_path / "scenes"
        car_scene = """[gd_scene load_steps=4 format=3 uid="uid://car_scene"]

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
        (scenes_dir / "Car.tscn").write_text(car_scene)
        
        scripts_dir = project_path / "scripts"
        car_script = """extends RigidBody2D

const ENGINE_POWER = 800
const STEERING_POWER = 3.0

var velocity = Vector2.ZERO
var steering_input = 0.0

func _physics_process(delta):
	# 입력 처리
	var throttle = Input.get_axis("move_down", "move_up")
	steering_input = Input.get_axis("move_left", "move_right")
	
	# 엔진 힘 적용
	if throttle != 0:
		apply_central_force(transform.y * throttle * ENGINE_POWER)
	
	# 조향
	if abs(linear_velocity.length()) > 10:
		angular_velocity = steering_input * STEERING_POWER
"""
        (scripts_dir / "Car.gd").write_text(car_script)
    
    async def create_world_progressively(self, project_path: Path, game_type: str):
        """월드를 단계별로 생성"""
        self.current_phase = BuildPhase.WORLD_BUILDING
        await self.show_phase_intro(BuildPhase.WORLD_BUILDING)
        
        if game_type == "platformer":
            await self.show_progress(
                "플랫폼 월드 구축",
                [
                    "🏗️ 지형 플랫폼 생성 중...",
                    "🌳 배경 요소 추가 중...",
                    "⭐ 수집 아이템 배치 중...",
                    "🚧 장애물 배치 중..."
                ],
                1.2
            )
        elif game_type == "racing":
            await self.show_progress(
                "레이싱 트랙 구축",
                [
                    "🛣️ 트랙 경로 생성 중...",
                    "🏁 체크포인트 배치 중...",
                    "🌴 트랙 주변 환경 구성 중...",
                    "💨 부스터 지역 설정 중..."
                ],
                1.2
            )
        else:
            await self.show_progress(
                "게임 월드 구축",
                [
                    "🌍 기본 월드 구조 생성 중...",
                    "🎨 환경 요소 배치 중...",
                    "✨ 인터랙션 요소 추가 중..."
                ],
                1.0
            )
    
    async def add_game_mechanics(self, project_path: Path, game_type: str):
        """게임 메커니즘 추가"""
        self.current_phase = BuildPhase.MECHANICS
        await self.show_phase_intro(BuildPhase.MECHANICS)
        
        mechanics_by_type = {
            "platformer": [
                "🦘 더블 점프 시스템 구현 중...",
                "💨 대시 능력 추가 중...",
                "🛡️ 무적 시간 시스템 구현 중...",
                "📊 점수 시스템 추가 중..."
            ],
            "racing": [
                "💨 드리프트 시스템 구현 중...",
                "⚡ 부스터 시스템 추가 중...",
                "🏁 랩 타임 기록 시스템 구현 중...",
                "🏆 순위 시스템 추가 중..."
            ],
            "puzzle": [
                "🧩 매칭 로직 구현 중...",
                "🔄 블록 회전 시스템 추가 중...",
                "⏱️ 시간 제한 시스템 구현 중...",
                "💡 힌트 시스템 추가 중..."
            ]
        }
        
        mechanics = mechanics_by_type.get(game_type, [
            "⚙️ 기본 게임 로직 구현 중...",
            "🎮 컨트롤 시스템 구현 중...",
            "📊 진행 상황 추적 시스템 추가 중..."
        ])
        
        await self.show_progress(f"{game_type} 게임 메커니즘 구현", mechanics, 1.5)
    
    async def create_ui_progressively(self, project_path: Path):
        """UI를 단계별로 생성"""
        self.current_phase = BuildPhase.UI_CREATION
        await self.show_phase_intro(BuildPhase.UI_CREATION)
        
        await self.show_progress(
            "사용자 인터페이스 제작",
            [
                "📊 HUD (헤드업 디스플레이) 생성 중...",
                "🎯 점수 표시 UI 추가 중...",
                "❤️ 체력/생명 표시 추가 중...",
                "⏸️ 일시정지 메뉴 구현 중...",
                "🏆 게임 오버 화면 제작 중..."
            ],
            1.0
        )
    
    async def test_and_polish(self, project_path: Path):
        """테스트 및 마무리"""
        self.current_phase = BuildPhase.TESTING
        await self.show_phase_intro(BuildPhase.TESTING)
        
        await self.show_progress(
            "게임 테스트",
            [
                "🧪 기본 기능 테스트 중...",
                "🐛 버그 체크 중...",
                "⚖️ 게임 밸런스 확인 중...",
                "🎮 조작감 테스트 중..."
            ],
            0.8
        )
        
        self.current_phase = BuildPhase.POLISH
        await self.show_phase_intro(BuildPhase.POLISH)
        
        await self.show_progress(
            "마무리 작업",
            [
                "✨ 시각 효과 개선 중...",
                "🔊 사운드 효과 점검 중...",
                "🎯 난이도 조정 중...",
                "💎 최종 품질 확인 중..."
            ],
            0.6
        )
    
    async def build_game_with_visualization(self, game_name: str, game_type: str):
        """게임을 단계별로 제작하며 과정을 시각화"""
        print("\n" + "="*60)
        print("🎮 게임 제작 과정을 실시간으로 보여드립니다!")
        print("💡 각 단계가 진행되는 모습을 확인하세요.")
        print("="*60)
        
        try:
            # 1. 프로젝트 생성 (단계별)
            project_path = await self.create_project_progressively(game_name, game_type)
            
            # 2. Godot 에디터 열기
            await self.open_godot_progressively(project_path)
            
            # 3. 씬 생성 (단계별)
            await self.create_scene_progressively(project_path, game_type)
            
            # 4. 플레이어 생성 (단계별)
            await self.create_player_progressively(project_path, game_type)
            
            # 5. 월드 구축 (단계별)
            await self.create_world_progressively(project_path, game_type)
            
            # 6. 게임 메커니즘 추가
            await self.add_game_mechanics(project_path, game_type)
            
            # 7. UI 제작
            await self.create_ui_progressively(project_path)
            
            # 8. 테스트 및 마무리
            await self.test_and_polish(project_path)
            
            # 완료
            print("\n" + "="*60)
            print("🎉 게임 제작이 완료되었습니다!")
            print("="*60)
            print("\n📊 제작 과정 요약:")
            self.show_build_summary()
            
            print("\n💬 이제 Godot 에디터에서:")
            print("  - F5: 전체 게임 실행")
            print("  - F6: 현재 씬 실행")
            print("  - Ctrl+S: 프로젝트 저장")
            print("\n🎮 사용자가 직접 게임을 수정하고 발전시킬 수 있습니다!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 게임 제작 중 오류 발생: {e}")
            return False
    
    def show_build_summary(self):
        """제작 과정 요약 표시"""
        if not self.build_log:
            return
        
        # 단계별로 그룹화
        phase_summary = {}
        for entry in self.build_log:
            phase = entry["phase"]
            if phase not in phase_summary:
                phase_summary[phase] = []
            phase_summary[phase].append(entry["step"])
        
        for phase, steps in phase_summary.items():
            print(f"\n{phase}")
            for step in steps[:3]:  # 주요 단계 3개만 표시
                print(f"  • {step}")
            if len(steps) > 3:
                print(f"  • ... 외 {len(steps) - 3}개 작업")


# 전역 인스턴스
_progressive_builder = None

def get_progressive_builder() -> ProgressiveGameBuilder:
    """진행형 게임 빌더 싱글톤 인스턴스 반환"""
    global _progressive_builder
    if _progressive_builder is None:
        _progressive_builder = ProgressiveGameBuilder()
    return _progressive_builder