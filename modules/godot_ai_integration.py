#!/usr/bin/env python3
"""
Godot AI 통합 시스템
Godot을 AI 작업에 최적화된 형태로 변형하여 완전한 자동화 환경 구축
"""

import os
import sys
import json
import asyncio
import logging
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import xml.etree.ElementTree as ET

class GodotAIIntegration:
    """Godot AI 통합 및 변형 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger("GodotAIIntegration")
        self.platform = platform.system().lower()
        self.architecture = platform.machine().lower()
        
        # 경로 설정
        self.autoci_root = Path(__file__).parent.parent
        self.godot_dir = self.autoci_root / "godot_ai"
        self.plugins_dir = self.godot_dir / "plugins"
        self.templates_dir = self.godot_dir / "templates"
        self.tools_dir = self.godot_dir / "tools"
        
        # Godot 버전 정보
        self.godot_version = "4.3-stable"
        self.godot_urls = self._get_godot_download_urls()
        
        # AI 플러그인 정보
        self.ai_plugins = self._get_ai_plugin_specs()
        
        # 초기화
        self._ensure_directories()
        
    def _get_godot_download_urls(self) -> Dict[str, str]:
        """플랫폼별 Godot 다운로드 URL"""
        base_url = f"https://downloads.tuxfamily.org/godotengine/4.3"
        
        urls = {
            "linux_x64": f"{base_url}/Godot_v{self.godot_version}_linux.x86_64.zip",
            "linux_arm64": f"{base_url}/Godot_v{self.godot_version}_linux.arm64.zip",
            "windows_x64": f"{base_url}/Godot_v{self.godot_version}_win64.exe.zip",
            "windows_x32": f"{base_url}/Godot_v{self.godot_version}_win32.exe.zip",
            "macos": f"{base_url}/Godot_v{self.godot_version}_macos.universal.zip",
            "headless": f"{base_url}/Godot_v{self.godot_version}_linux_headless.x86_64.zip"
        }
        
        return urls
        
    def _get_ai_plugin_specs(self) -> List[Dict[str, Any]]:
        """AI 플러그인 사양"""
        return [
            {
                "name": "AutoCI_AI_Controller",
                "description": "AI 자동 제어 핵심 플러그인",
                "version": "1.0.0",
                "script_paths": [
                    "addons/autoci_ai/plugin.cfg",
                    "addons/autoci_ai/plugin.gd",
                    "addons/autoci_ai/ai_controller.gd",
                    "addons/autoci_ai/scene_generator.gd",
                    "addons/autoci_ai/resource_manager.gd",
                    "addons/autoci_ai/automation_api.gd"
                ]
            },
            {
                "name": "AutoCI_Scene_Automation",
                "description": "씬 자동화 전용 플러그인",
                "version": "1.0.0",
                "script_paths": [
                    "addons/scene_automation/plugin.cfg",
                    "addons/scene_automation/plugin.gd",
                    "addons/scene_automation/auto_composer.gd",
                    "addons/scene_automation/layout_optimizer.gd"
                ]
            },
            {
                "name": "AutoCI_Resource_Generator",
                "description": "리소스 자동 생성 플러그인",
                "version": "1.0.0",
                "script_paths": [
                    "addons/resource_gen/plugin.cfg",
                    "addons/resource_gen/plugin.gd",
                    "addons/resource_gen/texture_generator.gd",
                    "addons/resource_gen/audio_generator.gd",
                    "addons/resource_gen/material_generator.gd"
                ]
            }
        ]
        
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.godot_dir,
            self.plugins_dir,
            self.templates_dir,
            self.tools_dir,
            self.godot_dir / "bin",
            self.godot_dir / "export_templates",
            self.godot_dir / "projects"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    async def setup_ai_optimized_godot(self) -> bool:
        """AI 최적화된 Godot 환경 구축"""
        self.logger.info("AI 최적화 Godot 환경 구축 시작")
        
        try:
            # 1. Godot 엔진 다운로드 및 설치
            await self._download_and_install_godot()
            
            # 2. AI 플러그인 생성 및 설치
            await self._create_ai_plugins()
            
            # 3. 프로젝트 템플릿 생성
            await self._create_project_templates()
            
            # 4. AI 자동화 도구 설치
            await self._install_automation_tools()
            
            # 5. Godot 설정 최적화
            await self._optimize_godot_settings()
            
            # 6. 통합 테스트
            await self._test_ai_integration()
            
            self.logger.info("AI 최적화 Godot 환경 구축 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"Godot AI 통합 실패: {e}")
            return False
            
    async def _download_and_install_godot(self):
        """Godot 엔진 다운로드 및 설치"""
        self.logger.info("Godot 엔진 다운로드 시작")
        
        # 플랫폼 및 아키텍처에 따른 URL 선택
        platform_key = self._get_platform_key()
        download_url = self.godot_urls.get(platform_key)
        
        if not download_url:
            raise ValueError(f"지원하지 않는 플랫폼: {self.platform}_{self.architecture}")
            
        # 다운로드
        godot_zip_path = self.godot_dir / "godot.zip"
        await self._download_file(download_url, godot_zip_path)
        
        # 압축 해제
        with zipfile.ZipFile(godot_zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.godot_dir / "bin")
            
        # 실행 파일 권한 설정 및 심볼릭 링크 생성
        await self._setup_godot_executable()
        
        # 헤드리스 버전도 다운로드 (서버 환경용)
        if platform_key != "headless":
            headless_url = self.godot_urls.get("headless")
            if headless_url:
                headless_zip_path = self.godot_dir / "godot_headless.zip"
                await self._download_file(headless_url, headless_zip_path)
                
                with zipfile.ZipFile(headless_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.godot_dir / "bin")
                    
        # 정리
        godot_zip_path.unlink(missing_ok=True)
        if 'headless_zip_path' in locals():
            headless_zip_path.unlink(missing_ok=True)
            
        self.logger.info("Godot 엔진 설치 완료")
        
    def _get_platform_key(self) -> str:
        """플랫폼 키 반환"""
        if self.platform == "linux":
            if "arm" in self.architecture or "aarch64" in self.architecture:
                return "linux_arm64"
            else:
                return "linux_x64"
        elif self.platform == "windows":
            if "64" in self.architecture:
                return "windows_x64"
            else:
                return "windows_x32"
        elif self.platform == "darwin":
            return "macos"
        else:
            return "linux_x64"  # 기본값
            
    async def _download_file(self, url: str, output_path: Path):
        """파일 다운로드"""
        self.logger.info(f"다운로드 중: {url}")
        
        def download():
            urllib.request.urlretrieve(url, output_path)
            
        # 비동기 실행
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, download)
        
    async def _setup_godot_executable(self):
        """Godot 실행 파일 설정"""
        bin_dir = self.godot_dir / "bin"
        
        # 실행 파일 찾기
        godot_executable = None
        for file_path in bin_dir.rglob("*"):
            if file_path.is_file() and ("godot" in file_path.name.lower() or "Godot" in file_path.name):
                if not file_path.name.endswith(('.zip', '.txt', '.md')):
                    godot_executable = file_path
                    break
                    
        if godot_executable:
            # 실행 권한 부여 (Unix 계열)
            if self.platform != "windows":
                os.chmod(godot_executable, 0o755)
                
            # 심볼릭 링크 생성
            symlink_path = self.godot_dir / "godot"
            if symlink_path.exists():
                symlink_path.unlink()
                
            if self.platform != "windows":
                symlink_path.symlink_to(godot_executable)
            else:
                # Windows에서는 복사
                shutil.copy2(godot_executable, symlink_path.with_suffix('.exe'))
                
            self.logger.info(f"Godot 실행 파일 설정 완료: {godot_executable}")
        else:
            raise FileNotFoundError("Godot 실행 파일을 찾을 수 없습니다")
            
    async def _create_ai_plugins(self):
        """AI 플러그인 생성"""
        self.logger.info("AI 플러그인 생성 시작")
        
        for plugin_spec in self.ai_plugins:
            await self._create_single_plugin(plugin_spec)
            
        self.logger.info("모든 AI 플러그인 생성 완료")
        
    async def _create_single_plugin(self, plugin_spec: Dict[str, Any]):
        """개별 플러그인 생성"""
        plugin_name = plugin_spec["name"]
        self.logger.info(f"플러그인 생성: {plugin_name}")
        
        # 플러그인 디렉토리 생성
        plugin_dir = self.plugins_dir / plugin_name.lower()
        plugin_dir.mkdir(parents=True, exist_ok=True)
        
        # plugin.cfg 생성
        await self._create_plugin_config(plugin_dir, plugin_spec)
        
        # plugin.gd 생성
        await self._create_plugin_script(plugin_dir, plugin_spec)
        
        # 기타 스크립트 생성
        await self._create_plugin_components(plugin_dir, plugin_spec)
        
    async def _create_plugin_config(self, plugin_dir: Path, plugin_spec: Dict[str, Any]):
        """plugin.cfg 파일 생성"""
        config_content = f'''[plugin]

name="{plugin_spec["name"]}"
description="{plugin_spec["description"]}"
author="AutoCI AI System"
version="{plugin_spec["version"]}"
script="plugin.gd"
'''
        
        config_path = plugin_dir / "plugin.cfg"
        with open(config_path, 'w') as f:
            f.write(config_content)
            
    async def _create_plugin_script(self, plugin_dir: Path, plugin_spec: Dict[str, Any]):
        """plugin.gd 스크립트 생성"""
        plugin_name = plugin_spec["name"]
        
        if plugin_name == "AutoCI_AI_Controller":
            script_content = self._get_ai_controller_plugin_script()
        elif plugin_name == "AutoCI_Scene_Automation":
            script_content = self._get_scene_automation_plugin_script()
        elif plugin_name == "AutoCI_Resource_Generator":
            script_content = self._get_resource_generator_plugin_script()
        else:
            script_content = self._get_default_plugin_script(plugin_name)
            
        script_path = plugin_dir / "plugin.gd"
        with open(script_path, 'w') as f:
            f.write(script_content)
            
    def _get_ai_controller_plugin_script(self) -> str:
        """AI 컨트롤러 플러그인 스크립트"""
        return '''@tool
extends EditorPlugin

# AutoCI AI Controller Plugin
# AI 자동 제어 핵심 플러그인

var ai_controller
var automation_dock

func _enter_tree():
    print("AutoCI AI Controller 플러그인 활성화")
    
    # AI 컨트롤러 초기화
    ai_controller = preload("res://addons/autoci_ai/ai_controller.gd").new()
    add_child(ai_controller)
    
    # 자동화 독 추가
    automation_dock = preload("res://addons/autoci_ai/automation_dock.tscn").instantiate()
    add_control_to_dock(DOCK_SLOT_LEFT_UR, automation_dock)
    
    # 메뉴 항목 추가
    add_tool_menu_item("AI 프로젝트 생성", _on_ai_create_project)
    add_tool_menu_item("AI 씬 최적화", _on_ai_optimize_scene)
    add_tool_menu_item("AI 리소스 생성", _on_ai_generate_resources)
    
    # 자동화 시작
    ai_controller.start_automation()

func _exit_tree():
    print("AutoCI AI Controller 플러그인 비활성화")
    
    # 정리
    if automation_dock:
        remove_control_from_docks(automation_dock)
        automation_dock.queue_free()
        
    remove_tool_menu_item("AI 프로젝트 생성")
    remove_tool_menu_item("AI 씬 최적화")
    remove_tool_menu_item("AI 리소스 생성")
    
    if ai_controller:
        ai_controller.stop_automation()
        ai_controller.queue_free()

func _on_ai_create_project():
    """AI 프로젝트 생성"""
    if ai_controller:
        ai_controller.create_ai_project()

func _on_ai_optimize_scene():
    """AI 씬 최적화"""
    if ai_controller:
        ai_controller.optimize_current_scene()

func _on_ai_generate_resources():
    """AI 리소스 생성"""
    if ai_controller:
        ai_controller.generate_game_resources()

func has_main_screen():
    return false

func get_plugin_name():
    return "AutoCI AI Controller"
'''

    def _get_scene_automation_plugin_script(self) -> str:
        """씬 자동화 플러그인 스크립트"""
        return '''@tool
extends EditorPlugin

# AutoCI Scene Automation Plugin
# 씬 자동화 전용 플러그인

var scene_composer
var layout_optimizer

func _enter_tree():
    print("AutoCI Scene Automation 플러그인 활성화")
    
    # 씬 컴포저 초기화
    scene_composer = preload("res://addons/scene_automation/auto_composer.gd").new()
    layout_optimizer = preload("res://addons/scene_automation/layout_optimizer.gd").new()
    
    add_child(scene_composer)
    add_child(layout_optimizer)
    
    # 컨텍스트 메뉴 추가
    add_tool_submenu_item("AI Scene Tools", _create_scene_menu())

func _exit_tree():
    print("AutoCI Scene Automation 플러그인 비활성화")
    
    if scene_composer:
        scene_composer.queue_free()
    if layout_optimizer:
        layout_optimizer.queue_free()

func _create_scene_menu():
    var menu = PopupMenu.new()
    menu.add_item("AI 씬 구성", 0)
    menu.add_item("레이아웃 최적화", 1)
    menu.add_item("노드 자동 배치", 2)
    
    menu.id_pressed.connect(_on_scene_menu_pressed)
    return menu

func _on_scene_menu_pressed(id: int):
    match id:
        0:
            scene_composer.compose_intelligent_scene()
        1:
            layout_optimizer.optimize_current_layout()
        2:
            scene_composer.auto_place_nodes()

func get_plugin_name():
    return "AutoCI Scene Automation"
'''

    def _get_resource_generator_plugin_script(self) -> str:
        """리소스 생성기 플러그인 스크립트"""
        return '''@tool
extends EditorPlugin

# AutoCI Resource Generator Plugin
# 리소스 자동 생성 플러그인

var texture_gen
var audio_gen
var material_gen

func _enter_tree():
    print("AutoCI Resource Generator 플러그인 활성화")
    
    # 생성기 초기화
    texture_gen = preload("res://addons/resource_gen/texture_generator.gd").new()
    audio_gen = preload("res://addons/resource_gen/audio_generator.gd").new()
    material_gen = preload("res://addons/resource_gen/material_generator.gd").new()
    
    add_child(texture_gen)
    add_child(audio_gen)
    add_child(material_gen)
    
    # 리소스 생성 메뉴 추가
    add_tool_submenu_item("AI Resource Generator", _create_resource_menu())

func _exit_tree():
    print("AutoCI Resource Generator 플러그인 비활성화")
    
    if texture_gen:
        texture_gen.queue_free()
    if audio_gen:
        audio_gen.queue_free()
    if material_gen:
        material_gen.queue_free()

func _create_resource_menu():
    var menu = PopupMenu.new()
    menu.add_item("AI 텍스처 생성", 0)
    menu.add_item("AI 오디오 생성", 1)
    menu.add_item("AI 머티리얼 생성", 2)
    menu.add_item("배치 리소스 생성", 3)
    
    menu.id_pressed.connect(_on_resource_menu_pressed)
    return menu

func _on_resource_menu_pressed(id: int):
    match id:
        0:
            texture_gen.generate_procedural_textures()
        1:
            audio_gen.generate_game_audio()
        2:
            material_gen.generate_materials()
        3:
            _batch_generate_resources()

func _batch_generate_resources():
    """배치 리소스 생성"""
    texture_gen.generate_procedural_textures()
    audio_gen.generate_game_audio()
    material_gen.generate_materials()

func get_plugin_name():
    return "AutoCI Resource Generator"
'''

    def _get_default_plugin_script(self, plugin_name: str) -> str:
        """기본 플러그인 스크립트"""
        return f'''@tool
extends EditorPlugin

# {plugin_name}
# AutoCI AI Generated Plugin

func _enter_tree():
    print("{plugin_name} 플러그인 활성화")

func _exit_tree():
    print("{plugin_name} 플러그인 비활성화")

func get_plugin_name():
    return "{plugin_name}"
'''

    async def _create_plugin_components(self, plugin_dir: Path, plugin_spec: Dict[str, Any]):
        """플러그인 컴포넌트 생성"""
        plugin_name = plugin_spec["name"]
        
        if plugin_name == "AutoCI_AI_Controller":
            await self._create_ai_controller_components(plugin_dir)
        elif plugin_name == "AutoCI_Scene_Automation":
            await self._create_scene_automation_components(plugin_dir)
        elif plugin_name == "AutoCI_Resource_Generator":
            await self._create_resource_generator_components(plugin_dir)
            
    async def _create_ai_controller_components_legacy(self, plugin_dir: Path):
        """AI 컨트롤러 컴포넌트 생성 (legacy)"""
        # ai_controller.gd
        ai_controller_script = '''extends Node
class_name AIController

# AutoCI AI Controller
# 핵심 AI 자동화 컨트롤러

var python_bridge
var automation_active: bool = false

signal ai_task_completed(task_name: String, result: Dictionary)
signal automation_status_changed(active: bool)

func _ready():
    print("AI Controller 초기화")
    setup_python_bridge()

func setup_python_bridge():
    """Python AI 시스템과의 브리지 설정"""
    # Python 스크립트 실행을 위한 설정
    python_bridge = {
        "ai_godot_automation": "python modules/ai_godot_automation.py",
        "scene_composer": "python modules/ai_scene_composer.py",
        "resource_manager": "python modules/ai_resource_manager.py",
        "gameplay_generator": "python modules/ai_gameplay_generator.py"
    }

func start_automation():
    """자동화 시작"""
    automation_active = true
    automation_status_changed.emit(true)
    print("AI 자동화 시작됨")

func stop_automation():
    """자동화 중지"""
    automation_active = false
    automation_status_changed.emit(false)
    print("AI 자동화 중지됨")

func create_ai_project():
    """AI 프로젝트 생성"""
    var task_data = {
        "command": "create_project",
        "game_type": "platformer",
        "project_name": "AI_Generated_Game"
    }
    
    execute_ai_task("ai_godot_automation", task_data)

func optimize_current_scene():
    """현재 씬 최적화"""
    var current_scene = EditorInterface.get_edited_scene_root()
    if current_scene:
        var task_data = {
            "command": "optimize_scene",
            "scene_path": current_scene.scene_file_path
        }
        
        execute_ai_task("scene_composer", task_data)

func generate_game_resources():
    """게임 리소스 생성"""
    var task_data = {
        "command": "generate_resources",
        "resource_types": ["texture", "audio", "material"]
    }
    
    execute_ai_task("resource_manager", task_data)

func execute_ai_task(system: String, task_data: Dictionary):
    """AI 작업 실행"""
    if not automation_active:
        print("자동화가 비활성화되어 있습니다")
        return
    
    var command = python_bridge.get(system, "")
    if command.is_empty():
        print("알 수 없는 AI 시스템: ", system)
        return
    
    # Python 스크립트 실행 (실제 구현에서는 더 정교한 통신 필요)
    var result = OS.execute("python", ["-c", "print('AI task executed')"])
    
    ai_task_completed.emit(system, {"status": "completed", "result": result})

func _process(delta):
    """실시간 AI 모니터링"""
    if automation_active:
        # AI 시스템 상태 체크
        pass
'''

        with open(plugin_dir / "ai_controller.gd", 'w') as f:
            f.write(ai_controller_script)
            
        # automation_dock.tscn (간단한 UI)
        dock_scene = '''[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://addons/autoci_ai/automation_dock.gd" id="1"]

[node name="AutomationDock" type="VBoxContainer"]
script = ExtResource("1")

[node name="Label" type="Label" parent="."]
layout_mode = 2
text = "AutoCI AI Automation"
horizontal_alignment = 1

[node name="StartButton" type="Button" parent="."]
layout_mode = 2
text = "Start AI Automation"

[node name="StopButton" type="Button" parent="."]
layout_mode = 2
text = "Stop AI Automation"

[node name="StatusLabel" type="Label" parent="."]
layout_mode = 2
text = "Status: Stopped"
'''

        with open(plugin_dir / "automation_dock.tscn", 'w') as f:
            f.write(dock_scene)
            
        # automation_dock.gd
        dock_script = '''extends VBoxContainer

var ai_controller

func _ready():
    # 버튼 연결
    $StartButton.pressed.connect(_on_start_pressed)
    $StopButton.pressed.connect(_on_stop_pressed)
    
    # AI 컨트롤러 찾기
    ai_controller = get_node("/root/AIController")
    if ai_controller:
        ai_controller.automation_status_changed.connect(_on_automation_status_changed)

func _on_start_pressed():
    if ai_controller:
        ai_controller.start_automation()

func _on_stop_pressed():
    if ai_controller:
        ai_controller.stop_automation()

func _on_automation_status_changed(active: bool):
    $StatusLabel.text = "Status: " + ("Running" if active else "Stopped")
'''

        with open(plugin_dir / "automation_dock.gd", 'w') as f:
            f.write(dock_script)
            
    async def _create_project_templates(self):
        """AI 최적화 프로젝트 템플릿 생성"""
        self.logger.info("프로젝트 템플릿 생성 시작")
        
        game_types = ["platformer", "racing", "puzzle", "rpg"]
        
        for game_type in game_types:
            template_dir = self.templates_dir / f"ai_{game_type}_template"
            template_dir.mkdir(parents=True, exist_ok=True)
            
            # project.godot 템플릿
            await self._create_project_template(template_dir, game_type)
            
            # 기본 씬 구조
            await self._create_template_scenes(template_dir, game_type)
            
            # AI 설정 파일
            await self._create_ai_config_template(template_dir, game_type)
            
        self.logger.info("모든 프로젝트 템플릿 생성 완료")
        
    async def _create_project_template(self, template_dir: Path, game_type: str):
        """프로젝트 템플릿 생성"""
        project_content = f'''[application]

config/name="AI {game_type.title()} Game"
config/description="AI Generated {game_type.title()} Game by AutoCI"
run/main_scene="res://scenes/Main.tscn"
config/features=PackedStringArray("4.3", "Forward Plus")
config/icon="res://icon.svg"

[autoload]

AIGameManager="*res://scripts/AIGameManager.gd"
AIResourceManager="*res://scripts/AIResourceManager.gd"

[debug]

file_logging/enable_file_logging=true

[input]

# AI 생성 입력 맵
{self._get_input_map_for_game_type(game_type)}

[layer_names]

# AI 생성 레이어
{self._get_layer_names_for_game_type(game_type)}

[physics]

# 게임 타입별 물리 설정
{self._get_physics_settings_for_game_type(game_type)}

[rendering]

renderer/rendering_method="forward_plus"
textures/canvas_textures/default_texture_filter=0
'''

        with open(template_dir / "project.godot", 'w') as f:
            f.write(project_content)
            
    def _get_input_map_for_game_type(self, game_type: str) -> str:
        """게임 타입별 입력 맵"""
        input_maps = {
            "platformer": '''
move_left={
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":65,"key_label":0,"unicode":97,"echo":false,"script":null)
, Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":4194319,"key_label":0,"unicode":0,"echo":false,"script":null)
]
}
move_right={
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":68,"key_label":0,"unicode":100,"echo":false,"script":null)
, Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":4194321,"key_label":0,"unicode":0,"echo":false,"script":null)
]
}
jump={
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":32,"key_label":0,"unicode":32,"echo":false,"script":null)
]
}''',
            "racing": '''
accelerate={
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":87,"key_label":0,"unicode":119,"echo":false,"script":null)
]
}
brake={
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":83,"key_label":0,"unicode":115,"echo":false,"script":null)
]
}
steer_left={
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":65,"key_label":0,"unicode":97,"echo":false,"script":null)
]
}
steer_right={
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":68,"key_label":0,"unicode":100,"echo":false,"script":null)
]
}''',
            "puzzle": '''
select={
"deadzone": 0.5,
"events": [Object(InputEventMouseButton,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"button_mask":1,"position":Vector2(0, 0),"global_position":Vector2(0, 0),"factor":1.0,"button_index":1,"canceled":false,"pressed":true,"double_click":false,"script":null)
]
}
reset={
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":82,"key_label":0,"unicode":114,"echo":false,"script":null)
]
}''',
            "rpg": '''
move_up={
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":87,"key_label":0,"unicode":119,"echo":false,"script":null)
]
}
move_down={
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":83,"key_label":0,"unicode":115,"echo":false,"script":null)
]
}
move_left={
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":65,"key_label":0,"unicode":97,"echo":false,"script":null)
]
}
move_right={
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":68,"key_label":0,"unicode":100,"echo":false,"script":null)
]
}
interact={
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":69,"key_label":0,"unicode":101,"echo":false,"script":null)
]
}'''
        }
        
        return input_maps.get(game_type, "")
        
    def _get_layer_names_for_game_type(self, game_type: str) -> str:
        """게임 타입별 레이어 이름"""
        layer_configs = {
            "platformer": '''
2d_physics/layer_1="Player"
2d_physics/layer_2="Enemy"
2d_physics/layer_3="Platform"
2d_physics/layer_4="Collectible"
2d_physics/layer_5="Environment"''',
            "racing": '''
3d_physics/layer_1="Car"
3d_physics/layer_2="Track"
3d_physics/layer_3="Obstacle"
3d_physics/layer_4="Checkpoint"
3d_physics/layer_5="Environment"''',
            "puzzle": '''
2d_physics/layer_1="Piece"
2d_physics/layer_2="Target"
2d_physics/layer_3="UI"
2d_physics/layer_4="Background"''',
            "rpg": '''
2d_physics/layer_1="Player"
2d_physics/layer_2="NPC"
2d_physics/layer_3="Enemy"
2d_physics/layer_4="Environment"
2d_physics/layer_5="Item"
2d_physics/layer_6="Trigger"'''
        }
        
        return layer_configs.get(game_type, "")
        
    def _get_physics_settings_for_game_type(self, game_type: str) -> str:
        """게임 타입별 물리 설정"""
        physics_configs = {
            "platformer": '''
2d/default_gravity=980
2d/default_linear_damp=0.0
2d/default_angular_damp=1.0''',
            "racing": '''
3d/default_gravity=9.8
3d/default_linear_damp=0.1
3d/default_angular_damp=0.1''',
            "puzzle": '''
2d/default_gravity=0
2d/default_linear_damp=1.0
2d/default_angular_damp=1.0''',
            "rpg": '''
2d/default_gravity=0
2d/default_linear_damp=5.0
2d/default_angular_damp=5.0'''
        }
        
        return physics_configs.get(game_type, "")
        
    async def _install_automation_tools(self):
        """자동화 도구 설치"""
        self.logger.info("자동화 도구 설치 시작")
        
        # Python-Godot 브리지 스크립트 생성
        bridge_script = '''#!/usr/bin/env python3
"""
AutoCI-Godot Bridge
Python AI 시스템과 Godot 간의 통신 브리지
"""

import sys
import json
import asyncio
from pathlib import Path

# AutoCI 모듈 경로 추가
autoci_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(autoci_path))

from modules.ai_godot_automation import AIGodotAutomation
from modules.ai_scene_composer import AISceneComposer
from modules.ai_resource_manager import AIResourceManager
from modules.ai_gameplay_generator import AIGameplayGenerator

class GodotAIBridge:
    """Godot-AI 브리지"""
    
    def __init__(self):
        self.godot_automation = AIGodotAutomation()
        self.scene_composer = AISceneComposer()
        self.resource_manager = AIResourceManager(Path.cwd())
        self.gameplay_generator = AIGameplayGenerator()
        
    async def execute_command(self, command_data: dict):
        """명령 실행"""
        command = command_data.get("command")
        
        if command == "create_project":
            return await self._create_project(command_data)
        elif command == "optimize_scene":
            return await self._optimize_scene(command_data)
        elif command == "generate_resources":
            return await self._generate_resources(command_data)
        elif command == "generate_gameplay":
            return await self._generate_gameplay(command_data)
        else:
            return {"error": f"Unknown command: {command}"}
            
    async def _create_project(self, data: dict):
        project = await self.godot_automation.create_complete_game_project(
            data.get("project_name", "AI_Game"),
            data.get("project_path", "./ai_game"),
            data.get("game_type", "platformer")
        )
        return {"success": True, "project": project.name if project else None}
        
    async def _optimize_scene(self, data: dict):
        # 씬 최적화 로직
        return {"success": True, "message": "Scene optimized"}
        
    async def _generate_resources(self, data: dict):
        # 리소스 생성 로직
        return {"success": True, "message": "Resources generated"}
        
    async def _generate_gameplay(self, data: dict):
        # 게임플레이 생성 로직
        return {"success": True, "message": "Gameplay generated"}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command_json = sys.argv[1]
        command_data = json.loads(command_json)
        
        bridge = GodotAIBridge()
        result = asyncio.run(bridge.execute_command(command_data))
        
        print(json.dumps(result))
    else:
        print(json.dumps({"error": "No command provided"}))
'''

        bridge_path = self.tools_dir / "godot_ai_bridge.py"
        with open(bridge_path, 'w') as f:
            f.write(bridge_script)
            
        # 실행 권한 부여
        os.chmod(bridge_path, 0o755)
        
        self.logger.info("자동화 도구 설치 완료")
        
    async def _optimize_godot_settings(self):
        """Godot 설정 최적화"""
        self.logger.info("Godot 설정 최적화 시작")
        
        # editor_settings-4.tres 최적화
        editor_settings = '''[gd_resource type="EditorSettings" format=3]

[resource]

# AI 최적화 설정
interface/editor/show_update_spinner = false
interface/editor/update_continuously = true
interface/editor/separate_distraction_mode = true
interface/scene_tabs/restore_scenes_on_load = true

# 자동화 친화적 설정
filesystem/file_dialog/show_hidden_files = true
filesystem/file_dialog/display_mode = 0
filesystem/import/blender/enabled = false
filesystem/import/fbx/enabled = false

# 성능 최적화
rendering/gl_compatibility/driver.windows = "opengl3"
rendering/gl_compatibility/driver.linuxbsd = "opengl3"
rendering/vulkan/rendering/back_end = 1

# AI 통합 설정
network/http_proxy/host = ""
network/http_proxy/port = -1
'''

        settings_dir = Path.home() / ".config" / "godot"
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        settings_path = settings_dir / "editor_settings-4.tres"
        with open(settings_path, 'w') as f:
            f.write(editor_settings)
            
        self.logger.info("Godot 설정 최적화 완료")
        
    async def _test_ai_integration(self):
        """AI 통합 테스트"""
        self.logger.info("AI 통합 테스트 시작")
        
        try:
            # Godot 실행 파일 테스트
            godot_exe = self.godot_dir / "godot"
            if self.platform == "windows":
                godot_exe = godot_exe.with_suffix(".exe")
                
            if not godot_exe.exists():
                raise FileNotFoundError("Godot 실행 파일이 없습니다")
                
            # 버전 확인
            result = subprocess.run([str(godot_exe), "--version"], 
                                  capture_output=True, text=True, timeout=10)
                                  
            if result.returncode == 0:
                self.logger.info(f"Godot 버전 확인: {result.stdout.strip()}")
            else:
                raise RuntimeError("Godot 실행 실패")
                
            # 플러그인 테스트
            for plugin_spec in self.ai_plugins:
                plugin_name = plugin_spec["name"].lower()
                plugin_dir = self.plugins_dir / plugin_name
                
                if not (plugin_dir / "plugin.cfg").exists():
                    raise FileNotFoundError(f"플러그인 {plugin_name} 설정 파일이 없습니다")
                    
            self.logger.info("모든 AI 통합 테스트 통과")
            
        except Exception as e:
            self.logger.error(f"AI 통합 테스트 실패: {e}")
            raise
            
    async def create_ai_ready_project(self, project_name: str, game_type: str, 
                                    output_path: Path) -> bool:
        """AI 준비된 프로젝트 생성"""
        self.logger.info(f"AI 준비 프로젝트 생성: {project_name} ({game_type})")
        
        try:
            # 템플릿 복사
            template_dir = self.templates_dir / f"ai_{game_type}_template"
            if not template_dir.exists():
                raise FileNotFoundError(f"템플릿이 없습니다: {game_type}")
                
            # 프로젝트 디렉토리 생성
            project_dir = output_path / project_name
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # 템플릿 복사
            shutil.copytree(template_dir, project_dir, dirs_exist_ok=True)
            
            # 플러그인 설치
            plugins_target_dir = project_dir / "addons"
            plugins_target_dir.mkdir(exist_ok=True)
            
            for plugin_spec in self.ai_plugins:
                plugin_name = plugin_spec["name"].lower()
                plugin_source = self.plugins_dir / plugin_name
                plugin_target = plugins_target_dir / plugin_name.replace("autoci_", "")
                
                if plugin_source.exists():
                    shutil.copytree(plugin_source, plugin_target, dirs_exist_ok=True)
                    
            # 프로젝트 설정 업데이트
            await self._update_project_settings(project_dir, project_name, game_type)
            
            self.logger.info(f"AI 준비 프로젝트 생성 완료: {project_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"프로젝트 생성 실패: {e}")
            return False
            
    async def _update_project_settings(self, project_dir: Path, project_name: str, game_type: str):
        """프로젝트 설정 업데이트"""
        project_file = project_dir / "project.godot"
        
        if project_file.exists():
            content = project_file.read_text()
            content = content.replace("AI {game_type.title()} Game", project_name)
            project_file.write_text(content)
            
    def get_godot_executable_path(self) -> Optional[Path]:
        """Godot 실행 파일 경로 반환"""
        godot_exe = self.godot_dir / "godot"
        if self.platform == "windows":
            godot_exe = godot_exe.with_suffix(".exe")
            
        return godot_exe if godot_exe.exists() else None
        
    def get_ai_plugin_list(self) -> List[str]:
        """설치된 AI 플러그인 목록 반환"""
        return [plugin["name"] for plugin in self.ai_plugins]
        
    def get_integration_status(self) -> Dict[str, Any]:
        """통합 상태 반환"""
        godot_exe = self.get_godot_executable_path()
        
        return {
            "godot_installed": godot_exe is not None,
            "godot_path": str(godot_exe) if godot_exe else None,
            "plugins_installed": len(self.get_ai_plugin_list()),
            "templates_available": len(list(self.templates_dir.glob("ai_*_template"))),
            "tools_available": len(list(self.tools_dir.glob("*.py")))
        }

    async def _create_ai_controller_components(self, plugin_dir: Path):
        """AI 컨트롤러 컴포넌트 생성"""
        # ai_controller.gd 생성
        ai_controller_script = '''@tool
extends Node

# AutoCI AI Controller Core
# Python AI 시스템과 Godot 간의 핵심 제어 인터페이스

signal ai_command_received(command: String, params: Dictionary)
signal automation_status_changed(status: String)

var python_bridge
var automation_active = false
var project_analyzer

func _ready():
    print("AI Controller 초기화 중...")
    setup_python_bridge()
    setup_project_analyzer()

func setup_python_bridge():
    """Python 브리지 설정"""
    python_bridge = preload("res://addons/autoci_ai/python_bridge.gd").new()
    add_child(python_bridge)
    python_bridge.connect("command_received", _on_python_command)

func setup_project_analyzer():
    """프로젝트 분석기 설정"""
    project_analyzer = preload("res://addons/autoci_ai/project_analyzer.gd").new()
    add_child(project_analyzer)

func start_automation():
    """자동화 시작"""
    automation_active = true
    automation_status_changed.emit("started")
    print("AI 자동화 시작됨")

func stop_automation():
    """자동화 중지"""
    automation_active = false
    automation_status_changed.emit("stopped")
    print("AI 자동화 중지됨")

func create_ai_project():
    """AI 프로젝트 생성"""
    var project_data = {
        "game_type": "platformer",
        "ai_optimized": true,
        "auto_generation": true
    }
    
    if python_bridge:
        python_bridge.send_command("create_project", project_data)

func optimize_current_scene():
    """현재 씬 최적화"""
    var current_scene = EditorInterface.get_edited_scene_root()
    if current_scene:
        var scene_data = project_analyzer.analyze_scene(current_scene)
        python_bridge.send_command("optimize_scene", scene_data)

func generate_game_resources():
    """게임 리소스 생성"""
    var resource_request = {
        "type": "complete_set",
        "game_type": "platformer",
        "quality": "high"
    }
    
    python_bridge.send_command("generate_resources", resource_request)

func _on_python_command(command: String, params: Dictionary):
    """Python 명령 처리"""
    match command:
        "create_scene":
            _create_scene_from_data(params)
        "add_node":
            _add_node_to_scene(params)
        "modify_properties":
            _modify_node_properties(params)
        "generate_script":
            _generate_node_script(params)

func _create_scene_from_data(data: Dictionary):
    """데이터로부터 씬 생성"""
    var scene = PackedScene.new()
    var root_node = _create_node_from_data(data.get("root", {}))
    
    scene.pack(root_node)
    
    var scene_path = "res://scenes/" + data.get("name", "ai_generated") + ".tscn"
    ResourceSaver.save(scene, scene_path)
    
    print("AI 씬 생성 완료: ", scene_path)

func _create_node_from_data(node_data: Dictionary) -> Node:
    """데이터로부터 노드 생성"""
    var node_type = node_data.get("type", "Node2D")
    var node = ClassDB.instantiate(node_type)
    
    # 기본 속성 설정
    node.name = node_data.get("name", "AINode")
    
    # 위치, 스케일, 회전 설정
    if node.has_method("set_position") and node_data.has("position"):
        var pos = node_data["position"]
        node.set_position(Vector2(pos[0], pos[1]))
    
    # 자식 노드들 생성
    for child_data in node_data.get("children", []):
        var child = _create_node_from_data(child_data)
        node.add_child(child)
    
    return node

func _add_node_to_scene(params: Dictionary):
    """씬에 노드 추가"""
    var current_scene = EditorInterface.get_edited_scene_root()
    if current_scene:
        var new_node = _create_node_from_data(params)
        current_scene.add_child(new_node)
        new_node.set_owner(current_scene)

func _modify_node_properties(params: Dictionary):
    """노드 속성 수정"""
    var node_path = params.get("path", "")
    var properties = params.get("properties", {})
    
    var current_scene = EditorInterface.get_edited_scene_root()
    if current_scene:
        var target_node = current_scene.get_node(node_path)
        if target_node:
            for prop in properties:
                target_node.set(prop, properties[prop])

func _generate_node_script(params: Dictionary):
    """노드 스크립트 생성"""
    var script_content = params.get("content", "")
    var script_path = params.get("path", "res://scripts/ai_generated.gd")
    
    var script = GDScript.new()
    script.source_code = script_content
    
    ResourceSaver.save(script, script_path)
    print("AI 스크립트 생성 완료: ", script_path)
'''
        
        ai_controller_path = plugin_dir / "ai_controller.gd"
        with open(ai_controller_path, 'w') as f:
            f.write(ai_controller_script)

        # scene_generator.gd 생성
        scene_generator_script = '''@tool
extends Node

# AutoCI Scene Generator
# AI 기반 씬 자동 생성

func generate_platformer_scene(scene_name: String) -> PackedScene:
    """플랫포머 씬 생성"""
    var scene = PackedScene.new()
    var root = Node2D.new()
    root.name = scene_name
    
    # 플레이어 추가
    var player = CharacterBody2D.new()
    player.name = "Player"
    player.position = Vector2(100, 400)
    root.add_child(player)
    
    # 플랫폼 추가
    for i in range(5):
        var platform = StaticBody2D.new()
        platform.name = "Platform" + str(i)
        platform.position = Vector2(200 + i * 300, 500 + randf_range(-50, 50))
        root.add_child(platform)
    
    scene.pack(root)
    return scene

func generate_racing_scene(scene_name: String) -> PackedScene:
    """레이싱 씬 생성"""
    var scene = PackedScene.new()
    var root = Node3D.new()
    root.name = scene_name
    
    # 차량 추가
    var vehicle = RigidBody3D.new()
    vehicle.name = "Vehicle"
    vehicle.position = Vector3(0, 1, 0)
    root.add_child(vehicle)
    
    # 트랙 요소들 추가
    for i in range(10):
        var track_piece = StaticBody3D.new()
        track_piece.name = "TrackPiece" + str(i)
        track_piece.position = Vector3(i * 10, 0, 0)
        root.add_child(track_piece)
    
    scene.pack(root)
    return scene
'''
        
        scene_generator_path = plugin_dir / "scene_generator.gd"
        with open(scene_generator_path, 'w') as f:
            f.write(scene_generator_script)

        # resource_manager.gd 생성
        resource_manager_script = '''@tool
extends Node

# AutoCI Resource Manager
# AI 기반 리소스 자동 관리

func generate_player_texture() -> ImageTexture:
    """플레이어 텍스처 생성"""
    var image = Image.create(32, 32, false, Image.FORMAT_RGBA8)
    
    # 간단한 플레이어 스프라이트 생성
    for x in range(32):
        for y in range(32):
            if x > 8 and x < 24 and y > 8 and y < 24:
                image.set_pixel(x, y, Color.BLUE)
            else:
                image.set_pixel(x, y, Color.TRANSPARENT)
    
    var texture = ImageTexture.new()
    texture.set_image(image)
    return texture

func generate_platform_texture() -> ImageTexture:
    """플랫폼 텍스처 생성"""
    var image = Image.create(64, 16, false, Image.FORMAT_RGBA8)
    
    # 플랫폼 텍스처 생성
    for x in range(64):
        for y in range(16):
            image.set_pixel(x, y, Color.BROWN)
    
    var texture = ImageTexture.new()
    texture.set_image(image)
    return texture

func save_generated_texture(texture: ImageTexture, path: String):
    """생성된 텍스처 저장"""
    ResourceSaver.save(texture, path)
    print("텍스처 저장 완료: ", path)
'''
        
        resource_manager_path = plugin_dir / "resource_manager.gd"
        with open(resource_manager_path, 'w') as f:
            f.write(resource_manager_script)

    async def _create_scene_automation_components(self, plugin_dir: Path):
        """씬 자동화 컴포넌트 생성"""
        # auto_composer.gd 생성
        auto_composer_script = '''@tool
extends Node

# AutoCI Auto Composer
# 지능형 씬 자동 구성

enum PlacementStrategy {
    RANDOM,
    GRID,
    ORGANIC,
    BALANCED,
    GUIDED
}

func compose_scene_intelligently(game_type: String, elements: Array, strategy: PlacementStrategy) -> Node:
    """지능형 씬 구성"""
    var root = Node2D.new()
    root.name = "AI_ComposedScene"
    
    match strategy:
        PlacementStrategy.RANDOM:
            _place_elements_randomly(root, elements)
        PlacementStrategy.GRID:
            _place_elements_in_grid(root, elements)
        PlacementStrategy.ORGANIC:
            _place_elements_organically(root, elements)
        PlacementStrategy.BALANCED:
            _place_elements_balanced(root, elements)
        PlacementStrategy.GUIDED:
            _place_elements_guided(root, elements, game_type)
    
    return root

func _place_elements_randomly(root: Node, elements: Array):
    """무작위 배치"""
    for element_data in elements:
        var node = _create_element(element_data)
        node.position = Vector2(
            randf_range(0, 1920),
            randf_range(0, 1080)
        )
        root.add_child(node)

func _place_elements_in_grid(root: Node, elements: Array):
    """격자 배치"""
    var grid_size = int(sqrt(elements.size())) + 1
    var cell_width = 1920.0 / grid_size
    var cell_height = 1080.0 / grid_size
    
    for i in range(elements.size()):
        var element_data = elements[i]
        var node = _create_element(element_data)
        
        var grid_x = i % grid_size
        var grid_y = i / grid_size
        
        node.position = Vector2(
            grid_x * cell_width + cell_width / 2,
            grid_y * cell_height + cell_height / 2
        )
        root.add_child(node)

func _place_elements_organically(root: Node, elements: Array):
    """유기적 배치"""
    var placed_positions = []
    var min_distance = 100.0
    
    for element_data in elements:
        var node = _create_element(element_data)
        var position = _find_organic_position(placed_positions, min_distance)
        
        node.position = position
        placed_positions.append(position)
        root.add_child(node)

func _place_elements_balanced(root: Node, elements: Array):
    """균형 배치"""
    # 화면을 4개 구역으로 나누어 균등 배치
    var zones = [
        Rect2(0, 0, 960, 540),      # 좌상
        Rect2(960, 0, 960, 540),    # 우상
        Rect2(0, 540, 960, 540),    # 좌하
        Rect2(960, 540, 960, 540)   # 우하
    ]
    
    for i in range(elements.size()):
        var element_data = elements[i]
        var node = _create_element(element_data)
        var zone = zones[i % 4]
        
        node.position = Vector2(
            zone.position.x + randf() * zone.size.x,
            zone.position.y + randf() * zone.size.y
        )
        root.add_child(node)

func _place_elements_guided(root: Node, elements: Array, game_type: String):
    """가이드된 배치"""
    for element_data in elements:
        var node = _create_element(element_data)
        var position = _get_guided_position(element_data, game_type)
        
        node.position = position
        root.add_child(node)

func _create_element(element_data: Dictionary) -> Node:
    """요소 생성"""
    var node_type = element_data.get("type", "Node2D")
    var node = ClassDB.instantiate(node_type)
    node.name = element_data.get("name", "Element")
    return node

func _find_organic_position(existing_positions: Array, min_distance: float) -> Vector2:
    """유기적 위치 찾기"""
    var max_attempts = 50
    
    for attempt in range(max_attempts):
        var pos = Vector2(randf_range(50, 1870), randf_range(50, 1030))
        var valid = true
        
        for existing_pos in existing_positions:
            if pos.distance_to(existing_pos) < min_distance:
                valid = false
                break
        
        if valid:
            return pos
    
    # 실패시 랜덤 위치 반환
    return Vector2(randf_range(50, 1870), randf_range(50, 1030))

func _get_guided_position(element_data: Dictionary, game_type: String) -> Vector2:
    """가이드된 위치 계산"""
    var element_type = element_data.get("type", "")
    
    match game_type:
        "platformer":
            return _get_platformer_position(element_type)
        "racing":
            return _get_racing_position(element_type)
        "puzzle":
            return _get_puzzle_position(element_type)
        _:
            return Vector2(960, 540)  # 중앙

func _get_platformer_position(element_type: String) -> Vector2:
    """플랫포머 요소 위치"""
    match element_type:
        "Player":
            return Vector2(100, 400)
        "Enemy":
            return Vector2(randf_range(500, 1500), randf_range(300, 600))
        "Platform":
            return Vector2(randf_range(200, 1800), randf_range(400, 800))
        _:
            return Vector2(randf_range(100, 1820), randf_range(100, 980))

func _get_racing_position(element_type: String) -> Vector2:
    """레이싱 요소 위치"""
    match element_type:
        "Vehicle":
            return Vector2(100, 500)
        "Checkpoint":
            return Vector2(randf_range(300, 1600), randf_range(400, 600))
        _:
            return Vector2(randf_range(100, 1820), randf_range(100, 980))

func _get_puzzle_position(element_type: String) -> Vector2:
    """퍼즐 요소 위치"""
    # 격자 기반 배치
    var grid_x = randi() % 8
    var grid_y = randi() % 6
    return Vector2(grid_x * 240 + 120, grid_y * 180 + 90)
'''
        
        auto_composer_path = plugin_dir / "auto_composer.gd"
        with open(auto_composer_path, 'w') as f:
            f.write(auto_composer_script)

        # layout_optimizer.gd 생성
        layout_optimizer_script = '''@tool
extends Node

# AutoCI Layout Optimizer
# 레이아웃 최적화 시스템

func optimize_scene_layout(scene_root: Node) -> void:
    """씬 레이아웃 최적화"""
    if not scene_root:
        return
    
    _resolve_overlaps(scene_root)
    _optimize_performance(scene_root)
    _ensure_accessibility(scene_root)

func _resolve_overlaps(scene_root: Node) -> void:
    """겹침 해결"""
    var nodes_with_positions = []
    _collect_positioned_nodes(scene_root, nodes_with_positions)
    
    var min_distance = 30.0
    
    for i in range(nodes_with_positions.size()):
        for j in range(i + 1, nodes_with_positions.size()):
            var node1 = nodes_with_positions[i]
            var node2 = nodes_with_positions[j]
            
            if not node1.has_method("get_position") or not node2.has_method("get_position"):
                continue
                
            var distance = node1.get_position().distance_to(node2.get_position())
            
            if distance < min_distance:
                _separate_nodes(node1, node2, min_distance)

func _optimize_performance(scene_root: Node) -> void:
    """성능 최적화"""
    var node_count = _count_all_nodes(scene_root)
    var max_nodes = 100
    
    if node_count > max_nodes:
        print("경고: 노드 수가 많습니다 (", node_count, "/", max_nodes, ")")
        _suggest_optimizations(scene_root)

func _ensure_accessibility(scene_root: Node) -> void:
    """접근성 확인"""
    var positioned_nodes = []
    _collect_positioned_nodes(scene_root, positioned_nodes)
    
    var center = Vector2(960, 540)
    var max_distance = 800
    
    for node in positioned_nodes:
        if node.has_method("get_position"):
            var distance = node.get_position().distance_to(center)
            if distance > max_distance:
                _move_node_closer(node, center, max_distance)

func _collect_positioned_nodes(node: Node, collection: Array) -> void:
    """위치가 있는 노드들 수집"""
    if node.has_method("get_position"):
        collection.append(node)
    
    for child in node.get_children():
        _collect_positioned_nodes(child, collection)

func _count_all_nodes(node: Node) -> int:
    """모든 노드 수 계산"""
    var count = 1
    for child in node.get_children():
        count += _count_all_nodes(child)
    return count

func _separate_nodes(node1: Node, node2: Node, min_distance: float) -> void:
    """노드 분리"""
    var pos1 = node1.get_position()
    var pos2 = node2.get_position()
    
    var direction = (pos2 - pos1).normalized()
    var new_pos2 = pos1 + direction * min_distance
    
    if node2.has_method("set_position"):
        node2.set_position(new_pos2)

func _move_node_closer(node: Node, center: Vector2, max_distance: float) -> void:
    """노드를 중앙에 가깝게 이동"""
    var current_pos = node.get_position()
    var direction = (current_pos - center).normalized()
    var new_pos = center + direction * max_distance
    
    if node.has_method("set_position"):
        node.set_position(new_pos)

func _suggest_optimizations(scene_root: Node) -> void:
    """최적화 제안"""
    print("최적화 제안:")
    print("- 불필요한 노드 제거")
    print("- 노드 그룹화 고려")
    print("- 오브젝트 풀링 사용")
'''
        
        layout_optimizer_path = plugin_dir / "layout_optimizer.gd"
        with open(layout_optimizer_path, 'w') as f:
            f.write(layout_optimizer_script)

    async def _create_resource_generator_components(self, plugin_dir: Path):
        """리소스 생성기 컴포넌트 생성"""
        # texture_generator.gd 생성
        texture_generator_script = '''@tool
extends Node

# AutoCI Texture Generator
# AI 기반 텍스처 자동 생성

enum TextureType {
    SOLID,
    GRADIENT,
    NOISE,
    PATTERN,
    SPRITE
}

func generate_texture(type: TextureType, width: int, height: int, params: Dictionary = {}) -> ImageTexture:
    """텍스처 생성"""
    var image = Image.create(width, height, false, Image.FORMAT_RGBA8)
    
    match type:
        TextureType.SOLID:
            _fill_solid_color(image, params.get("color", Color.WHITE))
        TextureType.GRADIENT:
            _fill_gradient(image, params)
        TextureType.NOISE:
            _fill_noise(image, params)
        TextureType.PATTERN:
            _fill_pattern(image, params)
        TextureType.SPRITE:
            _draw_sprite(image, params)
    
    var texture = ImageTexture.new()
    texture.set_image(image)
    return texture

func _fill_solid_color(image: Image, color: Color) -> void:
    """단색 채우기"""
    image.fill(color)

func _fill_gradient(image: Image, params: Dictionary) -> void:
    """그라디언트 채우기"""
    var start_color = params.get("start_color", Color.BLACK)
    var end_color = params.get("end_color", Color.WHITE)
    var direction = params.get("direction", "horizontal")
    
    var width = image.get_width()
    var height = image.get_height()
    
    for y in range(height):
        for x in range(width):
            var ratio: float
            
            if direction == "horizontal":
                ratio = float(x) / (width - 1)
            else:  # vertical
                ratio = float(y) / (height - 1)
            
            var color = start_color.lerp(end_color, ratio)
            image.set_pixel(x, y, color)

func _fill_noise(image: Image, params: Dictionary) -> void:
    """노이즈 채우기"""
    var noise_intensity = params.get("intensity", 0.5)
    var base_color = params.get("base_color", Color.GRAY)
    
    var width = image.get_width()
    var height = image.get_height()
    
    for y in range(height):
        for x in range(width):
            var noise_value = randf_range(-noise_intensity, noise_intensity)
            var color = Color(
                clamp(base_color.r + noise_value, 0.0, 1.0),
                clamp(base_color.g + noise_value, 0.0, 1.0),
                clamp(base_color.b + noise_value, 0.0, 1.0),
                base_color.a
            )
            image.set_pixel(x, y, color)

func _fill_pattern(image: Image, params: Dictionary) -> void:
    """패턴 채우기"""
    var pattern_type = params.get("pattern", "checkerboard")
    var color1 = params.get("color1", Color.BLACK)
    var color2 = params.get("color2", Color.WHITE)
    var scale = params.get("scale", 8)
    
    var width = image.get_width()
    var height = image.get_height()
    
    for y in range(height):
        for x in range(width):
            var color: Color
            
            match pattern_type:
                "checkerboard":
                    if (x / scale + y / scale) % 2 == 0:
                        color = color1
                    else:
                        color = color2
                "stripes_horizontal":
                    if (y / scale) % 2 == 0:
                        color = color1
                    else:
                        color = color2
                "stripes_vertical":
                    if (x / scale) % 2 == 0:
                        color = color1
                    else:
                        color = color2
                _:
                    color = color1
            
            image.set_pixel(x, y, color)

func _draw_sprite(image: Image, params: Dictionary) -> void:
    """스프라이트 그리기"""
    var sprite_type = params.get("sprite_type", "player")
    
    match sprite_type:
        "player":
            _draw_player_sprite(image)
        "enemy":
            _draw_enemy_sprite(image)
        "collectible":
            _draw_collectible_sprite(image)
        "platform":
            _draw_platform_sprite(image)

func _draw_player_sprite(image: Image) -> void:
    """플레이어 스프라이트 그리기"""
    var width = image.get_width()
    var height = image.get_height()
    var center_x = width / 2
    var center_y = height / 2
    
    # 머리 (원)
    _draw_circle(image, center_x, center_y - height / 4, width / 6, Color.BEIGE)
    
    # 몸 (사각형)
    _draw_rectangle(image, center_x - width / 6, center_y - height / 8, 
                   width / 3, height / 2, Color.BLUE)
    
    # 팔과 다리 (선)
    _draw_line(image, center_x - width / 4, center_y, center_x + width / 4, center_y, Color.BEIGE)
    _draw_line(image, center_x, center_y + height / 4, center_x - width / 8, center_y + height / 2, Color.BLUE)
    _draw_line(image, center_x, center_y + height / 4, center_x + width / 8, center_y + height / 2, Color.BLUE)

func _draw_enemy_sprite(image: Image) -> void:
    """적 스프라이트 그리기"""
    var width = image.get_width()
    var height = image.get_height()
    
    # 간단한 적 스프라이트 (삼각형)
    _draw_triangle(image, width / 2, height / 4, width / 4, height * 3 / 4, 
                  width * 3 / 4, height * 3 / 4, Color.RED)

func _draw_collectible_sprite(image: Image) -> void:
    """수집 아이템 스프라이트 그리기"""
    var width = image.get_width()
    var height = image.get_height()
    
    # 별 모양
    _draw_star(image, width / 2, height / 2, width / 3, Color.YELLOW)

func _draw_platform_sprite(image: Image) -> void:
    """플랫폼 스프라이트 그리기"""
    var width = image.get_width()
    var height = image.get_height()
    
    # 플랫폼 (그라디언트 사각형)
    _draw_rectangle(image, 0, 0, width, height, Color.BROWN)

func _draw_circle(image: Image, cx: int, cy: int, radius: int, color: Color) -> void:
    """원 그리기"""
    for y in range(max(0, cy - radius), min(image.get_height(), cy + radius + 1)):
        for x in range(max(0, cx - radius), min(image.get_width(), cx + radius + 1)):
            var dx = x - cx
            var dy = y - cy
            if dx * dx + dy * dy <= radius * radius:
                image.set_pixel(x, y, color)

func _draw_rectangle(image: Image, x: int, y: int, w: int, h: int, color: Color) -> void:
    """사각형 그리기"""
    for py in range(max(0, y), min(image.get_height(), y + h)):
        for px in range(max(0, x), min(image.get_width(), x + w)):
            image.set_pixel(px, py, color)

func _draw_line(image: Image, x1: int, y1: int, x2: int, y2: int, color: Color) -> void:
    """선 그리기 (Bresenham 알고리즘)"""
    var dx = abs(x2 - x1)
    var dy = abs(y2 - y1)
    var sx = 1 if x1 < x2 else -1
    var sy = 1 if y1 < y2 else -1
    var err = dx - dy
    
    var x = x1
    var y = y1
    
    while true:
        if x >= 0 and x < image.get_width() and y >= 0 and y < image.get_height():
            image.set_pixel(x, y, color)
        
        if x == x2 and y == y2:
            break
            
        var e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

func _draw_triangle(image: Image, x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, color: Color) -> void:
    """삼각형 그리기"""
    # 간단한 구현: 세 변을 선으로 그리기
    _draw_line(image, x1, y1, x2, y2, color)
    _draw_line(image, x2, y2, x3, y3, color)
    _draw_line(image, x3, y3, x1, y1, color)

func _draw_star(image: Image, cx: int, cy: int, radius: int, color: Color) -> void:
    """별 그리기"""
    var points = []
    for i in range(10):
        var angle = i * PI / 5
        var r = radius if i % 2 == 0 else radius / 2
        var x = cx + int(cos(angle) * r)
        var y = cy + int(sin(angle) * r)
        points.append(Vector2(x, y))
    
    # 별의 선들 그리기
    for i in range(points.size()):
        var next_i = (i + 1) % points.size()
        _draw_line(image, points[i].x, points[i].y, points[next_i].x, points[next_i].y, color)
'''
        
        texture_generator_path = plugin_dir / "texture_generator.gd"
        with open(texture_generator_path, 'w') as f:
            f.write(texture_generator_script)

        # audio_generator.gd 생성
        audio_generator_script = '''@tool
extends Node

# AutoCI Audio Generator
# AI 기반 오디오 자동 생성

enum AudioType {
    TONE,
    NOISE,
    MELODY,
    EFFECT
}

func generate_audio(type: AudioType, duration: float, params: Dictionary = {}) -> AudioStream:
    """오디오 생성"""
    var sample_rate = 44100
    var samples = int(duration * sample_rate)
    var data = PackedFloat32Array()
    data.resize(samples)
    
    match type:
        AudioType.TONE:
            _generate_tone(data, sample_rate, params)
        AudioType.NOISE:
            _generate_noise(data, params)
        AudioType.MELODY:
            _generate_melody(data, sample_rate, params)
        AudioType.EFFECT:
            _generate_effect(data, sample_rate, params)
    
    var audio_stream = AudioStreamWAV.new()
    audio_stream.data = data.to_byte_array()
    audio_stream.format = AudioStreamWAV.FORMAT_32_FLOAT
    audio_stream.mix_rate = sample_rate
    audio_stream.stereo = false
    
    return audio_stream

func _generate_tone(data: PackedFloat32Array, sample_rate: int, params: Dictionary) -> void:
    """톤 생성"""
    var frequency = params.get("frequency", 440.0)  # A4
    var amplitude = params.get("amplitude", 0.5)
    
    for i in range(data.size()):
        var t = float(i) / sample_rate
        data[i] = amplitude * sin(2 * PI * frequency * t)

func _generate_noise(data: PackedFloat32Array, params: Dictionary) -> void:
    """노이즈 생성"""
    var amplitude = params.get("amplitude", 0.3)
    
    for i in range(data.size()):
        data[i] = amplitude * randf_range(-1.0, 1.0)

func _generate_melody(data: PackedFloat32Array, sample_rate: int, params: Dictionary) -> void:
    """멜로디 생성"""
    var notes = params.get("notes", [261.63, 293.66, 329.63, 349.23])  # C, D, E, F
    var note_duration = params.get("note_duration", 0.5)
    var amplitude = params.get("amplitude", 0.4)
    
    var samples_per_note = int(note_duration * sample_rate)
    
    for i in range(data.size()):
        var note_index = (i / samples_per_note) % notes.size()
        var frequency = notes[note_index]
        var t = float(i % samples_per_note) / sample_rate
        data[i] = amplitude * sin(2 * PI * frequency * t)

func _generate_effect(data: PackedFloat32Array, sample_rate: int, params: Dictionary) -> void:
    """효과음 생성"""
    var effect_type = params.get("effect_type", "jump")
    var amplitude = params.get("amplitude", 0.6)
    
    match effect_type:
        "jump":
            _generate_jump_sound(data, sample_rate, amplitude)
        "collect":
            _generate_collect_sound(data, sample_rate, amplitude)
        "hit":
            _generate_hit_sound(data, sample_rate, amplitude)
        "explosion":
            _generate_explosion_sound(data, sample_rate, amplitude)

func _generate_jump_sound(data: PackedFloat32Array, sample_rate: int, amplitude: float) -> void:
    """점프 사운드"""
    for i in range(data.size()):
        var t = float(i) / sample_rate
        var frequency = 300 + 200 * exp(-t * 5)  # 주파수가 감소
        data[i] = amplitude * sin(2 * PI * frequency * t) * exp(-t * 3)

func _generate_collect_sound(data: PackedFloat32Array, sample_rate: int, amplitude: float) -> void:
    """수집 사운드"""
    for i in range(data.size()):
        var t = float(i) / sample_rate
        var frequency = 400 + 300 * t  # 주파수가 증가
        data[i] = amplitude * sin(2 * PI * frequency * t) * exp(-t * 2)

func _generate_hit_sound(data: PackedFloat32Array, sample_rate: int, amplitude: float) -> void:
    """타격 사운드"""
    for i in range(data.size()):
        var t = float(i) / sample_rate
        var noise = randf_range(-0.3, 0.3)
        var tone = 150 * sin(2 * PI * 100 * t)
        data[i] = amplitude * (noise + tone) * exp(-t * 8)

func _generate_explosion_sound(data: PackedFloat32Array, sample_rate: int, amplitude: float) -> void:
    """폭발 사운드"""
    for i in range(data.size()):
        var t = float(i) / sample_rate
        var noise = randf_range(-1.0, 1.0)
        var bass = 0.3 * sin(2 * PI * 60 * t)
        data[i] = amplitude * (noise * 0.7 + bass) * exp(-t * 2)
'''
        
        audio_generator_path = plugin_dir / "audio_generator.gd"
        with open(audio_generator_path, 'w') as f:
            f.write(audio_generator_script)

    async def _create_template_scenes(self, template_dir: Path, game_type: str):
        """템플릿 씬 생성"""
        scenes_dir = template_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        
        if game_type == "platformer":
            await self._create_platformer_scenes(scenes_dir)
        elif game_type == "racing":
            await self._create_racing_scenes(scenes_dir)
        elif game_type == "puzzle":
            await self._create_puzzle_scenes(scenes_dir)
        elif game_type == "rpg":
            await self._create_rpg_scenes(scenes_dir)

    async def _create_platformer_scenes(self, scenes_dir: Path):
        """플랫포머 씬 생성"""
        # Main.tscn 생성
        main_scene = '''[gd_scene load_steps=3 format=3 uid="uid://platformer_main"]

[ext_resource type="Script" path="res://scripts/GameManager.gd" id="1"]
[ext_resource type="PackedScene" path="res://scenes/Player.tscn" id="2"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

[node name="Player" parent="." instance=ExtResource("2")]
position = Vector2(100, 400)

[node name="UI" type="CanvasLayer" parent="."]

[node name="HUD" type="Control" parent="UI"]
layout_mode = 3
anchors_preset = 15

[node name="HealthBar" type="ProgressBar" parent="UI/HUD"]
layout_mode = 0
offset_right = 200.0
offset_bottom = 20.0
'''
        
        with open(scenes_dir / "Main.tscn", 'w') as f:
            f.write(main_scene)

        # Player.tscn 생성
        player_scene = '''[gd_scene load_steps=3 format=3 uid="uid://platformer_player"]

[ext_resource type="Script" path="res://scripts/Player.gd" id="1"]

[sub_resource type="RectangleShape2D" id="RectangleShape2D_1"]
size = Vector2(32, 32)

[node name="Player" type="CharacterBody2D"]
script = ExtResource("1")

[node name="Sprite2D" type="Sprite2D" parent="."]
modulate = Color(0, 0, 1, 1)
texture = preload("res://assets/textures/player.tres")

[node name="CollisionShape2D" type="CollisionShape2D" parent="."]
shape = SubResource("RectangleShape2D_1")
'''
        
        with open(scenes_dir / "Player.tscn", 'w') as f:
            f.write(player_scene)

    async def _create_racing_scenes(self, scenes_dir: Path):
        """레이싱 씬 생성"""
        # Main.tscn 생성
        main_scene = '''[gd_scene load_steps=2 format=3 uid="uid://racing_main"]

[ext_resource type="Script" path="res://scripts/RaceManager.gd" id="1"]

[node name="Main" type="Node3D"]
script = ExtResource("1")

[node name="Track" type="Node3D" parent="."]

[node name="Vehicle" type="RigidBody3D" parent="."]
position = Vector3(0, 1, 0)

[node name="Camera3D" type="Camera3D" parent="Vehicle"]
transform = Transform3D(1, 0, 0, 0, 0.707107, 0.707107, 0, -0.707107, 0.707107, 0, 5, 10)
'''
        
        with open(scenes_dir / "Main.tscn", 'w') as f:
            f.write(main_scene)

    async def _create_puzzle_scenes(self, scenes_dir: Path):
        """퍼즐 씬 생성"""
        # Main.tscn 생성
        main_scene = '''[gd_scene load_steps=2 format=3 uid="uid://puzzle_main"]

[ext_resource type="Script" path="res://scripts/PuzzleManager.gd" id="1"]

[node name="Main" type="Control"]
layout_mode = 3
anchors_preset = 15
script = ExtResource("1")

[node name="GameBoard" type="GridContainer" parent="."]
layout_mode = 1
anchors_preset = 8
anchor_left = 0.5
anchor_top = 0.5
anchor_right = 0.5
anchor_bottom = 0.5
columns = 4
'''
        
        with open(scenes_dir / "Main.tscn", 'w') as f:
            f.write(main_scene)

    async def _create_rpg_scenes(self, scenes_dir: Path):
        """RPG 씬 생성"""
        # Main.tscn 생성
        main_scene = '''[gd_scene load_steps=3 format=3 uid="uid://rpg_main"]

[ext_resource type="Script" path="res://scripts/WorldManager.gd" id="1"]
[ext_resource type="PackedScene" path="res://scenes/Player.tscn" id="2"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

[node name="Player" parent="." instance=ExtResource("2")]
position = Vector2(500, 500)

[node name="NPCs" type="Node2D" parent="."]

[node name="Environment" type="Node2D" parent="."]

[node name="UI" type="CanvasLayer" parent="."]

[node name="HUD" type="Control" parent="UI"]
layout_mode = 3
anchors_preset = 15
'''
        
        with open(scenes_dir / "Main.tscn", 'w') as f:
            f.write(main_scene)

    async def _create_ai_config_template(self, template_dir: Path, game_type: str):
        """AI 설정 템플릿 생성"""
        ai_config = {
            "game_type": game_type,
            "ai_settings": {
                "auto_generation": True,
                "smart_placement": True,
                "balance_optimization": True,
                "performance_monitoring": True
            },
            "generation_parameters": {
                "scene_complexity": "medium",
                "resource_quality": "high",
                "difficulty_progression": "balanced"
            },
            "automation": {
                "enabled": True,
                "interval_minutes": 30,
                "auto_save": True,
                "auto_backup": True
            }
        }
        
        config_path = template_dir / "ai_config.json"
        with open(config_path, 'w') as f:
            json.dump(ai_config, f, indent=2)

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Godot AI 통합 시스템")
    parser.add_argument("--install", action="store_true", help="Godot 자동 설치")
    parser.add_argument("--plugins", action="store_true", help="AI 플러그인 설치")
    parser.add_argument("--templates", action="store_true", help="프로젝트 템플릿 생성")
    parser.add_argument("--bridge", action="store_true", help="Python-Godot 브리지 설정")
    parser.add_argument("--status", action="store_true", help="통합 상태 확인")
    parser.add_argument("--test", action="store_true", help="테스트 프로젝트 생성")
    
    args = parser.parse_args()
    
    print("🤖 Godot AI 통합 시스템")
    print("=" * 60)
    
    async def run_integration():
        integration = GodotAIIntegration()
        
        if args.install:
            print("📥 Godot 엔진 설치 중...")
            try:
                await integration._download_and_install_godot()
                print("✅ Godot 설치 완료!")
            except Exception as e:
                print(f"❌ Godot 설치 실패: {e}")
                return
                
        if args.plugins:
            print("🔌 AI 플러그인 설치 중...")
            try:
                await integration._create_ai_plugins()
                print("✅ AI 플러그인 설치 완료!")
            except Exception as e:
                print(f"❌ 플러그인 설치 실패: {e}")
                return
                
        if args.templates:
            print("📋 프로젝트 템플릿 생성 중...")
            try:
                await integration._create_project_templates()
                print("✅ 프로젝트 템플릿 생성 완료!")
            except Exception as e:
                print(f"❌ 템플릿 생성 실패: {e}")
                return
                
        if args.bridge:
            print("🌉 Python-Godot 브리지 설정 중...")
            try:
                await integration._install_automation_tools()
                print("✅ 브리지 설정 완료!")
            except Exception as e:
                print(f"❌ 브리지 설정 실패: {e}")
                return
                
        if args.status:
            print("📊 통합 상태 확인 중...")
            status = integration.get_integration_status()
            print(f"Godot 설치됨: {'✅' if status['godot_installed'] else '❌'}")
            print(f"플러그인 수: {status['plugins_installed']}")
            print(f"템플릿 수: {status['templates_available']}")
            print(f"도구 수: {status['tools_available']}")
            return
            
        if args.test:
            print("🧪 테스트 프로젝트 생성 중...")
            try:
                test_success = await integration.create_ai_ready_project(
                    "AI_Test_Game", "platformer", Path("/tmp")
                )
                if test_success:
                    print("✅ 테스트 프로젝트 생성 완료!")
                else:
                    print("❌ 프로젝트 생성 실패")
            except Exception as e:
                print(f"❌ 테스트 프로젝트 생성 실패: {e}")
            return
            
        # 인수가 없으면 전체 설치
        if not any([args.install, args.plugins, args.templates, args.bridge, args.status, args.test]):
            print("🚀 전체 AI 최적화 환경 구축 중...")
            success = await integration.setup_ai_optimized_godot()
            
            if success:
                print("✅ Godot AI 통합 완료!")
                
                # 상태 확인
                status = integration.get_integration_status()
                print(f"Godot 설치됨: {'✅' if status['godot_installed'] else '❌'}")
                print(f"플러그인 수: {status['plugins_installed']}")
                print(f"템플릿 수: {status['templates_available']}")
                print(f"도구 수: {status['tools_available']}")
            else:
                print("❌ Godot AI 통합 실패")
            
    asyncio.run(run_integration())

if __name__ == "__main__":
    main()