#!/usr/bin/env python3
"""
Godot Live Integration - AutoCI와 Godot 실시간 통합
autoci 실행 시 자동으로 Godot을 열고 진행상황을 표시
"""

import os
import sys
import asyncio
import subprocess
import json
import socket
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

class GodotLiveIntegration:
    """Godot 실시간 통합 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger("GodotLiveIntegration")
        self.project_root = Path(__file__).parent.parent
        self.godot_path = self._find_godot_executable()
        self.godot_process = None
        self.server_socket = None
        self.is_connected = False
        self.current_project = None
        self.stats = {
            "start_time": datetime.now(),
            "tasks_completed": 0,
            "current_task": "대기 중",
            "ai_model": "미선택",
            "learning_progress": 0,
            "games_created": 0
        }
        
    def _find_godot_executable(self) -> Optional[Path]:
        """Godot 실행파일 찾기"""
        # 설치된 Godot 경로들
        possible_paths = [
            self.project_root / "godot_ai" / "godot",
            self.project_root / "godot_ai" / "Godot.exe",
            self.project_root / "godot_ai" / "Godot_v4.3-stable_linux.x86_64",
            self.project_root / "godot_ai" / "Godot_v4.3-stable_win64.exe",
            Path("/usr/local/bin/godot"),
            Path("/opt/godot/godot"),
            Path("C:/Program Files/Godot/Godot.exe"),
            Path("C:/Godot/Godot.exe")
        ]
        
        # WSL에서 Windows Godot 찾기
        if "microsoft" in os.uname().release.lower():
            windows_paths = [
                Path("/mnt/c/Program Files/Godot/Godot.exe"),
                Path("/mnt/c/Godot/Godot.exe"),
                Path("/mnt/d/Godot/Godot.exe")
            ]
            possible_paths.extend(windows_paths)
        
        for path in possible_paths:
            if path.exists():
                return path
                
        return None
        
    async def start_godot_with_dashboard(self):
        """Godot을 대시보드 프로젝트와 함께 시작"""
        if not self.godot_path:
            self.logger.warning("Godot 실행파일을 찾을 수 없습니다.")
            print("⚠️  Godot이 설치되지 않았습니다. 진행상황 표시 없이 계속합니다.")
            return False
            
        # AutoCI 대시보드 프로젝트 경로
        dashboard_project = self.project_root / "godot_projects" / "autoci_dashboard"
        
        # 대시보드 프로젝트가 없으면 생성
        if not dashboard_project.exists():
            await self._create_dashboard_project(dashboard_project)
        
        # 통신 서버 시작
        await self._start_communication_server()
        
        # Godot 실행
        try:
            cmd = [str(self.godot_path), str(dashboard_project / "project.godot")]
            
            # WSL에서 Windows Godot 실행 시
            if "microsoft" in os.uname().release.lower() and str(self.godot_path).startswith("/mnt/"):
                # Windows 경로로 변환
                win_path = str(self.godot_path).replace("/mnt/c/", "C:/").replace("/mnt/d/", "D:/")
                win_project = str(dashboard_project).replace("/mnt/d/", "D:/").replace("/", "\\")
                cmd = ["cmd.exe", "/c", win_path, win_project + "\\project.godot"]
            
            self.godot_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            print("🎮 Godot AI 대시보드가 시작되었습니다!")
            print("   진행상황이 Godot 창에 실시간으로 표시됩니다.")
            
            # 연결 대기
            await asyncio.sleep(2)
            return True
            
        except Exception as e:
            self.logger.error(f"Godot 시작 실패: {e}")
            print(f"⚠️  Godot 시작 실패: {e}")
            return False
    
    async def _create_dashboard_project(self, project_path: Path):
        """AutoCI 대시보드 프로젝트 생성"""
        print("📊 AutoCI 대시보드 프로젝트 생성 중...")
        
        project_path.mkdir(parents=True, exist_ok=True)
        
        # project.godot 생성
        project_config = """
; Engine configuration file.
; It's best edited using the editor UI and not directly,
; since the properties are organized in sections here.

[application]

config/name="AutoCI Dashboard"
config/description="AutoCI 실시간 진행상황 대시보드"
run/main_scene="res://main.tscn"
config/features=PackedStringArray("4.3", "GL Compatibility")
config/icon="res://icon.svg"

[display]

window/size/viewport_width=1280
window/size/viewport_height=720
window/size/resizable=true

[rendering]

renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"
"""
        (project_path / "project.godot").write_text(project_config)
        
        # 메인 씬 생성
        main_scene = """[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://Dashboard.gd" id="1"]
[ext_resource type="Theme" path="res://theme.tres" id="2"]

[node name="Dashboard" type="Control"]
anchor_right = 1.0
anchor_bottom = 1.0
theme = ExtResource("2")
script = ExtResource("1")

[node name="Background" type="ColorRect" parent="."]
anchor_right = 1.0
anchor_bottom = 1.0
color = Color(0.1, 0.1, 0.15, 1)

[node name="Title" type="Label" parent="."]
anchor_left = 0.5
anchor_right = 0.5
margin_left = -300.0
margin_top = 20.0
margin_right = 300.0
margin_bottom = 80.0
theme_override_font_sizes/font_size = 36
text = "🚀 AutoCI 실시간 대시보드"
horizontal_alignment = 1

[node name="StatsContainer" type="VBoxContainer" parent="."]
margin_left = 50.0
margin_top = 120.0
margin_right = 600.0
margin_bottom = 680.0

[node name="CurrentTask" type="RichTextLabel" parent="StatsContainer"]
custom_minimum_size = Vector2(0, 60)
bbcode_enabled = true
text = "[b]현재 작업:[/b] 대기 중..."

[node name="Progress" type="ProgressBar" parent="StatsContainer"]
custom_minimum_size = Vector2(0, 30)
value = 0.0

[node name="Stats" type="RichTextLabel" parent="StatsContainer"]
custom_minimum_size = Vector2(0, 400)
bbcode_enabled = true
text = "[b]통계[/b]"

[node name="LogContainer" type="VBoxContainer" parent="."]
margin_left = 650.0
margin_top = 120.0
margin_right = 1230.0
margin_bottom = 680.0

[node name="LogTitle" type="Label" parent="LogContainer"]
text = "실시간 로그"
theme_override_font_sizes/font_size = 24

[node name="LogScroll" type="ScrollContainer" parent="LogContainer"]
custom_minimum_size = Vector2(0, 520)

[node name="LogText" type="RichTextLabel" parent="LogContainer/LogScroll"]
custom_minimum_size = Vector2(560, 500)
bbcode_enabled = true
scroll_following = true
"""
        (project_path / "main.tscn").write_text(main_scene)
        
        # 대시보드 스크립트 생성
        dashboard_script = '''extends Control

var socket: StreamPeerTCP
var is_connected := false
var reconnect_timer := 0.0

func _ready():
    print("AutoCI Dashboard 시작")
    _connect_to_autoci()
    
func _connect_to_autoci():
    socket = StreamPeerTCP.new()
    var result = socket.connect_to_host("127.0.0.1", 9999)
    if result == OK:
        is_connected = true
        print("AutoCI에 연결됨")
    else:
        print("AutoCI 연결 실패, 재시도 예정...")

func _process(delta):
    if not is_connected:
        reconnect_timer += delta
        if reconnect_timer > 3.0:
            reconnect_timer = 0.0
            _connect_to_autoci()
        return
    
    # 데이터 수신
    if socket.get_available_bytes() > 0:
        var data = socket.get_string(socket.get_available_bytes())
        _process_data(data)

func _process_data(data: String):
    try:
        var json_data = JSON.parse_string(data)
        if json_data:
            _update_dashboard(json_data)
    except:
        pass

func _update_dashboard(data: Dictionary):
    # 현재 작업 업데이트
    if data.has("current_task"):
        $StatsContainer/CurrentTask.text = "[b]현재 작업:[/b] " + data.current_task
    
    # 진행률 업데이트
    if data.has("progress"):
        $StatsContainer/Progress.value = data.progress
    
    # 통계 업데이트
    if data.has("stats"):
        var stats_text = "[b]📊 시스템 통계[/b]\\n\\n"
        stats_text += "⏱️ 실행 시간: " + data.stats.get("uptime", "0") + "\\n"
        stats_text += "🎮 생성된 게임: " + str(data.stats.get("games_created", 0)) + "개\\n"
        stats_text += "📚 학습한 주제: " + str(data.stats.get("topics_learned", 0)) + "개\\n"
        stats_text += "🔧 완료된 작업: " + str(data.stats.get("tasks_completed", 0)) + "개\\n"
        stats_text += "🤖 AI 모델: " + data.stats.get("ai_model", "미선택") + "\\n"
        $StatsContainer/Stats.text = stats_text
    
    # 로그 추가
    if data.has("log"):
        var log_entry = "[color=#" + data.get("color", "ffffff") + "]"
        log_entry += "[" + data.get("time", "") + "] "
        log_entry += data.log + "[/color]\\n"
        $LogContainer/LogScroll/LogText.append_text(log_entry)
'''
        (project_path / "Dashboard.gd").write_text(dashboard_script)
        
        # 테마 파일 생성 (간단한 다크 테마)
        theme_content = """[gd_resource type="Theme" format=3]

[resource]
default_font_size = 16
"""
        (project_path / "theme.tres").write_text(theme_content)
        
        # 아이콘 생성 (간단한 SVG)
        icon_svg = """<svg height="128" width="128" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="10" rx="20" ry="20" width="108" height="108" style="fill:#1a1a2e;"/>
  <text x="64" y="75" font-family="Arial" font-size="48" text-anchor="middle" fill="#f39c12">AI</text>
</svg>"""
        (project_path / "icon.svg").write_text(icon_svg)
        
        print("✅ 대시보드 프로젝트 생성 완료")
    
    async def _start_communication_server(self):
        """AutoCI와 Godot 간 통신 서버 시작"""
        def server_thread():
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind(('127.0.0.1', 9999))
                self.server_socket.listen(1)
                self.server_socket.settimeout(0.1)  # Non-blocking
                
                while True:
                    try:
                        client, addr = self.server_socket.accept()
                        self.is_connected = True
                        self.logger.info(f"Godot 연결됨: {addr}")
                    except socket.timeout:
                        pass
                    except Exception as e:
                        if self.server_socket:
                            break
                    time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"서버 오류: {e}")
        
        thread = threading.Thread(target=server_thread, daemon=True)
        thread.start()
    
    async def update_dashboard(self, data: Dict[str, Any]):
        """대시보드 업데이트"""
        if not self.is_connected:
            return
        
        # 실행 시간 계산
        uptime = datetime.now() - self.stats["start_time"]
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        
        # 데이터 준비
        update_data = {
            "current_task": data.get("task", self.stats["current_task"]),
            "progress": data.get("progress", 0),
            "stats": {
                "uptime": f"{hours}시간 {minutes}분",
                "games_created": self.stats["games_created"],
                "topics_learned": data.get("topics_learned", 0),
                "tasks_completed": self.stats["tasks_completed"],
                "ai_model": self.stats["ai_model"]
            },
            "log": data.get("log", ""),
            "time": datetime.now().strftime("%H:%M:%S"),
            "color": data.get("color", "ffffff")
        }
        
        # 통계 업데이트
        if "task" in data:
            self.stats["current_task"] = data["task"]
        if "tasks_completed" in data:
            self.stats["tasks_completed"] = data["tasks_completed"]
        if "games_created" in data:
            self.stats["games_created"] = data["games_created"]
        if "ai_model" in data:
            self.stats["ai_model"] = data["ai_model"]
        
        # Godot에 전송
        try:
            if hasattr(self, 'client_socket') and self.client_socket:
                message = json.dumps(update_data) + "\n"
                self.client_socket.send(message.encode())
        except:
            self.is_connected = False
    
    async def log_to_dashboard(self, message: str, color: str = "ffffff"):
        """대시보드에 로그 메시지 전송"""
        await self.update_dashboard({
            "log": message,
            "color": color
        })
    
    def stop_godot(self):
        """Godot 종료"""
        if self.godot_process:
            self.godot_process.terminate()
            self.godot_process = None
        
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        
        self.is_connected = False

# 전역 인스턴스
_godot_integration = None

def get_godot_integration() -> GodotLiveIntegration:
    """Godot 통합 인스턴스 가져오기"""
    global _godot_integration
    if _godot_integration is None:
        _godot_integration = GodotLiveIntegration()
    return _godot_integration

async def start_godot_dashboard():
    """Godot 대시보드 시작 (외부에서 호출용)"""
    integration = get_godot_integration()
    return await integration.start_godot_with_dashboard()

async def update_godot_dashboard(data: Dict[str, Any]):
    """Godot 대시보드 업데이트 (외부에서 호출용)"""
    integration = get_godot_integration()
    await integration.update_dashboard(data)

async def log_to_godot(message: str, color: str = "ffffff"):
    """Godot에 로그 전송 (외부에서 호출용)"""
    integration = get_godot_integration()
    await integration.log_to_dashboard(message, color)