#!/usr/bin/env python3
"""
Godot 실시간 대시보드 시스템
WSL 환경에서 Windows Godot을 자동으로 열고 실시간으로 작업 상황을 표시
"""

import asyncio
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import socket
import threading
import queue
import urllib.request
import zipfile

class GodotRealtimeDashboard:
    """Godot 실시간 대시보드 관리자"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.godot_project_path = self.project_root / "godot_dashboard"
        self.godot_executable = None
        self.dashboard_port = 12345
        self.is_running = False
        self.message_queue = queue.Queue()
        self.error_queue = queue.Queue()
        self.stats = {
            "start_time": datetime.now(),
            "tasks_completed": 0,
            "errors_count": 0,
            "current_task": "초기화 중...",
            "progress": 0,
            "ai_status": "준비 중"
        }
        
    def find_godot_executable(self) -> Optional[str]:
        """Godot 실행 파일 찾기 (WSL에서 Windows 경로)"""
        # 저장된 경로 먼저 확인
        saved_path = self._load_godot_path()
        if saved_path and Path(saved_path).exists():
            print(f"✅ 저장된 Godot 경로 사용: {saved_path}")
            return saved_path
        
        # AI 수정된 Godot 우선 확인
        print("🔍 AI 수정된 Godot을 찾는 중...")
        
        # 수정된 Godot 빌드 경로들 (우선순위 높음)
        modified_godot_paths = [
            # Windows AI 빌드 (최우선)
            str(self.project_root / "godot_ai_build" / "output" / "godot.windows.editor.x86_64.exe"),
            str(self.project_root / "godot_ai_build" / "output" / "godot.ai.editor.windows.x86_64.exe"),
            # AutoCI 프로젝트의 수정된 Godot (빌드 예정 경로)
            str(self.project_root / "godot_modified" / "bin" / "godot.windows.editor.x86_64.exe"),
            str(self.project_root / "godot_modified" / "godot.windows.editor.x86_64.exe"),
            # 간단한 빌드로 설정된 Godot
            str(self.project_root / "godot_ai_build" / "Godot_v4.3-stable_win64.exe"),
            # 상위 디렉토리의 수정된 Godot
            str(self.project_root.parent / "godot-modified" / "bin" / "godot.windows.editor.x86_64.exe"),
            str(self.project_root.parent / "godot-ai" / "bin" / "godot.windows.editor.x86_64.exe"),
            str(self.project_root.parent / "godot-build" / "bin" / "godot.windows.editor.x86_64.exe"),
            # Windows 경로의 수정된 Godot
            "/mnt/d/godot-modified/bin/godot.windows.editor.x86_64.exe",
            "/mnt/c/godot-modified/bin/godot.windows.editor.x86_64.exe",
            # 프로젝트 내 AI 폴더
            str(self.project_root / "godot_ai" / "godot.windows.editor.x86_64.exe"),
            # Linux AI 빌드 (WSL에서만 사용 가능한 경우를 위해 마지막에 배치)
            str(self.project_root / "godot_ai_build" / "output" / "godot.ai.editor.linux.x86_64"),
            # 일반 Linux Godot
            str(self.project_root / "godot_engine" / "Godot_v4.3-stable_linux.x86_64"),
        ]
        
        # 수정된 Godot 먼저 확인
        for path in modified_godot_paths:
            if path and Path(path).exists():
                print(f"✅ AI 수정된 Godot 찾음: {path}")
                return path
        
        # 일반 Godot 경로들 확인
        possible_paths = [
            # 일반적인 설치 경로
            "/mnt/c/Program Files/Godot/Godot.exe",
            "/mnt/c/Program Files (x86)/Godot/Godot.exe",
            "/mnt/d/Godot/Godot.exe",
            "/mnt/c/Godot/Godot.exe",
            # Steam 설치 경로
            "/mnt/c/Program Files (x86)/Steam/steamapps/common/Godot Engine/godot.windows.opt.tools.64.exe",
            "/mnt/c/Program Files/Steam/steamapps/common/Godot Engine/godot.windows.opt.tools.64.exe",
            # 사용자 다운로드 폴더
            f"/mnt/c/Users/{os.environ.get('USER', '')}/Downloads/Godot_v4.3-stable_win64.exe",
            f"/mnt/c/Users/{os.environ.get('USER', '')}/Downloads/Godot.exe",
            # 프로젝트 디렉토리의 Godot
            str(self.project_root / "Godot_v4.3-stable_win64.exe"),
            str(self.project_root / "Godot_v4.2-stable_win64.exe"),
            str(self.project_root / "Godot.exe"),
            # godot_bin 디렉토리
            str(self.project_root / "godot_bin" / "Godot_v4.3-stable_win64.exe"),
            str(self.project_root / "godot_bin" / "Godot.exe"),
        ]
        
        # 와일드카드 패턴으로 검색
        wildcard_patterns = [
            "/mnt/c/Users/*/Downloads/Godot*.exe",
            "/mnt/d/*/Godot*.exe",
            "/mnt/c/*/Godot*.exe",
        ]
        
        # 일반 경로 확인
        for path in possible_paths:
            if path and Path(path).exists():
                print(f"✅ Godot 찾음: {path}")
                return path
        
        # 와일드카드 패턴 검색
        import glob
        for pattern in wildcard_patterns:
            matches = glob.glob(pattern)
            if matches:
                # 가장 최신 버전 선택
                matches.sort(reverse=True)
                print(f"✅ Godot 찾음: {matches[0]}")
                return matches[0]
                
        # wsl 명령으로 Windows에서 찾기
        try:
            # Windows에서 where 명령 실행
            result = subprocess.run(
                ["cmd.exe", "/c", "where", "godot"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                godot_path = result.stdout.strip().split('\n')[0]
                # Windows 경로를 WSL 경로로 변환
                if godot_path.startswith("C:"):
                    wsl_path = "/mnt/c" + godot_path[2:].replace('\\', '/')
                    print(f"✅ Godot 찾음 (PATH): {wsl_path}")
                    return wsl_path
                elif godot_path.startswith("D:"):
                    wsl_path = "/mnt/d" + godot_path[2:].replace('\\', '/')
                    print(f"✅ Godot 찾음 (PATH): {wsl_path}")
                    return wsl_path
        except:
            pass
            
        return None
        
    async def create_dashboard_project(self):
        """대시보드용 Godot 프로젝트 생성"""
        self.godot_project_path.mkdir(exist_ok=True)
        
        # project.godot 파일 생성
        project_config = """
; Engine configuration file.
; It's best edited using the editor UI and not directly,
; since the properties are organized in sections here.

[application]

config/name="AutoCI Dashboard"
config/features=PackedStringArray("4.3", "GL Compatibility")
run/main_scene="res://Dashboard.tscn"
config/icon="res://icon.svg"

[display]

window/size/viewport_width=1280
window/size/viewport_height=720
window/size/resizable=false

[rendering]

renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"
"""
        
        (self.godot_project_path / "project.godot").write_text(project_config)
        
        # 메인 대시보드 씬 생성
        dashboard_scene = """[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://Dashboard.gd" id="1"]
[ext_resource type="Theme" path="res://DashboardTheme.tres" id="2"]

[node name="Dashboard" type="Control"]
anchor_right = 1.0
anchor_bottom = 1.0
theme = ExtResource("2")
script = ExtResource("1")

[node name="Background" type="ColorRect" parent="."]
anchor_right = 1.0
anchor_bottom = 1.0
color = Color(0.1, 0.1, 0.15, 1.0)

[node name="Title" type="Label" parent="."]
anchor_left = 0.5
anchor_right = 0.5
offset_left = -200.0
offset_top = 20.0
offset_right = 200.0
offset_bottom = 60.0
theme_override_font_sizes/font_size = 32
text = "AutoCI 실시간 대시보드"
horizontal_alignment = 1

[node name="MainPanel" type="Panel" parent="."]
anchor_left = 0.05
anchor_top = 0.15
anchor_right = 0.95
anchor_bottom = 0.95
modulate = Color(1, 1, 1, 0.95)

[node name="StatusContainer" type="VBoxContainer" parent="MainPanel"]
anchor_right = 1.0
anchor_bottom = 1.0
margin_left = 20.0
margin_top = 20.0
margin_right = -20.0
margin_bottom = -20.0

[node name="CurrentTask" type="RichTextLabel" parent="MainPanel/StatusContainer"]
custom_minimum_size = Vector2(0, 60)
bbcode_enabled = true
text = "[b]현재 작업:[/b] 초기화 중..."

[node name="ProgressBar" type="ProgressBar" parent="MainPanel/StatusContainer"]
custom_minimum_size = Vector2(0, 30)
value = 0.0

[node name="HSeparator" type="HSeparator" parent="MainPanel/StatusContainer"]
custom_minimum_size = Vector2(0, 20)

[node name="StatsGrid" type="GridContainer" parent="MainPanel/StatusContainer"]
columns = 2
custom_constants/h_separation = 50

[node name="TasksLabel" type="Label" parent="MainPanel/StatusContainer/StatsGrid"]
text = "완료된 작업:"

[node name="TasksValue" type="Label" parent="MainPanel/StatusContainer/StatsGrid"]
text = "0"

[node name="ErrorsLabel" type="Label" parent="MainPanel/StatusContainer/StatsGrid"]
text = "오류 수:"

[node name="ErrorsValue" type="Label" parent="MainPanel/StatusContainer/StatsGrid"]
text = "0"
modulate = Color(1, 0.3, 0.3, 1)

[node name="TimeLabel" type="Label" parent="MainPanel/StatusContainer/StatsGrid"]
text = "실행 시간:"

[node name="TimeValue" type="Label" parent="MainPanel/StatusContainer/StatsGrid"]
text = "00:00:00"

[node name="AIStatusLabel" type="Label" parent="MainPanel/StatusContainer/StatsGrid"]
text = "AI 상태:"

[node name="AIStatusValue" type="Label" parent="MainPanel/StatusContainer/StatsGrid"]
text = "준비 중"
modulate = Color(0.3, 1, 0.3, 1)

[node name="HSeparator2" type="HSeparator" parent="MainPanel/StatusContainer"]
custom_minimum_size = Vector2(0, 20)

[node name="LogLabel" type="Label" parent="MainPanel/StatusContainer"]
text = "실시간 로그:"

[node name="LogScroll" type="ScrollContainer" parent="MainPanel/StatusContainer"]
size_flags_vertical = 3

[node name="LogText" type="RichTextLabel" parent="MainPanel/StatusContainer/LogScroll"]
size_flags_horizontal = 3
size_flags_vertical = 3
bbcode_enabled = true
scroll_following = true

[node name="ErrorPanel" type="Panel" parent="."]
visible = false
anchor_left = 0.2
anchor_top = 0.3
anchor_right = 0.8
anchor_bottom = 0.7
modulate = Color(1, 0.8, 0.8, 0.95)

[node name="ErrorTitle" type="Label" parent="ErrorPanel"]
anchor_left = 0.5
anchor_right = 0.5
offset_left = -100.0
offset_top = 10.0
offset_right = 100.0
offset_bottom = 40.0
text = "⚠️ 오류 발생"
theme_override_colors/font_color = Color(1, 0, 0, 1)
theme_override_font_sizes/font_size = 24
horizontal_alignment = 1

[node name="ErrorText" type="RichTextLabel" parent="ErrorPanel"]
anchor_left = 0.05
anchor_top = 0.2
anchor_right = 0.95
anchor_bottom = 0.8
bbcode_enabled = true

[node name="CloseButton" type="Button" parent="ErrorPanel"]
anchor_left = 0.5
anchor_top = 0.85
anchor_right = 0.5
anchor_bottom = 0.95
offset_left = -50.0
offset_right = 50.0
text = "닫기"

[connection signal="pressed" from="ErrorPanel/CloseButton" to="." method="_on_error_close"]
"""
        
        (self.godot_project_path / "Dashboard.tscn").write_text(dashboard_scene)
        
        # 대시보드 스크립트 생성
        dashboard_script = """extends Control

var socket : StreamPeerTCP
var update_timer : Timer
var start_time : float
var log_lines : Array = []
const MAX_LOG_LINES = 100

func _ready():
	start_time = Time.get_ticks_msec() / 1000.0
	
	# 타이머 설정
	update_timer = Timer.new()
	update_timer.wait_time = 0.1
	update_timer.timeout.connect(_update_dashboard)
	add_child(update_timer)
	update_timer.start()
	
	# 소켓 연결
	_connect_to_autoci()
	
func _connect_to_autoci():
	socket = StreamPeerTCP.new()
	var error = socket.connect_to_host("127.0.0.1", 12345)
	if error != OK:
		add_log("[color=red]AutoCI 연결 실패[/color]")
	else:
		add_log("[color=green]AutoCI 연결 성공[/color]")

func _update_dashboard():
	# 실행 시간 업데이트
	var elapsed = Time.get_ticks_msec() / 1000.0 - start_time
	var hours = int(elapsed / 3600)
	var minutes = int((elapsed % 3600) / 60)
	var seconds = int(elapsed % 60)
	$MainPanel/StatusContainer/StatsGrid/TimeValue.text = "%02d:%02d:%02d" % [hours, minutes, seconds]
	
	# 소켓에서 데이터 읽기
	if socket and socket.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		var available = socket.get_available_bytes()
		if available > 0:
			var data = socket.get_string(available)
			_process_data(data)

func _process_data(data: String):
	# JSON 데이터 파싱
	var json = JSON.new()
	var parse_result = json.parse(data)
	if parse_result != OK:
		return
		
	var msg = json.data
	
	match msg.get("type", ""):
		"status":
			_update_status(msg)
		"log":
			add_log(msg.get("message", ""))
		"error":
			_show_error(msg.get("message", ""))
		"progress":
			$MainPanel/StatusContainer/ProgressBar.value = msg.get("value", 0)

func _update_status(status: Dictionary):
	$MainPanel/StatusContainer/CurrentTask.text = "[b]현재 작업:[/b] " + status.get("current_task", "")
	$MainPanel/StatusContainer/StatsGrid/TasksValue.text = str(status.get("tasks_completed", 0))
	$MainPanel/StatusContainer/StatsGrid/ErrorsValue.text = str(status.get("errors_count", 0))
	$MainPanel/StatusContainer/StatsGrid/AIStatusValue.text = status.get("ai_status", "")
	
	if status.has("progress"):
		$MainPanel/StatusContainer/ProgressBar.value = status.get("progress", 0)

func add_log(message: String):
	var timestamp = Time.get_time_string_from_system()
	log_lines.append("[color=gray]%s[/color] %s" % [timestamp, message])
	
	# 최대 라인 수 제한
	if log_lines.size() > MAX_LOG_LINES:
		log_lines.pop_front()
	
	# 로그 텍스트 업데이트
	$MainPanel/StatusContainer/LogScroll/LogText.text = "\\n".join(log_lines)

func _show_error(error_message: String):
	$ErrorPanel.visible = true
	$ErrorPanel/ErrorText.text = "[color=red]" + error_message + "[/color]"
	add_log("[color=red]오류: " + error_message + "[/color]")

func _on_error_close():
	$ErrorPanel.visible = false

func _exit_tree():
	if socket:
		socket.disconnect_from_host()
"""
        
        (self.godot_project_path / "Dashboard.gd").write_text(dashboard_script)
        
        # 테마 파일 생성
        theme_content = """[gd_resource type="Theme" format=3]

[resource]
default_font_size = 16
"""
        (self.godot_project_path / "DashboardTheme.tres").write_text(theme_content)
        
        # 아이콘 생성 (간단한 SVG)
        icon_svg = """<svg height="128" width="128" xmlns="http://www.w3.org/2000/svg">
<rect x="10" y="10" width="108" height="108" rx="20" fill="#4a90e2"/>
<text x="64" y="75" font-size="48" text-anchor="middle" fill="white">AI</text>
</svg>"""
        (self.godot_project_path / "icon.svg").write_text(icon_svg)
        
    async def start_dashboard(self, project_path: str = None):
        """Godot 대시보드 시작"""
        # Godot 실행 파일 찾기
        print("🔍 Godot 실행 파일을 찾는 중...")
        self.godot_executable = self.find_godot_executable()
        
        if not self.godot_executable:
            print("⚠️  Godot을 찾을 수 없습니다.")
            print("\n🛠️  AI 수정된 Godot을 사용하시는 경우:")
            print("1. Godot 소스를 빌드한 후 다음 경로 중 하나에 배치:")
            print(f"   - {self.get_windows_path(str(self.project_root.parent))}/godot-modified/bin/godot.windows.editor.x86_64.exe")
            print("   - D:\\godot-modified\\bin\\godot.windows.editor.x86_64.exe")
            print("   - C:\\godot-modified\\bin\\godot.windows.editor.x86_64.exe")
            print("\n💡 일반 Godot 설치 방법:")
            print("1. https://godotengine.org/download 에서 Windows 버전 다운로드")
            print("2. 다운로드한 파일을 다음 위치 중 하나에 저장:")
            print("   - C:\\Program Files\\Godot\\")
            print("   - D:\\Godot\\")
            print(f"   - {self.get_windows_path(str(self.project_root))}\\")
            print("3. 또는 PATH 환경변수에 Godot 경로 추가")
            
            # 수정된 Godot 사용 여부 확인
            use_modified = input("\n수정된 Godot을 사용하시나요? (y/N): ")
            if use_modified.lower() == 'y':
                custom_path = input("수정된 Godot 실행 파일의 전체 경로를 입력하세요 (취소: Enter): ").strip()
                if custom_path:
                    # Windows 경로를 WSL 경로로 변환
                    if custom_path.startswith("C:\\") or custom_path.startswith("c:\\"):
                        wsl_path = "/mnt/c" + custom_path[2:].replace('\\', '/')
                    elif custom_path.startswith("D:\\") or custom_path.startswith("d:\\"):
                        wsl_path = "/mnt/d" + custom_path[2:].replace('\\', '/')
                    else:
                        wsl_path = custom_path
                    
                    if Path(wsl_path).exists():
                        self.godot_executable = wsl_path
                        print(f"✅ 수정된 Godot 설정 완료: {wsl_path}")
                        # 설정 저장
                        self._save_godot_path(wsl_path)
                        return True
                    else:
                        print(f"❌ 파일을 찾을 수 없습니다: {wsl_path}")
                        return False
            else:
                # AI Godot 설정 도우미 실행
                setup_choice = input("\nAI Godot 설정 도우미를 실행하시겠습니까? (y/N): ")
                if setup_choice.lower() == 'y':
                    print("\n🔧 AI Godot 설정 도우미 실행...")
                    try:
                        result = subprocess.run(
                            ["python3", str(self.project_root / "setup_ai_godot.py")],
                            capture_output=False
                        )
                        if result.returncode == 0:
                            # 설정 완료 후 다시 찾기
                            self.godot_executable = self.find_godot_executable()
                            if self.godot_executable:
                                print("\n✅ AI Godot 설정 완료!")
                                return True
                    except Exception as e:
                        print(f"❌ 설정 도우미 실행 실패: {e}")
                
                print("\n💡 수동 설정 방법:")
                print("1. AI 수정된 Godot 빌드: python3 setup_custom_godot.py")
                print("2. 설정 도우미 실행: python3 setup_ai_godot.py")
                print("3. 다시 autoci 실행")
                print("\n터미널 모드로 계속 진행합니다.")
                return False
            
        # 프로젝트 경로 설정
        if project_path:
            # 기존 프로젝트 사용
            self.godot_project_path = Path(project_path)
            print(f"📁 기존 프로젝트 사용: {self.godot_project_path}")
            
            # 대시보드 스크립트 추가
            await self.inject_dashboard_to_project()
        else:
            # 대시보드 프로젝트 생성
            await self.create_dashboard_project()
        
        # 소켓 서버 시작
        self.start_socket_server()
        
        # Godot 실행 (WSL에서 Windows 프로그램 실행)
        try:
            # Windows 경로로 변환
            win_project_path = self.get_windows_path(str(self.godot_project_path))
            
            # Windows 실행 파일인지 확인
            if self.godot_executable.endswith('.exe'):
                # cmd.exe를 통해 Windows에서 Godot 실행
                subprocess.Popen([
                    "cmd.exe", "/c", "start", "", 
                    self.godot_executable.replace('/mnt/c/', 'C:\\').replace('/mnt/d/', 'D:\\').replace('/', '\\'),
                    "--path", win_project_path
                ])
                print("✅ Windows Godot이 시작되었습니다!")
            else:
                # Linux 실행 파일인 경우 직접 실행
                subprocess.Popen([
                    self.godot_executable,
                    "--path", str(self.godot_project_path)
                ])
                print("✅ Linux Godot이 시작되었습니다!")
            
            self.is_running = True
            print("📊 AutoCI 대시보드가 자동으로 표시됩니다.")
            
            # 연결 대기
            await asyncio.sleep(2)
            
            return True
            
        except Exception as e:
            print(f"❌ Godot 실행 실패: {e}")
            return False
            
    def start_socket_server(self):
        """대시보드와 통신할 소켓 서버 시작"""
        def server_thread():
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('127.0.0.1', self.dashboard_port))
            server_socket.listen(1)
            server_socket.settimeout(1.0)
            
            while self.is_running:
                try:
                    client, addr = server_socket.accept()
                    client.settimeout(0.1)
                    
                    # 클라이언트 처리
                    while self.is_running:
                        # 메시지 큐에서 데이터 가져오기
                        try:
                            msg = self.message_queue.get_nowait()
                            data = json.dumps(msg) + "\n"
                            client.send(data.encode())
                        except queue.Empty:
                            pass
                            
                        # 오류 큐 확인
                        try:
                            error = self.error_queue.get_nowait()
                            error_msg = {
                                "type": "error",
                                "message": error
                            }
                            data = json.dumps(error_msg) + "\n"
                            client.send(data.encode())
                        except queue.Empty:
                            pass
                            
                        # 주기적 상태 업데이트
                        status_msg = {
                            "type": "status",
                            **self.stats
                        }
                        data = json.dumps(status_msg) + "\n"
                        try:
                            client.send(data.encode())
                        except:
                            break
                            
                        time.sleep(0.1)
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.is_running:
                        print(f"소켓 서버 오류: {e}")
                        
            server_socket.close()
            
        thread = threading.Thread(target=server_thread, daemon=True)
        thread.start()
        
    def update_status(self, current_task: str, progress: float = None, ai_status: str = None):
        """상태 업데이트"""
        self.stats["current_task"] = current_task
        if progress is not None:
            self.stats["progress"] = progress
        if ai_status:
            self.stats["ai_status"] = ai_status
            
        # 메시지 큐에 추가
        self.message_queue.put({
            "type": "status",
            **self.stats
        })
        
    def add_log(self, message: str):
        """로그 추가"""
        self.message_queue.put({
            "type": "log",
            "message": message
        })
        
    def report_error(self, error: str):
        """오류 보고"""
        self.stats["errors_count"] += 1
        self.error_queue.put(error)
        
    def task_completed(self):
        """작업 완료 카운트"""
        self.stats["tasks_completed"] += 1
        
    async def inject_dashboard_to_project(self):
        """기존 프로젝트에 대시보드 추가"""
        # AutoCI 대시보드 신 추가
        dashboard_scene_path = self.godot_project_path / "AutoCIDashboard.tscn"
        dashboard_script_path = self.godot_project_path / "AutoCIDashboard.gd"
        
        if not dashboard_scene_path.exists():
            # 대시보드 신 생성
            dashboard_scene = """[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://AutoCIDashboard.gd" id="1"]

[node name="AutoCIDashboard" type="CanvasLayer"]
layer = 10
script = ExtResource("1")

[node name="DashboardUI" type="Control" parent="."]
anchor_right = 1.0
anchor_bottom = 1.0
mouse_filter = 2
"""
            dashboard_scene_path.write_text(dashboard_scene)
            
        if not dashboard_script_path.exists():
            # 대시보드 스크립트 생성
            dashboard_script = """extends CanvasLayer

var socket : StreamPeerTCP
var panel : Panel
var log_label : RichTextLabel
var stats_label : Label
var progress_bar : ProgressBar

func _ready():
	# UI 생성
	create_ui()
	
	# 소켓 연결
	socket = StreamPeerTCP.new()
	socket.connect_to_host("127.0.0.1", 12345)
	
	# 업데이트 타이머
	var timer = Timer.new()
	timer.wait_time = 0.1
	timer.timeout.connect(_update_dashboard)
	add_child(timer)
	timer.start()

func create_ui():
	# 메인 패널
	panel = Panel.new()
	panel.set_anchors_and_offsets_preset(Control.PRESET_TOP_RIGHT)
	panel.size = Vector2(400, 600)
	panel.position = Vector2(get_viewport().size.x - 420, 20)
	panel.modulate = Color(1, 1, 1, 0.9)
	$DashboardUI.add_child(panel)
	
	# 타이틀
	var title = Label.new()
	title.text = "AutoCI Dashboard"
	title.add_theme_font_size_override("font_size", 20)
	title.position = Vector2(150, 10)
	panel.add_child(title)
	
	# 통계 레이블
	stats_label = Label.new()
	stats_label.position = Vector2(10, 50)
	stats_label.size = Vector2(380, 100)
	stats_label.text = "AI 상태: 준비 중..."
	panel.add_child(stats_label)
	
	# 진행률 바
	progress_bar = ProgressBar.new()
	progress_bar.position = Vector2(10, 160)
	progress_bar.size = Vector2(380, 20)
	progress_bar.value = 0
	panel.add_child(progress_bar)
	
	# 로그 레이블
	log_label = RichTextLabel.new()
	log_label.position = Vector2(10, 200)
	log_label.size = Vector2(380, 380)
	log_label.bbcode_enabled = true
	log_label.scroll_following = true
	panel.add_child(log_label)
	
	# 최소화 버튼
	var minimize_btn = Button.new()
	minimize_btn.text = "_"
	minimize_btn.position = Vector2(360, 10)
	minimize_btn.size = Vector2(30, 30)
	minimize_btn.pressed.connect(_toggle_panel)
	panel.add_child(minimize_btn)

func _update_dashboard():
	if socket and socket.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		var available = socket.get_available_bytes()
		if available > 0:
			var data = socket.get_string(available)
			_process_data(data)

func _process_data(data: String):
	var json = JSON.new()
	var parse_result = json.parse(data)
	if parse_result != OK:
		return
		
	var msg = json.data
	
	match msg.get("type", ""):
		"status":
			_update_status(msg)
		"log":
			add_log(msg.get("message", ""))
		"progress":
			progress_bar.value = msg.get("value", 0)

func _update_status(status: Dictionary):
	var status_text = "AI 상태: " + status.get("ai_status", "")
	status_text += "\n현재 작업: " + status.get("current_task", "")
	status_text += "\n완료된 작업: " + str(status.get("tasks_completed", 0))
	status_text += "\n오류 수: " + str(status.get("errors_count", 0))
	stats_label.text = status_text
	
	if status.has("progress"):
		progress_bar.value = status.get("progress", 0)

func add_log(message: String):
	var timestamp = Time.get_time_string_from_system()
	log_label.append_text("[color=gray]%s[/color] %s\n" % [timestamp, message])

func _toggle_panel():
	panel.visible = !panel.visible

func _exit_tree():
	if socket:
		socket.disconnect_from_host()
"""
            dashboard_script_path.write_text(dashboard_script)
            
        # project.godot에 autoload 추가
        project_file = self.godot_project_path / "project.godot"
        if project_file.exists():
            content = project_file.read_text()
            
            # autoload 섹션 찾기
            if "[autoload]" not in content:
                content += "\n\n[autoload]\n"
                
            # AutoCIDashboard autoload 추가
            if "AutoCIDashboard=" not in content:
                autoload_section = content.find("[autoload]")
                if autoload_section != -1:
                    # [autoload] 다음 줄에 추가
                    next_line = content.find("\n", autoload_section) + 1
                    content = content[:next_line] + 'AutoCIDashboard="*res://AutoCIDashboard.tscn"\n' + content[next_line:]
                    project_file.write_text(content)
                    print("✅ AutoCI 대시보드가 프로젝트에 추가되었습니다.")
    
    def get_windows_path(self, wsl_path: str) -> str:
        """경로 변환 함수"""
        if wsl_path.startswith("/mnt/"):
            drive = wsl_path[5]
            rest = wsl_path[7:]
            return f"{drive.upper()}:\\{rest.replace('/', '\\')}"
        return wsl_path.replace('/', '\\')
    
    async def download_godot(self) -> bool:
        """Godot 자동 다운로드"""
        print("\n📥 Godot 4.3 안정 버전을 다운로드합니다...")
        
        # 다운로드 URL (Godot 4.3 Windows 64비트)
        godot_url = "https://github.com/godotengine/godot/releases/download/4.3-stable/Godot_v4.3-stable_win64.exe.zip"
        
        # 다운로드할 디렉토리 생성
        godot_bin_dir = self.project_root / "godot_bin"
        godot_bin_dir.mkdir(exist_ok=True)
        
        zip_path = godot_bin_dir / "godot.zip"
        exe_path = godot_bin_dir / "Godot_v4.3-stable_win64.exe"
        
        try:
            # 이미 다운로드되어 있는지 확인
            if exe_path.exists():
                print("✅ Godot이 이미 다운로드되어 있습니다.")
                return True
            
            # 다운로드 진행률 표시 함수
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                progress_bar = "█" * int(percent // 2) + "▒" * (50 - int(percent // 2))
                print(f"\r진행률: [{progress_bar}] {percent:.1f}%", end="")
            
            # 다운로드
            print(f"다운로드 중: {godot_url}")
            urllib.request.urlretrieve(godot_url, zip_path, reporthook=download_progress)
            print("\n✅ 다운로드 완료!")
            
            # ZIP 압축 해제
            print("📦 압축 해제 중...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(godot_bin_dir)
            
            # 압축 파일 삭제
            zip_path.unlink()
            
            # 실행 권한 설정 (WSL에서는 필요 없지만 일관성을 위해)
            if exe_path.exists():
                print("✅ Godot 설치 완료!")
                return True
            else:
                print("❌ 압축 해제 후 실행 파일을 찾을 수 없습니다.")
                return False
                
        except Exception as e:
            print(f"\n❌ 다운로드 중 오류 발생: {e}")
            
            # 대체 다운로드 방법 안내
            print("\n💡 수동 다운로드 방법:")
            print("1. 브라우저에서 다음 URL 접속:")
            print("   https://godotengine.org/download/windows/")
            print("2. 'Godot Engine - Windows' 다운로드")
            print(f"3. 다운로드한 파일을 다음 위치에 저장:")
            print(f"   {self.get_windows_path(str(godot_bin_dir))}")
            
            return False
    
    def _save_godot_path(self, path: str):
        """Godot 경로 설정 저장"""
        config_file = self.project_root / ".godot_config.json"
        config = {"godot_path": path}
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"설정 저장 실패: {e}")
    
    def _load_godot_path(self) -> Optional[str]:
        """저장된 Godot 경로 불러오기"""
        config_file = self.project_root / ".godot_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return config.get("godot_path")
            except Exception:
                pass
        return None
    
    def stop(self):
        """대시보드 중지"""
        self.is_running = False
        print("Godot 대시보드를 종료합니다...")

# 전역 대시보드 인스턴스
_dashboard_instance = None

def get_dashboard() -> GodotRealtimeDashboard:
    """대시보드 싱글톤 인스턴스 가져오기"""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = GodotRealtimeDashboard()
    return _dashboard_instance

async def test_dashboard():
    """대시보드 테스트"""
    dashboard = get_dashboard()
    
    if await dashboard.start_dashboard():
        # 테스트 메시지들
        await asyncio.sleep(2)
        
        dashboard.update_status("AutoCI 시스템 초기화 중...", 10, "시작 중")
        dashboard.add_log("AutoCI 24시간 개발 시스템이 시작되었습니다.")
        
        await asyncio.sleep(1)
        
        dashboard.update_status("AI 모델 로딩 중...", 30, "모델 로딩")
        dashboard.add_log("AI 모델을 불러오는 중입니다...")
        
        await asyncio.sleep(1)
        
        dashboard.update_status("게임 프로젝트 생성 중...", 50, "활성")
        dashboard.add_log("새로운 게임 프로젝트를 생성합니다.")
        dashboard.task_completed()
        
        await asyncio.sleep(1)
        
        dashboard.report_error("테스트 오류입니다. 실제 오류가 아닙니다.")
        
        await asyncio.sleep(1)
        
        dashboard.update_status("코드 생성 중...", 80, "코드 생성")
        dashboard.add_log("AI가 게임 코드를 작성하고 있습니다...")
        
        await asyncio.sleep(10)
        
        dashboard.stop()

if __name__ == "__main__":
    asyncio.run(test_dashboard())