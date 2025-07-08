#!/usr/bin/env python3
"""
AutoCI Create 자동화 통합 모듈
godot_automation_system과 연동하여 게임 개발 자동화
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.godot_automation_system import GodotAutomationSystem
from core_system.continuous_learning_system import ContinuousLearningSystem

logger = logging.getLogger(__name__)

class AutoCICreateAutomation:
    """AutoCI Create 명령어를 위한 자동화 시스템"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.automation = GodotAutomationSystem(project_path)
        
        # 작업 큐
        self.task_queue = []
        self.completed_tasks = []
        
        # 학습 시스템 연동
        self.learning_system = None
        try:
            self.learning_system = ContinuousLearningSystem()
            logger.info("🤖 학습 시스템 연동 완료")
        except:
            logger.warning("⚠️ 학습 시스템을 사용할 수 없습니다")
            
    def analyze_project_requirements(self) -> Dict[str, Any]:
        """프로젝트 요구사항 분석"""
        requirements = {
            "assets_needed": [],
            "scenes_to_create": [],
            "scripts_to_write": [],
            "ui_elements": []
        }
        
        # project.godot 파일 분석
        project_file = self.project_path / "project.godot"
        if project_file.exists():
            # 간단한 분석 (실제로는 더 정교한 파싱 필요)
            content = project_file.read_text(encoding='utf-8')
            
            if "2D" in content:
                requirements["assets_needed"].extend([
                    {"type": "sprite", "description": "2D 캐릭터 스프라이트"},
                    {"type": "texture", "description": "배경 텍스처"},
                    {"type": "ui", "description": "UI 요소 (버튼, 패널 등)"}
                ])
                requirements["scenes_to_create"].extend([
                    "Player.tscn",
                    "MainMenu.tscn",
                    "GameLevel.tscn"
                ])
                
            if "3D" in content:
                requirements["assets_needed"].extend([
                    {"type": "model", "description": "3D 캐릭터 모델"},
                    {"type": "environment", "description": "환경 모델"},
                    {"type": "texture", "description": "3D 텍스처"}
                ])
                
        return requirements
        
    def create_asset_request_ui(self):
        """자원 요청 UI 생성 (웹 인터페이스)"""
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>AutoCI 자원 요청</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .asset-request {
            border: 2px dashed #3498db;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            background: #f8f9fa;
        }
        .upload-area {
            border: 3px dashed #2ecc71;
            padding: 40px;
            text-align: center;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            background: #e8f5e9;
        }
        .status {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .pending { background: #fff3cd; }
        .completed { background: #d4edda; }
        .processing { background: #cce5ff; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; }
        button {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #2980b9;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 AutoCI 게임 자원 요청</h1>
        <p>AutoCI가 게임 개발을 위해 다음 자원들을 요청합니다:</p>
        
        <div id="requests"></div>
        
        <div class="upload-area" id="dropZone">
            <h3>📁 파일을 여기에 드래그하거나 클릭하여 업로드</h3>
            <input type="file" id="fileInput" multiple style="display: none;">
        </div>
        
        <div id="status"></div>
        
        <button onclick="checkProgress()">🔄 진행 상황 확인</button>
        <button onclick="startAutomation()">🚀 자동화 시작</button>
    </div>
    
    <script>
        // 파일 업로드 처리
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.background = '#e8f5e9';
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.style.background = '';
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            handleFiles(e.dataTransfer.files);
        });
        
        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });
        
        function handleFiles(files) {
            // 파일 처리 로직
            console.log('업로드된 파일:', files);
            updateStatus('파일 업로드 중...', 'processing');
        }
        
        function updateStatus(message, type) {
            const status = document.getElementById('status');
            status.innerHTML = `<div class="status ${type}">${message}</div>`;
        }
        
        function checkProgress() {
            // 진행 상황 확인
            fetch('/api/progress')
                .then(response => response.json())
                .then(data => {
                    updateStatus(`완료: ${data.completed}/${data.total}`, 'processing');
                });
        }
        
        function startAutomation() {
            if (confirm('자동화를 시작하시겠습니까?')) {
                updateStatus('자동화 진행 중...', 'processing');
                fetch('/api/start', { method: 'POST' });
            }
        }
        
        // 요청 목록 로드
        fetch('/api/requests')
            .then(response => response.json())
            .then(data => {
                const container = document.getElementById('requests');
                data.forEach(req => {
                    container.innerHTML += `
                        <div class="asset-request">
                            <h3>${req.type}</h3>
                            <p>${req.description}</p>
                            <small>폴더: ${req.folder}</small>
                        </div>
                    `;
                });
            });
    </script>
</body>
</html>"""
        
        # HTML 파일 저장
        ui_path = self.project_path / "autoci_asset_requests.html"
        ui_path.write_text(html_content, encoding='utf-8')
        logger.info(f"📋 자원 요청 UI 생성: {ui_path}")
        
        return str(ui_path)
        
    async def automated_game_creation(self, game_type: str = "2d_platformer"):
        """자동화된 게임 생성 워크플로우"""
        logger.info(f"🎮 {game_type} 게임 자동 생성 시작")
        
        # 1. 프로젝트 구조 생성
        await self._create_project_structure(game_type)
        
        # 2. 필요한 자원 요청
        await self._request_required_assets(game_type)
        
        # 3. 씬 자동 생성
        await self._create_scenes(game_type)
        
        # 4. 스크립트 생성
        await self._generate_scripts(game_type)
        
        # 5. 테스트 및 최적화
        await self._test_and_optimize()
        
        logger.info("✅ 게임 생성 완료!")
        
    async def _create_project_structure(self, game_type: str):
        """프로젝트 구조 생성"""
        structures = {
            "2d_platformer": [
                "scenes/player",
                "scenes/enemies",
                "scenes/levels",
                "scenes/ui",
                "scripts/player",
                "scripts/enemies",
                "scripts/game",
                "assets/sprites",
                "assets/sounds",
                "assets/music"
            ],
            "3d_fps": [
                "scenes/player",
                "scenes/weapons",
                "scenes/enemies",
                "scenes/levels",
                "scripts/player",
                "scripts/weapons",
                "scripts/ai",
                "assets/models",
                "assets/textures",
                "assets/sounds"
            ]
        }
        
        for dir_path in structures.get(game_type, []):
            full_path = self.project_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
    async def _request_required_assets(self, game_type: str):
        """필요한 자원 요청"""
        asset_requests = {
            "2d_platformer": [
                ("sprite", "Player character sprite with idle, walk, jump animations"),
                ("sprite", "Enemy sprites (at least 3 different types)"),
                ("texture", "Tileset for platforms and environment"),
                ("ui", "UI elements - health bar, score, buttons"),
                ("sfx", "Jump, coin collect, hit, game over sounds"),
                ("music", "Background music - menu and gameplay")
            ],
            "3d_fps": [
                ("model", "Player hands and weapon models"),
                ("model", "Enemy character models"),
                ("environment", "Level environment models"),
                ("texture", "Textures for all models"),
                ("sfx", "Weapon sounds, footsteps, impacts"),
                ("music", "Ambient and action music")
            ]
        }
        
        for asset_type, description in asset_requests.get(game_type, []):
            folder = self.automation.request_user_asset(asset_type, description)
            logger.info(f"📦 자원 요청: {description} -> {folder}")
            
    async def _create_scenes(self, game_type: str):
        """씬 자동 생성"""
        if game_type == "2d_platformer":
            # Player 씬 생성
            self.automation.automated_workflow("Create player scene with sprite and collision")
            await asyncio.sleep(2)
            
            # Level 씬 생성
            self.automation.automated_workflow("Create level with tilemap and platforms")
            await asyncio.sleep(2)
            
            # UI 씬 생성
            self.automation.automated_workflow("Create UI with health bar and score")
            
    async def _generate_scripts(self, game_type: str):
        """스크립트 자동 생성"""
        scripts = {
            "2d_platformer": {
                "Player.gd": """extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

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
""",
                "Enemy.gd": """extends CharacterBody2D

@export var speed = 100.0
@export var detection_range = 200.0

var player = null
var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

func _ready():
    player = get_tree().get_first_node_in_group("player")

func _physics_process(delta):
    if not is_on_floor():
        velocity.y += gravity * delta
    
    if player and global_position.distance_to(player.global_position) < detection_range:
        var direction = (player.global_position - global_position).normalized()
        velocity.x = direction.x * speed
    else:
        velocity.x = 0
    
    move_and_slide()
"""
            }
        }
        
        for filename, content in scripts.get(game_type, {}).items():
            script_path = self.project_path / "scripts" / filename
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(content, encoding='utf-8')
            logger.info(f"📝 스크립트 생성: {filename}")
            
    async def _test_and_optimize(self):
        """테스트 및 최적화"""
        # 씬 실행
        self.automation.automated_workflow("Play current scene for testing")
        await asyncio.sleep(5)
        
        # 성능 분석
        state = self.automation.capture_state()
        
        # 학습 기반 최적화
        if self.learning_system:
            # 최적화 제안 생성
            optimization_prompt = """
            Based on the current Godot project state, suggest optimizations for:
            1. Performance improvements
            2. Code structure
            3. Asset organization
            """
            
            # 학습 시스템 활용 (실제 구현 필요)
            logger.info("🔧 학습 기반 최적화 진행 중...")
            
    def generate_progress_report(self) -> Dict[str, Any]:
        """진행 상황 보고서 생성"""
        report = {
            "project_path": str(self.project_path),
            "automation_success_rate": self.automation.success_count / max(1, self.automation.total_attempts),
            "completed_tasks": len(self.completed_tasks),
            "pending_tasks": len(self.task_queue),
            "learning_history": self.automation.learning_history[-10:],  # 최근 10개
            "requested_assets": self._get_asset_request_status(),
            "timestamp": datetime.now().isoformat()
        }
        
        # 보고서 저장
        report_path = self.project_path / "autoci_progress_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        return report
        
    def _get_asset_request_status(self) -> Dict[str, int]:
        """자원 요청 상태 확인"""
        request_file = self.automation.user_assets_dir / "asset_requests.json"
        if request_file.exists():
            with open(request_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    "total": len(data.get("requests", [])),
                    "completed": len(data.get("completed", [])),
                    "pending": len(data.get("pending", []))
                }
        return {"total": 0, "completed": 0, "pending": 0}

# CLI 통합을 위한 메인 함수
async def main(project_path: str, game_type: str = "2d_platformer"):
    """AutoCI Create 자동화 실행"""
    automation = AutoCICreateAutomation(project_path)
    
    # 자원 요청 UI 생성
    ui_path = automation.create_asset_request_ui()
    print(f"\n📋 자원 요청 UI를 확인하세요: {ui_path}")
    print("필요한 자원을 해당 폴더에 넣어주세요.\n")
    
    # 자동화 실행
    await automation.automated_game_creation(game_type)
    
    # 진행 보고서
    report = automation.generate_progress_report()
    print(f"\n📊 진행 보고서:")
    print(f"- 성공률: {report['automation_success_rate']:.1%}")
    print(f"- 완료된 작업: {report['completed_tasks']}")
    print(f"- 대기 중인 작업: {report['pending_tasks']}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AutoCI Create 자동화")
    parser.add_argument("project_path", help="Godot 프로젝트 경로")
    parser.add_argument("--type", default="2d_platformer", help="게임 타입")
    
    args = parser.parse_args()
    asyncio.run(main(args.project_path, args.type))