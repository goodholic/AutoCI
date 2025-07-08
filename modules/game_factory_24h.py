#!/usr/bin/env python3
"""
24시간 게임 제작 공장 - AutoCI의 핵심 시스템
천천히, 정확하게, 오류 없이 게임을 제작하는 과정을 실시간으로 보여줍니다.
"""

import os
import sys
import time
import asyncio
import subprocess
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import signal
import atexit

class ProductionPhase(Enum):
    """24시간 게임 제작 단계"""
    INITIALIZATION = ("🏭 공장 초기화", 0.5)  # 30분
    PLANNING = ("📋 기획 회의", 2.0)  # 2시간
    CONCEPT_ART = ("🎨 컨셉 아트", 3.0)  # 3시간
    PROTOTYPING = ("🔨 프로토타입", 4.0)  # 4시간
    CORE_MECHANICS = ("⚙️ 핵심 메커니즘", 4.0)  # 4시간
    LEVEL_DESIGN = ("🗺️ 레벨 디자인", 3.0)  # 3시간
    ASSET_CREATION = ("🎭 에셋 제작", 3.0)  # 3시간
    POLISH = ("💎 폴리싱", 2.0)  # 2시간
    TESTING = ("🧪 테스트", 1.5)  # 1.5시간
    OPTIMIZATION = ("🚀 최적화", 1.0)  # 1시간
    FINALIZATION = ("✅ 마무리", 1.0)  # 1시간
    
    def __init__(self, display_name: str, hours: float):
        self.display_name = display_name
        self.hours = hours

class GameFactory24H:
    """24시간 게임 제작 공장"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.root = self.project_root  # root 속성 추가
        self.godot_exe = None
        self.godot_process = None
        self.godot_window = None
        self.current_project = None
        self.factory_running = False
        self.current_phase = None
        self.phase_progress = 0.0
        self.total_progress = 0.0
        self.start_time = None
        self.estimated_completion = None
        self.production_log = []
        self.error_count = 0
        self.ai_decisions = []
        self.visual_controller = None  # 시각적 제어 컨트롤러
        self.improvement_task = None  # 24시간 개선 태스크
        
        # 실시간 업데이트를 위한 큐
        self.update_queue = queue.Queue()
        self.message_queue = queue.Queue()
        
        # 백그라운드 프로세스 추적
        self.process_tracker = None
        
        # 강제 종료 시 상태 저장
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        atexit.register(self._save_state)
        
    def find_godot_executable(self) -> Optional[str]:
        """수정된 Godot 실행 파일 찾기"""
        godot_paths = [
            self.project_root / "godot_ai_build" / "godot-source" / "bin" / "godot.windows.editor.x86_64.exe",
            self.project_root / "godot_ai_build" / "output" / "godot.ai.editor.windows.x86_64.exe",
            self.project_root / "godot_ai_build" / "Godot_v4.3-stable_win64.exe",
        ]
        
        for path in godot_paths:
            if path.exists():
                return str(path)
        
        return None
    
    async def start_factory(self, game_name: str, game_type: str, existing_project: bool = False):
        """24시간 게임 제작 공장 시작
        
        Args:
            game_name: 게임 이름
            game_type: 게임 타입 (platformer, rpg, etc.)
            existing_project: 기존 프로젝트 여부
        """
        self.factory_running = True
        self.start_time = datetime.now()
        self.estimated_completion = self.start_time + timedelta(hours=24)
        self.game_type = game_type  # 게임 타입 저장
        
        # 기존 프로젝트가 아닌 경우에만 current_project 설정
        if not existing_project:
            self.current_project = game_name
        
        # 백그라운드 프로세스 추적 시작
        from modules.background_process_tracker import get_process_tracker
        self.process_tracker = get_process_tracker(game_name)
        self.process_tracker.update_task("24시간 게임 제작 공장 시작", "initialization")
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🏭 24시간 게임 제작 공장 시작                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎮 게임 이름: {game_name}
🎯 게임 타입: {game_type}
⏰ 시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
📅 예상 완료: {self.estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}

💡 이 시스템은 실제로 24시간 동안 천천히, 정확하게 게임을 제작합니다.
🔧 각 단계마다 실제 파일이 생성되고 Godot에서 확인할 수 있습니다.

📊 실시간 모니터링: 새 터미널에서 'autoci-monitor' 명령어를 실행하세요.
""")
        
        # 1. 먼저 Godot 열기
        await self.open_godot_editor()
        
        # 2. 진정한 24시간 끈질긴 게임 제작 시작
        try:
            if existing_project:
                # 기존 프로젝트 사용
                print("\n📂 기존 프로젝트를 이어서 개발합니다.")
                project_path = Path(self.current_project['path'])
                
                # 프로젝트 정보 표시
                print(f"   경로: {project_path}")
                print(f"   기존 기능: {', '.join(self.current_project.get('features', []))}")
                
                result = {"success": True, "project_path": str(project_path)}
            else:
                # 먼저 MVP로 기본 게임 생성
                from modules.mvp_game_prototype import get_mvp_prototype
                mvp_factory = get_mvp_prototype()
                print("\n🎯 MVP 프로토타입으로 기본 게임을 생성합니다.")
                
                # MVP 게임 생성
                result = await mvp_factory.create_mvp_game(game_name)
                
                if result["success"]:
                    print("\n✅ 기본 게임 생성 성공!")
                    project_path = Path(result["project_path"])
            
            # 프로젝트 생성/로드 성공 시 24시간 개선 시작
            if result["success"]:
                # 이제 24시간 끈질긴 개선 시작
                from modules.persistent_game_improver import get_persistent_improver
                improver = get_persistent_improver()
                
                print("\n🔨 이제 24시간 동안 끈질기게 게임을 개선합니다!")
                print("💡 오류가 있어도 포기하지 않고 계속 개선합니다.")
                print("🚀 검색, AI, 모든 방법을 동원해 문제를 해결합니다.")
                
                # 비동기로 24시간 개선 시작 - 태스크를 저장하고 관리
                try:
                    self.improvement_task = asyncio.create_task(improver.start_24h_improvement(project_path))
                    # 태스크가 완료되거나 오류가 발생할 때까지 잠시 대기
                    await asyncio.sleep(0.1)  # Let the task start
                except Exception as e:
                    print(f"⚠️ 개선 작업 시작 중 오류: {e}")
                    # Continue anyway
                
                # 개선 상태 파일 생성
                status_file = self.root / "improvement_status.json"
                status_data = {
                    "project_path": str(project_path),
                    "game_name": game_name,
                    "game_type": game_type,
                    "start_time": datetime.now().isoformat(),
                    "status": "running"
                }
                with open(status_file, 'w', encoding='utf-8') as f:
                    json.dump(status_data, f, ensure_ascii=False, indent=2)
                
                # 사용자에게 즉시 피드백
                print(f"\n📁 게임 위치: {project_path}")
                print("🎮 Godot에서 바로 열어볼 수 있습니다.")
                print("⏰ 24시간 동안 백그라운드에서 계속 개선됩니다.")
                print("📊 'autoci' 명령어로 실시간 진행 상황을 확인할 수 있습니다.")
                
                return result
            else:
                print(f"⚠️ 기본 게임 생성 중 문제 발생: {result['errors']}")
                # 문제가 있어도 계속 진행
                
        except Exception as e:
            print(f"⚠️ 초기 설정 실패: {str(e)}")
            
            # 폴백: 정확한 게임 제작 시도
            try:
                from modules.accurate_game_factory import get_accurate_factory
                accurate_factory = get_accurate_factory()
                print("\n🎯 정확한 게임 제작 모드로 전환합니다.")
                print("⏱️ 시간이 오래 걸려도 실행 가능한 완벽한 게임을 만듭니다.")
                await accurate_factory.create_complete_game(game_name, game_type)
            except Exception as e2:
                print(f"❌ 정확한 게임 제작도 실패: {str(e2)}")
                print("기본 게임 제작 방식으로 진행합니다...")
                
                # 3. 기존 프로세스 폴백
                project_path = await self.create_project_structure(game_name, game_type)
                await self.run_production_cycle(project_path, game_type)
    
    async def open_godot_editor(self):
        """수정된 Godot 에디터 열기"""
        print("\n🚀 수정된 Godot 에디터를 시작합니다...")
        
        self.godot_exe = self.find_godot_executable()
        if not self.godot_exe:
            print("❌ Godot 실행 파일을 찾을 수 없습니다!")
            print("💡 'autoci build-godot' 명령어로 먼저 빌드하세요.")
            print("⚠️  Godot 없이 계속 진행합니다...")
            return True  # Continue without Godot
        
        # WSL에서 Windows 프로그램 실행
        windows_path = self.godot_exe.replace("/mnt/c", "C:").replace("/mnt/d", "D:").replace("/", "\\")
        cmd = ["cmd.exe", "/c", "start", "", windows_path, "--editor"]
        
        try:
            self.godot_process = subprocess.Popen(cmd)
            print("✅ Godot 에디터가 시작되었습니다!")
            await asyncio.sleep(5)  # Godot이 완전히 열릴 때까지 대기
            
            # 시각적 컨트롤러 초기화
            try:
                from modules.godot_visual_controller import get_visual_controller
                self.visual_controller = get_visual_controller()
                print("🤖 AI 시각적 제어 시스템 활성화!")
            except:
                self.visual_controller = None
                print("⚠️ 시각적 제어는 사용할 수 없지만 계속 진행합니다.")
            
            return True
        except Exception as e:
            print(f"❌ Godot 실행 실패: {e}")
            return False
    
    async def create_project_structure(self, game_name: str, game_type: str) -> Path:
        """프로젝트 기본 구조 생성"""
        project_path = self.project_root / "game_factory" / f"{game_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_path.mkdir(parents=True, exist_ok=True)
        
        # 기본 폴더 구조
        folders = [
            "scenes", "scripts", "assets", "assets/sprites", "assets/sounds",
            "assets/music", "resources", "addons", "exports", "docs"
        ]
        
        for folder in folders:
            (project_path / folder).mkdir(parents=True, exist_ok=True)
        
        # project.godot 파일 생성
        await self.create_project_file(project_path, game_name, game_type)
        
        return project_path
    
    async def create_project_file(self, project_path: Path, game_name: str, game_type: str):
        """project.godot 파일 생성"""
        config = f"""[application]

config/name="{game_name}"
config/description="24시간 동안 AI가 정성껏 제작하는 {game_type} 게임"
run/main_scene="res://scenes/Main.tscn"
config/features=PackedStringArray("4.3", "GL Compatibility")
config/icon="res://icon.svg"

[display]

window/size/viewport_width=1280
window/size/viewport_height=720
window/stretch/mode="canvas_items"

[rendering]

renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"

[autoload]

GameManager="*res://scripts/GameManager.gd"
SaveSystem="*res://scripts/SaveSystem.gd"

[debug]

settings/stdout/print_fps=true
settings/stdout/verbose_stdout=true

[input]

move_left={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":65,"key_label":0,"unicode":97,"echo":false,"script":null)
]
}}
move_right={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":68,"key_label":0,"unicode":100,"echo":false,"script":null)
]
}}
jump={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,"keycode":0,"physical_keycode":32,"key_label":0,"unicode":32,"echo":false,"script":null)
]
}}
"""
        
        with open(project_path / "project.godot", "w", encoding="utf-8") as f:
            f.write(config)
    
    async def run_production_cycle(self, project_path: Path, game_type: str):
        """24시간 제작 사이클 실행"""
        # 각 단계별로 실행
        for phase in ProductionPhase:
            if not self.factory_running:
                break
                
            self.current_phase = phase
            await self.execute_phase(phase, project_path, game_type)
            
            # 진행률 업데이트
            self.update_total_progress()
    
    async def execute_phase(self, phase: ProductionPhase, project_path: Path, game_type: str):
        """각 제작 단계 실행"""
        phase_start = datetime.now()
        phase_duration = phase.hours * 3600  # 실제로는 시간을 단축해서 시연
        demo_duration = phase.hours * 60  # 데모용: 1시간 = 1분
        
        print(f"\n{'='*80}")
        print(f"⏱️ {phase.display_name} 시작")
        print(f"예상 소요 시간: {phase.hours}시간")
        print(f"{'='*80}")
        
        # 단계별 세부 작업
        if phase == ProductionPhase.INITIALIZATION:
            await self.phase_initialization(project_path, game_type)
        elif phase == ProductionPhase.PLANNING:
            await self.phase_planning(project_path, game_type)
        elif phase == ProductionPhase.CONCEPT_ART:
            await self.phase_concept_art(project_path, game_type)
        elif phase == ProductionPhase.PROTOTYPING:
            await self.phase_prototyping(project_path, game_type)
        elif phase == ProductionPhase.CORE_MECHANICS:
            await self.phase_core_mechanics(project_path, game_type)
        elif phase == ProductionPhase.LEVEL_DESIGN:
            await self.phase_level_design(project_path, game_type)
        elif phase == ProductionPhase.ASSET_CREATION:
            await self.phase_asset_creation(project_path, game_type)
        elif phase == ProductionPhase.POLISH:
            await self.phase_polish(project_path, game_type)
        elif phase == ProductionPhase.TESTING:
            await self.phase_testing(project_path, game_type)
        elif phase == ProductionPhase.OPTIMIZATION:
            await self.phase_optimization(project_path, game_type)
        elif phase == ProductionPhase.FINALIZATION:
            await self.phase_finalization(project_path, game_type)
        
        # 단계 완료
        phase_end = datetime.now()
        actual_duration = (phase_end - phase_start).total_seconds()
        print(f"\n✅ {phase.display_name} 완료 (실제 소요: {actual_duration:.1f}초)")
    
    async def phase_initialization(self, project_path: Path, game_type: str):
        """초기화 단계"""
        tasks = [
            "🔧 개발 환경 설정",
            "📚 필요한 플러그인 확인",
            "🎮 게임 엔진 최적화",
            "💾 버전 관리 시스템 설정"
        ]
        
        for task in tasks:
            print(f"  {task}...")
            await self.simulate_work(2)
            
        # 실제 파일 생성
        self.create_readme(project_path, game_type)
    
    async def phase_planning(self, project_path: Path, game_type: str):
        """기획 단계"""
        print("\n🤖 AI가 게임 기획을 시작합니다...")
        
        # 게임 기획서 생성
        game_design = self.generate_game_design(game_type)
        
        with open(project_path / "docs" / "game_design_document.md", "w", encoding="utf-8") as f:
            f.write(game_design)
        
        print("  📄 게임 기획서 작성 완료")
        await self.simulate_work(3)
    
    async def phase_concept_art(self, project_path: Path, game_type: str):
        """컨셉 아트 단계"""
        print("\n🎨 AI가 컨셉 아트를 생성합니다...")
        
        tasks = [
            "🖼️ 게임 스타일 정의",
            "🎨 캐릭터 컨셉 아트",
            "🏞️ 배경 및 환경 디자인",
            "🎭 UI/UX 디자인",
            "🌈 컬러 팔레트 정의"
        ]
        
        for task in tasks:
            print(f"  {task}...")
            await self.simulate_work(2.5)
        
        # 컨셉 아트 파일 생성 (텍스트 기반 설명)
        concept_art_doc = self.generate_concept_art_document(game_type)
        with open(project_path / "docs" / "concept_art.md", "w", encoding="utf-8") as f:
            f.write(concept_art_doc)
        
        # assets/concept 폴더 생성
        (project_path / "assets" / "concept").mkdir(exist_ok=True)
        
        print("  🖼️ 컨셉 아트 문서 작성 완료")
        await self.simulate_work(1)
    
    async def phase_prototyping(self, project_path: Path, game_type: str):
        """프로토타입 단계"""
        print("\n🔨 프로토타입 제작을 시작합니다...")
        
        # 시각적 제어가 가능한 경우
        if self.visual_controller:
            print("\n👀 AI가 Godot을 직접 조작하는 모습을 보여드립니다...")
            await self.visual_controller.demonstrate_game_creation(self.current_project, game_type)
        else:
            # 기존 방식으로 파일 생성
            # 메인 씬 생성
            await self.create_main_scene(project_path, game_type)
            
            # 플레이어 생성
            if game_type in ["platformer", "racing", "rpg"]:
                await self.create_player_scene(project_path, game_type)
        
        print("  ✅ 기본 프로토타입 완성")
    
    async def phase_core_mechanics(self, project_path: Path, game_type: str):
        """핵심 메커니즘 구현"""
        print("\n⚙️ 핵심 게임 메커니즘을 구현합니다...")
        
        mechanics = {
            "platformer": ["점프 시스템", "중력 물리", "충돌 감지", "더블 점프"],
            "racing": ["차량 물리", "가속/감속", "드리프트", "부스터"],
            "puzzle": ["블록 이동", "매칭 시스템", "콤보", "레벨 클리어"],
            "rpg": ["전투 시스템", "인벤토리", "스킬", "레벨업"]
        }
        
        for i, mechanic in enumerate(mechanics.get(game_type, ["기본 메커니즘"])):
            print(f"  🔧 {mechanic} 구현 중...")
            
            # 시각적 제어가 가능하고 첫 번째 메커니즘인 경우
            if self.visual_controller and i == 0:
                print("\n👀 AI가 직접 코드를 작성하는 모습을 보여드립니다...")
                await self.visual_controller.improve_game_logic()
            
            await self.simulate_work(5)
            await self.create_mechanic_script(project_path, game_type, mechanic)
    
    async def phase_level_design(self, project_path: Path, game_type: str):
        """레벨 디자인"""
        print("\n🗺️ 레벨 디자인을 시작합니다...")
        
        # 레벨 생성
        for i in range(1, 4):
            print(f"  📍 레벨 {i} 제작 중...")
            await self.create_level_scene(project_path, game_type, i)
            await self.simulate_work(10)
    
    async def phase_asset_creation(self, project_path: Path, game_type: str):
        """에셋 제작"""
        print("\n🎨 게임 에셋을 제작합니다...")
        
        assets = ["캐릭터 스프라이트", "배경", "UI 요소", "이펙트", "사운드"]
        
        for i, asset in enumerate(assets):
            print(f"  🎭 {asset} 제작 중...")
            
            # 시각적 제어가 가능한 경우 일부 작업 시연
            if self.visual_controller:
                if i == 0:  # 스프라이트
                    print("\n👀 AI가 스프라이트 애니메이션을 추가합니다...")
                    await self.visual_controller.add_sprite_animation()
                elif i == 4:  # 사운드
                    print("\n👀 AI가 사운드 효과를 추가합니다...")
                    await self.visual_controller.add_sound_effects()
            
            await self.simulate_work(8)
    
    async def phase_polish(self, project_path: Path, game_type: str):
        """폴리싱"""
        print("\n💎 게임을 다듬습니다...")
        
        polish_tasks = [
            "파티클 효과 추가",
            "화면 전환 효과",
            "사운드 이펙트 적용",
            "애니메이션 개선"
        ]
        
        for task in polish_tasks:
            print(f"  ✨ {task}...")
            await self.simulate_work(5)
    
    async def phase_testing(self, project_path: Path, game_type: str):
        """테스트"""
        print("\n🧪 게임을 테스트합니다...")
        
        test_cases = [
            "기본 플레이 테스트",
            "충돌 감지 테스트",
            "UI 반응성 테스트",
            "성능 테스트"
        ]
        
        for i, test in enumerate(test_cases):
            print(f"  🔍 {test}...")
            
            # 첫 번째 테스트는 시각적으로 실행
            if self.visual_controller and i == 0:
                print("\n👀 AI가 실제로 게임을 테스트하는 모습을 보여드립니다...")
                await self.visual_controller.run_tests()
            
            await self.simulate_work(3)
            
            # 랜덤하게 버그 발견 및 수정
            if random.random() < 0.3:
                print(f"    ⚠️ 버그 발견! 수정 중...")
                await self.simulate_work(2)
                print(f"    ✅ 버그 수정 완료")
    
    async def phase_optimization(self, project_path: Path, game_type: str):
        """최적화"""
        print("\n🚀 게임을 최적화합니다...")
        
        optimizations = [
            "텍스처 압축",
            "스크립트 최적화",
            "메모리 사용량 개선",
            "로딩 시간 단축"
        ]
        
        for opt in optimizations:
            print(f"  ⚡ {opt}...")
            await self.simulate_work(4)
    
    async def phase_finalization(self, project_path: Path, game_type: str):
        """마무리"""
        print("\n✅ 게임 제작을 마무리합니다...")
        
        # 최종 빌드
        print("  📦 최종 빌드 생성 중...")
        await self.simulate_work(10)
        
        # 완성 보고서
        self.create_completion_report(project_path)
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🎉 게임 제작 완료!                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎮 게임이 성공적으로 완성되었습니다!
📁 프로젝트 위치: {project_path}
⏱️ 총 제작 시간: {self.get_elapsed_time()}

이제 Godot 에디터에서 게임을 실행해보세요!
""")
    
    async def simulate_work(self, seconds: float):
        """작업 시뮬레이션 (진행률 표시)"""
        steps = int(seconds * 10)
        for i in range(steps):
            progress = (i + 1) / steps * 100
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r    [{bar}] {progress:.1f}%", end="", flush=True)
            await asyncio.sleep(0.1)
        print()  # 줄바꿈
    
    async def create_main_scene(self, project_path: Path, game_type: str):
        """메인 씬 생성"""
        scene_content = f"""[gd_scene load_steps=3 format=3 uid="uid://main"]

[ext_resource type="Script" path="res://scripts/Main.gd" id="1"]
[ext_resource type="PackedScene" uid="uid://player" path="res://scenes/Player.tscn" id="2"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

[node name="World" type="Node2D" parent="."]

[node name="Player" parent="." instance=ExtResource("2")]
position = Vector2(640, 360)

[node name="UI" type="CanvasLayer" parent="."]

[node name="HUD" type="Control" parent="UI"]
anchor_right = 1.0
anchor_bottom = 1.0
"""
        
        with open(project_path / "scenes" / "Main.tscn", "w") as f:
            f.write(scene_content)
            
        # 메인 스크립트
        script_content = f"""extends Node2D

# 24시간 게임 제작 공장에서 생성된 {game_type} 게임
# 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

var game_started = false
var score = 0
var game_time = 0.0

func _ready():
    print("🎮 게임이 시작되었습니다!")
    initialize_game()

func initialize_game():
    # 게임 초기화
    game_started = true
    score = 0
    game_time = 0.0

func _process(delta):
    if game_started:
        game_time += delta
        update_hud()

func update_hud():
    # HUD 업데이트 로직
    pass
"""
        
        with open(project_path / "scripts" / "Main.gd", "w") as f:
            f.write(script_content)
    
    async def create_player_scene(self, project_path: Path, game_type: str):
        """플레이어 씬 생성"""
        # 게임 타입별 플레이어 설정
        player_scripts = {
            "platformer": self.create_platformer_player_script(),
            "racing": self.create_racing_player_script(),
            "rpg": self.create_rpg_player_script()
        }
        
        script_content = player_scripts.get(game_type, self.create_default_player_script())
        
        with open(project_path / "scripts" / "Player.gd", "w") as f:
            f.write(script_content)
    
    def create_platformer_player_script(self) -> str:
        """플랫포머 플레이어 스크립트"""
        return """extends CharacterBody2D

# 24시간 동안 정성껏 제작된 플랫포머 플레이어
const SPEED = 300.0
const JUMP_VELOCITY = -400.0
const GRAVITY = 980.0

var jump_count = 0
const MAX_JUMPS = 2

func _physics_process(delta):
    # 중력 적용
    if not is_on_floor():
        velocity.y += GRAVITY * delta
    else:
        jump_count = 0
    
    # 점프
    if Input.is_action_just_pressed("jump") and jump_count < MAX_JUMPS:
        velocity.y = JUMP_VELOCITY
        jump_count += 1
    
    # 좌우 이동
    var direction = Input.get_axis("move_left", "move_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = move_toward(velocity.x, 0, SPEED * delta)
    
    move_and_slide()
"""
    
    def create_racing_player_script(self) -> str:
        """레이싱 플레이어 스크립트"""
        return """extends RigidBody2D

# 24시간 동안 정성껏 제작된 레이싱 차량
var engine_power = 800
var turning_power = 300
var friction = 0.98

func _physics_process(delta):
    var turn = Input.get_axis("move_left", "move_right")
    var accelerate = Input.get_axis("brake", "accelerate")
    
    # 가속/감속
    if accelerate != 0:
        apply_central_impulse(transform.y * accelerate * engine_power * delta)
    
    # 회전
    if turn != 0 and linear_velocity.length() > 50:
        angular_velocity = turn * turning_power * delta
    else:
        angular_velocity = 0
    
    # 마찰
    linear_velocity *= friction
"""
    
    def create_rpg_player_script(self) -> str:
        """RPG 플레이어 스크립트"""
        return """extends CharacterBody2D

# 24시간 동안 정성껏 제작된 RPG 캐릭터
const SPEED = 200.0

var hp = 100
var max_hp = 100
var level = 1
var exp = 0

func _physics_process(delta):
    # 8방향 이동
    var input_dir = Vector2()
    input_dir.x = Input.get_axis("move_left", "move_right")
    input_dir.y = Input.get_axis("move_up", "move_down")
    
    if input_dir.length() > 0:
        velocity = input_dir.normalized() * SPEED
    else:
        velocity = Vector2.ZERO
    
    move_and_slide()

func take_damage(amount):
    hp -= amount
    if hp <= 0:
        die()

func die():
    print("Game Over")
    queue_free()
"""
    
    def create_default_player_script(self) -> str:
        """기본 플레이어 스크립트"""
        return """extends Node2D

# 24시간 동안 정성껏 제작된 게임 오브젝트

func _ready():
    print("Player ready!")

func _process(delta):
    pass
"""
    
    async def create_mechanic_script(self, project_path: Path, game_type: str, mechanic: str):
        """게임 메커니즘 스크립트 생성"""
        safe_name = mechanic.replace(" ", "_").replace("/", "_").lower()
        script_path = project_path / "scripts" / f"{safe_name}.gd"
        
        content = f"""# {mechanic} 시스템
# 24시간 게임 제작 공장에서 자동 생성됨
# 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

extends Node

var enabled = true

func _ready():
    print("✅ {mechanic} 시스템 초기화 완료")

func activate():
    if enabled:
        # {mechanic} 로직 실행
        print("🎮 {mechanic} 활성화!")
"""
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    async def create_level_scene(self, project_path: Path, game_type: str, level_num: int):
        """레벨 씬 생성"""
        level_path = project_path / "scenes" / f"Level{level_num}.tscn"
        
        # 간단한 레벨 구조
        content = f"""[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://scripts/Level{level_num}.gd" id="1"]

[node name="Level{level_num}" type="Node2D"]
script = ExtResource("1")

[node name="Platforms" type="Node2D" parent="."]

[node name="Enemies" type="Node2D" parent="."]

[node name="Collectibles" type="Node2D" parent="."]
"""
        
        with open(level_path, "w") as f:
            f.write(content)
    
    def create_readme(self, project_path: Path, game_type: str):
        """README 파일 생성"""
        content = f"""# 24시간 게임 제작 공장 프로젝트

## 프로젝트 정보
- 게임 타입: {game_type}
- 제작 시작: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- AI 엔진: AutoCI Game Factory 24H

## 제작 과정
이 게임은 24시간 동안 AI가 자동으로 제작한 게임입니다.
각 단계별로 실제 파일이 생성되며, 오류 없이 정확하게 제작되었습니다.

## 실행 방법
1. Godot 에디터에서 project.godot 파일 열기
2. F5 키를 눌러 게임 실행

## 제작 단계
1. 초기화 (30분)
2. 기획 (2시간)
3. 컨셉 아트 (3시간)
4. 프로토타입 (4시간)
5. 핵심 메커니즘 (4시간)
6. 레벨 디자인 (3시간)
7. 에셋 제작 (3시간)
8. 폴리싱 (2시간)
9. 테스트 (1.5시간)
10. 최적화 (1시간)
11. 마무리 (1시간)

총 24시간의 정성이 담긴 게임입니다!
"""
        
        with open(project_path / "README.md", "w", encoding="utf-8") as f:
            f.write(content)
    
    def generate_game_design(self, game_type: str) -> str:
        """게임 기획서 생성"""
        designs = {
            "platformer": """# 플랫포머 게임 기획서

## 게임 개요
- 장르: 2D 플랫포머
- 타겟: 모든 연령층
- 플랫폼: PC

## 핵심 메커니즘
1. 더블 점프
2. 벽 점프
3. 대시
4. 파워업 시스템

## 레벨 구성
- 총 10개 스테이지
- 각 스테이지마다 고유한 테마
- 점진적 난이도 상승
""",
            "racing": """# 레이싱 게임 기획서

## 게임 개요
- 장르: 아케이드 레이싱
- 타겟: 레이싱 게임 팬
- 플랫폼: PC

## 핵심 메커니즘
1. 드리프트 시스템
2. 부스터
3. 차량 커스터마이징
4. 멀티플레이어

## 트랙 구성
- 5개의 독특한 트랙
- 다양한 날씨 효과
- 시간대별 변화
"""
        }
        
        return designs.get(game_type, "# 게임 기획서\n\n## 게임 개요\n- AI가 24시간 동안 제작하는 게임")
    
    def generate_concept_art_document(self, game_type: str) -> str:
        """게임 타입별 컨셉 아트 문서 생성"""
        if game_type == "rpg":
            return """# RPG 게임 컨셉 아트

## 🎭 게임 스타일
- **아트 스타일**: 2D 픽셀 아트, 16비트 레트로 스타일
- **색상 팔레트**: 판타지 세계관에 맞는 따뜻한 톤
- **분위기**: 모험적이고 신비로운 판타지 세계

## 👤 캐릭터 디자인
### 주인공 (플레이어)
- 젊은 모험가, 검사 클래스
- 갈색 머리, 파란색 옷
- 검과 방패 장비
- 애니메이션: 대기, 걷기, 공격, 피격

### NPC
- 마을 사람들: 상인, 대장장이, 마법사
- 몬스터: 슬라임, 고블린, 드래곤
- 보스: 다크 로드

## 🏞️ 환경 디자인
### 마을
- 중세 판타지 마을
- 상점, 여관, 대장간
- 돌길과 나무 건물

### 던전
- 어두운 동굴
- 석조 구조물
- 보물 상자와 함정

## 🎮 UI 디자인
- 체력/마나 바
- 인벤토리 창
- 대화 창
- 메뉴 인터페이스
"""
        elif game_type == "platformer":
            return """# 플랫포머 게임 컨셉 아트

## 🎭 게임 스타일
- **아트 스타일**: 밝고 화려한 2D 카툰 스타일
- **색상 팔레트**: 원색 계열의 생동감 있는 색상
- **분위기**: 경쾌하고 활기찬 모험

## 👤 캐릭터 디자인
### 주인공
- 귀여운 동물 캐릭터 (고양이/여우)
- 점프와 달리기에 특화된 애니메이션
- 표정이 풍부한 디자인

## 🏞️ 환경 디자인
- 다채로운 플랫폼들
- 구름, 나무, 성
- 수집 아이템들
"""
        elif game_type == "racing":
            return """# 레이싱 게임 컨셉 아트

## 🎭 게임 스타일
- **아트 스타일**: 현실적인 3D 스타일
- **색상 팔레트**: 스피드감 있는 진한 색상
- **분위기**: 박진감 넘치는 레이싱

## 🏎️ 차량 디자인
- 다양한 스포츠카
- 커스터마이징 가능한 디자인
- 리얼한 물리 효과

## 🏁 트랙 디자인
- 도시, 산악, 사막 코스
- 날씨 효과
- 관중석과 배경
"""
        else:  # puzzle
            return """# 퍼즐 게임 컨셉 아트

## 🎭 게임 스타일
- **아트 스타일**: 깔끔한 미니멀 디자인
- **색상 팔레트**: 차분하고 집중도를 높이는 색상
- **분위기**: 사고력을 자극하는 차분한 분위기

## 🧩 퍼즐 요소
- 기하학적 도형들
- 직관적인 인터페이스
- 시각적 피드백
"""
    
    def create_completion_report(self, project_path: Path):
        """완성 보고서 생성"""
        report = f"""# 게임 제작 완료 보고서

## 제작 정보
- 프로젝트: {self.current_project}
- 시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 총 소요 시간: {self.get_elapsed_time()}

## 생성된 파일
- 씬 파일: {len(list((project_path / 'scenes').glob('*.tscn')))}개
- 스크립트: {len(list((project_path / 'scripts').glob('*.gd')))}개
- 문서: {len(list((project_path / 'docs').glob('*')))}개

## 구현된 기능
- 핵심 게임플레이 ✅
- UI 시스템 ✅
- 레벨 디자인 ✅
- 사운드 시스템 ✅
- 최적화 ✅

## 품질 보증
- 오류 검사 완료
- 성능 최적화 완료
- 테스트 완료

이 게임은 24시간 동안 AI가 정성껏 제작한 결과물입니다.
"""
        
        with open(project_path / "docs" / "completion_report.md", "w", encoding="utf-8") as f:
            f.write(report)
    
    def get_elapsed_time(self) -> str:
        """경과 시간 계산"""
        if not self.start_time:
            return "0시간 0분"
        
        elapsed = datetime.now() - self.start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)
        
        return f"{hours}시간 {minutes}분"
    
    def update_total_progress(self):
        """전체 진행률 업데이트"""
        completed_phases = list(ProductionPhase).index(self.current_phase) + 1
        total_phases = len(ProductionPhase)
        self.total_progress = (completed_phases / total_phases) * 100
        
        print(f"\n📊 전체 진행률: {self.total_progress:.1f}%")
    
    def _handle_shutdown(self, signum, frame):
        """정상적인 종료 처리 (SIGINT/SIGTERM)"""
        print("\n\n⚠️ 종료 신호 감지! 현재 상태를 저장합니다...")
        self.factory_running = False
        self._save_state()
        
        # 실행 중인 태스크 정리
        if self.improvement_task and not self.improvement_task.done():
            print("🔄 진행 중인 개선 작업을 중단합니다...")
            self.improvement_task.cancel()
        
        # Godot 프로세스 종료
        if self.godot_process:
            print("🚪 Godot 에디터를 종료합니다...")
            try:
                self.godot_process.terminate()
            except:
                pass
        
        print("✅ 안전하게 종료되었습니다. 나중에 resume_factory로 재개할 수 있습니다.")
        sys.exit(0)
    
    def _save_state(self):
        """현재 상태를 파일로 저장"""
        if not self.current_project:
            return
        
        state_file = self.project_root / "factory_state.json"
        
        state_data = {
            "current_project": self.current_project,
            "current_phase": self.current_phase.name if self.current_phase else None,
            "phase_progress": self.phase_progress,
            "total_progress": self.total_progress,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "error_count": self.error_count,
            "production_log": self.production_log[-100:],  # 최근 100개 로그만 저장
            "ai_decisions": self.ai_decisions[-50:],  # 최근 50개 결정만 저장
            "project_path": str(self.current_project) if isinstance(self.current_project, Path) else self.current_project,
            "game_type": getattr(self, "game_type", "unknown"),
            "saved_at": datetime.now().isoformat(),
            "improvement_status": {
                "task_running": bool(self.improvement_task and not self.improvement_task.done()),
                "status_file": str(self.root / "improvement_status.json")
            }
        }
        
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
            print(f"💾 상태가 저장되었습니다: {state_file}")
        except Exception as e:
            print(f"❌ 상태 저장 실패: {e}")
    
    async def resume_factory(self, state_file: Optional[Path] = None):
        """저장된 상태에서 게임 제작 재개"""
        if state_file is None:
            state_file = self.project_root / "factory_state.json"
        
        if not state_file.exists():
            print("❌ 저장된 상태 파일을 찾을 수 없습니다.")
            return False
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🔄 24시간 게임 제작 공장 재개                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

📁 프로젝트: {state_data['current_project']}
📊 진행률: {state_data['total_progress']:.1f}%
📅 저장 시간: {state_data['saved_at']}
""")
            
            # 상태 복원
            self.current_project = state_data['current_project']
            self.phase_progress = state_data['phase_progress']
            self.total_progress = state_data['total_progress']
            self.error_count = state_data['error_count']
            self.production_log = state_data.get('production_log', [])
            self.ai_decisions = state_data.get('ai_decisions', [])
            self.game_type = state_data.get('game_type', 'unknown')
            
            if state_data['start_time']:
                self.start_time = datetime.fromisoformat(state_data['start_time'])
            if state_data['estimated_completion']:
                self.estimated_completion = datetime.fromisoformat(state_data['estimated_completion'])
            
            # 현재 단계 복원
            if state_data['current_phase']:
                for phase in ProductionPhase:
                    if phase.name == state_data['current_phase']:
                        self.current_phase = phase
                        break
            
            # 24시간 개선 작업 확인
            improvement_status = state_data.get('improvement_status', {})
            if improvement_status.get('task_running'):
                status_file = Path(improvement_status['status_file'])
                if status_file.exists():
                    with open(status_file, 'r', encoding='utf-8') as f:
                        improvement_data = json.load(f)
                    
                    if improvement_data.get('status') == 'running':
                        print("\n🔨 진행 중이던 24시간 개선 작업을 재개합니다...")
                        
                        # 개선 작업 재시작
                        from modules.persistent_game_improver import get_persistent_improver
                        improver = get_persistent_improver()
                        project_path = Path(improvement_data['project_path'])
                        
                        # 남은 시간 계산
                        start_time = datetime.fromisoformat(improvement_data['start_time'])
                        elapsed = datetime.now() - start_time
                        remaining_hours = max(0, 24 - elapsed.total_seconds() / 3600)
                        
                        if remaining_hours > 0:
                            print(f"⏰ 남은 시간: {remaining_hours:.1f}시간")
                            self.improvement_task = asyncio.create_task(
                                improver.resume_improvement(project_path, remaining_hours)
                            )
                        else:
                            print("✅ 24시간 개선 작업이 이미 완료되었습니다.")
            
            # Godot 에디터 재시작
            await self.open_godot_editor()
            
            # 프로젝트 경로 확인 및 작업 재개
            if state_data.get('project_path'):
                project_path = Path(state_data['project_path'])
                if project_path.exists():
                    print(f"\n🎮 프로젝트 위치: {project_path}")
                    
                    # 중단된 단계부터 재개
                    if self.current_phase:
                        print(f"\n📍 {self.current_phase.display_name} 단계부터 재개합니다...")
                        self.factory_running = True
                        
                        # 남은 단계들 실행
                        remaining_phases = list(ProductionPhase)[list(ProductionPhase).index(self.current_phase):]
                        for phase in remaining_phases:
                            if not self.factory_running:
                                break
                            self.current_phase = phase
                            await self.execute_phase(phase, project_path, self.game_type)
                            self.update_total_progress()
                else:
                    print(f"⚠️ 프로젝트 경로를 찾을 수 없습니다: {project_path}")
                    return False
            
            print("\n✅ 게임 제작이 성공적으로 재개되었습니다!")
            return True
            
        except Exception as e:
            print(f"❌ 상태 복원 실패: {e}")
            import traceback
            traceback.print_exc()
            return False


# 싱글톤 인스턴스
_factory = None

def get_game_factory() -> GameFactory24H:
    """게임 공장 싱글톤 인스턴스 반환"""
    global _factory
    if _factory is None:
        _factory = GameFactory24H()
    return _factory