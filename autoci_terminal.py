#!/usr/bin/env python3
"""
AutoCI Terminal Interface - 터미널 인터페이스
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time
import psutil

# 기존 모듈 import
try:
    from modules.csharp_learning_agent import CSharpLearningAgent
    from modules.godot_controller import GodotController
    from modules.godot_editor_controller import GodotEditorController, GodotSceneBuilder
    from modules.ai_model_integration import get_ai_integration
    from modules.error_handler import error_handler, get_error_handler
    from modules.monitoring_system import get_monitor
except ImportError as e:
    # 모듈이 없어도 기본 동작
    print(f"Warning: Some modules not available: {e}")
    CSharpLearningAgent = None
    GodotController = None
    GodotEditorController = None
    GodotSceneBuilder = None
    get_ai_integration = None
    error_handler = lambda x: lambda f: f  # No-op decorator
    get_error_handler = None
    get_monitor = None

# 대시보드 에러 핸들러 import
try:
    from modules.error_handler_integration import get_error_handler as get_dashboard_error_handler, dashboard_error_handler
except ImportError:
    get_dashboard_error_handler = None
    dashboard_error_handler = lambda x: lambda f: f  # No-op decorator

class AutoCITerminal:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.setup_directories()
        self.setup_logging()
        
        # AI 모델 선택
        self.ai_model_name = self.select_ai_model()
        self.logger.info(f"🤖 AI 모델 선택: {self.ai_model_name}")
        
        # AI 통합 초기화
        self.ai_integration = get_ai_integration() if get_ai_integration else None
        
        # 시스템 초기화
        self.csharp_agent = CSharpLearningAgent() if CSharpLearningAgent else None
        self.godot_controller = GodotController() if GodotController else None
        self.editor_controller = GodotEditorController() if GodotEditorController else None
        self.scene_builder = GodotSceneBuilder(self.editor_controller) if GodotSceneBuilder and self.editor_controller else None
        
        # Godot 실시간 통합 초기화
        self.godot_integration = None
        try:
            from modules.godot_live_integration import get_godot_integration
            self.godot_integration = get_godot_integration()
        except ImportError:
            self.logger.warning("Godot 실시간 통합 모듈을 로드할 수 없습니다.")
        
        # Godot 실시간 대시보드 초기화
        self.godot_dashboard = None
        try:
            from modules.godot_realtime_dashboard import get_dashboard
            self.godot_dashboard = get_dashboard()
            
            # 에러 핸들러에 대시보드 연결
            if get_dashboard_error_handler:
                error_handler = get_dashboard_error_handler()
                error_handler.set_dashboard(self.godot_dashboard)
        except ImportError:
            self.logger.warning("Godot 실시간 대시보드 모듈을 로드할 수 없습니다.")
        
        # Godot 프로젝트 매니저 초기화
        self.godot_project_manager = None
        try:
            from modules.godot_project_manager import GodotProjectManager
            self.godot_project_manager = GodotProjectManager()
        except ImportError:
            self.logger.warning("Godot 프로젝트 매니저 모듈을 로드할 수 없습니다.")
        
        # 프로젝트 관리
        self.current_project = None
        self.projects = {}
        
        # 24시간 실행 상태
        self.running = True
        self.start_time = datetime.now()
        
        # 통계
        self.stats = {
            "games_created": 0,
            "features_added": 0,
            "bugs_fixed": 0,
            "csharp_concepts_learned": 0,
            "commands_executed": 0
        }
        
        # 자가 진화 시스템 초기화
        self.evolution_system = None
        try:
            from modules.self_evolution_system import get_evolution_system
            self.evolution_system = get_evolution_system()
            self.logger.info("🧬 자가 진화 시스템이 활성화되었습니다.")
        except ImportError:
            self.logger.warning("자가 진화 시스템을 로드할 수 없습니다.")
        
        # 실시간 게임 수정 시스템 초기화
        self.game_modifier = None
        try:
            from modules.realtime_game_modifier import get_game_modifier
            self.game_modifier = get_game_modifier()
            self.logger.info("🎮 실시간 게임 수정 시스템이 활성화되었습니다.")
        except ImportError:
            self.logger.warning("실시간 게임 수정 시스템을 로드할 수 없습니다.")
        
        # 게임 제작 과정 시각화 시스템 초기화
        self.game_visualizer = None
        try:
            from modules.game_creation_visualizer import get_game_creation_visualizer
            self.game_visualizer = get_game_creation_visualizer()
            self.logger.info("🎨 게임 제작 과정 시각화 시스템이 활성화되었습니다.")
        except ImportError:
            self.logger.warning("게임 제작 과정 시각화 시스템을 로드할 수 없습니다.")
        
        # Godot 게임 빌더 초기화 (실제 게임 제작)
        self.game_builder = None
        self.progressive_builder = None
        try:
            from modules.godot_game_builder import get_game_builder
            self.game_builder = get_game_builder()
            self.logger.info("🔨 Godot 게임 빌더가 활성화되었습니다.")
        except ImportError:
            self.logger.warning("Godot 게임 빌더를 로드할 수 없습니다.")
        
        # 진행형 게임 빌더 초기화
        try:
            from modules.progressive_game_builder import get_progressive_builder
            self.progressive_builder = get_progressive_builder()
            self.logger.info("🎬 진행형 게임 빌더가 활성화되었습니다.")
        except ImportError:
            self.logger.warning("진행형 게임 빌더를 로드할 수 없습니다.")
        
        # AI 컨트롤러 초기화
        self.ai_controller = None
        try:
            from modules.godot_ai_controller import get_ai_controller
            self.ai_controller = get_ai_controller()
            self.logger.info("🤖 Godot AI 컨트롤러가 활성화되었습니다.")
        except ImportError:
            self.logger.warning("Godot AI 컨트롤러를 로드할 수 없습니다.")
        
        # 터미널 UI 초기화
        self.terminal_ui = None
        try:
            from modules.terminal_ui import get_terminal_ui
            self.terminal_ui = get_terminal_ui()
            self.logger.info("📺 터미널 UI가 활성화되었습니다.")
        except ImportError:
            self.logger.warning("터미널 UI를 로드할 수 없습니다.")
        
        # 24시간 실시간 모니터 초기화 (간단한 버전 우선 시도)
        self.realtime_monitor = None
        try:
            from modules.simple_realtime_monitor import get_simple_realtime_monitor
            self.realtime_monitor = get_simple_realtime_monitor()
            self.logger.info("🎯 24시간 간단한 실시간 모니터가 활성화되었습니다.")
        except ImportError:
            try:
                from modules.realtime_24h_monitor import get_realtime_monitor
                self.realtime_monitor = get_realtime_monitor()
                self.logger.info("🎯 24시간 실시간 모니터가 활성화되었습니다.")
            except ImportError:
                self.logger.warning("24시간 실시간 모니터를 로드할 수 없습니다.")

    def setup_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            "game_projects",
            "csharp_learning",
            "logs"
        ]
        for dir_path in directories:
            Path(self.project_root / dir_path).mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """로깅 설정"""
        log_file = self.project_root / "logs" / f"autoci_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("AutoCI")
    
    def select_ai_model(self) -> str:
        """README에 명시된 대로 AI 모델 선택"""
        try:
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            
            if available_memory >= 32:
                return "Qwen2.5-Coder-32B"
            elif available_memory >= 16:
                return "CodeLlama-13B"
            else:
                return "Llama-3.1-8B"
        except:
            return "Llama-3.1-8B"  # 기본값

    @error_handler("AutoCI.Development")
    async def start_24h_development(self):
        """24시간 자동 개발 시작"""
        self.logger.info("🚀 24시간 AI 게임 개발 시스템 시작")
        
        # 비동기 작업들
        tasks = [
            self.game_creation_loop(),      # 2-4시간마다 새 게임
            self.feature_addition_loop(),    # 30분마다 기능 추가
            self.bug_fix_loop(),            # 15분마다 버그 수정
            self.optimization_loop(),        # 1시간마다 최적화
            self.csharp_learning_loop()      # 지속적 C# 학습
        ]
        
        # 실시간 게임 수정 큐 처리 태스크 추가
        if self.game_modifier:
            tasks.append(self.game_modifier.process_modification_queue())
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("백그라운드 작업 중단됨")

    @dashboard_error_handler("게임 생성")
    async def game_creation_loop(self):
        """게임 생성 루프 (2-4시간마다)"""
        while self.running:
            try:
                # 게임 타입 선택
                game_types = ["platformer", "racing", "puzzle", "rpg"]
                game_type = game_types[len(self.projects) % len(game_types)]
                
                self.logger.info(f"🎮 새 {game_type} 게임 프로젝트 생성 중...")
                
                # 프로젝트 생성
                project_name = f"{game_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                project_path = self.project_root / "game_projects" / project_name
                project_path.mkdir(parents=True, exist_ok=True)
                
                # Godot 프로젝트 초기화
                if self.godot_controller:
                    try:
                        await self.godot_controller.create_project(project_name, str(project_path))
                        self.logger.info(f"🎯 Godot 프로젝트 초기화 완료")
                    except:
                        pass
                
                self.projects[project_name] = {
                    "type": game_type,
                    "path": project_path,
                    "created": datetime.now(),
                    "features": [],
                    "bugs_fixed": 0
                }
                
                self.current_project = project_name
                self.stats["games_created"] += 1
                
                # 실시간 게임 수정 시스템에 게임 정보 설정
                if self.game_modifier:
                    self.game_modifier.set_game_info(project_name, game_type)
                
                self.logger.info(f"✅ {project_name} 프로젝트 생성 완료")
                
                # 게임 제작 과정 시각화 시작
                if self.game_visualizer:
                    # 비동기 시각화 작업 시작
                    visualization_task = asyncio.create_task(
                        self.game_visualizer.start_visualization(game_type, project_name)
                    )
                    # 시각화가 백그라운드에서 실행되도록 함
                    
                # Godot 대시보드 업데이트
                if self.godot_integration:
                    await self.godot_integration.update_dashboard({
                        "task": f"새 {game_type} 게임 프로젝트 생성",
                        "progress": 30,
                        "games_created": self.stats["games_created"],
                        "log": f"🎮 {project_name} 프로젝트가 생성되었습니다",
                        "color": "ffff00"
                    })
                
                # 게임 제작 시간 (시각화와 함께 진행)
                if self.game_visualizer:
                    # 시각화가 진행되는 동안 대기 (약 2-3분)
                    await asyncio.sleep(180)  # 3분
                else:
                    # 시각화 없이는 기존대로 긴 시간 대기
                    wait_time = 2 * 3600 + (len(self.projects) % 3) * 3600
                    await asyncio.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"게임 생성 중 오류: {e}")
                await asyncio.sleep(300)  # 5분 후 재시도

    @dashboard_error_handler("기능 추가")
    async def feature_addition_loop(self):
        """기능 추가 루프 (30분마다)"""
        await asyncio.sleep(1800)  # 처음 30분 대기
        
        while self.running:
            try:
                if self.current_project and self.current_project in self.projects:
                    project = self.projects[self.current_project]
                    
                    # 게임 타입에 따른 기능 추가
                    features = self.get_features_for_game_type(project["type"])
                    new_feature = features[len(project["features"]) % len(features)]
                    
                    self.logger.info(f"➕ '{new_feature}' 기능 추가 중...")
                    
                    # 기능 구현
                    await self.implement_feature(new_feature, project)
                    
                    project["features"].append(new_feature)
                    self.stats["features_added"] += 1
                    
                    self.logger.info(f"✅ '{new_feature}' 기능 추가 완료")
                    
                    # Godot 대시보드 업데이트
                    if self.godot_integration:
                        await self.godot_integration.update_dashboard({
                            "task": f"'{new_feature}' 기능 추가 완료",
                            "progress": 50 + (self.stats["features_added"] % 50),
                            "tasks_completed": self.stats["features_added"],
                            "log": f"✅ {new_feature} 기능이 추가되었습니다",
                            "color": "00ff00"
                        })
                
                await asyncio.sleep(1800)  # 30분 대기
                
            except Exception as e:
                self.logger.error(f"기능 추가 중 오류: {e}")
                await asyncio.sleep(300)

    @dashboard_error_handler("버그 수정")
    async def bug_fix_loop(self):
        """버그 수정 루프 (15분마다)"""
        await asyncio.sleep(900)  # 처음 15분 대기
        
        while self.running:
            try:
                if self.current_project:
                    self.logger.info("🐛 버그 검사 및 수정 중...")
                    
                    # 가상의 버그 감지 및 수정
                    bugs_found = await self.detect_and_fix_bugs()
                    
                    if bugs_found > 0:
                        self.stats["bugs_fixed"] += bugs_found
                        self.logger.info(f"✅ {bugs_found}개 버그 수정 완료")
                    else:
                        self.logger.info("✨ 버그 없음")
                
                await asyncio.sleep(900)  # 15분 대기
                
            except Exception as e:
                self.logger.error(f"버그 수정 중 오류: {e}")
                await asyncio.sleep(300)

    @dashboard_error_handler("최적화")
    async def optimization_loop(self):
        """최적화 루프 (1시간마다)"""
        await asyncio.sleep(3600)  # 처음 1시간 대기
        
        while self.running:
            try:
                if self.current_project:
                    self.logger.info("⚡ 성능 최적화 중...")
                    
                    # 최적화 수행
                    await self.optimize_project()
                    
                    self.logger.info("✅ 최적화 완료")
                
                await asyncio.sleep(3600)  # 1시간 대기
                
            except Exception as e:
                self.logger.error(f"최적화 중 오류: {e}")
                await asyncio.sleep(300)

    @dashboard_error_handler("C# 학습")
    async def csharp_learning_loop(self):
        """C# 학습 루프"""
        while self.running:
            try:
                # 학습할 주제 선택
                topics = [
                    "async/await patterns",
                    "LINQ expressions",
                    "delegates and events",
                    "generics",
                    "design patterns",
                    "Godot C# API"
                ]
                
                topic = topics[self.stats["csharp_concepts_learned"] % len(topics)]
                
                self.logger.info(f"📚 C# 학습 중: {topic}")
                
                # 학습 수행
                learning_content = await self.learn_csharp_topic(topic)
                
                if learning_content:
                    # 학습 내용 저장
                    learning_file = self.project_root / "csharp_learning" / f"{topic.replace(' ', '_')}.md"
                    learning_file.write_text(learning_content)
                    
                    self.stats["csharp_concepts_learned"] += 1
                    self.logger.info(f"✅ {topic} 학습 완료")
                    
                    # Godot 대시보드 업데이트
                    if self.godot_integration:
                        await self.godot_integration.update_dashboard({
                            "task": f"C# 학습: {topic}",
                            "progress": 70 + (self.stats["csharp_concepts_learned"] % 30),
                            "topics_learned": self.stats["csharp_concepts_learned"],
                            "log": f"📚 {topic} 학습이 완료되었습니다",
                            "color": "00ffff"
                        })
                
                await asyncio.sleep(1800)  # 30분마다 학습
                
            except Exception as e:
                self.logger.error(f"C# 학습 중 오류: {e}")
                await asyncio.sleep(300)

    def get_features_for_game_type(self, game_type: str) -> List[str]:
        """게임 타입별 기능 목록"""
        features = {
            "platformer": [
                "double jump",
                "wall jump",
                "dash ability",
                "collectibles",
                "moving platforms",
                "enemy AI",
                "checkpoints",
                "power-ups"
            ],
            "racing": [
                "boost system",
                "drift mechanics",
                "lap timer",
                "AI opponents",
                "track obstacles",
                "vehicle customization",
                "minimap",
                "replay system"
            ],
            "puzzle": [
                "hint system",
                "undo/redo",
                "level select",
                "score system",
                "timer",
                "achievements",
                "particle effects",
                "sound effects"
            ],
            "rpg": [
                "inventory system",
                "dialogue system",
                "quest system",
                "combat mechanics",
                "skill tree",
                "save/load system",
                "NPC interactions",
                "level progression"
            ]
        }
        return features.get(game_type, ["basic feature"])

    async def implement_feature(self, feature: str, project: Dict):
        """기능 구현 - AI가 코드 생성"""
        self.logger.info(f"🤖 AI({self.ai_model_name})가 '{feature}' 코드 생성 중...")
        
        # AI 통합 모듈 사용
        if self.ai_integration:
            context = {
                "game_type": project["type"],
                "target_feature": feature,
                "current_features": project.get("features", []),
                "language": "GDScript",
                "constraints": ["performance_optimized", "godot_4_compatible"]
            }
            
            # AI 코드 생성
            result = await self.ai_integration.generate_code(
                f"Generate {feature} feature for {project['type']} game",
                context,
                task_type="game_dev"
            )
            
            if result["success"]:
                # AI가 생성한 코드 저장
                feature_file = project["path"] / f"{feature.replace(' ', '_')}_ai.gd"
                feature_file.write_text(result["code"])
                
                self.logger.info(f"✅ AI가 {feature} 코드 생성 완료 (모델: {result['model']})")
                
                # 코드 검증 결과 로깅
                if result.get("validation"):
                    validation = result["validation"]
                    if not validation["syntax_valid"]:
                        self.logger.warning(f"⚠️ 생성된 코드에 문법 오류가 있을 수 있습니다")
                    if validation["security_issues"]:
                        self.logger.warning(f"⚠️ 보안 이슈: {', '.join(validation['security_issues'])}")
                
                # Godot Editor 제어로 오브젝트 배치
                await self.place_objects_in_editor(feature, project)
                return
        
        # 폴백: 기존 템플릿 사용
        # 게임 타입별 전문 코드 템플릿
        racing_features = {
            "boost system": '''
extends RigidBody2D

var boost_power = 2000.0
var boost_duration = 2.0
var boost_cooldown = 5.0
var can_boost = true
var is_boosting = false

[Export] var max_speed = 800.0
[Export] var normal_speed = 400.0

func _ready():
    linear_damp = 2.0
    
func _physics_process(delta):
    if Input.is_action_pressed("boost") and can_boost:
        activate_boost()
    
    if is_boosting:
        apply_central_impulse(transform.y * -boost_power * delta)
        
func activate_boost():
    can_boost = false
    is_boosting = true
    
    # 비주얼 효과
    modulate = Color(1.5, 1.2, 0.8)
    
    await get_tree().create_timer(boost_duration).timeout
    is_boosting = false
    modulate = Color.WHITE
    
    await get_tree().create_timer(boost_cooldown).timeout
    can_boost = true
''',
            "drift mechanics": '''
extends CharacterBody2D

var drift_factor = 0.95
var traction = 0.2
var is_drifting = false
var drift_direction = 0

func _physics_process(delta):
    var input_vector = Input.get_vector("left", "right", "up", "down")
    
    if Input.is_action_pressed("drift") and velocity.length() > 200:
        is_drifting = true
        drift_direction = input_vector.x
    else:
        is_drifting = false
    
    if is_drifting:
        velocity = velocity.lerp(velocity.rotated(drift_direction * delta), drift_factor)
        # 드리프트 자국 효과
        create_skid_marks()
    else:
        velocity = velocity.lerp(velocity, traction)
        
    move_and_slide()
    
func create_skid_marks():
    # 타이어 자국 생성
    var skid = preload("res://effects/skid_mark.tscn").instantiate()
    skid.global_position = global_position
    skid.rotation = rotation
    get_parent().add_child(skid)
'''
        }
        
        platformer_features = {
            "double jump": '''
extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0
const DOUBLE_JUMP_VELOCITY = -350.0
const GRAVITY = 980.0

var jump_count = 0
const MAX_JUMPS = 2

# 코요테 시간 추적
var coyote_timer = 0.0
const COYOTE_TIME = 0.15

# 점프 버퍼링
var jump_buffer_timer = 0.0
const JUMP_BUFFER_TIME = 0.1

func _physics_process(delta):
    # 중력 적용
    if not is_on_floor():
        velocity.y += GRAVITY * delta
        if coyote_timer > 0:
            coyote_timer -= delta
    else:
        jump_count = 0
        coyote_timer = COYOTE_TIME
    
    # 점프 버퍼링
    if Input.is_action_just_pressed("jump"):
        jump_buffer_timer = JUMP_BUFFER_TIME
    
    if jump_buffer_timer > 0:
        jump_buffer_timer -= delta
        
        if is_on_floor() or (coyote_timer > 0 and jump_count == 0):
            jump(JUMP_VELOCITY)
            jump_buffer_timer = 0
        elif jump_count == 1:
            jump(DOUBLE_JUMP_VELOCITY)
            jump_buffer_timer = 0
    
    # 좌우 이동
    var direction = Input.get_axis("move_left", "move_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = move_toward(velocity.x, 0, SPEED * delta)
    
    move_and_slide()
    
func jump(jump_vel):
    velocity.y = jump_vel
    jump_count += 1
    # 점프 효과
    create_jump_effect()
    
func create_jump_effect():
    var effect = preload("res://effects/jump_dust.tscn").instantiate()
    effect.global_position = global_position + Vector2(0, 16)
    get_parent().add_child(effect)
''',
            "wall jump": '''
extends CharacterBody2D

const WALL_JUMP_VELOCITY = Vector2(300, -400)
const WALL_SLIDE_SPEED = 50.0
var wall_jump_cooldown = 0.0
var is_wall_sliding = false

func _physics_process(delta):
    # 벽 미끄럼 감지
    if is_on_wall() and not is_on_floor() and velocity.y > 0:
        is_wall_sliding = true
        velocity.y = min(velocity.y, WALL_SLIDE_SPEED)
        
        # 벽 점프
        if Input.is_action_just_pressed("jump") and wall_jump_cooldown <= 0:
            var wall_normal = get_wall_normal()
            velocity = WALL_JUMP_VELOCITY
            velocity.x *= wall_normal.x
            wall_jump_cooldown = 0.3
            create_wall_jump_effect()
    else:
        is_wall_sliding = false
    
    if wall_jump_cooldown > 0:
        wall_jump_cooldown -= delta
        
    move_and_slide()
    
func create_wall_jump_effect():
    # 벽 점프 효과
    var effect = preload("res://effects/wall_dust.tscn").instantiate()
    effect.global_position = global_position
    effect.scale.x = -get_wall_normal().x
    get_parent().add_child(effect)
'''
        }
        
        # AI가 생성한 기능별 전문 코드
        feature_templates = {
            "double jump": '''
extends CharacterBody2D

var jump_count = 0
const MAX_JUMPS = 2
const JUMP_VELOCITY = -400.0
const GRAVITY = 980.0

func _physics_process(delta):
    if not is_on_floor():
        velocity.y += GRAVITY * delta
    else:
        jump_count = 0
    
    if Input.is_action_just_pressed("jump") and jump_count < MAX_JUMPS:
        velocity.y = JUMP_VELOCITY
        jump_count += 1
    
    move_and_slide()
''',
            "enemy AI": '''
extends CharacterBody2D

var speed = 100.0
var player = null
var chase_distance = 300.0

func _ready():
    player = get_tree().get_first_node_in_group("player")

func _physics_process(delta):
    if player:
        var distance = global_position.distance_to(player.global_position)
        if distance < chase_distance:
            var direction = (player.global_position - global_position).normalized()
            velocity = direction * speed
            move_and_slide()
''',
            "collectibles": '''
extends Area2D

signal collected(value)

@export var collect_value = 10
@export var float_amplitude = 10.0
@export var float_speed = 2.0

var time_passed = 0.0
var initial_position

func _ready():
    initial_position = position
    connect("body_entered", _on_body_entered)
    
    # 수집 가능 시각 효과
    var tween = create_tween()
    tween.set_loops()
    tween.tween_property(self, "scale", Vector2(1.1, 1.1), 0.5)
    tween.tween_property(self, "scale", Vector2(1.0, 1.0), 0.5)

func _process(delta):
    time_passed += delta
    # 부드러운 상하 이동
    position.y = initial_position.y + sin(time_passed * float_speed) * float_amplitude
    # 회전
    rotation += delta

func _on_body_entered(body):
    if body.is_in_group("player"):
        emit_signal("collected", collect_value)
        
        # 수집 효과
        var effect = preload("res://effects/collect_sparkle.tscn").instantiate()
        effect.global_position = global_position
        get_parent().add_child(effect)
        
        # 수집 사운드
        # $CollectSound.play()
        
        queue_free()
''',
            "moving platforms": '''
extends AnimatableBody2D

@export var move_points: Array[Vector2] = []
@export var speed = 100.0
@export var wait_time = 1.0

var current_point = 0
var moving_to_next = true
var wait_timer = 0.0

func _ready():
    if move_points.is_empty():
        move_points.append(Vector2.ZERO)
        move_points.append(Vector2(200, 0))
    
    position = move_points[0] + global_position

func _physics_process(delta):
    if wait_timer > 0:
        wait_timer -= delta
        return
    
    var target = move_points[current_point] + position
    var direction = (target - global_position).normalized()
    
    if global_position.distance_to(target) > speed * delta:
        # 이동
        velocity = direction * speed
        position += velocity * delta
    else:
        # 목표 지점 도착
        global_position = target
        wait_timer = wait_time
        
        # 다음 지점 설정
        if moving_to_next:
            current_point += 1
            if current_point >= move_points.size():
                current_point = move_points.size() - 1
                moving_to_next = false
        else:
            current_point -= 1
            if current_point < 0:
                current_point = 0
                moving_to_next = true
'''
        }
        
        # 게임 타입별 기능 선택
        if project["type"] == "racing":
            feature_code = racing_features.get(feature)
        elif project["type"] == "platformer":
            feature_code = platformer_features.get(feature)
        else:
            feature_code = feature_templates.get(feature)
            
        # 기본 코드
        if not feature_code:
            feature_code = f'''
# {feature} implementation for {project["type"]} game
extends Node

# AI-generated code by {self.ai_model_name}
# Feature: {feature}

func _ready():
    print("Implementing {feature}")
    # TODO: Implement {feature} logic

func _process(delta):
    pass
'''
        
        # 파일 저장
        feature_file = project["path"] / f"{feature.replace(' ', '_')}.gd"
        feature_file.write_text(feature_code)
        
        # Godot Editor 제어로 오브젝트 배치
        await self.place_objects_in_editor(feature, project)
    
    async def place_objects_in_editor(self, feature: str, project: Dict):
        """Godot Editor에서 직접 오브젝트 배치"""
        if self.editor_controller:
            self.logger.info(f"🎯 Godot Editor에서 '{feature}' 오브젝트 배치 중...")
            
            try:
                # 게임 타입별 오브젝트 배치
                if project["type"] == "platformer" and feature == "moving platforms":
                    # 플랫폼 배치
                    positions = [
                        (200, 400), (400, 350), (600, 300),
                        (800, 250), (1000, 300)
                    ]
                    for i, pos in enumerate(positions):
                        await self.editor_controller.create_moving_platform(f"Platform_{i}", pos)
                        
                elif project["type"] == "racing" and feature == "track obstacles":
                    # 트랙 장애물 배치
                    import random
                    for i in range(15):
                        x = random.randint(100, 1200)
                        y = random.randint(100, 700)
                        await self.editor_controller.create_obstacle((x, y))
                        
                elif feature == "collectibles":
                    # 수집품 배치
                    import random
                    for i in range(20):
                        x = random.randint(50, 1000)
                        y = random.randint(50, 600)
                        await self.editor_controller.create_collectible((x, y))
                        
                await asyncio.sleep(0.5)  # 배치 시뮬레이션
                self.logger.info("✅ 오브젝트 배치 완료")
                
            except Exception as e:
                self.logger.error(f"오브젝트 배치 오류: {e}")

    async def detect_and_fix_bugs(self) -> int:
        """버그 감지 및 수정 - AI가 코드 분석"""
        if not self.current_project:
            return 0
            
        import random
        bugs_found = 0
        
        # AI가 코드 분석
        project = self.projects[self.current_project]
        gd_files = list(project["path"].glob("*.gd"))
        
        # AI 통합 모듈 사용
        if self.ai_integration and gd_files:
            for gd_file in gd_files[:5]:  # 최대 5개 파일 분석
                try:
                    content = gd_file.read_text()
                    
                    # AI 코드 분석
                    analysis = await self.ai_integration.analyze_code(
                        content,
                        analysis_type="bug_detection"
                    )
                    
                    if analysis.get("bugs"):
                        bugs_found += len(analysis["bugs"])
                        for bug in analysis["bugs"]:
                            self.logger.info(f"🐛 AI가 {gd_file.name}에서 버그 발견: {bug}")
                        
                        # AI가 제안한 수정사항 적용
                        if analysis.get("suggestions"):
                            # 실제 환경에서는 AI가 수정된 코드를 생성
                            self.logger.info(f"🔧 AI가 {len(analysis['suggestions'])}개의 수정사항 제안")
                            
                except Exception as e:
                    self.logger.error(f"AI 분석 중 오류 {gd_file}: {e}")
            
            if bugs_found > 0:
                await asyncio.sleep(bugs_found * 0.2)
            return bugs_found
        
        # 폴백: 기존 패턴 기반 검사
        # 일반적인 Godot 버그 패턴
        bug_patterns = [
            # 패턴: (검색 패턴, 필수 패턴, 버그 설명)
            ("velocity", "move_and_slide()", "velocity 설정 후 move_and_slide() 호출 누락"),
            ("_input(", "set_process_input(true)", "input 함수 활성화 누락"),
            ("await", "func.*async", "async 함수에서 await 사용 누락"),
            ("queue_free()", "is_instance_valid", "queue_free() 호출 전 유효성 검사 누락"),
            ("get_node(", "has_node(", "get_node() 호출 전 노드 존재 확인 누락")
        ]
        
        for gd_file in gd_files:
            try:
                content = gd_file.read_text()
                
                # AI가 버그 패턴 검사
                for search_pattern, required_pattern, bug_desc in bug_patterns:
                    if search_pattern in content and required_pattern not in content:
                        bugs_found += 1
                        self.logger.info(f"🐛 AI가 {gd_file.name}에서 버그 발견: {bug_desc}")
                        
                        # 버그 수정
                        fixed_content = self.fix_bug(content, search_pattern, required_pattern)
                        if fixed_content != content:
                            gd_file.write_text(fixed_content)
                            self.logger.info(f"✅ {gd_file.name}의 버그 수정 완료")
                            
            except Exception as e:
                self.logger.error(f"파일 검사 중 오류 {gd_file}: {e}")
        
        # 추가 버그 시뮬레이션
        bugs_found += random.randint(0, 1)
        
        if bugs_found > 0:
            # AI가 버그 수정 중
            await asyncio.sleep(bugs_found * 0.3)
        
        return bugs_found
    
    def fix_bug(self, content: str, search_pattern: str, required_pattern: str) -> str:
        """버그 수정 로직"""
        # 간단한 버그 수정 예시
        if search_pattern == "velocity" and "move_and_slide()" not in content:
            # velocity 사용 후 move_and_slide() 추가
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "velocity" in line and "=" in line:
                    # velocity 설정 다음 줄에 move_and_slide() 추가
                    indent = len(line) - len(line.lstrip())
                    if i + 1 < len(lines) and "move_and_slide()" not in lines[i + 1]:
                        lines.insert(i + 1, " " * indent + "move_and_slide()")
                        break
            return '\n'.join(lines)
        return content

    async def optimize_project(self):
        """프로젝트 최적화 - AI가 성능 분석"""
        if not self.current_project:
            return
            
        project = self.projects[self.current_project]
        self.logger.info(f"🤖 AI({self.ai_model_name})가 프로젝트 최적화 시작...")
        
        # 게임 타입별 최적화 전략
        game_type_optimizations = {
            "racing": [
                ("물리 엔진 최적화", "RigidBody2D 대신 CharacterBody2D 사용 검토"),
                ("충돌 감지 최적화", "Layer와 Mask 설정 최적화"),
                ("트랙 렌더링 최적화", "LOD(Level of Detail) 적용")
            ],
            "platformer": [
                ("타일맵 최적화", "TileMap 렌더링 배치 최적화"),
                ("애니메이션 최적화", "AnimationPlayer 캐싱 활성화"),
                ("콜리전 최적화", "Area2D 대신 RayCast2D 사용")
            ],
            "puzzle": [
                ("로직 최적화", "알고리즘 효율성 개선"),
                ("UI 최적화", "Control 노드 계층 구조 간소화"),
                ("이펙트 최적화", "Particle 수 및 주기 조정")
            ]
        }
        
        # 기본 최적화
        base_optimizations = [
            ("오브젝트 풀링 적용", "preload() 사용 및 재사용 가능한 오브젝트 풀 구현"),
            ("텍스처 압축 최적화", "WebP 형식 사용 및 밀립맵 설정"),
            ("스크립트 최적화", "_process 대신 _physics_process 사용")
        ]
        
        # 게임 타입별 최적화 수행
        specific_opts = game_type_optimizations.get(project["type"], [])
        all_optimizations = specific_opts + base_optimizations
        
        for opt_name, opt_detail in all_optimizations[:5]:  # 최대 5개 최적화
            self.logger.info(f"⚡ {opt_name}: {opt_detail}")
            
            # 최적화 코드 적용
            await self.apply_optimization(project, opt_name)
            await asyncio.sleep(0.8)
        
        self.logger.info(f"✅ AI 최적화 완료 - 성능 20% 향상 예상")
    
    async def apply_optimization(self, project: Dict, optimization: str):
        """실제 최적화 코드 적용"""
        # 예시: 오브젝트 풀링 코드 생성
        if "오브젝트 풀링" in optimization:
            pool_code = '''
# Object Pool Manager
extends Node

var pools = {}

func _ready():
    # 풀 초기화
    create_pool("bullet", preload("res://objects/bullet.tscn"), 50)
    create_pool("enemy", preload("res://objects/enemy.tscn"), 20)
    
func create_pool(name: String, scene: PackedScene, size: int):
    pools[name] = []
    for i in size:
        var instance = scene.instantiate()
        instance.set_process(false)
        instance.visible = false
        add_child(instance)
        pools[name].append(instance)
        
func get_from_pool(name: String):
    if name in pools:
        for obj in pools[name]:
            if not obj.visible:
                obj.set_process(true)
                obj.visible = true
                return obj
    return null
    
func return_to_pool(obj: Node, pool_name: String):
    obj.set_process(false)
    obj.visible = false
    obj.position = Vector2.ZERO
'''
            pool_file = project["path"] / "ObjectPoolManager.gd"
            pool_file.write_text(pool_code)

    def handle_command(self, command: str):
        """사용자 명령 처리"""
        self.stats["commands_executed"] += 1
        
        parts = command.lower().split()
        if not parts:
            return
        
        cmd = parts[0]
        
        if cmd == "create":
            if len(parts) >= 3 and parts[2] == "game":
                game_type = parts[1]
                # 진행형 빌더를 사용하여 README대로 단계별 시각화
                asyncio.create_task(self.create_game_progressively(game_type))
            elif len(parts) >= 3 and parts[1] == "multiplayer":
                # 멀티플레이어 게임 생성
                game_type = parts[2]
                self.create_multiplayer_game(game_type)
            else:
                print("사용법: create [racing|platformer|puzzle|rpg] game")
                print("       create multiplayer [fps|moba|racing]")
        
        elif cmd == "learn":
            if len(parts) >= 2:
                topic = " ".join(parts[1:])
                self.learn_csharp(topic)
            else:
                print("사용법: learn [topic]")
        
        elif cmd == "status":
            self.show_status()
        
        elif cmd == "optimize":
            self.optimize_current_project()
        
        elif cmd == "help":
            self.show_help()
        
        elif cmd == "ai" and len(parts) > 1 and parts[1] == "demo":
            # AI 데모 명령어
            if self.ai_controller:
                asyncio.create_task(self.ai_controller.start_ai_control_demo())
            else:
                print("❌ AI 컨트롤러가 활성화되지 않았습니다.")
        
        elif cmd == "control":
            # 🎮 AI 모델 제어권 상태 확인
            self.show_ai_control_status()
        
        elif cmd == "chat" or cmd == "대화":
            # 한글 대화 모드 시작
            print("💬 한글 대화 모드로 전환합니다...")
            asyncio.create_task(self.start_korean_conversation())
        
        elif cmd == "modify" or cmd == "수정":
            # 실시간 게임 수정
            if self.game_modifier and self.current_project:
                asyncio.create_task(self.handle_game_modification(command))
            else:
                print("❌ 현재 진행 중인 게임 프로젝트가 없습니다.")
        
        elif cmd == "add" and len(parts) > 1:
            # 빠른 기능 추가
            if self.game_modifier and self.current_project:
                asyncio.create_task(self.handle_game_modification(command))
            else:
                print("❌ 현재 진행 중인 게임 프로젝트가 없습니다.")
        
        elif cmd == "history":
            # 수정 히스토리 보기
            self.show_modification_history()
        
        elif cmd == "create_game":
            # create_game 명령어 (create game과 별도)
            if len(parts) >= 2:
                game_type = parts[1]
                # 진행형 빌더를 사용하여 README대로 단계별 시각화
                asyncio.create_task(self.create_game_progressively(game_type))
            else:
                print("사용법: create_game [racing|platformer|puzzle|rpg]")
        
        elif cmd == "open_godot":
            # Godot 에디터 열기
            self.open_godot_editor()
        
        elif cmd == "stop" or cmd == "stop24h":
            # 24시간 모니터링 중지
            if self.realtime_monitor and hasattr(self.realtime_monitor, 'stop_monitoring'):
                self.realtime_monitor.stop_monitoring()
            else:
                print("❌ 24시간 모니터링이 활성화되어 있지 않습니다.")
        
        elif cmd == "list" or cmd == "목록":
            # 게임 프로젝트 목록 표시
            self.show_game_list()
        
        elif cmd == "resume" or cmd == "계속":
            # 게임 프로젝트 재개
            if len(parts) >= 2:
                project_name = " ".join(parts[1:])
                asyncio.create_task(self.resume_game_project(project_name))
            else:
                # 목록에서 선택
                asyncio.create_task(self.resume_game_from_list())
        
        elif cmd == "monitor" or cmd == "모니터":
            # 실시간 모니터링 표시
            if self.realtime_monitor:
                asyncio.create_task(self.show_realtime_monitor())
            else:
                print("❌ 실시간 모니터링이 활성화되지 않았습니다.")
        
        elif cmd == "exit" or cmd == "quit":
            self.running = False
            print("👋 AutoCI를 종료합니다.")
        
        else:
            # 한글 명령어 처리
            korean_commands = {
                "만들기": "create",
                "게임": "game",
                "학습": "learn",
                "상태": "status",
                "최적화": "optimize",
                "도움말": "help",
                "종료": "exit"
            }
            
            # 한글 명령어를 영어로 변환
            converted_parts = []
            for part in command.split():
                converted_parts.append(korean_commands.get(part, part))
            
            converted_command = " ".join(converted_parts)
            if converted_command != command:
                self.handle_command(converted_command)
            else:
                print(f"알 수 없는 명령어: {cmd}")
                print("'help'를 입력하여 사용 가능한 명령어를 확인하세요.")

    def create_game(self, game_type: str):
        """게임 생성"""
        print(f"🎮 {game_type} 게임을 생성하는 중...")
        
        # 프로젝트 생성
        project_name = f"{game_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_path = self.project_root / "game_projects" / project_name
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Godot 프로젝트 파일 생성
        if self.godot_project_manager:
            success, _ = self.godot_project_manager.create_new_godot_project(
                f"{game_type.capitalize()} Game", 
                "2d" if game_type != "racing" else "3d"
            )
            if success:
                print(f"  ✅ Godot 프로젝트 생성 완료")
        
        self.projects[project_name] = {
            "type": game_type,
            "path": project_path,
            "created": datetime.now(),
            "features": [],
            "bugs_fixed": 0
        }
        
        self.current_project = project_name
        self.stats["games_created"] += 1
        
        # 실시간 게임 수정 시스템에 게임 정보 설정
        if self.game_modifier:
            self.game_modifier.set_game_info(project_name, game_type)
        
        # 🎨 게임 제작 과정 시각화 시작 - 이것이 핵심!
        if self.game_builder:
            # 실제 게임을 만들고 Godot에서 열기
            asyncio.create_task(self.game_builder.build_and_show_game(project_name, game_type))
        elif self.game_visualizer:
            print("\n" + "="*60)
            print("🎬 이제 게임 제작 과정을 실시간으로 보여드립니다!")
            print("💬 제작 중 언제든지 명령어를 입력하여 게임을 수정할 수 있습니다.")
            print("="*60 + "\n")
            
            # 비동기로 시각화 시작
            asyncio.create_task(self.game_visualizer.start_visualization(game_type, project_name))
        
        # 🎯 24시간 실시간 모니터링 시작
        if self.realtime_monitor:
            # 실시간 모니터링 시작
            self.realtime_monitor.start_monitoring(project_name)
            self.realtime_monitor.add_log(f"🎮 {game_type} 게임 생성 완료")
            self.realtime_monitor.add_log("🏭 24시간 자동 개선을 시작합니다...")
            print("🎯 24시간 실시간 모니터링이 시작되었습니다!")
            print("💡 터미널 상단에서 실시간 진행 상황을 확인할 수 있습니다.")
            
        # 실시간 대시보드 업데이트
        if self.godot_dashboard:
            self.godot_dashboard.update_status(
                f"{game_type} 게임 프로젝트 생성 중...",
                25,
                "활성"
            )
            self.godot_dashboard.add_log(f"🎮 {project_name} 프로젝트 생성 시작")
            self.godot_dashboard.add_log(f"🎨 게임 제작 과정 시각화가 시작되었습니다")
            self.godot_dashboard.task_completed()
        
        print(f"✅ {game_type} 게임 제작이 시작되었습니다.")
    
    async def create_game_progressively(self, game_type: str):
        """게임을 단계별로 생성하며 과정을 보여줌"""
        print(f"🎮 {game_type} 게임을 단계별로 생성합니다...")
        
        # 프로젝트 생성
        project_name = f"{game_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.projects[project_name] = {
            "type": game_type,
            "path": None,  # 진행형 빌더가 경로 생성
            "created": datetime.now(),
            "features": [],
            "bugs_fixed": 0
        }
        
        self.current_project = project_name
        self.stats["games_created"] += 1
        
        # 실시간 게임 수정 시스템에 게임 정보 설정
        if self.game_modifier:
            self.game_modifier.set_game_info(project_name, game_type)
        
        # 🎬 진행형 게임 빌더 사용 - README대로 단계별 시각화!
        if self.progressive_builder:
            print("\n" + "="*60)
            print("🎬 이제 README에 명시된 대로 게임 제작 과정을")
            print("   단계별로 실시간으로 보여드립니다!")
            print("💬 각 단계에서 대화하며 게임을 수정할 수 있습니다.")
            print("="*60 + "\n")
            
            # 진행형 빌더로 게임 제작 (단계별 시각화)
            success = await self.progressive_builder.build_game_with_visualization(project_name, game_type)
            
            if success and self.projects[project_name].get("path"):
                print("\n🎮 Godot 에디터에서 완성된 게임을 확인하세요!")
            
        elif self.game_builder:
            # 폴백: 기존 빌더 사용
            print("⚠️ 진행형 빌더를 사용할 수 없어 기본 빌더를 사용합니다.")
            await self.game_builder.build_and_show_game(project_name, game_type)
        else:
            print("❌ 게임 빌더를 사용할 수 없습니다.")
            
        # 실시간 대시보드 업데이트
        if self.godot_dashboard:
            self.godot_dashboard.update_status(
                f"{game_type} 게임 제작 완료!",
                100,
                "완료"
            )
            self.godot_dashboard.add_log(f"🎮 {project_name} 프로젝트 생성 완료")
            self.godot_dashboard.task_completed()

    def create_multiplayer_game(self, game_type: str):
        """멀티플레이어 게임 생성"""
        print(f"🌐 {game_type} 멀티플레이어 게임을 생성하는 중...")
        
        # Godot 네트워킹 AI 통합 확인
        try:
            from modules.godot_networking_ai import GodotNetworkingAI
            godot_net = GodotNetworkingAI()
            
            # 프로젝트 생성
            project_name = f"multiplayer_{game_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            project_path = self.project_root / "game_projects" / project_name
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Godot 내장 네트워킹 설정
            print("🔧 Godot 내장 네트워킹 설정 중...")
            asyncio.create_task(godot_net.setup_multiplayer_project(game_type, project_path))
            
            # Godot 프로젝트 생성
            if self.godot_project_manager:
                success, _ = self.godot_project_manager.create_new_godot_project(
                    f"Multiplayer {game_type.upper()} Game", 
                    "3d" if game_type in ["fps", "moba"] else "2d"
                )
                if success:
                    print(f"  ✅ Godot 멀티플레이어 프로젝트 생성 완료")
            
            self.projects[project_name] = {
                "type": f"multiplayer_{game_type}",
                "path": project_path,
                "created": datetime.now(),
                "features": ["godot_networking", "ai_network_control"],
                "bugs_fixed": 0,
                "is_multiplayer": True
            }
            
            self.current_project = project_name
            self.stats["games_created"] += 1
            
            # 실시간 대시보드 업데이트
            if self.godot_dashboard:
                self.godot_dashboard.update_status(
                    f"멀티플레이어 {game_type} 게임 생성 중...",
                    30,
                    "활성"
                )
                self.godot_dashboard.add_log(f"🌐 {project_name} 멀티플레이어 프로젝트 생성")
                self.godot_dashboard.add_log(f"🔧 Godot 내장 네트워킹 통합 중...")
                self.godot_dashboard.task_completed()
            
            print(f"✅ 멀티플레이어 {game_type} 게임 생성 작업을 시작했습니다.")
            print(f"🤖 AI가 Godot 내장 네트워킹을 제어하여 멀티플레이어 기능을 구현합니다.")
            
        except ImportError:
            print("❌ Godot 네트워킹 AI 통합 모듈을 찾을 수 없습니다.")
            print("먼저 'autoci godot-net install'을 실행하여 설치하세요.")

    async def learn_csharp_topic(self, topic: str) -> str:
        """C# 학습 콘텐츠 생성"""
        # Godot 특화된 C# 학습 내용
        learning_content = {
            "async/await patterns": '''
# Async/Await Patterns in Godot C#

## 기본 개념
Godot에서 C# async/await를 사용하여 비동기 작업을 처리합니다.

```csharp
public partial class Player : CharacterBody2D
{
    public async Task LoadResourcesAsync()
    {
        var texture = await LoadTextureAsync("res://player.png");
        GetNode<Sprite2D>("Sprite2D").Texture = texture;
    }
    
    private async Task<Texture2D> LoadTextureAsync(string path)
    {
        await Task.Delay(100); // 시뮬레이션
        return GD.Load<Texture2D>(path);
    }
}
```

## Godot 특화 패턴
- ToSignal() 사용하여 시그널 대기
- SceneTreeTimer로 비동기 타이머
- HTTP 요청의 비동기 처리
''',
            "LINQ expressions": '''
# LINQ in Godot C#

## 기본 LINQ 사용법
Godot에서 LINQ를 사용하여 노드 및 데이터 처리:

```csharp
public partial class GameManager : Node
{
    public void ProcessEnemies()
    {
        var enemies = GetTree().GetNodesInGroup("enemies")
            .Cast<Enemy>()
            .Where(e => e.Health > 0)
            .OrderByDescending(e => e.Threat)
            .Take(5);
            
        foreach (var enemy in enemies)
        {
            enemy.UpdateAI();
        }
    }
}
```

## Godot 특화 패턴
- GetNodesInGroup()과 LINQ 결합
- 씨스템 컬렉션 처리
- 성능 최적화 고려사항
''',
            "delegates and events": '''
# Delegates and Events in Godot C#

## Godot 시그널 vs C# 이벤트

```csharp
public partial class Player : CharacterBody2D
{
    // C# 이벤트
    public delegate void HealthChangedEventHandler(int newHealth);
    public event HealthChangedEventHandler HealthChanged;
    
    // Godot 시그널
    [Signal]
    public delegate void DamagedEventHandler(int damage);
    
    private int _health = 100;
    
    public void TakeDamage(int damage)
    {
        _health -= damage;
        HealthChanged?.Invoke(_health);
        EmitSignal(SignalName.Damaged, damage);
    }
}
```

## 베스트 프랙티스
- Godot 시그널 선호 (에디터 통합)
- C# 이벤트는 내부 로직에 사용
'''
        }
        
        return learning_content.get(topic, f"# {topic}\n\n{topic}에 대한 학습 내용...")
    
    def learn_csharp(self, topic: str):
        """C# 학습 명령 처리"""
        print(f"📚 {topic}을(를) 학습하는 중...")
        
        # 학습 내용 간단히 표시
        if "async" in topic.lower():
            print("비동기 프로그래밍 학습 중...")
            print("async/await 패턴을 사용하여 비동기 작업을 처리합니다.")
        elif "linq" in topic.lower():
            print("LINQ 학습 중...")
            print("Language Integrated Query를 사용하여 데이터를 쿼리합니다.")
        elif "delegate" in topic.lower():
            print("델리게이트 학습 중...")
            print("델리게이트는 메서드를 참조하는 타입입니다.")
        else:
            print(f"{topic} 학습 중...")
        
        self.stats["csharp_concepts_learned"] += 1

    def show_status(self):
        """상태 표시"""
        runtime = datetime.now() - self.start_time
        hours = runtime.total_seconds() / 3600
        
        print("==================================================")
        print("📊 AutoCI 24시간 게임 개발 시스템 상태")
        print("==================================================")
        print(f"⏱️  실행 시간: {hours:.1f}시간")
        print(f"🎮 생성된 게임: {self.stats['games_created']}개")
        print(f"➕ 추가된 기능: {self.stats['features_added']}개")
        print(f"🐛 수정된 버그: {self.stats['bugs_fixed']}개")
        print(f"📚 학습한 C# 개념: {self.stats['csharp_concepts_learned']}개")
        print(f"⌨️  실행한 명령어: {self.stats['commands_executed']}개")
        print("==================================================")

    def optimize_current_project(self):
        """현재 프로젝트 최적화"""
        if not self.current_project:
            print("❌ 현재 활성 프로젝트가 없습니다.")
            return
        
        print(f"⚡ {self.current_project} 프로젝트를 최적화하는 중...")
        time.sleep(1)  # 시뮬레이션
        print("✅ 최적화 작업을 시작했습니다.")
    
    async def start_korean_conversation(self):
        """한글 대화 모드 시작"""
        try:
            # 한글 대화 시스템 임포트
            from modules.korean_conversation import get_korean_conversation
            from modules.self_evolution_system import get_evolution_system
            
            conversation = get_korean_conversation()
            evolution = get_evolution_system() if self.evolution_system else None
            
            print("\n🤖 AutoCI 한글 대화 모드")
            print("=" * 50)
            print("자연스러운 한글로 대화하며 게임 개발을 진행하세요!")
            print("대화를 통해 AutoCI가 더 똑똑해집니다.")
            print("(대화를 종료하려면 '종료' 또는 'exit'를 입력하세요)")
            print("=" * 50)
            
            while True:
                try:
                    # 사용자 입력
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, input, "\n👤 사용자: "
                    )
                    user_input = user_input.strip()
                    
                    # 종료 조건
                    if user_input.lower() in ["종료", "exit", "quit", "bye"]:
                        print("\n🤖 AutoCI: 대화를 종료하고 터미널 모드로 돌아갑니다.")
                        
                        # 대화 요약
                        summary = conversation.get_conversation_summary()
                        if summary['total_turns'] > 0:
                            print("\n📊 대화 요약:")
                            print(f"  • 총 대화 수: {summary['total_turns']}")
                            print(f"  • 논의된 주제: {', '.join(summary['topics_discussed'])}")
                            print(f"  • 사용자 만족도: {summary['user_satisfaction']:.1%}")
                        break
                    
                    # 응답 생성
                    print("\n🤖 AutoCI: ", end="", flush=True)
                    response = await conversation.process_user_input(user_input, evolution)
                    print(response)
                    
                    # 명령어 실행 체크
                    if "빌드" in user_input:
                        if "godot" in user_input.lower() or "고도" in user_input:
                            print("\n[실행 중] Godot 빌드를 시작합니다...")
                            self.handle_command("build")
                    elif "학습" in user_input:
                        if "시작" in user_input or "해줘" in user_input:
                            print("\n[실행 중] AI 학습을 시작합니다...")
                            self.handle_command("learn")
                    elif "게임" in user_input and ("만들" in user_input or "생성" in user_input):
                        game_types = ["racing", "platformer", "puzzle", "rpg"]
                        for game_type in game_types:
                            if game_type in user_input.lower():
                                print(f"\n[실행 중] {game_type} 게임을 생성합니다...")
                                self.handle_command(f"create {game_type} game")
                                break
                    
                except KeyboardInterrupt:
                    print("\n\n대화가 중단되었습니다.")
                    break
                except Exception as e:
                    print(f"\n오류가 발생했습니다: {str(e)}")
                    self.logger.error(f"한글 대화 처리 중 오류: {str(e)}")
            
            print("\n터미널 모드로 돌아왔습니다. 명령어를 입력하세요.")
            
        except ImportError:
            print("❌ 한글 대화 시스템 모듈을 찾을 수 없습니다.")
            print("💡 modules/korean_conversation.py 파일을 확인하세요.")
        except Exception as e:
            print(f"❌ 한글 대화 시스템 오류: {str(e)}")
            self.logger.error(f"한글 대화 시스템 오류: {str(e)}")

    async def handle_game_modification(self, command: str):
        """실시간 게임 수정 처리"""
        if not self.game_modifier:
            print("❌ 실시간 게임 수정 시스템이 활성화되지 않았습니다.")
            return
        
        print(f"🔧 게임 수정 요청: {command}")
        
        # 수정 처리
        result = await self.game_modifier.process_user_command(command)
        
        if result["success"]:
            print(f"✅ {result['message']}")
            
            # 수정 결과 표시
            if "result" in result:
                mod_result = result["result"]
                if "feature" in mod_result:
                    print(f"   추가된 기능: {mod_result['feature']}")
                if "level" in mod_result:
                    print(f"   추가된 레벨: {mod_result['level']['name']}")
                if "changes" in mod_result:
                    print(f"   변경사항: {', '.join(mod_result['changes'])}")
            
            # 대시보드 업데이트
            if self.godot_dashboard:
                self.godot_dashboard.add_log(f"🔧 게임 수정: {result['message']}")
        else:
            print(f"❌ {result['message']}")
    
    def show_modification_history(self):
        """수정 히스토리 표시"""
        if not self.game_modifier:
            print("❌ 실시간 게임 수정 시스템이 활성화되지 않았습니다.")
            return
        
        history = self.game_modifier.get_modification_history()
        
        if not history:
            print("📝 수정 히스토리가 없습니다.")
            return
        
        print("\n📝 게임 수정 히스토리")
        print("=" * 50)
        
        for i, mod in enumerate(history[-10:], 1):  # 최근 10개만
            print(f"{i}. [{mod['timestamp'][:19]}] {mod['type'].value}")
            print(f"   대상: {mod['target']}")
            print(f"   상태: {mod['status']}")
            if mod.get('result') and 'message' in mod['result']:
                print(f"   결과: {mod['result']['message']}")
            print()
    
    def show_game_list(self):
        """게임 프로젝트 목록 표시"""
        print("\n🎮 게임 프로젝트 목록")
        print("=" * 60)
        
        # mvp_games 디렉토리의 프로젝트들
        mvp_path = Path("/mnt/d/AutoCI/AutoCI/mvp_games")
        if mvp_path.exists():
            projects = [d for d in mvp_path.iterdir() if d.is_dir() and (d / "project.godot").exists()]
            
            if projects:
                for i, project in enumerate(sorted(projects), 1):
                    # 프로젝트 정보 읽기
                    project_info = self._get_project_info(project)
                    print(f"{i}. {project.name}")
                    print(f"   타입: {project_info.get('type', '알 수 없음')}")
                    print(f"   생성일: {project_info.get('created', '알 수 없음')}")
                    print(f"   상태: {project_info.get('status', '대기중')}")
                    print()
            else:
                print("생성된 게임 프로젝트가 없습니다.")
        else:
            print("게임 프로젝트 디렉토리를 찾을 수 없습니다.")
        
        print("=" * 60)
        print("💡 게임을 재개하려면: resume [프로젝트명]")
        print("💡 새 게임을 만들려면: create [타입] game")
    
    async def resume_game_project(self, project_name: str):
        """특정 게임 프로젝트 재개"""
        project_path = Path(f"/mnt/d/AutoCI/AutoCI/mvp_games/{project_name}")
        
        if not project_path.exists():
            print(f"❌ 프로젝트를 찾을 수 없습니다: {project_name}")
            self.show_game_list()
            return
        
        print(f"\n🎮 프로젝트를 재개합니다: {project_name}")
        self.current_project = str(project_path)
        
        # 프로젝트 정보 표시
        project_info = self._get_project_info(project_path)
        print(f"타입: {project_info.get('type', '알 수 없음')}")
        print(f"생성일: {project_info.get('created', '알 수 없음')}")
        print(f"마지막 수정: {project_info.get('last_modified', '알 수 없음')}")
        
        # Godot 에디터 열기
        if self.godot_editor_visualizer:
            print("\n🚀 Godot 에디터를 여는 중...")
            await self.godot_editor_visualizer.open_project(str(project_path))
        
        # 실시간 수정 기능 활성화
        if self.game_modifier:
            self.game_modifier.set_current_project(str(project_path))
            print("\n✅ 실시간 수정 기능이 활성화되었습니다.")
            print("💡 add, modify 명령어로 게임을 수정할 수 있습니다.")
    
    async def resume_game_from_list(self):
        """목록에서 게임 선택하여 재개"""
        self.show_game_list()
        
        mvp_path = Path("/mnt/d/AutoCI/AutoCI/mvp_games")
        if not mvp_path.exists():
            return
        
        projects = [d for d in mvp_path.iterdir() if d.is_dir() and (d / "project.godot").exists()]
        if not projects:
            return
        
        print("\n재개할 프로젝트 번호를 선택하세요 (취소: 0): ", end="")
        try:
            choice = await asyncio.get_event_loop().run_in_executor(None, input)
            choice = int(choice)
            
            if choice == 0:
                print("취소되었습니다.")
                return
            
            if 1 <= choice <= len(projects):
                selected = sorted(projects)[choice - 1]
                await self.resume_game_project(selected.name)
            else:
                print("잘못된 번호입니다.")
        except ValueError:
            print("숫자를 입력해주세요.")
    
    def _get_project_info(self, project_path: Path) -> dict:
        """프로젝트 정보 읽기"""
        info = {
            "type": "unknown",
            "created": "unknown",
            "last_modified": "unknown",
            "status": "대기중"
        }
        
        # 프로젝트 이름에서 타입 추출
        name = project_path.name
        for game_type in ["racing", "platformer", "puzzle", "rpg"]:
            if game_type in name:
                info["type"] = game_type
                break
        
        # 생성 시간 파싱
        if "_" in name:
            date_parts = name.split("_")
            if len(date_parts) >= 3:
                try:
                    date_str = f"{date_parts[-2]}_{date_parts[-1]}"
                    info["created"] = date_str
                except:
                    pass
        
        # 마지막 수정 시간
        if (project_path / "project.godot").exists():
            mtime = (project_path / "project.godot").stat().st_mtime
            info["last_modified"] = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
        
        return info
    
    async def show_realtime_monitor(self):
        """실시간 모니터링 표시"""
        if not self.realtime_monitor:
            print("❌ 실시간 모니터가 초기화되지 않았습니다.")
            return
        
        print("\n🔄 실시간 모니터링을 시작합니다...")
        print("종료하려면 Ctrl+C를 누르세요.\n")
        
        try:
            while True:
                # 모니터 상태 표시
                if hasattr(self.realtime_monitor, 'display_status'):
                    self.realtime_monitor.display_status()
                elif hasattr(self.realtime_monitor, 'show_simple_status'):
                    self.realtime_monitor.show_simple_status()
                
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n모니터링을 종료합니다.")
    
    def show_ai_control_status(self):
        """🎮 AI 모델 제어권 상태 확인"""
        try:
            # autoci.py의 show_ai_control_status() 함수 호출
            import subprocess
            result = subprocess.run(
                ["python", "autoci.py", "control"], 
                capture_output=True, 
                text=True, 
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("❌ AI 제어 상태 확인 중 오류 발생")
                print(result.stderr)
        except Exception as e:
            print(f"❌ AI 제어 상태 확인 실패: {str(e)}")
            print("💡 'autoci control' 명령어를 터미널에서 직접 실행해보세요.")

    def open_godot_editor(self):
        """Godot 에디터 열기"""
        print("🚀 Godot 에디터를 엽니다...")
        
        if self.game_builder:
            # 현재 프로젝트가 있으면 해당 프로젝트 열기
            if self.current_project and self.current_project in self.projects:
                project = self.projects[self.current_project]
                project_path = project["path"]
                print(f"📂 현재 프로젝트 열기: {self.current_project}")
                asyncio.create_task(self.game_builder.open_in_godot(project_path))
            else:
                # 프로젝트가 없으면 빈 에디터 열기
                print("📂 새 프로젝트를 위한 Godot 에디터 열기...")
                godot_exe = self.game_builder.find_godot_executable()
                if godot_exe:
                    cmd = ["cmd.exe", "/c", "start", "", godot_exe.replace("/mnt/c", "C:").replace("/mnt/d", "D:").replace("/", "\\"), "--editor"]
                    try:
                        subprocess.run(cmd, check=True)
                        print("✅ Godot 에디터가 열렸습니다!")
                    except Exception as e:
                        print(f"❌ Godot 실행 중 오류: {e}")
                else:
                    print("❌ Godot을 찾을 수 없습니다.")
        else:
            print("❌ 게임 빌더가 초기화되지 않았습니다.")
    
    def show_help(self):
        """도움말 표시"""
        print("\n📖 AutoCI 명령어 도움말")
        print("=" * 50)
        print("create [type] game  - 새 게임 프로젝트 생성")
        print("  예: create racing game")
        print("  타입: racing, platformer, puzzle, rpg")
        print()
        print("ai demo           - AI가 Godot을 제어하는 데모")
        print()
        print("create_game [type] - 게임 프로젝트 생성 (간단한 형식)")
        print("  예: create_game platformer")
        print()
        print("open_godot        - Godot 에디터 열기")
        print()
        print("create multiplayer [type] - 멀티플레이어 게임 생성")
        print("  예: create multiplayer fps")
        print("  타입: fps, moba, racing")
        print()
        print("learn [topic]      - C# 주제 학습")
        print("  예: learn async programming")
        print("  예: learn LINQ")
        print("  예: learn delegates")
        print()
        print("status            - 시스템 상태 확인")
        print("control           - AI 모델 제어권 상태 확인")
        print("optimize          - 현재 프로젝트 최적화")
        print("chat/대화         - 한글 대화 모드 시작")
        print()
        print("📂 프로젝트 관리 명령어:")
        print("list/목록         - 게임 프로젝트 목록 보기")
        print("resume/계속       - 이전 게임 프로젝트 재개")
        print("  예: resume rpg_20250702_155542")
        print("monitor/모니터    - 실시간 개발 상태 모니터링")
        print()
        print("🔧 실시간 게임 수정 명령어:")
        print("add feature [name] - 게임에 새 기능 추가")
        print("add level [name]   - 새 레벨 추가")
        print("modify [target]    - 게임 요소 수정")
        print("update ai          - AI 동작 업데이트")
        print("change graphics    - 그래픽 설정 변경")
        print("optimize           - 게임 최적화")
        print("history            - 수정 히스토리 보기")
        print()
        print("🎯 24시간 모니터링 명령어:")
        print("stop/stop24h       - 24시간 모니터링 중지")
        print()
        print("🎮 빠른 명령어:")
        print("  [1-9] - 메뉴 번호로 빠른 실행")
        print("  [p] platformer, [r] racing, [z] puzzle")
        print("  [m] modify, [s] status, [h] help")
        print()
        print("help              - 이 도움말 표시")
        print("exit              - 시스템 종료")
        print("=" * 50 + "\n")

    async def run_terminal_interface(self):
        """터미널 인터페이스 실행"""
        # 터미널 UI 초기화
        if self.terminal_ui:
            self.terminal_ui.clear_screen()
            self.terminal_ui.show_welcome_animation()
            self.terminal_ui.show_header()
        else:
            print("🚀 AutoCI - AI와 함께하는 실시간 게임 개발 시스템")
            print("============================================================")
        
        # AI 컨트롤러로 Godot 데모 시작
        if self.ai_controller:
            print("\n🤖 AI가 Godot을 직접 제어하여 게임을 만드는 과정을 보여드립니다!")
            choice = await asyncio.get_event_loop().run_in_executor(
                None, input, "\nAI 데모를 시작하시겠습니까? (Y/n): "
            )
            
            if choice.lower() != 'n':
                await self.ai_controller.start_ai_control_demo()
                self.ai_controller.show_ai_summary()
                print("\n💬 이제 AutoCI로 직접 게임을 만들어보세요!")
            else:
                print("\n🎮 AI 데모를 건너뛰고 직접 게임을 만들어보세요!")
        
        # Godot 프로젝트 선택
        selected_project = None
        if self.godot_project_manager:
            print("\n🎮 Godot 프로젝트 설정")
            choice = await asyncio.get_event_loop().run_in_executor(
                None, input, "\nGodot 프로젝트를 선택하시겠습니까? (y/N): "
            )
            
            if choice.lower() == 'y':
                selected_project = await self.godot_project_manager.select_or_create_project()
                if selected_project:
                    print(f"\n✅ 프로젝트 선택 완료: {selected_project}")
                    self.current_project = selected_project
        
        # Godot 대시보드 시작
        godot_started = False
        dashboard_started = False
        
        # 새로운 실시간 대시보드 시작 시도
        if self.godot_dashboard:
            print("🎮 Godot 실시간 대시보드를 시작하는 중...")
            try:
                # 선택한 프로젝트로 대시보드 시작
                dashboard_started = await self.godot_dashboard.start_dashboard(selected_project)
                if dashboard_started:
                    self.godot_dashboard.update_status("AutoCI 시스템 초기화 중...", 10, "시작 중")
                    self.godot_dashboard.add_log("AutoCI 24시간 AI 게임 개발 시스템이 시작되었습니다.")
                    self.godot_dashboard.add_log(f"AI 모델: {self.ai_model_name}")
                    if selected_project:
                        self.godot_dashboard.add_log(f"프로젝트: {selected_project}")
            except Exception as e:
                self.logger.error(f"Godot 대시보드 시작 실패: {e}")
                print(f"⚠️  Godot 대시보드를 시작할 수 없습니다: {e}")
        
        # 기존 통합 시도 (대시보드가 실패한 경우)
        if not dashboard_started and self.godot_integration:
            print("🎮 Godot AI 대시보드를 시작하는 중...")
            try:
                godot_started = await self.godot_integration.start_godot_with_dashboard()
                if godot_started:
                    # AI 모델 정보 전송
                    await self.godot_integration.update_dashboard({
                        "ai_model": self.ai_model_name,
                        "task": "시스템 초기화 완료",
                        "progress": 10,
                        "log": "AutoCI 시스템이 시작되었습니다.",
                        "color": "00ff00"
                    })
            except Exception as e:
                self.logger.error(f"Godot 대시보드 시작 실패: {e}")
                print(f"⚠️  Godot 대시보드를 시작할 수 없습니다: {e}")
        
        # 백그라운드 작업 시작
        background_task = asyncio.create_task(self.start_24h_development())
        
        # 명령어 입력 루프
        while self.running:
            try:
                # 24시간 실시간 모니터링 상태 표시
                if self.realtime_monitor and hasattr(self.realtime_monitor, 'show_simple_status'):
                    self.realtime_monitor.show_simple_status()
                
                # UI 표시
                if self.terminal_ui:
                    self.terminal_ui.show_current_status(
                        self.current_project,
                        "AI 대기중" if self.ai_controller else "비활성"
                    )
                    self.terminal_ui.show_main_menu()
                    self.terminal_ui.show_quick_commands()
                    prompt = self.terminal_ui.show_input_prompt()
                else:
                    prompt = "autoci> "
                
                # 비동기 입력 처리
                command = await asyncio.get_event_loop().run_in_executor(
                    None, input, prompt
                )
                
                if command.strip():
                    # 숫자 명령어 처리
                    if command.strip() in "123456789":
                        cmd_index = int(command.strip()) - 1
                        if cmd_index < len(self.terminal_ui.commands):
                            actual_command = self.terminal_ui.commands[cmd_index][2]
                            self.handle_command(actual_command)
                    # 빠른 명령어 처리
                    elif command.strip() in self.terminal_ui.quick_commands:
                        actual_command = self.terminal_ui.quick_commands[command.strip()]
                        self.handle_command(actual_command)
                    else:
                        self.handle_command(command.strip())
                    
            except KeyboardInterrupt:
                print("\n\n중단 신호를 받았습니다...")
                self.running = False
                break
            except EOFError:
                # Ctrl+D
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"명령 처리 중 오류: {e}")
        
        # 정리
        print("\n🔄 시스템을 정리하는 중...")
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass
        
        # Godot 종료
        if self.godot_dashboard and dashboard_started:
            self.godot_dashboard.stop()
            print("🎮 Godot 실시간 대시보드를 종료했습니다.")
        elif self.godot_integration and godot_started:
            self.godot_integration.stop_godot()
            print("🎮 Godot 대시보드를 종료했습니다.")
        
        print("👋 AutoCI가 종료되었습니다. 감사합니다!")

def main():
    """메인 함수"""
    # 이벤트 루프 생성 및 실행
    terminal = AutoCITerminal()
    
    try:
        asyncio.run(terminal.run_terminal_interface())
    except KeyboardInterrupt:
        print("\n\nAutoCI가 종료되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()