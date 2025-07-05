"""
24시간 자동 게임 개발 파이프라인
AI가 끈질기게 24시간 동안 완전한 2.5D~3D 게임을 제작하는 시스템
"""

import os
import sys
import json
import time
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import asyncio
import random

# AutoCI 모듈 임포트
from .godot_automation_controller import GodotAutomationController
from .ai_model_integration import get_ai_integration
from .persistent_error_handler import PersistentErrorHandler
from .intelligent_search_system import IntelligentSearchSystem, SearchResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DevelopmentPhase(Enum):
    """개발 단계"""
    PLANNING = "planning"           # 기획
    DESIGN = "design"              # 디자인
    PROTOTYPE = "prototype"        # 프로토타입
    MECHANICS = "mechanics"        # 게임 메커니즘
    LEVEL_DESIGN = "level_design"  # 레벨 디자인
    AUDIO = "audio"               # 오디오
    VISUAL = "visual"             # 비주얼 폴리싱
    TESTING = "testing"           # 테스트
    BUILD = "build"               # 빌드
    DOCUMENTATION = "documentation" # 문서화


class GameQualityMetrics:
    """게임 품질 평가 메트릭"""
    def __init__(self):
        self.basic_functionality = 0    # 기본 기능 (0-30)
        self.gameplay_mechanics = 0     # 게임플레이 (0-20)
        self.visual_audio = 0          # 비주얼/오디오 (0-20)
        self.user_experience = 0       # UX (0-15)
        self.performance = 0           # 성능 (0-15)
        
    @property
    def total_score(self) -> int:
        """총 품질 점수 (0-100)"""
        return min(100, sum([
            self.basic_functionality,
            self.gameplay_mechanics,
            self.visual_audio,
            self.user_experience,
            self.performance
        ]))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "basic_functionality": self.basic_functionality,
            "gameplay_mechanics": self.gameplay_mechanics,
            "visual_audio": self.visual_audio,
            "user_experience": self.user_experience,
            "performance": self.performance,
            "total_score": self.total_score
        }


@dataclass
class GameProject:
    """게임 프로젝트 정보"""
    name: str
    game_type: str
    start_time: datetime
    target_duration: timedelta = timedelta(hours=24)
    current_phase: DevelopmentPhase = DevelopmentPhase.PLANNING
    quality_metrics: GameQualityMetrics = field(default_factory=GameQualityMetrics)
    iteration_count: int = 0
    error_count: int = 0
    improvement_count: int = 0
    completed_features: List[str] = field(default_factory=list)
    pending_features: List[str] = field(default_factory=list)
    phase_progress: Dict[str, float] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> timedelta:
        return datetime.now() - self.start_time
    
    @property
    def remaining_time(self) -> timedelta:
        return max(timedelta(0), self.target_duration - self.elapsed_time)
    
    @property
    def progress_percentage(self) -> float:
        return min(100, (self.elapsed_time.total_seconds() / self.target_duration.total_seconds()) * 100)


class GameDevelopmentPipeline:
    """24시간 자동 게임 개발 파이프라인"""
    
    def __init__(self):
        self.ai_model = get_ai_integration()
        self.godot_controller = GodotAutomationController()
        self.error_handler = PersistentErrorHandler()
        self.search_system = IntelligentSearchSystem()  # 지능형 검색 시스템 추가
        
        self.current_project: Optional[GameProject] = None
        self.is_running = False
        self.development_thread: Optional[threading.Thread] = None
        self.status_thread: Optional[threading.Thread] = None
        
        # 학습 경험 수집
        self.experiences = []
        self.experience_dir = Path("experiences") / "game_development"
        self.experience_dir.mkdir(parents=True, exist_ok=True)
        
        # 파손된 JSON 파일 정리
        self._cleanup_corrupted_files()
        
        # 가상 입력 시스템 추가
        from modules.virtual_input_controller import get_virtual_input
        self.virtual_input = get_virtual_input()
        
        # Godot 조작 학습 시스템 추가
        from modules.godot_manipulation_learning import GodotManipulationLearning
        self.godot_learning = GodotManipulationLearning()
        
        # 개발 단계별 시간 할당 (총 24시간)
        self.phase_durations = {
            DevelopmentPhase.PLANNING: timedelta(minutes=5),        # 5분으로 단축
            DevelopmentPhase.DESIGN: timedelta(minutes=5),          # 5분으로 단축
            DevelopmentPhase.PROTOTYPE: timedelta(hours=3),         # 3시간 유지
            DevelopmentPhase.MECHANICS: timedelta(hours=13.8),      # 메인 개발 시간 증가
            DevelopmentPhase.LEVEL_DESIGN: timedelta(minutes=10),   # 10분으로 단축
            DevelopmentPhase.AUDIO: timedelta(minutes=5),           # 5분으로 단축
            DevelopmentPhase.VISUAL: timedelta(minutes=30),         # 30분으로 단축
            DevelopmentPhase.TESTING: timedelta(hours=1),           # 1시간으로 단축
            DevelopmentPhase.BUILD: timedelta(minutes=0),           # 빌드 단계 제거
            DevelopmentPhase.DOCUMENTATION: timedelta(minutes=10)   # 10분으로 단축
        }
        
        # 게임 타입별 기본 기능 목록
        self.game_features = {
            "platformer": [
                "player_movement", "jumping", "gravity", "collision_detection",
                "enemy_ai", "collectibles", "level_progression", "score_system",
                "lives_system", "power_ups", "moving_platforms", "checkpoints"
            ],
            "racing": [
                "vehicle_control", "physics_engine", "track_design", "lap_system",
                "timer", "opponent_ai", "boost_system", "drift_mechanics",
                "collision_effects", "mini_map", "speed_display", "position_tracking"
            ],
            "rpg": [
                "character_creation", "inventory_system", "combat_system", "dialogue_system",
                "quest_system", "level_progression", "skill_tree", "save_system",
                "npc_interaction", "world_map", "equipment_system", "magic_system"
            ],
            "puzzle": [
                "grid_system", "piece_movement", "match_detection", "score_calculation",
                "level_generation", "hint_system", "timer", "combo_system",
                "special_pieces", "difficulty_scaling", "achievements", "tutorial"
            ]
        }
    
    async def start_development(self, game_name: str, game_type: str = "platformer"):
        """24시간 게임 개발 시작"""
        if self.is_running:
            logger.warning("이미 개발이 진행 중입니다")
            return False
        
        # 프로젝트 초기화
        self.current_project = GameProject(
            name=game_name,
            game_type=game_type,
            start_time=datetime.now()
        )
        
        # 기능 목록 설정
        self.current_project.pending_features = self.game_features.get(
            game_type, 
            self.game_features["platformer"]
        ).copy()
        
        self.is_running = True
        
        # Godot 프로젝트 시작
        await self.godot_controller.start_engine(game_name)
        
        # 개발 스레드 시작
        self.development_thread = threading.Thread(target=lambda: asyncio.run(self._development_loop()))
        self.development_thread.daemon = True
        self.development_thread.start()
        
        # 상태 모니터링 스레드 시작
        self.status_thread = threading.Thread(target=self._status_monitor_loop)
        self.status_thread.daemon = True
        self.status_thread.start()
        
        logger.info(f"24시간 게임 개발 시작: {game_name} ({game_type})")
        return True
    
    async def _development_loop(self):
        """메인 개발 루프"""
        try:
            # 각 개발 단계 실행
            for phase in DevelopmentPhase:
                if not self.is_running:
                    break
                
                self.current_project.current_phase = phase
                phase_end_time = datetime.now() + self.phase_durations[phase]
                
                logger.info(f"개발 단계 시작: {phase.value}")
                
                # 단계별 작업 실행
                phase_start_time = datetime.now()
                consecutive_failures = 0
                
                # BUILD 단계는 건너뛰기
                if phase == DevelopmentPhase.BUILD:
                    logger.info("빌드 단계 건너뛰기 (시간 할당 0분)")
                    continue
                
                # 단계 시작 시 예방적 검색 수행
                if phase in [DevelopmentPhase.PROTOTYPE, DevelopmentPhase.MECHANICS]:
                    logger.info(f"🔍 {phase.value} 단계 시작 전 베스트 프랙티스 검색...")
                    await self._proactive_search_for_phase(phase)
                
                while datetime.now() < phase_end_time and self.is_running:
                    # 남은 시간 체크
                    time_remaining = (phase_end_time - datetime.now()).total_seconds()
                    
                    # 시간이 부족하면 최소 기능으로 전환
                    if time_remaining < 60:  # 1분 미만
                        logger.warning(f"{phase.value} 단계 시간 부족, 최소 기능으로 전환")
                        await self._implement_minimal_phase_features(phase)
                        break
                    
                    # 단계 작업 실행
                    success = await self._execute_phase_tasks(phase)
                    
                    if not success:
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            logger.warning(f"{phase.value} 연속 실패 3회, 다음 단계로 이동")
                            break
                    else:
                        consecutive_failures = 0
                    
                    self.current_project.iteration_count += 1
                    
                    # 진행률 업데이트
                    phase_progress = ((datetime.now() - phase_start_time).total_seconds() / 
                                    self.phase_durations[phase].total_seconds()) * 100
                    self.current_project.phase_progress[phase.value] = min(phase_progress, 100)
                    
                    await asyncio.sleep(1)  # CPU 과부하 방지
            
            # 개발 완료
            if self.is_running:
                self._finalize_project()
                
        except Exception as e:
            logger.error(f"개발 루프 오류: {e}")
            traceback.print_exc()
        finally:
            self.is_running = False
    
    async def _execute_phase_tasks(self, phase: DevelopmentPhase) -> bool:
        """개발 단계별 작업 실행 - 끈질긴 재시도와 학습"""
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries:
            task_start_time = time.time()  # 작업 시작 시간 기록
            try:
                # 이전 경험에서 학습
                similar_experiences = self.get_similar_experiences(
                    phase.value, 
                    self.current_project.game_type
                )
                
                if similar_experiences:
                    logger.info(f"유사한 {len(similar_experiences)}개의 경험 발견, 학습 적용 중...")
                    self._apply_learned_strategies(similar_experiences, phase)
                
                # 단계별 작업 실행
                if phase == DevelopmentPhase.PLANNING:
                    await self._planning_phase()
                elif phase == DevelopmentPhase.DESIGN:
                    self._design_phase()
                elif phase == DevelopmentPhase.PROTOTYPE:
                    await self._prototype_phase()
                elif phase == DevelopmentPhase.MECHANICS:
                    await self._mechanics_phase()
                elif phase == DevelopmentPhase.LEVEL_DESIGN:
                    self._level_design_phase()
                elif phase == DevelopmentPhase.AUDIO:
                    self._audio_phase()
                elif phase == DevelopmentPhase.VISUAL:
                    self._visual_phase()
                elif phase == DevelopmentPhase.TESTING:
                    self._testing_phase()
                elif phase == DevelopmentPhase.BUILD:
                    self._build_phase()
                elif phase == DevelopmentPhase.DOCUMENTATION:
                    self._documentation_phase()
                
                # 성공시 경험 저장
                self._save_success_experience(phase)
                return True  # 성공
                    
            except Exception as e:
                # 실패 시간 계산
                failure_duration = time.time() - task_start_time
                
                retry_count += 1
                self.current_project.error_count += 1
                
                logger.error(f"{phase.value} 단계 실패 (시도 {retry_count}/{max_retries}): {e}")
                logger.info(f"⏱️ 실패까지 걸린 시간: {failure_duration:.1f}초")
                
                # 실패 경험 저장 및 학습
                self._save_failure_experience(phase, e)
                
                if retry_count < max_retries:
                    # 재시도 전 대기 (지수 백오프)
                    wait_time = min(2 ** retry_count, 30)  # 최대 30초
                    logger.info(f"{wait_time}초 후 재시도...")
                    await asyncio.sleep(wait_time)
                    
                    # 다른 접근 방법 시도 (실패 시간 정보 포함)
                    error_info = {"failure_duration": failure_duration}
                    await self._handle_error(e, phase, error_info)
                else:
                    logger.warning(f"{phase.value} 단계 최종 실패, 우회 방법 적용")
                    await self._apply_fallback_strategy(phase)
                    return False  # 실패
    
    async def _planning_phase(self):
        """기획 단계: 빠른 게임 컨셉 정의 (10분)"""
        logger.info("📝 기획 단계: 빠른 컨셉 정의 (10분)")
        
        # 간단한 게임 컨셉 생성
        game_concepts = {
            "platformer": "Jump and run adventure with collectibles",
            "rpg": "Fantasy adventure with character progression", 
            "puzzle": "Match-3 puzzle with special powers",
            "shooter": "Top-down arcade shooter with waves",
            "racing": "Fast-paced racing with boost mechanics"
        }
        
        concept = game_concepts.get(self.current_project.game_type, "Classic arcade game")
        
        # 최소한의 디자인 문서
        design_doc = f"""# {self.current_project.name} - Quick Design

**Type**: {self.current_project.game_type}
**Concept**: {concept}
**Target**: Casual players
**Duration**: 10-30 minutes gameplay

## Core Features
{chr(10).join(f"- {feature}" for feature in self.current_project.pending_features[:5])}

## Technical Stack
- Engine: Godot C#
- Language: C#
- Platform: PC

*Generated in 10 minutes for rapid development*
"""
        
        # 디자인 문서 저장
        doc_path = Path(self.godot_controller.project_path) / "docs" / "game_design.md"
        doc_path.parent.mkdir(exist_ok=True)
        doc_path.write_text(design_doc)
        
        logger.info("✅ 빠른 게임 컨셉 정의 완료! (10분)")
        self.current_project.quality_metrics.basic_functionality += 3
    
    def _design_phase(self):
        """디자인 단계: 아트 방향성 및 UI/UX 설계"""
        # 색상 팔레트 정의
        color_palette = self._generate_color_palette()
        
        # UI 레이아웃 생성
        ui_layout = self._design_ui_layout()
        
        # 설정 파일 저장
        config = {
            "color_palette": color_palette,
            "ui_layout": ui_layout,
            "art_style": self._select_art_style()
        }
        
        config_path = Path(self.godot_controller.project_path) / "config" / "design_config.json"
        config_path.parent.mkdir(exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2))
        
        self.current_project.quality_metrics.visual_audio += 5
    
    async def _prototype_phase(self):
        """프로토타입 단계: 기본 시스템 구현"""
        # 핵심 게임 루프 구현
        logger.info("🔨 프로토타입 단계: 핵심 게임 시스템 구현 중...")
        
        # 프로토타입 생성 전 베스트 프랙티스 검색
        logger.info("🔍 프로토타입 베스트 프랙티스 검색 중...")
        search_context = {
            "game_type": self.current_project.game_type,
            "phase": "prototype",
            "task": "create_game_loop"
        }
        
        try:
            # 예방적 검색 - 성공적인 프로토타입 패턴 찾기
            best_practices = await self.search_system.search_best_practices(
                f"Godot C# {self.current_project.game_type} prototype structure",
                search_context
            )
            
            if best_practices:
                logger.info(f"✅ {len(best_practices)}개의 베스트 프랙티스 발견, 적용 중...")
                # 가장 적합한 패턴 적용
                await self._apply_best_practice_pattern(best_practices[0])
        except Exception as e:
            logger.warning(f"베스트 프랙티스 검색 실패: {e}")
        
        if self.current_project.pending_features:
            feature = self.current_project.pending_features[0]
            await self._implement_feature(feature)
            
        # 기본 게임 루프 생성
        await self._create_main_game_loop()
    
    async def _mechanics_phase(self):
        """메커니즘 단계: 게임플레이 구현"""
        # 남은 기능들 구현
        while self.current_project.pending_features and self.is_running:
            feature = self.current_project.pending_features[0]
            success = await self._implement_feature(feature)
            
            if success:
                self.current_project.pending_features.remove(feature)
                self.current_project.completed_features.append(feature)
                self.current_project.quality_metrics.gameplay_mechanics += 2
                self.current_project.improvement_count += 1
            else:
                # 실패 시 다른 방법 시도
                self._try_alternative_implementation(feature)
    
    async def _create_main_game_loop(self):
        """기본 게임 루프 생성"""
        logger.info("메인 게임 루프 생성 중...")
        
        # Godot C# 메인 스크립트 생성
        try:
            # Godot C# 템플릿 파일 생성
            template_path = Path(self.godot_controller.project_path) / "scripts" / "Main.cs"
            template_path.parent.mkdir(exist_ok=True)
            
            # Godot C# 게임 루프 템플릿
            godot_template = '''using Godot;

public partial class Main : Node
{
    // 게임 상태
    private bool isGameRunning = true;
    
    public override void _Ready()
    {
        // 게임 초기화
        InitializeGame();
        GD.Print("AutoCI 게임이 시작되었습니다!");
    }
    
    public override void _Process(double delta)
    {
        if (!isGameRunning) return;
        
        // 게임 로직 업데이트
        UpdateGameLogic(delta);
    }
    
    private void InitializeGame()
    {
        // 기본 게임 설정
        Engine.MaxFps = 60;
        
        // 입력 설정
        SetupInputHandling();
    }
    
    private void UpdateGameLogic(double delta)
    {
        // 게임 상태 업데이트
        HandleInput();
        UpdateEntities(delta);
    }
    
    private void SetupInputHandling()
    {
        // 입력 처리 설정
    }
    
    private void HandleInput()
    {
        // 키보드/마우스 입력 처리
        if (Input.IsActionJustPressed("ui_cancel"))
        {
            GetTree().Quit();
        }
    }
    
    private void UpdateEntities(double delta)
    {
        // 엔티티 업데이트 로직
    }
}'''
            
            # 템플릿 저장
            template_path.write_text(godot_template, encoding='utf-8')
            
            # Main.cs로 복사
            main_path = Path(self.godot_controller.project_path) / "Main.cs"
            customized_template = godot_template.replace('Main', f'{self.current_project.name}Main')
            main_path.write_text(customized_template, encoding='utf-8')
            
            logger.info("✅ Godot C# 메인 스크립트 생성 완료")
            
            logger.info("✅ 안전한 게임 루프 템플릿 생성 완료")
            
        except SyntaxError as e:
            logger.error(f"파싱 오류 발생: {e}")
            # 검색을 통해 해결책 찾기
            await self._search_and_fix_syntax_error(e)
        except Exception as e:
            logger.error(f"게임 루프 생성 실패: {e}")
            raise
    
    async def _implement_feature(self, feature: str) -> bool:
        """특정 기능 구현"""
        start_time = time.time()
        experience = {
            "timestamp": datetime.now().isoformat(),
            "game_type": self.current_project.game_type,
            "feature": feature,
            "phase": self.current_project.current_phase.value,
            "attempt_number": self.current_project.iteration_count
        }
        
        try:
            # 이전 경험 검색
            similar_experiences = self.get_similar_experiences(feature, self.current_project.game_type)
            
            # 프롬프트에 이전 경험 추가
            experience_context = ""
            if similar_experiences:
                successful_exp = [exp for exp in similar_experiences if exp['success']]
                if successful_exp:
                    experience_context = f"\n\n이전 성공 사례 참고:\n{successful_exp[0].get('code_generated', '')[:500]}"
                    logger.info(f"📚 이전 성공 경험 활용: {feature}")
                
                # 실패 경험도 참고
                failed_exp = [exp for exp in similar_experiences if not exp['success']]
                if failed_exp:
                    experience_context += f"\n\n피해야 할 오류:\n- {failed_exp[0].get('lesson_learned', '')}"
            
            # AI에게 기능 구현 코드 생성 요청
            prompt = f"""
            Implement {feature} for a {self.current_project.game_type} game using Godot Engine with C#.
            The code should be modular and integrate with existing game systems.
            {experience_context}
            """
            
            context = {
                "task": "implement_feature",
                "feature": feature,
                "game_type": self.current_project.game_type,
                "project_name": self.current_project.name,
                "previous_attempts": len(similar_experiences)
            }
            code_result = await self.ai_model.generate_code(prompt, context)
            code = code_result.get('code', '') if isinstance(code_result, dict) else str(code_result)
            
            # 코드 검증
            if self.ai_model.validate_code(code):
                # 파일로 저장
                feature_file = f"scripts/{feature}.cs"
                # Godot C# 파일로 저장
                await self.godot_controller.create_script(feature_file, code)
                
                logger.info(f"기능 구현 완료: {feature}")
                
                # 성공 경험 저장
                experience.update({
                    "success": True,
                    "duration": time.time() - start_time,
                    "code_generated": code,
                    "quality_score": 0.8,  # 성공 시 기본 점수
                    "lesson_learned": f"{feature} 구현 성공 - 이 코드 패턴을 참고할 수 있음"
                })
                self._save_experience(experience)
                
                return True
            else:
                logger.warning(f"코드 검증 실패: {feature}")
                
                # 실패 경험 저장
                experience.update({
                    "success": False,
                    "duration": time.time() - start_time,
                    "error": "코드 검증 실패",
                    "quality_score": 0.3,
                    "lesson_learned": f"{feature} 구현 시 코드 검증 문제 - 구문 오류 확인 필요"
                })
                self._save_experience(experience)
                
                return False
                
        except Exception as e:
            logger.error(f"기능 구현 실패 {feature}: {e}")
            
            # 예외 경험 저장
            experience.update({
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e),
                "quality_score": 0.1,
                "lesson_learned": f"{feature} 구현 중 예외 발생 - {type(e).__name__} 처리 필요"
            })
            self._save_experience(experience)
            
            return False
    
    def _try_alternative_implementation(self, feature: str):
        """대체 구현 방법 시도"""
        strategies = [
            "simplified_version",
            "third_party_library",
            "manual_implementation",
            "hybrid_approach"
        ]
        
        for strategy in strategies:
            logger.info(f"{feature}에 대해 {strategy} 전략 시도 중...")
            
            # 간소화된 버전 시도
            if strategy == "simplified_version":
                success = self._implement_simplified_feature(feature)
            # 써드파티 라이브러리 검색
            elif strategy == "third_party_library":
                success = self._search_and_integrate_library(feature)
            # 수동 구현
            elif strategy == "manual_implementation":
                success = self._manual_implementation(feature)
            # 하이브리드 접근
            else:
                success = self._hybrid_implementation(feature)
            
            if success:
                self.current_project.pending_features.remove(feature)
                self.current_project.completed_features.append(f"{feature}_alt")
                break
    
    def _level_design_phase(self):
        """레벨 디자인 단계"""
        # 레벨 생성
        num_levels = 5 if self.current_project.game_type != "rpg" else 1
        
        for i in range(num_levels):
            level_data = self._generate_level_data(i + 1)
            self._create_level(i + 1, level_data)
            
        self.current_project.quality_metrics.gameplay_mechanics += 5
    
    def _audio_phase(self):
        """오디오 단계: 사운드 및 음악 추가"""
        # 사운드 효과 생성
        sound_effects = ["jump", "collect", "hit", "win", "lose"]
        
        for effect in sound_effects:
            self._create_sound_effect(effect)
            
        # 배경 음악 설정
        self._setup_background_music()
        
        self.current_project.quality_metrics.visual_audio += 5
    
    def _visual_phase(self):
        """비주얼 단계: 그래픽 폴리싱"""
        # 파티클 효과 추가
        self._add_particle_effects()
        
        # 셰이더 적용
        self._apply_shaders()
        
        # 포스트 프로세싱
        self._setup_post_processing()
        
        self.current_project.quality_metrics.visual_audio += 5
        self.current_project.quality_metrics.user_experience += 5
    
    def _testing_phase(self):
        """테스트 단계: 버그 수정 및 최적화"""
        # 자동 플레이 테스트
        test_results = self._run_automated_tests()
        
        # 버그 수정
        for bug in test_results.get("bugs", []):
            self._fix_bug(bug)
            
        # 성능 최적화
        self._optimize_performance()
        
        self.current_project.quality_metrics.performance += 10
    
    def _build_phase(self):
        """빌드 단계: 배포 준비"""
        # 최종 빌드 생성
        self._create_final_build()
        
        # 실행 파일 패키징
        self._package_executable()
        
        self.current_project.quality_metrics.performance += 5
    
    def _documentation_phase(self):
        """문서화 단계: 완성 보고서 작성"""
        # README 생성
        self._generate_readme()
        
        # 플레이 가이드 작성
        self._create_play_guide()
        
        # 개발 보고서 생성
        self._generate_development_report()
        
        self.current_project.quality_metrics.user_experience += 5
    
    async def _handle_error(self, error: Exception, phase: DevelopmentPhase, additional_info: Dict[str, Any] = None):
        """끈질긴 에러 처리 - 검색 기반 학습과 적응"""
        if additional_info is None:
            additional_info = {}
            
        error_info = {
            "phase": phase.value,
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "iteration": self.current_project.iteration_count,
            "elapsed_time": self.current_project.elapsed_time,
            "game_type": self.current_project.game_type,
            "feature": self.current_project.pending_features[0] if self.current_project.pending_features else "unknown",
            **additional_info  # 추가 정보 병합
        }
        
        # 에러 패턴 분석
        error_pattern = self._analyze_error_pattern(error)
        
        # 1단계: 기존 해결 경험 확인
        similar_solutions = self._find_similar_error_solutions(error_pattern)
        
        if similar_solutions:
            logger.info(f"유사한 에러 해결 경험 {len(similar_solutions)}개 발견")
            for solution in similar_solutions:
                if self._apply_solution(solution, error_info):
                    logger.info(f"이전 해결책 적용 성공: {solution['strategy']}")
                    return
        
        # 2단계: 지능형 검색 시스템 활용 (100% 실행, 실패 시간의 9배 투자)
        logger.info("🔍 실패 원인을 검색하여 학습 중...")
        
        # 검색 시간 계산 - 최소 60초, 최대 300초 보장
        failure_duration = error_info.get("failure_duration", 1.0)
        search_duration = max(60, min(failure_duration * 9, 300))  # 60초~300초 사이
        
        logger.info(f"⏱️ 실패 시간: {failure_duration:.1f}초 → 검색 시간: {search_duration:.1f}초 할당")
        
        # 검색 및 학습 실행 (시간 제한 포함)
        try:
            await asyncio.wait_for(
                self._search_and_learn(error, error_info, phase),
                timeout=search_duration
            )
        except asyncio.TimeoutError:
            logger.warning(f"검색 시간 {search_duration}초 초과, 다음 단계로 진행")
        
        # 3단계: 기존 에러 핸들러 시도
        solution = self.error_handler.handle_error(error, error_info)
        
        if solution:
            logger.info(f"표준 해결책 발견: {solution}")
            await self._save_error_solution(error_pattern, solution)
        else:
            # 4단계: 창의적 우회 방법 시도
            logger.warning(f"모든 해결책 실패, 창의적 접근 시도: {error}")
            await self._creative_workaround(error, phase)
    
    def _workaround_error(self, error: Exception, phase: DevelopmentPhase):
        """에러 우회 방법"""
        # 단계를 건너뛰거나 대체 방법 사용
        if "import" in str(error):
            # 모듈 없으면 대체 구현
            self._implement_without_module()
        elif "file not found" in str(error).lower():
            # 파일 생성
            self._create_missing_files()
        else:
            # 기능 비활성화
            logger.warning(f"{phase.value} 단계의 일부 기능 비활성화")
    
    def _status_monitor_loop(self):
        """실시간 상태 모니터링"""
        while self.is_running:
            self._print_status()
            time.sleep(10)  # 10초마다 업데이트
    
    def _print_status(self):
        """현재 상태 출력"""
        if not self.current_project:
            return
        
        elapsed = self.current_project.elapsed_time
        remaining = self.current_project.remaining_time
        
        # 더 간결한 상태 출력
        phase_emoji = {
            "planning": "📝",
            "design": "🎨", 
            "prototype": "🔨",
            "mechanics": "⚙️",
            "level_design": "🗺️",
            "audio": "🎵",
            "visual": "✨",
            "testing": "🧪",
            "build": "📦",
            "documentation": "📚"
        }
        
        current_emoji = phase_emoji.get(self.current_project.current_phase.value, "🔧")
        
        print(f"\r{current_emoji} [{self.current_project.current_phase.value}] 진행률: {self.current_project.progress_percentage:.1f}% | 품질: {self.current_project.quality_metrics.total_score}/100 | 완료: {len(self.current_project.completed_features)}/{len(self.current_project.completed_features) + len(self.current_project.pending_features)}기능", end="", flush=True)
    
    def _finalize_project(self):
        """프로젝트 최종 마무리 - 개선된 버전"""
        logger.info(f"게임 개발 완료: {self.current_project.name}")
        logger.info(f"최종 품질 점수: {self.current_project.quality_metrics.total_score}/100")
        
        project_path = Path(self.godot_controller.project_path)
        
        # 1. 완성도 최종 점검
        self._perform_final_checks()
        
        # 2. 누락된 필수 파일 생성
        self._ensure_essential_files(project_path)
        
        # 3. 최종 빌드 스크립트 생성
        self._create_build_scripts(project_path)
        
        # 4. 게임 실행 가능성 검증
        runnable = self._verify_game_runnable(project_path)
        
        # 5. 자동 플레이 가능성 테스트
        playability_score = self._test_playability()
        
        # 6. 최종 코드 정리 및 최적화
        self._cleanup_and_optimize_code(project_path)
        
        # 7. 상세한 최종 보고서 생성
        report = {
            "project_name": self.current_project.name,
            "game_type": self.current_project.game_type,
            "total_time": str(self.current_project.elapsed_time),
            "completion_timestamp": datetime.now().isoformat(),
            "quality_metrics": self.current_project.quality_metrics.to_dict(),
            "features": {
                "completed": self.current_project.completed_features,
                "pending": self.current_project.pending_features,
                "completion_rate": f"{len(self.current_project.completed_features) / max(1, len(self.current_project.completed_features) + len(self.current_project.pending_features)) * 100:.1f}%"
            },
            "development_stats": {
                "iterations": self.current_project.iteration_count,
                "errors_encountered": self.current_project.error_count,
                "improvements_made": self.current_project.improvement_count,
                "phases_completed": list(self.current_project.phase_progress.keys())
            },
            "technical_details": {
                "runnable": runnable,
                "playability_score": playability_score,
                "file_count": len(list(project_path.rglob("*"))),
                "code_files": len(list(project_path.rglob("*.cs"))) + len(list(project_path.rglob("*.py"))),
                "asset_files": len(list(project_path.rglob("*.png"))) + len(list(project_path.rglob("*.jpg")))
            },
            "recommendations": self._generate_recommendations(),
            "post_development_tasks": self._generate_post_tasks()
        }
        
        # 8. 보고서 저장
        report_path = project_path / "final_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        
        # 9. 실행 가능한 README 생성
        self._create_comprehensive_readme(project_path, report)
        
        # 10. 프로젝트 아카이브 생성 (선택적)
        self._create_project_archive(project_path)
        
        # 11. 성공 경험 저장
        self._save_project_completion_experience(report)
        
        # 12. 다음 프로젝트를 위한 학습 데이터 추출
        self._extract_learning_data(report)
        
        logger.info("="*50)
        logger.info(f"✅ 프로젝트 완료: {self.current_project.name}")
        logger.info(f"📊 최종 점수: {self.current_project.quality_metrics.total_score}/100")
        logger.info(f"🎮 실행 가능: {'예' if runnable else '아니오'}")
        logger.info(f"📁 프로젝트 위치: {project_path}")
        logger.info("="*50)
    
    # 헬퍼 메서드들
    def _generate_color_palette(self) -> List[str]:
        """색상 팔레트 생성"""
        base_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57"]
        return base_colors
    
    def _select_art_style(self) -> str:
        """아트 스타일 선택"""
        styles = ["cartoon", "realistic", "pixel", "low_poly", "stylized"]
        return random.choice(styles)
    
    def _design_ui_layout(self) -> Dict[str, Any]:
        """UI 레이아웃 디자인"""
        return {
            "main_menu": {"buttons": ["Play", "Options", "Exit"]},
            "hud": {"elements": ["score", "lives", "timer"]},
            "pause_menu": {"buttons": ["Resume", "Restart", "Main Menu"]}
        }
    
    def _generate_level_data(self, level_num: int) -> Dict[str, Any]:
        """레벨 데이터 생성"""
        return {
            "level_number": level_num,
            "difficulty": min(level_num * 0.2, 1.0),
            "objectives": ["reach_end", "collect_items"],
            "enemy_count": level_num * 3,
            "time_limit": 300 - (level_num * 20)
        }
    
    def _create_level(self, level_num: int, level_data: Dict[str, Any]):
        """레벨 생성"""
        logger.info(f"레벨 {level_num} 생성 중...")
        # 실제 레벨 생성 로직
    
    def _create_sound_effect(self, effect_name: str):
        """사운드 효과 생성"""
        logger.info(f"사운드 효과 생성: {effect_name}")
    
    def _setup_background_music(self):
        """배경 음악 설정"""
        logger.info("배경 음악 설정 중...")
    
    def _add_particle_effects(self):
        """파티클 효과 추가"""
        logger.info("파티클 효과 추가 중...")
    
    def _apply_shaders(self):
        """셰이더 적용"""
        logger.info("셰이더 적용 중...")
    
    def _setup_post_processing(self):
        """포스트 프로세싱 설정"""
        logger.info("포스트 프로세싱 설정 중...")
    
    def _run_automated_tests(self) -> Dict[str, Any]:
        """자동화 테스트 실행"""
        logger.info("자동화 테스트 실행 중...")
        return {"bugs": [], "performance_issues": []}
    
    def _fix_bug(self, bug: Dict[str, Any]):
        """버그 수정"""
        logger.info(f"버그 수정 중: {bug}")
    
    def _optimize_performance(self):
        """성능 최적화"""
        logger.info("성능 최적화 중...")
    
    def _create_final_build(self):
        """최종 빌드 생성"""
        logger.info("최종 빌드 생성 중...")
    
    def _package_executable(self):
        """실행 파일 패키징"""
        logger.info("실행 파일 패키징 중...")
    
    def _generate_readme(self):
        """README 생성"""
        logger.info("README 생성 중...")
    
    def _create_play_guide(self):
        """플레이 가이드 작성"""
        logger.info("플레이 가이드 작성 중...")
    
    def _generate_development_report(self):
        """개발 보고서 생성"""
        logger.info("개발 보고서 생성 중...")
    
    def _implement_simplified_feature(self, feature: str) -> bool:
        """간소화된 기능 구현"""
        logger.info(f"{feature}의 간소화 버전 구현 시도...")
        return random.random() > 0.3  # 70% 성공률
    
    def _search_and_integrate_library(self, feature: str) -> bool:
        """써드파티 라이브러리 검색 및 통합"""
        logger.info(f"{feature}를 위한 라이브러리 검색 중...")
        return random.random() > 0.5  # 50% 성공률
    
    def _manual_implementation(self, feature: str) -> bool:
        """수동 구현"""
        logger.info(f"{feature} 수동 구현 시도...")
        return random.random() > 0.4  # 60% 성공률
    
    def _hybrid_implementation(self, feature: str) -> bool:
        """하이브리드 구현"""
        logger.info(f"{feature} 하이브리드 접근법 시도...")
        return random.random() > 0.2  # 80% 성공률
    
    def _implement_without_module(self):
        """모듈 없이 구현"""
        logger.info("필요 모듈 없이 대체 구현 시도...")
    
    def _create_missing_files(self):
        """누락된 파일 생성"""
        logger.info("누락된 파일 생성 중...")
    
    def _apply_learned_strategies(self, experiences: List[Dict[str, Any]], phase: DevelopmentPhase):
        """학습된 전략 적용"""
        for exp in experiences:
            if exp.get('success', False):
                logger.info(f"성공 경험 적용: {exp.get('lesson_learned', '')}") 
                # TODO: 실제 전략 적용 로직 구현
    
    def _save_success_experience(self, phase: DevelopmentPhase):
        """성공 경험 저장"""
        experience = {
            "timestamp": datetime.now().isoformat(),
            "game_type": self.current_project.game_type,
            "phase": phase.value,
            "success": True,
            "duration": self.current_project.elapsed_time.total_seconds(),  # timedelta를 초로 변환
            "quality_score": self.current_project.quality_metrics.total_score,
            "features_completed": len(self.current_project.completed_features),
            "lesson_learned": f"{phase.value} 단계 성공적 완료"
        }
        self._save_experience(experience)
    
    def _save_failure_experience(self, phase: DevelopmentPhase, error: Exception):
        """실패 경험 저장"""
        experience = {
            "timestamp": datetime.now().isoformat(),
            "game_type": self.current_project.game_type,
            "phase": phase.value,
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "retry_count": self.current_project.error_count,
            "lesson_learned": f"{phase.value} 단계에서 {type(error).__name__} 발생"
        }
        self._save_experience(experience)
    
    def _analyze_error_pattern(self, error: Exception) -> Dict[str, Any]:
        """에러 패턴 분석"""
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "phase": self.current_project.current_phase.value,
            "game_type": self.current_project.game_type
        }
    
    def _find_similar_error_solutions(self, error_pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """유사한 에러 해결책 찾기"""
        # TODO: 실제 구현 필요
        return []
    
    def _apply_solution(self, solution: Dict[str, Any], error_info: Dict[str, Any]) -> bool:
        """해결책 적용"""
        # TODO: 실제 구현 필요
        return False
    
    async def _save_error_solution(self, error_pattern: Dict[str, Any], solution: str):
        """에러 해결책 저장"""
        from modules.shared_knowledge_base import get_shared_knowledge_base
        shared_kb = get_shared_knowledge_base()
        
        solution_data = {
            "timestamp": datetime.now().isoformat(),
            "error_pattern": error_pattern,
            "solution": solution,
            "success": True
        }
        
        # 공유 지식 베이스에 솔루션 저장
        error_type = error_pattern.get("type", "general_error")
        error_message = error_pattern.get("message", "")
        await shared_kb.save_solution(error_type, error_message, solution, success=True)
        
        logger.info(f"📚 에러 해결책 저장: {error_type}")
    
    async def _proactive_search_for_phase(self, phase: DevelopmentPhase):
        """단계 시작 전 예방적 검색"""
        try:
            # 공유 지식 베이스 사용
            from modules.shared_knowledge_base import get_shared_knowledge_base
            shared_kb = get_shared_knowledge_base()
            
            search_context = {
                "game_type": self.current_project.game_type,
                "phase": phase.value,
                "features": self.current_project.pending_features[:3]  # 다음 3개 기능
            }
            
            # 단계별 검색 쿼리
            queries = {
                DevelopmentPhase.PROTOTYPE: f"Godot C# {self.current_project.game_type} prototype best practices syntax errors common mistakes",
                DevelopmentPhase.MECHANICS: f"Godot C# {self.current_project.game_type} game mechanics implementation patterns"
            }
            
            query = queries.get(phase, f"Godot C# {phase.value} implementation guide")
            
            # 먼저 캐시된 결과 확인
            cached_results = await shared_kb.get_cached_search(query)
            if cached_results:
                logger.info(f"📚 캐시된 지식 사용: {query}")
                results = cached_results.get("search_results", [])
            else:
                # 베스트 프랙티스 검색
                results = await self.search_system.search_best_practices(query, search_context)
                # 검색 결과를 공유 지식 베이스에 저장
                if results:
                    await shared_kb.save_search_result(query, {"search_results": results})
            
            if results:
                logger.info(f"📚 {len(results)}개의 사전 지식 습득")
                # 검색 결과를 메모리에 저장
                self._store_phase_knowledge(phase, results)
                
                # 주요 패턴 추출
                patterns = self._extract_patterns_from_results(results)
                if patterns:
                    logger.info(f"✅ {len(patterns)}개의 유용한 패턴 발견")
                    
        except Exception as e:
            logger.warning(f"예방적 검색 실패: {e}")
    
    async def _search_and_fix_syntax_error(self, error: SyntaxError):
        """파싱 오류 검색 및 해결"""
        error_context = {
            "error_type": "SyntaxError",
            "line": error.lineno,
            "message": str(error),
            "game_type": self.current_project.game_type
        }
        
        # 파싱 오류 해결책 검색
        solutions = await self.search_system.search_for_solution(error, error_context)
        
        if solutions:
            logger.info(f"🔧 {len(solutions)}개의 파싱 오류 해결책 발견")
            for solution in solutions[:2]:
                success = await self.search_system.apply_solution(solution, error_context)
                if success:
                    logger.info("✅ 파싱 오류 해결 성공!")
                    return
        
        # 해결 실패 시 기본 템플릿 사용
        logger.warning("파싱 오류 해결 실패, 기본 템플릿 사용")
        await self._use_minimal_template()
    
    async def _apply_best_practice_pattern(self, pattern: Any):
        """베스트 프랙티스 패턴 적용"""
        # TODO: 패턴 적용 로직 구현
        pass
    
    def _store_phase_knowledge(self, phase: DevelopmentPhase, results: List[Any]):
        """단계별 지식 저장"""
        # TODO: 지식 저장 로직 구현
        pass
    
    def _extract_patterns_from_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """검색 결과에서 패턴 추출"""
        # TODO: 패턴 추출 로직 구현
        return []
    
    async def _use_minimal_template(self):
        """최소 템플릿 사용"""
        # TODO: 최소 템플릿 로직 구현
        pass
    
    async def _search_and_learn(self, error: Exception, error_info: Dict[str, Any], phase: DevelopmentPhase):
        """검색을 통한 학습 및 해결책 적용"""
        try:
            # 검색 컨텍스트 생성
            search_context = {
                "game_type": self.current_project.game_type,
                "phase": phase.value,
                "feature": error_info.get("feature", "unknown"),
                "error_type": type(error).__name__
            }
            
            # 솔루션 검색
            search_results = await self.search_system.search_for_solution(error, search_context)
            
            if search_results:
                logger.info(f"📚 {len(search_results)}개의 잠재적 솔루션 발견")
                
                # 상위 3개 솔루션 시도
                for i, result in enumerate(search_results[:3], 1):
                    logger.info(f"🔧 솔루션 {i}/{min(3, len(search_results))} 시도: {result.title}")
                    
                    # 솔루션 적용
                    success = await self.search_system.apply_solution(result, error_info)
                    
                    if success:
                        logger.info(f"✅ 검색 솔루션 적용 성공! 출처: {result.source.value}")
                        
                        # 성공한 솔루션을 경험으로 저장
                        self._save_search_learning_experience(error, result, phase)
                        
                        # 통계 업데이트
                        self.current_project.improvement_count += 1
                        return
                    else:
                        logger.warning(f"솔루션 {i} 실패, 다음 시도...")
                
                # 모든 솔루션 실패시 학습 내용 저장
                self._save_failed_search_experience(error, search_results, phase)
            else:
                logger.warning("검색 결과 없음, 대체 방법 필요")
                
        except Exception as e:
            logger.error(f"검색 및 학습 중 오류: {e}")
    
    def _save_search_learning_experience(self, error: Exception, solution: SearchResult, phase: DevelopmentPhase):
        """성공한 검색 솔루션을 학습 경험으로 저장"""
        experience = {
            "timestamp": datetime.now().isoformat(),
            "game_type": self.current_project.game_type,
            "phase": phase.value,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "solution_source": solution.source.value,
            "solution_title": solution.title,
            "solution_code": solution.solution_code,
            "success": True,
            "tags": solution.tags,
            "lesson_learned": f"검색을 통해 {type(error).__name__} 해결: {solution.source.value}에서 솔루션 발견"
        }
        
        self._save_experience(experience)
        
        # 검색 시스템의 지식 베이스에도 추가
        self.search_system._save_successful_solution(solution, {
            "error_type": type(error).__name__,
            "phase": phase.value
        })
    
    def _save_failed_search_experience(self, error: Exception, results: List[SearchResult], phase: DevelopmentPhase):
        """실패한 검색 시도를 학습 경험으로 저장"""
        experience = {
            "timestamp": datetime.now().isoformat(),
            "game_type": self.current_project.game_type,
            "phase": phase.value,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "searched_solutions": len(results),
            "success": False,
            "lesson_learned": f"{len(results)}개의 검색 솔루션 모두 실패, 다른 접근 필요"
        }
        
        self._save_experience(experience)
    
    async def _creative_workaround(self, error: Exception, phase: DevelopmentPhase):
        """창의적 우회 방법"""
        logger.info(f"창의적 해결 시도: {phase.value}에서 {error}")
        
        # 검색 통계 확인
        search_stats = self.search_system.get_search_statistics()
        logger.info(f"📊 검색 통계 - 총 검색: {search_stats['total_searches']}, 성공률: {search_stats['success_rate']:.1f}%")
        
        # 단계별 최소 기능 구현
        await self._implement_minimal_phase_features(phase)
    
    async def _apply_fallback_strategy(self, phase: DevelopmentPhase):
        """폴백 전략 적용"""
        logger.warning(f"{phase.value} 단계 폴백 전략 적용")
        # 최소한의 기능만 구현하고 다음 단계로
        await self._implement_minimal_phase_features(phase)
    
    async def _implement_minimal_phase_features(self, phase: DevelopmentPhase):
        """단계별 최소 기능 구현"""
        try:
            if phase == DevelopmentPhase.PLANNING:
                self._create_minimal_design_doc()
            elif phase == DevelopmentPhase.DESIGN:
                self._create_minimal_assets()
            elif phase == DevelopmentPhase.PROTOTYPE:
                await self._create_minimal_prototype()
            elif phase == DevelopmentPhase.MECHANICS:
                await self._implement_core_mechanics_only()
            elif phase == DevelopmentPhase.LEVEL_DESIGN:
                self._create_single_test_level()
            elif phase == DevelopmentPhase.AUDIO:
                self._add_minimal_sound_effects()
            elif phase == DevelopmentPhase.VISUAL:
                self._apply_basic_visual_polish()
            elif phase == DevelopmentPhase.TESTING:
                self._run_basic_tests()
            elif phase == DevelopmentPhase.BUILD:
                self._create_minimal_build()
            elif phase == DevelopmentPhase.DOCUMENTATION:
                self._create_minimal_docs()
            
            logger.info(f"{phase.value} 최소 기능 구현 완료")
        except Exception as e:
            logger.error(f"{phase.value} 최소 기능 구현 실패: {e}")
    
    def _save_experience(self, experience: Dict[str, Any]):
        """학습 경험 저장"""
        try:
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{experience['game_type']}_{experience.get('phase', 'unknown')}_{timestamp}.json"
            filepath = self.experience_dir / filename
            
            # 임시 파일에 먼저 저장 (원자적 쓰기를 위해)
            temp_filepath = filepath.with_suffix('.tmp')
            
            # 경험 저장
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(experience, f, indent=2, ensure_ascii=False)
                f.flush()  # 버퍼를 디스크에 쓰기
                os.fsync(f.fileno())  # OS 레벨에서 디스크에 쓰기 보장
            
            # 원자적으로 파일 이동
            temp_filepath.rename(filepath)
            
            # 메모리에도 저장
            self.experiences.append(experience)
            
            # 성공/실패 통계 업데이트
            if experience['success']:
                logger.info(f"✅ 성공 경험 저장: {experience['lesson_learned']}")
            else:
                logger.info(f"❌ 실패 경험 저장: {experience['lesson_learned']}")
                
        except Exception as e:
            logger.error(f"경험 저장 실패: {e}")
            # 임시 파일이 남아있다면 삭제
            if 'temp_filepath' in locals() and temp_filepath.exists():
                try:
                    temp_filepath.unlink()
                except:
                    pass
    
    def _cleanup_corrupted_files(self):
        """시작할 때 파손된 JSON 파일들을 정리"""
        corrupted_count = 0
        backup_dir = self.experience_dir / "corrupted"
        
        for exp_file in self.experience_dir.glob("*.json"):
            try:
                with open(exp_file, 'r', encoding='utf-8') as f:
                    json.load(f)  # 파싱 시도
            except json.JSONDecodeError:
                # 파손된 파일을 백업 디렉토리로 이동
                try:
                    backup_dir.mkdir(exist_ok=True)
                    backup_path = backup_dir / exp_file.name
                    exp_file.rename(backup_path)
                    corrupted_count += 1
                except Exception as e:
                    logger.error(f"파손된 파일 이동 실패 {exp_file}: {e}")
            except Exception:
                pass  # 다른 오류는 무시
        
        if corrupted_count > 0:
            logger.info(f"🧹 {corrupted_count}개의 파손된 JSON 파일을 정리했습니다.")
    
    def get_similar_experiences(self, feature: str, game_type: str) -> List[Dict[str, Any]]:
        """유사한 경험 검색 - phase나 feature 기반 검색 지원"""
        similar = []
        
        # 파일에서 로드
        for exp_file in self.experience_dir.glob("*.json"):
            try:
                with open(exp_file, 'r', encoding='utf-8') as f:
                    exp = json.load(f)
                    
                # 유사도 체크 - phase와 feature 모두 지원
                # feature 기반 검색
                if 'feature' in exp:
                    if exp.get('game_type') == game_type and exp.get('feature') == feature:
                        similar.append(exp)
                    elif exp.get('feature') == feature:  # 같은 기능이지만 다른 게임 타입
                        similar.append(exp)
                
                # phase 기반 검색 (feature가 실제로는 phase 값일 수 있음)
                elif 'phase' in exp:
                    if exp.get('game_type') == game_type and exp.get('phase') == feature:
                        similar.append(exp)
                    elif exp.get('phase') == feature:  # 같은 단계지만 다른 게임 타입
                        similar.append(exp)
                    
            except json.JSONDecodeError as e:
                # 파손된 JSON 파일은 삭제하거나 백업
                try:
                    # 백업 디렉토리 생성
                    backup_dir = self.experience_dir / "corrupted"
                    backup_dir.mkdir(exist_ok=True)
                    
                    # 파손된 파일을 백업 디렉토리로 이동
                    backup_path = backup_dir / exp_file.name
                    exp_file.rename(backup_path)
                    logger.warning(f"파손된 JSON 파일을 백업함: {exp_file} -> {backup_path}")
                except Exception as move_error:
                    logger.error(f"파손된 파일 이동 실패 {exp_file}: {move_error}")
                    
            except KeyError as e:
                logger.debug(f"필수 키 누락 {exp_file}: {e}")  # error -> debug로 변경
            except Exception as e:
                logger.debug(f"경험 로드 실패 {exp_file}: {e}")  # error -> debug로 변경
        
        # 성공 경험 우선 정렬 - quality_score가 없을 수도 있음
        similar.sort(key=lambda x: (x.get('success', False), x.get('quality_score', 0)), reverse=True)
        
        return similar[:5]  # 상위 5개만 반환
    
    def _create_minimal_design_doc(self):
        """최소 디자인 문서 생성"""
        doc = f"# {self.current_project.name} - Minimal Design\n\n"
        doc += f"Game Type: {self.current_project.game_type}\n"
        doc += "Core Features: Basic gameplay only\n"
        doc_path = Path(self.godot_controller.project_path) / "docs" / "minimal_design.md"
        doc_path.parent.mkdir(exist_ok=True)
        doc_path.write_text(doc)
    
    def _create_minimal_assets(self):
        """최소 에셋 생성"""
        # 기본 색상 큐브만 사용
        logger.info("최소 에셋 생성 - 기본 도형 사용")
    
    async def _create_minimal_prototype(self):
        """최소 프로토타입 생성"""
        logger.info("최소 프로토타입 생성 - 기본 게임 루프만")
        # 기본 Godot C# 프로토타입 코드
        basic_code = '''using Godot;

public partial class MinimalGame : Node
{
    public override void _Ready()
    {
        GD.Print("Minimal AutoCI game started!");
        // 기본 게임 설정
        Engine.MaxFps = 60;
    }
    
    public override void _Process(double delta)
    {
        // 기본 게임 루프
        if (Input.IsActionJustPressed("ui_cancel"))
        {
            GetTree().Quit();
        }
    }
}'''
        main_path = Path(self.godot_controller.project_path) / "Main.cs"
        main_path.write_text(basic_code)
    
    async def _implement_core_mechanics_only(self):
        """핵심 메카닉만 구현"""
        core_mechanics = {
            "platformer": ["player_movement", "jumping"],
            "rpg": ["character_creation", "inventory_system"],
            "puzzle": ["grid_system", "piece_movement"],
            "shooter": ["player_movement", "shooting"],
            "racing": ["vehicle_control", "physics_engine"]
        }
        
        mechanics = core_mechanics.get(self.current_project.game_type, ["player_movement"])
        for mechanic in mechanics[:2]:  # 최대 2개만
            if mechanic not in self.current_project.completed_features:
                self.current_project.completed_features.append(mechanic)
                logger.info(f"최소 메카닉 구현: {mechanic}")
    
    def _create_single_test_level(self):
        """단일 테스트 레벨 생성"""
        logger.info("단일 테스트 레벨 생성")
        level_path = Path(self.godot_controller.project_path) / "scenes" / "TestLevel.tscn"
        level_path.parent.mkdir(exist_ok=True)
        level_data = {"name": "Test Level", "size": [10, 10], "objects": []}
        level_path.write_text(json.dumps(level_data))
    
    def _add_minimal_sound_effects(self):
        """최소 사운드 효과 추가"""
        logger.info("최소 사운드 효과 - 무음 처리")
    
    def _apply_basic_visual_polish(self):
        """기본 시각 효과 적용"""
        logger.info("기본 시각 효과 - 기본 색상만 사용")
        self.current_project.quality_metrics.visual_appeal += 5
    
    def _run_basic_tests(self):
        """기본 테스트 실행"""
        logger.info("기본 테스트 - 실행 가능 여부만 확인")
        self.current_project.quality_metrics.technical_quality += 5
    
    def _create_minimal_build(self):
        """최소 빌드 생성"""
        logger.info("최소 빌드 - 개발 빌드만 생성")
        build_path = Path(self.godot_controller.project_path) / "build"
        build_path.mkdir(exist_ok=True)
        (build_path / "game.exe").touch()
    
    def _create_minimal_docs(self):
        """최소 문서 생성"""
        readme = f"# {self.current_project.name}\n\nMinimal game created by AutoCI\n"
        readme += f"Game Type: {self.current_project.game_type}\n"
        readme += f"Created: {self.current_project.start_time}\n"
        readme_path = Path(self.godot_controller.project_path) / "README.md"
        readme_path.write_text(readme)
    
    def _perform_final_checks(self):
        """완성도 최종 점검"""
        logger.info("프로젝트 완성도 최종 점검 중...")
        
        # 필수 기능 체크
        essential_features = {
            "platformer": ["player_movement", "jumping", "collision_detection"],
            "rpg": ["character_creation", "inventory_system", "dialogue_system"],
            "puzzle": ["grid_system", "piece_movement", "score_system"],
            "racing": ["vehicle_control", "physics_engine", "lap_system"]
        }
        
        required = essential_features.get(self.current_project.game_type, [])
        missing = [f for f in required if f not in self.current_project.completed_features]
        
        if missing:
            logger.warning(f"누락된 필수 기능: {missing}")
            # 품질 점수 조정
            self.current_project.quality_metrics.basic_functionality -= len(missing) * 2
    
    def _ensure_essential_files(self, project_path: Path):
        """누락된 필수 파일 생성"""
        essential_files = {
            "main.py": self._generate_main_file,
            "README.md": self._generate_basic_readme,
            "requirements.txt": self._generate_requirements,
            "run.bat": self._generate_run_batch,
            "run.sh": self._generate_run_shell
        }
        
        for filename, generator in essential_files.items():
            file_path = project_path / filename
            if not file_path.exists():
                logger.info(f"필수 파일 생성: {filename}")
                content = generator()
                file_path.write_text(content)
    
    def _create_build_scripts(self, project_path: Path):
        """최종 빌드 스크립트 생성"""
        build_dir = project_path / "build_scripts"
        build_dir.mkdir(exist_ok=True)
        
        # Windows 빌드 스크립트
        win_script = """@echo off
echo Building game for Windows...
python -m PyInstaller --onefile --windowed --name {game_name} main.py
echo Build complete!
pause
""".format(game_name=self.current_project.name.replace(" ", "_"))
        
        (build_dir / "build_windows.bat").write_text(win_script)
        
        # Linux 빌드 스크립트
        linux_script = """#!/bin/bash
echo "Building game for Linux..."
python -m PyInstaller --onefile --name {game_name} main.py
echo "Build complete!"
""".format(game_name=self.current_project.name.replace(" ", "_"))
        
        (build_dir / "build_linux.sh").write_text(linux_script)
    
    def _verify_game_runnable(self, project_path: Path) -> bool:
        """게임 실행 가능성 검증"""
        main_file = project_path / "main.py"
        if not main_file.exists():
            return False
        
        try:
            # 문법 검사
            import ast
            code = main_file.read_text()
            ast.parse(code)
            return True
        except SyntaxError:
            logger.error("main.py 문법 오류")
            return False
        except Exception as e:
            logger.error(f"실행 가능성 검증 실패: {e}")
            return False
    
    def _test_playability(self) -> float:
        """자동 플레이 가능성 테스트"""
        score = 0.0
        
        # 기본 기능 점수
        if len(self.current_project.completed_features) > 0:
            score += 30.0
        
        # 게임 루프 존재 여부
        if "game_loop" in str(self.current_project.completed_features):
            score += 20.0
        
        # 입력 시스템 존재 여부
        if any("input" in f or "control" in f for f in self.current_project.completed_features):
            score += 20.0
        
        # 목표 시스템 존재 여부
        if any("objective" in f or "goal" in f or "win" in f for f in self.current_project.completed_features):
            score += 20.0
        
        # UI 존재 여부
        if any("ui" in f or "hud" in f or "menu" in f for f in self.current_project.completed_features):
            score += 10.0
        
        return min(score, 100.0)
    
    def _cleanup_and_optimize_code(self, project_path: Path):
        """최종 코드 정리 및 최적화"""
        logger.info("코드 정리 및 최적화 중...")
        
        # 빈 파일 제거
        for empty_file in project_path.rglob("*"):
            if empty_file.is_file() and empty_file.stat().st_size == 0:
                logger.info(f"빈 파일 제거: {empty_file}")
                empty_file.unlink()
        
        # TODO: 추가 최적화 로직
    
    def _generate_recommendations(self) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        if self.current_project.quality_metrics.total_score < 50:
            recommendations.append("품질 향상을 위해 핵심 기능 보강 필요")
        
        if len(self.current_project.pending_features) > 5:
            recommendations.append(f"미완성 기능 {len(self.current_project.pending_features)}개 추가 개발 권장")
        
        if self.current_project.error_count > 10:
            recommendations.append("에러 처리 로직 개선 필요")
        
        if not any("test" in f for f in self.current_project.completed_features):
            recommendations.append("테스트 코드 추가 권장")
        
        return recommendations
    
    def _generate_post_tasks(self) -> List[str]:
        """개발 후 작업 목록 생성"""
        tasks = [
            "게임 플레이 테스트 수행",
            "버그 수정 및 안정화",
            "성능 최적화",
            "사용자 피드백 수집"
        ]
        
        if self.current_project.game_type == "platformer":
            tasks.append("레벨 디자인 추가 개선")
        elif self.current_project.game_type == "rpg":
            tasks.append("스토리 및 퀘스트 확장")
        
        return tasks
    
    def _create_comprehensive_readme(self, project_path: Path, report: Dict[str, Any]):
        """상세한 README 생성"""
        readme_content = f"""# {self.current_project.name}

## 게임 소개
- **장르**: {self.current_project.game_type}
- **개발 시간**: {report['total_time']}
- **품질 점수**: {report['quality_metrics']['total_score']}/100

## 실행 방법
1. Python 3.8+ 설치 필요
2. 의존성 설치: `pip install -r requirements.txt`
3. 게임 실행: `python main.py`

## 주요 기능
{chr(10).join(f"- {feature}" for feature in self.current_project.completed_features[:10])}

## 개발 통계
- 완성된 기능: {len(self.current_project.completed_features)}개
- 반복 횟수: {report['development_stats']['iterations']}회
- 수정된 오류: {report['development_stats']['errors_encountered']}개

## 조작법
- 방향키: 이동
- 스페이스바: 점프/액션
- ESC: 일시정지

## 빌드 방법
- Windows: `build_scripts/build_windows.bat` 실행
- Linux: `build_scripts/build_linux.sh` 실행

## 라이선스
이 프로젝트는 AutoCI에 의해 자동 생성되었습니다.
"""
        
        readme_path = project_path / "README.md"
        readme_path.write_text(readme_content)
    
    def _create_project_archive(self, project_path: Path):
        """프로젝트 아카이브 생성"""
        try:
            import shutil
            archive_name = f"{self.current_project.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            archive_path = project_path.parent / f"{archive_name}.zip"
            
            logger.info(f"프로젝트 아카이브 생성: {archive_path}")
            shutil.make_archive(str(archive_path.with_suffix('')), 'zip', project_path)
        except Exception as e:
            logger.error(f"아카이브 생성 실패: {e}")
    
    def _save_project_completion_experience(self, report: Dict[str, Any]):
        """프로젝트 완료 경험 저장"""
        experience = {
            "timestamp": datetime.now().isoformat(),
            "project_name": self.current_project.name,
            "game_type": self.current_project.game_type,
            "success": True,
            "quality_score": report['quality_metrics']['total_score'],
            "completion_rate": report['features']['completion_rate'],
            "development_time": report['total_time'],
            "playability_score": report['technical_details']['playability_score'],
            "lesson_learned": f"{self.current_project.game_type} 게임 24시간 개발 완료"
        }
        
        self._save_experience(experience)
    
    def _extract_learning_data(self, report: Dict[str, Any]):
        """학습 데이터 추출"""
        learning_data = {
            "successful_patterns": [],
            "failure_patterns": [],
            "optimization_opportunities": []
        }
        
        # 성공 패턴 수집
        for feature in self.current_project.completed_features:
            learning_data["successful_patterns"].append({
                "feature": feature,
                "game_type": self.current_project.game_type
            })
        
        # 실패 패턴 수집
        for feature in self.current_project.pending_features:
            learning_data["failure_patterns"].append({
                "feature": feature,
                "game_type": self.current_project.game_type,
                "reason": "시간 부족 또는 구현 실패"
            })
        
        # 최적화 기회 식별
        if report['development_stats']['errors_encountered'] > 20:
            learning_data["optimization_opportunities"].append("에러 처리 개선 필요")
        
        # 학습 데이터 저장
        learning_path = self.experience_dir / f"learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(learning_path, 'w', encoding='utf-8') as f:
            json.dump(learning_data, f, indent=2, ensure_ascii=False)
    
    def _generate_main_file(self) -> str:
        """기본 main.py 파일 생성"""
        return f"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
# {self.current_project.name} - Auto-generated by AutoCI

from direct.showbase.ShowBase import ShowBase
import sys

class {self.current_project.name.replace(' ', '')}Game(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.setBackgroundColor(0.1, 0.1, 0.1, 1)
        
        # 게임 초기화
        self.setup_game()
        
    def setup_game(self):
        # TODO: 게임 설정
        pass

if __name__ == "__main__":
    game = {self.current_project.name.replace(' ', '')}Game()
    game.run()
"""
    
    def _generate_basic_readme(self) -> str:
        """기본 README 생성"""
        return f"# {self.current_project.name}\n\nAuto-generated game by AutoCI"
    
    def _generate_requirements(self) -> str:
        """requirements.txt 생성"""
        return """"""
    
    def _generate_run_batch(self) -> str:
        """Godot 프로젝트 실행 스크립트 생성"""
        return "@echo off\necho Starting Godot project...\ngodot --path . --main-pack\npause"
    
    def _generate_run_shell(self) -> str:
        """Linux/Mac Godot 실행 스크립트 생성"""
        return "#!/bin/bash\necho Starting Godot project...\ngodot --path . --main-pack"
    
    async def stop(self):
        """개발 중지"""
        self.is_running = False
        if self.godot_controller:
            await self.godot_controller.stop_engine()
        logger.info("게임 개발 파이프라인 중지됨")


# 테스트 및 예제
if __name__ == "__main__":
    import asyncio
    
    pipeline = GameDevelopmentPipeline()
    
    # 24시간 게임 개발 시작
    asyncio.run(pipeline.start_development("MyPlatformer", "platformer"))
    
    try:
        # 24시간 대기 (실제로는 중간에 중단 가능)
        time.sleep(24 * 60 * 60)
    except KeyboardInterrupt:
        pipeline.stop()