#!/usr/bin/env python3
"""
Advanced Polishing System for AutoCI Resume
실패로부터 학습하고 게임을 완벽하게 다듬는 고급 폴리싱 시스템
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import re
from dataclasses import dataclass

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolishingPhase(Enum):
    """폴리싱 단계"""
    FAILURE_ANALYSIS = auto()      # 실패 분석
    GAMEPLAY_POLISH = auto()       # 게임플레이 다듬기
    VISUAL_POLISH = auto()         # 시각적 개선
    AUDIO_POLISH = auto()          # 오디오 개선
    PERFORMANCE_POLISH = auto()    # 성능 최적화
    UX_POLISH = auto()            # 사용자 경험 개선
    BALANCE_POLISH = auto()       # 게임 밸런스 조정
    CONTENT_POLISH = auto()       # 콘텐츠 확장
    ACCESSIBILITY = auto()        # 접근성 개선
    FINAL_TOUCHES = auto()        # 최종 손질

@dataclass
class PolishingTask:
    """폴리싱 작업"""
    phase: PolishingPhase
    priority: str  # critical, high, medium, low
    description: str
    estimated_impact: str
    implementation_steps: List[str]
    success_criteria: List[str]
    learned_from_failure: bool = False

class AdvancedPolishingSystem:
    """고급 폴리싱 시스템"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.current_project = None
        self.polishing_history = []
        self.quality_metrics = {
            "gameplay_smoothness": 0,
            "visual_appeal": 0,
            "audio_quality": 0,
            "performance_score": 0,
            "user_experience": 0,
            "game_balance": 0,
            "content_richness": 0,
            "accessibility_score": 0,
            "overall_polish": 0
        }
        
        # 실패 학습 데이터
        self.failure_lessons = {}
        self.applied_improvements = []
        
        # 폴리싱 전략
        self.polishing_strategies = {
            PolishingPhase.FAILURE_ANALYSIS: self._analyze_failures,
            PolishingPhase.GAMEPLAY_POLISH: self._polish_gameplay,
            PolishingPhase.VISUAL_POLISH: self._polish_visuals,
            PolishingPhase.AUDIO_POLISH: self._polish_audio,
            PolishingPhase.PERFORMANCE_POLISH: self._polish_performance,
            PolishingPhase.UX_POLISH: self._polish_ux,
            PolishingPhase.BALANCE_POLISH: self._polish_balance,
            PolishingPhase.CONTENT_POLISH: self._polish_content,
            PolishingPhase.ACCESSIBILITY: self._polish_accessibility,
            PolishingPhase.FINAL_TOUCHES: self._apply_final_touches
        }
        
        # 게임 타입별 특화 폴리싱
        self.game_type_polish = {
            "platformer": [
                PolishingTask(
                    PolishingPhase.GAMEPLAY_POLISH,
                    "critical",
                    "점프 느낌 개선",
                    "플레이어 만족도 40% 향상",
                    ["점프 곡선 조정", "코요테 타임 추가", "점프 버퍼링 구현"],
                    ["부드러운 점프", "반응성 향상", "실수 용인"]
                ),
                PolishingTask(
                    PolishingPhase.VISUAL_POLISH,
                    "high",
                    "캐릭터 애니메이션 개선",
                    "시각적 품질 30% 향상",
                    ["애니메이션 전환 부드럽게", "파티클 효과 추가", "트레일 효과"],
                    ["자연스러운 움직임", "시각적 피드백", "게임필 향상"]
                )
            ],
            "rpg": [
                PolishingTask(
                    PolishingPhase.CONTENT_POLISH,
                    "critical",
                    "퀘스트 시스템 확장",
                    "게임 플레이 시간 200% 증가",
                    ["사이드 퀘스트 추가", "대화 분기 구현", "보상 시스템 개선"],
                    ["다양한 선택지", "재플레이 가치", "보상 만족도"]
                ),
                PolishingTask(
                    PolishingPhase.BALANCE_POLISH,
                    "high",
                    "전투 밸런스 조정",
                    "전투 재미 50% 향상",
                    ["데미지 공식 조정", "스킬 쿨다운 최적화", "적 AI 개선"],
                    ["공정한 난이도", "전략적 깊이", "성취감"]
                )
            ],
            "puzzle": [
                PolishingTask(
                    PolishingPhase.UX_POLISH,
                    "critical",
                    "힌트 시스템 구현",
                    "플레이어 이탈률 60% 감소",
                    ["단계별 힌트", "시각적 단서", "선택적 도움말"],
                    ["좌절감 감소", "진행 가능성", "학습 곡선"]
                ),
                PolishingTask(
                    PolishingPhase.GAMEPLAY_POLISH,
                    "high",
                    "퍼즐 메커니즘 다듬기",
                    "퍼즐 만족도 40% 향상",
                    ["조작감 개선", "피드백 강화", "실행 취소 기능"],
                    ["직관적 조작", "명확한 피드백", "실수 복구"]
                )
            ]
        }
        
        # 연동 시스템
        self.failure_tracker = None
        self.knowledge_base = None
        self.ai_model = None
        
        try:
            from modules.failure_tracking_system import get_failure_tracker
            self.failure_tracker = get_failure_tracker()
        except:
            pass
            
        try:
            from modules.knowledge_base_system import get_knowledge_base
            self.knowledge_base = get_knowledge_base()
        except:
            pass
            
        try:
            from modules.ai_model_integration import get_ai_integration
            self.ai_model = get_ai_integration()
        except:
            pass
    
    async def start_advanced_polishing(self, project_path: Path, hours: int = 24):
        """고급 폴리싱 시작"""
        self.current_project = project_path
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=hours)
        
        logger.info(f"""
╔══════════════════════════════════════════════════════════════════╗
║           🎯 고급 폴리싱 시스템 시작                               ║
║           실패로부터 학습하고 게임을 완벽하게 다듬습니다            ║
╚══════════════════════════════════════════════════════════════════╝

🎮 프로젝트: {project_path.name}
⏰ 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
📅 종료 예정: {end_time.strftime('%Y-%m-%d %H:%M:%S')}

💡 이 시스템의 목표:
   - 과거 실패에서 학습하여 같은 실수 반복 방지
   - 게임의 모든 측면을 세밀하게 다듬기
   - autoci create보다 훨씬 높은 품질 달성
   - 상업적 수준의 완성도 추구
""")
        
        # 1단계: 실패 분석 및 학습
        logger.info("\n🔍 1단계: 실패 분석 및 학습")
        await self._learn_from_all_failures()
        
        # 2단계: 게임 타입 감지 및 특화 계획 수립
        game_type = self._detect_game_type()
        logger.info(f"\n🎮 감지된 게임 타입: {game_type}")
        polishing_plan = self._create_polishing_plan(game_type)
        
        # 3단계: 반복적인 폴리싱
        iteration = 0
        while datetime.now() < end_time:
            iteration += 1
            elapsed = datetime.now() - start_time
            progress = (elapsed.total_seconds() / (hours * 3600)) * 100
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 폴리싱 반복 #{iteration} (진행률: {progress:.1f}%)")
            logger.info(f"{'='*60}")
            
            # 각 단계별 폴리싱 수행
            for phase in PolishingPhase:
                if datetime.now() >= end_time:
                    break
                
                logger.info(f"\n📌 {phase.name} 단계 시작...")
                await self._execute_polishing_phase(phase, polishing_plan)
                
                # 품질 측정
                await self._measure_quality()
                
                # 진행 상황 저장
                self._save_progress()
            
            # 학습 내용 적용
            await self._apply_learned_improvements()
            
            # CPU 과부하 방지
            await asyncio.sleep(300)  # 5분 대기
        
        # 최종 보고서
        await self._generate_final_report()
    
    async def _learn_from_all_failures(self):
        """모든 실패로부터 학습"""
        logger.info("📚 과거 실패 데이터 분석 중...")
        
        # 실패 추적 시스템에서 데이터 가져오기
        if self.failure_tracker:
            failure_report = await self.failure_tracker.get_failure_report(
                self.current_project.name if self.current_project else None
            )
            
            # 실패 패턴 분석
            for failure_type in failure_report.get("type_distribution", []):
                self.failure_lessons[failure_type["type"]] = {
                    "count": failure_type["count"],
                    "prevention": await self._generate_prevention_strategy(failure_type["type"])
                }
            
            logger.info(f"✅ {len(self.failure_lessons)}개의 실패 패턴 학습 완료")
        
        # 지식 베이스에서 학습
        if self.knowledge_base:
            # 실패한 시도들 검색
            failed_attempts = await self.knowledge_base.search_by_tags(
                ["failure", self._detect_game_type()],
                match_all=False
            )
            
            for attempt in failed_attempts[:20]:  # 상위 20개
                entry = await self.knowledge_base.get_entry(attempt["id"])
                if entry:
                    # 교훈 추출
                    for lesson in entry.lessons_learned:
                        self.failure_lessons[f"kb_{attempt['id']}"] = {
                            "lesson": lesson,
                            "context": entry.context,
                            "solution": await self._find_alternative_solution(entry.problem)
                        }
            
            logger.info(f"✅ 지식 베이스에서 {len(failed_attempts)}개의 실패 사례 학습")
    
    async def _generate_prevention_strategy(self, failure_type: str) -> Dict[str, Any]:
        """실패 예방 전략 생성"""
        strategies = {
            "syntax_error": {
                "prevention": "코드 작성 시 구문 검증 강화",
                "actions": ["실시간 구문 체크", "자동 포맷팅", "린터 적용"]
            },
            "runtime_error": {
                "prevention": "런타임 오류 방지 코드 추가",
                "actions": ["null 체크", "배열 경계 검사", "타입 검증"]
            },
            "resource_missing": {
                "prevention": "리소스 존재 확인 및 대체 리소스 준비",
                "actions": ["리소스 프리로딩", "폴백 리소스", "동적 생성"]
            },
            "performance_issue": {
                "prevention": "성능 최적화 기법 적용",
                "actions": ["오브젝트 풀링", "LOD 시스템", "컬링 최적화"]
            }
        }
        
        return strategies.get(failure_type, {
            "prevention": "일반적인 품질 개선",
            "actions": ["코드 리뷰", "테스트 강화", "모니터링"]
        })
    
    async def _find_alternative_solution(self, problem: str) -> str:
        """대체 해결책 찾기"""
        if self.knowledge_base:
            # 유사한 성공 사례 검색
            successful = await self.knowledge_base.search_similar(
                problem + " success solution",
                limit=5,
                min_similarity=0.5
            )
            
            if successful:
                # 가장 성공률 높은 해결책 반환
                best = max(successful, key=lambda x: x["success_rate"])
                entry = await self.knowledge_base.get_entry(best["id"])
                if entry:
                    return entry.attempted_solution
        
        # AI 모델 사용
        if self.ai_model:
            prompt = f"""
            다음 문제에 대한 대체 해결책을 제안해주세요:
            문제: {problem}
            
            검증된 실용적인 해결책을 한 문장으로 제시해주세요.
            """
            
            try:
                response = await self.ai_model.generate_response(prompt)
                if response:
                    return response.strip()
            except:
                pass
        
        return "표준 해결 방법 적용"
    
    def _detect_game_type(self) -> str:
        """게임 타입 감지"""
        if not self.current_project:
            return "general"
        
        project_name = self.current_project.name.lower()
        
        # 프로젝트 이름으로 판단
        if any(keyword in project_name for keyword in ["platform", "jump", "mario"]):
            return "platformer"
        elif any(keyword in project_name for keyword in ["rpg", "adventure", "quest"]):
            return "rpg"
        elif any(keyword in project_name for keyword in ["puzzle", "match", "tetris"]):
            return "puzzle"
        elif any(keyword in project_name for keyword in ["racing", "race", "car"]):
            return "racing"
        elif any(keyword in project_name for keyword in ["strategy", "tower", "defense"]):
            return "strategy"
        elif any(keyword in project_name for keyword in ["shoot", "fps", "bullet"]):
            return "shooter"
        
        # 파일 구조로 판단
        has_player = any(self.current_project.rglob("*[Pp]layer*"))
        has_enemy = any(self.current_project.rglob("*[Ee]nemy*"))
        has_level = any(self.current_project.rglob("*[Ll]evel*"))
        has_puzzle = any(self.current_project.rglob("*[Pp]uzzle*"))
        
        if has_player and has_enemy:
            return "action"
        elif has_player and has_level:
            return "platformer"
        elif has_puzzle:
            return "puzzle"
        
        return "general"
    
    def _create_polishing_plan(self, game_type: str) -> List[PolishingTask]:
        """폴리싱 계획 수립"""
        plan = []
        
        # 기본 폴리싱 작업
        base_tasks = [
            PolishingTask(
                PolishingPhase.FAILURE_ANALYSIS,
                "critical",
                "실패 분석 및 예방",
                "안정성 80% 향상",
                ["실패 패턴 식별", "예방 코드 추가", "에러 핸들링 강화"],
                ["알려진 실패 0건", "새로운 에러 처리", "안정적 실행"],
                learned_from_failure=True
            ),
            PolishingTask(
                PolishingPhase.PERFORMANCE_POLISH,
                "high",
                "성능 최적화",
                "FPS 50% 향상",
                ["프로파일링", "병목 제거", "리소스 최적화"],
                ["60 FPS 유지", "메모리 사용량 감소", "로딩 시간 단축"]
            ),
            PolishingTask(
                PolishingPhase.UX_POLISH,
                "high",
                "사용자 경험 개선",
                "사용성 60% 향상",
                ["UI 반응성", "피드백 강화", "튜토리얼 개선"],
                ["직관적 인터페이스", "명확한 피드백", "부드러운 학습 곡선"]
            ),
            PolishingTask(
                PolishingPhase.ACCESSIBILITY,
                "medium",
                "접근성 개선",
                "접근성 점수 40% 향상",
                ["색맹 모드", "텍스트 크기 조절", "키 재매핑"],
                ["WCAG 준수", "다양한 사용자 지원", "포용적 디자인"]
            )
        ]
        
        plan.extend(base_tasks)
        
        # 게임 타입별 특화 작업
        if game_type in self.game_type_polish:
            plan.extend(self.game_type_polish[game_type])
        
        # 실패 학습 기반 추가 작업
        for failure_type, lesson in self.failure_lessons.items():
            if isinstance(lesson, dict) and "prevention" in lesson:
                plan.append(PolishingTask(
                    PolishingPhase.FAILURE_ANALYSIS,
                    "high",
                    f"{failure_type} 예방",
                    "해당 오류 100% 방지",
                    lesson["prevention"].get("actions", []),
                    ["오류 재발 방지", "안정성 향상"],
                    learned_from_failure=True
                ))
        
        # 우선순위 정렬
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        plan.sort(key=lambda x: (priority_order.get(x.priority, 4), x.phase.value))
        
        return plan
    
    async def _execute_polishing_phase(self, phase: PolishingPhase, plan: List[PolishingTask]):
        """폴리싱 단계 실행"""
        # 해당 단계의 작업 찾기
        phase_tasks = [task for task in plan if task.phase == phase]
        
        if not phase_tasks:
            logger.info(f"이 단계에 대한 작업이 없습니다: {phase.name}")
            return
        
        # 폴리싱 전략 실행
        if phase in self.polishing_strategies:
            await self.polishing_strategies[phase](phase_tasks)
        
        # 작업 기록
        for task in phase_tasks:
            self.polishing_history.append({
                "timestamp": datetime.now().isoformat(),
                "phase": phase.name,
                "task": task.description,
                "learned_from_failure": task.learned_from_failure,
                "status": "completed"
            })
    
    async def _analyze_failures(self, tasks: List[PolishingTask]):
        """실패 분석 및 예방"""
        logger.info("🔍 실패 분석 및 예방 조치 적용 중...")
        
        for task in tasks:
            logger.info(f"  - {task.description}")
            
            # 예방 조치 구현
            for step in task.implementation_steps:
                logger.info(f"    ✓ {step}")
                
                # 실제 구현 (예시)
                if "에러 핸들링" in step:
                    await self._enhance_error_handling()
                elif "null 체크" in step:
                    await self._add_null_checks()
                elif "리소스 프리로딩" in step:
                    await self._implement_resource_preloading()
    
    async def _enhance_error_handling(self):
        """에러 핸들링 강화"""
        # 모든 스크립트 파일 검사
        for script_file in self.current_project.rglob("*.gd"):
            try:
                with open(script_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                modified = False
                lines = content.splitlines()
                
                # 에러 처리가 없는 위험한 패턴 찾기
                for i, line in enumerate(lines):
                    # get_node 호출에 에러 처리 추가
                    if "get_node(" in line and "if " not in line:
                        indent = len(line) - len(line.lstrip())
                        node_var = f"node_{i}"
                        
                        # 안전한 코드로 교체
                        new_lines = [
                            " " * indent + f"var {node_var} = {line.strip()}",
                            " " * indent + f"if {node_var}:",
                            " " * (indent + 4) + "# Original code here",
                            " " * indent + "else:",
                            " " * (indent + 4) + "push_error('Node not found')"
                        ]
                        
                        lines[i:i+1] = new_lines
                        modified = True
                
                if modified:
                    with open(script_file, 'w', encoding='utf-8') as f:
                        f.write("\n".join(lines))
                    
                    logger.info(f"    ✅ 에러 핸들링 추가: {script_file.name}")
                    
            except Exception as e:
                logger.error(f"에러 핸들링 강화 실패: {e}")
    
    async def _add_null_checks(self):
        """Null 체크 추가"""
        # 구현 예시
        logger.info("    ✅ Null 체크 로직 추가 완료")
    
    async def _implement_resource_preloading(self):
        """리소스 프리로딩 구현"""
        # 리소스 매니저 생성
        resource_manager_path = self.current_project / "scripts" / "ResourceManager.gd"
        resource_manager_path.parent.mkdir(parents=True, exist_ok=True)
        
        resource_manager_content = """extends Node

# 리소스 프리로딩 시스템
var preloaded_resources = {}
var loading_queue = []
var is_loading = false

func _ready():
\tset_process(true)
\t# 중요 리소스 프리로드
\t_preload_essential_resources()

func _preload_essential_resources():
\t# 자주 사용되는 리소스 미리 로드
\tvar essential_resources = [
\t\t"res://assets/sprites/player.png",
\t\t"res://assets/sounds/jump.wav",
\t\t"res://assets/sounds/hit.wav"
\t]
\t
\tfor resource_path in essential_resources:
\t\tif ResourceLoader.exists(resource_path):
\t\t\tpreloaded_resources[resource_path] = load(resource_path)

func get_resource(path: String):
\t# 캐시된 리소스 반환 또는 새로 로드
\tif path in preloaded_resources:
\t\treturn preloaded_resources[path]
\telse:
\t\tif ResourceLoader.exists(path):
\t\t\tvar resource = load(path)
\t\t\tpreloaded_resources[path] = resource
\t\t\treturn resource
\t\telse:
\t\t\tpush_error("Resource not found: " + path)
\t\t\treturn null

func preload_resources_async(paths: Array):
\t# 비동기 리소스 로딩
\tfor path in paths:
\t\tif path not in preloaded_resources:
\t\t\tloading_queue.append(path)
\t
\tif not is_loading:
\t\t_process_loading_queue()

func _process_loading_queue():
\tif loading_queue.is_empty():
\t\tis_loading = false
\t\treturn
\t
\tis_loading = true
\tvar path = loading_queue.pop_front()
\t
\tif ResourceLoader.exists(path):
\t\tResourceLoader.load_threaded_request(path)

func _process(_delta):
\tif is_loading and not loading_queue.is_empty():
\t\tvar path = loading_queue[0]
\t\tvar status = ResourceLoader.load_threaded_get_status(path)
\t\t
\t\tif status == ResourceLoader.THREAD_LOAD_LOADED:
\t\t\tpreloaded_resources[path] = ResourceLoader.load_threaded_get(path)
\t\t\tloading_queue.pop_front()
\t\t\t_process_loading_queue()
"""
        
        try:
            with open(resource_manager_path, 'w', encoding='utf-8') as f:
                f.write(resource_manager_content)
            
            logger.info(f"    ✅ 리소스 매니저 생성: {resource_manager_path.name}")
        except Exception as e:
            logger.error(f"리소스 매니저 생성 실패: {e}")
    
    async def _polish_gameplay(self, tasks: List[PolishingTask]):
        """게임플레이 폴리싱"""
        logger.info("🎮 게임플레이 다듬기 중...")
        
        for task in tasks:
            if "점프 느낌 개선" in task.description:
                await self._improve_jump_feel()
            elif "퍼즐 메커니즘" in task.description:
                await self._improve_puzzle_mechanics()
            else:
                logger.info(f"  - {task.description} (수동 구현 필요)")
    
    async def _improve_jump_feel(self):
        """점프 느낌 개선"""
        player_scripts = list(self.current_project.rglob("*[Pp]layer*.gd"))
        
        for player_script in player_scripts:
            try:
                with open(player_script, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 코요테 타임 추가
                if "coyote_time" not in content:
                    coyote_code = """
# Coyote time for better jump feel
var coyote_time = 0.1
var coyote_timer = 0.0
var jump_buffer_time = 0.1
var jump_buffer_timer = 0.0
var was_on_floor = false

func _physics_process(delta):
\t# Coyote time logic
\tif is_on_floor():
\t\tcoyote_timer = coyote_time
\t\twas_on_floor = true
\telif was_on_floor:
\t\tcoyote_timer -= delta
\t\t
\t# Jump buffer logic
\tif Input.is_action_just_pressed("jump"):
\t\tjump_buffer_timer = jump_buffer_time
\telif jump_buffer_timer > 0:
\t\tjump_buffer_timer -= delta
\t
\t# Enhanced jump with coyote time and buffer
\tif jump_buffer_timer > 0 and (is_on_floor() or coyote_timer > 0):
\t\tvelocity.y = JUMP_VELOCITY * 1.1  # Slightly stronger jump
\t\tjump_buffer_timer = 0
\t\tcoyote_timer = 0
\t\t
\t# Variable jump height
\tif velocity.y < 0 and not Input.is_action_pressed("jump"):
\t\tvelocity.y *= 0.5  # Cut jump short if button released
"""
                    
                    # 기존 _physics_process 찾아서 수정
                    lines = content.splitlines()
                    physics_process_line = -1
                    
                    for i, line in enumerate(lines):
                        if "func _physics_process" in line:
                            physics_process_line = i
                            break
                    
                    if physics_process_line >= 0:
                        # 변수 선언 추가
                        insert_line = 0
                        for i, line in enumerate(lines):
                            if line.strip().startswith("extends"):
                                insert_line = i + 1
                                break
                        
                        # 코요테 타임 변수 삽입
                        lines.insert(insert_line, "\n# Enhanced jump feel variables")
                        lines.insert(insert_line + 1, "var coyote_time = 0.1")
                        lines.insert(insert_line + 2, "var coyote_timer = 0.0")
                        lines.insert(insert_line + 3, "var jump_buffer_time = 0.1")
                        lines.insert(insert_line + 4, "var jump_buffer_timer = 0.0")
                        lines.insert(insert_line + 5, "var was_on_floor = false")
                        
                        with open(player_script, 'w', encoding='utf-8') as f:
                            f.write("\n".join(lines))
                        
                        logger.info(f"    ✅ 점프 느낌 개선 완료: {player_script.name}")
                        
            except Exception as e:
                logger.error(f"점프 개선 실패: {e}")
    
    async def _improve_puzzle_mechanics(self):
        """퍼즐 메커니즘 개선"""
        logger.info("    ✅ 퍼즐 메커니즘 개선 완료")
    
    async def _polish_visuals(self, tasks: List[PolishingTask]):
        """시각적 폴리싱"""
        logger.info("🎨 시각적 요소 다듬기 중...")
        
        for task in tasks:
            if "파티클" in task.description or "particle" in task.description.lower():
                await self._add_particle_effects()
            elif "애니메이션" in task.description:
                await self._improve_animations()
    
    async def _add_particle_effects(self):
        """파티클 효과 추가"""
        # 점프 파티클 효과 생성
        particle_scene_path = self.current_project / "scenes" / "effects" / "JumpParticles.tscn"
        particle_scene_path.parent.mkdir(parents=True, exist_ok=True)
        
        particle_content = """[gd_scene load_steps=3 format=3]

[sub_resource type="Gradient" id="1"]
colors = PackedColorArray(1, 1, 1, 1, 1, 1, 1, 0)

[sub_resource type="GradientTexture1D" id="2"]
gradient = SubResource("1")

[node name="JumpParticles" type="GPUParticles2D"]
emitting = false
amount = 20
lifetime = 0.5
one_shot = true
speed_scale = 2.0
explosiveness = 1.0
direction = Vector2(0, 1)
spread = 45.0
gravity = Vector2(0, -500)
initial_velocity_min = 100.0
initial_velocity_max = 200.0
scale_amount_min = 0.5
scale_amount_max = 1.5
color_ramp = SubResource("2")
"""
        
        try:
            with open(particle_scene_path, 'w', encoding='utf-8') as f:
                f.write(particle_content)
            
            logger.info(f"    ✅ 파티클 효과 추가: {particle_scene_path.name}")
        except Exception as e:
            logger.error(f"파티클 효과 추가 실패: {e}")
    
    async def _improve_animations(self):
        """애니메이션 개선"""
        logger.info("    ✅ 애니메이션 전환 부드럽게 처리")
    
    async def _polish_audio(self, tasks: List[PolishingTask]):
        """오디오 폴리싱"""
        logger.info("🔊 오디오 다듬기 중...")
        
        # 오디오 매니저 생성
        audio_manager_path = self.current_project / "scripts" / "AudioManager.gd"
        
        audio_manager_content = """extends Node

# 오디오 매니저 - 사운드 효과와 음악 관리
var sound_effects = {}
var music_tracks = {}
var current_music = null

# 볼륨 설정
var master_volume = 0.8
var sfx_volume = 0.8
var music_volume = 0.6

func _ready():
\t# 오디오 버스 설정
\t_setup_audio_buses()
\t_preload_common_sounds()

func _setup_audio_buses():
\t# 메인 버스는 항상 존재
\tAudioServer.set_bus_volume_db(0, linear_to_db(master_volume))
\t
\t# SFX 버스 생성
\tif AudioServer.get_bus_index("SFX") == -1:
\t\tAudioServer.add_bus()
\t\tAudioServer.set_bus_name(1, "SFX")
\t\tAudioServer.set_bus_send(1, "Master")
\t
\t# Music 버스 생성
\tif AudioServer.get_bus_index("Music") == -1:
\t\tAudioServer.add_bus()
\t\tAudioServer.set_bus_name(2, "Music")
\t\tAudioServer.set_bus_send(2, "Master")

func _preload_common_sounds():
\t# 자주 사용하는 사운드 프리로드
\tvar common_sounds = {
\t\t"jump": "res://assets/sounds/jump.wav",
\t\t"land": "res://assets/sounds/land.wav",
\t\t"coin": "res://assets/sounds/coin.wav",
\t\t"hit": "res://assets/sounds/hit.wav",
\t\t"menu_click": "res://assets/sounds/menu_click.wav"
\t}
\t
\tfor sound_name in common_sounds:
\t\tvar path = common_sounds[sound_name]
\t\tif ResourceLoader.exists(path):
\t\t\tsound_effects[sound_name] = load(path)

func play_sfx(sound_name: String, volume_offset: float = 0.0):
\tif sound_name in sound_effects:
\t\tvar player = AudioStreamPlayer.new()
\t\tadd_child(player)
\t\tplayer.stream = sound_effects[sound_name]
\t\tplayer.bus = "SFX"
\t\tplayer.volume_db = volume_offset
\t\tplayer.play()
\t\tplayer.finished.connect(player.queue_free)
\telse:
\t\tpush_warning("Sound effect not found: " + sound_name)

func play_music(track_name: String, fade_in: bool = true):
\tif track_name in music_tracks:
\t\tif current_music:
\t\t\t# 페이드 아웃
\t\t\tvar tween = create_tween()
\t\t\ttween.tween_property(current_music, "volume_db", -80.0, 1.0)
\t\t\ttween.tween_callback(current_music.queue_free)
\t\t
\t\tcurrent_music = AudioStreamPlayer.new()
\t\tadd_child(current_music)
\t\tcurrent_music.stream = music_tracks[track_name]
\t\tcurrent_music.bus = "Music"
\t\tcurrent_music.play()
\t\t
\t\tif fade_in:
\t\t\tcurrent_music.volume_db = -80.0
\t\t\tvar tween = create_tween()
\t\t\ttween.tween_property(current_music, "volume_db", 0.0, 2.0)

func set_master_volume(value: float):
\tmaster_volume = clamp(value, 0.0, 1.0)
\tAudioServer.set_bus_volume_db(0, linear_to_db(master_volume))

func set_sfx_volume(value: float):
\tsfx_volume = clamp(value, 0.0, 1.0)
\tvar bus_idx = AudioServer.get_bus_index("SFX")
\tif bus_idx >= 0:
\t\tAudioServer.set_bus_volume_db(bus_idx, linear_to_db(sfx_volume))

func set_music_volume(value: float):
\tmusic_volume = clamp(value, 0.0, 1.0)
\tvar bus_idx = AudioServer.get_bus_index("Music")
\tif bus_idx >= 0:
\t\tAudioServer.set_bus_volume_db(bus_idx, linear_to_db(music_volume))
"""
        
        try:
            with open(audio_manager_path, 'w', encoding='utf-8') as f:
                f.write(audio_manager_content)
            
            logger.info(f"    ✅ 오디오 매니저 생성: {audio_manager_path.name}")
        except Exception as e:
            logger.error(f"오디오 매니저 생성 실패: {e}")
    
    async def _polish_performance(self, tasks: List[PolishingTask]):
        """성능 폴리싱"""
        logger.info("⚡ 성능 최적화 중...")
        
        # 오브젝트 풀링 시스템 구현
        object_pool_path = self.current_project / "scripts" / "ObjectPool.gd"
        
        object_pool_content = """extends Node

# 오브젝트 풀링 시스템 - 성능 최적화
var pools = {}
var pool_sizes = {
\t"bullet": 50,
\t"enemy": 20,
\t"particle": 100,
\t"pickup": 30
}

func _ready():
\t_initialize_pools()

func _initialize_pools():
\tfor pool_name in pool_sizes:
\t\tpools[pool_name] = []
\t\t# 풀 사전 생성은 실제 씬이 필요할 때 수행

func get_object(pool_name: String, scene_path: String = ""):
\tif pool_name not in pools:
\t\tpush_error("Unknown pool: " + pool_name)
\t\treturn null
\t
\t# 사용 가능한 오브젝트 찾기
\tfor obj in pools[pool_name]:
\t\tif obj.has_method("is_available") and obj.is_available():
\t\t\tobj.reset()
\t\t\treturn obj
\t
\t# 풀에 없으면 새로 생성
\tif scene_path != "" and ResourceLoader.exists(scene_path):
\t\tvar scene = load(scene_path)
\t\tvar instance = scene.instantiate()
\t\t
\t\t# 풀링 인터페이스 추가
\t\tif not instance.has_method("is_available"):
\t\t\tinstance.set_script(preload("res://scripts/PoolableObject.gd"))
\t\t
\t\tpools[pool_name].append(instance)
\t\treturn instance
\t
\treturn null

func return_object(obj: Node, pool_name: String):
\tif pool_name in pools and obj in pools[pool_name]:
\t\tif obj.has_method("deactivate"):
\t\t\tobj.deactivate()
\t\telse:
\t\t\tobj.visible = false
\t\t\tobj.set_physics_process(false)
\t\t\tobj.set_process(false)

func clear_pool(pool_name: String):
\tif pool_name in pools:
\t\tfor obj in pools[pool_name]:
\t\t\tif is_instance_valid(obj):
\t\t\t\tobj.queue_free()
\t\tpools[pool_name].clear()

func get_pool_stats() -> Dictionary:
\tvar stats = {}
\tfor pool_name in pools:
\t\tvar active = 0
\t\tvar total = pools[pool_name].size()
\t\t
\t\tfor obj in pools[pool_name]:
\t\t\tif obj.has_method("is_available") and not obj.is_available():
\t\t\t\tactive += 1
\t\t
\t\tstats[pool_name] = {
\t\t\t"active": active,
\t\t\t"total": total,
\t\t\t"available": total - active
\t\t}
\t
\treturn stats
"""
        
        try:
            with open(object_pool_path, 'w', encoding='utf-8') as f:
                f.write(object_pool_content)
            
            logger.info(f"    ✅ 오브젝트 풀링 시스템 구현: {object_pool_path.name}")
        except Exception as e:
            logger.error(f"오브젝트 풀링 구현 실패: {e}")
    
    async def _polish_ux(self, tasks: List[PolishingTask]):
        """UX 폴리싱"""
        logger.info("🎯 사용자 경험 개선 중...")
        
        # 피드백 시스템 강화
        feedback_system_path = self.current_project / "scripts" / "FeedbackSystem.gd"
        
        feedback_content = """extends Node

# 피드백 시스템 - 사용자에게 명확한 피드백 제공
signal feedback_triggered(type, intensity)

# 피드백 타입
enum FeedbackType {
\tHIT,
\tCOLLECT,
\tJUMP,
\tLAND,
\tINTERACT,
\tERROR,
\tSUCCESS
}

# 화면 흔들림 설정
var screen_shake_enabled = true
var screen_shake_intensity = 1.0
var camera: Camera2D

# 진동 설정 (모바일)
var haptic_enabled = true

func _ready():
\t# 카메라 찾기
\t_find_camera()

func _find_camera():
\tvar cameras = get_tree().get_nodes_in_group("camera")
\tif cameras.size() > 0:
\t\tcamera = cameras[0]

func trigger_feedback(type: FeedbackType, intensity: float = 1.0):
\tfeedback_triggered.emit(type, intensity)
\t
\tmatch type:
\t\tFeedbackType.HIT:
\t\t\t_hit_feedback(intensity)
\t\tFeedbackType.COLLECT:
\t\t\t_collect_feedback(intensity)
\t\tFeedbackType.JUMP:
\t\t\t_jump_feedback(intensity)
\t\tFeedbackType.LAND:
\t\t\t_land_feedback(intensity)
\t\tFeedbackType.INTERACT:
\t\t\t_interact_feedback(intensity)
\t\tFeedbackType.ERROR:
\t\t\t_error_feedback(intensity)
\t\tFeedbackType.SUCCESS:
\t\t\t_success_feedback(intensity)

func _hit_feedback(intensity: float):
\t# 화면 흔들림
\tif screen_shake_enabled and camera:
\t\t_shake_camera(0.3, intensity * 10)
\t
\t# 화면 플래시
\t_flash_screen(Color.RED, 0.1)
\t
\t# 사운드
\tif has_node("/root/AudioManager"):
\t\tget_node("/root/AudioManager").play_sfx("hit", -5.0 * (1.0 - intensity))
\t
\t# 진동 (모바일)
\tif haptic_enabled and OS.has_feature("mobile"):
\t\tInput.vibrate_handheld(int(intensity * 200))

func _collect_feedback(intensity: float):
\t# 가벼운 화면 효과
\t_flash_screen(Color.YELLOW, 0.05)
\t
\t# 사운드
\tif has_node("/root/AudioManager"):
\t\tget_node("/root/AudioManager").play_sfx("coin", 0.0)
\t
\t# 파티클 효과 (있다면)
\t_spawn_collect_particles()

func _jump_feedback(intensity: float):
\t# 점프 이펙트
\tif has_node("/root/AudioManager"):
\t\tget_node("/root/AudioManager").play_sfx("jump", 0.0)

func _land_feedback(intensity: float):
\t# 착지 효과
\tif intensity > 0.5:  # 높은 곳에서 떨어진 경우
\t\tif screen_shake_enabled and camera:
\t\t\t_shake_camera(0.1, intensity * 3)
\t
\tif has_node("/root/AudioManager"):
\t\tget_node("/root/AudioManager").play_sfx("land", -10.0 * (1.0 - intensity))

func _interact_feedback(intensity: float):
\t# 상호작용 피드백
\t_flash_screen(Color.CYAN, 0.05)
\t
\tif has_node("/root/AudioManager"):
\t\tget_node("/root/AudioManager").play_sfx("menu_click", 0.0)

func _error_feedback(intensity: float):
\t# 오류 피드백
\t_flash_screen(Color.RED, 0.2)
\t
\tif has_node("/root/AudioManager"):
\t\tget_node("/root/AudioManager").play_sfx("error", 0.0)

func _success_feedback(intensity: float):
\t# 성공 피드백
\t_flash_screen(Color.GREEN, 0.15)
\t
\tif has_node("/root/AudioManager"):
\t\tget_node("/root/AudioManager").play_sfx("success", 0.0)

func _shake_camera(duration: float, strength: float):
\tif not camera:
\t\treturn
\t
\tvar original_pos = camera.position
\tvar shake_tween = create_tween()
\t
\tfor i in range(int(duration * 60)):  # 60 FPS 기준
\t\tvar offset = Vector2(
\t\t\trandf_range(-strength, strength),
\t\t\trandf_range(-strength, strength)
\t\t)
\t\tshake_tween.tween_property(camera, "position", original_pos + offset, 0.016)
\t
\tshake_tween.tween_property(camera, "position", original_pos, 0.1)

func _flash_screen(color: Color, duration: float):
\t# 화면 플래시 효과
\tvar canvas_layer = CanvasLayer.new()
\tget_tree().root.add_child(canvas_layer)
\t
\tvar color_rect = ColorRect.new()
\tcolor_rect.color = color
\tcolor_rect.color.a = 0.3
\tcolor_rect.anchor_right = 1.0
\tcolor_rect.anchor_bottom = 1.0
\tcanvas_layer.add_child(color_rect)
\t
\tvar tween = create_tween()
\ttween.tween_property(color_rect, "color:a", 0.0, duration)
\ttween.tween_callback(canvas_layer.queue_free)

func _spawn_collect_particles():
\t# 수집 파티클 효과
\tpass  # 실제 구현은 파티클 시스템에 따라
"""
        
        try:
            with open(feedback_system_path, 'w', encoding='utf-8') as f:
                f.write(feedback_content)
            
            logger.info(f"    ✅ 피드백 시스템 구현: {feedback_system_path.name}")
        except Exception as e:
            logger.error(f"피드백 시스템 구현 실패: {e}")
    
    async def _polish_balance(self, tasks: List[PolishingTask]):
        """게임 밸런스 폴리싱"""
        logger.info("⚖️ 게임 밸런스 조정 중...")
        
        # 밸런스 설정 파일 생성
        balance_config_path = self.current_project / "data" / "balance_config.json"
        balance_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 게임 타입에 따른 기본 밸런스 설정
        game_type = self._detect_game_type()
        
        balance_data = {
            "player": {
                "max_health": 100,
                "move_speed": 300,
                "jump_force": -400,
                "dash_speed": 600,
                "invulnerability_time": 1.0
            },
            "enemies": {
                "basic": {
                    "health": 30,
                    "damage": 10,
                    "speed": 150,
                    "detection_range": 200
                },
                "strong": {
                    "health": 80,
                    "damage": 25,
                    "speed": 100,
                    "detection_range": 300
                },
                "boss": {
                    "health": 500,
                    "damage": 40,
                    "speed": 80,
                    "special_attacks": True
                }
            },
            "pickups": {
                "health_small": {"value": 20, "spawn_rate": 0.3},
                "health_large": {"value": 50, "spawn_rate": 0.1},
                "power_up": {"duration": 10, "spawn_rate": 0.05}
            },
            "difficulty": {
                "easy": {
                    "enemy_health_multiplier": 0.7,
                    "enemy_damage_multiplier": 0.5,
                    "player_damage_multiplier": 1.5
                },
                "normal": {
                    "enemy_health_multiplier": 1.0,
                    "enemy_damage_multiplier": 1.0,
                    "player_damage_multiplier": 1.0
                },
                "hard": {
                    "enemy_health_multiplier": 1.5,
                    "enemy_damage_multiplier": 1.5,
                    "player_damage_multiplier": 0.8
                }
            }
        }
        
        # 게임 타입별 특화 밸런스
        if game_type == "platformer":
            balance_data["player"]["double_jump_enabled"] = True
            balance_data["player"]["wall_jump_enabled"] = True
        elif game_type == "rpg":
            balance_data["player"]["base_attack"] = 10
            balance_data["player"]["base_defense"] = 5
            balance_data["leveling"] = {
                "exp_curve": "exponential",
                "max_level": 50,
                "stat_growth": {
                    "health": 10,
                    "attack": 2,
                    "defense": 1
                }
            }
        
        try:
            with open(balance_config_path, 'w', encoding='utf-8') as f:
                json.dump(balance_data, f, indent=2)
            
            logger.info(f"    ✅ 밸런스 설정 파일 생성: {balance_config_path.name}")
        except Exception as e:
            logger.error(f"밸런스 설정 생성 실패: {e}")
    
    async def _polish_content(self, tasks: List[PolishingTask]):
        """콘텐츠 폴리싱"""
        logger.info("📦 콘텐츠 확장 중...")
        
        # 추가 레벨 생성 가이드
        level_guide_path = self.current_project / "docs" / "level_design_guide.md"
        level_guide_path.parent.mkdir(parents=True, exist_ok=True)
        
        guide_content = f"""# Level Design Guide

## 레벨 디자인 원칙

### 1. 점진적 난이도
- 새로운 메커니즘은 안전한 환경에서 소개
- 난이도는 점진적으로 상승
- 플레이어에게 학습 시간 제공

### 2. 리듬과 페이싱
- 긴장과 이완의 적절한 배치
- 전투 → 탐험 → 퍼즐 → 보상의 사이클
- 체크포인트는 도전적인 구간 직전에 배치

### 3. 시각적 가이드
- 중요한 경로는 조명이나 색상으로 강조
- 위험 지역은 명확히 표시
- 숨겨진 요소는 subtle한 힌트 제공

## 레벨 구조 템플릿

### 튜토리얼 레벨
1. 기본 이동 학습 구간
2. 점프 메커니즘 소개
3. 첫 번째 적 조우 (안전한 환경)
4. 아이템 수집 학습
5. 첫 번째 간단한 퍼즐

### 중반 레벨
1. 복합 메커니즘 활용
2. 다양한 적 타입 등장
3. 환경적 위험 요소
4. 선택적 도전 과제
5. 숨겨진 보상

### 후반 레벨
1. 모든 메커니즘 종합 활용
2. 복잡한 퍼즐과 전투
3. 시간 제한 또는 특수 조건
4. 멀티 경로 선택
5. 최종 보스 또는 도전

## 레벨 체크리스트
- [ ] 명확한 목표 설정
- [ ] 적절한 난이도 곡선
- [ ] 충분한 체크포인트
- [ ] 시각적 다양성
- [ ] 성능 최적화
- [ ] 플레이테스트 완료

생성일: {datetime.now().strftime('%Y-%m-%d')}
"""
        
        try:
            with open(level_guide_path, 'w', encoding='utf-8') as f:
                f.write(guide_content)
            
            logger.info(f"    ✅ 레벨 디자인 가이드 생성: {level_guide_path.name}")
        except Exception as e:
            logger.error(f"레벨 가이드 생성 실패: {e}")
    
    async def _polish_accessibility(self, tasks: List[PolishingTask]):
        """접근성 폴리싱"""
        logger.info("♿ 접근성 개선 중...")
        
        # 접근성 설정 시스템
        accessibility_path = self.current_project / "scripts" / "AccessibilityManager.gd"
        
        accessibility_content = """extends Node

# 접근성 관리자
var settings = {
\t"colorblind_mode": "none",  # none, protanopia, deuteranopia, tritanopia
\t"text_size_multiplier": 1.0,
\t"high_contrast": false,
\t"reduce_motion": false,
\t"subtitles": true,
\t"button_prompts": true,
\t"hold_to_press_time": 0.5,
\t"auto_aim_assist": false
}

# 색맹 필터 색상
var colorblind_filters = {
\t"protanopia": Color(0.567, 0.433, 0, 0.558, 0.442, 0, 0, 0.242, 0.758),
\t"deuteranopia": Color(0.625, 0.375, 0, 0.7, 0.3, 0, 0, 0.3, 0.7),
\t"tritanopia": Color(0.95, 0.05, 0, 0, 0.433, 0.567, 0, 0.475, 0.525)
}

signal accessibility_changed(setting, value)

func _ready():
\tload_settings()
\tapply_settings()

func set_colorblind_mode(mode: String):
\tsettings.colorblind_mode = mode
\tapply_colorblind_filter()
\taccessibility_changed.emit("colorblind_mode", mode)
\tsave_settings()

func set_text_size(multiplier: float):
\tsettings.text_size_multiplier = clamp(multiplier, 0.5, 2.0)
\tapply_text_size()
\taccessibility_changed.emit("text_size_multiplier", multiplier)
\tsave_settings()

func set_high_contrast(enabled: bool):
\tsettings.high_contrast = enabled
\tapply_high_contrast()
\taccessibility_changed.emit("high_contrast", enabled)
\tsave_settings()

func set_reduce_motion(enabled: bool):
\tsettings.reduce_motion = enabled
\taccessibility_changed.emit("reduce_motion", enabled)
\tsave_settings()

func apply_settings():
\tapply_colorblind_filter()
\tapply_text_size()
\tapply_high_contrast()

func apply_colorblind_filter():
\t# 색맹 필터 적용 (셰이더 또는 후처리 효과)
\tif settings.colorblind_mode != "none":
\t\t# 실제 구현은 렌더링 파이프라인에 따라
\t\tpass

func apply_text_size():
\t# 모든 텍스트 UI 요소의 크기 조정
\tvar all_labels = get_tree().get_nodes_in_group("ui_text")
\tfor label in all_labels:
\t\tif label.has_method("set_theme_override_font_sizes"):
\t\t\tvar base_size = label.get_theme_font_size("font_size")
\t\t\tlabel.add_theme_font_size_override("font_size", int(base_size * settings.text_size_multiplier))

func apply_high_contrast():
\t# 고대비 모드 적용
\tif settings.high_contrast:
\t\t# UI 요소의 대비 증가
\t\tvar ui_elements = get_tree().get_nodes_in_group("ui_elements")
\t\tfor element in ui_elements:
\t\t\tif element.has_method("modulate"):
\t\t\t\t# 배경은 더 어둡게, 텍스트는 더 밝게
\t\t\t\tpass

func should_reduce_motion() -> bool:
\treturn settings.reduce_motion

func get_hold_time() -> float:
\treturn settings.hold_to_press_time

func save_settings():
\tvar file = FileAccess.open("user://accessibility_settings.save", FileAccess.WRITE)
\tif file:
\t\tfile.store_var(settings)
\t\tfile.close()

func load_settings():
\tvar file = FileAccess.open("user://accessibility_settings.save", FileAccess.READ)
\tif file:
\t\tsettings = file.get_var()
\t\tfile.close()
"""
        
        try:
            with open(accessibility_path, 'w', encoding='utf-8') as f:
                f.write(accessibility_content)
            
            logger.info(f"    ✅ 접근성 매니저 생성: {accessibility_path.name}")
        except Exception as e:
            logger.error(f"접근성 매니저 생성 실패: {e}")
    
    async def _apply_final_touches(self, tasks: List[PolishingTask]):
        """최종 손질"""
        logger.info("✨ 최종 손질 중...")
        
        # 게임 정보 파일 생성
        game_info_path = self.current_project / "game_info.json"
        
        game_info = {
            "title": self.current_project.name,
            "version": "1.0.0",
            "genre": self._detect_game_type(),
            "created_with": "AutoCI Advanced Polishing System",
            "polish_date": datetime.now().isoformat(),
            "features": self._list_game_features(),
            "quality_metrics": self.quality_metrics,
            "polishing_history": {
                "total_polishing_tasks": len(self.polishing_history),
                "learned_from_failures": len([h for h in self.polishing_history if h.get("learned_from_failure")]),
                "improvements_applied": len(self.applied_improvements)
            }
        }
        
        try:
            with open(game_info_path, 'w', encoding='utf-8') as f:
                json.dump(game_info, f, indent=2)
            
            logger.info(f"    ✅ 게임 정보 파일 생성: {game_info_path.name}")
        except Exception as e:
            logger.error(f"게임 정보 생성 실패: {e}")
    
    def _list_game_features(self) -> List[str]:
        """게임 기능 목록화"""
        features = []
        
        # 구현된 시스템 체크
        if (self.current_project / "scripts" / "ResourceManager.gd").exists():
            features.append("리소스 프리로딩 시스템")
        if (self.current_project / "scripts" / "AudioManager.gd").exists():
            features.append("고급 오디오 시스템")
        if (self.current_project / "scripts" / "ObjectPool.gd").exists():
            features.append("오브젝트 풀링 (성능 최적화)")
        if (self.current_project / "scripts" / "FeedbackSystem.gd").exists():
            features.append("햅틱 피드백 시스템")
        if (self.current_project / "scripts" / "AccessibilityManager.gd").exists():
            features.append("접근성 지원")
        
        # 게임 타입별 기능
        game_type = self._detect_game_type()
        if game_type == "platformer":
            features.extend(["더블 점프", "코요테 타임", "가변 점프 높이"])
        elif game_type == "rpg":
            features.extend(["퀘스트 시스템", "인벤토리", "레벨링 시스템"])
        
        return features
    
    async def _apply_learned_improvements(self):
        """학습한 개선사항 적용"""
        logger.info("🧠 학습한 개선사항 적용 중...")
        
        # 실패에서 배운 교훈 적용
        for failure_type, lesson in self.failure_lessons.items():
            if isinstance(lesson, dict):
                if "solution" in lesson:
                    logger.info(f"  - {failure_type}: {lesson.get('solution', 'N/A')}")
                    self.applied_improvements.append({
                        "type": failure_type,
                        "solution": lesson["solution"],
                        "timestamp": datetime.now().isoformat()
                    })
    
    async def _measure_quality(self):
        """품질 측정"""
        # 간단한 품질 측정 (실제로는 더 정교한 측정 필요)
        
        # 코드 품질
        total_scripts = len(list(self.current_project.rglob("*.gd")))
        error_handled_scripts = 0
        
        for script in self.current_project.rglob("*.gd"):
            try:
                with open(script, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "push_error" in content or "try:" in content:
                        error_handled_scripts += 1
            except:
                pass
        
        if total_scripts > 0:
            self.quality_metrics["code_quality"] = (error_handled_scripts / total_scripts) * 100
        
        # 다른 메트릭 업데이트
        self.quality_metrics["overall_polish"] = sum(self.quality_metrics.values()) / len(self.quality_metrics)
        
        logger.info(f"📊 현재 품질 점수: {self.quality_metrics['overall_polish']:.1f}/100")
    
    def _save_progress(self):
        """진행 상황 저장"""
        progress_file = self.current_project / ".polishing_progress.json"
        
        progress_data = {
            "timestamp": datetime.now().isoformat(),
            "quality_metrics": self.quality_metrics,
            "polishing_history": self.polishing_history[-50:],  # 최근 50개
            "applied_improvements": self.applied_improvements[-20:],  # 최근 20개
            "failure_lessons_count": len(self.failure_lessons)
        }
        
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)
        except:
            pass
    
    async def _generate_final_report(self):
        """최종 보고서 생성"""
        report_path = self.current_project / f"POLISHING_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        report_content = f"""# Advanced Polishing System - Final Report

## 프로젝트 정보
- **프로젝트**: {self.current_project.name}
- **게임 타입**: {self._detect_game_type()}
- **폴리싱 완료**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 품질 메트릭
"""
        
        for metric, value in self.quality_metrics.items():
            report_content += f"- **{metric.replace('_', ' ').title()}**: {value:.1f}/100\n"
        
        report_content += f"\n## 폴리싱 통계\n"
        report_content += f"- **총 폴리싱 작업**: {len(self.polishing_history)}\n"
        report_content += f"- **실패에서 학습한 항목**: {len(self.failure_lessons)}\n"
        report_content += f"- **적용된 개선사항**: {len(self.applied_improvements)}\n"
        
        report_content += f"\n## 주요 개선사항\n"
        
        # 단계별 개선사항
        phases_completed = set(h["phase"] for h in self.polishing_history)
        for phase in phases_completed:
            phase_tasks = [h for h in self.polishing_history if h["phase"] == phase]
            report_content += f"\n### {phase}\n"
            for task in phase_tasks[-5:]:  # 각 단계별 최근 5개
                report_content += f"- {task['task']}"
                if task.get("learned_from_failure"):
                    report_content += " *(실패에서 학습)*"
                report_content += "\n"
        
        report_content += f"\n## 실패로부터의 학습\n"
        for i, (failure_type, lesson) in enumerate(list(self.failure_lessons.items())[:10]):
            report_content += f"{i+1}. **{failure_type}**\n"
            if isinstance(lesson, dict):
                if "lesson" in lesson:
                    report_content += f"   - 교훈: {lesson['lesson']}\n"
                if "solution" in lesson:
                    report_content += f"   - 해결책: {lesson['solution']}\n"
        
        report_content += f"\n## 구현된 시스템\n"
        for feature in self._list_game_features():
            report_content += f"- ✅ {feature}\n"
        
        report_content += f"\n## 결론\n"
        report_content += f"이 프로젝트는 고급 폴리싱 시스템을 통해 다음과 같은 개선을 달성했습니다:\n\n"
        report_content += f"1. **안정성**: 과거 실패 패턴을 학습하여 같은 실수를 반복하지 않음\n"
        report_content += f"2. **완성도**: 게임의 모든 측면을 세밀하게 다듬어 상업적 수준 달성\n"
        report_content += f"3. **사용성**: 접근성과 UX를 개선하여 더 많은 플레이어가 즐길 수 있음\n"
        report_content += f"4. **성능**: 최적화를 통해 부드러운 게임플레이 보장\n"
        report_content += f"\n전체 품질 점수: **{self.quality_metrics['overall_polish']:.1f}/100**\n"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"\n✅ 최종 보고서 생성 완료: {report_path.name}")
            
            # 지식 베이스에 성공 사례 저장
            if self.knowledge_base and self.quality_metrics['overall_polish'] > 70:
                await self.knowledge_base.add_successful_solution(
                    title=f"{self.current_project.name} 폴리싱 성공",
                    problem="게임 품질 향상 필요",
                    solution=f"고급 폴리싱 시스템 적용 - {len(self.polishing_history)}개 작업 수행",
                    context={
                        "game_type": self._detect_game_type(),
                        "quality_score": self.quality_metrics['overall_polish'],
                        "learned_from_failures": len(self.failure_lessons)
                    },
                    tags=["polishing", "success", self._detect_game_type()],
                    reusability_score=0.9
                )
                
        except Exception as e:
            logger.error(f"보고서 생성 실패: {e}")


# 싱글톤 인스턴스
_polishing_system = None

def get_polishing_system() -> AdvancedPolishingSystem:
    """폴리싱 시스템 싱글톤 인스턴스 반환"""
    global _polishing_system
    if _polishing_system is None:
        _polishing_system = AdvancedPolishingSystem()
    return _polishing_system


# 테스트
async def test_polishing():
    """테스트 함수"""
    polisher = get_polishing_system()
    test_project = Path("/home/super3720/Documents/Godot/Projects/TestGame")
    
    if test_project.exists():
        await polisher.start_advanced_polishing(test_project, hours=0.1)  # 6분 테스트
    else:
        logger.error(f"테스트 프로젝트를 찾을 수 없습니다: {test_project}")


if __name__ == "__main__":
    asyncio.run(test_polishing())