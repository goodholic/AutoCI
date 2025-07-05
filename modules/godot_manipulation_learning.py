"""
Godot 엔진 조작 학습 모듈
24시간 연속 조작을 통한 실패 경험 축적 및 학습
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from modules.virtual_input_controller import get_virtual_input, InputMode
from modules.ai_model_integration import get_ai_integration
from core_system.continuous_learning_system import ContinuousLearningSystem

logger = logging.getLogger(__name__)


class GodotAction(Enum):
    """Godot 조작 액션 타입"""
    CREATE_NODE = "create_node"
    ADD_SCRIPT = "add_script"
    MODIFY_PROPERTY = "modify_property"
    CONNECT_SIGNAL = "connect_signal"
    RUN_SCENE = "run_scene"
    DEBUG_GAME = "debug_game"
    CREATE_ANIMATION = "create_animation"
    SETUP_PHYSICS = "setup_physics"
    CONFIGURE_UI = "configure_ui"
    IMPORT_ASSET = "import_asset"


@dataclass
class ManipulationResult:
    """조작 결과 기록"""
    action: GodotAction
    success: bool
    error_message: Optional[str]
    time_taken: float
    steps_completed: int
    total_steps: int
    screenshot_path: Optional[str]
    learned_pattern: Optional[Dict[str, Any]]


class GodotManipulationLearning:
    """Godot 엔진 조작 학습 시스템"""
    
    def __init__(self):
        self.virtual_input = get_virtual_input()
        self.ai_model = get_ai_integration()
        self.learning_system = ContinuousLearningSystem()
        
        # 학습 데이터 저장 경로
        self.data_dir = Path("continuous_learning/godot_manipulation")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 실패 패턴 데이터베이스
        self.failure_patterns: Dict[str, List[Dict]] = {
            "node_creation_failures": [],
            "script_errors": [],
            "signal_connection_issues": [],
            "performance_problems": [],
            "ui_layout_mistakes": []
        }
        
        # 성공 패턴 데이터베이스
        self.success_patterns: Dict[str, List[Dict]] = {
            "efficient_workflows": [],
            "optimized_structures": [],
            "reusable_components": []
        }
        
        self._load_learning_data()
        
    def _load_learning_data(self):
        """기존 학습 데이터 로드"""
        failure_file = self.data_dir / "failure_patterns.json"
        success_file = self.data_dir / "success_patterns.json"
        
        if failure_file.exists():
            self.failure_patterns = json.loads(failure_file.read_text())
        if success_file.exists():
            self.success_patterns = json.loads(success_file.read_text())
            
    def _save_learning_data(self):
        """학습 데이터 저장"""
        failure_file = self.data_dir / "failure_patterns.json"
        success_file = self.data_dir / "success_patterns.json"
        
        failure_file.write_text(json.dumps(self.failure_patterns, indent=2))
        success_file.write_text(json.dumps(self.success_patterns, indent=2))
        
    async def continuous_manipulation_learning(self, duration_hours: int = 24):
        """24시간 연속 조작 학습"""
        logger.info(f"🎮 Godot 조작 학습 시작 ({duration_hours}시간)")
        
        await self.virtual_input.activate()
        self.virtual_input.set_mode(InputMode.GODOT_EDITOR)
        
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        manipulation_count = 0
        failure_count = 0
        success_count = 0
        
        # 다양한 조작 시나리오
        scenarios = [
            self._scenario_create_platformer,
            self._scenario_create_rpg_system,
            self._scenario_create_ui_menu,
            self._scenario_setup_multiplayer,
            self._scenario_optimize_performance,
            self._scenario_create_particle_effects,
            self._scenario_implement_save_system,
            self._scenario_create_inventory
        ]
        
        while time.time() < end_time:
            # 랜덤 시나리오 선택
            scenario = scenarios[manipulation_count % len(scenarios)]
            
            try:
                logger.info(f"\n🔄 조작 #{manipulation_count + 1}: {scenario.__name__}")
                
                # 조작 실행 및 결과 기록
                result = await scenario()
                
                if result.success:
                    success_count += 1
                    self._record_success(result)
                    logger.info(f"✅ 성공! (시간: {result.time_taken:.1f}초)")
                else:
                    failure_count += 1
                    self._record_failure(result)
                    logger.warning(f"❌ 실패: {result.error_message}")
                    
                    # 실패에서 학습
                    await self._learn_from_failure(result)
                
                # AI 모델에 경험 전달
                await self._update_ai_knowledge(result)
                
                manipulation_count += 1
                
                # 진행 상황 출력
                elapsed = time.time() - start_time
                remaining = end_time - time.time()
                success_rate = (success_count / manipulation_count) * 100 if manipulation_count > 0 else 0
                
                logger.info(f"""
📊 진행 상황:
   총 조작: {manipulation_count}회
   성공: {success_count}회
   실패: {failure_count}회
   성공률: {success_rate:.1f}%
   경과: {elapsed/3600:.1f}시간
   남은 시간: {remaining/3600:.1f}시간
""")
                
                # 주기적으로 학습 데이터 저장
                if manipulation_count % 10 == 0:
                    self._save_learning_data()
                    await self._consolidate_learning()
                
                # 다음 조작까지 대기
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"조작 중 오류: {e}")
                failure_count += 1
                
        await self.virtual_input.deactivate()
        
        # 최종 보고서
        self._generate_final_report(manipulation_count, success_count, failure_count)
        
    async def _scenario_create_platformer(self) -> ManipulationResult:
        """플랫폼 게임 생성 시나리오"""
        start_time = time.time()
        steps_completed = 0
        total_steps = 8
        
        try:
            # 1. 새 씬 생성
            await self.virtual_input.execute_macro("godot_new_scene")
            steps_completed += 1
            
            # 2. 기본 구조 생성
            await self.virtual_input.godot_create_node("Node2D", "Main")
            await asyncio.sleep(0.5)
            steps_completed += 1
            
            # 3. 플레이어 생성
            await self.virtual_input.godot_create_node("CharacterBody2D", "Player")
            await self.virtual_input.godot_create_node("CollisionShape2D", "PlayerCollision")
            await self.virtual_input.godot_create_node("Sprite2D", "PlayerSprite")
            steps_completed += 1
            
            # 4. 플레이어 스크립트 추가
            player_script = await self.ai_model.generate_game_code(
                "Create Godot 4 C# platformer player controller with double jump"
            )
            await self.virtual_input.godot_add_script(player_script)
            steps_completed += 1
            
            # 5. 타일맵 생성
            await self.virtual_input.godot_create_node("TileMap", "World")
            steps_completed += 1
            
            # 6. 카메라 설정
            await self.virtual_input.godot_create_node("Camera2D", "GameCamera")
            await self._set_camera_follow_player()
            steps_completed += 1
            
            # 7. UI 생성
            await self._create_game_ui()
            steps_completed += 1
            
            # 8. 씬 저장 및 테스트
            await self.virtual_input.execute_macro("godot_save")
            await self.virtual_input.type_text("Platformer_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".tscn")
            await self.virtual_input.press_key("enter")
            steps_completed += 1
            
            return ManipulationResult(
                action=GodotAction.CREATE_NODE,
                success=True,
                error_message=None,
                time_taken=time.time() - start_time,
                steps_completed=steps_completed,
                total_steps=total_steps,
                screenshot_path=None,
                learned_pattern={
                    "workflow": "platformer_creation",
                    "key_steps": ["scene_setup", "player_controller", "tilemap", "camera_follow"],
                    "time_efficiency": "good"
                }
            )
            
        except Exception as e:
            return ManipulationResult(
                action=GodotAction.CREATE_NODE,
                success=False,
                error_message=str(e),
                time_taken=time.time() - start_time,
                steps_completed=steps_completed,
                total_steps=total_steps,
                screenshot_path=None,
                learned_pattern={
                    "failure_point": f"step_{steps_completed + 1}",
                    "error_type": type(e).__name__
                }
            )
            
    async def _scenario_create_rpg_system(self) -> ManipulationResult:
        """RPG 시스템 생성 시나리오"""
        start_time = time.time()
        steps_completed = 0
        total_steps = 10
        
        try:
            # RPG 시스템 구현
            # 1. 캐릭터 스탯 시스템
            await self._create_character_stats_system()
            steps_completed += 2
            
            # 2. 인벤토리 시스템
            await self._create_inventory_system()
            steps_completed += 2
            
            # 3. 전투 시스템
            await self._create_combat_system()
            steps_completed += 2
            
            # 4. 대화 시스템
            await self._create_dialogue_system()
            steps_completed += 2
            
            # 5. 퀘스트 시스템
            await self._create_quest_system()
            steps_completed += 2
            
            return ManipulationResult(
                action=GodotAction.CREATE_NODE,
                success=True,
                error_message=None,
                time_taken=time.time() - start_time,
                steps_completed=steps_completed,
                total_steps=total_steps,
                screenshot_path=None,
                learned_pattern={
                    "workflow": "rpg_system_creation",
                    "complexity": "high",
                    "reusable_components": ["stats", "inventory", "combat", "dialogue", "quest"]
                }
            )
            
        except Exception as e:
            return ManipulationResult(
                action=GodotAction.CREATE_NODE,
                success=False,
                error_message=str(e),
                time_taken=time.time() - start_time,
                steps_completed=steps_completed,
                total_steps=total_steps,
                screenshot_path=None,
                learned_pattern=None
            )
            
    async def _learn_from_failure(self, result: ManipulationResult):
        """실패로부터 학습"""
        failure_data = {
            "timestamp": datetime.now().isoformat(),
            "action": result.action.value,
            "error": result.error_message,
            "context": result.learned_pattern,
            "recovery_attempts": []
        }
        
        # 에러 타입별 분류
        if "node" in str(result.error_message).lower():
            self.failure_patterns["node_creation_failures"].append(failure_data)
        elif "script" in str(result.error_message).lower():
            self.failure_patterns["script_errors"].append(failure_data)
        elif "signal" in str(result.error_message).lower():
            self.failure_patterns["signal_connection_issues"].append(failure_data)
        else:
            self.failure_patterns["performance_problems"].append(failure_data)
            
        # AI에게 실패 패턴 학습 요청
        learning_prompt = f"""
        Godot 조작 실패 분석:
        액션: {result.action.value}
        에러: {result.error_message}
        실패 단계: {result.steps_completed}/{result.total_steps}
        
        이 실패를 피하기 위한 개선 방법을 제시하세요.
        """
        
        improvement = await self.ai_model.generate(learning_prompt)
        failure_data["recovery_attempts"].append({
            "suggestion": improvement,
            "timestamp": datetime.now().isoformat()
        })
        
    def _record_success(self, result: ManipulationResult):
        """성공 패턴 기록"""
        if result.learned_pattern:
            workflow_type = result.learned_pattern.get("workflow", "unknown")
            
            success_data = {
                "timestamp": datetime.now().isoformat(),
                "action": result.action.value,
                "time_taken": result.time_taken,
                "pattern": result.learned_pattern,
                "efficiency_score": self._calculate_efficiency(result)
            }
            
            if workflow_type in ["platformer_creation", "rpg_system_creation"]:
                self.success_patterns["efficient_workflows"].append(success_data)
            else:
                self.success_patterns["optimized_structures"].append(success_data)
                
    def _record_failure(self, result: ManipulationResult):
        """실패 패턴 기록"""
        # 실패 데이터는 _learn_from_failure에서 처리
        pass
        
    def _calculate_efficiency(self, result: ManipulationResult) -> float:
        """효율성 점수 계산"""
        # 시간 효율성 (빠를수록 높은 점수)
        time_score = max(0, 1 - (result.time_taken / 300))  # 5분 기준
        
        # 완성도 (완료된 단계 비율)
        completion_score = result.steps_completed / result.total_steps
        
        return (time_score + completion_score) / 2
        
    async def _update_ai_knowledge(self, result: ManipulationResult):
        """AI 지식 베이스 업데이트"""
        # 조작 경험을 학습 시스템에 전달
        experience = {
            "type": "godot_manipulation",
            "action": result.action.value,
            "success": result.success,
            "time_taken": result.time_taken,
            "learned_pattern": result.learned_pattern,
            "timestamp": time.time()
        }
        
        # 연속 학습 시스템에 경험 추가
        if hasattr(self.learning_system, 'add_experience'):
            self.learning_system.add_experience(experience)
            
    async def _consolidate_learning(self):
        """학습 내용 통합 및 최적화"""
        logger.info("🧠 학습 내용 통합 중...")
        
        # 실패 패턴 분석
        common_failures = self._analyze_failure_patterns()
        
        # 성공 패턴 최적화
        optimized_workflows = self._optimize_success_patterns()
        
        # 새로운 전략 생성
        new_strategies = await self._generate_new_strategies(
            common_failures, optimized_workflows
        )
        
        # 지식 베이스에 저장
        knowledge_update = {
            "timestamp": datetime.now().isoformat(),
            "failures_analyzed": len(common_failures),
            "workflows_optimized": len(optimized_workflows),
            "new_strategies": new_strategies
        }
        
        knowledge_file = self.data_dir / f"knowledge_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        knowledge_file.write_text(json.dumps(knowledge_update, indent=2))
        
    def _analyze_failure_patterns(self) -> List[Dict]:
        """실패 패턴 분석"""
        common_failures = []
        
        for category, failures in self.failure_patterns.items():
            if len(failures) >= 3:  # 3회 이상 발생한 패턴
                # 가장 빈번한 에러 찾기
                error_counts = {}
                for failure in failures:
                    error = failure.get("error", "unknown")
                    error_counts[error] = error_counts.get(error, 0) + 1
                    
                most_common = max(error_counts.items(), key=lambda x: x[1])
                common_failures.append({
                    "category": category,
                    "error": most_common[0],
                    "frequency": most_common[1]
                })
                
        return common_failures
        
    def _optimize_success_patterns(self) -> List[Dict]:
        """성공 패턴 최적화"""
        optimized = []
        
        for category, successes in self.success_patterns.items():
            if successes:
                # 가장 효율적인 워크플로우 찾기
                best_workflow = max(
                    successes,
                    key=lambda x: x.get("efficiency_score", 0)
                )
                optimized.append({
                    "category": category,
                    "best_practice": best_workflow
                })
                
        return optimized
        
    async def _generate_new_strategies(
        self,
        failures: List[Dict],
        successes: List[Dict]
    ) -> List[Dict]:
        """새로운 전략 생성"""
        strategies = []
        
        for failure in failures:
            # AI에게 개선 전략 요청
            prompt = f"""
            Godot 조작에서 반복되는 실패:
            카테고리: {failure['category']}
            에러: {failure['error']}
            빈도: {failure['frequency']}회
            
            이를 해결하기 위한 새로운 접근 방법을 제시하세요.
            """
            
            strategy = await self.ai_model.generate(prompt)
            strategies.append({
                "problem": failure,
                "solution": strategy
            })
            
        return strategies
        
    def _generate_final_report(self, total: int, success: int, failure: int):
        """최종 학습 보고서 생성"""
        report = {
            "summary": {
                "total_manipulations": total,
                "successful": success,
                "failed": failure,
                "success_rate": (success / total * 100) if total > 0 else 0,
                "timestamp": datetime.now().isoformat()
            },
            "failure_analysis": self._analyze_failure_patterns(),
            "best_practices": self._optimize_success_patterns(),
            "learned_patterns": {
                "node_creation": len(self.success_patterns.get("efficient_workflows", [])),
                "error_recovery": len(self.failure_patterns.get("script_errors", [])),
                "optimization": len(self.success_patterns.get("optimized_structures", []))
            }
        }
        
        report_file = self.data_dir / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.write_text(json.dumps(report, indent=2))
        
        logger.info(f"""
🎯 Godot 조작 학습 완료!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
총 조작 횟수: {total}
성공: {success} ({success/total*100:.1f}%)
실패: {failure} ({failure/total*100:.1f}%)

주요 학습 내용:
- 효율적인 워크플로우: {len(self.success_patterns.get('efficient_workflows', []))}개
- 에러 해결 패턴: {len(self.failure_patterns.get('script_errors', []))}개
- 최적화 전략: {len(self.success_patterns.get('optimized_structures', []))}개

보고서 저장: {report_file}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
        
    # Helper methods for scenarios
    async def _set_camera_follow_player(self):
        """카메라가 플레이어를 따라가도록 설정"""
        # Inspector에서 카메라 속성 설정
        await self.virtual_input.move_mouse(1500, 400)
        await self.virtual_input.click()
        await self.virtual_input.type_text("true")  # Enabled
        await self.virtual_input.press_key("tab")
        await self.virtual_input.type_text("Player")  # Follow target
        
    async def _create_game_ui(self):
        """게임 UI 생성"""
        await self.virtual_input.godot_create_node("CanvasLayer", "UI")
        await self.virtual_input.godot_create_node("Control", "HUD")
        await self.virtual_input.godot_create_node("Label", "ScoreLabel")
        await self.virtual_input.godot_create_node("Label", "LivesLabel")
        
    async def _create_character_stats_system(self):
        """캐릭터 스탯 시스템 생성"""
        stats_script = await self.ai_model.generate_game_code(
            "Create character stats system with HP, MP, STR, DEX, INT for Godot C#"
        )
        await self.virtual_input.godot_add_script(stats_script)
        
    async def _create_inventory_system(self):
        """인벤토리 시스템 생성"""
        inventory_script = await self.ai_model.generate_game_code(
            "Create grid-based inventory system with drag and drop for Godot C#"
        )
        await self.virtual_input.godot_add_script(inventory_script)
        
    async def _create_combat_system(self):
        """전투 시스템 생성"""
        combat_script = await self.ai_model.generate_game_code(
            "Create turn-based combat system with skills and combos for Godot C#"
        )
        await self.virtual_input.godot_add_script(combat_script)
        
    async def _create_dialogue_system(self):
        """대화 시스템 생성"""
        dialogue_script = await self.ai_model.generate_game_code(
            "Create branching dialogue system with choices for Godot C#"
        )
        await self.virtual_input.godot_add_script(dialogue_script)
        
    async def _create_quest_system(self):
        """퀘스트 시스템 생성"""
        quest_script = await self.ai_model.generate_game_code(
            "Create quest system with objectives and rewards for Godot C#"
        )
        await self.virtual_input.godot_add_script(quest_script)
        
    async def _scenario_create_ui_menu(self) -> ManipulationResult:
        """UI 메뉴 생성 시나리오"""
        # 구현 생략 (위와 유사한 패턴)
        pass
        
    async def _scenario_setup_multiplayer(self) -> ManipulationResult:
        """멀티플레이어 설정 시나리오"""
        # 구현 생략
        pass
        
    async def _scenario_optimize_performance(self) -> ManipulationResult:
        """성능 최적화 시나리오"""
        # 구현 생략
        pass
        
    async def _scenario_create_particle_effects(self) -> ManipulationResult:
        """파티클 효과 생성 시나리오"""
        # 구현 생략
        pass
        
    async def _scenario_implement_save_system(self) -> ManipulationResult:
        """저장 시스템 구현 시나리오"""
        # 구현 생략
        pass
        
    async def _scenario_create_inventory(self) -> ManipulationResult:
        """인벤토리 생성 시나리오"""
        # 구현 생략
        pass