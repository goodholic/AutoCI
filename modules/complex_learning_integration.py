"""
복합 학습 통합 시스템
autoci create와 autoci learn에서 가상 입력과 Godot 조작 학습을 통합
"""

import asyncio
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

from modules.godot_manipulation_learning import GodotManipulationLearning
from modules.virtual_input_controller import get_virtual_input, InputMode
from core_system.continuous_learning_system import ContinuousLearningSystem
from modules.game_development_pipeline import GameDevelopmentPipeline

logger = logging.getLogger(__name__)


class ComplexLearningIntegration:
    """복합 학습 통합 시스템"""
    
    def __init__(self):
        self.godot_learning = GodotManipulationLearning()
        self.virtual_input = get_virtual_input()
        self.continuous_learning = ContinuousLearningSystem()
        self.game_pipeline = GameDevelopmentPipeline()
        
        # 학습 통계
        self.stats = {
            "total_manipulations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "learned_patterns": 0,
            "applied_fixes": 0
        }
        
        # 통합 학습 데이터
        self.integrated_knowledge = {
            "godot_operations": {},
            "code_patterns": {},
            "optimization_strategies": {},
            "error_recovery": {}
        }
        
    async def integrated_create_with_learning(self, game_name: str, game_type: str):
        """
        autoci create - 게임 생성과 동시에 Godot 조작 학습
        24시간 동안 실제 게임을 만들면서 조작 방법 학습
        """
        logger.info(f"🎮 복합 학습 모드로 게임 개발 시작: {game_name}")
        
        # 두 개의 병렬 태스크 실행
        create_task = asyncio.create_task(
            self._create_game_with_monitoring(game_name, game_type)
        )
        
        learning_task = asyncio.create_task(
            self._parallel_manipulation_learning()
        )
        
        # 24시간 동안 병렬 실행
        try:
            await asyncio.gather(create_task, learning_task)
        except Exception as e:
            logger.error(f"복합 학습 중 오류: {e}")
        finally:
            await self._save_integrated_knowledge()
            
    async def _create_game_with_monitoring(self, game_name: str, game_type: str):
        """게임 생성하면서 모든 조작 모니터링"""
        # 가상 입력 활성화
        await self.virtual_input.activate()
        self.virtual_input.set_mode(InputMode.GODOT_EDITOR)
        
        # 게임 개발 시작
        await self.game_pipeline.start_development(game_name, game_type)
        
        # 개발 중 모든 조작 기록
        while self.game_pipeline.is_running:
            # 현재 수행 중인 작업 모니터링
            current_phase = self.game_pipeline.current_project.current_phase
            
            # 가상 입력 액션 기록
            actions = self.virtual_input.get_action_history()
            
            # 학습 데이터로 변환
            for action in actions:
                self._process_action_for_learning(action, current_phase)
                
            await asyncio.sleep(1)
            
        await self.virtual_input.deactivate()
        
    async def _parallel_manipulation_learning(self):
        """병렬로 Godot 조작 학습 실행"""
        # 별도의 가상 화면에서 학습
        learning_duration = 24  # 24시간
        await self.godot_learning.continuous_manipulation_learning(learning_duration)
        
    def _process_action_for_learning(self, action: Dict, phase: Any):
        """조작 액션을 학습 데이터로 처리"""
        action_data = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase.value if hasattr(phase, 'value') else str(phase),
            "action_type": action.get("type"),
            "details": action.get("data"),
            "context": {
                "mode": action.get("mode"),
                "success": True  # 실제 결과 확인 필요
            }
        }
        
        # 조작 타입별 분류
        action_type = action.get("type", "unknown")
        if action_type not in self.integrated_knowledge["godot_operations"]:
            self.integrated_knowledge["godot_operations"][action_type] = []
            
        self.integrated_knowledge["godot_operations"][action_type].append(action_data)
        self.stats["total_manipulations"] += 1
        
    async def integrated_learn_with_manipulation(self, duration_hours: int = 24):
        """
        autoci learn - 연속 학습과 동시에 Godot 조작 연습
        이론 학습과 실전 조작을 병행
        """
        logger.info("🧠 복합 학습 모드 시작 (이론 + 실전)")
        
        # 세 가지 학습을 병렬로 실행
        tasks = [
            # 1. 기존 연속 학습 (이론)
            asyncio.create_task(
                self.continuous_learning.continuous_learning_loop(duration_hours)
            ),
            
            # 2. Godot 조작 학습 (실전)
            asyncio.create_task(
                self.godot_learning.continuous_manipulation_learning(duration_hours)
            ),
            
            # 3. 학습 내용 실시간 통합
            asyncio.create_task(
                self._integrate_learning_results(duration_hours)
            )
        ]
        
        await asyncio.gather(*tasks)
        
    async def _integrate_learning_results(self, duration_hours: int):
        """학습 결과를 실시간으로 통합"""
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        while time.time() < end_time:
            # 이론 학습 결과 수집
            theory_knowledge = await self._collect_theory_knowledge()
            
            # 실전 조작 결과 수집
            practice_knowledge = await self._collect_practice_knowledge()
            
            # 통합 및 패턴 추출
            integrated_patterns = self._extract_integrated_patterns(
                theory_knowledge, practice_knowledge
            )
            
            # 지식 베이스 업데이트
            self._update_integrated_knowledge(integrated_patterns)
            
            # 30분마다 통합
            await asyncio.sleep(1800)
            
    async def _collect_theory_knowledge(self) -> Dict:
        """이론 학습에서 지식 수집"""
        # continuous_learning 폴더에서 최근 학습 데이터 읽기
        learning_dir = Path("continuous_learning/answers")
        today = datetime.now().strftime("%Y%m%d")
        today_dir = learning_dir / today
        
        knowledge = {
            "godot_concepts": [],
            "csharp_patterns": [],
            "optimization_tips": []
        }
        
        if today_dir.exists():
            for file in today_dir.glob("*.json"):
                try:
                    data = json.loads(file.read_text())
                    # 주제별 분류
                    if "godot" in data.get("topic", "").lower():
                        knowledge["godot_concepts"].append(data)
                    elif "csharp" in data.get("topic", "").lower():
                        knowledge["csharp_patterns"].append(data)
                    elif "optimization" in data.get("operation", "").lower():
                        knowledge["optimization_tips"].append(data)
                except Exception as e:
                    logger.error(f"학습 데이터 읽기 오류: {e}")
                    
        return knowledge
        
    async def _collect_practice_knowledge(self) -> Dict:
        """실전 조작에서 지식 수집"""
        # Godot 조작 학습 데이터 수집
        return {
            "successful_workflows": self.godot_learning.success_patterns,
            "common_errors": self.godot_learning.failure_patterns,
            "efficiency_metrics": self._calculate_efficiency_metrics()
        }
        
    def _extract_integrated_patterns(self, theory: Dict, practice: Dict) -> Dict:
        """이론과 실전을 통합하여 패턴 추출"""
        patterns = {
            "theory_to_practice": [],  # 이론이 실전에 적용된 사례
            "practice_insights": [],   # 실전에서 발견한 새로운 통찰
            "optimization_opportunities": []  # 개선 가능 영역
        }
        
        # 이론과 실전 매칭
        for concept in theory.get("godot_concepts", []):
            # 관련 실전 경험 찾기
            related_practice = self._find_related_practice(
                concept, practice.get("successful_workflows", {})
            )
            
            if related_practice:
                patterns["theory_to_practice"].append({
                    "theory": concept,
                    "practice": related_practice,
                    "effectiveness": self._evaluate_effectiveness(concept, related_practice)
                })
                
        # 실전에서만 발견된 패턴
        for workflow_type, workflows in practice.get("successful_workflows", {}).items():
            if len(workflows) >= 3:  # 3회 이상 성공한 패턴
                patterns["practice_insights"].append({
                    "workflow": workflow_type,
                    "frequency": len(workflows),
                    "average_efficiency": self._calculate_average_efficiency(workflows)
                })
                
        return patterns
        
    def _find_related_practice(self, theory: Dict, practices: Dict) -> Optional[Dict]:
        """이론과 관련된 실전 경험 찾기"""
        theory_keywords = theory.get("keywords", [])
        
        for practice_type, practice_list in practices.items():
            for practice in practice_list:
                practice_pattern = practice.get("pattern", {})
                if any(keyword in str(practice_pattern) for keyword in theory_keywords):
                    return practice
                    
        return None
        
    def _evaluate_effectiveness(self, theory: Dict, practice: Dict) -> float:
        """이론이 실전에 얼마나 효과적으로 적용되었는지 평가"""
        # 간단한 평가 로직
        practice_efficiency = practice.get("efficiency_score", 0.5)
        theory_quality = theory.get("quality", 0.7)
        
        return (practice_efficiency + theory_quality) / 2
        
    def _calculate_average_efficiency(self, workflows: List[Dict]) -> float:
        """평균 효율성 계산"""
        if not workflows:
            return 0.0
            
        total_efficiency = sum(w.get("efficiency_score", 0) for w in workflows)
        return total_efficiency / len(workflows)
        
    def _calculate_efficiency_metrics(self) -> Dict:
        """효율성 메트릭 계산"""
        total = self.stats["total_manipulations"]
        success = self.stats["successful_operations"]
        
        return {
            "success_rate": (success / total) if total > 0 else 0,
            "error_recovery_rate": self._calculate_error_recovery_rate(),
            "learning_velocity": self.stats["learned_patterns"] / max(1, total),
            "application_rate": self.stats["applied_fixes"] / max(1, self.stats["learned_patterns"])
        }
        
    def _calculate_error_recovery_rate(self) -> float:
        """에러 복구율 계산"""
        total_errors = sum(
            len(errors) for errors in self.godot_learning.failure_patterns.values()
        )
        
        recovered_errors = sum(
            1 for errors in self.godot_learning.failure_patterns.values()
            for error in errors
            if error.get("recovery_attempts")
        )
        
        return (recovered_errors / total_errors) if total_errors > 0 else 0
        
    def _update_integrated_knowledge(self, patterns: Dict):
        """통합 지식 베이스 업데이트"""
        # 이론-실전 연결 저장
        for item in patterns.get("theory_to_practice", []):
            effectiveness = item.get("effectiveness", 0)
            if effectiveness > 0.7:  # 효과적인 패턴만 저장
                self.integrated_knowledge["code_patterns"][
                    item["theory"].get("topic", "unknown")
                ] = item
                
        # 실전 통찰 저장
        for insight in patterns.get("practice_insights", []):
            if insight.get("average_efficiency", 0) > 0.8:
                self.integrated_knowledge["optimization_strategies"][
                    insight["workflow"]
                ] = insight
                
        self.stats["learned_patterns"] = len(self.integrated_knowledge["code_patterns"])
        
    async def apply_integrated_knowledge(self):
        """
        autoci fix - 통합된 지식을 엔진에 적용
        이론과 실전을 결합한 개선 사항 적용
        """
        logger.info("🔧 통합 지식을 엔진에 적용합니다")
        
        # 1. 코드 패턴 개선
        code_improvements = await self._generate_code_improvements()
        
        # 2. 워크플로우 최적화
        workflow_optimizations = await self._optimize_workflows()
        
        # 3. 에러 처리 강화
        error_handlers = await self._enhance_error_handling()
        
        # 4. 실전 적용
        await self._apply_improvements_to_engine(
            code_improvements,
            workflow_optimizations,
            error_handlers
        )
        
        # 5. 결과 보고
        self._generate_application_report()
        
    async def _generate_code_improvements(self) -> List[Dict]:
        """코드 개선 사항 생성"""
        improvements = []
        
        for pattern_name, pattern_data in self.integrated_knowledge["code_patterns"].items():
            improvement = {
                "pattern": pattern_name,
                "original": pattern_data.get("theory", {}).get("answer", ""),
                "optimized": await self._optimize_code_pattern(pattern_data),
                "benefits": self._analyze_benefits(pattern_data)
            }
            improvements.append(improvement)
            
        return improvements
        
    async def _optimize_code_pattern(self, pattern_data: Dict) -> str:
        """코드 패턴 최적화"""
        # AI를 사용하여 코드 최적화
        from modules.ai_model_integration import get_ai_integration
        ai = get_ai_integration()
        
        prompt = f"""
        다음 코드 패턴을 실전 경험을 바탕으로 최적화하세요:
        
        원본 코드:
        {pattern_data.get('theory', {}).get('answer', '')}
        
        실전 경험:
        {pattern_data.get('practice', {}).get('pattern', {})}
        
        효율성: {pattern_data.get('effectiveness', 0):.2f}
        
        더 효율적이고 실용적인 코드로 개선해주세요.
        """
        
        return await ai.generate(prompt)
        
    def _analyze_benefits(self, pattern_data: Dict) -> List[str]:
        """개선의 이점 분석"""
        benefits = []
        
        effectiveness = pattern_data.get("effectiveness", 0)
        if effectiveness > 0.8:
            benefits.append("높은 실전 효과성")
        if effectiveness > 0.9:
            benefits.append("검증된 최적 패턴")
            
        practice = pattern_data.get("practice", {})
        if practice.get("efficiency_score", 0) > 0.85:
            benefits.append("뛰어난 실행 효율성")
            
        return benefits
        
    async def _optimize_workflows(self) -> List[Dict]:
        """워크플로우 최적화"""
        optimizations = []
        
        for workflow_name, workflow_data in self.integrated_knowledge["optimization_strategies"].items():
            optimization = {
                "workflow": workflow_name,
                "current_efficiency": workflow_data.get("average_efficiency", 0),
                "optimization_steps": self._generate_optimization_steps(workflow_data),
                "expected_improvement": self._estimate_improvement(workflow_data)
            }
            optimizations.append(optimization)
            
        return optimizations
        
    def _generate_optimization_steps(self, workflow_data: Dict) -> List[str]:
        """최적화 단계 생성"""
        steps = []
        
        efficiency = workflow_data.get("average_efficiency", 0)
        
        if efficiency < 0.7:
            steps.append("불필요한 단계 제거")
            steps.append("자동화 가능 부분 식별")
        if efficiency < 0.8:
            steps.append("단축키 및 매크로 활용")
            steps.append("병렬 처리 가능 작업 분리")
        if efficiency < 0.9:
            steps.append("미세 조정 및 최적화")
            
        return steps
        
    def _estimate_improvement(self, workflow_data: Dict) -> float:
        """예상 개선율 추정"""
        current_efficiency = workflow_data.get("average_efficiency", 0.5)
        # 현재 효율성이 낮을수록 개선 여지가 큼
        return min(0.3, (1 - current_efficiency) * 0.5)
        
    async def _enhance_error_handling(self) -> List[Dict]:
        """에러 처리 강화"""
        handlers = []
        
        for error_type, errors in self.godot_learning.failure_patterns.items():
            if len(errors) >= 2:  # 2회 이상 발생한 에러
                handler = {
                    "error_type": error_type,
                    "frequency": len(errors),
                    "recovery_strategies": self._compile_recovery_strategies(errors),
                    "prevention_measures": await self._generate_prevention_measures(error_type, errors)
                }
                handlers.append(handler)
                
        return handlers
        
    def _compile_recovery_strategies(self, errors: List[Dict]) -> List[str]:
        """복구 전략 컴파일"""
        strategies = []
        
        for error in errors:
            for attempt in error.get("recovery_attempts", []):
                suggestion = attempt.get("suggestion", "")
                if suggestion and suggestion not in strategies:
                    strategies.append(suggestion)
                    
        return strategies[:5]  # 상위 5개만
        
    async def _generate_prevention_measures(self, error_type: str, errors: List[Dict]) -> List[str]:
        """예방 조치 생성"""
        # 에러 패턴 분석
        common_contexts = []
        for error in errors:
            context = error.get("context", {})
            if context:
                common_contexts.append(context)
                
        # AI를 사용하여 예방 조치 생성
        from modules.ai_model_integration import get_ai_integration
        ai = get_ai_integration()
        
        prompt = f"""
        다음 에러를 예방하기 위한 조치를 제안하세요:
        에러 타입: {error_type}
        발생 횟수: {len(errors)}
        공통 컨텍스트: {common_contexts[:3]}
        
        구체적이고 실행 가능한 예방 조치를 3개 제시하세요.
        """
        
        response = await ai.generate(prompt)
        # 응답을 리스트로 파싱
        return response.split('\n')[:3]
        
    async def _apply_improvements_to_engine(
        self,
        code_improvements: List[Dict],
        workflow_optimizations: List[Dict],
        error_handlers: List[Dict]
    ):
        """개선 사항을 엔진에 적용"""
        logger.info("🚀 개선 사항을 엔진에 적용 중...")
        
        # 1. 코드 템플릿 업데이트
        for improvement in code_improvements:
            await self._update_code_template(improvement)
            self.stats["applied_fixes"] += 1
            
        # 2. 워크플로우 매크로 생성
        for optimization in workflow_optimizations:
            await self._create_workflow_macro(optimization)
            self.stats["applied_fixes"] += 1
            
        # 3. 에러 핸들러 등록
        for handler in error_handlers:
            await self._register_error_handler(handler)
            self.stats["applied_fixes"] += 1
            
        logger.info(f"✅ 총 {self.stats['applied_fixes']}개 개선 사항 적용 완료")
        
    async def _update_code_template(self, improvement: Dict):
        """코드 템플릿 업데이트"""
        # 실제 구현은 프로젝트 구조에 따라
        template_file = Path("templates") / f"{improvement['pattern']}.cs"
        template_file.parent.mkdir(exist_ok=True)
        template_file.write_text(improvement["optimized"])
        
    async def _create_workflow_macro(self, optimization: Dict):
        """워크플로우 매크로 생성"""
        # 가상 입력 시스템에 매크로 추가
        macro_name = f"optimized_{optimization['workflow']}"
        macro_steps = []
        
        for step in optimization["optimization_steps"]:
            # 단계를 실제 액션으로 변환
            if "단축키" in step:
                macro_steps.append({"type": "key", "keys": ["ctrl", "shift", "o"]})
            elif "자동화" in step:
                macro_steps.append({"type": "wait", "duration": 0.1})
                
        self.virtual_input.macro_library[macro_name] = macro_steps
        
    async def _register_error_handler(self, handler: Dict):
        """에러 핸들러 등록"""
        # 에러 처리 시스템에 핸들러 추가
        error_handlers_file = Path("error_handlers") / f"{handler['error_type']}.json"
        error_handlers_file.parent.mkdir(exist_ok=True)
        error_handlers_file.write_text(json.dumps(handler, indent=2))
        
    def _generate_application_report(self):
        """적용 결과 보고서 생성"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "improvements": {
                "code_patterns": len(self.integrated_knowledge["code_patterns"]),
                "workflows": len(self.integrated_knowledge["optimization_strategies"]),
                "error_handlers": len(self.integrated_knowledge["error_recovery"])
            },
            "effectiveness": {
                "learning_efficiency": self._calculate_learning_efficiency(),
                "application_success_rate": self._calculate_application_success_rate()
            }
        }
        
        report_file = Path("reports") / f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2))
        
        logger.info(f"""
🎯 통합 학습 적용 완료!
━━━━━━━━━━━━━━━━━━━━━━━━
총 조작: {self.stats['total_manipulations']}회
성공: {self.stats['successful_operations']}회
학습된 패턴: {self.stats['learned_patterns']}개
적용된 개선: {self.stats['applied_fixes']}개

학습 효율성: {self._calculate_learning_efficiency():.1%}
적용 성공률: {self._calculate_application_success_rate():.1%}

보고서: {report_file}
━━━━━━━━━━━━━━━━━━━━━━━━
""")
        
    def _calculate_learning_efficiency(self) -> float:
        """학습 효율성 계산"""
        if self.stats["total_manipulations"] == 0:
            return 0.0
            
        return self.stats["learned_patterns"] / self.stats["total_manipulations"]
        
    def _calculate_application_success_rate(self) -> float:
        """적용 성공률 계산"""
        if self.stats["learned_patterns"] == 0:
            return 0.0
            
        return self.stats["applied_fixes"] / self.stats["learned_patterns"]
        
    async def _save_integrated_knowledge(self):
        """통합 지식 저장"""
        knowledge_file = Path("integrated_knowledge") / f"knowledge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        knowledge_file.parent.mkdir(exist_ok=True)
        knowledge_file.write_text(json.dumps(self.integrated_knowledge, indent=2))
        
        logger.info(f"💾 통합 지식 저장: {knowledge_file}")


# 싱글톤 인스턴스
_complex_learning = None


def get_complex_learning() -> ComplexLearningIntegration:
    """복합 학습 시스템 싱글톤 반환"""
    global _complex_learning
    if _complex_learning is None:
        _complex_learning = ComplexLearningIntegration()
    return _complex_learning