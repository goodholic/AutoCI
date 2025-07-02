#!/usr/bin/env python3
"""
개발 경험 수집기 - 24시간 게임 개발 중 발견된 가치있는 정보와 패턴을 수집하고 저장
모든 성공적인 해결책, 유용한 코드 패턴, 최적화 방법을 학습하고 저장합니다.
"""

import os
import sys
import json
import time
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class ExperienceType(Enum):
    """경험 타입"""
    ERROR_SOLUTION = auto()          # 오류 해결책
    GAME_MECHANIC = auto()          # 게임 메카닉 구현
    CODE_PATTERN = auto()           # 유용한 코드 패턴
    PERFORMANCE_OPT = auto()         # 성능 최적화
    RESOURCE_GENERATION = auto()    # 리소스 생성 패턴
    COMMUNITY_SOLUTION = auto()     # 커뮤니티 솔루션
    AI_DISCOVERY = auto()           # AI가 발견한 방법
    WORKAROUND = auto()            # 우회 방법
    BEST_PRACTICE = auto()         # 모범 사례
    CREATIVE_SOLUTION = auto()     # 창의적 해결책

class DevelopmentExperienceCollector:
    """개발 경험 수집기"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.knowledge_base_path = self.project_root / "continuous_learning" / "development_knowledge"
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        
        # 지식 저장소
        self.error_solutions = {}              # 오류별 해결책
        self.successful_mechanics = []         # 성공적인 게임 메카닉
        self.code_patterns = {}               # 유용한 코드 패턴
        self.performance_optimizations = []    # 성능 최적화 방법
        self.resource_patterns = {}           # 리소스 생성 패턴
        self.community_wisdom = []            # 커뮤니티 지혜
        self.ai_discoveries = []              # AI 발견
        
        # 통계
        self.total_experiences = 0
        self.successful_applications = 0
        self.learning_sessions = []
        
        # 실시간 수집 상태
        self.active_problems = {}             # 현재 해결 중인 문제들
        self.monitoring_enabled = True
        
        # 학습 패턴 인식
        self.pattern_frequency = defaultdict(int)
        self.solution_effectiveness = defaultdict(float)
        
        # 경험 평가 기준
        self.experience_scores = defaultdict(float)
        
        # 기존 지식 로드
        self._load_existing_knowledge()
    
    def _load_existing_knowledge(self):
        """기존 지식 로드"""
        knowledge_file = self.knowledge_base_path / "collected_knowledge.json"
        if knowledge_file.exists():
            try:
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.error_solutions = data.get('error_solutions', {})
                    self.successful_mechanics = data.get('successful_mechanics', [])
                    self.code_patterns = data.get('code_patterns', {})
                    self.performance_optimizations = data.get('performance_optimizations', [])
                    self.resource_patterns = data.get('resource_patterns', {})
                    self.community_wisdom = data.get('community_wisdom', [])
                    self.ai_discoveries = data.get('ai_discoveries', [])
                    self.total_experiences = data.get('total_experiences', 0)
                    logger.info(f"기존 지식 로드 완료: {self.total_experiences}개의 경험")
            except Exception as e:
                logger.error(f"지식 로드 실패: {e}")
    
    def _save_knowledge(self):
        """지식 저장"""
        knowledge_file = self.knowledge_base_path / "collected_knowledge.json"
        data = {
            'error_solutions': self.error_solutions,
            'successful_mechanics': self.successful_mechanics,
            'code_patterns': self.code_patterns,
            'performance_optimizations': self.performance_optimizations,
            'resource_patterns': self.resource_patterns,
            'community_wisdom': self.community_wisdom,
            'ai_discoveries': self.ai_discoveries,
            'total_experiences': self.total_experiences,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"지식 저장 완료: {self.total_experiences}개의 경험")
        except Exception as e:
            logger.error(f"지식 저장 실패: {e}")
    
    async def start_monitoring(self, project_path: Path):
        """실시간 모니터링 시작"""
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    📚 개발 경험 수집기 시작                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 프로젝트: {project_path.name}
📊 기존 경험: {self.total_experiences}개
🔍 모니터링 중...
""")
        
        self.monitoring_enabled = True
        
        # 주기적으로 지식 저장
        while self.monitoring_enabled:
            await asyncio.sleep(300)  # 5분마다
            self._save_knowledge()
            await self._analyze_patterns()
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_enabled = False
        self._save_knowledge()
        self._generate_learning_report()
    
    async def collect_error_solution(self, error: Dict[str, Any], solution: Dict[str, Any], success: bool):
        """오류 해결책 수집"""
        error_hash = self._get_error_hash(error)
        
        experience = {
            'error': error,
            'solution': solution,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'attempts': solution.get('attempts', 1),
            'strategy': solution.get('strategy', 'unknown')
        }
        
        if error_hash not in self.error_solutions:
            self.error_solutions[error_hash] = []
        
        self.error_solutions[error_hash].append(experience)
        
        if success:
            self.total_experiences += 1
            self.successful_applications += 1
            logger.info(f"✅ 성공적인 오류 해결책 수집: {error.get('type', 'unknown')}")
            
            # 효과성 점수 업데이트
            self.solution_effectiveness[solution.get('strategy', 'unknown')] += 1.0
        else:
            # 실패한 시도도 학습
            self.solution_effectiveness[solution.get('strategy', 'unknown')] -= 0.1
    
    async def collect_game_mechanic(self, mechanic_name: str, implementation: Dict[str, Any], performance_metrics: Dict[str, Any] = None):
        """성공적인 게임 메카닉 수집"""
        mechanic = {
            'name': mechanic_name,
            'implementation': implementation,
            'performance': performance_metrics,
            'timestamp': datetime.now().isoformat(),
            'code_snippet': implementation.get('code', ''),
            'description': implementation.get('description', ''),
            'complexity': self._evaluate_complexity(implementation.get('code', ''))
        }
        
        self.successful_mechanics.append(mechanic)
        self.total_experiences += 1
        
        logger.info(f"🎮 게임 메카닉 수집: {mechanic_name}")
        
        # 패턴 빈도 업데이트
        self.pattern_frequency[f"mechanic_{mechanic_name}"] += 1
    
    async def collect_code_pattern(self, pattern_name: str, pattern_code: str, use_case: str, effectiveness: float = 1.0):
        """유용한 코드 패턴 수집"""
        pattern_hash = hashlib.md5(pattern_code.encode()).hexdigest()[:8]
        
        pattern = {
            'name': pattern_name,
            'code': pattern_code,
            'use_case': use_case,
            'effectiveness': effectiveness,
            'timestamp': datetime.now().isoformat(),
            'applications': 1
        }
        
        if pattern_hash in self.code_patterns:
            # 이미 있는 패턴이면 적용 횟수 증가
            self.code_patterns[pattern_hash]['applications'] += 1
            self.code_patterns[pattern_hash]['effectiveness'] = (
                self.code_patterns[pattern_hash]['effectiveness'] + effectiveness
            ) / 2
        else:
            self.code_patterns[pattern_hash] = pattern
            self.total_experiences += 1
        
        logger.info(f"📝 코드 패턴 수집: {pattern_name}")
        self.pattern_frequency[f"pattern_{pattern_name}"] += 1
    
    async def collect_performance_optimization(self, optimization: Dict[str, Any]):
        """성능 최적화 방법 수집"""
        opt_data = {
            'type': optimization.get('type', 'general'),
            'before_metrics': optimization.get('before', {}),
            'after_metrics': optimization.get('after', {}),
            'improvement': self._calculate_improvement(
                optimization.get('before', {}),
                optimization.get('after', {})
            ),
            'method': optimization.get('method', ''),
            'code_changes': optimization.get('code_changes', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        self.performance_optimizations.append(opt_data)
        self.total_experiences += 1
        
        improvement_percent = opt_data['improvement']
        logger.info(f"🚀 성능 최적화 수집: {optimization.get('type', 'general')} - {improvement_percent:.1f}% 개선")
    
    async def collect_resource_pattern(self, resource_type: str, generation_method: Dict[str, Any]):
        """리소스 생성 패턴 수집"""
        pattern = {
            'resource_type': resource_type,
            'method': generation_method.get('method', ''),
            'parameters': generation_method.get('parameters', {}),
            'code': generation_method.get('code', ''),
            'success_rate': generation_method.get('success_rate', 1.0),
            'timestamp': datetime.now().isoformat()
        }
        
        if resource_type not in self.resource_patterns:
            self.resource_patterns[resource_type] = []
        
        self.resource_patterns[resource_type].append(pattern)
        self.total_experiences += 1
        
        logger.info(f"🎨 리소스 패턴 수집: {resource_type}")
    
    async def collect_community_solution(self, problem: str, solution: Dict[str, Any], source: str):
        """커뮤니티 솔루션 수집"""
        community_knowledge = {
            'problem': problem,
            'solution': solution,
            'source': source,  # Discord, Reddit, Forums 등
            'votes': solution.get('votes', 0),
            'verified': solution.get('verified', False),
            'timestamp': datetime.now().isoformat()
        }
        
        self.community_wisdom.append(community_knowledge)
        self.total_experiences += 1
        
        logger.info(f"💬 커뮤니티 솔루션 수집: {source} - {problem[:50]}...")
    
    async def collect_ai_discovery(self, discovery: Dict[str, Any]):
        """AI가 발견한 방법 수집"""
        ai_knowledge = {
            'discovery_type': discovery.get('type', 'general'),
            'description': discovery.get('description', ''),
            'code': discovery.get('code', ''),
            'context': discovery.get('context', ''),
            'effectiveness': discovery.get('effectiveness', 1.0),
            'creativity_score': discovery.get('creativity_score', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        self.ai_discoveries.append(ai_knowledge)
        self.total_experiences += 1
        
        logger.info(f"🤖 AI 발견 수집: {discovery.get('type', 'general')}")
    
    def search_similar_problems(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """유사한 문제와 해결책 검색"""
        problem_hash = self._get_error_hash(problem)
        similar_solutions = []
        
        # 직접 매칭
        if problem_hash in self.error_solutions:
            similar_solutions.extend(self.error_solutions[problem_hash])
        
        # 유사도 기반 검색
        problem_keywords = self._extract_keywords(str(problem))
        
        for error_hash, solutions in self.error_solutions.items():
            if error_hash != problem_hash:
                for solution in solutions:
                    similarity = self._calculate_similarity(
                        problem_keywords,
                        self._extract_keywords(str(solution['error']))
                    )
                    if similarity > 0.6:  # 60% 이상 유사도
                        similar_solutions.append({
                            'solution': solution,
                            'similarity': similarity
                        })
        
        # 유사도 순으로 정렬
        similar_solutions.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return similar_solutions[:5]  # 상위 5개 반환
    
    def get_best_practices(self, category: str = None) -> List[Dict[str, Any]]:
        """카테고리별 모범 사례 반환"""
        best_practices = []
        
        # 효과적인 솔루션 추출
        for strategy, score in self.solution_effectiveness.items():
            if score > 5:  # 5번 이상 성공한 전략
                best_practices.append({
                    'type': 'error_solution',
                    'strategy': strategy,
                    'success_score': score,
                    'category': 'problem_solving'
                })
        
        # 자주 사용되는 패턴
        for pattern_name, frequency in self.pattern_frequency.items():
            if frequency > 3:  # 3번 이상 사용된 패턴
                best_practices.append({
                    'type': 'pattern',
                    'name': pattern_name,
                    'frequency': frequency,
                    'category': 'code_pattern'
                })
        
        # 높은 개선율의 최적화
        for opt in self.performance_optimizations:
            if opt['improvement'] > 20:  # 20% 이상 개선
                best_practices.append({
                    'type': 'optimization',
                    'method': opt['type'],
                    'improvement': opt['improvement'],
                    'category': 'performance'
                })
        
        # 카테고리 필터링
        if category:
            best_practices = [bp for bp in best_practices if bp.get('category') == category]
        
        # 점수 순으로 정렬
        best_practices.sort(
            key=lambda x: x.get('success_score', x.get('frequency', x.get('improvement', 0))),
            reverse=True
        )
        
        return best_practices[:10]  # 상위 10개
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """학습 인사이트 반환"""
        insights = {
            'total_experiences': self.total_experiences,
            'success_rate': self.successful_applications / max(self.total_experiences, 1),
            'most_effective_strategies': [],
            'common_patterns': [],
            'optimization_summary': {},
            'ai_creativity_score': 0,
            'community_contribution': len(self.community_wisdom)
        }
        
        # 가장 효과적인 전략
        sorted_strategies = sorted(
            self.solution_effectiveness.items(),
            key=lambda x: x[1],
            reverse=True
        )
        insights['most_effective_strategies'] = sorted_strategies[:5]
        
        # 일반적인 패턴
        sorted_patterns = sorted(
            self.pattern_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        insights['common_patterns'] = sorted_patterns[:10]
        
        # 최적화 요약
        if self.performance_optimizations:
            avg_improvement = sum(opt['improvement'] for opt in self.performance_optimizations) / len(self.performance_optimizations)
            insights['optimization_summary'] = {
                'average_improvement': avg_improvement,
                'total_optimizations': len(self.performance_optimizations),
                'best_optimization': max(self.performance_optimizations, key=lambda x: x['improvement'])
            }
        
        # AI 창의성 점수
        if self.ai_discoveries:
            avg_creativity = sum(ai['creativity_score'] for ai in self.ai_discoveries) / len(self.ai_discoveries)
            insights['ai_creativity_score'] = avg_creativity
        
        return insights
    
    async def integrate_with_improver(self, improver_instance):
        """persistent_game_improver와 통합"""
        # 개선자의 로그를 모니터링하고 성공적인 개선 수집
        original_fix = improver_instance._persistently_fix_error
        
        async def wrapped_fix(error, *args, **kwargs):
            # 문제 추적 시작
            problem_id = self._start_tracking_problem(error)
            
            # 원래 함수 실행
            result = await original_fix(error, *args, **kwargs)
            
            # 결과 수집
            if result:
                solution_data = self._end_tracking_problem(problem_id, success=True)
                await self.collect_error_solution(error, solution_data, True)
            
            return result
        
        improver_instance._persistently_fix_error = wrapped_fix
        logger.info("✅ persistent_game_improver와 통합 완료")
    
    async def integrate_with_extreme_engine(self, engine_instance):
        """extreme_persistence_engine과 통합"""
        # 극한 엔진의 창의적인 해결책 수집
        original_solve = engine_instance.solve_with_extreme_persistence
        
        async def wrapped_solve(error, project_path, remaining_hours):
            # 해결 과정 모니터링
            start_attempts = engine_instance.total_attempts
            
            result = await original_solve(error, project_path, remaining_hours)
            
            # 창의적인 해결책 수집
            if result:
                end_attempts = engine_instance.total_attempts
                creativity_score = min((end_attempts - start_attempts) / 10, 10)
                
                await self.collect_ai_discovery({
                    'type': 'extreme_persistence',
                    'description': f"극한의 끈질김으로 {end_attempts - start_attempts}번 시도 끝에 해결",
                    'context': str(error),
                    'creativity_score': creativity_score,
                    'effectiveness': 1.0
                })
            
            return result
        
        engine_instance.solve_with_extreme_persistence = wrapped_solve
        logger.info("✅ extreme_persistence_engine과 통합 완료")
    
    async def integrate_with_ai_controller(self, controller_instance):
        """ai_model_controller와 통합"""
        # AI 모델 컨트롤러의 품질 응답 수집
        original_evaluate = controller_instance.evaluate_response_quality
        
        def wrapped_evaluate(question, response, model_name):
            # 원래 평가 실행
            quality = original_evaluate(question, response, model_name)
            
            # 높은 품질의 응답은 패턴으로 저장
            if quality.is_acceptable and quality.score > 0.8:
                asyncio.create_task(self.collect_code_pattern(
                    f"high_quality_{model_name}_{question.get('type', 'general')}",
                    response[:500],  # 처음 500자만 저장
                    f"{model_name}에서 생성한 고품질 응답 패턴",
                    effectiveness=quality.score
                ))
            
            # 실패한 응답에서도 학습
            elif not quality.is_acceptable:
                self.pattern_frequency[f"failed_{model_name}_{quality.issues[0] if quality.issues else 'unknown'}"] += 1
            
            return quality
        
        controller_instance.evaluate_response_quality = wrapped_evaluate
        logger.info("✅ ai_model_controller와 통합 완료")
    
    def _get_error_hash(self, error: Dict[str, Any]) -> str:
        """오류 해시 생성"""
        error_str = f"{error.get('type', '')}_{error.get('description', '')}"
        return hashlib.md5(error_str.encode()).hexdigest()[:8]
    
    def _evaluate_complexity(self, code: str) -> int:
        """코드 복잡도 평가"""
        if not code:
            return 0
        
        # 간단한 복잡도 계산
        lines = code.strip().split('\n')
        complexity = len(lines)
        
        # 제어문 개수
        control_statements = ['if ', 'for ', 'while ', 'match ', 'func ']
        for line in lines:
            for stmt in control_statements:
                if stmt in line:
                    complexity += 2
        
        return min(complexity, 100)
    
    def _calculate_improvement(self, before: Dict[str, Any], after: Dict[str, Any]) -> float:
        """개선율 계산"""
        if not before or not after:
            return 0.0
        
        # FPS 개선
        if 'fps' in before and 'fps' in after:
            fps_before = before.get('fps', 30)
            fps_after = after.get('fps', 30)
            if fps_before > 0:
                return ((fps_after - fps_before) / fps_before) * 100
        
        # 메모리 개선
        if 'memory' in before and 'memory' in after:
            mem_before = before.get('memory', 100)
            mem_after = after.get('memory', 100)
            if mem_before > 0:
                return ((mem_before - mem_after) / mem_before) * 100
        
        return 0.0
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """텍스트에서 키워드 추출"""
        # 간단한 키워드 추출
        words = text.lower().split()
        keywords = set()
        
        for word in words:
            if len(word) > 3:  # 3글자 이상
                keywords.add(word.strip('.,!?;:()[]{}'))
        
        return keywords
    
    def _calculate_similarity(self, keywords1: Set[str], keywords2: Set[str]) -> float:
        """키워드 유사도 계산"""
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _start_tracking_problem(self, problem: Dict[str, Any]) -> str:
        """문제 추적 시작"""
        problem_id = f"prob_{int(time.time() * 1000)}"
        self.active_problems[problem_id] = {
            'problem': problem,
            'start_time': datetime.now(),
            'attempts': []
        }
        return problem_id
    
    def _end_tracking_problem(self, problem_id: str, success: bool) -> Dict[str, Any]:
        """문제 추적 종료"""
        if problem_id not in self.active_problems:
            return {}
        
        problem_data = self.active_problems.pop(problem_id)
        duration = (datetime.now() - problem_data['start_time']).total_seconds()
        
        return {
            'duration': duration,
            'attempts': len(problem_data['attempts']),
            'success': success,
            'strategy': 'tracked_solution'
        }
    
    async def _analyze_patterns(self):
        """패턴 분석"""
        insights = self.get_learning_insights()
        
        print(f"\n📊 학습 패턴 분석:")
        print(f"- 총 경험: {insights['total_experiences']}")
        print(f"- 성공률: {insights['success_rate']:.1%}")
        print(f"- 가장 효과적인 전략: {insights['most_effective_strategies'][:3]}")
        
        # 새로운 패턴 발견
        await self._discover_new_patterns()
    
    async def _discover_new_patterns(self):
        """새로운 패턴 발견"""
        # 반복되는 해결책에서 패턴 찾기
        for error_hash, solutions in self.error_solutions.items():
            if len(solutions) > 5:  # 5번 이상 발생한 오류
                successful_solutions = [s for s in solutions if s['success']]
                if successful_solutions:
                    # 공통 패턴 추출
                    common_strategy = max(
                        set(s['strategy'] for s in successful_solutions),
                        key=lambda x: sum(1 for s in successful_solutions if s['strategy'] == x)
                    )
                    
                    await self.collect_code_pattern(
                        f"auto_discovered_{error_hash}",
                        f"# 자동 발견 패턴: {common_strategy}",
                        f"오류 {error_hash}에 대한 효과적인 해결 패턴",
                        effectiveness=0.8
                    )
    
    def _generate_learning_report(self):
        """학습 보고서 생성"""
        report_path = self.knowledge_base_path / f"learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        insights = self.get_learning_insights()
        best_practices = self.get_best_practices()
        
        report = f"""# 개발 경험 학습 보고서

## 📊 전체 통계
- 총 수집된 경험: {self.total_experiences}
- 성공적인 적용: {self.successful_applications}
- 성공률: {insights['success_rate']:.1%}
- 커뮤니티 기여: {insights['community_contribution']}

## 🏆 가장 효과적인 전략
"""
        
        for strategy, score in insights['most_effective_strategies'][:10]:
            report += f"- {strategy}: {score:.1f}점\n"
        
        report += "\n## 📈 성능 최적화 요약\n"
        if insights['optimization_summary']:
            report += f"- 평균 개선율: {insights['optimization_summary']['average_improvement']:.1f}%\n"
            report += f"- 총 최적화 수: {insights['optimization_summary']['total_optimizations']}\n"
        
        report += "\n## 💡 모범 사례\n"
        for practice in best_practices:
            report += f"- [{practice['type']}] {practice.get('name', practice.get('strategy', practice.get('method', 'Unknown')))}\n"
        
        report += f"\n## 🤖 AI 창의성 점수: {insights['ai_creativity_score']:.1f}/10\n"
        
        report += "\n## 📚 학습된 패턴\n"
        for pattern_name, frequency in list(insights['common_patterns'])[:20]:
            report += f"- {pattern_name}: {frequency}회 사용\n"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 학습 보고서 생성: {report_path}")

# 싱글톤 인스턴스
_collector_instance = None

def get_experience_collector():
    """경험 수집기 인스턴스 반환"""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = DevelopmentExperienceCollector()
    return _collector_instance

# 간편 사용을 위한 래퍼 함수들
async def collect_error_solution(error: Dict[str, Any], solution: Dict[str, Any], success: bool):
    """오류 해결책 수집"""
    collector = get_experience_collector()
    await collector.collect_error_solution(error, solution, success)

async def collect_game_mechanic(name: str, implementation: Dict[str, Any], metrics: Dict[str, Any] = None):
    """게임 메카닉 수집"""
    collector = get_experience_collector()
    await collector.collect_game_mechanic(name, implementation, metrics)

async def collect_optimization(optimization: Dict[str, Any]):
    """성능 최적화 수집"""
    collector = get_experience_collector()
    await collector.collect_performance_optimization(optimization)

def search_solutions(problem: Dict[str, Any]) -> List[Dict[str, Any]]:
    """유사한 문제 해결책 검색"""
    collector = get_experience_collector()
    return collector.search_similar_problems(problem)

def get_best_practices(category: str = None) -> List[Dict[str, Any]]:
    """모범 사례 조회"""
    collector = get_experience_collector()
    return collector.get_best_practices(category)