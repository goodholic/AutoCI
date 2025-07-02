#!/usr/bin/env python3
"""
AutoCI 학습 통합 모듈
AutoCI의 모든 개발 활동을 실시간으로 학습 데이터로 변환합니다.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .realtime_learning_integrator import get_realtime_integrator
from .development_experience_collector import get_experience_collector

logger = logging.getLogger(__name__)

class AutoCILearningIntegration:
    """AutoCI 학습 통합 관리자"""
    
    def __init__(self):
        self.integrator = get_realtime_integrator()
        self.collector = get_experience_collector()
        self.is_active = False
        
    async def initialize(self, continuous_learning_system=None, ai_controller=None):
        """학습 통합 초기화"""
        logger.info("AutoCI 학습 통합 초기화 시작...")
        
        # 시스템 연결
        await self.integrator.connect_systems(
            continuous_learning_system=continuous_learning_system,
            experience_collector=self.collector,
            ai_model_controller=ai_controller
        )
        
        # 실시간 처리 시작
        await self.integrator.start_realtime_processing()
        self.is_active = True
        
        logger.info("✅ AutoCI 학습 통합 초기화 완료")
    
    async def on_build_success(self, project_info: Dict[str, Any]):
        """빌드 성공 시 학습"""
        if not self.is_active:
            return
        
        # 성공적인 빌드 구성을 학습 데이터로
        await self.integrator.on_new_experience('game_mechanic', {
            'name': '빌드 구성',
            'description': f"{project_info.get('name', 'Godot 프로젝트')} 빌드 성공",
            'code_snippet': project_info.get('build_script', ''),
            'performance': {
                'build_time': project_info.get('build_time', 0),
                'output_size': project_info.get('output_size', 0)
            },
            'effectiveness': 1.0
        })
    
    async def on_error_fixed(self, error: Dict[str, Any], solution: Dict[str, Any]):
        """오류 수정 시 학습"""
        if not self.is_active:
            return
        
        # 오류 해결 경험 수집
        await self.collector.collect_error_solution(error, solution, True)
        
        # 학습 데이터로 변환
        await self.integrator.on_new_experience('error_solution', {
            'error': error,
            'solution': solution,
            'success': True,
            'effectiveness': solution.get('confidence', 0.8)
        })
    
    async def on_optimization_applied(self, optimization: Dict[str, Any]):
        """최적화 적용 시 학습"""
        if not self.is_active:
            return
        
        # 최적화 경험 수집
        await self.collector.collect_performance_optimization(optimization)
        
        # 학습 데이터로 변환
        await self.integrator.on_new_experience('performance_opt', optimization)
    
    async def on_pattern_discovered(self, pattern: Dict[str, Any]):
        """패턴 발견 시 학습"""
        if not self.is_active:
            return
        
        # 코드 패턴 수집
        await self.collector.collect_code_pattern(
            pattern.get('name', 'pattern'),
            pattern.get('code', ''),
            pattern.get('use_case', 'general'),
            pattern.get('effectiveness', 0.8)
        )
        
        # 학습 데이터로 변환
        await self.integrator.on_new_experience('code_pattern', pattern)
    
    async def on_ai_insight(self, insight: Dict[str, Any]):
        """AI 인사이트 발생 시 학습"""
        if not self.is_active:
            return
        
        # AI 발견 수집
        await self.collector.collect_ai_discovery(insight)
        
        # 학습 데이터로 변환
        await self.integrator.on_new_experience('ai_discovery', insight)
    
    async def on_community_solution(self, problem: str, solution: Dict[str, Any], source: str):
        """커뮤니티 솔루션 발견 시 학습"""
        if not self.is_active:
            return
        
        # 커뮤니티 솔루션 수집
        await self.collector.collect_community_solution(problem, solution, source)
        
        # 학습 데이터로 변환
        await self.integrator.on_new_experience('community_solution', {
            'problem': problem,
            'solution': solution,
            'source': source,
            'effectiveness': solution.get('votes', 0) / 10  # 투표수 기반 효과성
        })
    
    def get_learning_status(self) -> Dict[str, Any]:
        """학습 상태 조회"""
        return {
            'active': self.is_active,
            'integration_status': self.integrator.get_integration_status(),
            'collection_insights': self.collector.get_learning_insights(),
            'best_practices': self.collector.get_best_practices()
        }
    
    async def generate_learning_report(self) -> Dict[str, Any]:
        """학습 보고서 생성"""
        # 통합 보고서
        await self.integrator._generate_integration_report()
        
        # 수집 보고서
        self.collector._generate_learning_report()
        
        # 특화 데이터셋 생성
        datasets = {
            'error_solutions': await self.integrator.create_specialized_training_dataset('error'),
            'game_mechanics': await self.integrator.create_specialized_training_dataset('game'),
            'optimizations': await self.integrator.create_specialized_training_dataset('performance'),
            'ai_insights': await self.integrator.create_specialized_training_dataset('ai')
        }
        
        return {
            'datasets_created': len(datasets),
            'total_qa_pairs': sum(d['statistics']['total_pairs'] for d in datasets.values()),
            'report_generated': True
        }
    
    async def shutdown(self):
        """학습 통합 종료"""
        if self.is_active:
            logger.info("AutoCI 학습 통합 종료 중...")
            
            # 최종 보고서 생성
            await self.generate_learning_report()
            
            # 통합 중지
            await self.integrator.stop_realtime_processing()
            
            # 수집기 중지
            self.collector.stop_monitoring()
            
            self.is_active = False
            logger.info("✅ AutoCI 학습 통합 종료 완료")

# 싱글톤 인스턴스
_autoci_learning = None

def get_autoci_learning():
    """AutoCI 학습 통합 인스턴스 반환"""
    global _autoci_learning
    if _autoci_learning is None:
        _autoci_learning = AutoCILearningIntegration()
    return _autoci_learning

# AutoCI 통합을 위한 간편 함수들
async def init_learning(continuous_learning_system=None, ai_controller=None):
    """학습 초기화"""
    learning = get_autoci_learning()
    await learning.initialize(continuous_learning_system, ai_controller)
    return learning

async def learn_from_build(project_info: Dict[str, Any]):
    """빌드에서 학습"""
    learning = get_autoci_learning()
    await learning.on_build_success(project_info)

async def learn_from_error(error: Dict[str, Any], solution: Dict[str, Any]):
    """오류 해결에서 학습"""
    learning = get_autoci_learning()
    await learning.on_error_fixed(error, solution)

async def learn_from_optimization(optimization: Dict[str, Any]):
    """최적화에서 학습"""
    learning = get_autoci_learning()
    await learning.on_optimization_applied(optimization)

async def learn_from_pattern(pattern: Dict[str, Any]):
    """패턴에서 학습"""
    learning = get_autoci_learning()
    await learning.on_pattern_discovered(pattern)

async def learn_from_ai(insight: Dict[str, Any]):
    """AI 인사이트에서 학습"""
    learning = get_autoci_learning()
    await learning.on_ai_insight(insight)

def get_learning_status():
    """학습 상태 조회"""
    learning = get_autoci_learning()
    return learning.get_learning_status()

async def shutdown_learning():
    """학습 종료"""
    learning = get_autoci_learning()
    await learning.shutdown()

# AutoCI 메인 프로세스에 통합하기 위한 데코레이터
def with_learning(func):
    """함수 실행 결과를 자동으로 학습하는 데코레이터"""
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        
        # 결과에서 학습 가능한 데이터 추출
        if isinstance(result, dict):
            if 'error' in result and 'solution' in result:
                await learn_from_error(result['error'], result['solution'])
            elif 'optimization' in result:
                await learn_from_optimization(result['optimization'])
            elif 'pattern' in result:
                await learn_from_pattern(result['pattern'])
            elif 'ai_insight' in result:
                await learn_from_ai(result['ai_insight'])
        
        return result
    
    return wrapper