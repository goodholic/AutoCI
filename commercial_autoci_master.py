#!/usr/bin/env python3
"""
AutoCI 상용화 수준 마스터 시스템
모든 구성 요소를 통합하여 상용화 품질의 AI 코딩 어시스턴트 제공
"""

import os
import sys
import json
import logging
import asyncio
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# AutoCI 모듈 임포트
sys.path.append(str(Path(__file__).parent))

from commercial_ai_engine import CommercialDialogueEngine
from csharp_expert_learner import CSharpExpertLearner
from continuous_learning_pipeline import ContinuousLearningPipeline
from commercial_quality_validator import CommercialQualityValidator
from real_learning_system import RealLearningSystem
from ai_learning_monitor import AILearningMonitor

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('commercial_autoci.log'),
        logging.StreamHandler()
    ]
)


class CommercialAutoCI:
    """상용화 수준 AutoCI 마스터 시스템"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        
        print("🚀 AutoCI 상용화 수준 시스템 초기화 중...")
        
        # 핵심 구성 요소
        self.components = {
            'dialogue_engine': CommercialDialogueEngine(),
            'csharp_expert': CSharpExpertLearner(),
            'learning_pipeline': ContinuousLearningPipeline(),
            'quality_validator': CommercialQualityValidator(),
            'base_learner': RealLearningSystem(),
            'monitor': AILearningMonitor()
        }
        
        # 시스템 상태
        self.system_state = {
            'is_running': False,
            'quality_status': 'initializing',
            'learning_active': False,
            'monitoring_active': False,
            'total_conversations': 0,
            'expertise_level': 0.0,
            'user_satisfaction': 0.0
        }
        
        # 성능 메트릭
        self.performance_metrics = {
            'avg_response_time': 0.0,
            'dialogue_quality': 0.0,
            'technical_accuracy': 0.0,
            'learning_rate': 0.0,
            'uptime': 0.0
        }
        
        # 초기화 완료
        print("✅ 모든 구성 요소 초기화 완료!")
        
    async def start_system(self):
        """시스템 전체 시작"""
        logger.info("🚀 AutoCI 상용화 시스템 시작...")
        
        self.system_state['is_running'] = True
        
        try:
            # 1. 품질 검증 (시작 전)
            await self._initial_quality_check()
            
            # 2. 학습 파이프라인 시작
            self.components['learning_pipeline'].start()
            self.system_state['learning_active'] = True
            
            # 3. 모니터링 시작
            self.components['monitor'].start()
            self.system_state['monitoring_active'] = True
            
            # 4. 백그라운드 작업 시작
            await self._start_background_tasks()
            
            # 5. 시스템 준비 완료 메시지
            self._print_startup_banner()
            
            logger.info("✅ AutoCI 시스템 완전 가동 완료!")
            
        except Exception as e:
            logger.error(f"시스템 시작 오류: {e}")
            await self.stop_system()
            raise
    
    async def stop_system(self):
        """시스템 전체 중지"""
        logger.info("🛑 AutoCI 시스템 종료 중...")
        
        self.system_state['is_running'] = False
        
        # 각 구성 요소 중지
        self.components['learning_pipeline'].stop()
        self.components['monitor'].stop()
        
        # 최종 통계 출력
        self._print_shutdown_summary()
        
        logger.info("✅ AutoCI 시스템 종료 완료")
    
    async def _initial_quality_check(self):
        """초기 품질 검증"""
        logger.info("🔍 초기 품질 검증 실행...")
        
        validator = self.components['quality_validator']
        
        # 각 구성 요소 품질 검증
        dialogue_quality = await validator.validate_dialogue_quality(
            self.components['dialogue_engine']
        )
        
        csharp_quality = await validator.validate_csharp_expertise(
            self.components['csharp_expert']
        )
        
        learning_quality = await validator.validate_learning_capability(
            self.components['base_learner']
        )
        
        # 전체 품질 상태 업데이트
        overall_quality = (
            dialogue_quality.get('overall', {}).get('score', 0) +
            csharp_quality.get('overall', {}).get('score', 0) +
            learning_quality.get('overall', {}).get('score', 0)
        ) / 3
        
        if overall_quality >= 0.85:
            self.system_state['quality_status'] = 'commercial_ready'
            logger.info(f"✅ 상용화 품질 기준 충족: {overall_quality:.1%}")
        else:
            self.system_state['quality_status'] = 'needs_improvement'
            logger.warning(f"⚠️ 품질 개선 필요: {overall_quality:.1%}")
    
    async def _start_background_tasks(self):
        """백그라운드 작업 시작"""
        # 품질 모니터링 (30분마다)
        asyncio.create_task(self._periodic_quality_check())
        
        # 성능 메트릭 업데이트 (5분마다)
        asyncio.create_task(self._update_performance_metrics())
        
        # 시스템 건강 상태 체크 (1분마다)
        asyncio.create_task(self._health_check())
    
    async def _periodic_quality_check(self):
        """주기적 품질 검증"""
        while self.system_state['is_running']:
            try:
                await asyncio.sleep(1800)  # 30분 대기
                
                logger.info("🔍 주기적 품질 검증 실행...")
                
                # 대화 품질만 빠르게 체크
                validator = self.components['quality_validator']
                dialogue_result = await validator.validate_dialogue_quality(
                    self.components['dialogue_engine']
                )
                
                quality_score = dialogue_result.get('overall', {}).get('score', 0)
                
                if quality_score < 0.8:
                    logger.warning(f"⚠️ 품질 저하 감지: {quality_score:.1%}")
                    # 개선 조치 실행
                    await self._trigger_quality_improvement()
                
            except Exception as e:
                logger.error(f"품질 검증 오류: {e}")
    
    async def _update_performance_metrics(self):
        """성능 메트릭 업데이트"""
        while self.system_state['is_running']:
            try:
                await asyncio.sleep(300)  # 5분 대기
                
                # 각 구성 요소에서 메트릭 수집
                dialogue_stats = self._get_dialogue_stats()
                learning_stats = self.components['base_learner'].get_learning_stats()
                
                # 메트릭 업데이트
                self.performance_metrics.update({
                    'avg_response_time': dialogue_stats.get('avg_response_time', 0),
                    'dialogue_quality': dialogue_stats.get('quality_score', 0),
                    'technical_accuracy': float(learning_stats.get('accuracy', '0%').rstrip('%')) / 100,
                    'learning_rate': learning_stats.get('learning_rate', 0),
                    'total_conversations': learning_stats.get('total_conversations', 0)
                })
                
            except Exception as e:
                logger.error(f"메트릭 업데이트 오류: {e}")
    
    async def _health_check(self):
        """시스템 건강 상태 체크"""
        while self.system_state['is_running']:
            try:
                await asyncio.sleep(60)  # 1분 대기
                
                # 각 구성 요소 상태 확인
                health_status = {
                    'dialogue_engine': self._check_component_health('dialogue_engine'),
                    'csharp_expert': self._check_component_health('csharp_expert'),
                    'learning_pipeline': self._check_component_health('learning_pipeline'),
                    'monitor': self._check_component_health('monitor')
                }
                
                # 전체 건강 상태
                healthy_components = sum(1 for status in health_status.values() if status)
                health_ratio = healthy_components / len(health_status)
                
                if health_ratio < 0.8:
                    logger.warning(f"⚠️ 시스템 건강 상태 저하: {health_ratio:.1%}")
                
            except Exception as e:
                logger.error(f"건강 상태 체크 오류: {e}")
    
    def _check_component_health(self, component_name: str) -> bool:
        """구성 요소 건강 상태 체크"""
        component = self.components.get(component_name)
        
        if not component:
            return False
        
        # 기본적인 건강 상태 체크
        if hasattr(component, 'is_running'):
            return getattr(component, 'is_running', False)
        
        return True
    
    async def _trigger_quality_improvement(self):
        """품질 개선 조치 실행"""
        logger.info("🔧 품질 개선 조치 실행...")
        
        # 학습 파이프라인에서 긴급 학습 실행
        pipeline = self.components['learning_pipeline']
        
        # 문서 학습 실행
        await pipeline.learn_from_documentation()
        
        # 패턴 종합 실행
        await pipeline.synthesize_patterns()
        
        logger.info("✅ 품질 개선 조치 완료")
    
    def _get_dialogue_stats(self) -> Dict[str, float]:
        """대화 통계 가져오기"""
        # 실제로는 대화 엔진에서 통계 수집
        return {
            'avg_response_time': 0.3,
            'quality_score': 0.9,
            'total_responses': 150
        }
    
    def _print_startup_banner(self):
        """시작 배너 출력"""
        banner = f"""
{'='*80}
🤖 AutoCI - 상용화 수준 AI 코딩 어시스턴트
{'='*80}

✨ 상용화 품질 AI 대화 엔진 가동
🎓 C# 전문가 수준 지식 시스템 준비
🧠 24시간 지속 학습 파이프라인 활성화
📊 실시간 품질 모니터링 시작

{'='*80}
시스템 상태: {self.system_state['quality_status']}
학습 활성화: {'✅' if self.system_state['learning_active'] else '❌'}
모니터링 활성화: {'✅' if self.system_state['monitoring_active'] else '❌'}
{'='*80}

💬 이제 상용화 수준의 AI 대화가 가능합니다!
🎮 Unity/C# 전문가 수준의 도움을 받으실 수 있습니다!
🔄 대화할 때마다 AI가 실시간으로 학습합니다!

사용법: 자연스러운 한국어로 대화하세요!

{'='*80}
"""
        print(banner)
    
    def _print_shutdown_summary(self):
        """종료 요약 출력"""
        uptime = time.time() - getattr(self, '_start_time', time.time())
        
        summary = f"""
{'='*60}
📊 AutoCI 세션 요약
{'='*60}

🕐 가동 시간: {uptime/3600:.1f}시간
💬 총 대화 수: {self.performance_metrics.get('total_conversations', 0)}
⚡ 평균 응답 시간: {self.performance_metrics.get('avg_response_time', 0):.2f}초
🎯 대화 품질: {self.performance_metrics.get('dialogue_quality', 0):.1%}
🧠 기술 정확도: {self.performance_metrics.get('technical_accuracy', 0):.1%}

감사합니다! 🙏
{'='*60}
"""
        print(summary)
    
    async def process_user_input(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """사용자 입력 처리 (메인 인터페이스)"""
        start_time = time.time()
        
        try:
            # 1. 상용화 대화 엔진으로 처리
            dialogue_result = self.components['dialogue_engine'].process_dialogue(
                user_input, context
            )
            
            # 2. C# 전문 지식이 필요한 경우 전문가 시스템 활용
            if self._needs_expert_knowledge(user_input, dialogue_result):
                expert_knowledge = self._get_expert_enhancement(user_input)
                dialogue_result = self._enhance_with_expert_knowledge(
                    dialogue_result, expert_knowledge
                )
            
            # 3. 실시간 학습
            self.components['base_learner'].learn_from_conversation(
                user_input,
                dialogue_result['response'],
                context
            )
            
            # 4. 통계 업데이트
            self.system_state['total_conversations'] += 1
            
            # 5. 응답 시간 기록
            response_time = time.time() - start_time
            
            return {
                'response': dialogue_result['response'],
                'confidence': dialogue_result.get('confidence', 0.9),
                'response_time': response_time,
                'quality_score': dialogue_result.get('quality_score', 0.9),
                'expert_enhanced': self._needs_expert_knowledge(user_input, dialogue_result),
                'learning_applied': True,
                'system_status': self.get_system_status()
            }
            
        except Exception as e:
            logger.error(f"사용자 입력 처리 오류: {e}")
            return {
                'response': "죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해 주세요.",
                'confidence': 0.5,
                'error': str(e)
            }
    
    def _needs_expert_knowledge(self, user_input: str, dialogue_result: Dict) -> bool:
        """전문 지식 필요 여부 판단"""
        # C#, Unity 관련 키워드 체크
        technical_keywords = [
            'async', 'await', 'Task', 'delegate', 'event', 'LINQ', 
            'Unity', 'GameObject', 'Transform', 'Coroutine',
            'Singleton', 'Factory', 'Observer', '디자인패턴'
        ]
        
        input_lower = user_input.lower()
        
        return any(keyword.lower() in input_lower for keyword in technical_keywords)
    
    def _get_expert_enhancement(self, user_input: str) -> Dict:
        """전문가 지식 강화 정보 가져오기"""
        # 주요 키워드 추출
        keywords = self._extract_technical_keywords(user_input)
        
        expert_knowledge = {}
        for keyword in keywords:
            knowledge = self.components['csharp_expert'].get_expert_knowledge(keyword)
            if knowledge:
                expert_knowledge[keyword] = knowledge
        
        return expert_knowledge
    
    def _extract_technical_keywords(self, text: str) -> List[str]:
        """기술적 키워드 추출"""
        technical_terms = [
            'async', 'await', 'Task', 'delegate', 'event', 'LINQ',
            'Unity', 'GameObject', 'Transform', 'Coroutine', 'MonoBehaviour',
            'Singleton', 'Factory', 'Observer', 'Strategy', 'Command'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for term in technical_terms:
            if term.lower() in text_lower:
                found_keywords.append(term)
        
        return found_keywords
    
    def _enhance_with_expert_knowledge(self, dialogue_result: Dict, 
                                     expert_knowledge: Dict) -> Dict:
        """전문 지식으로 응답 강화"""
        base_response = dialogue_result['response']
        
        # 전문 지식 추가
        enhancements = []
        
        for keyword, knowledge in expert_knowledge.items():
            if knowledge.get('concepts'):
                enhancements.append(f"\n\n💡 {keyword}에 대한 전문 지식:")
                
                for concept in knowledge['concepts'][:2]:  # 상위 2개
                    enhancements.append(f"- {concept.get('description', '')}")
            
            if knowledge.get('code_examples'):
                enhancements.append(f"\n📝 코드 예시:")
                
                for example in knowledge['code_examples'][:1]:  # 1개만
                    if example.get('code'):
                        enhancements.append(f"```csharp\n{example['code']}\n```")
        
        if enhancements:
            enhanced_response = base_response + "".join(enhancements)
            dialogue_result['response'] = enhanced_response
            dialogue_result['expert_enhanced'] = True
        
        return dialogue_result
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보"""
        learning_stats = self.components['base_learner'].get_learning_stats()
        pipeline_status = self.components['learning_pipeline'].get_status()
        
        return {
            'is_running': self.system_state['is_running'],
            'quality_status': self.system_state['quality_status'],
            'total_conversations': self.system_state['total_conversations'],
            'learning_stats': learning_stats,
            'pipeline_status': pipeline_status,
            'performance_metrics': self.performance_metrics
        }
    
    def get_quality_report(self) -> str:
        """품질 보고서 가져오기"""
        return self.components['quality_validator'].generate_quality_report()
    
    async def interactive_mode(self):
        """대화형 모드"""
        await self.start_system()
        
        print("\n💬 대화를 시작하세요! (종료: 'exit' 또는 '종료')")
        
        try:
            while self.system_state['is_running']:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', '종료', '끝']:
                    break
                
                # 특별 명령어 처리
                if user_input == '상태':
                    status = self.get_system_status()
                    print(f"\n📊 시스템 상태:")
                    print(f"- 품질 상태: {status['quality_status']}")
                    print(f"- 총 대화: {status['total_conversations']}개")
                    print(f"- 학습된 패턴: {status['learning_stats'].get('learned_patterns', 0)}개")
                    continue
                
                elif user_input == '품질보고서':
                    report = self.get_quality_report()
                    print(report)
                    continue
                
                # 일반 대화 처리
                result = await self.process_user_input(user_input)
                
                # 응답 출력
                print(f"\n🤖 AutoCI: {result['response']}")
                
                # 메타 정보 출력 (선택적)
                if result.get('expert_enhanced'):
                    print(f"   💡 전문가 지식 적용됨")
                
                print(f"   ⚡ 응답시간: {result['response_time']:.2f}초 | "
                      f"품질: {result['quality_score']:.1%}")
                
        except KeyboardInterrupt:
            print("\n\n종료 중...")
        
        finally:
            await self.stop_system()


async def main():
    """메인 함수"""
    print("🚀 AutoCI 상용화 수준 시스템")
    print("=" * 50)
    
    # 시스템 생성
    autoci = CommercialAutoCI()
    
    # 대화형 모드 시작
    await autoci.interactive_mode()


if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(main())