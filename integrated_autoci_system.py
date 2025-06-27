#!/usr/bin/env python3
"""
AutoCI 통합 시스템
한국어 AI + 실제 학습 + 모니터링을 하나로 통합
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# AutoCI 모듈 임포트
sys.path.append(str(Path(__file__).parent))

from advanced_korean_ai import AdvancedKoreanAI
from real_learning_system import RealLearningSystem
from ai_learning_monitor import AILearningMonitor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedAutoCI:
    """통합 AutoCI 시스템"""
    
    def __init__(self):
        print("🚀 AutoCI 통합 시스템 초기화 중...")
        
        # 각 컴포넌트 초기화
        self.korean_ai = AdvancedKoreanAI()
        self.learning_system = RealLearningSystem()
        self.monitor = AILearningMonitor()
        
        # 통합 상태
        self.is_running = False
        self.conversation_count = 0
        self.learning_enabled = True
        
        # 대화 컨텍스트
        self.current_context = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'project_path': None,
            'user_profile': {}
        }
        
        print("✅ 모든 시스템이 준비되었습니다!")
        
    def start(self):
        """통합 시스템 시작"""
        self.is_running = True
        
        # 백그라운드 학습 시작
        self.learning_system.start_background_learning()
        logger.info("🧠 백그라운드 학습 시작")
        
        # 모니터링 시작
        self.monitor.start()
        logger.info("📊 1분 모니터링 시작")
        
        # 초기 인사
        self.print_welcome()
        
    def stop(self):
        """통합 시스템 중지"""
        self.is_running = False
        
        # 각 시스템 중지
        self.learning_system.stop_background_learning()
        self.monitor.stop()
        
        # 최종 통계 출력
        self.print_final_stats()
        
    def print_welcome(self):
        """환영 메시지"""
        print("\n" + "="*70)
        print("🤖 AutoCI - 진짜 학습하는 한국어 AI 코딩 어시스턴트")
        print("="*70)
        print("✨ ChatGPT처럼 자연스러운 한국어 대화")
        print("🧠 실제로 대화에서 학습하여 계속 똑똑해짐")
        print("📊 1분마다 학습 상태 자동 모니터링")
        print("🎮 Unity/C# 전문가 수준의 도움")
        print("="*70)
        print("💬 자유롭게 대화해보세요! (종료: 'exit' 또는 '종료')")
        print()
        
    def process_input(self, user_input: str) -> str:
        """사용자 입력 처리"""
        # 종료 명령 체크
        if user_input.lower() in ['exit', 'quit', '종료', '끝', '나가기']:
            return None
            
        # 한국어 AI로 분석
        analysis = self.korean_ai.analyze_input(user_input)
        
        # 학습 시스템에서 유사한 대화 검색
        similar_conversations = self.learning_system.get_similar_conversations(user_input, k=3)
        
        # 응답 생성 (학습된 내용 참고)
        response = self._generate_integrated_response(analysis, similar_conversations)
        
        # 대화를 학습 시스템에 저장
        if self.learning_enabled:
            learning_result = self.learning_system.learn_from_conversation(
                user_input, 
                response,
                self.current_context
            )
            
            # 대화 카운트 증가
            self.conversation_count += 1
            
            # 주기적으로 학습 상태 표시
            if self.conversation_count % 5 == 0:
                self._show_learning_progress()
                
        return response
        
    def _generate_integrated_response(self, analysis: Dict, similar_convs: List[Dict]) -> str:
        """통합 응답 생성"""
        # 기본 응답 생성
        base_response = self.korean_ai.generate_response(analysis)
        
        # 학습된 내용이 있으면 참고
        if similar_convs and analysis['intent'] in ['question', 'request']:
            # 가장 평가가 좋았던 응답 참고
            best_conv = max(similar_convs, key=lambda x: x.get('feedback', 0))
            if best_conv.get('feedback', 0) > 0.8:
                # 이전에 좋은 평가를 받은 응답이 있음
                base_response += f"\n\n💡 이전에 비슷한 질문이 있었어요:\n"
                base_response += f"Q: {best_conv['user_input'][:50]}...\n"
                base_response += f"A: {best_conv['ai_response'][:100]}..."
                
        # 현재 학습 중인 내용 언급
        if analysis['topic'] in ['unity', 'csharp', 'coding']:
            stats = self.learning_system.get_learning_stats()
            if stats['topics_learned']:
                base_response += f"\n\n📚 최근에 {', '.join(stats['topics_learned'][:3])}에 대해 학습했어요!"
                
        return base_response
        
    def _show_learning_progress(self):
        """학습 진행 상황 표시"""
        stats = self.learning_system.get_learning_stats()
        
        print("\n" + "-"*50)
        print("🧠 학습 진행 상황:")
        print(f"  • 총 대화: {stats['total_conversations']}개")
        print(f"  • 학습한 패턴: {stats['learned_patterns']}개")
        print(f"  • 정확도: {stats['accuracy']}")
        print("-"*50 + "\n")
        
    def get_feedback(self, feedback_text: str):
        """사용자 피드백 처리"""
        # 피드백 분석
        is_positive = any(word in feedback_text for word in 
                         ['좋아', '고마워', '감사', '도움', '최고', '훌륭'])
        
        feedback_score = 0.9 if is_positive else 0.3
        
        # 마지막 대화에 대한 피드백으로 학습
        if self.conversation_count > 0:
            self.learning_system.learn_from_feedback(
                self.conversation_count,
                feedback_score
            )
            
        # 한국어 AI도 피드백 학습
        self.korean_ai.learn_from_feedback(feedback_text, is_positive)
        
        return "피드백 감사합니다! 더 나은 도움을 드릴 수 있도록 노력하겠습니다. 😊"
        
    def show_status(self) -> str:
        """현재 상태 표시"""
        # 학습 통계
        learning_stats = self.learning_system.get_learning_stats()
        
        # 모니터링 데이터
        if hasattr(self.monitor, 'metrics_history') and self.monitor.metrics_history:
            latest_metrics = self.monitor.metrics_history[-1]
            cpu_usage = latest_metrics.get('cpu_percent', 0)
            memory_usage = latest_metrics.get('memory', {}).get('percent', 0)
        else:
            cpu_usage = 0
            memory_usage = 0
            
        status = f"""
📊 AutoCI 상태 리포트
{'='*50}
🧠 학습 상태:
  • 총 대화 수: {learning_stats['total_conversations']}
  • 학습된 패턴: {learning_stats['learned_patterns']}
  • 정확도: {learning_stats['accuracy']}
  • 학습률: {learning_stats['learning_rate']}
  
💻 시스템 상태:
  • CPU 사용률: {cpu_usage:.1f}%
  • 메모리 사용률: {memory_usage:.1f}%
  • 현재 세션 대화: {self.conversation_count}개
  
📚 최근 학습 주제:
  {', '.join(learning_stats.get('topics_learned', [])[:5])}
  
🔍 자주 발생한 에러:
"""
        
        error_patterns = learning_stats.get('error_patterns', {})
        for error_type, count in list(error_patterns.items())[:3]:
            status += f"  • {error_type}: {count}회\n"
            
        return status
        
    def print_final_stats(self):
        """최종 통계 출력"""
        print("\n" + "="*70)
        print("👋 AutoCI 사용 통계")
        print("="*70)
        print(self.show_status())
        print("="*70)
        print("감사합니다! 다음에 또 만나요! 😊")
        
    def interactive_mode(self):
        """대화형 모드"""
        self.start()
        
        try:
            while self.is_running:
                # 사용자 입력
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                    
                # 피드백 처리
                if user_input.startswith("피드백:"):
                    feedback_response = self.get_feedback(user_input[4:].strip())
                    print(f"🤖 AutoCI: {feedback_response}")
                    continue
                    
                # 상태 확인
                if user_input in ['상태', 'status', '통계']:
                    print(self.show_status())
                    continue
                    
                # 일반 대화 처리
                response = self.process_input(user_input)
                
                if response is None:
                    break
                    
                # 응답 출력
                print(f"\n🤖 AutoCI: {response}")
                
                # 가끔 학습 팁 제공
                if self.conversation_count % 10 == 0:
                    print("\n💡 Tip: '피드백: [메시지]'로 저의 답변을 평가해주시면 더 잘 학습할 수 있어요!")
                    
        except KeyboardInterrupt:
            print("\n\n종료 중...")
            
        finally:
            self.stop()


def main():
    """메인 함수"""
    # 통합 시스템 생성
    autoci = IntegratedAutoCI()
    
    # 대화형 모드 시작
    autoci.interactive_mode()


if __name__ == "__main__":
    main()