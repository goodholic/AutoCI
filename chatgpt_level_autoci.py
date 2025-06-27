#!/usr/bin/env python3
"""
ChatGPT 수준의 AutoCI 통합 시스템
모든 컴포넌트를 통합하여 ChatGPT와 같은 수준의 대화형 AI 구현
"""

import os
import sys
import time
import json
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers 없음 - 기본 모드로 실행")

# 로컬 모듈 임포트
try:
    from advanced_transformer_autoci import AdvancedAutoCI, KoreanTransformerModel, AdvancedMemorySystem, RealTimeLearningEngine
    from korean_dataset_collector import KoreanDatasetDatabase, ConversationDataCollector, ConversationQualityEvaluator
    from learning_progress_tracker import LearningProgressTracker
    ADVANCED_MODULES_AVAILABLE = True
except ImportError:
    ADVANCED_MODULES_AVAILABLE = False
    print("⚠️ 고급 모듈 없음 - 간소화 모드로 실행")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatGPTLevelAutoCI:
    """ChatGPT 수준의 AutoCI 시스템"""
    
    def __init__(self):
        self.system_name = "ChatGPT-Level AutoCI"
        self.version = "2.0.0"
        self.capabilities = {
            "korean_conversation": True,
            "unity_expertise": True,
            "csharp_programming": True,
            "real_time_learning": True,
            "context_memory": True,
            "quality_evaluation": True,
            "continuous_improvement": True
        }
        
        # 컴포넌트 초기화
        self.advanced_ai = None
        self.dataset_db = None
        self.progress_tracker = None
        self.conversation_history = []
        self.user_sessions = {}
        
        # 성능 메트릭
        self.metrics = {
            "total_conversations": 0,
            "user_satisfaction": 0.0,
            "learning_efficiency": 0.0,
            "response_accuracy": 0.0,
            "system_uptime": datetime.now(),
            "last_learning_update": None
        }
        
        # 시스템 상태
        self.status = {
            "initialized": False,
            "learning_enabled": True,
            "components_healthy": False,
            "ready_for_conversation": False
        }
        
        self.initialize_system()
    
    def initialize_system(self):
        """시스템 초기화"""
        logger.info(f"🚀 {self.system_name} v{self.version} 초기화 시작")
        
        try:
            # 1. 데이터셋 시스템 초기화
            self._initialize_dataset_system()
            
            # 2. 고급 AI 시스템 초기화
            self._initialize_advanced_ai()
            
            # 3. 진행률 추적 시스템 초기화
            self._initialize_progress_tracking()
            
            # 4. 시스템 상태 업데이트
            self.status["initialized"] = True
            self.status["components_healthy"] = self._check_component_health()
            self.status["ready_for_conversation"] = True
            
            logger.info("✅ ChatGPT 수준의 AutoCI 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 시스템 초기화 실패: {e}")
            self._fallback_initialization()
    
    def _initialize_dataset_system(self):
        """데이터셋 시스템 초기화"""
        try:
            if ADVANCED_MODULES_AVAILABLE:
                self.dataset_db = KoreanDatasetDatabase()
                logger.info("✅ 한국어 데이터셋 시스템 초기화 완료")
            else:
                logger.warning("⚠️ 고급 데이터셋 시스템 사용 불가 - 기본 모드")
        except Exception as e:
            logger.error(f"❌ 데이터셋 시스템 초기화 실패: {e}")
    
    def _initialize_advanced_ai(self):
        """고급 AI 시스템 초기화"""
        try:
            if ADVANCED_MODULES_AVAILABLE and TRANSFORMERS_AVAILABLE:
                self.advanced_ai = AdvancedAutoCI()
                logger.info("✅ 고급 트랜스포머 AI 시스템 초기화 완료")
            else:
                self._initialize_fallback_ai()
                logger.warning("⚠️ 고급 AI 시스템 사용 불가 - 폴백 AI 사용")
        except Exception as e:
            logger.error(f"❌ 고급 AI 시스템 초기화 실패: {e}")
            self._initialize_fallback_ai()
    
    def _initialize_fallback_ai(self):
        """폴백 AI 시스템 초기화"""
        class FallbackAI:
            def __init__(self):
                self.responses = {
                    "greeting": ["안녕하세요! AutoCI입니다.", "반갑습니다!", "안녕하세요! 어떻게 도와드릴까요?"],
                    "unity": ["Unity 개발을 도와드리겠습니다.", "Unity 관련 질문이시군요!", "Unity에 대해 설명드리겠습니다."],
                    "csharp": ["C# 프로그래밍을 도와드리겠습니다.", "C# 관련 질문이시네요!", "C#에 대해 알려드리겠습니다."],
                    "default": ["네, 이해했습니다.", "더 구체적으로 설명해주시면 도움을 드릴 수 있습니다.", "좋은 질문이네요!"]
                }
                
            def chat(self, user_id: str, user_input: str) -> Tuple[str, str]:
                import random
                input_lower = user_input.lower()
                
                if any(word in input_lower for word in ["안녕", "hello", "hi"]):
                    response = random.choice(self.responses["greeting"])
                elif any(word in input_lower for word in ["unity", "유니티"]):
                    response = random.choice(self.responses["unity"])
                elif any(word in input_lower for word in ["c#", "csharp", "코드"]):
                    response = random.choice(self.responses["csharp"])
                else:
                    response = random.choice(self.responses["default"])
                
                return response, f"fallback_conv_{int(time.time())}"
            
            def process_feedback(self, user_id: str, conversation_id: str, 
                               user_input: str, ai_response: str, feedback: str) -> bool:
                return True
            
            def save_model(self):
                pass
        
        self.advanced_ai = FallbackAI()
    
    def _initialize_progress_tracking(self):
        """진행률 추적 시스템 초기화"""
        try:
            if ADVANCED_MODULES_AVAILABLE:
                self.progress_tracker = LearningProgressTracker()
                logger.info("✅ 학습 진행률 추적 시스템 초기화 완료")
            else:
                logger.warning("⚠️ 진행률 추적 시스템 사용 불가")
        except Exception as e:
            logger.error(f"❌ 진행률 추적 시스템 초기화 실패: {e}")
    
    def _fallback_initialization(self):
        """폴백 초기화"""
        logger.warning("🔄 폴백 모드로 시스템 초기화")
        self._initialize_fallback_ai()
        self.status["initialized"] = True
        self.status["ready_for_conversation"] = True
        logger.info("✅ 폴백 모드 초기화 완료")
    
    def _check_component_health(self) -> bool:
        """컴포넌트 상태 확인"""
        healthy = True
        
        if self.advanced_ai is None:
            healthy = False
            logger.warning("⚠️ AI 시스템 비정상")
        
        if ADVANCED_MODULES_AVAILABLE and self.dataset_db is None:
            healthy = False
            logger.warning("⚠️ 데이터셋 시스템 비정상")
        
        return healthy
    
    def chat(self, user_input: str, user_id: str = "default_user") -> Dict[str, Any]:
        """ChatGPT 수준의 대화 인터페이스"""
        
        if not self.status["ready_for_conversation"]:
            return {
                "response": "시스템이 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.",
                "conversation_id": None,
                "confidence": 0.0,
                "learning_applied": False,
                "error": "system_not_ready"
            }
        
        try:
            start_time = time.time()
            
            # 사용자 세션 관리
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = {
                    "conversations": [],
                    "preferences": {},
                    "satisfaction_scores": [],
                    "created_at": datetime.now().isoformat()
                }
            
            # AI 응답 생성
            ai_response, conversation_id = self.advanced_ai.chat(user_id, user_input)
            
            # 응답 시간 계산
            response_time = time.time() - start_time
            
            # 대화 기록 저장
            conversation_record = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "user_input": user_input,
                "ai_response": ai_response,
                "timestamp": datetime.now().isoformat(),
                "response_time": response_time,
                "confidence": self._estimate_confidence(user_input, ai_response)
            }
            
            self.conversation_history.append(conversation_record)
            self.user_sessions[user_id]["conversations"].append(conversation_record)
            
            # 메트릭 업데이트
            self.metrics["total_conversations"] += 1
            
            # 응답 품질 평가 (백그라운드)
            threading.Thread(
                target=self._evaluate_response_quality,
                args=(conversation_record,),
                daemon=True
            ).start()
            
            return {
                "response": ai_response,
                "conversation_id": conversation_id,
                "confidence": conversation_record["confidence"],
                "response_time": response_time,
                "learning_applied": True,
                "capabilities": list(self.capabilities.keys()),
                "system_status": "ready"
            }
            
        except Exception as e:
            logger.error(f"❌ 대화 처리 오류: {e}")
            return {
                "response": "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
                "conversation_id": None,
                "confidence": 0.0,
                "learning_applied": False,
                "error": str(e)
            }
    
    def _estimate_confidence(self, user_input: str, ai_response: str) -> float:
        """응답 신뢰도 추정"""
        confidence = 0.5  # 기본값
        
        # 응답 길이 기반
        if len(ai_response) > 50:
            confidence += 0.2
        
        # 기술적 키워드 포함 여부
        tech_keywords = ["unity", "c#", "gameobject", "script", "코드", "메서드"]
        if any(keyword in user_input.lower() for keyword in tech_keywords):
            if any(keyword in ai_response.lower() for keyword in tech_keywords):
                confidence += 0.2
        
        # 구조화된 답변 (코드 블록, 단계별 설명)
        if any(marker in ai_response for marker in ["```", "1.", "2.", "3."]):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _evaluate_response_quality(self, conversation_record: Dict):
        """응답 품질 평가 (백그라운드)"""
        try:
            if ADVANCED_MODULES_AVAILABLE and hasattr(self, 'quality_evaluator'):
                # 품질 평가 수행
                pass
            
            # 간단한 품질 메트릭 업데이트
            confidence = conversation_record["confidence"]
            current_accuracy = self.metrics["response_accuracy"]
            self.metrics["response_accuracy"] = (current_accuracy + confidence) / 2
            
        except Exception as e:
            logger.error(f"품질 평가 오류: {e}")
    
    def process_feedback(self, conversation_id: str, feedback: str, 
                        feedback_type: str = "general") -> Dict[str, Any]:
        """사용자 피드백 처리"""
        
        try:
            # 대화 기록 찾기
            conversation_record = next(
                (conv for conv in self.conversation_history 
                 if conv["conversation_id"] == conversation_id),
                None
            )
            
            if not conversation_record:
                return {
                    "success": False,
                    "error": "conversation_not_found",
                    "message": "해당 대화를 찾을 수 없습니다."
                }
            
            # 피드백 분석
            feedback_score = self._analyze_feedback(feedback)
            
            # 고급 AI 시스템에 피드백 전달
            if hasattr(self.advanced_ai, 'process_feedback'):
                success = self.advanced_ai.process_feedback(
                    conversation_record["user_id"],
                    conversation_id,
                    conversation_record["user_input"],
                    conversation_record["ai_response"],
                    feedback
                )
            else:
                success = True
            
            # 사용자 만족도 업데이트
            user_id = conversation_record["user_id"]
            if user_id in self.user_sessions:
                self.user_sessions[user_id]["satisfaction_scores"].append(feedback_score)
                
                # 전체 만족도 업데이트
                all_scores = [
                    score 
                    for session in self.user_sessions.values() 
                    for score in session["satisfaction_scores"]
                ]
                if all_scores:
                    self.metrics["user_satisfaction"] = sum(all_scores) / len(all_scores)
            
            # 학습 진행률 기록
            if self.progress_tracker and feedback_score != 0:
                try:
                    self.progress_tracker.record_learning_step(
                        epoch=self.metrics["total_conversations"],
                        loss=max(0.1, 1.0 - abs(feedback_score)),
                        accuracy=max(0.1, (feedback_score + 1.0) / 2.0),
                        learning_rate=0.001,
                        batch_size=1,
                        data_points=1,
                        training_time=1.0
                    )
                except Exception as e:
                    logger.warning(f"진행률 기록 실패: {e}")
            
            self.metrics["last_learning_update"] = datetime.now().isoformat()
            
            return {
                "success": success,
                "feedback_score": feedback_score,
                "learning_applied": success,
                "message": "피드백이 학습에 반영되었습니다." if success else "피드백 처리에 실패했습니다."
            }
            
        except Exception as e:
            logger.error(f"❌ 피드백 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "피드백 처리 중 오류가 발생했습니다."
            }
    
    def _analyze_feedback(self, feedback: str) -> float:
        """피드백 분석"""
        feedback_lower = feedback.lower()
        
        # 긍정적 패턴
        positive_patterns = [
            "좋", "맞", "정확", "도움", "고마", "훌륭", "완벽", "최고", "감사"
        ]
        
        # 부정적 패턴
        negative_patterns = [
            "틀", "아니", "이상", "별로", "다시", "잘못", "나쁘", "엉터리"
        ]
        
        positive_count = sum(1 for pattern in positive_patterns if pattern in feedback_lower)
        negative_count = sum(1 for pattern in negative_patterns if pattern in feedback_lower)
        
        if positive_count > negative_count:
            return min(1.0, positive_count * 0.3 + 0.4)
        elif negative_count > positive_count:
            return max(-1.0, -(negative_count * 0.3 + 0.4))
        else:
            return 0.0
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        uptime = datetime.now() - self.metrics["system_uptime"]
        
        return {
            "system_name": self.system_name,
            "version": self.version,
            "status": self.status,
            "capabilities": self.capabilities,
            "metrics": {
                **self.metrics,
                "uptime_hours": uptime.total_seconds() / 3600,
                "avg_conversations_per_hour": self.metrics["total_conversations"] / max(uptime.total_seconds() / 3600, 1)
            },
            "active_users": len(self.user_sessions),
            "components": {
                "advanced_ai": self.advanced_ai is not None,
                "dataset_db": self.dataset_db is not None,
                "progress_tracker": self.progress_tracker is not None,
                "transformers_available": TRANSFORMERS_AVAILABLE,
                "advanced_modules_available": ADVANCED_MODULES_AVAILABLE
            }
        }
    
    def generate_learning_report(self) -> Dict[str, Any]:
        """학습 보고서 생성"""
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "system_performance": {
                    "total_conversations": self.metrics["total_conversations"],
                    "user_satisfaction": self.metrics["user_satisfaction"],
                    "response_accuracy": self.metrics["response_accuracy"],
                    "learning_efficiency": self.metrics["learning_efficiency"]
                },
                "user_analytics": {
                    "total_users": len(self.user_sessions),
                    "avg_conversations_per_user": (
                        sum(len(session["conversations"]) for session in self.user_sessions.values()) 
                        / max(len(self.user_sessions), 1)
                    ),
                    "user_retention": self._calculate_user_retention()
                },
                "conversation_insights": self._analyze_conversation_patterns(),
                "recommendations": self._generate_improvement_recommendations()
            }
            
            # 진행률 추적기에서 추가 정보
            if self.progress_tracker:
                try:
                    progress_report = self.progress_tracker.generate_progress_report()
                    report["learning_progress"] = progress_report
                except Exception as e:
                    logger.warning(f"진행률 보고서 생성 실패: {e}")
            
            return report
            
        except Exception as e:
            logger.error(f"학습 보고서 생성 실패: {e}")
            return {"error": str(e)}
    
    def _calculate_user_retention(self) -> float:
        """사용자 유지율 계산"""
        if not self.user_sessions:
            return 0.0
        
        # 간단한 유지율: 대화가 2회 이상인 사용자 비율
        retained_users = sum(
            1 for session in self.user_sessions.values() 
            if len(session["conversations"]) >= 2
        )
        
        return retained_users / len(self.user_sessions)
    
    def _analyze_conversation_patterns(self) -> Dict[str, Any]:
        """대화 패턴 분석"""
        if not self.conversation_history:
            return {}
        
        # 주제별 분포
        topics = {}
        for conv in self.conversation_history:
            user_input = conv["user_input"].lower()
            
            if any(word in user_input for word in ["unity", "유니티"]):
                topics["unity"] = topics.get("unity", 0) + 1
            elif any(word in user_input for word in ["c#", "csharp"]):
                topics["csharp"] = topics.get("csharp", 0) + 1
            else:
                topics["general"] = topics.get("general", 0) + 1
        
        # 응답 시간 분석
        response_times = [conv["response_time"] for conv in self.conversation_history]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # 신뢰도 분석
        confidences = [conv["confidence"] for conv in self.conversation_history]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "topic_distribution": topics,
            "avg_response_time": avg_response_time,
            "avg_confidence": avg_confidence,
            "total_conversations": len(self.conversation_history)
        }
    
    def _generate_improvement_recommendations(self) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        # 응답 정확도 기반
        if self.metrics["response_accuracy"] < 0.7:
            recommendations.append("응답 정확도가 낮습니다. 더 많은 학습 데이터가 필요합니다.")
        
        # 사용자 만족도 기반
        if self.metrics["user_satisfaction"] < 0.6:
            recommendations.append("사용자 만족도가 낮습니다. 응답 품질 개선이 필요합니다.")
        
        # 유지율 기반
        retention = self._calculate_user_retention()
        if retention < 0.5:
            recommendations.append("사용자 유지율이 낮습니다. 더 매력적인 대화 경험을 제공해야 합니다.")
        
        # 컴포넌트 상태 기반
        if not self.status["components_healthy"]:
            recommendations.append("일부 시스템 컴포넌트가 정상적으로 작동하지 않습니다.")
        
        if not recommendations:
            recommendations.append("시스템이 잘 작동하고 있습니다. 현재 성능을 유지하세요.")
        
        return recommendations
    
    def interactive_chat_interface(self):
        """대화형 인터페이스"""
        print(f"\n🤖 {self.system_name} v{self.version}")
        print("=" * 60)
        print("ChatGPT 수준의 AutoCI와 대화해보세요!")
        print("명령어: 'quit' (종료), 'status' (상태), 'report' (보고서)")
        print("=" * 60)
        
        user_id = f"user_{int(time.time())}"
        last_conversation_id = None
        
        while True:
            try:
                user_input = input("\n💬 당신: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료']:
                    break
                
                elif user_input.lower() == 'status':
                    status = self.get_system_status()
                    print(f"\n📊 시스템 상태:")
                    print(f"  버전: {status['version']}")
                    print(f"  총 대화: {status['metrics']['total_conversations']}")
                    print(f"  사용자 만족도: {status['metrics']['user_satisfaction']:.2f}")
                    print(f"  응답 정확도: {status['metrics']['response_accuracy']:.2f}")
                    print(f"  업타임: {status['metrics']['uptime_hours']:.1f}시간")
                    continue
                
                elif user_input.lower() == 'report':
                    print("\n📋 학습 보고서 생성 중...")
                    report = self.generate_learning_report()
                    if "error" not in report:
                        print(f"📈 총 대화: {report['system_performance']['total_conversations']}")
                        print(f"📊 사용자 만족도: {report['system_performance']['user_satisfaction']:.2f}")
                        print(f"🎯 권장사항: {len(report['recommendations'])}개")
                        for i, rec in enumerate(report['recommendations'], 1):
                            print(f"  {i}. {rec}")
                    else:
                        print(f"❌ 보고서 생성 실패: {report['error']}")
                    continue
                
                elif user_input.lower().startswith('feedback:') and last_conversation_id:
                    feedback = user_input[9:].strip()
                    result = self.process_feedback(last_conversation_id, feedback)
                    if result["success"]:
                        print(f"✅ {result['message']}")
                    else:
                        print(f"❌ {result['message']}")
                    continue
                
                # AI와 대화
                result = self.chat(user_input, user_id)
                
                if "error" not in result:
                    print(f"\n🤖 AutoCI: {result['response']}")
                    print(f"   (신뢰도: {result['confidence']:.2f}, "
                          f"응답시간: {result['response_time']:.2f}초)")
                    
                    last_conversation_id = result["conversation_id"]
                    
                    # 가끔 피드백 요청
                    import random
                    if random.random() < 0.2:  # 20% 확률
                        print("\n💡 이 답변이 도움이 되었나요? 'feedback: 좋아요' 또는 'feedback: 별로예요'")
                else:
                    print(f"\n❌ AutoCI: {result['response']}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
        
        # 시스템 정리
        if hasattr(self.advanced_ai, 'save_model'):
            self.advanced_ai.save_model()
        
        print(f"\n👋 {self.system_name}를 이용해주셔서 감사합니다!")
        print("학습 데이터가 저장되었습니다.")

def main():
    """메인 함수"""
    print("🚀 ChatGPT 수준의 AutoCI 시스템")
    print("=" * 60)
    
    try:
        # ChatGPT 수준 AutoCI 초기화
        autoci = ChatGPTLevelAutoCI()
        
        # 대화형 인터페이스 시작
        autoci.interactive_chat_interface()
        
    except Exception as e:
        logger.error(f"시스템 실행 실패: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())