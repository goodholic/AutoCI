#!/usr/bin/env python3
"""
AutoCI 실제 학습 AI 개념 - 진짜 학습하는 시스템
(현재는 개념 설명용)
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import hashlib

class LearningKoreanAI:
    """실제로 학습하는 한국어 AI (개념)"""
    
    def __init__(self):
        # 학습 데이터 저장소
        self.conversation_memory = []
        self.pattern_weights = {}
        self.response_feedback = {}
        self.learning_enabled = True
        
        # 학습 통계
        self.learning_stats = {
            "total_conversations": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "pattern_updates": 0,
            "last_learning": None
        }
        
        print("🧠 학습 AI 초기화됨 - 실제 학습 모드")
    
    def process_conversation(self, user_input: str, ai_response: str, user_feedback: Optional[str] = None):
        """대화를 처리하고 학습"""
        
        # 1. 대화 저장
        conversation_id = self._generate_conversation_id(user_input)
        conversation_data = {
            "id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response,
            "user_feedback": user_feedback,
            "learned": False
        }
        
        self.conversation_memory.append(conversation_data)
        self.learning_stats["total_conversations"] += 1
        
        # 2. 사용자 피드백이 있으면 학습
        if user_feedback:
            self._learn_from_feedback(conversation_data)
        
        # 3. 패턴 분석 및 가중치 업데이트
        self._update_pattern_weights(user_input, ai_response)
        
        # 4. 자동 학습 (백그라운드)
        if self.learning_enabled:
            self._background_learning()
        
        print(f"🧠 학습 완료: 총 {len(self.conversation_memory)}개 대화 학습됨")
    
    def _learn_from_feedback(self, conversation: Dict):
        """사용자 피드백으로부터 학습"""
        feedback = conversation["user_feedback"].lower()
        
        # 긍정적 피드백
        positive_keywords = ["좋아", "맞아", "정확해", "도움됐어", "고마워", "잘했어"]
        # 부정적 피드백  
        negative_keywords = ["틀려", "아니야", "이상해", "별로", "다시", "잘못"]
        
        if any(keyword in feedback for keyword in positive_keywords):
            self._reinforce_response(conversation, reward=1.0)
            self.learning_stats["positive_feedback"] += 1
            print("✅ 긍정적 피드백 학습")
            
        elif any(keyword in feedback for keyword in negative_keywords):
            self._penalize_response(conversation, penalty=-0.5)
            self.learning_stats["negative_feedback"] += 1
            print("❌ 부정적 피드백 학습")
        
        conversation["learned"] = True
        self.learning_stats["last_learning"] = datetime.now().isoformat()
    
    def _reinforce_response(self, conversation: Dict, reward: float):
        """좋은 응답 패턴 강화"""
        user_input = conversation["user_input"]
        ai_response = conversation["ai_response"]
        
        # 성공한 패턴의 가중치 증가
        pattern_key = self._extract_pattern(user_input)
        if pattern_key in self.pattern_weights:
            self.pattern_weights[pattern_key] += reward
        else:
            self.pattern_weights[pattern_key] = reward
        
        # 응답 템플릿 점수 향상
        response_key = self._hash_response(ai_response)
        if response_key in self.response_feedback:
            self.response_feedback[response_key] += reward
        else:
            self.response_feedback[response_key] = reward
    
    def _penalize_response(self, conversation: Dict, penalty: float):
        """나쁜 응답 패턴 약화"""
        user_input = conversation["user_input"]
        ai_response = conversation["ai_response"]
        
        # 실패한 패턴의 가중치 감소
        pattern_key = self._extract_pattern(user_input)
        if pattern_key in self.pattern_weights:
            self.pattern_weights[pattern_key] += penalty  # penalty는 음수
        
        # 응답 템플릿 점수 하락
        response_key = self._hash_response(ai_response)
        if response_key in self.response_feedback:
            self.response_feedback[response_key] += penalty
    
    def _update_pattern_weights(self, user_input: str, ai_response: str):
        """패턴 가중치 업데이트"""
        # 사용 빈도에 따른 패턴 강화
        pattern = self._extract_pattern(user_input)
        
        if pattern in self.pattern_weights:
            self.pattern_weights[pattern] += 0.1  # 사용할 때마다 약간 증가
        else:
            self.pattern_weights[pattern] = 0.1
        
        self.learning_stats["pattern_updates"] += 1
    
    def _background_learning(self):
        """백그라운드 학습 (시뮬레이션)"""
        # 실제로는 여기서 신경망 훈련이나 패턴 분석이 일어남
        if len(self.conversation_memory) % 10 == 0:  # 10개 대화마다
            print("🔄 백그라운드 학습 중...")
            time.sleep(0.1)  # 학습 시뮬레이션
            print("✅ 패턴 분석 완료")
    
    def _extract_pattern(self, text: str) -> str:
        """텍스트에서 패턴 추출"""
        # 간단한 패턴 추출 (실제로는 더 복잡)
        import re
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(text.replace(' ', ''))
        
        if korean_chars / total_chars > 0.8:
            return "korean_high"
        elif korean_chars / total_chars > 0.3:
            return "korean_mixed"
        else:
            return "english"
    
    def _hash_response(self, response: str) -> str:
        """응답을 해시화"""
        return hashlib.md5(response.encode()).hexdigest()[:8]
    
    def _generate_conversation_id(self, user_input: str) -> str:
        """대화 ID 생성"""
        timestamp = str(int(time.time()))
        input_hash = hashlib.md5(user_input.encode()).hexdigest()[:6]
        return f"conv_{timestamp}_{input_hash}"
    
    def get_learning_status(self) -> Dict:
        """학습 상태 반환"""
        return {
            "learning_enabled": self.learning_enabled,
            "stats": self.learning_stats,
            "memory_size": len(self.conversation_memory),
            "pattern_count": len(self.pattern_weights),
            "top_patterns": sorted(self.pattern_weights.items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
        }
    
    def save_learning_data(self, filename: str = "autoci_learning_data.json"):
        """학습 데이터 저장"""
        learning_data = {
            "conversations": self.conversation_memory,
            "pattern_weights": self.pattern_weights,
            "response_feedback": self.response_feedback,
            "stats": self.learning_stats,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(learning_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 학습 데이터 저장됨: {filename}")
    
    def load_learning_data(self, filename: str = "autoci_learning_data.json"):
        """학습 데이터 로드"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                learning_data = json.load(f)
            
            self.conversation_memory = learning_data.get("conversations", [])
            self.pattern_weights = learning_data.get("pattern_weights", {})
            self.response_feedback = learning_data.get("response_feedback", {})
            self.learning_stats = learning_data.get("stats", self.learning_stats)
            
            print(f"📂 학습 데이터 로드됨: {len(self.conversation_memory)}개 대화")
            
        except FileNotFoundError:
            print("📂 새로운 학습 데이터 시작")

def demonstrate_learning():
    """학습 AI 데모"""
    print("🧠 실제 학습하는 AI 데모")
    print("=" * 50)
    
    ai = LearningKoreanAI()
    
    # 샘플 대화들
    conversations = [
        ("안녕하세요", "안녕하세요! 반갑습니다!", "좋아"),
        ("Unity 도와줘", "Unity 개발을 도와드리겠습니다!", "도움됐어"),
        ("너 이름이 뭐야", "제 이름은 AutoCI입니다.", "아니야 틀려"),
        ("고마워", "천만에요! 도움이 되어서 기뻐요!", "맞아")
    ]
    
    for user_input, ai_response, feedback in conversations:
        print(f"\n사용자: {user_input}")
        print(f"AI: {ai_response}")
        print(f"피드백: {feedback}")
        
        ai.process_conversation(user_input, ai_response, feedback)
    
    # 학습 상태 출력
    print(f"\n📊 학습 상태:")
    status = ai.get_learning_status()
    for key, value in status["stats"].items():
        print(f"  {key}: {value}")
    
    print(f"\n🔝 상위 패턴:")
    for pattern, weight in status["top_patterns"]:
        print(f"  {pattern}: {weight:.2f}")
    
    # 학습 데이터 저장
    ai.save_learning_data()

if __name__ == "__main__":
    demonstrate_learning() 