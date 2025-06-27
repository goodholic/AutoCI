#!/usr/bin/env python3
"""
간단한 신경망 학습 시스템 테스트 (PyTorch 없이)
"""

import sys
import os
import time
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class SimpleConversationData:
    """간단한 대화 데이터 구조"""
    user_input: str
    ai_response: str
    feedback_score: float  # -1.0 to 1.0
    timestamp: str

class SimpleNeuralNetwork:
    """간단한 신경망 (NumPy 기반)"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 5, output_size: int = 1):
        # 가중치 초기화 (Xavier 초기화)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.learning_rate = 0.01
    
    def sigmoid(self, x):
        """시그모이드 활성화 함수"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        """시그모이드 미분"""
        return x * (1 - x)
    
    def forward(self, X):
        """순전파"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        """역전파"""
        m = X.shape[0]
        
        # 출력층 오차
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # 은닉층 오차
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # 가중치 업데이트
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=100):
        """학습"""
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            self.backward(X, y, output)
        return losses
    
    def predict(self, X):
        """예측"""
        return self.forward(X)

class SimpleLearningAutoCI:
    """간단한 학습 AutoCI 시스템 (PyTorch 없이)"""
    
    def __init__(self):
        self.neural_network = SimpleNeuralNetwork()
        self.conversations: List[SimpleConversationData] = []
        self.learning_enabled = True
        
        # 통계
        self.stats = {
            "total_conversations": 0,
            "total_training_epochs": 0,
            "average_quality_score": 0.0,
            "last_learning_time": None,
            "model_accuracy": 0.0
        }
        
        # 간단한 단어 임베딩 (해시 기반)
        self.vocab = {}
        self.vocab_size = 0
        
        print("🧠 간단한 학습 AI 시스템 초기화 완료")
    
    def text_to_vector(self, text: str, vector_size: int = 10) -> np.ndarray:
        """텍스트를 벡터로 변환 (간단한 해시 기반)"""
        words = text.lower().split()
        vector = np.zeros(vector_size)
        
        for i, word in enumerate(words[:vector_size]):
            # 단어의 해시를 벡터 인덱스로 사용
            word_hash = hash(word) % vector_size
            vector[word_hash] = 1.0
            
            # 단어 길이와 위치 정보 추가
            vector[word_hash] += len(word) * 0.1
            vector[word_hash] += (i + 1) * 0.01
        
        # 정규화
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def estimate_feedback_score(self, user_input: str, ai_response: str) -> float:
        """피드백 점수 자동 추정"""
        # 간단한 규칙 기반 추정
        score = 0.0
        
        # 긍정적 키워드
        positive_keywords = ["좋아", "맞아", "정확", "도움", "고마워", "잘했어", "훌륭"]
        negative_keywords = ["틀려", "아니야", "이상해", "별로", "다시", "잘못", "나빠"]
        
        user_lower = user_input.lower()
        response_lower = ai_response.lower()
        
        # 사용자 입력 분석
        for keyword in positive_keywords:
            if keyword in user_lower:
                score += 0.2
        
        for keyword in negative_keywords:
            if keyword in user_lower:
                score -= 0.2
        
        # 응답 길이 분석 (적절한 길이가 좋음)
        response_length = len(ai_response)
        if 10 <= response_length <= 100:
            score += 0.1
        elif response_length < 5 or response_length > 200:
            score -= 0.1
        
        # 유니티/C# 관련 키워드
        unity_keywords = ["unity", "유니티", "c#", "script", "스크립트", "게임", "오브젝트"]
        if any(keyword in user_lower for keyword in unity_keywords):
            if any(keyword in response_lower for keyword in unity_keywords):
                score += 0.1
        
        return np.clip(score, -1.0, 1.0)
    
    def process_conversation(self, user_input: str, ai_response: str, 
                           feedback_score: Optional[float] = None) -> Dict:
        """대화 처리 및 학습"""
        
        # 피드백 점수 계산
        if feedback_score is None:
            feedback_score = self.estimate_feedback_score(user_input, ai_response)
        
        # 대화 데이터 생성
        conversation = SimpleConversationData(
            user_input=user_input,
            ai_response=ai_response,
            feedback_score=feedback_score,
            timestamp=datetime.now().isoformat()
        )
        
        self.conversations.append(conversation)
        self.stats["total_conversations"] += 1
        
        # 학습 트리거 (강한 피드백이 있을 때)
        if self.learning_enabled and abs(feedback_score) > 0.3:
            self.trigger_learning()
        
        return {
            "conversation_id": len(self.conversations),
            "feedback_score": feedback_score,
            "learning_triggered": abs(feedback_score) > 0.3
        }
    
    def trigger_learning(self):
        """학습 트리거"""
        if len(self.conversations) < 5:
            return
        
        print("🔄 신경망 학습 시작...")
        
        # 최근 대화들로 학습 데이터 준비
        recent_conversations = self.conversations[-10:]
        
        X = []
        y = []
        
        for conv in recent_conversations:
            input_vector = self.text_to_vector(conv.user_input)
            X.append(input_vector)
            
            # 피드백 점수를 0-1 범위로 변환
            normalized_score = (conv.feedback_score + 1.0) / 2.0
            y.append([normalized_score])
        
        if len(X) > 0:
            X = np.array(X)
            y = np.array(y)
            
            # 신경망 학습
            losses = self.neural_network.train(X, y, epochs=50)
            
            self.stats["total_training_epochs"] += 50
            self.stats["last_learning_time"] = datetime.now().isoformat()
            
            final_loss = losses[-1] if losses else 0.0
            print(f"✅ 학습 완료 - 최종 손실: {final_loss:.4f}")
    
    def generate_response(self, user_input: str) -> Tuple[str, float]:
        """응답 생성"""
        
        # 신경망으로 품질 점수 예측
        try:
            input_vector = self.text_to_vector(user_input)
            quality_score = self.neural_network.predict(input_vector.reshape(1, -1))[0][0]
            
            # 0-1 범위를 -1~1 범위로 변환
            confidence = (quality_score - 0.5) * 2.0
            
        except Exception as e:
            quality_score = 0.5
            confidence = 0.0
        
        # 간단한 규칙 기반 응답 생성
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["안녕", "hello", "hi"]):
            response = "안녕하세요! AutoCI입니다. 어떻게 도와드릴까요?"
        elif any(word in user_lower for word in ["unity", "유니티"]):
            response = "Unity 개발을 도와드리겠습니다! 어떤 부분이 궁금하신가요?"
        elif any(word in user_lower for word in ["코드", "code", "스크립트"]):
            response = "코드를 분석해드리겠습니다. 코드를 보여주세요."
        elif any(word in user_lower for word in ["오류", "에러", "error"]):
            response = "오류를 해결해드리겠습니다. 어떤 오류인지 알려주세요."
        elif any(word in user_lower for word in ["고마워", "감사"]):
            response = "천만에요! 더 도움이 필요하시면 언제든 말씀해주세요."
        else:
            response = "네, 이해했습니다. 더 자세히 설명해주시면 도움을 드릴 수 있습니다."
        
        return response, abs(confidence)
    
    def get_learning_status(self) -> Dict:
        """학습 상태 반환"""
        return {
            "stats": self.stats,
            "total_conversations": len(self.conversations),
            "neural_network_weights": {
                "W1_shape": self.neural_network.W1.shape,
                "W2_shape": self.neural_network.W2.shape
            }
        }

def test_simple_neural_learning():
    """간단한 신경망 학습 테스트"""
    print("🚀 간단한 신경망 학습 시스템 테스트")
    print("=" * 60)
    
    # AI 시스템 초기화
    ai = SimpleLearningAutoCI()
    print("✅ AI 시스템 초기화 성공")
    
    # 테스트 대화들
    test_conversations = [
        ("안녕하세요", "안녕하세요! AutoCI입니다.", 1.0),
        ("Unity 도움이 필요해요", "Unity 개발을 도와드리겠습니다!", 0.8),
        ("코드 분석해주세요", "코드를 분석해드리겠습니다.", 0.6),
        ("이상한 응답이네요", "죄송합니다. 다시 시도해보겠습니다.", -0.5),
        ("고마워요", "천만에요! 도움이 되어서 기뻐요!", 0.9),
        ("Unity에서 오브젝트 생성하는 방법", "Instantiate 함수를 사용하시면 됩니다.", 0.7),
        ("스크립트 오류가 있어요", "어떤 오류인지 알려주시면 도와드리겠습니다.", 0.5),
        ("잘못된 답변", "죄송합니다. 개선하겠습니다.", -0.3)
    ]
    
    print(f"\n🗣️ {len(test_conversations)}개 테스트 대화 처리 중...")
    
    # 대화 처리 및 학습
    for i, (user_input, ai_response, feedback) in enumerate(test_conversations, 1):
        print(f"\n테스트 {i}/{len(test_conversations)}:")
        print(f"사용자: {user_input}")
        print(f"AI: {ai_response}")
        print(f"피드백: {feedback}")
        
        result = ai.process_conversation(user_input, ai_response, feedback)
        
        print(f"✅ 대화 ID: {result['conversation_id']}")
        print(f"✅ 학습 트리거: {result['learning_triggered']}")
        
        # 잠시 대기
        time.sleep(0.5)
    
    print(f"\n📊 최종 학습 상태:")
    status = ai.get_learning_status()
    for key, value in status["stats"].items():
        print(f"  {key}: {value}")
    
    print(f"\n🤖 응답 생성 테스트:")
    test_inputs = [
        "안녕하세요",
        "Unity 스크립트 도움",
        "코드 오류 해결",
        "감사합니다"
    ]
    
    for test_input in test_inputs:
        response, confidence = ai.generate_response(test_input)
        print(f"입력: {test_input}")
        print(f"응답: {response}")
        print(f"신뢰도: {confidence:.2f}\n")
    
    print("🎉 모든 테스트 완료!")
    print("✅ 간단한 신경망 기반 학습 시스템이 정상 작동합니다.")
    
    return ai

if __name__ == "__main__":
    test_simple_neural_learning()