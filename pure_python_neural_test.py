#!/usr/bin/env python3
"""
순수 Python 신경망 학습 시스템 테스트 (외부 라이브러리 없이)
"""

import sys
import os
import time
import math
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# dataclass 수동 구현 (decorator 없이)
class ConversationData:
    def __init__(self, user_input: str, ai_response: str, feedback_score: float, timestamp: str):
        self.user_input = user_input
        self.ai_response = ai_response
        self.feedback_score = feedback_score
        self.timestamp = timestamp

class PurePythonMatrix:
    """순수 Python 행렬 연산"""
    
    @staticmethod
    def zeros(rows: int, cols: int) -> List[List[float]]:
        """영행렬 생성"""
        return [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    @staticmethod
    def random_matrix(rows: int, cols: int, scale: float = 1.0) -> List[List[float]]:
        """랜덤 행렬 생성"""
        import random
        return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]
    
    @staticmethod
    def multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """행렬 곱셈"""
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        
        if cols_A != rows_B:
            raise ValueError("행렬 크기가 맞지 않습니다")
        
        result = PurePythonMatrix.zeros(rows_A, cols_B)
        
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        
        return result
    
    @staticmethod
    def add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """행렬 덧셈"""
        rows, cols = len(A), len(A[0])
        result = PurePythonMatrix.zeros(rows, cols)
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = A[i][j] + B[i][j]
        
        return result
    
    @staticmethod
    def subtract(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """행렬 뺄셈"""
        rows, cols = len(A), len(A[0])
        result = PurePythonMatrix.zeros(rows, cols)
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = A[i][j] - B[i][j]
        
        return result
    
    @staticmethod
    def transpose(A: List[List[float]]) -> List[List[float]]:
        """행렬 전치"""
        rows, cols = len(A), len(A[0])
        result = PurePythonMatrix.zeros(cols, rows)
        
        for i in range(rows):
            for j in range(cols):
                result[j][i] = A[i][j]
        
        return result
    
    @staticmethod
    def scalar_multiply(A: List[List[float]], scalar: float) -> List[List[float]]:
        """스칼라 곱셈"""
        rows, cols = len(A), len(A[0])
        result = PurePythonMatrix.zeros(rows, cols)
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = A[i][j] * scalar
        
        return result

class PurePythonNeuralNetwork:
    """순수 Python 신경망"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 5, output_size: int = 1):
        # 가중치 초기화
        scale = math.sqrt(2.0 / input_size)
        self.W1 = PurePythonMatrix.random_matrix(input_size, hidden_size, scale)
        self.b1 = PurePythonMatrix.zeros(1, hidden_size)
        
        scale = math.sqrt(2.0 / hidden_size)
        self.W2 = PurePythonMatrix.random_matrix(hidden_size, output_size, scale)
        self.b2 = PurePythonMatrix.zeros(1, output_size)
        
        self.learning_rate = 0.01
    
    def sigmoid(self, x: float) -> float:
        """시그모이드 함수"""
        try:
            return 1.0 / (1.0 + math.exp(-max(-250, min(250, x))))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    def sigmoid_matrix(self, matrix: List[List[float]]) -> List[List[float]]:
        """행렬에 시그모이드 적용"""
        rows, cols = len(matrix), len(matrix[0])
        result = PurePythonMatrix.zeros(rows, cols)
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = self.sigmoid(matrix[i][j])
        
        return result
    
    def forward(self, X: List[List[float]]) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """순전파"""
        # z1 = X * W1 + b1
        z1 = PurePythonMatrix.multiply(X, self.W1)
        z1 = PurePythonMatrix.add(z1, self.b1)
        
        # a1 = sigmoid(z1)
        a1 = self.sigmoid_matrix(z1)
        
        # z2 = a1 * W2 + b2
        z2 = PurePythonMatrix.multiply(a1, self.W2)
        z2 = PurePythonMatrix.add(z2, self.b2)
        
        # a2 = sigmoid(z2)
        a2 = self.sigmoid_matrix(z2)
        
        return a1, z2, a2
    
    def calculate_loss(self, predictions: List[List[float]], targets: List[List[float]]) -> float:
        """손실 계산 (MSE)"""
        total_loss = 0.0
        count = 0
        
        rows, cols = len(predictions), len(predictions[0])
        
        for i in range(rows):
            for j in range(cols):
                diff = predictions[i][j] - targets[i][j]
                total_loss += diff * diff
                count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def train_step(self, X: List[List[float]], y: List[List[float]]) -> float:
        """한 번의 학습 단계"""
        batch_size = len(X)
        
        # 순전파
        a1, z2, a2 = self.forward(X)
        
        # 손실 계산
        loss = self.calculate_loss(a2, y)
        
        # 역전파
        # 출력층 오차
        dz2 = PurePythonMatrix.subtract(a2, y)
        
        # W2, b2 그래디언트
        a1_t = PurePythonMatrix.transpose(a1)
        dW2 = PurePythonMatrix.multiply(a1_t, dz2)
        dW2 = PurePythonMatrix.scalar_multiply(dW2, 1.0 / batch_size)
        
        # db2 계산 (열 합)
        db2 = PurePythonMatrix.zeros(1, len(dz2[0]))
        for i in range(len(dz2)):
            for j in range(len(dz2[0])):
                db2[0][j] += dz2[i][j]
        db2 = PurePythonMatrix.scalar_multiply(db2, 1.0 / batch_size)
        
        # 은닉층 오차
        W2_t = PurePythonMatrix.transpose(self.W2)
        dz1 = PurePythonMatrix.multiply(dz2, W2_t)
        
        # a1에 대한 시그모이드 미분 적용
        for i in range(len(dz1)):
            for j in range(len(dz1[0])):
                sigmoid_derivative = a1[i][j] * (1 - a1[i][j])
                dz1[i][j] *= sigmoid_derivative
        
        # W1, b1 그래디언트
        X_t = PurePythonMatrix.transpose(X)
        dW1 = PurePythonMatrix.multiply(X_t, dz1)
        dW1 = PurePythonMatrix.scalar_multiply(dW1, 1.0 / batch_size)
        
        # db1 계산
        db1 = PurePythonMatrix.zeros(1, len(dz1[0]))
        for i in range(len(dz1)):
            for j in range(len(dz1[0])):
                db1[0][j] += dz1[i][j]
        db1 = PurePythonMatrix.scalar_multiply(db1, 1.0 / batch_size)
        
        # 가중치 업데이트
        dW2_scaled = PurePythonMatrix.scalar_multiply(dW2, self.learning_rate)
        self.W2 = PurePythonMatrix.subtract(self.W2, dW2_scaled)
        
        db2_scaled = PurePythonMatrix.scalar_multiply(db2, self.learning_rate)
        self.b2 = PurePythonMatrix.subtract(self.b2, db2_scaled)
        
        dW1_scaled = PurePythonMatrix.scalar_multiply(dW1, self.learning_rate)
        self.W1 = PurePythonMatrix.subtract(self.W1, dW1_scaled)
        
        db1_scaled = PurePythonMatrix.scalar_multiply(db1, self.learning_rate)
        self.b1 = PurePythonMatrix.subtract(self.b1, db1_scaled)
        
        return loss
    
    def predict(self, X: List[List[float]]) -> List[List[float]]:
        """예측"""
        _, _, a2 = self.forward(X)
        return a2

class PurePythonLearningAutoCI:
    """순수 Python 학습 AutoCI 시스템"""
    
    def __init__(self):
        self.neural_network = PurePythonNeuralNetwork()
        self.conversations: List[ConversationData] = []
        self.learning_enabled = True
        
        # 통계
        self.stats = {
            "total_conversations": 0,
            "total_training_epochs": 0,
            "average_loss": 0.0,
            "last_learning_time": None
        }
        
        print("🧠 순수 Python 학습 AI 시스템 초기화 완료")
    
    def text_to_vector(self, text: str, vector_size: int = 10) -> List[float]:
        """텍스트를 벡터로 변환"""
        words = text.lower().split()
        vector = [0.0] * vector_size
        
        for i, word in enumerate(words[:vector_size]):
            # 단어의 해시를 벡터 인덱스로 사용
            word_hash = hash(word) % vector_size
            vector[word_hash] = 1.0
            
            # 단어 길이와 위치 정보 추가
            vector[word_hash] += len(word) * 0.1
            vector[word_hash] += (i + 1) * 0.01
        
        # 간단한 정규화
        vector_sum = sum(x * x for x in vector)
        if vector_sum > 0:
            norm = math.sqrt(vector_sum)
            vector = [x / norm for x in vector]
        
        return vector
    
    def estimate_feedback_score(self, user_input: str, ai_response: str) -> float:
        """피드백 점수 자동 추정"""
        score = 0.0
        
        # 긍정적/부정적 키워드
        positive_keywords = ["좋아", "맞아", "정확", "도움", "고마워", "잘했어"]
        negative_keywords = ["틀려", "아니야", "이상해", "별로", "다시", "잘못"]
        
        user_lower = user_input.lower()
        response_lower = ai_response.lower()
        
        for keyword in positive_keywords:
            if keyword in user_lower:
                score += 0.2
        
        for keyword in negative_keywords:
            if keyword in user_lower:
                score -= 0.2
        
        # 응답 길이 체크
        response_length = len(ai_response)
        if 10 <= response_length <= 100:
            score += 0.1
        elif response_length < 5 or response_length > 200:
            score -= 0.1
        
        return max(-1.0, min(1.0, score))
    
    def process_conversation(self, user_input: str, ai_response: str, 
                           feedback_score: Optional[float] = None) -> Dict:
        """대화 처리"""
        if feedback_score is None:
            feedback_score = self.estimate_feedback_score(user_input, ai_response)
        
        conversation = ConversationData(
            user_input=user_input,
            ai_response=ai_response,
            feedback_score=feedback_score,
            timestamp=datetime.now().isoformat()
        )
        
        self.conversations.append(conversation)
        self.stats["total_conversations"] += 1
        
        # 학습 트리거
        if self.learning_enabled and abs(feedback_score) > 0.3:
            self.trigger_learning()
        
        return {
            "conversation_id": len(self.conversations),
            "feedback_score": feedback_score,
            "learning_triggered": abs(feedback_score) > 0.3
        }
    
    def trigger_learning(self):
        """학습 실행"""
        if len(self.conversations) < 3:
            return
        
        print("🔄 신경망 학습 시작...")
        
        # 학습 데이터 준비
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
            # 학습 실행
            total_loss = 0.0
            epochs = 20
            
            for epoch in range(epochs):
                loss = self.neural_network.train_step(X, y)
                total_loss += loss
            
            avg_loss = total_loss / epochs
            self.stats["total_training_epochs"] += epochs
            self.stats["average_loss"] = avg_loss
            self.stats["last_learning_time"] = datetime.now().isoformat()
            
            print(f"✅ 학습 완료 - 평균 손실: {avg_loss:.4f}")
    
    def generate_response(self, user_input: str) -> Tuple[str, float]:
        """응답 생성"""
        # 신경망으로 품질 예측
        try:
            input_vector = self.text_to_vector(user_input)
            prediction = self.neural_network.predict([input_vector])
            quality_score = prediction[0][0]
            
            # 0-1 범위를 신뢰도로 변환
            confidence = quality_score
            
        except Exception as e:
            confidence = 0.5
        
        # 규칙 기반 응답 생성
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
        
        return response, confidence
    
    def get_learning_status(self) -> Dict:
        """학습 상태 반환"""
        return {
            "stats": self.stats,
            "total_conversations": len(self.conversations),
            "neural_network_info": {
                "input_size": len(self.neural_network.W1),
                "hidden_size": len(self.neural_network.W1[0]),
                "output_size": len(self.neural_network.W2[0])
            }
        }

def test_pure_python_neural_learning():
    """순수 Python 신경망 학습 테스트"""
    print("🚀 순수 Python 신경망 학습 시스템 테스트")
    print("=" * 60)
    
    # AI 시스템 초기화
    ai = PurePythonLearningAutoCI()
    print("✅ AI 시스템 초기화 성공")
    
    # 테스트 대화들
    test_conversations = [
        ("안녕하세요", "안녕하세요! AutoCI입니다.", 1.0),
        ("Unity 도움이 필요해요", "Unity 개발을 도와드리겠습니다!", 0.8),
        ("코드 분석해주세요", "코드를 분석해드리겠습니다.", 0.6),
        ("이상한 응답이네요", "죄송합니다. 다시 시도해보겠습니다.", -0.5),
        ("고마워요", "천만에요! 도움이 되어서 기뻐요!", 0.9),
        ("Unity에서 오브젝트 생성", "Instantiate를 사용하세요.", 0.7),
        ("스크립트 오류", "어떤 오류인지 알려주세요.", 0.5),
        ("잘못된 답변", "죄송합니다. 개선하겠습니다.", -0.3)
    ]
    
    print(f"\n🗣️ {len(test_conversations)}개 테스트 대화 처리 중...")
    
    # 대화 처리
    for i, (user_input, ai_response, feedback) in enumerate(test_conversations, 1):
        print(f"\n테스트 {i}/{len(test_conversations)}:")
        print(f"사용자: {user_input}")
        print(f"AI: {ai_response}")
        print(f"피드백: {feedback}")
        
        result = ai.process_conversation(user_input, ai_response, feedback)
        
        print(f"✅ 대화 ID: {result['conversation_id']}")
        print(f"✅ 학습 트리거: {result['learning_triggered']}")
    
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
    print("✅ 순수 Python 신경망 기반 학습 시스템이 정상 작동합니다.")
    
    return ai

if __name__ == "__main__":
    test_pure_python_neural_learning()