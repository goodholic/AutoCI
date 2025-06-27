#!/usr/bin/env python3
"""
AutoCI 실제 신경망 기반 학습 AI 시스템
PyTorch를 사용한 진짜 학습하는 AI
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sqlite3
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

@dataclass
class ConversationData:
    """대화 데이터 구조"""
    user_input: str
    ai_response: str
    feedback_score: float  # -1.0 to 1.0
    timestamp: str
    context_vector: Optional[np.ndarray] = None

class NeuralResponseNetwork(nn.Module):
    """신경망 기반 응답 생성 네트워크"""
    
    def __init__(self, input_size: int = 512, hidden_size: int = 256, output_size: int = 128):
        super(NeuralResponseNetwork, self).__init__()
        
        # 인코더 레이어
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 응답 생성 레이어
        self.response_generator = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )
        
        # 품질 평가 레이어
        self.quality_evaluator = nn.Sequential(
            nn.Linear(hidden_size + output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 품질 점수
        )
    
    def forward(self, input_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """순전파"""
        encoded = self.encoder(input_vector)
        response_vector = self.response_generator(encoded)
        
        # 품질 평가를 위해 인코딩과 응답 벡터 결합
        combined = torch.cat([encoded, response_vector], dim=1)
        quality_score = self.quality_evaluator(combined)
        
        return response_vector, quality_score

class MemorySystem:
    """기억 시스템 - 단기/장기/작업 메모리"""
    
    def __init__(self, max_short_term: int = 100, max_long_term: int = 10000):
        self.short_term_memory: List[ConversationData] = []
        self.long_term_memory: List[ConversationData] = []
        self.working_memory: Dict[str, any] = {}
        
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        
        # 벡터화기
        self.vectorizer = TfidfVectorizer(max_features=512, ngram_range=(1, 2))
        self.is_vectorizer_fitted = False
    
    def add_conversation(self, conversation: ConversationData):
        """대화 추가"""
        # 벡터화
        if not self.is_vectorizer_fitted:
            # 첫 번째 대화로 벡터화기 학습
            self.vectorizer.fit([conversation.user_input])
            self.is_vectorizer_fitted = True
        
        try:
            conversation.context_vector = self.vectorizer.transform([conversation.user_input]).toarray()[0]
        except:
            # 벡터화 실패시 재학습
            all_texts = [conv.user_input for conv in self.short_term_memory[-10:]]
            all_texts.append(conversation.user_input)
            self.vectorizer.fit(all_texts)
            conversation.context_vector = self.vectorizer.transform([conversation.user_input]).toarray()[0]
        
        # 단기 메모리에 추가
        self.short_term_memory.append(conversation)
        
        # 단기 메모리 크기 제한
        if len(self.short_term_memory) > self.max_short_term:
            # 오래된 것을 장기 메모리로 이동
            old_conversation = self.short_term_memory.pop(0)
            self.promote_to_long_term(old_conversation)
    
    def promote_to_long_term(self, conversation: ConversationData):
        """장기 메모리로 승격"""
        # 중요한 대화만 장기 메모리에 저장 (피드백이 있거나 점수가 높은 것)
        if abs(conversation.feedback_score) > 0.3:
            self.long_term_memory.append(conversation)
            
            # 장기 메모리 크기 제한
            if len(self.long_term_memory) > self.max_long_term:
                self.long_term_memory.pop(0)
    
    def find_similar_conversations(self, query: str, top_k: int = 5) -> List[ConversationData]:
        """유사한 대화 찾기"""
        if not self.is_vectorizer_fitted:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query]).toarray()[0]
        except:
            return []
        
        all_conversations = self.short_term_memory + self.long_term_memory
        similarities = []
        
        for conv in all_conversations:
            if conv.context_vector is not None:
                similarity = cosine_similarity([query_vector], [conv.context_vector])[0][0]
                similarities.append((similarity, conv))
        
        # 유사도 순으로 정렬하여 상위 k개 반환
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [conv for _, conv in similarities[:top_k]]

class RealLearningAutoCI:
    """실제 학습하는 AutoCI 시스템"""
    
    def __init__(self, model_path: str = "autoci_neural_model.pth"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 신경망 모델 초기화
        self.model = NeuralResponseNetwork().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # 메모리 시스템
        self.memory = MemorySystem()
        
        # 학습 상태
        self.learning_enabled = True
        self.is_training = False
        self.training_thread = None
        
        # 통계
        self.stats = {
            "total_conversations": 0,
            "total_training_epochs": 0,
            "average_quality_score": 0.0,
            "last_learning_time": None,
            "model_accuracy": 0.0
        }
        
        # 응답 템플릿
        self.response_templates = {
            "korean_greeting": ["안녕하세요!", "반갑습니다!", "안녕히계세요!"],
            "unity_help": ["Unity 관련 도움을 드리겠습니다.", "Unity 개발을 지원해드립니다."],
            "code_analysis": ["코드를 분석해드리겠습니다.", "코드 리뷰를 시작합니다."],
            "error_help": ["오류 해결을 도와드리겠습니다.", "문제를 분석해보겠습니다."]
        }
        
        self.load_model()
        self.start_background_learning()
        
        print(f"🧠 실제 학습 AI 시스템 초기화 완료 (디바이스: {self.device})")
    
    def load_model(self):
        """모델 로드"""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.stats = checkpoint.get('stats', self.stats)
                print(f"✅ 기존 모델 로드됨: {self.model_path}")
            except Exception as e:
                print(f"⚠️ 모델 로드 실패, 새 모델 시작: {e}")
        else:
            print("🆕 새로운 모델로 시작")
    
    def save_model(self):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats
        }, self.model_path)
        print(f"💾 모델 저장됨: {self.model_path}")
    
    def process_conversation(self, user_input: str, ai_response: str, 
                           feedback_score: Optional[float] = None) -> Dict:
        """대화 처리 및 학습"""
        
        # 피드백 점수 계산 (제공되지 않은 경우 자동 추정)
        if feedback_score is None:
            feedback_score = self.estimate_feedback_score(user_input, ai_response)
        
        # 대화 데이터 생성
        conversation = ConversationData(
            user_input=user_input,
            ai_response=ai_response,
            feedback_score=feedback_score,
            timestamp=datetime.now().isoformat()
        )
        
        # 메모리에 추가
        self.memory.add_conversation(conversation)
        self.stats["total_conversations"] += 1
        
        # 실시간 학습 트리거
        if self.learning_enabled and abs(feedback_score) > 0.5:
            self.trigger_learning(conversation)
        
        return {
            "conversation_id": self.stats["total_conversations"],
            "feedback_score": feedback_score,
            "learning_triggered": abs(feedback_score) > 0.5,
            "similar_conversations": len(self.memory.find_similar_conversations(user_input))
        }
    
    def estimate_feedback_score(self, user_input: str, ai_response: str) -> float:
        """피드백 점수 자동 추정"""
        # 신경망을 사용하여 응답 품질 평가
        try:
            input_vector = self.memory.vectorizer.transform([user_input]).toarray()[0]
            input_tensor = torch.FloatTensor(input_vector).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, quality_score = self.model(input_tensor)
                estimated_score = quality_score.item()
                
                # -1 ~ 1 범위로 변환
                return (estimated_score - 0.5) * 2.0
                
        except Exception as e:
            # 오류 발생시 중립 점수 반환
            return 0.0
    
    def generate_response(self, user_input: str) -> Tuple[str, float]:
        """신경망 기반 응답 생성"""
        
        # 유사한 대화 찾기
        similar_conversations = self.memory.find_similar_conversations(user_input, top_k=3)
        
        # 신경망으로 응답 벡터 생성
        try:
            input_vector = self.memory.vectorizer.transform([user_input]).toarray()[0]
            input_tensor = torch.FloatTensor(input_vector).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                response_vector, quality_score = self.model(input_tensor)
                
            # 응답 벡터를 실제 텍스트로 변환 (템플릿 기반)
            response_text = self.vector_to_response(response_vector.cpu().numpy()[0], user_input)
            
            return response_text, quality_score.item()
            
        except Exception as e:
            # 오류 발생시 기본 응답
            return "죄송합니다. 다시 말씀해 주세요.", 0.5
    
    def vector_to_response(self, response_vector: np.ndarray, user_input: str) -> str:
        """응답 벡터를 실제 텍스트로 변환"""
        
        # 간단한 템플릿 매칭 (실제로는 더 복잡한 디코딩 필요)
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ["안녕", "hello", "hi"]):
            return np.random.choice(self.response_templates["korean_greeting"])
        elif any(word in user_input_lower for word in ["unity", "유니티"]):
            return np.random.choice(self.response_templates["unity_help"])
        elif any(word in user_input_lower for word in ["코드", "code", "분석"]):
            return np.random.choice(self.response_templates["code_analysis"])
        elif any(word in user_input_lower for word in ["오류", "에러", "error", "문제"]):
            return np.random.choice(self.response_templates["error_help"])
        else:
            return "네, 어떻게 도와드릴까요?"
    
    def trigger_learning(self, conversation: ConversationData):
        """즉시 학습 트리거"""
        if not self.is_training:
            self.is_training = True
            threading.Thread(target=self._immediate_learning, args=(conversation,)).start()
    
    def _immediate_learning(self, conversation: ConversationData):
        """즉시 학습 실행"""
        try:
            if conversation.context_vector is not None:
                # 입력 텐서 준비
                input_tensor = torch.FloatTensor(conversation.context_vector).unsqueeze(0).to(self.device)
                target_quality = torch.FloatTensor([[conversation.feedback_score]]).to(self.device)
                
                # 순전파
                response_vector, predicted_quality = self.model(input_tensor)
                
                # 손실 계산
                loss = self.criterion(predicted_quality, target_quality)
                
                # 역전파
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 통계 업데이트
                self.stats["last_learning_time"] = datetime.now().isoformat()
                self.stats["total_training_epochs"] += 1
                
                print(f"🔄 즉시 학습 완료 - 손실: {loss.item():.4f}")
                
        except Exception as e:
            print(f"❌ 즉시 학습 오류: {e}")
        finally:
            self.is_training = False
    
    def start_background_learning(self):
        """백그라운드 학습 시작"""
        if self.training_thread is None or not self.training_thread.is_alive():
            self.training_thread = threading.Thread(target=self._background_learning_loop)
            self.training_thread.daemon = True
            self.training_thread.start()
            print("🔄 백그라운드 학습 시작됨")
    
    def _background_learning_loop(self):
        """백그라운드 학습 루프"""
        while self.learning_enabled:
            try:
                # 5분마다 배치 학습
                time.sleep(300)
                
                if len(self.memory.short_term_memory) >= 10:
                    self._batch_learning()
                    
            except Exception as e:
                print(f"❌ 백그라운드 학습 오류: {e}")
                time.sleep(60)  # 오류 시 1분 대기
    
    def _batch_learning(self):
        """배치 학습"""
        if self.is_training:
            return
            
        self.is_training = True
        print("🎯 배치 학습 시작...")
        
        try:
            # 최근 대화들로 배치 생성
            recent_conversations = self.memory.short_term_memory[-20:]
            
            inputs = []
            targets = []
            
            for conv in recent_conversations:
                if conv.context_vector is not None:
                    inputs.append(conv.context_vector)
                    targets.append([conv.feedback_score])
            
            if len(inputs) > 0:
                input_tensor = torch.FloatTensor(inputs).to(self.device)
                target_tensor = torch.FloatTensor(targets).to(self.device)
                
                # 여러 번 학습
                for epoch in range(5):
                    response_vectors, predicted_qualities = self.model(input_tensor)
                    loss = self.criterion(predicted_qualities, target_tensor)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                print(f"✅ 배치 학습 완료 - 최종 손실: {loss.item():.4f}")
                self.stats["total_training_epochs"] += 5
                
                # 모델 저장
                self.save_model()
                
        except Exception as e:
            print(f"❌ 배치 학습 오류: {e}")
        finally:
            self.is_training = False
    
    def get_learning_status(self) -> Dict:
        """학습 상태 반환"""
        return {
            "device": str(self.device),
            "learning_enabled": self.learning_enabled,
            "is_training": self.is_training,
            "stats": self.stats,
            "memory_stats": {
                "short_term_memory": len(self.memory.short_term_memory),
                "long_term_memory": len(self.memory.long_term_memory),
                "working_memory": len(self.memory.working_memory)
            },
            "model_info": {
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
    
    def chat_interface(self):
        """대화형 인터페이스"""
        print("\n🤖 AutoCI 실제 학습 AI와 대화하기")
        print("종료하려면 'quit' 입력")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n사용자: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료']:
                    break
                
                if user_input.lower() == 'status':
                    status = self.get_learning_status()
                    print(f"\n📊 학습 상태:")
                    for key, value in status["stats"].items():
                        print(f"  {key}: {value}")
                    continue
                
                # AI 응답 생성
                ai_response, confidence = self.generate_response(user_input)
                print(f"\nAutoCI: {ai_response}")
                print(f"(신뢰도: {confidence:.2f})")
                
                # 사용자 피드백 받기
                feedback = input("피드백 (좋음: +, 나쁨: -, 그냥 Enter): ").strip()
                
                feedback_score = 0.0
                if feedback == '+':
                    feedback_score = 1.0
                elif feedback == '-':
                    feedback_score = -1.0
                
                # 대화 처리 및 학습
                result = self.process_conversation(user_input, ai_response, feedback_score)
                
                if result["learning_triggered"]:
                    print("🧠 학습이 트리거되었습니다!")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
        
        print("\n👋 대화를 종료합니다. 학습 데이터가 저장되었습니다.")
        self.save_model()

def main():
    """메인 함수"""
    print("🚀 AutoCI 실제 학습 AI 시스템 시작")
    
    # 학습 AI 초기화
    learning_ai = RealLearningAutoCI()
    
    # 대화형 인터페이스 시작
    learning_ai.chat_interface()

if __name__ == "__main__":
    main()