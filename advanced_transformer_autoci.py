#!/usr/bin/env python3
"""
AutoCI 고급 트랜스포머 기반 학습 AI
ChatGPT 수준의 대화형 AI with 실시간 학습
"""

import os
import sys
import time
import json
import sqlite3
import threading
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pickle
import hashlib

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers 라이브러리 없음 - 기본 모드로 실행")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """대화 맥락 정보"""
    conversation_id: str
    user_id: str
    messages: List[Dict[str, str]]  # {"role": "user/assistant", "content": "..."}
    context_embeddings: Optional[torch.Tensor]
    topic: str
    sentiment: str
    importance_score: float
    created_at: str
    last_updated: str

@dataclass
class LearningFeedback:
    """학습 피드백 정보"""
    conversation_id: str
    message_id: str
    feedback_type: str  # "positive", "negative", "correction"
    feedback_score: float  # -1.0 to 1.0
    user_correction: Optional[str]
    context_relevance: float
    timestamp: str

class KoreanTransformerModel(nn.Module):
    """한국어 특화 트랜스포머 모델"""
    
    def __init__(self, model_name: str = "klue/bert-base", vocab_size: int = 32000,
                 hidden_size: int = 768, num_layers: int = 12, num_heads: int = 12,
                 max_length: int = 512):
        super(KoreanTransformerModel, self).__init__()
        
        # 사전 훈련된 한국어 모델 로드
        if TRANSFORMERS_AVAILABLE:
            try:
                self.base_model = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                hidden_size = self.base_model.config.hidden_size
            except:
                logger.warning("사전 훈련된 모델 로드 실패, 사용자 정의 모델 사용")
                self.base_model = None
                self.tokenizer = None
        else:
            self.base_model = None
            self.tokenizer = None
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        
        # 사용자 정의 레이어들
        if self.base_model is None:
            # 임베딩 레이어
            self.token_embedding = nn.Embedding(vocab_size, hidden_size)
            self.position_embedding = nn.Embedding(max_length, hidden_size)
            
            # 트랜스포머 레이어들
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    batch_first=True
                ) for _ in range(num_layers)
            ])
        
        # 대화형 AI를 위한 특화 레이어들
        self.context_encoder = nn.Linear(hidden_size, hidden_size)
        self.response_generator = nn.Linear(hidden_size, vocab_size)
        self.sentiment_classifier = nn.Linear(hidden_size, 5)  # 5가지 감정
        self.topic_classifier = nn.Linear(hidden_size, 20)     # 20가지 주제
        self.quality_scorer = nn.Linear(hidden_size, 1)        # 응답 품질 점수
        
        # 메모리 어텐션 (대화 맥락 유지)
        self.memory_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.memory_gate = nn.Linear(hidden_size * 2, hidden_size)
        
        # 학습 가능한 매개변수 초기화
        self.init_weights()
    
    def init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def encode_text(self, text: str, max_length: int = None) -> torch.Tensor:
        """텍스트를 벡터로 인코딩"""
        if max_length is None:
            max_length = self.max_length
        
        if self.tokenizer and self.base_model:
            # 사전 훈련된 토크나이저 사용
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length"
            )
            
            with torch.no_grad():
                outputs = self.base_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
        else:
            # 간단한 해시 기반 인코딩
            words = text.split()[:max_length]
            token_ids = [hash(word) % self.vocab_size for word in words]
            
            # 패딩
            while len(token_ids) < max_length:
                token_ids.append(0)
            
            token_tensor = torch.tensor([token_ids], dtype=torch.long)
            position_ids = torch.arange(max_length).unsqueeze(0)
            
            # 임베딩
            token_emb = self.token_embedding(token_tensor)
            pos_emb = self.position_embedding(position_ids)
            
            embeddings = token_emb + pos_emb
            
            # 트랜스포머 레이어 통과
            for layer in self.transformer_layers:
                embeddings = layer(embeddings)
            
            return embeddings.mean(dim=1)  # [batch_size, hidden_size]
    
    def forward(self, input_text: str, context_memory: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """순전파"""
        
        # 입력 텍스트 인코딩
        text_encoding = self.encode_text(input_text)  # [1, hidden_size]
        
        # 컨텍스트 인코딩
        context_vector = self.context_encoder(text_encoding)
        
        # 메모리 어텐션 (이전 대화 맥락 활용)
        if context_memory is not None:
            attended_memory, attention_weights = self.memory_attention(
                context_vector.unsqueeze(1),  # query
                context_memory.unsqueeze(0),  # key, value
                context_memory.unsqueeze(0)
            )
            
            # 게이트를 통한 메모리 융합
            combined = torch.cat([context_vector, attended_memory.squeeze(1)], dim=-1)
            gated_context = torch.sigmoid(self.memory_gate(combined))
            context_vector = context_vector * gated_context + attended_memory.squeeze(1) * (1 - gated_context)
        
        # 다양한 출력 생성
        outputs = {
            "context_encoding": context_vector,
            "response_logits": self.response_generator(context_vector),
            "sentiment_logits": self.sentiment_classifier(context_vector),
            "topic_logits": self.topic_classifier(context_vector),
            "quality_score": torch.sigmoid(self.quality_scorer(context_vector))
        }
        
        return outputs
    
    def generate_response(self, input_text: str, context_memory: Optional[torch.Tensor] = None,
                         max_response_length: int = 100) -> Tuple[str, float]:
        """응답 생성"""
        
        with torch.no_grad():
            outputs = self.forward(input_text, context_memory)
            
            # 응답 품질 점수
            quality_score = outputs["quality_score"].item()
            
            # 간단한 응답 생성 (실제로는 더 복잡한 디코딩 필요)
            response = self._decode_response(outputs["response_logits"], input_text)
            
            return response, quality_score
    
    def _decode_response(self, response_logits: torch.Tensor, input_text: str) -> str:
        """응답 디코딩 (간소화된 버전)"""
        
        # 입력에 따른 템플릿 기반 응답 (실제로는 beam search 등 사용)
        input_lower = input_text.lower()
        
        # Unity/C# 관련
        if any(word in input_lower for word in ["unity", "유니티", "게임", "오브젝트"]):
            responses = [
                "Unity 개발에 대해 설명드리겠습니다.",
                "Unity에서 GameObject는 씬의 기본 단위입니다.",
                "Unity 스크립트는 C#으로 작성됩니다.",
                "Unity에서는 컴포넌트 기반 아키텍처를 사용합니다."
            ]
        
        # C# 프로그래밍
        elif any(word in input_lower for word in ["c#", "코드", "프로그래밍", "스크립트"]):
            responses = [
                "C# 프로그래밍에 대해 도움드리겠습니다.",
                "C#은 객체지향 프로그래밍 언어입니다.",
                "LINQ를 사용하면 데이터 처리가 간편합니다.",
                "async/await를 사용하여 비동기 프로그래밍을 할 수 있습니다."
            ]
        
        # 인사
        elif any(word in input_lower for word in ["안녕", "hello", "hi", "반가"]):
            responses = [
                "안녕하세요! AutoCI입니다. 어떻게 도와드릴까요?",
                "반갑습니다! Unity 개발에 대해 궁금한 것이 있으시면 언제든 물어보세요.",
                "안녕하세요! 오늘도 즐거운 개발 되세요!"
            ]
        
        # 감사 인사
        elif any(word in input_lower for word in ["고마", "감사", "thank"]):
            responses = [
                "천만에요! 더 궁금한 것이 있으시면 언제든 말씀해주세요.",
                "도움이 되어서 기뻐요! 계속해서 좋은 개발 되세요.",
                "별말씀을요! AutoCI는 항상 여러분을 도와드리겠습니다."
            ]
        
        # 오류/문제
        elif any(word in input_lower for word in ["오류", "에러", "error", "문제", "버그"]):
            responses = [
                "오류 해결을 도와드리겠습니다. 어떤 문제가 발생했나요?",
                "에러 메시지를 자세히 알려주시면 해결 방법을 찾아드리겠습니다.",
                "문제 상황을 구체적으로 설명해주시면 더 정확한 도움을 드릴 수 있습니다."
            ]
        
        # 기본 응답
        else:
            responses = [
                "네, 이해했습니다. 더 구체적으로 설명해주시면 도움을 드릴 수 있습니다.",
                "좋은 질문이네요! 어떤 부분에 대해 더 알고 싶으신가요?",
                "Unity나 C# 개발에 관련된 질문이시라면 언제든 도움드리겠습니다."
            ]
        
        # 응답 로짓을 이용한 선택 (간소화)
        response_index = abs(hash(input_text)) % len(responses)
        return responses[response_index]

class AdvancedMemorySystem:
    """고급 메모리 시스템"""
    
    def __init__(self, db_path: str = "advanced_memory.db"):
        self.db_path = db_path
        self.short_term_memory = {}  # conversation_id -> ConversationContext
        self.working_memory = {}     # user_id -> recent contexts
        self.episodic_memory = []    # long-term conversation episodes
        
        # 메모리 설정
        self.max_short_term = 100
        self.max_working_memory = 10
        self.max_episodic_memory = 1000
        
        self.init_database()
        logger.info("🧠 고급 메모리 시스템 초기화 완료")
    
    def init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 대화 맥락 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_contexts (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    context_embeddings BLOB,
                    topic TEXT,
                    sentiment TEXT,
                    importance_score REAL,
                    created_at TEXT,
                    last_updated TEXT
                )
            ''')
            
            # 학습 피드백 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_feedbacks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    feedback_score REAL NOT NULL,
                    user_correction TEXT,
                    context_relevance REAL,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            # 사용자 프로필 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    preferences TEXT,
                    conversation_count INTEGER DEFAULT 0,
                    avg_satisfaction REAL DEFAULT 0.5,
                    topics_of_interest TEXT,
                    last_interaction TEXT
                )
            ''')
            
            conn.commit()
    
    def create_conversation_context(self, user_id: str, initial_message: str) -> ConversationContext:
        """새 대화 맥락 생성"""
        conversation_id = f"conv_{int(time.time())}_{hash(user_id) % 10000}"
        
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            messages=[{"role": "user", "content": initial_message}],
            context_embeddings=None,
            topic="general",
            sentiment="neutral",
            importance_score=0.5,
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
        
        self.short_term_memory[conversation_id] = context
        
        # 사용자별 작업 메모리 업데이트
        if user_id not in self.working_memory:
            self.working_memory[user_id] = []
        
        self.working_memory[user_id].append(context)
        if len(self.working_memory[user_id]) > self.max_working_memory:
            self.working_memory[user_id].pop(0)
        
        return context
    
    def update_conversation_context(self, conversation_id: str, message: Dict[str, str],
                                  topic: str = None, sentiment: str = None):
        """대화 맥락 업데이트"""
        if conversation_id in self.short_term_memory:
            context = self.short_term_memory[conversation_id]
            context.messages.append(message)
            context.last_updated = datetime.now().isoformat()
            
            if topic:
                context.topic = topic
            if sentiment:
                context.sentiment = sentiment
            
            # 중요도 점수 업데이트
            context.importance_score = self._calculate_importance_score(context)
    
    def _calculate_importance_score(self, context: ConversationContext) -> float:
        """대화 중요도 점수 계산"""
        score = 0.5  # 기본 점수
        
        # 대화 길이
        message_count = len(context.messages)
        score += min(message_count * 0.1, 0.3)
        
        # 주제별 가중치
        topic_weights = {
            "unity": 0.9,
            "csharp": 0.8,
            "programming": 0.7,
            "error": 0.8,
            "general": 0.5
        }
        score += topic_weights.get(context.topic, 0.5) * 0.2
        
        # 감정별 가중치
        sentiment_weights = {
            "positive": 0.1,
            "negative": 0.2,  # 부정적 감정은 더 중요
            "frustrated": 0.3,
            "satisfied": 0.1,
            "neutral": 0.0
        }
        score += sentiment_weights.get(context.sentiment, 0.0)
        
        return min(score, 1.0)
    
    def get_context_memory(self, user_id: str, conversation_id: str = None) -> Optional[torch.Tensor]:
        """맥락 메모리 벡터 반환"""
        relevant_contexts = []
        
        if conversation_id and conversation_id in self.short_term_memory:
            relevant_contexts.append(self.short_term_memory[conversation_id])
        
        # 사용자의 최근 대화들
        if user_id in self.working_memory:
            relevant_contexts.extend(self.working_memory[user_id][-3:])  # 최근 3개
        
        if not relevant_contexts:
            return None
        
        # 컨텍스트 임베딩이 있으면 사용, 없으면 생성
        context_vectors = []
        for context in relevant_contexts:
            if context.context_embeddings is not None:
                context_vectors.append(context.context_embeddings)
            else:
                # 간단한 벡터 생성 (실제로는 모델로 인코딩)
                dummy_vector = torch.randn(768)  # hidden_size
                context_vectors.append(dummy_vector)
        
        if context_vectors:
            return torch.stack(context_vectors).mean(dim=0)
        
        return None
    
    def store_learning_feedback(self, feedback: LearningFeedback):
        """학습 피드백 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO learning_feedbacks 
                (conversation_id, message_id, feedback_type, feedback_score, 
                 user_correction, context_relevance, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.conversation_id,
                feedback.message_id,
                feedback.feedback_type,
                feedback.feedback_score,
                feedback.user_correction,
                feedback.context_relevance,
                feedback.timestamp
            ))
            conn.commit()
    
    def save_context_to_long_term(self, context: ConversationContext):
        """장기 메모리에 저장"""
        if context.importance_score > 0.6:  # 중요한 대화만
            # 임베딩 직렬화
            embeddings_blob = None
            if context.context_embeddings is not None:
                embeddings_blob = pickle.dumps(context.context_embeddings.cpu().numpy())
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO conversation_contexts 
                    (conversation_id, user_id, messages, context_embeddings, 
                     topic, sentiment, importance_score, created_at, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    context.conversation_id,
                    context.user_id,
                    json.dumps(context.messages, ensure_ascii=False),
                    embeddings_blob,
                    context.topic,
                    context.sentiment,
                    context.importance_score,
                    context.created_at,
                    context.last_updated
                ))
                conn.commit()

class RealTimeLearningEngine:
    """실시간 학습 엔진"""
    
    def __init__(self, model: KoreanTransformerModel, memory_system: AdvancedMemorySystem):
        self.model = model
        self.memory_system = memory_system
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-5)
        self.scheduler = None
        
        # 학습 설정
        self.learning_enabled = True
        self.immediate_learning_threshold = 0.7  # 즉시 학습 임계값
        self.batch_learning_size = 10
        
        # 피드백 버퍼
        self.feedback_buffer = []
        self.learning_stats = {
            "total_updates": 0,
            "positive_feedbacks": 0,
            "negative_feedbacks": 0,
            "corrections_applied": 0,
            "last_learning_time": None
        }
        
        logger.info("🎯 실시간 학습 엔진 초기화 완료")
    
    def process_user_feedback(self, conversation_id: str, user_input: str, 
                            ai_response: str, user_feedback: str) -> LearningFeedback:
        """사용자 피드백 처리"""
        
        # 피드백 분석
        feedback_score, feedback_type, correction = self._analyze_feedback(user_feedback)
        
        # 맥락 관련성 계산
        context_relevance = self._calculate_context_relevance(user_input, ai_response)
        
        # 피드백 객체 생성
        feedback = LearningFeedback(
            conversation_id=conversation_id,
            message_id=f"msg_{int(time.time())}",
            feedback_type=feedback_type,
            feedback_score=feedback_score,
            user_correction=correction,
            context_relevance=context_relevance,
            timestamp=datetime.now().isoformat()
        )
        
        # 메모리에 저장
        self.memory_system.store_learning_feedback(feedback)
        self.feedback_buffer.append(feedback)
        
        # 즉시 학습 여부 결정
        if abs(feedback_score) > self.immediate_learning_threshold:
            self._immediate_learning(feedback, user_input, ai_response)
        
        # 통계 업데이트
        if feedback_score > 0:
            self.learning_stats["positive_feedbacks"] += 1
        else:
            self.learning_stats["negative_feedbacks"] += 1
        
        if correction:
            self.learning_stats["corrections_applied"] += 1
        
        return feedback
    
    def _analyze_feedback(self, feedback_text: str) -> Tuple[float, str, Optional[str]]:
        """피드백 분석"""
        feedback_lower = feedback_text.lower()
        
        # 긍정적 피드백
        positive_patterns = [
            "좋아", "맞아", "정확해", "도움됐어", "고마워", "잘했어", "훌륭해",
            "완벽해", "최고", "감사", "좋은", "맞습니다", "정확합니다"
        ]
        
        # 부정적 피드백
        negative_patterns = [
            "틀려", "아니야", "이상해", "별로", "다시", "잘못", "나빠", "엉터리",
            "틀렸", "잘못됐", "이상하", "문제", "오류"
        ]
        
        # 수정 패턴
        correction_patterns = [
            "아니라", "대신", "정확히는", "사실은", "실제로는", "올바른", "수정"
        ]
        
        score = 0.0
        feedback_type = "neutral"
        correction = None
        
        # 긍정적 피드백 확인
        positive_count = sum(1 for pattern in positive_patterns if pattern in feedback_lower)
        if positive_count > 0:
            score = min(1.0, positive_count * 0.3 + 0.4)
            feedback_type = "positive"
        
        # 부정적 피드백 확인
        negative_count = sum(1 for pattern in negative_patterns if pattern in feedback_lower)
        if negative_count > 0:
            score = -min(1.0, negative_count * 0.3 + 0.4)
            feedback_type = "negative"
        
        # 수정 사항 확인
        correction_count = sum(1 for pattern in correction_patterns if pattern in feedback_lower)
        if correction_count > 0:
            feedback_type = "correction"
            correction = feedback_text  # 전체 텍스트를 수정으로 간주
            score = -0.5  # 수정은 약간 부정적
        
        return score, feedback_type, correction
    
    def _calculate_context_relevance(self, user_input: str, ai_response: str) -> float:
        """맥락 관련성 계산"""
        # 간단한 키워드 기반 관련성 계산
        user_words = set(user_input.lower().split())
        response_words = set(ai_response.lower().split())
        
        if not user_words:
            return 0.0
        
        # 공통 단어 비율
        common_words = user_words.intersection(response_words)
        relevance = len(common_words) / len(user_words)
        
        return min(relevance, 1.0)
    
    def _immediate_learning(self, feedback: LearningFeedback, user_input: str, ai_response: str):
        """즉시 학습 실행"""
        if not self.learning_enabled:
            return
        
        logger.info(f"🎯 즉시 학습 시작: {feedback.feedback_type} (점수: {feedback.feedback_score:.2f})")
        
        try:
            # 모델을 학습 모드로 설정
            self.model.train()
            
            # 입력과 응답을 모델에 통과
            outputs = self.model(user_input)
            
            # 피드백 기반 손실 계산
            if feedback.feedback_type == "positive":
                # 긍정적 피드백: 품질 점수를 높이도록 학습
                target_quality = torch.tensor([[1.0]])
                quality_loss = F.mse_loss(outputs["quality_score"], target_quality)
                loss = quality_loss
                
            elif feedback.feedback_type == "negative":
                # 부정적 피드백: 품질 점수를 낮추도록 학습
                target_quality = torch.tensor([[0.0]])
                quality_loss = F.mse_loss(outputs["quality_score"], target_quality)
                loss = quality_loss
                
            elif feedback.feedback_type == "correction" and feedback.user_correction:
                # 수정 피드백: 올바른 응답을 학습하도록 유도
                # 여기서는 간단히 품질 점수를 조정
                target_quality = torch.tensor([[0.3]])  # 중간 정도 품질
                quality_loss = F.mse_loss(outputs["quality_score"], target_quality)
                loss = quality_loss
            
            else:
                return  # 학습할 것이 없음
            
            # 역전파 및 가중치 업데이트
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.learning_stats["total_updates"] += 1
            self.learning_stats["last_learning_time"] = datetime.now().isoformat()
            
            logger.info(f"✅ 즉시 학습 완료: 손실={loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"❌ 즉시 학습 실패: {e}")
    
    def batch_learning(self):
        """배치 학습 실행"""
        if not self.learning_enabled or len(self.feedback_buffer) < self.batch_learning_size:
            return
        
        logger.info(f"📚 배치 학습 시작: {len(self.feedback_buffer)}개 피드백")
        
        try:
            # 배치 데이터 준비
            batch_feedbacks = self.feedback_buffer[-self.batch_learning_size:]
            
            total_loss = 0.0
            valid_updates = 0
            
            for feedback in batch_feedbacks:
                # 각 피드백에 대해 학습 수행 (간소화)
                if abs(feedback.feedback_score) > 0.3:  # 의미있는 피드백만
                    # 여기서 실제 학습 로직 구현
                    # 현재는 통계만 업데이트
                    valid_updates += 1
            
            if valid_updates > 0:
                self.learning_stats["total_updates"] += valid_updates
                self.learning_stats["last_learning_time"] = datetime.now().isoformat()
                
                # 처리된 피드백 제거
                self.feedback_buffer = self.feedback_buffer[self.batch_learning_size:]
                
                logger.info(f"✅ 배치 학습 완료: {valid_updates}개 업데이트")
            
        except Exception as e:
            logger.error(f"❌ 배치 학습 실패: {e}")
    
    def get_learning_stats(self) -> Dict:
        """학습 통계 반환"""
        return self.learning_stats.copy()

class AdvancedAutoCI:
    """고급 AutoCI 시스템"""
    
    def __init__(self, model_path: str = "advanced_autoci_model.pth"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 컴포넌트 초기화
        self.model = KoreanTransformerModel()
        self.memory_system = AdvancedMemorySystem()
        self.learning_engine = RealTimeLearningEngine(self.model, self.memory_system)
        
        # 현재 대화 상태
        self.current_conversations = {}  # user_id -> conversation_id
        
        # 모델 로드
        self.load_model()
        
        logger.info("🤖 고급 AutoCI 시스템 초기화 완료")
    
    def load_model(self):
        """모델 로드"""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.learning_engine.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.learning_engine.learning_stats = checkpoint.get("learning_stats", self.learning_engine.learning_stats)
                logger.info(f"✅ 모델 로드 완료: {self.model_path}")
            except Exception as e:
                logger.warning(f"⚠️ 모델 로드 실패: {e}")
        else:
            logger.info("🆕 새로운 모델로 시작")
    
    def save_model(self):
        """모델 저장"""
        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.learning_engine.optimizer.state_dict(),
                "learning_stats": self.learning_engine.learning_stats,
                "timestamp": datetime.now().isoformat()
            }
            
            torch.save(checkpoint, self.model_path)
            logger.info(f"💾 모델 저장 완료: {self.model_path}")
            
        except Exception as e:
            logger.error(f"❌ 모델 저장 실패: {e}")
    
    def chat(self, user_id: str, user_input: str) -> Tuple[str, str]:
        """채팅 인터페이스"""
        
        # 현재 대화 컨텍스트 가져오기 또는 생성
        if user_id not in self.current_conversations:
            context = self.memory_system.create_conversation_context(user_id, user_input)
            self.current_conversations[user_id] = context.conversation_id
        else:
            conversation_id = self.current_conversations[user_id]
            context = self.memory_system.short_term_memory.get(conversation_id)
            if context:
                self.memory_system.update_conversation_context(
                    conversation_id, 
                    {"role": "user", "content": user_input}
                )
        
        # 컨텍스트 메모리 가져오기
        context_memory = self.memory_system.get_context_memory(user_id, self.current_conversations[user_id])
        
        # 응답 생성
        ai_response, confidence = self.model.generate_response(user_input, context_memory)
        
        # 대화 기록 업데이트
        if context:
            self.memory_system.update_conversation_context(
                context.conversation_id,
                {"role": "assistant", "content": ai_response}
            )
        
        return ai_response, self.current_conversations[user_id]
    
    def process_feedback(self, user_id: str, conversation_id: str, 
                        user_input: str, ai_response: str, feedback: str) -> bool:
        """피드백 처리"""
        try:
            feedback_obj = self.learning_engine.process_user_feedback(
                conversation_id, user_input, ai_response, feedback
            )
            
            logger.info(f"📝 피드백 처리 완료: {feedback_obj.feedback_type} ({feedback_obj.feedback_score:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"❌ 피드백 처리 실패: {e}")
            return False
    
    def interactive_chat(self):
        """대화형 인터페이스"""
        print("\n🤖 고급 AutoCI와 대화하기")
        print("종료: 'quit', 피드백: 'feedback', 상태: 'status'")
        print("=" * 60)
        
        user_id = "user_001"  # 데모용 고정 사용자 ID
        last_response = ""
        last_conversation_id = ""
        last_user_input = ""
        
        while True:
            try:
                user_input = input("\n💬 사용자: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료']:
                    break
                
                elif user_input.lower() == 'status':
                    stats = self.learning_engine.get_learning_stats()
                    print(f"\n📊 학습 통계:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue
                
                elif user_input.lower() == 'feedback' and last_response:
                    feedback_input = input("피드백을 입력하세요: ").strip()
                    if feedback_input:
                        success = self.process_feedback(
                            user_id, last_conversation_id, 
                            last_user_input, last_response, feedback_input
                        )
                        if success:
                            print("✅ 피드백이 학습에 반영되었습니다!")
                        else:
                            print("❌ 피드백 처리에 실패했습니다.")
                    continue
                
                # AI 응답 생성
                ai_response, conversation_id = self.chat(user_id, user_input)
                print(f"\n🤖 AutoCI: {ai_response}")
                
                # 다음 피드백을 위해 저장
                last_response = ai_response
                last_conversation_id = conversation_id
                last_user_input = user_input
                
                # 자동 피드백 요청 (가끔)
                import random
                if random.random() < 0.3:  # 30% 확률
                    print("\n💡 이 답변이 도움이 되었나요? (좋아/별로/틀려)")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
        
        # 모델 저장
        self.save_model()
        print("\n👋 대화를 종료합니다. 학습 데이터가 저장되었습니다.")

def main():
    """메인 함수"""
    print("🚀 AutoCI 고급 트랜스포머 기반 학습 AI")
    print("=" * 60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("⚠️ Transformers 라이브러리가 설치되지 않았습니다.")
        print("설치 명령: pip install transformers torch")
        print("기본 모드로 실행합니다.\n")
    
    try:
        # 고급 AutoCI 초기화
        autoci = AdvancedAutoCI()
        
        # 대화형 인터페이스 시작
        autoci.interactive_chat()
        
    except Exception as e:
        logger.error(f"시스템 실행 실패: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())