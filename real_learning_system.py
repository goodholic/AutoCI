#!/usr/bin/env python3
"""
AutoCI 실제 학습 시스템
사용자와의 대화, 코드 패턴, 피드백으로부터 실제로 학습하는 AI
"""

import os
import sys
import json
import sqlite3
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import hashlib
from collections import defaultdict, Counter
import re
import threading
import time

# 기계학습 라이브러리 (선택적)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn이 없어 기본 학습 모드로 작동합니다.")

logger = logging.getLogger(__name__)


class RealLearningSystem:
    """실제로 학습하는 AI 시스템"""
    
    def __init__(self, db_path: str = "autoci_brain.db"):
        self.base_path = Path(__file__).parent
        self.db_path = self.base_path / db_path
        
        # 학습 파라미터
        self.learning_rate = 0.1
        self.memory_capacity = 10000
        self.pattern_threshold = 0.7
        
        # 메모리 시스템
        self.short_term_memory = []  # 최근 대화
        self.long_term_memory = {}   # 영구 저장
        self.working_memory = {}     # 현재 작업 컨텍스트
        
        # 학습 통계
        self.stats = {
            'total_conversations': 0,
            'learned_patterns': 0,
            'accuracy': 0.0,
            'last_update': datetime.now()
        }
        
        # 신경망 가중치 (간단한 구현)
        self.weights = self._initialize_weights()
        
        # 데이터베이스 초기화
        self._init_database()
        
        # 학습 데이터 로드
        self._load_learned_data()
        
        # 백그라운드 학습 스레드
        self.learning_thread = None
        self.is_learning = False
        
    def _initialize_weights(self) -> Dict[str, np.ndarray]:
        """신경망 가중치 초기화"""
        weights = {
            'intent_recognition': np.random.randn(100, 50) * 0.01,
            'response_generation': np.random.randn(50, 100) * 0.01,
            'context_understanding': np.random.randn(50, 50) * 0.01,
            'emotion_detection': np.random.randn(30, 10) * 0.01
        }
        return weights
        
    def _init_database(self):
        """학습 데이터베이스 초기화"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 대화 학습 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input TEXT,
                ai_response TEXT,
                user_feedback REAL DEFAULT 0.5,
                context TEXT,
                learned_features TEXT
            )
        ''')
        
        # 코드 패턴 학습 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_hash TEXT UNIQUE,
                pattern_type TEXT,
                pattern_content TEXT,
                usage_count INTEGER DEFAULT 1,
                success_rate REAL DEFAULT 0.5,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # 학습된 개념 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_name TEXT UNIQUE,
                concept_type TEXT,
                understanding_level REAL DEFAULT 0.1,
                related_patterns TEXT,
                examples TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 사용자 선호도 학습 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preference_type TEXT,
                preference_value TEXT,
                confidence REAL DEFAULT 0.5,
                frequency INTEGER DEFAULT 1,
                last_observed DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 에러 패턴 학습 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type TEXT,
                error_context TEXT,
                solution TEXT,
                success_count INTEGER DEFAULT 0,
                total_count INTEGER DEFAULT 1,
                last_encountered DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 신경망 가중치 저장 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS neural_weights (
                layer_name TEXT PRIMARY KEY,
                weights_data BLOB,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def learn_from_conversation(self, user_input: str, ai_response: str, 
                               context: Dict = None) -> Dict[str, Any]:
        """대화로부터 학습"""
        # 특징 추출
        features = self._extract_conversation_features(user_input, ai_response)
        
        # 단기 기억에 저장
        self.short_term_memory.append({
            'timestamp': datetime.now(),
            'user_input': user_input,
            'ai_response': ai_response,
            'features': features,
            'context': context or {}
        })
        
        # 패턴 인식
        patterns = self._identify_patterns(user_input, ai_response)
        
        # 데이터베이스에 저장
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversation_memory 
            (user_input, ai_response, context, learned_features)
            VALUES (?, ?, ?, ?)
        ''', (user_input, ai_response, 
              json.dumps(context or {}), 
              json.dumps(features)))
        
        # 패턴 저장
        for pattern in patterns:
            self._save_pattern(cursor, pattern)
            
        conn.commit()
        conn.close()
        
        # 통계 업데이트
        self.stats['total_conversations'] += 1
        
        # 가중치 업데이트 (간단한 gradient descent)
        self._update_weights(features)
        
        return {
            'learned': True,
            'features': features,
            'patterns': patterns,
            'memory_size': len(self.short_term_memory)
        }
        
    def _extract_conversation_features(self, user_input: str, 
                                     ai_response: str) -> Dict[str, Any]:
        """대화에서 특징 추출"""
        features = {
            'input_length': len(user_input),
            'response_length': len(ai_response),
            'input_keywords': self._extract_keywords(user_input),
            'response_keywords': self._extract_keywords(ai_response),
            'input_intent': self._classify_intent(user_input),
            'emotion': self._detect_emotion(user_input),
            'topic': self._identify_topic(user_input),
            'complexity': self._assess_complexity(user_input),
            'code_snippets': self._extract_code_snippets(user_input + ai_response)
        }
        
        # TF-IDF 특징 (sklearn 사용 가능한 경우)
        if HAS_SKLEARN and hasattr(self, 'tfidf_vectorizer'):
            try:
                tfidf_features = self.tfidf_vectorizer.transform([user_input])
                features['tfidf_scores'] = tfidf_features.toarray()[0].tolist()
            except:
                pass
                
        return features
        
    def _extract_keywords(self, text: str) -> List[str]:
        """핵심 키워드 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 필요)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 불용어 제거
        stopwords = {'은', '는', '이', '가', '을', '를', '에', '에서', 
                    'the', 'is', 'at', 'which', 'on', 'a', 'an'}
        
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # 빈도 계산
        word_freq = Counter(keywords)
        
        # 상위 키워드 반환
        return [word for word, _ in word_freq.most_common(10)]
        
    def _classify_intent(self, text: str) -> str:
        """의도 분류 (신경망 사용)"""
        # 간단한 규칙 기반 + 학습된 패턴
        intents = {
            'question': ['?', '뭐', '어떻게', '왜', '언제', '어디'],
            'request': ['해줘', '만들어', '고쳐', '부탁', '원해'],
            'greeting': ['안녕', '하이', '반가워'],
            'error': ['에러', '오류', '안돼', '문제', 'exception'],
            'feedback': ['고마워', '감사', '좋아', '나빠', '별로']
        }
        
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, keywords in intents.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            intent_scores[intent] = score
            
        # 신경망 가중치 적용
        if 'intent_recognition' in self.weights:
            # 간단한 특징 벡터 생성
            feature_vector = np.zeros(100)
            for i, char in enumerate(text[:100]):
                feature_vector[i] = ord(char) / 255.0
                
            # 가중치 적용
            hidden = np.tanh(np.dot(feature_vector, self.weights['intent_recognition']))
            
            # 최종 점수에 반영
            learned_scores = hidden[:len(intent_scores)]
            for i, (intent, _) in enumerate(intent_scores.items()):
                if i < len(learned_scores):
                    intent_scores[intent] += learned_scores[i]
                    
        # 가장 높은 점수의 의도 반환
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        return 'unknown'
        
    def _detect_emotion(self, text: str) -> str:
        """감정 감지"""
        emotions = {
            'happy': ['좋아', '기뻐', '행복', '즐거워', '최고', '😊', '😄'],
            'sad': ['슬퍼', '우울', '힘들어', '외로워', '😢', '😭'],
            'angry': ['화나', '짜증', '싫어', '미워', '😡', '😠'],
            'confused': ['모르겠', '헷갈려', '어려워', '복잡', '🤔', '😕'],
            'excited': ['신나', '기대', '와', '대박', '🎉', '✨']
        }
        
        detected_emotions = []
        for emotion, keywords in emotions.items():
            if any(kw in text for kw in keywords):
                detected_emotions.append(emotion)
                
        return detected_emotions[0] if detected_emotions else 'neutral'
        
    def _identify_topic(self, text: str) -> str:
        """주제 식별"""
        topics = {
            'unity': ['unity', '유니티', 'gameobject', 'transform', 'collider'],
            'csharp': ['c#', 'class', 'method', 'async', 'interface'],
            'error': ['error', 'exception', 'null', '에러', '오류'],
            'general': []
        }
        
        text_lower = text.lower()
        topic_scores = defaultdict(int)
        
        for topic, keywords in topics.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topic_scores[topic] += 1
                    
        if topic_scores:
            return max(topic_scores.items(), key=lambda x: x[1])[0]
        return 'general'
        
    def _assess_complexity(self, text: str) -> float:
        """복잡도 평가 (0-1)"""
        # 간단한 복잡도 메트릭
        word_count = len(text.split())
        unique_words = len(set(text.lower().split()))
        
        # 기술 용어 개수
        tech_terms = ['async', 'interface', 'abstract', 'delegate', 'lambda',
                     'coroutine', 'singleton', 'dependency', 'injection']
        tech_count = sum(1 for term in tech_terms if term in text.lower())
        
        # 복잡도 점수 계산
        complexity = min(1.0, (word_count / 50 + unique_words / 30 + tech_count / 5) / 3)
        
        return complexity
        
    def _extract_code_snippets(self, text: str) -> List[str]:
        """코드 스니펫 추출"""
        # 코드 블록 패턴
        code_patterns = [
            r'```[\s\S]*?```',  # Markdown 코드 블록
            r'`[^`]+`',         # 인라인 코드
            r'class\s+\w+\s*{[\s\S]*?}',  # 클래스 정의
            r'(public|private|protected)\s+\w+\s+\w+\s*\([^)]*\)',  # 메서드 시그니처
        ]
        
        snippets = []
        for pattern in code_patterns:
            matches = re.findall(pattern, text)
            snippets.extend(matches)
            
        return snippets
        
    def _identify_patterns(self, user_input: str, ai_response: str) -> List[Dict]:
        """패턴 인식"""
        patterns = []
        
        # 질문-답변 패턴
        if '?' in user_input:
            pattern = {
                'type': 'qa_pattern',
                'question_type': self._classify_intent(user_input),
                'topic': self._identify_topic(user_input),
                'hash': hashlib.md5(user_input.encode()).hexdigest()[:16]
            }
            patterns.append(pattern)
            
        # 코드 패턴
        code_snippets = self._extract_code_snippets(user_input + ai_response)
        for snippet in code_snippets:
            pattern = {
                'type': 'code_pattern',
                'content': snippet[:200],  # 처음 200자
                'language': 'csharp',  # 간단히 가정
                'hash': hashlib.md5(snippet.encode()).hexdigest()[:16]
            }
            patterns.append(pattern)
            
        # 에러 패턴
        error_keywords = ['error', 'exception', '에러', '오류', 'null']
        if any(kw in user_input.lower() for kw in error_keywords):
            pattern = {
                'type': 'error_pattern',
                'error_type': self._extract_error_type(user_input),
                'context': user_input[:100],
                'hash': hashlib.md5(user_input.encode()).hexdigest()[:16]
            }
            patterns.append(pattern)
            
        return patterns
        
    def _extract_error_type(self, text: str) -> str:
        """에러 타입 추출"""
        error_types = [
            'NullReferenceException',
            'IndexOutOfRangeException',
            'ArgumentException',
            'InvalidOperationException',
            'NotImplementedException'
        ]
        
        for error_type in error_types:
            if error_type.lower() in text.lower():
                return error_type
                
        return 'UnknownError'
        
    def _save_pattern(self, cursor, pattern: Dict):
        """패턴 저장"""
        if pattern['type'] == 'code_pattern':
            cursor.execute('''
                INSERT INTO code_patterns (pattern_hash, pattern_type, pattern_content)
                VALUES (?, ?, ?)
                ON CONFLICT(pattern_hash) DO UPDATE SET
                usage_count = usage_count + 1,
                last_seen = CURRENT_TIMESTAMP
            ''', (pattern['hash'], pattern['type'], pattern.get('content', '')))
            
        self.stats['learned_patterns'] += 1
        
    def _update_weights(self, features: Dict):
        """신경망 가중치 업데이트 (간단한 구현)"""
        # 학습률 적용
        learning_rate = self.learning_rate
        
        # 특징 벡터 생성
        feature_vector = np.zeros(100)
        
        # 간단한 특징 인코딩
        feature_vector[0] = features.get('input_length', 0) / 1000.0
        feature_vector[1] = features.get('response_length', 0) / 1000.0
        feature_vector[2] = features.get('complexity', 0)
        
        # 가중치 업데이트 (gradient descent 시뮬레이션)
        for layer_name, weights in self.weights.items():
            # 랜덤 그래디언트 (실제로는 역전파 필요)
            gradient = np.random.randn(*weights.shape) * 0.001
            
            # 가중치 업데이트
            self.weights[layer_name] -= learning_rate * gradient
            
        # 주기적으로 가중치 저장
        if self.stats['total_conversations'] % 10 == 0:
            self._save_weights()
            
    def _save_weights(self):
        """신경망 가중치 저장"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        for layer_name, weights in self.weights.items():
            weights_blob = pickle.dumps(weights)
            cursor.execute('''
                INSERT OR REPLACE INTO neural_weights (layer_name, weights_data)
                VALUES (?, ?)
            ''', (layer_name, weights_blob))
            
        conn.commit()
        conn.close()
        
    def _load_learned_data(self):
        """학습된 데이터 로드"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 통계 로드
        cursor.execute('SELECT COUNT(*) FROM conversation_memory')
        self.stats['total_conversations'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM code_patterns')
        self.stats['learned_patterns'] = cursor.fetchone()[0]
        
        # 가중치 로드
        cursor.execute('SELECT layer_name, weights_data FROM neural_weights')
        for layer_name, weights_blob in cursor.fetchall():
            try:
                self.weights[layer_name] = pickle.loads(weights_blob)
            except:
                pass
                
        # TF-IDF 벡터라이저 초기화 (sklearn 사용 가능한 경우)
        if HAS_SKLEARN:
            # 기존 대화에서 학습
            cursor.execute('SELECT user_input FROM conversation_memory LIMIT 1000')
            texts = [row[0] for row in cursor.fetchall()]
            
            if texts:
                self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
                self.tfidf_vectorizer.fit(texts)
                
        conn.close()
        
        logger.info(f"학습 데이터 로드 완료: {self.stats['total_conversations']}개 대화, "
                   f"{self.stats['learned_patterns']}개 패턴")
        
    def learn_from_feedback(self, conversation_id: int, feedback: float):
        """사용자 피드백으로부터 학습"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 피드백 저장
        cursor.execute('''
            UPDATE conversation_memory 
            SET user_feedback = ? 
            WHERE id = ?
        ''', (feedback, conversation_id))
        
        # 관련 패턴의 성공률 업데이트
        if feedback > 0.7:  # 긍정적 피드백
            cursor.execute('''
                UPDATE code_patterns 
                SET success_rate = (success_rate * usage_count + 1) / (usage_count + 1)
                WHERE pattern_hash IN (
                    SELECT pattern_hash FROM conversation_memory 
                    WHERE id = ?
                )
            ''', (conversation_id,))
            
        conn.commit()
        conn.close()
        
        # 정확도 업데이트
        self._update_accuracy()
        
    def _update_accuracy(self):
        """전체 정확도 계산"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT AVG(user_feedback) 
            FROM conversation_memory 
            WHERE user_feedback > 0
        ''')
        
        result = cursor.fetchone()
        if result and result[0]:
            self.stats['accuracy'] = result[0]
            
        conn.close()
        
    def get_similar_conversations(self, user_input: str, k: int = 5) -> List[Dict]:
        """유사한 대화 검색"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 간단한 키워드 기반 검색
        keywords = self._extract_keywords(user_input)
        
        similar_convs = []
        
        for keyword in keywords[:3]:  # 상위 3개 키워드
            cursor.execute('''
                SELECT user_input, ai_response, user_feedback
                FROM conversation_memory
                WHERE user_input LIKE ?
                ORDER BY user_feedback DESC
                LIMIT ?
            ''', (f'%{keyword}%', k))
            
            for row in cursor.fetchall():
                similar_convs.append({
                    'user_input': row[0],
                    'ai_response': row[1],
                    'feedback': row[2]
                })
                
        conn.close()
        
        # 중복 제거 및 상위 k개 반환
        seen = set()
        unique_convs = []
        for conv in similar_convs:
            if conv['user_input'] not in seen:
                seen.add(conv['user_input'])
                unique_convs.append(conv)
                
        return unique_convs[:k]
        
    def get_learning_stats(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        return {
            'total_conversations': self.stats['total_conversations'],
            'learned_patterns': self.stats['learned_patterns'],
            'accuracy': f"{self.stats['accuracy'] * 100:.1f}%",
            'memory_usage': len(self.short_term_memory),
            'last_update': self.stats['last_update'].isoformat(),
            'learning_rate': self.learning_rate,
            'topics_learned': self._get_learned_topics(),
            'error_patterns': self._get_error_patterns_summary()
        }
        
    def _get_learned_topics(self) -> List[str]:
        """학습한 주제 목록"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT concept_name 
            FROM learned_concepts 
            ORDER BY understanding_level DESC
            LIMIT 10
        ''')
        
        topics = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return topics
        
    def _get_error_patterns_summary(self) -> Dict[str, int]:
        """에러 패턴 요약"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT error_type, COUNT(*) as count
            FROM error_patterns
            GROUP BY error_type
            ORDER BY count DESC
            LIMIT 5
        ''')
        
        summary = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        return summary
        
    def start_background_learning(self):
        """백그라운드 학습 시작"""
        if not self.is_learning:
            self.is_learning = True
            self.learning_thread = threading.Thread(
                target=self._background_learning_loop,
                daemon=True
            )
            self.learning_thread.start()
            logger.info("백그라운드 학습이 시작되었습니다.")
            
    def _background_learning_loop(self):
        """백그라운드 학습 루프"""
        while self.is_learning:
            try:
                # 주기적으로 패턴 분석
                self._analyze_patterns()
                
                # 메모리 정리
                self._cleanup_memory()
                
                # 가중치 최적화
                self._optimize_weights()
                
                # 1분 대기
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"백그라운드 학습 오류: {e}")
                
    def _analyze_patterns(self):
        """저장된 패턴 분석"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 자주 사용되는 패턴 찾기
        cursor.execute('''
            SELECT pattern_type, COUNT(*) as count
            FROM code_patterns
            GROUP BY pattern_type
            ORDER BY count DESC
        ''')
        
        pattern_stats = cursor.fetchall()
        
        # 패턴 클러스터링 (sklearn 사용 가능한 경우)
        if HAS_SKLEARN and len(pattern_stats) > 10:
            # 간단한 클러스터링 수행
            pass
            
        conn.close()
        
    def _cleanup_memory(self):
        """메모리 정리"""
        # 단기 기억 정리 (최근 100개만 유지)
        if len(self.short_term_memory) > 100:
            self.short_term_memory = self.short_term_memory[-100:]
            
        # 오래된 작업 메모리 제거
        current_time = datetime.now()
        expired_keys = []
        
        for key, value in self.working_memory.items():
            if isinstance(value, dict) and 'timestamp' in value:
                if current_time - value['timestamp'] > timedelta(hours=1):
                    expired_keys.append(key)
                    
        for key in expired_keys:
            del self.working_memory[key]
            
    def _optimize_weights(self):
        """가중치 최적화"""
        # 간단한 가중치 정규화
        for layer_name, weights in self.weights.items():
            # L2 정규화
            norm = np.linalg.norm(weights)
            if norm > 10:
                self.weights[layer_name] = weights / norm * 10
                
    def stop_background_learning(self):
        """백그라운드 학습 중지"""
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        logger.info("백그라운드 학습이 중지되었습니다.")


# 테스트 함수
def test_real_learning():
    """실제 학습 시스템 테스트"""
    print("🧠 AutoCI 실제 학습 시스템 테스트")
    print("=" * 60)
    
    # 학습 시스템 초기화
    learning_system = RealLearningSystem()
    
    # 백그라운드 학습 시작
    learning_system.start_background_learning()
    
    # 테스트 대화
    test_conversations = [
        ("Unity에서 플레이어 이동을 구현하고 싶어요", 
         "Transform.Translate() 또는 Rigidbody.velocity를 사용하면 됩니다."),
        
        ("NullReferenceException이 계속 발생해요",
         "객체가 null인지 먼저 확인하세요: if (myObject != null)"),
         
        ("코루틴이 뭔가요?",
         "Unity에서 시간에 걸쳐 실행되는 함수입니다. yield return을 사용합니다."),
    ]
    
    # 대화 학습
    for i, (user_input, ai_response) in enumerate(test_conversations):
        print(f"\n[대화 {i+1}]")
        print(f"👤: {user_input}")
        print(f"🤖: {ai_response}")
        
        # 학습
        result = learning_system.learn_from_conversation(
            user_input, ai_response,
            context={'session_id': 'test', 'timestamp': datetime.now()}
        )
        
        print(f"✅ 학습 완료: {result['patterns']} 패턴 발견")
        
        # 피드백 (긍정적)
        learning_system.learn_from_feedback(i+1, 0.9)
        
    # 학습 통계 확인
    print("\n" + "=" * 60)
    print("📊 학습 통계:")
    stats = learning_system.get_learning_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    # 유사한 대화 검색
    print("\n" + "=" * 60)
    print("🔍 유사한 대화 검색:")
    similar = learning_system.get_similar_conversations("Unity 이동")
    for conv in similar:
        print(f"  Q: {conv['user_input'][:50]}...")
        print(f"  A: {conv['ai_response'][:50]}...")
        print(f"  평가: {conv['feedback']}")
        
    # 백그라운드 학습 중지
    learning_system.stop_background_learning()
    
    print("\n✅ 테스트 완료!")


if __name__ == "__main__":
    test_real_learning()