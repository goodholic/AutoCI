#!/usr/bin/env python3
"""
AutoCI 상용화 수준 AI 대화 엔진
ChatGPT 수준의 자연스럽고 전문적인 대화 능력
"""

import os
import sys
import json
import sqlite3
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import re
import threading
import time
import hashlib
from collections import defaultdict, deque
import pickle

# 고급 NLP 라이브러리
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  Transformers 라이브러리가 없습니다. 기본 모드로 작동합니다.")

logger = logging.getLogger(__name__)


class CommercialDialogueEngine:
    """상용화 수준의 AI 대화 엔진"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.base_path = Path(__file__).parent
        self.data_path = self.base_path / "commercial_data"
        self.data_path.mkdir(exist_ok=True)
        
        # 대화 품질 파라미터
        self.quality_threshold = 0.85  # 상용화 품질 기준
        self.response_creativity = 0.7  # 창의성 수준
        self.context_window = 10  # 문맥 창 크기
        
        # 대화 스타일 매트릭스
        self.dialogue_styles = {
            'professional': {'formality': 0.9, 'empathy': 0.6, 'clarity': 0.95},
            'friendly': {'formality': 0.4, 'empathy': 0.9, 'clarity': 0.8},
            'technical': {'formality': 0.8, 'empathy': 0.3, 'clarity': 1.0},
            'educational': {'formality': 0.7, 'empathy': 0.7, 'clarity': 0.9}
        }
        
        # 고급 대화 컴포넌트
        self.components = {
            'intent_analyzer': IntentAnalyzer(),
            'context_manager': ContextManager(),
            'response_generator': ResponseGenerator(),
            'quality_checker': QualityChecker(),
            'emotion_engine': EmotionEngine()
        }
        
        # 대화 메모리 (단기/장기)
        self.short_term_memory = deque(maxlen=50)
        self.long_term_memory = ConversationMemory()
        
        # 학습 데이터
        self.conversation_patterns = defaultdict(list)
        self.successful_responses = []
        
        # 초기화
        self._initialize_models()
        self._load_conversation_data()
        
    def _initialize_models(self):
        """AI 모델 초기화"""
        if HAS_TRANSFORMERS:
            try:
                # 한국어 BERT 모델 사용
                self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
                self.model = AutoModel.from_pretrained("klue/bert-base")
                logger.info("한국어 BERT 모델 로드 완료")
            except:
                HAS_TRANSFORMERS = False
                logger.warning("BERT 모델 로드 실패, 기본 모드 사용")
        
        # 커스텀 신경망
        self.dialogue_network = DialogueNeuralNetwork()
        
    def process_dialogue(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """상용화 수준의 대화 처리"""
        start_time = time.time()
        
        # 1. 입력 전처리 및 분석
        processed_input = self._preprocess_input(user_input)
        
        # 2. 의도 분석 (다중 의도 지원)
        intents = self.components['intent_analyzer'].analyze(processed_input)
        
        # 3. 문맥 관리
        context_data = self.components['context_manager'].update(
            processed_input, 
            self.short_term_memory,
            context
        )
        
        # 4. 감정 분석
        emotion_data = self.components['emotion_engine'].analyze(
            processed_input,
            context_data
        )
        
        # 5. 응답 생성 (다중 후보)
        response_candidates = self._generate_responses(
            processed_input,
            intents,
            context_data,
            emotion_data
        )
        
        # 6. 품질 검증 및 최적 응답 선택
        best_response = self._select_best_response(
            response_candidates,
            user_input,
            context_data
        )
        
        # 7. 대화 기록 및 학습
        self._record_conversation(
            user_input,
            best_response,
            intents,
            emotion_data,
            context_data
        )
        
        # 응답 시간 측정
        response_time = time.time() - start_time
        
        return {
            'response': best_response['text'],
            'confidence': best_response['confidence'],
            'intents': intents,
            'emotion': emotion_data,
            'context': context_data,
            'response_time': response_time,
            'quality_score': best_response['quality_score']
        }
    
    def _preprocess_input(self, text: str) -> Dict[str, Any]:
        """입력 전처리"""
        # 기본 정제
        cleaned_text = text.strip()
        
        # 문장 분리
        sentences = re.split(r'[.!?]+', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 토큰화
        tokens = []
        if HAS_TRANSFORMERS and hasattr(self, 'tokenizer'):
            encoded = self.tokenizer(cleaned_text, return_tensors='pt')
            tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        else:
            # 간단한 토큰화
            tokens = cleaned_text.split()
        
        # 언어 특성 분석
        features = {
            'length': len(cleaned_text),
            'sentence_count': len(sentences),
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            'question_marks': cleaned_text.count('?'),
            'exclamation_marks': cleaned_text.count('!'),
            'formal_markers': sum(1 for marker in ['습니다', '합니다', '세요'] if marker in cleaned_text),
            'informal_markers': sum(1 for marker in ['야', '어', '아'] if marker in cleaned_text)
        }
        
        return {
            'original': text,
            'cleaned': cleaned_text,
            'sentences': sentences,
            'tokens': tokens,
            'features': features
        }
    
    def _generate_responses(self, processed_input: Dict, intents: List[Dict],
                          context: Dict, emotion: Dict) -> List[Dict]:
        """다중 응답 후보 생성"""
        candidates = []
        
        # 1. 템플릿 기반 응답
        template_responses = self._generate_template_responses(intents)
        candidates.extend(template_responses)
        
        # 2. 학습된 패턴 기반 응답
        pattern_responses = self._generate_pattern_responses(
            processed_input['cleaned'],
            context
        )
        candidates.extend(pattern_responses)
        
        # 3. 생성 모델 기반 응답 (if available)
        if self.dialogue_network:
            generated_responses = self._generate_neural_responses(
                processed_input,
                intents,
                context,
                emotion
            )
            candidates.extend(generated_responses)
        
        # 4. 컨텍스트 인식 응답
        context_responses = self._generate_context_aware_responses(
            processed_input,
            context,
            emotion
        )
        candidates.extend(context_responses)
        
        # 각 후보에 점수 부여
        for candidate in candidates:
            candidate['quality_score'] = self._evaluate_response_quality(
                candidate['text'],
                processed_input['cleaned'],
                context
            )
        
        return candidates
    
    def _generate_template_responses(self, intents: List[Dict]) -> List[Dict]:
        """템플릿 기반 응답 생성"""
        templates = {
            'greeting': [
                "안녕하세요! 오늘 어떤 도움이 필요하신가요?",
                "반갑습니다! 무엇을 도와드릴까요?",
                "안녕하세요! C#과 Unity 관련 질문이 있으신가요?"
            ],
            'question': [
                "좋은 질문이네요! {topic}에 대해 설명드리겠습니다.",
                "{topic}에 대해 궁금하신 것 같네요. 자세히 알려드리겠습니다.",
                "네, {topic}에 대해 말씀드리자면..."
            ],
            'error': [
                "{error_type} 에러는 주로 {cause} 때문에 발생합니다. 해결 방법은...",
                "이런 에러를 해결하려면 먼저 {check_point}를 확인해보세요.",
                "{error_type}가 발생했군요. 다음 단계를 따라해보세요:"
            ],
            'request': [
                "네, {task}를 도와드리겠습니다. 먼저...",
                "{task}를 위한 코드를 작성해드리겠습니다.",
                "알겠습니다. {task}를 구현하는 방법은 다음과 같습니다:"
            ]
        }
        
        responses = []
        for intent in intents[:2]:  # 상위 2개 의도
            intent_type = intent['type']
            if intent_type in templates:
                for template in templates[intent_type][:2]:
                    response_text = template.format(**intent.get('params', {}))
                    responses.append({
                        'text': response_text,
                        'type': 'template',
                        'intent': intent_type,
                        'confidence': intent['confidence'] * 0.8
                    })
        
        return responses
    
    def _generate_pattern_responses(self, user_input: str, context: Dict) -> List[Dict]:
        """학습된 패턴 기반 응답"""
        responses = []
        
        # 유사한 대화 패턴 검색
        similar_patterns = self._find_similar_patterns(user_input)
        
        for pattern in similar_patterns[:3]:
            if pattern['success_rate'] > 0.7:
                responses.append({
                    'text': pattern['response'],
                    'type': 'pattern',
                    'confidence': pattern['similarity'] * pattern['success_rate'],
                    'pattern_id': pattern['id']
                })
        
        return responses
    
    def _generate_neural_responses(self, processed_input: Dict, intents: List[Dict],
                                  context: Dict, emotion: Dict) -> List[Dict]:
        """신경망 기반 응답 생성"""
        responses = []
        
        try:
            # 입력 인코딩
            input_vector = self._encode_input(processed_input, intents, context, emotion)
            
            # 신경망 통과
            with torch.no_grad():
                output = self.dialogue_network(input_vector)
                
            # 디코딩
            response_text = self._decode_output(output, context)
            
            responses.append({
                'text': response_text,
                'type': 'neural',
                'confidence': float(torch.sigmoid(output.max()).item())
            })
            
        except Exception as e:
            logger.error(f"신경망 응답 생성 오류: {e}")
        
        return responses
    
    def _generate_context_aware_responses(self, processed_input: Dict,
                                        context: Dict, emotion: Dict) -> List[Dict]:
        """컨텍스트 인식 응답 생성"""
        responses = []
        
        # 이전 대화 분석
        recent_topics = context.get('recent_topics', [])
        conversation_flow = context.get('flow', 'normal')
        
        # 대화 흐름에 맞는 응답
        if conversation_flow == 'problem_solving':
            responses.append({
                'text': "이전에 말씀하신 문제를 해결하기 위해 다음 단계를 진행해볼까요?",
                'type': 'contextual',
                'confidence': 0.7
            })
        elif conversation_flow == 'learning':
            responses.append({
                'text': "좋아요! 이제 더 깊이 있는 내용을 알아볼까요?",
                'type': 'contextual',
                'confidence': 0.7
            })
        
        # 감정 기반 응답
        if emotion.get('type') == 'frustrated':
            responses.append({
                'text': "어려우셨군요. 천천히 하나씩 해결해보겠습니다. 먼저...",
                'type': 'empathetic',
                'confidence': 0.8
            })
        
        return responses
    
    def _select_best_response(self, candidates: List[Dict], 
                            user_input: str, context: Dict) -> Dict:
        """최적 응답 선택"""
        if not candidates:
            return {
                'text': "죄송합니다. 이해하지 못했습니다. 다시 설명해주시겠어요?",
                'confidence': 0.3,
                'quality_score': 0.5
            }
        
        # 품질 점수로 정렬
        candidates.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # 상위 후보 중 다양성 고려하여 선택
        best_candidate = candidates[0]
        
        # 품질 임계값 확인
        if best_candidate['quality_score'] < self.quality_threshold:
            # 품질이 낮으면 안전한 응답으로 대체
            best_candidate = self._generate_safe_response(user_input, context)
        
        return best_candidate
    
    def _evaluate_response_quality(self, response: str, user_input: str, 
                                 context: Dict) -> float:
        """응답 품질 평가"""
        scores = []
        
        # 1. 관련성 점수
        relevance = self._calculate_relevance(response, user_input)
        scores.append(relevance)
        
        # 2. 일관성 점수
        consistency = self._calculate_consistency(response, context)
        scores.append(consistency)
        
        # 3. 완성도 점수
        completeness = self._calculate_completeness(response)
        scores.append(completeness)
        
        # 4. 자연스러움 점수
        naturalness = self._calculate_naturalness(response)
        scores.append(naturalness)
        
        # 5. 유용성 점수
        usefulness = self._calculate_usefulness(response, user_input)
        scores.append(usefulness)
        
        # 가중 평균
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]
        quality_score = sum(s * w for s, w in zip(scores, weights))
        
        return quality_score
    
    def _calculate_relevance(self, response: str, user_input: str) -> float:
        """관련성 계산"""
        # 키워드 중복도
        input_keywords = set(self._extract_keywords(user_input))
        response_keywords = set(self._extract_keywords(response))
        
        if not input_keywords:
            return 0.5
        
        overlap = len(input_keywords & response_keywords)
        relevance = overlap / len(input_keywords)
        
        return min(1.0, relevance * 1.5)  # 부스팅
    
    def _calculate_consistency(self, response: str, context: Dict) -> float:
        """일관성 계산"""
        # 이전 응답들과의 일관성 체크
        previous_responses = context.get('previous_responses', [])
        
        if not previous_responses:
            return 0.8
        
        # 모순 체크
        contradictions = 0
        for prev in previous_responses[-3:]:
            if self._has_contradiction(response, prev):
                contradictions += 1
        
        consistency = 1.0 - (contradictions * 0.3)
        return max(0.0, consistency)
    
    def _calculate_completeness(self, response: str) -> float:
        """완성도 계산"""
        # 문장 구조 완성도
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = 0
        
        for sentence in sentences:
            if sentence.strip() and len(sentence.split()) >= 3:
                valid_sentences += 1
        
        if not sentences:
            return 0.0
        
        return valid_sentences / len(sentences)
    
    def _calculate_naturalness(self, response: str) -> float:
        """자연스러움 계산"""
        # 간단한 규칙 기반 평가
        naturalness = 1.0
        
        # 반복 체크
        words = response.split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            naturalness *= unique_ratio
        
        # 문장 길이 분포
        sentences = re.split(r'[.!?]+', response)
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if lengths:
            avg_length = np.mean(lengths)
            if avg_length < 5 or avg_length > 20:
                naturalness *= 0.8
        
        return naturalness
    
    def _calculate_usefulness(self, response: str, user_input: str) -> float:
        """유용성 계산"""
        usefulness = 0.5
        
        # 구체적인 정보 포함 여부
        if any(pattern in response for pattern in ['다음과 같습니다', '방법은', '해결하려면']):
            usefulness += 0.2
        
        # 코드 포함 여부 (기술적 질문인 경우)
        if '코드' in user_input or 'code' in user_input.lower():
            if '```' in response or 'class' in response or 'public' in response:
                usefulness += 0.3
        
        # 단계별 설명 여부
        if any(marker in response for marker in ['첫째', '둘째', '1.', '2.', '먼저']):
            usefulness += 0.2
        
        return min(1.0, usefulness)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        # 불용어 제거
        stopwords = {'은', '는', '이', '가', '을', '를', '에', '에서', '와', '과',
                    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or'}
        
        words = re.findall(r'\w+', text.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 1]
        
        return keywords
    
    def _has_contradiction(self, text1: str, text2: str) -> bool:
        """모순 검사"""
        # 간단한 모순 패턴
        contradictions = [
            ('할 수 있습니다', '할 수 없습니다'),
            ('가능합니다', '불가능합니다'),
            ('맞습니다', '틀렸습니다'),
            ('예', '아니오')
        ]
        
        for pos, neg in contradictions:
            if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                return True
        
        return False
    
    def _find_similar_patterns(self, user_input: str) -> List[Dict]:
        """유사 패턴 검색"""
        similar_patterns = []
        
        # 데이터베이스에서 검색
        conn = sqlite3.connect(str(self.data_path / "dialogue_patterns.db"))
        cursor = conn.cursor()
        
        # 키워드 기반 검색
        keywords = self._extract_keywords(user_input)
        
        for keyword in keywords[:5]:
            cursor.execute('''
                SELECT id, input_pattern, response, success_rate
                FROM dialogue_patterns
                WHERE input_pattern LIKE ?
                ORDER BY success_rate DESC
                LIMIT 5
            ''', (f'%{keyword}%',))
            
            for row in cursor.fetchall():
                similarity = self._calculate_similarity(user_input, row[1])
                similar_patterns.append({
                    'id': row[0],
                    'input': row[1],
                    'response': row[2],
                    'success_rate': row[3],
                    'similarity': similarity
                })
        
        conn.close()
        
        # 유사도 순으로 정렬
        similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_patterns[:5]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산"""
        # 간단한 자카드 유사도
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _encode_input(self, processed_input: Dict, intents: List[Dict],
                     context: Dict, emotion: Dict) -> torch.Tensor:
        """입력 인코딩"""
        # 특징 벡터 생성
        features = []
        
        # 텍스트 특징
        features.extend([
            processed_input['features']['length'] / 200,
            processed_input['features']['sentence_count'] / 5,
            processed_input['features']['question_marks'] / 3,
        ])
        
        # 의도 특징
        intent_vector = [0] * 10  # 10개 의도 카테고리
        for intent in intents[:3]:
            intent_idx = hash(intent['type']) % 10
            intent_vector[intent_idx] = intent['confidence']
        features.extend(intent_vector)
        
        # 감정 특징
        emotion_vector = [0] * 5  # 5개 감정 카테고리
        emotion_idx = hash(emotion.get('type', 'neutral')) % 5
        emotion_vector[emotion_idx] = emotion.get('intensity', 0.5)
        features.extend(emotion_vector)
        
        # 텐서 변환
        return torch.tensor(features, dtype=torch.float32)
    
    def _decode_output(self, output: torch.Tensor, context: Dict) -> str:
        """출력 디코딩"""
        # 간단한 템플릿 기반 디코딩
        output_values = output.squeeze().tolist()
        
        # 응답 타입 결정
        response_type_idx = np.argmax(output_values[:5])
        response_types = ['informative', 'helpful', 'clarifying', 'encouraging', 'technical']
        response_type = response_types[response_type_idx]
        
        # 응답 템플릿
        templates = {
            'informative': "제가 알기로는 {topic}에 대해 {detail} 입니다.",
            'helpful': "도움이 되도록 {action}를 해드리겠습니다.",
            'clarifying': "혹시 {clarification}를 원하시는 건가요?",
            'encouraging': "잘하고 계십니다! {suggestion}를 해보시면 좋을 것 같아요.",
            'technical': "기술적으로 설명드리면 {technical_detail} 입니다."
        }
        
        # 컨텍스트 기반 파라미터 채우기
        template = templates.get(response_type, "네, 알겠습니다.")
        params = self._extract_template_params(template, context)
        
        return template.format(**params)
    
    def _extract_template_params(self, template: str, context: Dict) -> Dict[str, str]:
        """템플릿 파라미터 추출"""
        params = {}
        
        # 템플릿에서 필요한 파라미터 찾기
        placeholders = re.findall(r'{(\w+)}', template)
        
        for placeholder in placeholders:
            if placeholder == 'topic':
                params[placeholder] = context.get('current_topic', 'C# 프로그래밍')
            elif placeholder == 'detail':
                params[placeholder] = "중요한 개념"
            elif placeholder == 'action':
                params[placeholder] = "코드 예제를 작성"
            elif placeholder == 'clarification':
                params[placeholder] = "이 부분"
            elif placeholder == 'suggestion':
                params[placeholder] = "단계별로 접근"
            elif placeholder == 'technical_detail':
                params[placeholder] = "다음과 같은 원리로 작동합니다"
        
        return params
    
    def _generate_safe_response(self, user_input: str, context: Dict) -> Dict:
        """안전한 기본 응답 생성"""
        safe_responses = [
            "흥미로운 질문이네요. 조금 더 구체적으로 설명해주시면 정확한 답변을 드릴 수 있을 것 같습니다.",
            "네, 이해했습니다. 이 부분에 대해 더 자세히 알아보고 답변드리겠습니다.",
            "좋은 포인트를 짚어주셨네요. 제가 이해한 바로는..."
        ]
        
        import random
        return {
            'text': random.choice(safe_responses),
            'confidence': 0.6,
            'quality_score': 0.7
        }
    
    def _record_conversation(self, user_input: str, response: Dict,
                           intents: List[Dict], emotion: Dict, context: Dict):
        """대화 기록 및 학습"""
        # 단기 메모리에 추가
        self.short_term_memory.append({
            'timestamp': datetime.now(),
            'user_input': user_input,
            'response': response['text'],
            'intents': intents,
            'emotion': emotion,
            'quality_score': response['quality_score']
        })
        
        # 장기 메모리에 저장
        self.long_term_memory.store({
            'user_input': user_input,
            'response': response,
            'context': context,
            'metadata': {
                'intents': intents,
                'emotion': emotion,
                'timestamp': datetime.now()
            }
        })
        
        # 패턴 학습
        if response['quality_score'] > 0.8:
            self._learn_pattern(user_input, response['text'], intents)
    
    def _learn_pattern(self, user_input: str, response: str, intents: List[Dict]):
        """성공적인 패턴 학습"""
        pattern_key = intents[0]['type'] if intents else 'general'
        
        self.conversation_patterns[pattern_key].append({
            'input': user_input,
            'response': response,
            'timestamp': datetime.now()
        })
        
        # 주기적으로 데이터베이스에 저장
        if len(self.conversation_patterns[pattern_key]) % 10 == 0:
            self._save_patterns_to_db()
    
    def _save_patterns_to_db(self):
        """패턴을 데이터베이스에 저장"""
        conn = sqlite3.connect(str(self.data_path / "dialogue_patterns.db"))
        cursor = conn.cursor()
        
        # 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dialogue_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_pattern TEXT,
                response TEXT,
                pattern_type TEXT,
                success_rate REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 패턴 저장
        for pattern_type, patterns in self.conversation_patterns.items():
            for pattern in patterns[-5:]:  # 최근 5개만
                cursor.execute('''
                    INSERT INTO dialogue_patterns 
                    (input_pattern, response, pattern_type)
                    VALUES (?, ?, ?)
                ''', (pattern['input'], pattern['response'], pattern_type))
        
        conn.commit()
        conn.close()
    
    def _load_conversation_data(self):
        """대화 데이터 로드"""
        # 기존 패턴 로드
        db_path = self.data_path / "dialogue_patterns.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT pattern_type, input_pattern, response, success_rate
                FROM dialogue_patterns
                WHERE success_rate > 0.7
                ORDER BY success_rate DESC
                LIMIT 100
            ''')
            
            for row in cursor.fetchall():
                pattern_type, input_pattern, response, success_rate = row
                self.successful_responses.append({
                    'type': pattern_type,
                    'input': input_pattern,
                    'response': response,
                    'success_rate': success_rate
                })
            
            conn.close()


class IntentAnalyzer:
    """의도 분석기"""
    
    def __init__(self):
        self.intent_patterns = {
            'greeting': ['안녕', '하이', 'hello', '반가워'],
            'question': ['?', '뭐', '어떻게', '왜', '언제', '어디', '무엇'],
            'request': ['해줘', '만들어', '보여줘', '알려줘', '설명해'],
            'error': ['에러', '오류', 'error', 'exception', '안돼', '안됨'],
            'feedback': ['고마워', '감사', '좋아', '나빠', '별로', '최고'],
            'learning': ['배우고', '공부', '알고싶', '궁금'],
            'technical': ['코드', 'code', '함수', 'class', 'method', 'async'],
            'clarification': ['무슨 말', '이해가', '다시', '설명'],
            'confirmation': ['맞아', '그래', '네', '예', '확인'],
            'completion': ['완료', '끝', '다했', '마침']
        }
    
    def analyze(self, processed_input: Dict) -> List[Dict]:
        """다중 의도 분석"""
        text = processed_input['cleaned'].lower()
        intents = []
        
        # 패턴 매칭
        for intent_type, patterns in self.intent_patterns.items():
            confidence = 0.0
            matches = 0
            
            for pattern in patterns:
                if pattern in text:
                    matches += 1
            
            if matches > 0:
                confidence = min(1.0, matches / len(patterns) * 2)
                intents.append({
                    'type': intent_type,
                    'confidence': confidence,
                    'params': self._extract_params(text, intent_type)
                })
        
        # 신뢰도 순으로 정렬
        intents.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 의도가 없으면 기본값
        if not intents:
            intents.append({
                'type': 'general',
                'confidence': 0.5,
                'params': {}
            })
        
        return intents
    
    def _extract_params(self, text: str, intent_type: str) -> Dict:
        """의도별 파라미터 추출"""
        params = {}
        
        if intent_type == 'error':
            # 에러 타입 추출
            error_types = ['NullReferenceException', 'IndexOutOfRange', 
                          'ArgumentException', 'InvalidOperation']
            for error in error_types:
                if error.lower() in text.lower():
                    params['error_type'] = error
                    break
        
        elif intent_type == 'question':
            # 주제 추출
            if 'unity' in text:
                params['topic'] = 'Unity'
            elif 'c#' in text or 'csharp' in text:
                params['topic'] = 'C#'
        
        elif intent_type == 'request':
            # 작업 타입 추출
            if '코드' in text:
                params['task'] = '코드 작성'
            elif '설명' in text:
                params['task'] = '설명'
        
        return params


class ContextManager:
    """문맥 관리자"""
    
    def __init__(self):
        self.context_window = 10
        self.topic_tracker = TopicTracker()
        
    def update(self, processed_input: Dict, memory: deque, 
               external_context: Dict = None) -> Dict:
        """문맥 업데이트"""
        context = {
            'conversation_length': len(memory),
            'current_topic': self.topic_tracker.get_current_topic(processed_input),
            'recent_topics': self.topic_tracker.get_recent_topics(),
            'flow': self._analyze_conversation_flow(memory),
            'user_state': self._analyze_user_state(memory),
            'previous_responses': [m['response'] for m in list(memory)[-3:]]
        }
        
        if external_context:
            context.update(external_context)
        
        return context
    
    def _analyze_conversation_flow(self, memory: deque) -> str:
        """대화 흐름 분석"""
        if len(memory) < 2:
            return 'starting'
        
        recent_intents = []
        for m in list(memory)[-5:]:
            if 'intents' in m and m['intents']:
                recent_intents.append(m['intents'][0]['type'])
        
        # 흐름 패턴 인식
        if recent_intents.count('error') >= 2:
            return 'problem_solving'
        elif recent_intents.count('question') >= 3:
            return 'learning'
        elif recent_intents.count('request') >= 2:
            return 'task_oriented'
        
        return 'normal'
    
    def _analyze_user_state(self, memory: deque) -> Dict:
        """사용자 상태 분석"""
        if not memory:
            return {'engagement': 'neutral', 'expertise': 'unknown'}
        
        recent_emotions = []
        for m in list(memory)[-5:]:
            if 'emotion' in m:
                recent_emotions.append(m['emotion'].get('type', 'neutral'))
        
        # 참여도 계산
        if recent_emotions.count('excited') > 1:
            engagement = 'high'
        elif recent_emotions.count('frustrated') > 1:
            engagement = 'low'
        else:
            engagement = 'medium'
        
        return {
            'engagement': engagement,
            'expertise': self._estimate_expertise(memory)
        }
    
    def _estimate_expertise(self, memory: deque) -> str:
        """전문성 수준 추정"""
        technical_terms = 0
        total_inputs = len(memory)
        
        if total_inputs == 0:
            return 'unknown'
        
        for m in memory:
            input_text = m.get('user_input', '').lower()
            if any(term in input_text for term in 
                   ['async', 'delegate', 'interface', 'abstract', 'generic']):
                technical_terms += 1
        
        ratio = technical_terms / total_inputs
        
        if ratio > 0.3:
            return 'expert'
        elif ratio > 0.1:
            return 'intermediate'
        else:
            return 'beginner'


class ResponseGenerator:
    """응답 생성기"""
    
    def __init__(self):
        self.style_adapter = StyleAdapter()
        
    def generate(self, intent: str, context: Dict, style: str = 'professional') -> str:
        """스타일에 맞는 응답 생성"""
        base_response = self._generate_base_response(intent, context)
        styled_response = self.style_adapter.apply_style(base_response, style)
        
        return styled_response
    
    def _generate_base_response(self, intent: str, context: Dict) -> str:
        """기본 응답 생성"""
        # 의도별 기본 응답
        responses = {
            'greeting': "안녕하세요! 무엇을 도와드릴까요?",
            'question': "좋은 질문입니다. 설명드리겠습니다.",
            'error': "에러 해결을 도와드리겠습니다.",
            'request': "네, 바로 도와드리겠습니다."
        }
        
        return responses.get(intent, "네, 알겠습니다.")


class QualityChecker:
    """품질 검증기"""
    
    def check(self, response: str, context: Dict) -> Dict[str, float]:
        """응답 품질 검증"""
        return {
            'grammar': self._check_grammar(response),
            'coherence': self._check_coherence(response, context),
            'completeness': self._check_completeness(response),
            'appropriateness': self._check_appropriateness(response, context)
        }
    
    def _check_grammar(self, response: str) -> float:
        """문법 검사"""
        # 간단한 규칙 기반 검사
        score = 1.0
        
        # 문장 종결 확인
        if not response.strip().endswith(('.', '!', '?', '요')):
            score -= 0.2
        
        # 맞춤법 패턴
        common_errors = [
            ('됬', '됐'),
            ('햇', '했'),
            ('엤', '었')
        ]
        
        for error, _ in common_errors:
            if error in response:
                score -= 0.1
        
        return max(0.0, score)
    
    def _check_coherence(self, response: str, context: Dict) -> float:
        """일관성 검사"""
        # 이전 응답과의 일관성
        previous = context.get('previous_responses', [])
        
        if not previous:
            return 0.9
        
        # 주제 일관성
        current_topic = context.get('current_topic', '')
        if current_topic and current_topic.lower() not in response.lower():
            return 0.7
        
        return 0.9
    
    def _check_completeness(self, response: str) -> float:
        """완성도 검사"""
        # 응답 길이
        if len(response) < 10:
            return 0.5
        elif len(response) > 500:
            return 0.8
        
        return 1.0
    
    def _check_appropriateness(self, response: str, context: Dict) -> float:
        """적절성 검사"""
        user_state = context.get('user_state', {})
        
        # 사용자 수준에 맞는지
        if user_state.get('expertise') == 'beginner':
            # 너무 기술적인 용어가 많으면 감점
            technical_terms = ['polymorphism', 'delegate', 'lambda', 'LINQ']
            term_count = sum(1 for term in technical_terms if term in response)
            if term_count > 2:
                return 0.7
        
        return 0.9


class EmotionEngine:
    """감정 분석 엔진"""
    
    def analyze(self, processed_input: Dict, context: Dict) -> Dict:
        """감정 분석"""
        text = processed_input['cleaned']
        
        emotion_indicators = {
            'happy': ['좋아', '기뻐', '감사', '최고', '😊', '😄'],
            'sad': ['슬퍼', '아쉬워', '힘들어', '😢', '😭'],
            'angry': ['화나', '짜증', '싫어', '😡', '😠'],
            'frustrated': ['답답', '어려워', '모르겠', '안돼', '😤'],
            'excited': ['신나', '대박', '와', '기대', '🎉'],
            'confused': ['헷갈려', '이해가', '뭔지', '🤔'],
            'neutral': []
        }
        
        detected_emotion = 'neutral'
        max_score = 0
        
        for emotion, indicators in emotion_indicators.items():
            score = sum(1 for ind in indicators if ind in text)
            if score > max_score:
                max_score = score
                detected_emotion = emotion
        
        # 감정 강도 계산
        intensity = min(1.0, max_score / 3)
        
        return {
            'type': detected_emotion,
            'intensity': intensity,
            'indicators': max_score
        }


class TopicTracker:
    """주제 추적기"""
    
    def __init__(self):
        self.recent_topics = deque(maxlen=5)
        self.topic_keywords = {
            'unity': ['unity', '유니티', 'gameobject', 'transform', 'component'],
            'csharp': ['c#', 'class', 'method', 'variable', 'namespace'],
            'debugging': ['에러', 'error', '디버깅', 'exception', '오류'],
            'architecture': ['구조', 'pattern', '설계', 'architecture', '패턴'],
            'performance': ['성능', 'performance', '최적화', 'optimize', '속도']
        }
    
    def get_current_topic(self, processed_input: Dict) -> str:
        """현재 주제 추출"""
        text = processed_input['cleaned'].lower()
        
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            current_topic = max(topic_scores.items(), key=lambda x: x[1])[0]
            self.recent_topics.append(current_topic)
            return current_topic
        
        return 'general'
    
    def get_recent_topics(self) -> List[str]:
        """최근 주제 목록"""
        return list(self.recent_topics)


class StyleAdapter:
    """스타일 적응기"""
    
    def apply_style(self, text: str, style: str) -> str:
        """텍스트에 스타일 적용"""
        if style == 'professional':
            return self._make_professional(text)
        elif style == 'friendly':
            return self._make_friendly(text)
        elif style == 'technical':
            return self._make_technical(text)
        
        return text
    
    def _make_professional(self, text: str) -> str:
        """전문적인 스타일"""
        # 존댓말 변환
        replacements = [
            ('해', '하세요'),
            ('돼', '됩니다'),
            ('야', '셔야 합니다')
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        return text
    
    def _make_friendly(self, text: str) -> str:
        """친근한 스타일"""
        # 이모티콘 추가
        if text.endswith('!'):
            text += ' 😊'
        
        return text
    
    def _make_technical(self, text: str) -> str:
        """기술적인 스타일"""
        # 전문 용어 강조
        return text


class ConversationMemory:
    """대화 메모리 관리"""
    
    def __init__(self, db_path: str = "conversation_memory.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input TEXT,
                response TEXT,
                context TEXT,
                metadata TEXT,
                quality_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store(self, conversation: Dict):
        """대화 저장"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (user_input, response, context, metadata, quality_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            conversation['user_input'],
            json.dumps(conversation['response']),
            json.dumps(conversation.get('context', {})),
            json.dumps(conversation.get('metadata', {})),
            conversation['response'].get('quality_score', 0.5)
        ))
        
        conn.commit()
        conn.close()


class DialogueNeuralNetwork(nn.Module):
    """대화 신경망"""
    
    def __init__(self, input_size: int = 18, hidden_size: int = 128, output_size: int = 50):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# 테스트 함수
if __name__ == "__main__":
    print("🚀 상용화 수준 AI 대화 엔진 테스트")
    print("=" * 60)
    
    engine = CommercialDialogueEngine()
    
    test_inputs = [
        "안녕하세요! Unity 개발을 처음 시작하는데 도움이 필요해요.",
        "GameObject가 null인데 왜 NullReferenceException이 발생하나요?",
        "코루틴과 async/await 중 뭘 써야 할까요?",
        "정말 감사합니다! 덕분에 문제를 해결했어요!"
    ]
    
    for user_input in test_inputs:
        print(f"\n👤 User: {user_input}")
        
        result = engine.process_dialogue(user_input)
        
        print(f"🤖 AI: {result['response']}")
        print(f"   품질: {result['quality_score']:.2f} | "
              f"신뢰도: {result['confidence']:.2f} | "
              f"응답시간: {result['response_time']:.3f}초")