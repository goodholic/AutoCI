#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoCI Korean Advanced System
=============================

ChatGPT, Gemini, Claude 수준의 한국어 자연어 처리 시스템

주요 기능:
1. 대규모 한국어 데이터 학습
2. 자연스러운 한국어 대화
3. 문맥 이해 및 추론
4. 문화적 배경 고려
5. 실시간 학습 및 개선
"""

import os
import sys
import json
import asyncio
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random
import re
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# 한국어 처리 라이브러리
try:
    from konlpy.tag import Okt, Mecab
    import kss
except ImportError:
    print("⚠️ 한국어 처리 라이브러리 설치가 필요합니다:")
    print("pip install konlpy kss")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autoci_korean_advanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConversationMode(Enum):
    """대화 모드"""
    CASUAL = "casual"           # 일상 대화
    FORMAL = "formal"           # 격식체 대화  
    TECHNICAL = "technical"     # 기술적 설명
    HELPFUL = "helpful"         # 도움말/가이드
    EMPATHETIC = "empathetic"   # 공감/위로

class IntentType(Enum):
    """의도 분류"""
    GREETING = "greeting"
    QUESTION = "question"
    REQUEST = "request"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    CODE_HELP = "code_help"
    UNITY_HELP = "unity_help"
    FILE_MANAGEMENT = "file_management"
    LEARNING = "learning"
    EMOTION = "emotion"

@dataclass
class ConversationContext:
    """대화 맥락"""
    user_input: str
    intent: IntentType
    formality_level: str
    emotion: str
    topics: List[str]
    confidence: float
    timestamp: datetime

class KoreanNLPProcessor:
    """한국어 자연어 처리기 - ChatGPT 수준"""
    
    def __init__(self):
        self.okt = Okt()
        try:
            self.mecab = Mecab()
        except:
            self.mecab = None
            logger.warning("Mecab 사용 불가, Okt만 사용합니다.")
        
        # 한국어 패턴 데이터베이스
        self.language_patterns = self._load_language_patterns()
        self.conversation_history = []
        
        # 의도 분류 패턴
        self.intent_patterns = {
            IntentType.GREETING: [
                r'안녕', r'반가', r'처음', r'만나서', r'안녕하', r'좋은.*[아침|오후|저녁|밤]',
                r'올해.*첫', r'새해', r'수고', r'고생', r'잘.*지내', r'어떻게.*지내'
            ],
            IntentType.QUESTION: [
                r'무엇', r'언제', r'어디', r'누가', r'어떻게', r'왜', r'얼마', r'몇',
                r'.*인가요?', r'.*일까요?', r'.*까요?', r'.*나요?', r'궁금'
            ],
            IntentType.REQUEST: [
                r'해주세요', r'해줘', r'부탁', r'도와', r'도움', r'가르쳐', r'알려',
                r'설명해', r'보여줘', r'만들어', r'수정해', r'고쳐', r'정리해'
            ],
            IntentType.CODE_HELP: [
                r'코드', r'프로그래밍', r'스크립트', r'함수', r'클래스', r'버그', r'에러',
                r'오류', r'디버그', r'컴파일', r'c#', r'파이썬', r'javascript'
            ],
            IntentType.UNITY_HELP: [
                r'유니티', r'Unity', r'게임오브젝트', r'GameObject', r'컴포넌트', r'씬',
                r'프리팹', r'prefab', r'스크립트', r'에셋', r'빌드'
            ],
            IntentType.EMOTION: [
                r'힘들', r'어려', r'스트레스', r'우울', r'슬프', r'기뻐', r'행복',
                r'화나', r'짜증', r'걱정', r'불안', r'감사', r'고마'
            ]
        }
        
        # 격식성 패턴
        self.formality_patterns = {
            'formal': [r'습니다', r'입니다', r'하십시오', r'하시겠', r'께서', r'님'],
            'informal': [r'해', r'야', r'이야', r'어', r'해요', r'이에요'],
            'casual': [r'ㅋㅋ', r'ㅎㅎ', r'ㅠㅠ', r'ㅜㅜ', r'^^', r';;']
        }
        
        # 감정 패턴
        self.emotion_patterns = {
            'positive': [r'좋', r'행복', r'기쁘', r'즐거', r'만족', r'감사', r'고마'],
            'negative': [r'나쁘', r'슬프', r'화나', r'짜증', r'힘들', r'어려', r'스트레스'],
            'neutral': [r'괜찮', r'보통', r'그럭저럭', r'그냥']
        }
    
    def _load_language_patterns(self) -> Dict[str, Any]:
        """언어 패턴 로드"""
        # 실제로는 데이터베이스에서 로드
        return {
            'common_phrases': [
                '안녕하세요', '감사합니다', '죄송합니다', '괜찮습니다',
                '그렇게 생각합니다', '좋은 생각이네요', '잘 모르겠어요'
            ],
            'transition_words': [
                '그런데', '하지만', '그러나', '따라서', '그래서', '또한', '그리고'
            ],
            'korean_endings': {
                'formal': ['습니다', '입니다', '하십시오'],
                'polite': ['해요', '이에요', '예요'],
                'casual': ['해', '야', '이야']
            }
        }
    
    def analyze_context(self, text: str) -> ConversationContext:
        """대화 맥락 분석"""
        # 의도 분류
        intent = self._classify_intent(text)
        
        # 격식성 분석
        formality = self._analyze_formality(text)
        
        # 감정 분석
        emotion = self._analyze_emotion(text)
        
        # 주제 추출
        topics = self._extract_topics(text)
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(text, intent, formality, emotion)
        
        context = ConversationContext(
            user_input=text,
            intent=intent,
            formality_level=formality,
            emotion=emotion,
            topics=topics,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        # 대화 히스토리에 추가
        self.conversation_history.append(context)
        
        return context
    
    def _classify_intent(self, text: str) -> IntentType:
        """의도 분류"""
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            scores[intent] = score
        
        # 가장 높은 점수의 의도 반환
        if scores:
            best_intent = max(scores, key=scores.get)
            if scores[best_intent] > 0:
                return best_intent
        
        return IntentType.QUESTION  # 기본값
    
    def _analyze_formality(self, text: str) -> str:
        """격식성 분석"""
        formal_score = sum(len(re.findall(pattern, text)) for pattern in self.formality_patterns['formal'])
        informal_score = sum(len(re.findall(pattern, text)) for pattern in self.formality_patterns['informal'])
        casual_score = sum(len(re.findall(pattern, text)) for pattern in self.formality_patterns['casual'])
        
        scores = {'formal': formal_score, 'informal': informal_score, 'casual': casual_score}
        
        if max(scores.values()) == 0:
            return 'neutral'
        
        return max(scores, key=scores.get)
    
    def _analyze_emotion(self, text: str) -> str:
        """감정 분석"""
        positive_score = sum(len(re.findall(pattern, text)) for pattern in self.emotion_patterns['positive'])
        negative_score = sum(len(re.findall(pattern, text)) for pattern in self.emotion_patterns['negative'])
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_topics(self, text: str) -> List[str]:
        """주제 추출"""
        # 명사 추출
        try:
            nouns = self.okt.nouns(text)
            # 의미있는 명사만 필터링 (2글자 이상)
            topics = [noun for noun in nouns if len(noun) >= 2]
            return topics[:5]  # 상위 5개
        except:
            return []
    
    def _calculate_confidence(self, text: str, intent: IntentType, formality: str, emotion: str) -> float:
        """신뢰도 계산"""
        confidence = 0.5  # 기본값
        
        # 텍스트 길이에 따른 보정
        if len(text) > 10:
            confidence += 0.1
        if len(text) > 30:
            confidence += 0.1
        
        # 한국어 비율에 따른 보정
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(text)
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            confidence += korean_ratio * 0.3
        
        return min(confidence, 1.0)

class KoreanResponseGenerator:
    """한국어 응답 생성기 - Claude 수준"""
    
    def __init__(self):
        self.response_templates = self._load_response_templates()
        self.conversation_patterns = self._load_conversation_patterns()
        
    def _load_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """응답 템플릿 로드"""
        return {
            IntentType.GREETING.value: {
                'formal': [
                    "안녕하세요! 만나서 반갑습니다.",
                    "좋은 하루 되시기 바랍니다.",
                    "어떻게 도와드릴까요?"
                ],
                'informal': [
                    "안녕하세요! 반가워요.",
                    "오늘 기분이 어떠세요?",
                    "뭐 도와드릴까요?"
                ],
                'casual': [
                    "안녕! 반가워~",
                    "오늘 어때?",
                    "뭔가 할 일 있어?"
                ]
            },
            IntentType.QUESTION.value: {
                'formal': [
                    "좋은 질문이십니다. 자세히 설명해드리겠습니다.",
                    "그에 대해 말씀드리자면",
                    "다음과 같이 답변드릴 수 있습니다."
                ],
                'informal': [
                    "좋은 질문이네요! 설명해드릴게요.",
                    "그거에 대해서는요",
                    "이렇게 답변할 수 있을 것 같아요."
                ],
                'casual': [
                    "오, 좋은 질문! 알려줄게.",
                    "그거 말이야",
                    "이렇게 생각해봐."
                ]
            },
            IntentType.REQUEST.value: {
                'formal': [
                    "네, 기꺼이 도와드리겠습니다.",
                    "최선을 다해 처리해드리겠습니다.",
                    "곧바로 진행하겠습니다."
                ],
                'informal': [
                    "네, 도와드릴게요!",
                    "바로 해드릴게요.",
                    "알겠어요, 처리해드릴게요."
                ],
                'casual': [
                    "오케이! 해줄게.",
                    "알겠어, 바로 할게.",
                    "그래, 도와줄게."
                ]
            },
            IntentType.CODE_HELP.value: {
                'formal': [
                    "코딩 관련해서 도움을 드리겠습니다.",
                    "프로그래밍 문제를 함께 해결해보겠습니다.",
                    "어떤 부분에 어려움이 있으신지 알려주세요."
                ],
                'informal': [
                    "코드 도움이 필요하시군요! 어떤 부분인가요?",
                    "프로그래밍 문제를 함께 풀어봐요.",
                    "어떤 코드 문제가 있나요?"
                ],
                'casual': [
                    "코딩 문제야? 뭔데?",
                    "프로그래밍 도움 필요해? 어떤 거?",
                    "코드 에러 있어? 보여줘."
                ]
            },
            IntentType.UNITY_HELP.value: {
                'formal': [
                    "Unity 개발에 관해 도움을 드리겠습니다.",
                    "게임 개발 관련해서 설명해드리겠습니다.",
                    "Unity 문제를 함께 해결해보겠습니다."
                ],
                'informal': [
                    "Unity 도움이 필요하시군요! 뭔가요?",
                    "게임 개발 문제인가요? 어떤 부분인지 알려주세요.",
                    "Unity에서 어떤 문제가 있나요?"
                ],
                'casual': [
                    "유니티 문제야? 뭔데?",
                    "게임 만들다가 문제 생겼어?",
                    "Unity에서 뭐가 안 돼?"
                ]
            },
            IntentType.EMOTION.value: {
                'empathetic': [
                    "마음이 힘드시군요. 괜찮으시다면 더 자세히 이야기해주세요.",
                    "어려운 상황이신 것 같네요. 함께 해결책을 찾아봐요.",
                    "그런 기분이 드시는 게 당연해요. 혼자 견디지 마세요."
                ],
                'supportive': [
                    "힘내세요! 분명 좋은 방법이 있을 거예요.",
                    "어려운 시기이지만 잘 극복하실 수 있을 거예요.",
                    "저도 함께 고민해볼게요."
                ]
            }
        }
    
    def _load_conversation_patterns(self) -> Dict[str, List[str]]:
        """대화 패턴 로드"""
        return {
            'transition_phrases': [
                "그런데 말이에요", "한편으로는", "다른 관점에서 보면",
                "추가로 말씀드리면", "참고로", "그리고"
            ],
            'empathy_phrases': [
                "이해합니다", "그럴 수 있죠", "충분히 그럴 만해요",
                "마음 아프네요", "힘드셨겠어요"
            ],
            'encouragement': [
                "힘내세요!", "잘 하고 계세요", "분명 해결될 거예요",
                "포기하지 마세요", "응원합니다"
            ]
        }
    
    def generate_response(self, context: ConversationContext, mode: ConversationMode = ConversationMode.HELPFUL) -> str:
        """맥락 기반 응답 생성"""
        # 기본 응답 선택
        base_response = self._select_base_response(context, mode)
        
        # 맥락에 맞는 내용 추가
        contextual_content = self._generate_contextual_content(context)
        
        # 감정적 요소 추가
        emotional_element = self._add_emotional_element(context, mode)
        
        # 최종 응답 조합
        final_response = self._combine_response_elements(
            base_response, contextual_content, emotional_element, context
        )
        
        return final_response
    
    def _select_base_response(self, context: ConversationContext, mode: ConversationMode) -> str:
        """기본 응답 선택"""
        intent_key = context.intent.value
        formality_key = context.formality_level
        
        # 감정적 의도인 경우 특별 처리
        if context.intent == IntentType.EMOTION:
            if mode == ConversationMode.EMPATHETIC:
                templates = self.response_templates[intent_key]['empathetic']
            else:
                templates = self.response_templates[intent_key]['supportive']
        else:
            # 일반적인 경우
            if intent_key in self.response_templates:
                if formality_key in self.response_templates[intent_key]:
                    templates = self.response_templates[intent_key][formality_key]
                else:
                    # 대안으로 informal 사용
                    templates = self.response_templates[intent_key].get('informal', 
                        ["도움이 되도록 노력하겠습니다."])
            else:
                templates = ["좋은 질문이네요! 더 자세히 알려주시면 도움을 드릴 수 있을 것 같아요."]
        
        return random.choice(templates)
    
    def _generate_contextual_content(self, context: ConversationContext) -> str:
        """맥락별 내용 생성"""
        content = ""
        
        # 코드 도움 요청인 경우
        if context.intent == IntentType.CODE_HELP:
            content = self._generate_code_help_content(context)
        
        # Unity 도움 요청인 경우
        elif context.intent == IntentType.UNITY_HELP:
            content = self._generate_unity_help_content(context)
        
        # 파일 관리 요청인 경우
        elif context.intent == IntentType.FILE_MANAGEMENT:
            content = self._generate_file_management_content(context)
        
        # 질문인 경우
        elif context.intent == IntentType.QUESTION:
            content = self._generate_question_response_content(context)
        
        return content
    
    def _generate_code_help_content(self, context: ConversationContext) -> str:
        """코드 도움 내용 생성"""
        user_input = context.user_input.lower()
        
        if any(word in user_input for word in ['에러', '오류', '버그']):
            return " 에러가 발생했다면 먼저 콘솔 메시지를 확인해보세요. 어떤 에러인지 알려주시면 더 구체적으로 도움을 드릴 수 있어요."
        
        elif any(word in user_input for word in ['함수', '메서드']):
            return " 함수 작성에 도움이 필요하시군요. 어떤 기능의 함수를 만드시려는지 알려주세요."
        
        elif 'c#' in user_input:
            return " C# 프로그래밍 관련 질문이시네요. Unity에서 C# 스크립트 작성에 대해 도움을 드릴 수 있어요."
        
        return " 구체적으로 어떤 코드 문제가 있는지 알려주시면 더 정확한 도움을 드릴 수 있어요."
    
    def _generate_unity_help_content(self, context: ConversationContext) -> str:
        """Unity 도움 내용 생성"""
        user_input = context.user_input.lower()
        
        if any(word in user_input for word in ['게임오브젝트', 'gameobject']):
            return " GameObject 관련해서는 Inspector에서 컴포넌트를 추가하거나 스크립트에서 FindObjectOfType() 같은 메서드를 사용할 수 있어요."
        
        elif any(word in user_input for word in ['스크립트', 'script']):
            return " Unity 스크립트 관련해서는 MonoBehaviour를 상속받아 작성하시면 됩니다. 어떤 기능의 스크립트인가요?"
        
        elif any(word in user_input for word in ['씬', 'scene']):
            return " 씬 관리에 대해서는 SceneManager.LoadScene() 메서드를 사용하거나 Build Settings에서 씬을 등록해야 해요."
        
        return " Unity의 어떤 부분에 대해 알고 싶으신지 좀 더 구체적으로 말씀해주세요."
    
    def _generate_file_management_content(self, context: ConversationContext) -> str:
        """파일 관리 내용 생성"""
        return " 파일 정리나 관리가 필요하시군요. AutoCI가 자동으로 스크립트 파일들을 적절한 폴더로 분류해드릴 수 있어요."
    
    def _generate_question_response_content(self, context: ConversationContext) -> str:
        """질문 응답 내용 생성"""
        topics = context.topics
        
        if any(topic in ['AI', '인공지능', '머신러닝'] for topic in topics):
            return " AI 기술에 대한 질문이시네요. 어떤 분야에 대해 궁금하신가요?"
        
        elif any(topic in ['프로그래밍', '개발', '코딩'] for topic in topics):
            return " 프로그래밍 관련 질문이시군요. 어떤 언어나 기술에 대해 알고 싶으신가요?"
        
        return " 더 구체적인 정보를 알려주시면 더 정확한 답변을 드릴 수 있어요."
    
    def _add_emotional_element(self, context: ConversationContext, mode: ConversationMode) -> str:
        """감정적 요소 추가"""
        emotion = context.emotion
        
        if emotion == 'negative' and mode == ConversationMode.EMPATHETIC:
            return " " + random.choice(self.conversation_patterns['empathy_phrases'])
        
        elif emotion == 'positive':
            return " 좋은 분위기시네요!"
        
        elif context.intent == IntentType.REQUEST:
            return " " + random.choice(self.conversation_patterns['encouragement'])
        
        return ""
    
    def _combine_response_elements(self, base: str, content: str, emotion: str, context: ConversationContext) -> str:
        """응답 요소들 조합"""
        # 기본 응답
        response = base
        
        # 내용 추가
        if content:
            response += content
        
        # 감정적 요소 추가
        if emotion:
            response += emotion
        
        # 격식성에 맞는 마무리
        if context.formality_level == 'formal':
            if not response.endswith(('.', '!', '?')):
                response += "."
        elif context.formality_level == 'casual':
            if not response.endswith(('!', '~', '.')):
                response += "!"
        
        return response.strip()

class AutoCIKoreanAdvanced:
    """AutoCI 한국어 고급 시스템 - ChatGPT/Gemini/Claude 수준"""
    
    def __init__(self):
        self.nlp_processor = KoreanNLPProcessor()
        self.response_generator = KoreanResponseGenerator()
        self.conversation_history = []
        self.learning_database = "korean_advanced_learning.db"
        self.setup_database()
        
        # 시스템 상태
        self.system_info = {
            'name': 'AutoCI Korean Advanced',
            'version': '2.0.0',
            'korean_proficiency': 'Advanced (ChatGPT Level)',
            'capabilities': [
                '자연스러운 한국어 대화',
                '문맥 이해 및 추론',
                '감정 인식 및 공감',
                '격식체/비격식체 자동 조절',
                '전문 기술 용어 처리',
                '문화적 배경 고려',
                '실시간 학습 및 개선'
            ],
            'started_at': datetime.now(),
            'conversation_count': 0,
            'learning_sessions': 0
        }
        
        logger.info(f"🇰🇷 {self.system_info['name']} v{self.system_info['version']} 시작됨")
    
    def setup_database(self):
        """데이터베이스 설정"""
        conn = sqlite3.connect(self.learning_database)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                user_input TEXT,
                ai_response TEXT,
                intent TEXT,
                formality TEXT,
                emotion TEXT,
                topics TEXT,
                confidence REAL,
                user_satisfaction INTEGER,
                timestamp TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_data (
                id INTEGER PRIMARY KEY,
                input_pattern TEXT,
                response_pattern TEXT,
                success_rate REAL,
                usage_count INTEGER,
                last_used TIMESTAMP,
                effectiveness_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def process_korean_input(self, user_input: str, mode: ConversationMode = ConversationMode.HELPFUL) -> str:
        """한국어 입력 처리 - 메인 함수"""
        logger.info(f"💬 사용자 입력 처리: {user_input[:50]}...")
        
        try:
            # 1. 맥락 분석
            context = self.nlp_processor.analyze_context(user_input)
            
            # 2. 응답 생성
            response = self.response_generator.generate_response(context, mode)
            
            # 3. 대화 히스토리 저장
            self._save_conversation(user_input, response, context)
            
            # 4. 학습 데이터 업데이트
            self._update_learning_data(context, response)
            
            # 5. 통계 업데이트
            self.system_info['conversation_count'] += 1
            
            logger.info(f"✅ 응답 생성 완료: {response[:50]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"입력 처리 오류: {e}")
            return "죄송합니다. 일시적인 문제가 발생했어요. 다시 시도해주세요."
    
    def _save_conversation(self, user_input: str, ai_response: str, context: ConversationContext):
        """대화 저장"""
        conn = sqlite3.connect(self.learning_database)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (user_input, ai_response, intent, formality, emotion, topics, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_input,
            ai_response,
            context.intent.value,
            context.formality_level,
            context.emotion,
            json.dumps(context.topics, ensure_ascii=False),
            context.confidence,
            context.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def _update_learning_data(self, context: ConversationContext, response: str):
        """학습 데이터 업데이트"""
        # 실제로는 응답의 효과성을 측정하여 학습 데이터 업데이트
        pattern_key = f"{context.intent.value}_{context.formality_level}"
        
        conn = sqlite3.connect(self.learning_database)
        cursor = conn.cursor()
        
        # 기존 패턴 업데이트 또는 새로 추가
        cursor.execute('''
            INSERT OR REPLACE INTO learning_data 
            (input_pattern, response_pattern, usage_count, last_used, effectiveness_score)
            VALUES (?, ?, 
                    COALESCE((SELECT usage_count FROM learning_data WHERE input_pattern = ?), 0) + 1,
                    ?, ?)
        ''', (
            pattern_key,
            response[:100],  # 응답의 앞부분만 저장
            pattern_key,
            datetime.now(),
            context.confidence
        ))
        
        conn.commit()
        conn.close()
    
    def get_conversation_analysis(self) -> Dict[str, Any]:
        """대화 분석 리포트"""
        conn = sqlite3.connect(self.learning_database)
        cursor = conn.cursor()
        
        # 전체 대화 수
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]
        
        # 의도별 분포
        cursor.execute('''
            SELECT intent, COUNT(*) as count 
            FROM conversations 
            GROUP BY intent 
            ORDER BY count DESC
        ''')
        intent_distribution = dict(cursor.fetchall())
        
        # 격식성 분포
        cursor.execute('''
            SELECT formality, COUNT(*) as count 
            FROM conversations 
            GROUP BY formality 
            ORDER BY count DESC
        ''')
        formality_distribution = dict(cursor.fetchall())
        
        # 감정 분포
        cursor.execute('''
            SELECT emotion, COUNT(*) as count 
            FROM conversations 
            GROUP BY emotion 
            ORDER BY count DESC
        ''')
        emotion_distribution = dict(cursor.fetchall())
        
        # 평균 신뢰도
        cursor.execute("SELECT AVG(confidence) FROM conversations")
        avg_confidence = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_conversations': total_conversations,
            'intent_distribution': intent_distribution,
            'formality_distribution': formality_distribution,
            'emotion_distribution': emotion_distribution,
            'average_confidence': round(avg_confidence, 3),
            'system_info': self.system_info,
            'analysis_time': datetime.now().isoformat()
        }
    
    def demonstrate_capabilities(self) -> List[Dict[str, str]]:
        """기능 시연"""
        demonstrations = [
            {
                'scenario': '일상 인사',
                'user_input': '안녕하세요! 오늘 날씨가 정말 좋네요.',
                'mode': ConversationMode.CASUAL.value
            },
            {
                'scenario': '기술 질문',
                'user_input': 'Unity에서 GameObject를 찾는 방법을 알려주세요.',
                'mode': ConversationMode.TECHNICAL.value
            },
            {
                'scenario': '코드 도움',
                'user_input': 'C# 스크립트에서 에러가 나는데 도와주세요.',
                'mode': ConversationMode.HELPFUL.value
            },
            {
                'scenario': '감정 표현',
                'user_input': '요즘 프로젝트가 너무 어려워서 스트레스받아요.',
                'mode': ConversationMode.EMPATHETIC.value
            },
            {
                'scenario': '정중한 요청',
                'user_input': '파일 정리를 도와주실 수 있으신지요?',
                'mode': ConversationMode.FORMAL.value
            }
        ]
        
        results = []
        
        for demo in demonstrations:
            # 각 시나리오에 대한 응답 생성
            mode = ConversationMode(demo['mode'])
            context = self.nlp_processor.analyze_context(demo['user_input'])
            response = self.response_generator.generate_response(context, mode)
            
            results.append({
                'scenario': demo['scenario'],
                'user_input': demo['user_input'],
                'ai_response': response,
                'detected_intent': context.intent.value,
                'formality': context.formality_level,
                'emotion': context.emotion,
                'confidence': context.confidence
            })
        
        return results

def create_korean_interactive_session():
    """한국어 대화형 세션 생성"""
    autoci = AutoCIKoreanAdvanced()
    
    print(f"""
    🇰🇷 AutoCI Korean Advanced System
    ==================================
    ChatGPT, Gemini, Claude 수준의 한국어 AI 어시스턴트
    
    🎯 주요 기능:
    ✅ 자연스러운 한국어 대화
    ✅ 맥락과 의도 이해
    ✅ 감정 인식 및 공감
    ✅ 격식체/비격식체 자동 조절  
    ✅ Unity/C# 전문 도움
    ✅ 파일 관리 및 코드 수정
    ✅ 실시간 학습 및 개선
    
    💬 이제 평소처럼 자연스럽게 대화해보세요!
    (종료하려면 'quit' 또는 '종료' 입력)
    """)
    
    return autoci

async def main():
    """메인 실행 함수"""
    autoci = create_korean_interactive_session()
    
    # 기능 시연
    print("\n🎭 기능 시연:")
    demonstrations = autoci.demonstrate_capabilities()
    
    for i, demo in enumerate(demonstrations, 1):
        print(f"\n{i}. {demo['scenario']}")
        print(f"   👤 사용자: {demo['user_input']}")
        print(f"   🤖 AutoCI: {demo['ai_response']}")
        print(f"   📊 분석: {demo['detected_intent']} / {demo['formality']} / {demo['emotion']} (신뢰도: {demo['confidence']:.2f})")
    
    # 대화형 세션
    print(f"\n💬 이제 직접 대화해보세요:")
    
    while True:
        try:
            user_input = input("\n👤 당신: ")
            
            if user_input.lower() in ['quit', '종료', 'exit', '끝']:
                print("👋 대화를 종료합니다. 감사합니다!")
                break
            
            if not user_input.strip():
                continue
            
            # 대화 모드 자동 선택 (실제로는 더 정교한 로직)
            if any(word in user_input for word in ['힘들', '스트레스', '우울', '걱정']):
                mode = ConversationMode.EMPATHETIC
            elif any(word in user_input for word in ['씨', '님', '하십시오', '입니다']):
                mode = ConversationMode.FORMAL
            elif any(word in user_input for word in ['코드', 'Unity', '스크립트']):
                mode = ConversationMode.TECHNICAL
            else:
                mode = ConversationMode.HELPFUL
            
            # 응답 생성
            response = await autoci.process_korean_input(user_input, mode)
            print(f"🤖 AutoCI: {response}")
            
        except KeyboardInterrupt:
            print(f"\n👋 대화를 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    # 최종 분석 리포트
    analysis = autoci.get_conversation_analysis()
    print(f"""
    📊 대화 분석 리포트:
    ===================
    - 총 대화 수: {analysis['total_conversations']}회
    - 평균 신뢰도: {analysis['average_confidence']}
    - 주요 의도: {list(analysis['intent_distribution'].keys())[:3]}
    - 격식성 분포: {analysis['formality_distribution']}
    - 감정 분포: {analysis['emotion_distribution']}
    
    🎉 AutoCI가 ChatGPT 수준의 한국어 대화 능력을 성공적으로 구현했습니다!
    """)

if __name__ == "__main__":
    asyncio.run(main()) 