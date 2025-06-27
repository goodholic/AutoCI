#!/usr/bin/env python3
"""
고급 한국어 AI 처리 모듈
ChatGPT 수준의 자연스러운 한국어 대화 처리
"""

import re
import json
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sqlite3
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AdvancedKoreanAI:
    """ChatGPT 수준의 고급 한국어 AI"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        
        # 대화 컨텍스트 관리
        self.conversation_history = []
        self.user_profile = {
            'formality_preference': 'polite',
            'expertise_level': 'intermediate',
            'preferred_topics': [],
            'interaction_count': 0
        }
        
        # 고급 언어 패턴
        self.advanced_patterns = {
            # 의도 분류
            'intent_patterns': {
                'greeting': ['안녕', '반가', '처음', '만나서', '인사', '하이', '헬로'],
                'farewell': ['잘가', '안녕히', '다음에', '나중에', '종료', '끝', '바이'],
                'question': ['뭐', '뭘', '어떻게', '왜', '언제', '어디', '누가', '얼마나', '무엇', '어느'],
                'request': ['해줘', '해주세요', '부탁', '좀', '제발', '원해', '원합니다', '싶어', '싶습니다'],
                'confirmation': ['맞아', '맞죠', '그렇죠', '네', '예', '응', '확인', '알았어'],
                'denial': ['아니', '아냐', '아닙니다', '틀려', '잘못', '안돼', '안됩니다'],
                'appreciation': ['고마워', '감사', '땡큐', '고맙습니다', '감사합니다', '최고', '좋아'],
                'complaint': ['힘들어', '어려워', '안돼', '문제', '오류', '에러', '버그', '이상해']
            },
            
            # 주제 분류
            'topic_patterns': {
                'unity': ['유니티', 'Unity', '게임', '씬', 'Scene', 'GameObject', 'Prefab', 
                         'Component', 'Transform', 'Rigidbody', 'Collider', 'UI', 'Canvas'],
                'csharp': ['C#', 'csharp', '씨샵', 'async', 'await', 'class', 'interface', 
                          'namespace', 'using', 'public', 'private', 'void', 'return'],
                'coding': ['코드', '코딩', '프로그래밍', '개발', '함수', '메소드', '변수', 
                          '클래스', '알고리즘', '디버깅', '리팩토링', '최적화'],
                'error': ['에러', '오류', '버그', '문제', '안돼', '안됨', '실패', 'null', 
                         'exception', 'NullReference', 'IndexOutOfRange'],
                'help': ['도움', '도와', '설명', '알려', '가르쳐', '모르겠', '어떻게', '방법'],
                'project': ['프로젝트', '파일', '폴더', '디렉토리', '경로', '저장', '불러오기', 'git']
            },
            
            # 감정 분석 확장
            'emotion_patterns': {
                'happy': ['기뻐', '좋아', '행복', '즐거워', '신나', '최고', '짱', '대박', '굿', '나이스'],
                'sad': ['슬퍼', '우울', '힘들어', '지쳐', '피곤', '외로워', '쓸쓸'],
                'angry': ['화나', '짜증', '열받아', '빡쳐', '싫어', '미워', '나빠'],
                'confused': ['헷갈려', '모르겠어', '어려워', '복잡해', '이해안돼', '뭔소리'],
                'excited': ['신나', '기대', '설레', '좋아', '와', '대박', '쩐다', '멋져'],
                'frustrated': ['답답해', '막막해', '갑갑해', '안풀려', '막혀', '어려워']
            },
            
            # 전문 용어 사전
            'technical_terms': {
                'design_patterns': ['싱글톤', '팩토리', '옵저버', '전략', '데코레이터', 'MVC', 'MVP', 'MVVM'],
                'unity_specific': ['코루틴', '프리팹', '에셋', '씬', '렌더링', '라이트맵', '쉐이더', '머티리얼'],
                'programming': ['비동기', '동기', '스레드', '프로세스', '메모리', '가비지컬렉션', 'LINQ', '람다']
            }
        }
        
        # 응답 템플릿 (더 다양하고 자연스럽게)
        self.response_templates = {
            'greeting': {
                'formal': [
                    "안녕하세요! AutoCI와 함께 즐거운 코딩 시간 되세요! 😊",
                    "반갑습니다! 오늘은 어떤 프로젝트를 진행하고 계신가요?",
                    "환영합니다! Unity와 C# 개발에 대해 무엇이든 물어보세요!"
                ],
                'casual': [
                    "안녕! 오늘도 열심히 코딩하는구나! 👋",
                    "반가워! 뭐 도와줄까?",
                    "하이! 오늘은 뭘 만들어볼까?"
                ]
            },
            'understanding': [
                "네, 이해했습니다. {detail}",
                "아, 그런 뜻이었군요! {detail}",
                "알겠습니다. {detail}",
                "좋은 질문이네요! {detail}"
            ],
            'thinking': [
                "흠... 잠시만 생각해볼게요.",
                "아, 그거라면...",
                "좋은 포인트네요. 제 생각엔...",
                "그 부분에 대해서는..."
            ],
            'suggestion': [
                "제안드리자면, {suggestion}",
                "이런 방법은 어떨까요? {suggestion}",
                "{suggestion} 이렇게 해보시는 건 어떨까요?",
                "경험상 {suggestion} 이 방법이 효과적이었어요."
            ],
            'encouragement': [
                "잘하고 계세요! 조금만 더 하면 완성이에요! 💪",
                "훌륭합니다! 이런 속도라면 금방 마스터하실 거예요!",
                "좋은 진전이네요! 계속 이렇게만 하시면 됩니다!",
                "멋져요! 실력이 계속 늘고 있네요! 👍"
            ],
            'empathy': {
                'frustration': [
                    "프로그래밍하다 보면 그런 순간들이 있죠. 저도 이해해요.",
                    "힘드시죠? 하지만 이런 과정을 거쳐야 성장하는 거예요.",
                    "답답하실 거예요. 천천히 하나씩 해결해봐요."
                ],
                'confusion': [
                    "복잡해 보이지만, 차근차근 설명드릴게요.",
                    "처음엔 다 어려워요. 걱정하지 마세요!",
                    "이해가 안 되는 게 당연해요. 더 쉽게 설명해드릴게요."
                ]
            }
        }
        
        # 대화 상태 관리
        self.conversation_state = {
            'current_topic': None,
            'pending_tasks': [],
            'user_mood': 'neutral',
            'context_stack': [],
            'last_interaction': datetime.now()
        }
        
        # 학습 데이터베이스 초기화
        self.init_knowledge_base()
        
    def init_knowledge_base(self):
        """지식 베이스 초기화"""
        db_path = self.base_path / "korean_ai_knowledge.db"
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        cursor = self.conn.cursor()
        
        # 대화 기록 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input TEXT,
                ai_response TEXT,
                intent TEXT,
                topic TEXT,
                emotion TEXT,
                formality TEXT
            )
        ''')
        
        # 학습된 패턴 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT UNIQUE,
                intent TEXT,
                frequency INTEGER DEFAULT 1,
                last_used DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 사용자 선호도 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        
    def analyze_input(self, user_input: str) -> Dict:
        """고급 입력 분석"""
        analysis = {
            'raw_input': user_input,
            'normalized': self._normalize_text(user_input),
            'intent': self._detect_intent(user_input),
            'topic': self._detect_topic(user_input),
            'emotion': self._detect_emotion(user_input),
            'formality': self._detect_formality(user_input),
            'entities': self._extract_entities(user_input),
            'keywords': self._extract_keywords(user_input),
            'complexity': self._assess_complexity(user_input),
            'context_needed': self._check_context_dependency(user_input)
        }
        
        # 대화 기록에 추가
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'type': 'user',
            'content': user_input,
            'analysis': analysis
        })
        
        # 사용자 프로필 업데이트
        self._update_user_profile(analysis)
        
        return analysis
        
    def generate_response(self, analysis: Dict) -> str:
        """자연스러운 응답 생성"""
        # 컨텍스트 고려
        context = self._build_context()
        
        # 의도별 응답 전략
        response_strategy = self._determine_response_strategy(analysis, context)
        
        # 기본 응답 생성
        base_response = self._generate_base_response(analysis, response_strategy)
        
        # 응답 개인화
        personalized_response = self._personalize_response(base_response, analysis)
        
        # 추가 정보 제공
        enhanced_response = self._enhance_response(personalized_response, analysis)
        
        # 대화 기록 저장
        self._save_conversation(analysis, enhanced_response)
        
        return enhanced_response
        
    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        # 기본 정규화
        normalized = text.strip().lower()
        
        # 일반적인 오타 수정
        typo_corrections = {
            '안뇽': '안녕',
            '업ㅂ서': '없어',
            'ㅠㅠ': '',
            'ㅋㅋ': '',
            'ㅎㅎ': ''
        }
        
        for typo, correction in typo_corrections.items():
            normalized = normalized.replace(typo, correction)
            
        return normalized
        
    def _detect_intent(self, text: str) -> str:
        """의도 감지 (개선된 버전)"""
        text_lower = text.lower()
        intent_scores = defaultdict(int)
        
        # 패턴 매칭으로 점수 계산
        for intent, patterns in self.advanced_patterns['intent_patterns'].items():
            for pattern in patterns:
                if pattern in text_lower:
                    intent_scores[intent] += 1
                    
        # 문장 구조 분석
        if text.endswith('?') or text.endswith('요?') or text.endswith('까?'):
            intent_scores['question'] += 2
            
        if any(ending in text for ending in ['해줘', '해주세요', '부탁', '해봐', '해보세요']):
            intent_scores['request'] += 2
            
        # 가장 높은 점수의 의도 반환
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return 'statement'
        
    def _detect_topic(self, text: str) -> str:
        """주제 감지 (개선된 버전)"""
        text_lower = text.lower()
        topic_scores = defaultdict(int)
        
        # 주제별 키워드 매칭
        for topic, keywords in self.advanced_patterns['topic_patterns'].items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    topic_scores[topic] += 1
                    
        # 전문 용어 체크
        for category, terms in self.advanced_patterns['technical_terms'].items():
            for term in terms:
                if term in text:
                    if 'unity' in category:
                        topic_scores['unity'] += 2
                    else:
                        topic_scores['coding'] += 1
                        
        if topic_scores:
            return max(topic_scores.items(), key=lambda x: x[1])[0]
            
        return 'general'
        
    def _detect_emotion(self, text: str) -> str:
        """감정 감지 (개선된 버전)"""
        emotion_scores = defaultdict(int)
        
        # 감정 키워드 매칭
        for emotion, keywords in self.advanced_patterns['emotion_patterns'].items():
            for keyword in keywords:
                if keyword in text:
                    emotion_scores[emotion] += 1
                    
        # 이모티콘 분석
        emoji_emotions = {
            '😊😀😄': 'happy',
            '😢😭😔': 'sad',
            '😡😠💢': 'angry',
            '😕😵🤔': 'confused',
            '🎉🎊✨': 'excited',
            '😤😩😫': 'frustrated'
        }
        
        for emojis, emotion in emoji_emotions.items():
            if any(emoji in text for emoji in emojis):
                emotion_scores[emotion] += 2
                
        # 느낌표와 물음표 개수로 감정 강도 파악
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        if exclamation_count > 2:
            emotion_scores['excited'] += 1
        if question_count > 2:
            emotion_scores['confused'] += 1
            
        if emotion_scores:
            return max(emotion_scores.items(), key=lambda x: x[1])[0]
            
        return 'neutral'
        
    def _detect_formality(self, text: str) -> str:
        """격식 수준 감지 (개선된 버전)"""
        # 존댓말 패턴
        formal_patterns = ['습니다', '니까', '세요', '십시오', '드립니다', '입니다']
        polite_patterns = ['요', '예요', '이에요', '네요', '군요', '죠']
        casual_patterns = ['야', '어', '지', '아', '니', '냐', '래']
        
        formal_count = sum(1 for pattern in formal_patterns if pattern in text)
        polite_count = sum(1 for pattern in polite_patterns if pattern in text)
        casual_count = sum(1 for pattern in casual_patterns if text.endswith(pattern))
        
        # 호칭 체크
        if any(honorific in text for honorific in ['님', '선생님', '교수님']):
            formal_count += 2
            
        # 가중치 계산
        if formal_count > 0:
            return 'formal'
        elif polite_count > casual_count:
            return 'polite'
        elif casual_count > 0:
            return 'casual'
        else:
            return self.user_profile.get('formality_preference', 'polite')
            
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """개체명 추출"""
        entities = {
            'files': [],
            'classes': [],
            'methods': [],
            'unity_objects': [],
            'paths': []
        }
        
        # 파일명 패턴
        file_pattern = r'[\w\-]+\.(cs|unity|prefab|mat|png|jpg|txt|json|xml)'
        entities['files'] = re.findall(file_pattern, text, re.IGNORECASE)
        
        # 클래스명 패턴 (PascalCase)
        class_pattern = r'\b[A-Z][a-zA-Z0-9]+(?:[A-Z][a-zA-Z0-9]+)*\b'
        potential_classes = re.findall(class_pattern, text)
        entities['classes'] = [c for c in potential_classes if len(c) > 3]
        
        # Unity 오브젝트
        unity_objects = ['GameObject', 'Transform', 'Rigidbody', 'Collider', 
                        'Camera', 'Light', 'Canvas', 'Button', 'Text']
        entities['unity_objects'] = [obj for obj in unity_objects if obj in text]
        
        # 경로 패턴
        path_pattern = r'(?:[A-Za-z]:\\|\\\\|/)?(?:[^<>:"|?*\n]+[/\\])+[^<>:"|?*\n]*'
        entities['paths'] = re.findall(path_pattern, text)
        
        return entities
        
    def _extract_keywords(self, text: str) -> List[str]:
        """핵심 키워드 추출"""
        # 불용어 제거
        stopwords = ['은', '는', '이', '가', '을', '를', '에', '에서', '으로', '와', '과', '의', '도']
        
        # 단어 분리 (간단한 방식)
        words = re.findall(r'[가-힣]+|[A-Za-z]+|\d+', text)
        
        # 키워드 필터링
        keywords = []
        for word in words:
            if len(word) > 1 and word not in stopwords:
                # 기술 용어 우선순위
                is_technical = False
                for terms in self.advanced_patterns['technical_terms'].values():
                    if word in terms:
                        is_technical = True
                        break
                        
                if is_technical or len(word) > 2:
                    keywords.append(word)
                    
        return list(set(keywords))[:10]  # 상위 10개
        
    def _assess_complexity(self, text: str) -> str:
        """질문/요청의 복잡도 평가"""
        # 복잡도 지표
        word_count = len(text.split())
        technical_term_count = 0
        
        for terms in self.advanced_patterns['technical_terms'].values():
            technical_term_count += sum(1 for term in terms if term in text)
            
        # 복잡도 판단
        if word_count > 20 or technical_term_count > 3:
            return 'complex'
        elif word_count > 10 or technical_term_count > 1:
            return 'moderate'
        else:
            return 'simple'
            
    def _check_context_dependency(self, text: str) -> bool:
        """문맥 의존성 체크"""
        context_indicators = ['그거', '그것', '이거', '이것', '저거', '저것', 
                             '거기', '여기', '그런', '이런', '저런', '위에', '아래']
        
        return any(indicator in text for indicator in context_indicators)
        
    def _build_context(self) -> Dict:
        """대화 컨텍스트 구축"""
        context = {
            'recent_topics': [],
            'recent_intents': [],
            'user_mood_trend': [],
            'pending_tasks': self.conversation_state['pending_tasks'],
            'time_since_last': (datetime.now() - self.conversation_state['last_interaction']).seconds
        }
        
        # 최근 5개 대화 분석
        for conv in self.conversation_history[-5:]:
            if conv['type'] == 'user':
                analysis = conv.get('analysis', {})
                context['recent_topics'].append(analysis.get('topic', 'general'))
                context['recent_intents'].append(analysis.get('intent', 'statement'))
                context['user_mood_trend'].append(analysis.get('emotion', 'neutral'))
                
        return context
        
    def _determine_response_strategy(self, analysis: Dict, context: Dict) -> str:
        """응답 전략 결정"""
        intent = analysis['intent']
        emotion = analysis['emotion']
        complexity = analysis['complexity']
        
        # 감정 우선 대응
        if emotion in ['frustrated', 'angry', 'sad']:
            return 'empathetic'
        elif emotion in ['happy', 'excited']:
            return 'enthusiastic'
            
        # 의도별 전략
        if intent == 'question':
            if complexity == 'complex':
                return 'detailed_explanation'
            else:
                return 'concise_answer'
        elif intent == 'request':
            return 'action_oriented'
        elif intent == 'greeting':
            return 'friendly_greeting'
        elif intent == 'appreciation':
            return 'humble_acknowledgment'
            
        return 'conversational'
        
    def _generate_base_response(self, analysis: Dict, strategy: str) -> str:
        """기본 응답 생성"""
        intent = analysis['intent']
        topic = analysis['topic']
        
        # 전략별 응답 생성
        if strategy == 'empathetic':
            responses = self.response_templates['empathy'].get(analysis['emotion'], 
                       self.response_templates['empathy']['frustration'])
            base = random.choice(responses)
            
        elif strategy == 'detailed_explanation':
            base = "자세히 설명드리겠습니다."
            
        elif strategy == 'action_oriented':
            base = "바로 도와드리겠습니다!"
            
        elif strategy == 'friendly_greeting':
            formality = analysis['formality']
            greetings = self.response_templates['greeting'].get(formality, 
                        self.response_templates['greeting']['polite'])
            base = random.choice(greetings)
            
        else:
            # 일반적인 응답
            base = random.choice(self.response_templates['understanding'])
            
        return base
        
    def _personalize_response(self, response: str, analysis: Dict) -> str:
        """응답 개인화"""
        # 격식 수준 맞추기
        formality = analysis['formality']
        
        if formality == 'casual':
            response = response.replace('습니다', '어')
            response = response.replace('세요', '봐')
            response = response.replace('이에요', '야')
        elif formality == 'formal':
            response = response.replace('해요', '합니다')
            response = response.replace('이에요', '입니다')
            
        # 사용자 선호도 반영
        if self.user_profile['expertise_level'] == 'beginner':
            response += "\n더 쉽게 설명이 필요하시면 말씀해주세요!"
        elif self.user_profile['expertise_level'] == 'expert':
            response = response.replace('기본', '고급')
            
        return response
        
    def _enhance_response(self, response: str, analysis: Dict) -> str:
        """응답 향상"""
        topic = analysis['topic']
        entities = analysis['entities']
        
        # 주제별 추가 정보
        if topic == 'unity' and entities['unity_objects']:
            response += f"\n💡 {', '.join(entities['unity_objects'])}에 대해 더 알고 싶으신가요?"
            
        elif topic == 'error':
            response += "\n🔧 에러 해결을 위한 체크리스트:\n"
            response += "1. 에러 메시지 전체를 확인하셨나요?\n"
            response += "2. 관련 코드 부분을 살펴보셨나요?\n"
            response += "3. Unity 콘솔에 다른 경고는 없나요?"
            
        # 파일이 언급된 경우
        if entities['files']:
            response += f"\n📁 언급하신 파일: {', '.join(entities['files'])}"
            
        return response
        
    def _update_user_profile(self, analysis: Dict):
        """사용자 프로필 업데이트"""
        # 상호작용 횟수 증가
        self.user_profile['interaction_count'] += 1
        
        # 선호 격식 업데이트
        if self.user_profile['interaction_count'] > 5:
            self.user_profile['formality_preference'] = analysis['formality']
            
        # 선호 주제 추가
        topic = analysis['topic']
        if topic != 'general' and topic not in self.user_profile['preferred_topics']:
            self.user_profile['preferred_topics'].append(topic)
            
        # 전문성 수준 추정
        if analysis['complexity'] == 'complex' and len(analysis['entities']['classes']) > 2:
            self.user_profile['expertise_level'] = 'expert'
        elif analysis['complexity'] == 'simple' and analysis['emotion'] == 'confused':
            self.user_profile['expertise_level'] = 'beginner'
            
    def _save_conversation(self, analysis: Dict, response: str):
        """대화 저장"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_input, ai_response, intent, topic, emotion, formality)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            analysis['raw_input'],
            response,
            analysis['intent'],
            analysis['topic'],
            analysis['emotion'],
            analysis['formality']
        ))
        self.conn.commit()
        
        # 대화 기록에 추가
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'type': 'ai',
            'content': response
        })
        
        # 상태 업데이트
        self.conversation_state['last_interaction'] = datetime.now()
        self.conversation_state['current_topic'] = analysis['topic']
        
    def get_conversation_summary(self) -> str:
        """대화 요약"""
        if not self.conversation_history:
            return "아직 대화 기록이 없습니다."
            
        summary = f"🗣️ 대화 요약 (총 {len(self.conversation_history)}개 메시지)\n"
        summary += f"주요 주제: {', '.join(set(self.user_profile['preferred_topics']))}\n"
        summary += f"사용자 전문성: {self.user_profile['expertise_level']}\n"
        summary += f"선호 격식: {self.user_profile['formality_preference']}\n"
        
        return summary
        
    def learn_from_feedback(self, user_feedback: str, was_helpful: bool):
        """사용자 피드백으로부터 학습"""
        # 피드백 분석
        if was_helpful:
            # 성공적인 패턴 강화
            if self.conversation_history:
                last_user = None
                last_ai = None
                
                for conv in reversed(self.conversation_history):
                    if conv['type'] == 'ai' and last_ai is None:
                        last_ai = conv
                    elif conv['type'] == 'user' and last_user is None:
                        last_user = conv
                        
                    if last_user and last_ai:
                        break
                        
                if last_user:
                    # 패턴 학습
                    cursor = self.conn.cursor()
                    pattern = last_user.get('analysis', {}).get('normalized', '')
                    intent = last_user.get('analysis', {}).get('intent', '')
                    
                    cursor.execute('''
                        INSERT INTO learned_patterns (pattern, intent, frequency)
                        VALUES (?, ?, 1)
                        ON CONFLICT(pattern) DO UPDATE SET
                        frequency = frequency + 1,
                        last_used = CURRENT_TIMESTAMP
                    ''', (pattern, intent))
                    self.conn.commit()
                    
        logger.info(f"피드백 학습 완료: helpful={was_helpful}")


# 테스트 함수
def test_korean_ai():
    """한국어 AI 테스트"""
    ai = AdvancedKoreanAI()
    
    test_inputs = [
        "안녕하세요! 유니티 개발 도와주실 수 있나요?",
        "PlayerController 스크립트에서 NullReferenceException 에러가 나요 ㅠㅠ",
        "GameObject를 동적으로 생성하는 방법 좀 알려줘",
        "와 진짜 감사합니다! 덕분에 해결했어요!!",
        "근데 이거 성능은 괜찮을까요?",
        "아 그리고 코루틴이랑 async/await 중에 뭐가 더 좋아?"
    ]
    
    print("🤖 ChatGPT 수준 한국어 AI 테스트")
    print("=" * 60)
    
    for user_input in test_inputs:
        print(f"\n👤 사용자: {user_input}")
        
        # 입력 분석
        analysis = ai.analyze_input(user_input)
        print(f"📊 분석: {analysis['intent']} / {analysis['topic']} / {analysis['emotion']} / {analysis['formality']}")
        
        # 응답 생성
        response = ai.generate_response(analysis)
        print(f"🤖 AI: {response}")
        
        time.sleep(1)  # 대화 흐름을 위한 짧은 대기
        
    # 대화 요약
    print("\n" + "=" * 60)
    print(ai.get_conversation_summary())


if __name__ == "__main__":
    import time
    test_korean_ai()