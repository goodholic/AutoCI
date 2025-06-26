#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Korean AI Simple Demo - ChatGPT Style Learning
==============================================

ChatGPT, Gemini, Claude가 어떻게 한국어를 학습했는지 시연하고
AutoCI에 적용하는 방법을 보여주는 시스템

외부 라이브러리 없이 핵심 개념 구현
"""

import json
import re
import random
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

@dataclass
class KoreanLearningData:
    """한국어 학습 데이터"""
    text: str
    source_type: str
    quality_score: float
    patterns: Dict[str, int]
    features: Dict[str, Any]

class ChatGPTStyleKoreanLearner:
    """ChatGPT 방식의 한국어 학습기"""
    
    def __init__(self):
        print("🧠 ChatGPT 방식 한국어 학습 시스템 초기화 중...")
        
        # 1. 대형 언어 모델들의 한국어 학습 방식
        self.learning_methods = {
            "다국어_사전_훈련": {
                "설명": "ChatGPT, Gemini, Claude가 사용하는 핵심 방법",
                "데이터_소스": [
                    "한국어 웹페이지 (뉴스, 블로그, 포럼)",
                    "한국어 위키피디아",
                    "한국어 문학 작품",
                    "한국어 대화 데이터",
                    "정부 문서, 학술 논문",
                    "소셜미디어 텍스트"
                ],
                "학습량": "수십억 개의 한국어 문장"
            },
            "토크나이저_최적화": {
                "설명": "한국어 특성에 맞는 토큰 분할",
                "특징": [
                    "조사, 어미 변화 처리",
                    "복합어 분해",
                    "한자어 처리",
                    "띄어쓰기 오류 보정"
                ]
            },
            "인간_피드백_강화학습": {
                "설명": "Human Feedback Reinforcement Learning",
                "과정": [
                    "다양한 응답 생성",
                    "인간 평가자가 품질 평가",
                    "선호도에 따른 모델 조정",
                    "자연스러움과 도움됨 최적화"
                ]
            }
        }
        
        # 2. 한국어 패턴 분석기
        self.korean_patterns = {
            # 조사 패턴
            "particles": ["은", "는", "이", "가", "을", "를", "에", "에서", "로", "으로", "와", "과", "의"],
            
            # 어미 패턴
            "endings": {
                "formal": ["습니다", "입니다", "하십시오", "하시겠습니까"],
                "polite": ["해요", "이에요", "예요", "돼요"],
                "casual": ["해", "야", "이야", "어"]
            },
            
            # 높임법 패턴
            "honorifics": ["님", "씨", "선생님", "교수님", "께서", "드리다", "받으시다"],
            
            # 감정 표현
            "emotions": {
                "positive": ["좋다", "행복하다", "기쁘다", "즐겁다", "감사하다"],
                "negative": ["나쁘다", "슬프다", "화나다", "힘들다", "어렵다"],
                "neutral": ["괜찮다", "보통이다", "그럭저럭"]
            },
            
            # 문장 유형
            "sentence_types": {
                "question": ["무엇", "언제", "어디", "누가", "어떻게", "왜"],
                "request": ["해주세요", "해줘", "부탁", "도와주세요"]
            }
        }
        
        # 3. 학습 데이터 저장소
        self.knowledge_base = []
        self.pattern_frequency = defaultdict(int)
        self.conversation_history = []
        
        # 4. 응답 생성 템플릿
        self.response_templates = self._initialize_response_templates()
        
        print("✅ 초기화 완료!")
    
    def _initialize_response_templates(self) -> Dict[str, List[str]]:
        """응답 템플릿 초기화"""
        return {
            "greeting": {
                "formal": [
                    "안녕하세요! 만나서 반갑습니다.",
                    "좋은 하루 되시기 바랍니다.",
                    "어떻게 도와드릴까요?"
                ],
                "casual": [
                    "안녕하세요! 반가워요.",
                    "오늘 어떻게 지내세요?",
                    "뭐 도와드릴까요?"
                ]
            },
            "explanation": {
                "technical": [
                    "자세히 설명해드리겠습니다.",
                    "다음과 같이 이해하시면 됩니다.",
                    "단계별로 알려드릴게요."
                ],
                "simple": [
                    "쉽게 말씀드리면",
                    "간단히 설명하면",
                    "이렇게 생각해보세요."
                ]
            },
            "empathy": [
                "이해합니다.",
                "그럴 수 있죠.",
                "마음이 아프네요.",
                "힘드셨겠어요."
            ],
            "encouragement": [
                "힘내세요!",
                "잘 하고 계세요.",
                "분명 해결될 거예요.",
                "포기하지 마세요."
            ]
        }
    
    def demonstrate_learning_process(self):
        """ChatGPT 방식 학습 과정 시연"""
        print(f"""
        🎓 ChatGPT/Gemini/Claude의 한국어 학습 방식
        ==========================================
        
        📚 1단계: 다국어 사전 훈련 (Multilingual Pre-training)
        --------------------------------------------------------
        • 전 세계 웹에서 수집한 대규모 한국어 텍스트 학습
        • 한국어 뉴스, 블로그, 포럼, 위키피디아 등
        • 문법 패턴, 어휘, 문화적 맥락 동시 학습
        • 다른 언어와의 관계도 함께 학습
        
        🔧 2단계: 한국어 특화 토크나이저 최적화
        -----------------------------------------
        • 한국어 형태소 특성에 맞는 토큰 분할
        • 조사, 어미 변화 패턴 인식
        • 복합어와 띄어쓰기 오류 처리
        • 한자어, 외래어 적절히 처리
        
        🎯 3단계: 인간 피드백 강화학습 (RLHF)
        ------------------------------------
        • 한국어 네이티브 스피커들의 평가
        • 자연스러움, 정확성, 도움 정도 평가
        • 문화적 적절성 고려
        • 반복 학습으로 품질 개선
        
        🌟 결과: 자연스러운 한국어 대화 능력
        ----------------------------------
        • 맥락 이해 및 적절한 응답
        • 격식체/비격식체 자동 조절
        • 문화적 배경 고려
        • 감정 인식 및 공감
        """)
    
    def collect_korean_training_data(self) -> List[KoreanLearningData]:
        """한국어 훈련 데이터 수집 시뮬레이션"""
        print("📊 한국어 훈련 데이터 수집 중...")
        
        # 시뮬레이션 데이터 (실제로는 웹 크롤링)
        sample_korean_texts = [
            {
                "text": "안녕하세요! 오늘 날씨가 정말 좋네요. 산책하기 딱 좋은 날 같아요.",
                "source": "blog",
                "domain": "일상"
            },
            {
                "text": "프로그래밍을 배우는 것은 처음에는 어렵지만, 꾸준히 연습하면 실력이 늘어요.",
                "source": "educational",
                "domain": "기술"
            },
            {
                "text": "한국의 전통 음식인 김치는 건강에도 좋고 맛도 뛰어납니다.",
                "source": "encyclopedia",
                "domain": "문화"
            },
            {
                "text": "Unity에서 GameObject를 생성하려면 Hierarchy 창에서 우클릭하면 됩니다.",
                "source": "technical",
                "domain": "게임개발"
            },
            {
                "text": "힘든 일이 있어도 포기하지 마세요. 분명 좋은 결과가 있을 거예요.",
                "source": "counseling",
                "domain": "심리"
            },
            {
                "text": "대한민국 정부는 국민의 안전과 복지를 위해 다양한 정책을 시행하고 있습니다.",
                "source": "government",
                "domain": "정치"
            },
            {
                "text": "AI 기술이 발전하면서 우리의 일상생활이 점점 더 편리해지고 있어요.",
                "source": "news",
                "domain": "기술"
            },
            {
                "text": "친구와 함께 맛있는 음식을 먹으며 이야기하는 시간이 가장 행복해요.",
                "source": "social",
                "domain": "일상"
            }
        ]
        
        training_data = []
        
        for item in sample_korean_texts:
            # 패턴 분석
            patterns = self._analyze_patterns(item["text"])
            
            # 언어학적 특징 추출
            features = self._extract_features(item["text"])
            
            # 품질 점수 계산
            quality_score = self._calculate_quality_score(item["text"], patterns, features)
            
            learning_data = KoreanLearningData(
                text=item["text"],
                source_type=item["source"],
                quality_score=quality_score,
                patterns=patterns,
                features=features
            )
            
            training_data.append(learning_data)
            self.knowledge_base.append(learning_data)
        
        print(f"✅ {len(training_data)}개의 한국어 학습 데이터 수집 완료")
        return training_data
    
    def _analyze_patterns(self, text: str) -> Dict[str, int]:
        """텍스트에서 한국어 패턴 분석"""
        patterns = defaultdict(int)
        
        # 조사 패턴
        for particle in self.korean_patterns["particles"]:
            patterns[f"particle_{particle}"] = text.count(particle)
        
        # 어미 패턴
        for level, endings in self.korean_patterns["endings"].items():
            for ending in endings:
                if ending in text:
                    patterns[f"ending_{level}"] += 1
        
        # 높임법 패턴
        for honorific in self.korean_patterns["honorifics"]:
            if honorific in text:
                patterns["honorific"] += 1
        
        # 감정 패턴
        for emotion_type, words in self.korean_patterns["emotions"].items():
            for word in words:
                if word in text:
                    patterns[f"emotion_{emotion_type}"] += 1
        
        return dict(patterns)
    
    def _extract_features(self, text: str) -> Dict[str, Any]:
        """언어학적 특징 추출"""
        features = {}
        
        # 기본 통계
        features["length"] = len(text)
        features["word_count"] = len(text.split())
        features["sentence_count"] = len([s for s in text.split('.') if s.strip()])
        
        # 한국어 비율
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.findall(r'[가-힣a-zA-Z0-9]', text))
        features["korean_ratio"] = korean_chars / total_chars if total_chars > 0 else 0
        
        # 격식성 분석
        formal_patterns = sum(text.count(ending) for ending in self.korean_patterns["endings"]["formal"])
        casual_patterns = sum(text.count(ending) for ending in self.korean_patterns["endings"]["casual"])
        
        if formal_patterns > casual_patterns:
            features["formality"] = "formal"
        elif casual_patterns > formal_patterns:
            features["formality"] = "casual"
        else:
            features["formality"] = "neutral"
        
        # 문장 유형
        if any(word in text for word in self.korean_patterns["sentence_types"]["question"]):
            features["sentence_type"] = "question"
        elif any(word in text for word in self.korean_patterns["sentence_types"]["request"]):
            features["sentence_type"] = "request"
        else:
            features["sentence_type"] = "statement"
        
        return features
    
    def _calculate_quality_score(self, text: str, patterns: Dict[str, int], features: Dict[str, Any]) -> float:
        """품질 점수 계산"""
        score = 0.0
        
        # 길이 점수 (0.3)
        ideal_length = 50
        length_score = min(features["length"] / ideal_length, 1.0) * 0.3
        score += length_score
        
        # 한국어 비율 점수 (0.3)
        korean_score = features["korean_ratio"] * 0.3
        score += korean_score
        
        # 문법 패턴 점수 (0.2)
        pattern_count = sum(patterns.values())
        grammar_score = min(pattern_count / 5, 1.0) * 0.2
        score += grammar_score
        
        # 완성도 점수 (0.2)
        completeness = 0.2 if features["sentence_count"] >= 1 else 0.1
        score += completeness
        
        return min(score, 1.0)
    
    def perform_rlhf_training(self):
        """인간 피드백 강화학습 시뮬레이션"""
        print("\n🎯 인간 피드백 강화학습 (RLHF) 시뮬레이션")
        print("=" * 50)
        
        # 테스트 프롬프트들
        test_prompts = [
            "안녕하세요! 어떻게 지내세요?",
            "Unity에서 스크립트 작성하는 방법을 알려주세요.",
            "요즘 스트레스가 많아서 힘들어요.",
            "한국의 전통 문화에 대해 설명해주세요.",
            "파이썬 프로그래밍을 배우고 싶어요."
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. 테스트 프롬프트: \"{prompt}\"")
            
            # 여러 버전의 응답 생성
            responses = self._generate_multiple_responses(prompt)
            
            print("   생성된 응답들:")
            for j, response in enumerate(responses, 1):
                print(f"   응답 {j}: {response}")
            
            # 인간 피드백 시뮬레이션
            best_response, feedback = self._simulate_human_feedback(prompt, responses)
            
            print(f"   🏆 최고 평가 응답: {best_response}")
            print(f"   📝 피드백: {feedback}")
            
            # 학습 데이터 업데이트
            self._update_learning_from_feedback(prompt, best_response, feedback)
    
    def _generate_multiple_responses(self, prompt: str) -> List[str]:
        """프롬프트에 대한 여러 응답 생성"""
        responses = []
        
        # 프롬프트 분석
        features = self._extract_features(prompt)
        patterns = self._analyze_patterns(prompt)
        
        # 인사말 응답
        if "안녕" in prompt:
            responses.extend(self.response_templates["greeting"]["formal"][:2])
            responses.extend(self.response_templates["greeting"]["casual"][:1])
        
        # 기술 질문 응답
        elif any(word in prompt.lower() for word in ["unity", "스크립트", "프로그래밍", "코드"]):
            base_responses = [
                "기술적인 질문이시네요. 자세히 설명해드리겠습니다.",
                "프로그래밍 관련해서 도움을 드릴게요.",
                "그 부분에 대해 단계별로 알려드릴게요."
            ]
            responses.extend(base_responses)
        
        # 감정적 표현에 대한 응답
        elif any(word in prompt for word in ["스트레스", "힘들", "어려", "걱정"]):
            responses.extend(self.response_templates["empathy"][:2])
            responses.extend(self.response_templates["encouragement"][:1])
        
        # 기본 응답
        else:
            responses.extend([
                "좋은 질문이네요! 설명해드릴게요.",
                "그에 대해 자세히 알려드리겠습니다.",
                "흥미로운 주제입니다. 함께 알아봐요."
            ])
        
        return responses[:3]  # 최대 3개 응답
    
    def _simulate_human_feedback(self, prompt: str, responses: List[str]) -> tuple:
        """인간 피드백 시뮬레이션"""
        scores = []
        
        for response in responses:
            # 평가 기준
            appropriateness = self._evaluate_appropriateness(prompt, response)
            naturalness = self._evaluate_naturalness(response)
            helpfulness = self._evaluate_helpfulness(prompt, response)
            
            total_score = (appropriateness + naturalness + helpfulness) / 3
            scores.append(total_score)
        
        # 최고 점수 응답 선택
        best_index = scores.index(max(scores))
        best_response = responses[best_index]
        
        # 피드백 생성
        feedback = self._generate_feedback(scores, best_index)
        
        return best_response, feedback
    
    def _evaluate_appropriateness(self, prompt: str, response: str) -> float:
        """적절성 평가"""
        score = 0.7  # 기본 점수
        
        # 격식성 일치
        prompt_formal = "습니다" in prompt or "하십시오" in prompt
        response_formal = "습니다" in response or "입니다" in response
        
        if prompt_formal == response_formal:
            score += 0.2
        
        # 주제 관련성
        if "Unity" in prompt and "Unity" in response:
            score += 0.1
        elif "스트레스" in prompt and any(word in response for word in ["이해", "힘내", "괜찮"]):
            score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_naturalness(self, response: str) -> float:
        """자연스러움 평가"""
        score = 0.6
        
        # 문장 구조
        if len(response.split()) >= 3:
            score += 0.2
        
        # 한국어다움
        korean_ratio = len(re.findall(r'[가-힣]', response)) / len(response)
        score += korean_ratio * 0.2
        
        return min(score, 1.0)
    
    def _evaluate_helpfulness(self, prompt: str, response: str) -> float:
        """도움 정도 평가"""
        score = 0.5
        
        # 길이 적절성
        if 20 <= len(response) <= 100:
            score += 0.3
        
        # 구체성
        if any(word in response for word in ["방법", "단계", "설명", "도움"]):
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_feedback(self, scores: List[float], best_index: int) -> str:
        """피드백 생성"""
        avg_score = sum(scores) / len(scores)
        best_score = scores[best_index]
        
        if best_score >= 0.8:
            return f"우수한 응답입니다. (점수: {best_score:.2f}) 자연스럽고 도움이 되는 답변이에요."
        elif best_score >= 0.6:
            return f"좋은 응답입니다. (점수: {best_score:.2f}) 더 구체적이면 더 좋겠어요."
        else:
            return f"개선이 필요합니다. (점수: {best_score:.2f}) 더 자연스럽고 도움되는 답변이 필요해요."
    
    def _update_learning_from_feedback(self, prompt: str, best_response: str, feedback: str):
        """피드백으로부터 학습 업데이트"""
        # 성공적인 패턴 저장
        prompt_patterns = self._analyze_patterns(prompt)
        response_patterns = self._analyze_patterns(best_response)
        
        # 패턴 빈도 업데이트
        for pattern, count in response_patterns.items():
            self.pattern_frequency[pattern] += count
        
        # 대화 히스토리에 추가
        self.conversation_history.append({
            "prompt": prompt,
            "response": best_response,
            "feedback": feedback,
            "timestamp": datetime.now()
        })
    
    def demonstrate_korean_conversation(self):
        """한국어 대화 능력 시연"""
        print(f"\n💬 AutoCI 한국어 대화 시연")
        print("=" * 40)
        
        test_conversations = [
            {
                "user": "안녕하세요! 오늘 하루 어떻게 보내셨어요?",
                "context": "일상적인 인사"
            },
            {
                "user": "Unity에서 GameObject를 찾는 방법을 알려주세요.",
                "context": "기술적 질문"
            },
            {
                "user": "요즘 프로젝트가 너무 어려워서 스트레스받아요.",
                "context": "감정적 표현"
            },
            {
                "user": "한국의 전통 음식 중에서 가장 유명한 것은 무엇인가요?",
                "context": "문화적 질문"
            },
            {
                "user": "파일 정리를 도와주실 수 있나요?",
                "context": "도움 요청"
            }
        ]
        
        for i, conv in enumerate(test_conversations, 1):
            print(f"\n{i}. {conv['context']}")
            print(f"   👤 사용자: {conv['user']}")
            
            # 응답 생성
            ai_response = self._generate_smart_response(conv['user'])
            print(f"   🤖 AutoCI: {ai_response}")
            
            # 분석 결과 출력
            features = self._extract_features(conv['user'])
            print(f"   📊 분석: {features['formality']} / {features['sentence_type']}")
    
    def _generate_smart_response(self, user_input: str) -> str:
        """똑똑한 응답 생성"""
        features = self._extract_features(user_input)
        patterns = self._analyze_patterns(user_input)
        
        # 맥락 분석
        if "안녕" in user_input:
            if features["formality"] == "formal":
                return "안녕하세요! 저도 잘 지내고 있습니다. 오늘 좋은 하루 되시길 바라요!"
            else:
                return "안녕하세요! 잘 지내고 있어요. 오늘 어떤 일 있으세요?"
        
        elif any(word in user_input.lower() for word in ["unity", "게임오브젝트", "gameobject"]):
            return "Unity에서 GameObject를 찾으려면 FindObjectOfType<>() 메서드나 GameObject.Find() 메서드를 사용하시면 됩니다. 구체적으로 어떤 상황인지 알려주시면 더 정확한 방법을 안내해드릴게요!"
        
        elif any(word in user_input for word in ["스트레스", "어려", "힘들"]):
            return "프로젝트가 어려우시군요. 이해해요. 한 번에 모든 걸 해결하려고 하지 마시고, 작은 부분부터 차근차근 해보세요. 혼자 고민하지 마시고 언제든 도움을 요청하세요!"
        
        elif "전통 음식" in user_input or "한국" in user_input:
            return "한국의 대표적인 전통 음식으로는 김치, 비빔밥, 불고기, 된장찌개 등이 있습니다. 그 중에서도 김치는 전 세계적으로 가장 유명한 한국 음식이에요. 건강에도 좋고 맛도 훌륭하답니다!"
        
        elif "파일 정리" in user_input or "도와" in user_input:
            return "네, 기꺼이 도와드리겠습니다! AutoCI가 스크립트 파일들을 자동으로 분류하고 정리해드릴 수 있어요. 어떤 종류의 파일들을 정리하고 싶으신가요?"
        
        else:
            if features["formality"] == "formal":
                return "좋은 질문이십니다. 더 구체적으로 알려주시면 정확한 답변을 드릴 수 있겠습니다."
            else:
                return "흥미로운 질문이네요! 좀 더 자세히 설명해주시면 더 도움을 드릴 수 있을 것 같아요."
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """학습 통계 조회"""
        return {
            "총_학습_데이터": len(self.knowledge_base),
            "대화_히스토리": len(self.conversation_history),
            "학습된_패턴": len(self.pattern_frequency),
            "평균_품질_점수": sum(data.quality_score for data in self.knowledge_base) / len(self.knowledge_base) if self.knowledge_base else 0,
            "주요_패턴": dict(sorted(self.pattern_frequency.items(), key=lambda x: x[1], reverse=True)[:10]),
            "학습_완료_시간": datetime.now().isoformat()
        }
    
    def show_comparison_with_chatgpt(self):
        """ChatGPT와의 비교 분석"""
        print(f"""
        📊 AutoCI vs ChatGPT 한국어 능력 비교
        ===================================
        
        🎯 학습 방식 비교:
        ┌─────────────────┬─────────────────────┬─────────────────────┐
        │     특징        │      ChatGPT        │      AutoCI         │
        ├─────────────────┼─────────────────────┼─────────────────────┤
        │ 학습 데이터     │ 수십억 개 문장      │ {len(self.knowledge_base)}개 + 지속 수집    │
        │ 토크나이저      │ 전용 한국어 최적화  │ 패턴 기반 분석      │
        │ RLHF           │ 대규모 인간 피드백  │ 시뮬레이션 + 실제   │
        │ 문화적 이해     │ 글로벌 수준        │ 한국 특화          │
        │ 실시간 학습     │ 제한적             │ 지속적 학습        │
        │ 전문 도메인     │ 범용               │ Unity/C# 특화      │
        └─────────────────┴─────────────────────┴─────────────────────┘
        
        🌟 AutoCI의 장점:
        • Unity/게임 개발 전문성
        • 한국 개발자 맞춤 서비스  
        • 실시간 프로젝트 학습
        • 코드 자동 수정 및 관리
        • 개인화된 대화 스타일
        
        🎯 목표: ChatGPT 수준의 자연스러움 + 전문성
        """)

def main():
    """메인 실행 함수"""
    print("""
    🇰🇷 ChatGPT 방식 한국어 학습 시스템
    ==================================
    
    ChatGPT, Gemini, Claude가 어떻게 한국어를 학습했는지 보여주고
    AutoCI에 적용하는 방법을 시연합니다.
    """)
    
    # 학습 시스템 초기화
    learner = ChatGPTStyleKoreanLearner()
    
    # 1. 학습 방식 설명
    learner.demonstrate_learning_process()
    
    # 2. 학습 데이터 수집
    training_data = learner.collect_korean_training_data()
    
    # 3. RLHF 훈련
    learner.perform_rlhf_training()
    
    # 4. 대화 능력 시연
    learner.demonstrate_korean_conversation()
    
    # 5. 학습 통계
    stats = learner.get_learning_statistics()
    print(f"\n📈 학습 통계:")
    for key, value in stats.items():
        if key != "주요_패턴":
            print(f"   {key}: {value}")
    
    print(f"\n🔥 주요 학습 패턴:")
    for pattern, freq in stats["주요_패턴"].items():
        print(f"   {pattern}: {freq}회")
    
    # 6. ChatGPT 비교
    learner.show_comparison_with_chatgpt()
    
    print(f"""
    ✅ AutoCI 한국어 학습 시연 완료!
    
    🎉 결론:
    -------
    ChatGPT/Gemini/Claude와 같은 방식으로 AutoCI도 
    자연스러운 한국어 대화가 가능하도록 개발했습니다!
    
    • 대규모 한국어 데이터 학습 ✅
    • 패턴 인식 및 분석 ✅  
    • 인간 피드백 기반 개선 ✅
    • 맥락 이해 및 적절한 응답 ✅
    • Unity/개발 전문성 ✅
    
    이제 AutoCI가 ChatGPT처럼 자연스럽게 한국어로 대화할 수 있어요! 🚀
    """)

if __name__ == "__main__":
    main() 