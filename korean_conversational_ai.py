#!/usr/bin/env python3
"""
AutoCI 한국어 대화형 AI 시스템
자연스러운 한국어 대화 처리 및 의도 분석 시스템
"""

import re
import json
import random
from datetime import datetime
from pathlib import Path
import subprocess
import sys

class KoreanConversationalAI:
    """한국어 대화형 AI 엔진"""
    
    def __init__(self):
        self.conversation_history = []
        self.user_context = {}
        self.load_knowledge_base()
        
    def load_knowledge_base(self):
        """한국어 지식 베이스 로드"""
        self.intents = {
            'greeting': {
                'patterns': [
                    r'안녕.*?', r'하이.*?', r'반가.*?', r'헬로.*?', r'좋은.*?아침', 
                    r'좋은.*?저녁', r'어서.*?와', r'처음.*?뵙.*?', r'만나.*?반가.*?'
                ],
                'responses': [
                    "안녕하세요! 😊 AutoCI와 함께 코딩 여행을 시작해볼까요?",
                    "반갑습니다! 🤗 오늘 어떤 멋진 코드를 만들어볼까요?",
                    "하이! 👋 코딩 마법사가 되어 함께 문제를 해결해봐요!",
                    "환영합니다! ✨ 궁금한 것이 있으면 편하게 물어보세요!"
                ]
            },
            'code_help': {
                'patterns': [
                    r'.*?코드.*?도와.*?', r'.*?프로그래밍.*?도움.*?', r'.*?버그.*?찾.*?',
                    r'.*?에러.*?해결.*?', r'.*?최적화.*?', r'.*?성능.*?개선.*?',
                    r'.*?유니티.*?스크립트.*?', r'.*?C#.*?', r'.*?오류.*?수정.*?'
                ],
                'responses': [
                    "코드 문제를 해결해드릴게요! 🔧 어떤 부분이 어려우신가요?",
                    "프로그래밍 도움을 드리겠습니다! 💡 구체적으로 어떤 문제인지 알려주세요.",
                    "버그 사냥을 시작해볼까요? 🐛 코드를 보여주시거나 문제를 설명해주세요!",
                    "최적화 전문가가 왔어요! ⚡ 성능 개선이 필요한 부분을 알려주세요."
                ]
            },
            'unity_specific': {
                'patterns': [
                    r'.*?유니티.*?', r'.*?Unity.*?', r'.*?게임.*?개발.*?', r'.*?MonoBehaviour.*?',
                    r'.*?GameObject.*?', r'.*?Transform.*?', r'.*?Coroutine.*?', r'.*?씬.*?'
                ],
                'responses': [
                    "Unity 전문가가 도와드릴게요! 🎮 어떤 게임 기능을 만들고 계신가요?",
                    "게임 개발이 즐거우시죠? 🕹️ Unity에서 어떤 것이 궁금하신가요?",
                    "Unity 마스터와 함께 멋진 게임을 만들어봐요! 🚀 구체적인 질문을 해주세요.",
                    "MonoBehaviour의 마법을 부려볼까요? ✨ 어떤 스크립트 작업이 필요한가요?"
                ]
            },
            'file_management': {
                'patterns': [
                    r'.*?파일.*?정리.*?', r'.*?폴더.*?정리.*?', r'.*?스크립트.*?이동.*?',
                    r'.*?Assets.*?정리.*?', r'.*?Scripts.*?폴더.*?', r'.*?정리.*?해.*?'
                ],
                'responses': [
                    "파일 정리의 마법사가 왔어요! 🗂️ 어떤 폴더를 정리하고 싶으신가요?",
                    "깔끔한 프로젝트 구조를 만들어드릴게요! 📁 정리가 필요한 부분을 알려주세요.",
                    "스크립트 정리 전문가입니다! 📝 Assets 폴더 구조를 최적화해드릴게요!",
                    "폴더 정리로 생산성을 높여봐요! 💪 어떤 파일들이 문제가 되고 있나요?"
                ]
            },
            'compliments': {
                'patterns': [
                    r'.*?잘.*?했.*?', r'.*?좋.*?', r'.*?훌륭.*?', r'.*?멋.*?', r'.*?완벽.*?',
                    r'.*?고마.*?', r'.*?감사.*?', r'.*?최고.*?', r'.*?대단.*?'
                ],
                'responses': [
                    "와! 칭찬해주셔서 감사해요! 😄 더 열심히 도와드릴게요!",
                    "고마워요! 🥰 함께 더 멋진 코드를 만들어봐요!",
                    "기뻐요! 😊 앞으로도 최선을 다해 도와드리겠습니다!",
                    "감사합니다! 🙏 여러분과 함께 작업하는 것이 즐거워요!"
                ]
            },
            'questions': {
                'patterns': [
                    r'.*?어떻게.*?', r'.*?왜.*?', r'.*?무엇.*?', r'.*?언제.*?', r'.*?어디.*?',
                    r'.*?누가.*?', r'.*?뭐.*?', r'.*?어느.*?', r'.*?\?', r'.*?？'
                ],
                'responses': [
                    "좋은 질문이에요! 🤔 더 구체적으로 설명해주시면 정확한 답변을 드릴게요!",
                    "궁금한 것이 많으시네요! 💭 어떤 부분이 가장 알고 싶으신가요?",
                    "질문의 답을 찾아보겠습니다! 🔍 조금 더 자세히 말씀해주실 수 있나요?",
                    "흥미로운 질문이네요! 🧐 관련된 코드나 상황을 알려주시면 더 도움이 될 거예요!"
                ]
            },
            'emotions': {
                'patterns': [
                    r'.*?힘들.*?', r'.*?어렵.*?', r'.*?모르겠.*?', r'.*?막혔.*?', r'.*?답답.*?',
                    r'.*?스트레스.*?', r'.*?짜증.*?', r'.*?포기.*?'
                ],
                'responses': [
                    "힘드시겠지만 함께라면 해결할 수 있어요! 💪 한 단계씩 차근차근 해봐요!",
                    "어려운 문제일수록 해결했을 때 성취감이 크죠! 😤 포기하지 말고 도전해봐요!",
                    "막힌 부분이 있으시군요! 🤝 제가 도와드릴게요. 문제를 같이 살펴볼까요?",
                    "프로그래밍은 원래 어려워요! 🎯 하지만 그래서 더 재미있는 거죠. 함께 해결해봐요!"
                ]
            }
        }
        
        self.code_keywords = {
            'unity': ['유니티', 'unity', '게임', 'monobehaviour', 'gameobject', 'transform'],
            'csharp': ['c#', 'csharp', '시샵', 'C샵', '클래스', 'class', '메소드', 'method'],
            'performance': ['성능', '최적화', '속도', '메모리', '퍼포먼스', 'performance', 'optimization'],
            'error': ['에러', '오류', '버그', 'error', 'bug', '문제', '안됨', '작동안함']
        }
    
    def analyze_intent(self, user_input: str) -> dict:
        """사용자 입력의 의도 분석"""
        user_input = user_input.strip().lower()
        
        # 감정 및 의도 점수
        intent_scores = {}
        
        for intent_name, intent_data in self.intents.items():
            score = 0
            for pattern in intent_data['patterns']:
                if re.search(pattern, user_input):
                    score += 1
            intent_scores[intent_name] = score
        
        # 키워드 기반 추가 분석
        detected_topics = []
        for topic, keywords in self.code_keywords.items():
            if any(keyword in user_input for keyword in keywords):
                detected_topics.append(topic)
        
        # 가장 높은 점수의 의도 선택
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        return {
            'intent': best_intent[0] if best_intent[1] > 0 else 'general',
            'confidence': best_intent[1],
            'topics': detected_topics,
            'user_input': user_input
        }
    
    def generate_response(self, intent_analysis: dict) -> str:
        """분석된 의도를 바탕으로 응답 생성"""
        intent = intent_analysis['intent']
        topics = intent_analysis['topics']
        user_input = intent_analysis['user_input']
        
        # 기본 응답 선택
        if intent in self.intents and self.intents[intent]['responses']:
            response = random.choice(self.intents[intent]['responses'])
        else:
            response = "흥미로운 말씀이네요! 🤔 더 자세히 설명해주시면 더 도움이 될 것 같아요!"
        
        # 토픽에 따른 추가 정보
        additional_info = []
        
        if 'unity' in topics:
            additional_info.append("\n🎮 Unity 관련 작업이시군요! 어떤 게임 기능을 개발하고 계신가요?")
            
        if 'csharp' in topics:
            additional_info.append("\n💻 C# 코딩 작업이네요! 구체적으로 어떤 부분이 필요하신가요?")
            
        if 'performance' in topics:
            additional_info.append("\n⚡ 성능 최적화는 제 전문 분야예요! 어떤 부분의 성능을 개선하고 싶으신가요?")
            
        if 'error' in topics:
            additional_info.append("\n🔧 오류 해결을 도와드릴게요! 에러 메시지나 문제 상황을 자세히 알려주세요!")
        
        # 명령어 추천
        command_suggestions = self.suggest_commands(intent, topics)
        if command_suggestions:
            additional_info.append(f"\n💡 추천 명령어: {command_suggestions}")
        
        return response + "".join(additional_info)
    
    def suggest_commands(self, intent: str, topics: list) -> str:
        """의도와 토픽에 따른 명령어 추천"""
        suggestions = []
        
        if intent == 'code_help' or 'error' in topics:
            suggestions.append("'분석' (코드 분석)")
            suggestions.append("'개선 <파일명>' (자동 개선)")
            
        if intent == 'unity_specific' or 'unity' in topics:
            suggestions.append("'정리' (Unity 스크립트 정리)")
            suggestions.append("'프로젝트 <경로>' (Unity 프로젝트 설정)")
            
        if intent == 'file_management':
            suggestions.append("'정리' (파일 정리)")
            suggestions.append("'검색 <키워드>' (파일 검색)")
        
        return ", ".join(suggestions) if suggestions else ""
    
    def chat(self, user_input: str) -> str:
        """메인 대화 처리 함수"""
        # 대화 기록 저장
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'type': 'user'
        })
        
        # 의도 분석
        intent_analysis = self.analyze_intent(user_input)
        
        # 응답 생성
        response = self.generate_response(intent_analysis)
        
        # 응답 기록 저장
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'ai_response': response,
            'intent': intent_analysis,
            'type': 'ai'
        })
        
        return response
    
    def get_conversation_summary(self) -> str:
        """대화 요약 제공"""
        if not self.conversation_history:
            return "아직 대화가 시작되지 않았습니다."
        
        total_messages = len([msg for msg in self.conversation_history if msg['type'] == 'user'])
        
        recent_topics = []
        for msg in self.conversation_history[-6:]:  # 최근 6개 메시지
            if msg['type'] == 'ai' and 'intent' in msg:
                intent_info = msg['intent']
                if intent_info['topics']:
                    recent_topics.extend(intent_info['topics'])
        
        unique_topics = list(set(recent_topics))
        
        summary = f"📊 대화 요약:\n"
        summary += f"   💬 총 메시지: {total_messages}개\n"
        summary += f"   🏷️ 주요 주제: {', '.join(unique_topics) if unique_topics else '일반 대화'}\n"
        summary += f"   ⏰ 마지막 대화: {self.conversation_history[-1]['timestamp'][:19]}"
        
        return summary


def main():
    """메인 함수 - 대화형 AI 실행"""
    ai = KoreanConversationalAI()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║  🤖 AutoCI 한국어 대화형 AI 시스템 v2.0                          ║
║                                                                  ║
║  ✨ 자연스러운 한국어 대화 처리                                   ║
║  🧠 의도 분석 및 맥락 이해                                        ║
║  🎯 Unity & C# 전문 지원                                         ║
║                                                                  ║
║  💬 자유롭게 대화해보세요!                                        ║
║     (종료하려면 '종료', '바이바이', 또는 'exit' 입력)               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    while True:
        try:
            user_input = input("\n👤 당신: ").strip()
            
            if not user_input:
                continue
                
            # 종료 명령어 처리
            exit_commands = ['종료', '바이바이', '안녕히', '그만', 'exit', 'quit', 'bye']
            if any(cmd in user_input.lower() for cmd in exit_commands):
                print("\n🤖 AI: 안녕히 가세요! 😊 언제든 다시 오세요!")
                print(ai.get_conversation_summary())
                break
            
            # 특별 명령어 처리
            if user_input.lower() == '요약':
                print(f"\n🤖 AI: {ai.get_conversation_summary()}")
                continue
                
            if user_input.lower() == '도움말':
                print("""
🤖 AI: 저와 자연스럽게 대화하세요! 

💬 대화 예시:
   "안녕하세요!" - 인사하기
   "Unity 스크립트 정리해줘" - 작업 요청
   "코드에 버그가 있어" - 문제 상담
   "C# 성능 최적화 방법 알려줘" - 기술 질문
   "파일 정리가 필요해" - 파일 관리

🎯 전문 분야:
   • Unity 게임 개발
   • C# 프로그래밍
   • 코드 최적화
   • 파일 정리
   • 버그 해결

📋 특별 명령어:
   '요약' - 대화 요약 보기
   '종료' - 프로그램 종료
""")
                continue
            
            # AI 응답 생성
            response = ai.chat(user_input)
            print(f"\n🤖 AI: {response}")
            
        except KeyboardInterrupt:
            print("\n\n🤖 AI: 안녕히 가세요! 👋")
            print(ai.get_conversation_summary())
            break
        except Exception as e:
            print(f"\n❌ 오류가 발생했습니다: {e}")
            print("💡 다시 시도해주세요!")


if __name__ == "__main__":
    main() 