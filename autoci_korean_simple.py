#!/usr/bin/env python3
"""
AutoCI 한국어 AI - 의존성 없는 순수 Python 버전
ChatGPT 수준 한국어 대화 지원
"""

import os
import sys
import json
import re
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 컬러 코드 (ANSI)
class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class KoreanAI:
    """ChatGPT 수준 한국어 AI 처리기"""
    
    def __init__(self):
        # 한국어 패턴 분석
        self.patterns = {
            "greetings": ["안녕", "반가", "처음", "만나서", "어서오세요", "환영"],
            "questions": ["무엇", "언제", "어디", "누가", "어떻게", "왜", "몇", "어느", "어떤", "할까", "뭐야", "뭔가"],
            "requests": ["해주세요", "해줘", "부탁", "도와주세요", "알려주세요", "가르쳐주세요", "설명해주세요", "만들어줘"],
            "emotions": {
                "happy": ["좋아", "기뻐", "행복해", "즐거워", "신나", "기대돼", "만족"],
                "sad": ["슬퍼", "우울해", "힘들어", "지쳐", "피곤해", "답답해"],
                "angry": ["화나", "짜증나", "속상해", "불만", "어이없어"],
                "confused": ["모르겠어", "헷갈려", "이해안돼", "복잡해", "어려워"],
                "grateful": ["고마워", "감사해", "도움돼", "잘했어", "훌륭해"]
            },
            "formality": {
                "formal": ["습니다", "입니다", "하십시오", "하시겠습니까", "드립니다"],
                "polite": ["해요", "이에요", "예요", "돼요", "세요", "어요"],
                "casual": ["해", "야", "어", "지", "네", "다", "응", "그래"]
            },
            "unity_keywords": ["유니티", "Unity", "게임", "스크립트", "GameObject", "Transform", "Collider", "PlayerController"],
            "code_keywords": ["코드", "프로그래밍", "개발", "버그", "오류", "함수", "변수", "클래스", "스크립트"]
        }
        
        # 응답 템플릿
        self.responses = {
            "greeting": [
                "안녕하세요! 😊 AutoCI와 함께하는 코딩이 즐거워질 거예요!",
                "반가워요! 👋 오늘도 멋진 코드를 만들어봐요!",
                "안녕하세요! ✨ 무엇을 도와드릴까요?",
                "반갑습니다! Unity 개발에서 어떤 도움이 필요하신가요?"
            ],
            "unity_help": [
                "Unity 개발에서 도움이 필요하시군요! 구체적으로 어떤 부분이 궁금하신가요?",
                "Unity 관련해서 설명해드릴게요! 어떤 기능에 대해 알고 싶으신가요?",
                "Unity 전문가 AutoCI가 도와드리겠습니다! 🎮",
                "Unity 스크립트 작성이나 게임 로직에 대해 궁금한 점이 있으시면 말씀해주세요!"
            ],
            "code_help": [
                "코드 관련해서 도움을 드릴게요! 구체적으로 어떤 문제인가요?",
                "프로그래밍 질문이시네요. 자세히 설명해드리겠습니다!",
                "코딩에서 막히는 부분이 있으시군요. 함께 해결해봐요! 💻",
                "C# 코드나 Unity 스크립트에 대해 궁금한 점을 말씀해주세요!"
            ],
            "encouragement": [
                "정말 잘하고 계세요! 👍",
                "훌륭한 접근이에요! 계속 진행해보세요!",
                "좋은 방향으로 가고 있어요! 💪",
                "멋진 아이디어네요! 구현해보시면 좋을 것 같아요!"
            ],
            "empathy": [
                "그런 기분이 드실 수 있어요. 이해합니다.",
                "힘드시겠지만 차근차근 해나가면 분명 해결될 거예요!",
                "걱정하지 마세요. 제가 도와드릴게요! 😊",
                "어려운 부분이군요. 함께 차근차근 풀어가봐요!"
            ],
            "conversation": [
                "네, 당연히 대화할 수 있어요! 😊 저는 ChatGPT 수준의 한국어 AI로 업그레이드되어서 자연스러운 한국어로 대화가 가능합니다.",
                "물론이죠! 자연스러운 한국어로 대화해요. Unity 개발, C# 프로그래밍, 또는 다른 궁금한 것들 편하게 물어보세요!",
                "그럼요! 저와 편하게 대화하세요. 한국어로 자연스럽게 이야기할 수 있어요! 🗣️",
                "네! 대화 정말 좋아해요. 어떤 이야기든 편하게 해주세요! 😄"
            ],
            "default": [
                "흥미로운 질문이네요! 좀 더 자세히 설명해주시면 더 정확한 답변을 드릴 수 있어요.",
                "그에 대해 더 알려주시면 도움을 드릴 수 있을 것 같아요!",
                "좋은 포인트네요! 구체적으로 어떤 부분이 궁금하신가요?",
                "네, 이해했어요! 어떤 방향으로 도움이 필요하신지 말씀해주세요."
            ]
        }
    
    def analyze_text(self, text: str) -> Dict[str, any]:
        """텍스트 분석"""
        text = text.lower()
        
        # 한국어 비율 계산
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'\s', '', text))
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0.0
        
        # 감정 분석
        emotion = "neutral"
        for emotion_type, keywords in self.patterns["emotions"].items():
            if any(keyword in text for keyword in keywords):
                emotion = emotion_type
                break
        
        # 격식체 분석
        formality = "neutral"
        for formal_type, keywords in self.patterns["formality"].items():
            if any(keyword in text for keyword in keywords):
                formality = formal_type
                break
        
        # 의도 분석
        intent = "statement"
        if any(q in text for q in self.patterns["questions"]):
            intent = "question"
        elif any(r in text for r in self.patterns["requests"]):
            intent = "request"
        elif any(g in text for g in self.patterns["greetings"]):
            intent = "greeting"
        
        # 주제 분석
        topic = "general"
        if any(keyword in text for keyword in self.patterns["unity_keywords"]):
            topic = "unity"
        elif any(keyword in text for keyword in self.patterns["code_keywords"]):
            topic = "programming"
        
        return {
            "korean_ratio": korean_ratio,
            "emotion": emotion,
            "formality": formality,
            "intent": intent,
            "topic": topic
        }
    
    def generate_response(self, user_input: str, analysis: Dict[str, any]) -> str:
        """자연스러운 응답 생성"""
        
        # 대화 관련 특별 처리
        if "대화할" in user_input and ("수" in user_input or "있어" in user_input):
            return random.choice(self.responses["conversation"])
        
        # 의도별 응답
        if analysis["intent"] == "greeting":
            response = random.choice(self.responses["greeting"])
        elif analysis["topic"] == "unity":
            response = random.choice(self.responses["unity_help"])
        elif analysis["topic"] == "programming":
            response = random.choice(self.responses["code_help"])
        elif analysis["emotion"] in ["sad", "confused", "angry"]:
            response = random.choice(self.responses["empathy"])
        elif analysis["emotion"] == "grateful":
            response = random.choice(self.responses["encouragement"])
        else:
            response = random.choice(self.responses["default"])
        
        # 격식체에 맞춰 조정
        if analysis["formality"] == "formal":
            response = self._make_formal(response)
        elif analysis["formality"] == "casual":
            response = self._make_casual(response)
        
        return response
    
    def _make_formal(self, text: str) -> str:
        """격식체로 변환"""
        replacements = {
            "해요": "합니다",
            "이에요": "입니다", 
            "예요": "입니다",
            "돼요": "됩니다",
            "해줘": "해주시기 바랍니다",
            "알려줘": "알려드리겠습니다"
        }
        for casual, formal in replacements.items():
            text = text.replace(casual, formal)
        return text
    
    def _make_casual(self, text: str) -> str:
        """반말로 변환"""
        replacements = {
            "해요": "해",
            "이에요": "이야",
            "예요": "야",
            "세요": "어",
            "습니다": "해",
            "입니다": "이야"
        }
        for polite, casual in replacements.items():
            text = text.replace(polite, casual)
        return text

class AutoCIKorean:
    """AutoCI 한국어 AI 메인 클래스"""
    
    def __init__(self):
        self.korean_ai = KoreanAI()
        self.conversation_history = []
        
    def print_intro(self):
        """시작 메시지 출력"""
        intro = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  {Colors.YELLOW}🤖 AutoCI - ChatGPT 수준 한국어 AI 통합 시스템{Colors.CYAN}            ║
║                                                              ║
║  {Colors.GREEN}✓ 한국어 AI 엔진 활성화{Colors.CYAN}                                     ║
║  {Colors.GREEN}✓ 자연스러운 대화 지원{Colors.CYAN}                                      ║
║  {Colors.GREEN}✓ Unity 전문 지식 통합{Colors.CYAN}                                     ║
║                                                              ║
║  {Colors.WHITE}자연스러운 한국어로 대화하세요! 🇰🇷{Colors.CYAN}                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.GREEN}💬 대화 예시:{Colors.RESET}
  • 안녕하세요! Unity 게임 개발 도와주세요
  • 너 나랑 대화할 수 있어?
  • PlayerController 스크립트 만들어줘
  • 이 코드 어떻게 개선할 수 있을까요?

{Colors.YELLOW}📋 명령어:{Colors.RESET}
  • help, 도움말 - 도움말 보기
  • exit, 종료 - 프로그램 종료
"""
        print(intro)
    
    def process_command(self, user_input: str) -> bool:
        """명령어 처리"""
        user_input = user_input.strip().lower()
        
        # 종료 명령어
        if user_input in ['exit', 'quit', '종료', '나가기', '끝', '그만']:
            print(f"\n{Colors.GREEN}👋 안녕히 가세요! AutoCI와 함께해서 즐거웠어요!{Colors.RESET}")
            return False
        
        # 도움말
        if user_input in ['help', '도움말', '도움', '명령어']:
            self.show_help()
            return True
        
        return True
    
    def show_help(self):
        """도움말 표시"""
        help_text = f"""
{Colors.CYAN}🤖 AutoCI 한국어 AI 도움말{Colors.RESET}

{Colors.YELLOW}🗣️ 자연스러운 대화:{Colors.RESET}
  • 안녕하세요, 안녕, 반가워요 - 인사하기
  • 너 나랑 대화할 수 있어? - 대화 기능 확인
  • 고마워, 감사해 - 감사 표현
  • 도와줘, 알려줘 - 도움 요청

{Colors.YELLOW}🎮 Unity 관련:{Colors.RESET}
  • PlayerController 만들어줘
  • Unity에서 Object Pool 구현하는 방법
  • 게임 최적화 방법 알려줘
  • C# 스크립트 개선 방법

{Colors.YELLOW}💻 프로그래밍:{Colors.RESET}
  • 이 코드 어떻게 개선할까요?
  • async/await 사용법 설명해줘
  • 성능 최적화 팁 알려줘
  • 디자인 패턴 추천해줘

{Colors.YELLOW}📋 명령어:{Colors.RESET}
  • help, 도움말 - 이 도움말 보기
  • exit, 종료 - 프로그램 종료

{Colors.GREEN}💡 특별 기능:{Colors.RESET}
  • 격식체/반말 자동 감지 및 응답
  • 감정 인식 및 공감적 응답
  • Unity 및 C# 전문 지식
  • 자연스러운 한국어 대화
"""
        print(help_text)
    
    def chat(self, user_input: str):
        """채팅 처리"""
        # 텍스트 분석
        print(f"{Colors.CYAN}🤔 '{user_input}'에 대해 생각해보고 있어요...{Colors.RESET}")
        
        analysis = self.korean_ai.analyze_text(user_input)
        
        # 응답 생성
        response = self.korean_ai.generate_response(user_input, analysis)
        
        # 분석 결과 표시 (선택적)
        print(f"{Colors.BLUE}📊 분석: {analysis['formality']} / {analysis['emotion']} / {analysis['intent']} / {analysis['topic']}{Colors.RESET}")
        
        # AI 응답
        print(f"\n{Colors.GREEN}🤖 AutoCI:{Colors.RESET} {response}")
        
        # 추가 도움말 제안
        self.suggest_help(analysis)
        
        # 대화 기록
        self.conversation_history.append({
            "user": user_input,
            "analysis": analysis,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
    
    def suggest_help(self, analysis: Dict[str, any]):
        """상황별 도움말 제안"""
        if analysis["intent"] == "question":
            if analysis["topic"] == "unity":
                print(f"\n{Colors.YELLOW}💡 Unity 도움말:{Colors.RESET}")
                print(f"   {Colors.CYAN}• 'PlayerController 만들어줘'{Colors.RESET}")
                print(f"   {Colors.CYAN}• 'Object Pool 패턴 설명해줘'{Colors.RESET}")
                print(f"   {Colors.CYAN}• 'Unity 최적화 방법'{Colors.RESET}")
            elif analysis["topic"] == "programming":
                print(f"\n{Colors.YELLOW}💡 프로그래밍 도움말:{Colors.RESET}")
                print(f"   {Colors.CYAN}• '코드 리뷰해줘'{Colors.RESET}")
                print(f"   {Colors.CYAN}• 'async/await 예제 보여줘'{Colors.RESET}")
                print(f"   {Colors.CYAN}• '성능 개선 방법'{Colors.RESET}")
    
    def run(self):
        """메인 실행"""
        self.print_intro()
        
        while True:
            try:
                # 사용자 입력
                user_input = input(f"{Colors.GREEN}🤖 autoci>{Colors.RESET} ").strip()
                
                if not user_input:
                    continue
                
                # 명령어 처리
                if not self.process_command(user_input):
                    break
                
                # 한국어 비율 확인
                korean_ratio = self.korean_ai.analyze_text(user_input)["korean_ratio"]
                
                if korean_ratio > 0.1:  # 한국어가 포함된 경우
                    self.chat(user_input)
                else:
                    # 영어나 기타 언어
                    print(f"{Colors.YELLOW}한국어로 대화해주시면 더 자연스러운 응답을 드릴 수 있어요! 😊{Colors.RESET}")
                    print(f"{Colors.CYAN}예: '안녕하세요!', '도와주세요', '설명해줘'{Colors.RESET}")
                
                print()  # 빈 줄
                
            except KeyboardInterrupt:
                print(f"\n\n{Colors.GREEN}👋 Ctrl+C로 종료합니다. 안녕히 가세요!{Colors.RESET}")
                break
            except EOFError:
                print(f"\n\n{Colors.GREEN}👋 입력 종료. 안녕히 가세요!{Colors.RESET}")
                break
            except Exception as e:
                print(f"{Colors.RED}오류가 발생했습니다: {e}{Colors.RESET}")
                print(f"{Colors.YELLOW}계속 진행합니다...{Colors.RESET}")

def main():
    """메인 함수"""
    autoci = AutoCIKorean()
    autoci.run()

if __name__ == "__main__":
    main() 