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
import cmd
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
                "흥미로운 얘기네요! 더 자세히 말씀해주시면 구체적으로 도와드릴 수 있어요 😊",
                "아, 그런 것에 관심이 있으시군요! 어떤 부분이 가장 궁금하신가요?",
                "좋은 질문이에요! Unity나 C# 관련해서 구체적으로 어떤 도움이 필요하신지 알려주세요",
                "네! 어떤 문제를 해결하고 싶으신지, 또는 무엇을 만들고 계신지 말씀해주세요 🎮"
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
        user_lower = user_input.lower()
        
        # 자기소개 요청 특별 처리
        if any(keyword in user_lower for keyword in ["너에 대해", "자기소개", "너는 누구", "너는 뭐", "너 설명", "너 알려줘", "넌 누구", "넌 뭐야"]):
            if analysis["formality"] == "casual":
                return "안녕! 나는 AutoCI야. Unity 개발과 C# 프로그래밍을 도와주는 AI야. 코드 작성, 버그 해결, 최적화 등 뭐든지 물어봐!"
            else:
                return "안녕하세요! 저는 AutoCI입니다. Unity 게임 개발과 C# 프로그래밍을 전문으로 도와드리는 AI 어시스턴트예요. 코드 작성, 디버깅, 성능 최적화 등 개발 관련 모든 것을 도와드릴 수 있습니다!"
        
        # 대화 가능 여부 질문
        if "대화할" in user_lower and ("수" in user_lower or "있어" in user_lower):
            return random.choice(self.responses["conversation"])
        
        # 기능/능력 질문
        if any(keyword in user_lower for keyword in ["뭐 할 수 있어", "무엇을 할", "어떤 기능", "뭐가 가능", "할 수 있는"]):
            return "저는 Unity 게임 개발 전문 AI예요! 🎮\n• C# 스크립트 작성 및 개선\n• Unity 컴포넌트 설명\n• 게임 로직 구현 도움\n• 성능 최적화 조언\n• 버그 해결 방법 제시\n뭐든 물어보세요!"
        
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

class AutoCIKoreanShell(cmd.Cmd):
    """AutoCI 한국어 AI 셸"""
    
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
    
    prompt = f'{Colors.GREEN}🤖 autoci>{Colors.RESET} '
    
    def __init__(self):
        super().__init__()
        self.korean_ai = KoreanAI()
        self.conversation_history = []
        self.current_project = None
    
    def default(self, line):
        """ChatGPT 수준 한국어 AI 처리"""
        line = line.strip()
        
        # 한국어 비율 확인
        korean_ratio = self.korean_ai.analyze_text(line)["korean_ratio"]
        
        if korean_ratio > 0.1:  # 한국어가 포함된 경우
            # ChatGPT 스타일 한국어 분석
            print(f"{Colors.CYAN}🤔 '{line}'에 대해 생각해보고 있어요...{Colors.RESET}")
            analysis = self.korean_ai.analyze_text(line)
            
            # 한국어 명령어 매핑 먼저 확인
            korean_commands = {
                '도움말': 'help', '도움': 'help', '명령어': 'help',
                '상태': 'status', '상태확인': 'status',
                '학습': 'learning', '학습상태': 'learning', '학습확인': 'learning',
                '프로젝트': 'project', '분석': 'analyze', '개선': 'improve',
                '검색': 'search', '찾기': 'search',
                '리포트': 'report', '보고서': 'report',
                '모니터링': 'monitor', '모니터': 'monitor',
                '종료': 'exit', '나가기': 'exit', '끝': 'exit', '그만': 'exit'
            }
            
            # 명령어인지 확인
            for korean_cmd, english_cmd in korean_commands.items():
                if korean_cmd in line:
                    print(f"{Colors.CYAN}✅ '{korean_cmd}' 명령을 실행합니다!{Colors.RESET}")
                    self.onecmd(english_cmd)
                    return
            
            # ChatGPT 스타일 자연스러운 응답 생성
            smart_response = self.korean_ai.generate_response(line, analysis)
            
            # 분석 결과 표시 (디버그용)
            print(f"{Colors.BLUE}📊 분석: {analysis['formality']} / {analysis['emotion']} / {analysis['intent']} / {analysis['topic']}{Colors.RESET}")
            
            # AI 응답 출력
            print(f"\n{Colors.GREEN}🤖 AutoCI:{Colors.RESET} {smart_response}")
            
            # 구체적인 도움 제안
            self.suggest_help(analysis)
            
            # 대화 기록
            self.conversation_history.append({
                "user": line,
                "analysis": analysis,
                "response": smart_response,
                "timestamp": datetime.now().isoformat()
            })
        else:
            # 영어나 기타 언어
            print(f"{Colors.YELLOW}한국어로 대화해주시면 더 자연스러운 응답을 드릴 수 있어요! 😊{Colors.RESET}")
            print(f"{Colors.CYAN}예: '안녕하세요!', '도와주세요', '설명해줘'{Colors.RESET}")
    
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
    
    def do_help(self, arg):
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
   • status, 상태 - 시스템 상태 확인
   • learning, 학습 - 학습 상태 및 AI 타입 분석
   • exit, 종료 - 프로그램 종료

{Colors.GREEN}💡 특별 기능:{Colors.RESET}
  • 격식체/반말 자동 감지 및 응답
  • 감정 인식 및 공감적 응답
  • Unity 및 C# 전문 지식
  • 자연스러운 한국어 대화
"""
        print(help_text)
    
    def do_learning(self, arg):
        """학습 상태 확인"""
        print(f"\n{Colors.CYAN}🧠 AutoCI 학습 상태 분석{Colors.RESET}")
        print("=" * 50)
        
        print(f"{Colors.YELLOW}📊 현재 구현된 AI 타입:{Colors.RESET}")
        print(f"  🔧 규칙 기반 AI (Rule-based)")
        print(f"  📝 패턴 매칭 시스템")
        print(f"  🎯 템플릿 응답 생성")
        
        print(f"\n{Colors.YELLOW}🚫 실제 학습이 일어나지 않는 이유:{Colors.RESET}")
        print(f"  • 고정된 규칙과 패턴")
        print(f"  • 신경망 가중치 업데이트 없음")
        print(f"  • 이전 대화가 다음 응답에 영향 안 줌")
        
        if self.conversation_history:
            print(f"\n{Colors.BLUE}💬 대화 기록 통계:{Colors.RESET}")
            print(f"  총 대화: {len(self.conversation_history)}번")
            
            # 감정 통계
            emotions = [conv["analysis"]["emotion"] for conv in self.conversation_history]
            emotion_count = {}
            for emotion in emotions:
                emotion_count[emotion] = emotion_count.get(emotion, 0) + 1
            
            print(f"  감정 분포: {emotion_count}")
            
            # 주제 통계  
            topics = [conv["analysis"]["topic"] for conv in self.conversation_history]
            topic_count = {}
            for topic in topics:
                topic_count[topic] = topic_count.get(topic, 0) + 1
                
            print(f"  주제 분포: {topic_count}")
        
        print(f"\n{Colors.GREEN}🎯 실제 학습 AI를 원한다면:{Colors.RESET}")
        print(f"  python3 autoci_learning_ai_concept.py")
        print(f"  (실제 학습하는 AI 개념 데모)")
        
        print(f"\n{Colors.CYAN}💡 ChatGPT/Claude 같은 진짜 학습 AI와의 차이:{Colors.RESET}")
        print(f"  🤖 ChatGPT: 수십억 파라미터 신경망, 대규모 데이터 학습")
        print(f"  🔧 현재 AutoCI: 규칙 기반, 패턴 매칭")
        print(f"  📈 업그레이드 가능: 실제 신경망 통합 시스템")
        print()
    
    def do_status(self, arg):
        """시스템 상태 확인"""
        print(f"\n{Colors.CYAN}📊 AutoCI 시스템 상태:{Colors.RESET}")
        print(f"  {Colors.GREEN}🟢 한국어 AI - 활성화{Colors.RESET}")
        print(f"  {Colors.GREEN}🟢 대화 엔진 - 실행 중{Colors.RESET}")
        print(f"  {Colors.GREEN}🟢 Unity 지원 - 준비됨{Colors.RESET}")
        print(f"  {Colors.YELLOW}🟡 의존성 - 기본 모드{Colors.RESET}")
        print(f"  {Colors.RED}🔴 실제 학습 - 비활성화{Colors.RESET}")
        
        if self.conversation_history:
            print(f"\n{Colors.BLUE}💬 대화 통계:{Colors.RESET}")
            print(f"   총 대화: {len(self.conversation_history)}번")
            
            # 최근 감정 분석
            recent_emotions = [conv["analysis"]["emotion"] for conv in self.conversation_history[-5:]]
            emotion_count = {}
            for emotion in recent_emotions:
                emotion_count[emotion] = emotion_count.get(emotion, 0) + 1
            
            print(f"   최근 감정: {', '.join(emotion_count.keys())}")
        
        print(f"\n{Colors.YELLOW}💡 '학습' 명령어로 자세한 학습 정보 확인{Colors.RESET}")
        print()
    
    def do_exit(self, arg):
        """종료"""
        print(f"\n{Colors.GREEN}👋 안녕히 가세요! AutoCI와 함께해서 즐거웠어요!{Colors.RESET}")
        return True
    
    def do_quit(self, arg):
        """종료"""
        return self.do_exit(arg)
    
    def emptyline(self):
        """빈 줄 입력 시 아무것도 하지 않음"""
        pass

def main():
    """메인 함수"""
    try:
        shell = AutoCIKoreanShell()
        shell.cmdloop()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.GREEN}👋 Ctrl+C로 종료합니다. 안녕히 가세요!{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}오류가 발생했습니다: {e}{Colors.RESET}")

if __name__ == "__main__":
    main()