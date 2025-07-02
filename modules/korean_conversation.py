#!/usr/bin/env python3
"""
AutoCI 한글 대화 시스템
사용자와 자연스러운 한글 대화를 통해 AutoCI가 학습하고 발전
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('korean_conversation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """대화 턴 데이터"""
    turn_id: str
    user_message: str
    ai_response: str
    timestamp: datetime
    context: Dict[str, Any]
    intent: Optional[str] = None  # 의도 분류
    entities: Optional[List[str]] = None  # 추출된 엔티티

class KoreanConversationSystem:
    """한글 대화 시스템"""
    
    def __init__(self):
        self.conversation_history = []
        self.current_context = {}
        
        # 대화 의도 분류
        self.intent_patterns = {
            "질문": ["어떻게", "무엇", "뭐", "왜", "언제", "어디", "누가", "?"],
            "요청": ["해줘", "해주세요", "만들어", "생성", "실행", "보여줘", "알려줘"],
            "피드백": ["좋아", "나빠", "잘못", "틀렸", "맞아", "고마워", "감사"],
            "설명": ["설명", "알려", "가르쳐", "이해", "모르겠"],
            "명령": ["시작", "중지", "실행", "정지", "빌드", "테스트"],
            "대화": ["안녕", "반가워", "고마워", "미안", "괜찮아"]
        }
        
        # 엔티티 추출 패턴
        self.entity_patterns = {
            "godot": ["고도", "godot", "고돗", "엔진"],
            "csharp": ["씨샵", "c#", "csharp", "시샵"],
            "network": ["네트워크", "멀티플레이어", "서버", "클라이언트", "동기화"],
            "nakama": ["나카마", "nakama", "백엔드"],
            "ai": ["ai", "인공지능", "모델", "학습"],
            "build": ["빌드", "컴파일", "생성", "만들기"],
            "error": ["오류", "에러", "버그", "문제", "안됨", "안돼"]
        }
        
        # 대화 상태 관리
        self.conversation_state = {
            "topic": None,  # 현재 대화 주제
            "mood": "neutral",  # 대화 분위기
            "user_satisfaction": 0.5,  # 사용자 만족도
            "question_count": 0,  # 질문 횟수
            "command_count": 0,  # 명령 횟수
        }
        
        # 자연스러운 응답 템플릿
        self.response_templates = {
            "greeting": [
                "안녕하세요! AutoCI입니다. 무엇을 도와드릴까요? 😊",
                "반갑습니다! 오늘은 어떤 게임을 만들어볼까요?",
                "안녕하세요! Godot 개발에 대해 궁금한 점이 있으신가요?"
            ],
            "question_acknowledge": [
                "좋은 질문이네요! {topic}에 대해 설명드리겠습니다.",
                "아, {topic} 말씀이시군요. 제가 알려드릴게요.",
                "{topic}에 대해 궁금하신 거군요! 바로 답변드리겠습니다."
            ],
            "command_acknowledge": [
                "네, {action}을(를) 실행하겠습니다.",
                "알겠습니다! {action} 작업을 시작할게요.",
                "{action}을(를) 진행하겠습니다. 잠시만 기다려주세요."
            ],
            "feedback_positive": [
                "도움이 되어서 기쁩니다! 😊",
                "감사합니다! 더 나은 답변을 위해 계속 학습하겠습니다.",
                "좋은 피드백 감사합니다! 다른 질문도 편하게 해주세요."
            ],
            "feedback_negative": [
                "죄송합니다. 더 정확한 답변을 드리도록 노력하겠습니다.",
                "아직 부족한 부분이 있네요. 계속 학습해서 개선하겠습니다.",
                "피드백 감사합니다. 어떤 부분이 부족했는지 알려주시면 더 도움이 될 것 같아요."
            ],
            "clarification": [
                "혹시 {options} 중 어떤 것을 말씀하시는 건가요?",
                "좀 더 구체적으로 설명해주시면 정확히 도와드릴 수 있을 것 같아요.",
                "{topic}의 어떤 부분이 궁금하신가요?"
            ]
        }
        
        logger.info("한글 대화 시스템이 초기화되었습니다.")
    
    async def process_user_input(self, user_input: str, 
                               evolution_system=None) -> str:
        """사용자 입력 처리 및 응답 생성"""
        # 의도 분류
        intent = self._classify_intent(user_input)
        
        # 엔티티 추출
        entities = self._extract_entities(user_input)
        
        # 컨텍스트 업데이트
        self._update_context(intent, entities)
        
        # 응답 생성
        response = await self._generate_response(user_input, intent, entities)
        
        # 대화 턴 기록
        turn = ConversationTurn(
            turn_id=self._generate_turn_id(),
            user_message=user_input,
            ai_response=response,
            timestamp=datetime.now(),
            context=self.current_context.copy(),
            intent=intent,
            entities=entities
        )
        self.conversation_history.append(turn)
        
        # 자가 진화 시스템과 연동
        if evolution_system:
            # 질문을 자가 진화 시스템에 전달
            context = {
                "user_id": "korean_conversation",
                "language": "korean",
                "intent": intent,
                "entities": entities,
                "conversation_state": self.conversation_state.copy()
            }
            
            # 진화 시스템에서 더 나은 응답 생성
            evolved_response, response_id = await evolution_system.process_user_question(
                user_input, context
            )
            
            # 진화된 응답이 더 좋다면 사용
            if len(evolved_response) > len(response):
                response = evolved_response
                turn.ai_response = response
        
        # 대화 상태 업데이트
        self._update_conversation_state(intent, user_input, response)
        
        return response
    
    def _classify_intent(self, text: str) -> str:
        """의도 분류"""
        text_lower = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return intent
        
        return "대화"  # 기본값
    
    def _extract_entities(self, text: str) -> List[str]:
        """엔티티 추출"""
        text_lower = text.lower()
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                entities.append(entity_type)
        
        return entities
    
    def _update_context(self, intent: str, entities: List[str]):
        """컨텍스트 업데이트"""
        self.current_context["last_intent"] = intent
        self.current_context["last_entities"] = entities
        self.current_context["timestamp"] = datetime.now().isoformat()
        
        # 주제 추론
        if entities:
            self.conversation_state["topic"] = entities[0]
    
    async def _generate_response(self, user_input: str, 
                               intent: str, entities: List[str]) -> str:
        """응답 생성"""
        # 정보 수집기 임포트
        try:
            from modules.intelligent_information_gatherer import get_information_gatherer
            gatherer = get_information_gatherer()
            information_available = True
        except ImportError:
            information_available = False

        # 기본 응답 전략
        if intent == "대화":
            if any(greeting in user_input.lower() for greeting in ["안녕", "반가", "하이"]):
                return self._select_template("greeting")
        
        elif intent == "질문":
            topic = entities[0] if entities else "그것"
            response = self._select_template("question_acknowledge", topic=topic)
            
            # 정보 수집기를 사용하여 웹에서 답변 검색
            if information_available and topic in ["csharp", "godot"]:
                web_results = await gatherer.search_web_for_code(f"{topic} {user_input}")
                if web_results:
                    answer = "웹에서 다음과 같은 정보를 찾았습니다:\n\n"
                    for result in web_results:
                        answer += f"- 소스: {result['source']}\n"
                        answer += f"  설명: {result['explanation']}\n\n"
                    return answer

            # 기존 답변 생성 로직
            answer = await self._generate_answer(user_input, entities)
            return f"{response}\n\n{answer}"
        
        elif intent == "요청":
            action = self._extract_action(user_input)
            response = self._select_template("command_acknowledge", action=action)
            
            # 실제 명령 실행 (여기서는 시뮬레이션)
            result = await self._execute_command(action)
            return f"{response}\n\n{result}"
        
        elif intent == "피드백":
            if any(positive in user_input.lower() for positive in ["좋아", "맞아", "고마워"]):
                return self._select_template("feedback_positive")
            else:
                return self._select_template("feedback_negative")
        
        elif intent == "설명":
            if entities:
                topic = entities[0]
                return self._select_template("clarification", topic=topic)
        
        # 기본 응답
        return "네, 이해했습니다. 더 자세히 설명해주시면 더 정확히 도와드릴 수 있을 것 같아요."
    
    def _select_template(self, template_type: str, **kwargs) -> str:
        """템플릿 선택 및 포맷팅"""
        import random
        templates = self.response_templates.get(template_type, ["이해했습니다."])
        template = random.choice(templates)
        
        # 변수 치환
        for key, value in kwargs.items():
            template = template.replace(f"{{{key}}}", str(value))
        
        return template
    
    async def _generate_answer(self, question: str, entities: List[str]) -> str:
        """실제 답변 생성 (시뮬레이션)"""
        # 엔티티별 기본 답변
        answers = {
            "godot": "Godot은 오픈소스 게임 엔진으로, 2D와 3D 게임을 모두 개발할 수 있습니다. C#과 GDScript를 지원하며, 노드 기반의 씬 시스템을 사용합니다.",
            "csharp": "C#은 마이크로소프트에서 개발한 객체지향 프로그래밍 언어입니다. Godot 4.0부터 .NET 6를 지원하여 더욱 강력해졌습니다.",
            "network": "Godot의 내장 네트워킹은 MultiplayerAPI를 통해 쉽게 멀티플레이어 게임을 만들 수 있게 해줍니다. RPC와 자동 동기화를 지원합니다.",
            "nakama": "Nakama는 오픈소스 게임 서버로, 매치메이킹, 리더보드, 채팅 등의 기능을 제공합니다. Godot과 완벽하게 통합됩니다.",
            "ai": "AutoCI는 AI 모델을 활용하여 24시간 자동으로 게임을 개발할 수 있습니다. 사용자의 질문을 통해 계속 학습하고 발전합니다.",
            "build": "Godot 빌드는 `build-godot` 명령어로 쉽게 할 수 있습니다. Windows와 Linux 버전을 모두 지원합니다.",
            "error": "오류가 발생했다면 먼저 로그를 확인해보세요. 대부분의 문제는 종속성이나 경로 문제일 가능성이 높습니다."
        }
        
        # 엔티티에 맞는 답변 선택
        for entity in entities:
            if entity in answers:
                return answers[entity]
        
        # 기본 답변
        return "해당 주제에 대해 더 구체적인 정보가 필요합니다. 어떤 부분이 궁금하신가요?"
    
    def _extract_action(self, text: str) -> str:
        """명령어 추출"""
        actions = {
            "빌드": ["빌드", "컴파일", "만들어"],
            "실행": ["실행", "시작", "돌려"],
            "테스트": ["테스트", "검사", "확인"],
            "학습": ["학습", "공부", "배워"],
            "생성": ["생성", "만들기", "제작"]
        }
        
        text_lower = text.lower()
        for action, keywords in actions.items():
            if any(keyword in text_lower for keyword in keywords):
                return action
        
        return "작업"
    
    async def _execute_command(self, action: str) -> str:
        """명령 실행 (시뮬레이션)"""
        results = {
            "빌드": "✅ Godot 엔진 빌드가 시작되었습니다. 약 10-15분 정도 소요됩니다.",
            "실행": "🚀 AutoCI가 실행되었습니다. 24시간 자동 개발이 시작됩니다.",
            "테스트": "🧪 테스트를 실행 중입니다... 모든 테스트가 통과했습니다!",
            "학습": "📚 AI 학습 모드가 시작되었습니다. 5가지 핵심 주제를 학습합니다.",
            "생성": "🎮 새로운 게임 프로젝트를 생성했습니다."
        }
        
        await asyncio.sleep(1)  # 실행 시뮬레이션
        return results.get(action, "✅ 작업이 완료되었습니다.")
    
    def _update_conversation_state(self, intent: str, user_input: str, response: str):
        """대화 상태 업데이트"""
        # 의도별 카운트
        if intent == "질문":
            self.conversation_state["question_count"] += 1
        elif intent == "명령":
            self.conversation_state["command_count"] += 1
        
        # 사용자 만족도 추정 (간단한 휴리스틱)
        positive_signals = ["고마워", "좋아", "잘", "완벽", "최고"]
        negative_signals = ["안돼", "틀렸", "아니", "별로", "나쁘"]
        
        user_lower = user_input.lower()
        if any(signal in user_lower for signal in positive_signals):
            self.conversation_state["user_satisfaction"] = min(1.0, 
                self.conversation_state["user_satisfaction"] + 0.1)
        elif any(signal in user_lower for signal in negative_signals):
            self.conversation_state["user_satisfaction"] = max(0.0, 
                self.conversation_state["user_satisfaction"] - 0.1)
        
        # 분위기 업데이트
        if self.conversation_state["user_satisfaction"] > 0.7:
            self.conversation_state["mood"] = "positive"
        elif self.conversation_state["user_satisfaction"] < 0.3:
            self.conversation_state["mood"] = "negative"
        else:
            self.conversation_state["mood"] = "neutral"
    
    def _generate_turn_id(self) -> str:
        """대화 턴 ID 생성"""
        import hashlib
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"turn_{timestamp}_{random_part}"
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """대화 요약 정보"""
        if not self.conversation_history:
            return {
                "total_turns": 0,
                "topics_discussed": [],
                "user_satisfaction": 0.5,
                "most_common_intent": None
            }
        
        # 주제 통계
        topics = {}
        intents = {}
        
        for turn in self.conversation_history:
            # 주제 카운트
            for entity in turn.entities or []:
                topics[entity] = topics.get(entity, 0) + 1
            
            # 의도 카운트
            if turn.intent:
                intents[turn.intent] = intents.get(turn.intent, 0) + 1
        
        # 가장 많이 논의된 주제
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # 가장 흔한 의도
        most_common_intent = max(intents.items(), key=lambda x: x[1])[0] if intents else None
        
        return {
            "total_turns": len(self.conversation_history),
            "topics_discussed": [topic for topic, _ in top_topics],
            "user_satisfaction": self.conversation_state["user_satisfaction"],
            "most_common_intent": most_common_intent,
            "question_count": self.conversation_state["question_count"],
            "command_count": self.conversation_state["command_count"],
            "mood": self.conversation_state["mood"]
        }
    
    async def save_conversation(self, filepath: str):
        """대화 저장"""
        data = {
            "conversation_history": [asdict(turn) for turn in self.conversation_history],
            "conversation_state": self.conversation_state,
            "summary": self.get_conversation_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"대화가 {filepath}에 저장되었습니다.")
    
    async def load_conversation(self, filepath: str):
        """대화 불러오기"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 대화 기록 복원
        self.conversation_history = []
        for turn_data in data.get("conversation_history", []):
            turn = ConversationTurn(
                turn_id=turn_data["turn_id"],
                user_message=turn_data["user_message"],
                ai_response=turn_data["ai_response"],
                timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                context=turn_data["context"],
                intent=turn_data.get("intent"),
                entities=turn_data.get("entities")
            )
            self.conversation_history.append(turn)
        
        # 대화 상태 복원
        self.conversation_state = data.get("conversation_state", self.conversation_state)
        
        logger.info(f"대화가 {filepath}에서 불러와졌습니다.")


# 전역 인스턴스
_korean_conversation = None

def get_korean_conversation() -> KoreanConversationSystem:
    """한글 대화 시스템 싱글톤 인스턴스 반환"""
    global _korean_conversation
    if _korean_conversation is None:
        _korean_conversation = KoreanConversationSystem()
    return _korean_conversation


async def interactive_conversation():
    """대화형 한글 인터페이스"""
    from modules.self_evolution_system import get_evolution_system
    
    conversation = get_korean_conversation()
    evolution = get_evolution_system()
    
    print("🤖 AutoCI 한글 대화 시스템")
    print("=" * 50)
    print("안녕하세요! AutoCI입니다. 무엇을 도와드릴까요?")
    print("(종료하려면 '종료' 또는 'exit'를 입력하세요)")
    print("=" * 50)
    
    while True:
        try:
            # 사용자 입력
            user_input = input("\n👤 사용자: ").strip()
            
            # 종료 조건
            if user_input.lower() in ["종료", "exit", "quit", "bye"]:
                print("\n🤖 AutoCI: 대화를 종료합니다. 감사합니다! 👋")
                
                # 대화 요약
                summary = conversation.get_conversation_summary()
                print("\n📊 대화 요약:")
                print(f"  • 총 대화 수: {summary['total_turns']}")
                print(f"  • 논의된 주제: {', '.join(summary['topics_discussed'])}")
                print(f"  • 사용자 만족도: {summary['user_satisfaction']:.1%}")
                
                # 대화 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                await conversation.save_conversation(f"conversation_{timestamp}.json")
                break
            
            # 응답 생성
            print("\n🤖 AutoCI: ", end="", flush=True)
            response = await conversation.process_user_input(user_input, evolution)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n대화가 중단되었습니다.")
            break
        except Exception as e:
            print(f"\n오류가 발생했습니다: {str(e)}")
            logger.error(f"대화 처리 중 오류: {str(e)}")


if __name__ == "__main__":
    # 테스트용 대화형 인터페이스 실행
    asyncio.run(interactive_conversation())