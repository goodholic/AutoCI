"""
한글 대화 인터페이스
자연스러운 한국어로 AI와 대화하며 게임 개발
"""

import re
import json
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import os

# AI 모델 및 게임 개발 모듈
from .ai_model_integration import get_ai_integration
from .godot_automation_controller import GodotAutomationController
from .game_development_pipeline import GameDevelopmentPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentType(Enum):
    """사용자 의도 분류"""
    CREATE_GAME = "create_game"         # 게임 생성
    MODIFY_GAME = "modify_game"         # 게임 수정
    ADD_FEATURE = "add_feature"         # 기능 추가
    ASK_QUESTION = "ask_question"       # 질문
    GIVE_FEEDBACK = "give_feedback"     # 피드백
    STATUS_CHECK = "status_check"       # 상태 확인
    HELP = "help"                       # 도움말
    UNKNOWN = "unknown"                 # 알 수 없음


@dataclass
class ConversationContext:
    """대화 컨텍스트"""
    current_game: Optional[str] = None
    current_phase: Optional[str] = None
    history: List[Dict[str, str]] = None
    user_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
        if self.user_preferences is None:
            self.user_preferences = {}


class KoreanConversationInterface:
    """한글 대화 인터페이스"""
    
    def __init__(self):
        self.ai_model = get_ai_integration()
        self.godot_controller = GodotAutomationController()
        self.game_pipeline = GameDevelopmentPipeline()
        
        self.context = ConversationContext()
        self.is_active = False
        
        # 의도 패턴 정의
        self.intent_patterns = self._initialize_intent_patterns()
        
        # 응답 템플릿
        self.response_templates = self._initialize_response_templates()
        
    def _initialize_intent_patterns(self) -> Dict[IntentType, List[re.Pattern]]:
        """의도 인식 패턴 초기화"""
        return {
            IntentType.CREATE_GAME: [
                re.compile(r"(게임|game).*(만들|생성|제작|개발)", re.IGNORECASE),
                re.compile(r"(플랫폼|레이싱|RPG|퍼즐).*(만들|생성)", re.IGNORECASE),
                re.compile(r"create.*(game|racing|platformer)", re.IGNORECASE),
            ],
            IntentType.MODIFY_GAME: [
                re.compile(r"(수정|변경|바꾸|고치)", re.IGNORECASE),
                re.compile(r"(modify|change|update|fix)", re.IGNORECASE),
            ],
            IntentType.ADD_FEATURE: [
                re.compile(r"(기능|feature).*(추가|넣|구현)", re.IGNORECASE),
                re.compile(r"(추가|넣어|구현).*(기능|시스템)", re.IGNORECASE),
                re.compile(r"add.*(feature|system|functionality)", re.IGNORECASE),
            ],
            IntentType.ASK_QUESTION: [
                re.compile(r"(어떻게|무엇|뭐|왜|언제|어디)", re.IGNORECASE),
                re.compile(r"(방법|설명|알려)", re.IGNORECASE),
                re.compile(r"(how|what|why|when|where|explain)", re.IGNORECASE),
            ],
            IntentType.GIVE_FEEDBACK: [
                re.compile(r"(좋아|싫어|개선|피드백|의견)", re.IGNORECASE),
                re.compile(r"(good|bad|improve|feedback|opinion)", re.IGNORECASE),
            ],
            IntentType.STATUS_CHECK: [
                re.compile(r"(상태|진행|현재|지금)", re.IGNORECASE),
                re.compile(r"(status|progress|current|now)", re.IGNORECASE),
            ],
            IntentType.HELP: [
                re.compile(r"(도움|help|명령|사용법)", re.IGNORECASE),
            ],
        }
    
    def _initialize_response_templates(self) -> Dict[IntentType, List[str]]:
        """응답 템플릿 초기화"""
        return {
            IntentType.CREATE_GAME: [
                "네, {game_type} 게임을 만들어드리겠습니다! 🎮",
                "{game_type} 게임 제작을 시작합니다. 24시간 동안 열심히 개발하겠습니다!",
                "좋은 선택이네요! {game_type} 게임을 만들어보겠습니다.",
            ],
            IntentType.MODIFY_GAME: [
                "네, {aspect}을(를) 수정하겠습니다.",
                "{aspect} 부분을 개선해보겠습니다.",
                "알겠습니다. {aspect}을(를) 변경하도록 하겠습니다.",
            ],
            IntentType.ADD_FEATURE: [
                "{feature} 기능을 추가하겠습니다!",
                "좋은 아이디어네요! {feature}을(를) 구현해보겠습니다.",
                "{feature} 기능 추가 작업을 시작합니다.",
            ],
            IntentType.ASK_QUESTION: [
                "좋은 질문이네요! {answer}",
                "{answer}",
                "설명드리겠습니다. {answer}",
            ],
            IntentType.STATUS_CHECK: [
                "현재 상태를 알려드리겠습니다:\n{status}",
                "진행 상황입니다:\n{status}",
            ],
            IntentType.HELP: [
                """
사용 가능한 명령어:
• 게임 만들기: "플랫폼 게임 만들어줘", "레이싱 게임 제작해줘"
• 기능 추가: "점프 기능 추가해줘", "사운드 효과 넣어줘"
• 게임 수정: "속도 더 빠르게 해줘", "색상 바꿔줘"
• 상태 확인: "지금 상태 어때?", "진행 상황 알려줘"
• 질문하기: "Panda3D가 뭐야?", "어떻게 캐릭터 움직여?"
""",
            ],
        }
    
    async def start_conversation(self):
        """대화 시작"""
        self.is_active = True
        
        print("""
🎮 AutoCI 한글 대화 인터페이스에 오신 것을 환영합니다!
저와 자연스러운 한국어로 대화하며 게임을 개발할 수 있습니다.
'도움말'이라고 입력하시면 사용 가능한 명령어를 볼 수 있습니다.
종료하려면 'exit' 또는 '종료'를 입력하세요.
""")
        
        while self.is_active:
            try:
                # 사용자 입력 받기
                user_input = input("\n💬 사용자: ").strip()
                
                if user_input.lower() in ['exit', '종료', 'quit']:
                    await self.handle_exit()
                    break
                
                # 응답 생성
                response = await self.process_input(user_input)
                
                # 응답 출력
                print(f"\n🤖 AutoCI: {response}")
                
                # 대화 기록 저장
                self._save_conversation(user_input, response)
                
            except KeyboardInterrupt:
                await self.handle_exit()
                break
            except Exception as e:
                logger.error(f"대화 처리 중 오류: {e}")
                print("\n❌ 죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.")
    
    async def process_input(self, user_input: str) -> str:
        """사용자 입력 처리"""
        # 의도 파악
        intent, entities = self._analyze_intent(user_input)
        
        # 컨텍스트 업데이트
        self.context.history.append({
            "user": user_input,
            "intent": intent.value,
            "timestamp": datetime.now().isoformat()
        })
        
        # 의도에 따른 처리
        if intent == IntentType.CREATE_GAME:
            return await self._handle_create_game(user_input, entities)
        elif intent == IntentType.MODIFY_GAME:
            return await self._handle_modify_game(user_input, entities)
        elif intent == IntentType.ADD_FEATURE:
            return await self._handle_add_feature(user_input, entities)
        elif intent == IntentType.ASK_QUESTION:
            return await self._handle_question(user_input)
        elif intent == IntentType.GIVE_FEEDBACK:
            return await self._handle_feedback(user_input)
        elif intent == IntentType.STATUS_CHECK:
            return await self._handle_status_check()
        elif intent == IntentType.HELP:
            return self._get_help_message()
        else:
            return await self._handle_unknown(user_input)
    
    def _analyze_intent(self, user_input: str) -> Tuple[IntentType, Dict[str, Any]]:
        """사용자 의도 분석"""
        entities = {}
        
        # 각 의도 패턴 확인
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern.search(user_input):
                    # 엔티티 추출
                    entities = self._extract_entities(user_input, intent_type)
                    return intent_type, entities
        
        return IntentType.UNKNOWN, entities
    
    def _extract_entities(self, user_input: str, intent_type: IntentType) -> Dict[str, Any]:
        """엔티티 추출"""
        entities = {}
        
        if intent_type == IntentType.CREATE_GAME:
            # 게임 타입 추출
            game_types = {
                "플랫폼": "platformer",
                "레이싱": "racing",
                "알피지": "rpg",
                "RPG": "rpg",
                "퍼즐": "puzzle"
            }
            
            for korean, english in game_types.items():
                if korean in user_input:
                    entities["game_type"] = english
                    entities["game_type_korean"] = korean
                    break
            
            # 게임 이름 추출 (따옴표 안의 텍스트)
            name_match = re.search(r"['\"]([^'\"]+)['\"]", user_input)
            if name_match:
                entities["game_name"] = name_match.group(1)
        
        elif intent_type in [IntentType.ADD_FEATURE, IntentType.MODIFY_GAME]:
            # 기능/수정 대상 추출
            features = [
                "점프", "이동", "충돌", "사운드", "음악", "UI", 
                "메뉴", "점수", "레벨", "캐릭터", "애니메이션"
            ]
            
            for feature in features:
                if feature in user_input:
                    entities["feature"] = feature
                    break
        
        return entities
    
    async def _handle_create_game(self, user_input: str, entities: Dict[str, Any]) -> str:
        """게임 생성 처리"""
        game_type = entities.get("game_type", "platformer")
        game_type_korean = entities.get("game_type_korean", "플랫폼")
        game_name = entities.get("game_name", f"My{game_type.capitalize()}Game")
        
        # 게임 개발 시작
        success = await self.game_pipeline.start_development(game_name, game_type)
        
        if success:
            self.context.current_game = game_name
            response = self.response_templates[IntentType.CREATE_GAME][0].format(
                game_type=game_type_korean
            )
            response += f"\n\n프로젝트 이름: {game_name}"
            response += "\n24시간 자동 개발이 시작되었습니다."
            response += "\n실시간으로 개발 과정을 확인하실 수 있습니다."
        else:
            response = "죄송합니다. 게임 개발을 시작할 수 없습니다. 이미 진행 중인 프로젝트가 있는지 확인해주세요."
        
        return response
    
    async def _handle_modify_game(self, user_input: str, entities: Dict[str, Any]) -> str:
        """게임 수정 처리"""
        if not self.context.current_game:
            return "현재 개발 중인 게임이 없습니다. 먼저 게임을 만들어주세요."
        
        # AI를 통한 수정 사항 분석
        modification_prompt = f"""
        사용자 요청: {user_input}
        현재 게임: {self.context.current_game}
        
        사용자가 원하는 수정 사항을 구체적으로 분석하고 코드로 구현하세요.
        """
        
        context = {
            "task": "modify_game",
            "user_request": user_input,
            "current_game": self.context.current_game,
            "entities": entities
        }
        modification_result = await self.ai_model.generate_code(modification_prompt, context)
        modification_code = modification_result.get('code', '') if isinstance(modification_result, dict) else str(modification_result)
        
        # 수정 사항 적용
        # TODO: 실제 게임 코드 수정 로직
        
        aspect = entities.get("feature", "게임")
        response = self.response_templates[IntentType.MODIFY_GAME][0].format(aspect=aspect)
        response += "\n수정 작업을 진행하고 있습니다..."
        
        return response
    
    async def _handle_add_feature(self, user_input: str, entities: Dict[str, Any]) -> str:
        """기능 추가 처리"""
        if not self.context.current_game:
            return "현재 개발 중인 게임이 없습니다. 먼저 게임을 만들어주세요."
        
        feature = entities.get("feature", "새로운 기능")
        
        # 기능 추가 요청
        if hasattr(self.game_pipeline, 'add_feature'):
            # 게임 파이프라인에 기능 추가 요청
            pass
        
        response = self.response_templates[IntentType.ADD_FEATURE][0].format(feature=feature)
        response += f"\n{feature} 기능을 현재 개발 중인 게임에 추가하고 있습니다."
        
        return response
    
    async def _handle_question(self, user_input: str) -> str:
        """질문 처리"""
        # AI 모델을 통한 답변 생성
        question_prompt = f"""
        사용자 질문: {user_input}
        
        Panda3D 게임 개발과 관련된 질문에 친절하고 정확하게 한국어로 답변해주세요.
        코드 예제가 필요한 경우 포함해주세요.
        """
        
        context = {
            "task": "answer_question",
            "question": user_input,
            "topic": "Panda3D game development"
        }
        answer_result = await self.ai_model.generate_code(question_prompt, context, max_length=500)
        answer = answer_result.get('code', '') if isinstance(answer_result, dict) else str(answer_result)
        
        if answer:
            response = self.response_templates[IntentType.ASK_QUESTION][0].format(answer=answer)
        else:
            # 기본 답변
            response = self._generate_basic_answer(user_input)
        
        return response
    
    def _generate_basic_answer(self, question: str) -> str:
        """기본 답변 생성"""
        if "Panda3D" in question:
            return """
Panda3D는 Python으로 개발된 오픈소스 3D 게임 엔진입니다.
주요 특징:
• Python으로 쉽게 3D 게임 개발 가능
• 크로스 플랫폼 지원 (Windows, Linux, Mac)
• 강력한 렌더링 엔진과 물리 엔진 통합
• 대규모 월드 렌더링 지원
"""
        elif "캐릭터" in question and "움직" in question:
            return """
Panda3D에서 캐릭터를 움직이는 기본 방법:

```python
# 키보드 입력 받기
self.accept("arrow_left", self.move_left)
self.accept("arrow_right", self.move_right)

# 이동 함수
def move_left(self):
    pos = self.player.getPos()
    self.player.setPos(pos.x - 1, pos.y, pos.z)
```
"""
        else:
            return "흥미로운 질문이네요! 더 자세한 정보를 위해 Panda3D 문서를 참고하시거나, 구체적인 질문을 해주세요."
    
    async def _handle_feedback(self, user_input: str) -> str:
        """피드백 처리"""
        # 피드백 저장
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "game": self.context.current_game,
            "feedback": user_input,
            "context": self.context.history[-5:]  # 최근 5개 대화
        }
        
        # 피드백 파일에 저장
        feedback_path = "user_feedback"
        os.makedirs(feedback_path, exist_ok=True)
        
        feedback_file = f"{feedback_path}/feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        return "소중한 피드백 감사합니다! 더 나은 게임 개발을 위해 참고하겠습니다. 😊"
    
    async def _handle_status_check(self) -> str:
        """상태 확인 처리"""
        if not self.context.current_game:
            return "현재 개발 중인 게임이 없습니다."
        
        # 게임 개발 상태 가져오기
        if hasattr(self.game_pipeline, 'current_project') and self.game_pipeline.current_project:
            project = self.game_pipeline.current_project
            status = f"""
🎮 게임: {project.name}
📊 진행률: {project.progress_percentage:.1f}%
🔄 현재 단계: {project.current_phase.value}
⏱️ 경과 시간: {project.elapsed_time}
⏳ 남은 시간: {project.remaining_time}
✅ 완료된 기능: {len(project.completed_features)}개
📋 남은 기능: {len(project.pending_features)}개
🏆 품질 점수: {project.quality_metrics.total_score}/100
"""
        else:
            status = f"게임 '{self.context.current_game}' 개발 중입니다."
        
        response = self.response_templates[IntentType.STATUS_CHECK][0].format(status=status)
        return response
    
    def _get_help_message(self) -> str:
        """도움말 메시지"""
        return self.response_templates[IntentType.HELP][0]
    
    async def _handle_unknown(self, user_input: str) -> str:
        """알 수 없는 입력 처리"""
        # AI를 통한 의도 파악 시도
        intent_prompt = f"""
        사용자 입력: {user_input}
        
        이 입력이 게임 개발과 관련이 있나요? 
        관련이 있다면 어떤 작업을 원하는지 추측해주세요.
        """
        
        context = {
            "task": "understand_intent",
            "user_input": user_input
        }
        ai_result = await self.ai_model.generate_code(intent_prompt, context, max_length=200)
        ai_response = ai_result.get('code', '') if isinstance(ai_result, dict) else str(ai_result)
        
        if ai_response and "게임" in ai_response:
            return f"말씀하신 내용을 이해했습니다. {ai_response}"
        else:
            return """
죄송합니다. 무엇을 원하시는지 정확히 이해하지 못했습니다.
다음과 같이 말씀해주세요:
• "플랫폼 게임 만들어줘"
• "점프 기능 추가해줘"
• "현재 상태 알려줘"
• "도움말"
"""
    
    async def handle_exit(self):
        """종료 처리"""
        self.is_active = False
        
        if self.game_pipeline.is_running:
            self.game_pipeline.stop()
        
        print("\n👋 AutoCI 한글 대화 인터페이스를 종료합니다. 감사합니다!")
    
    def _save_conversation(self, user_input: str, response: str):
        """대화 내용 저장"""
        conversation_path = "conversations"
        os.makedirs(conversation_path, exist_ok=True)
        
        # 일별 대화 파일
        today = datetime.now().strftime('%Y%m%d')
        conversation_file = f"{conversation_path}/conversation_{today}.json"
        
        # 기존 대화 로드
        conversations = []
        if os.path.exists(conversation_file):
            with open(conversation_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        
        # 새 대화 추가
        conversations.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": response,
            "context": {
                "current_game": self.context.current_game,
                "current_phase": self.context.current_phase
            }
        })
        
        # 저장
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)


# CLI 실행
if __name__ == "__main__":
    interface = KoreanConversationInterface()
    asyncio.run(interface.start_conversation())