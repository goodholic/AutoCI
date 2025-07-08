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
    
    def __init__(self, game_factory=None):
        self.conversation_history = []
        self.current_context = {}
        self.game_factory = game_factory # GameFactory24H 인스턴스 저장
        
        # 대화 의도 분류
        self.intent_patterns = {
            "질문": ["어떻게", "무엇", "뭐", "왜", "언제", "어디", "누가", "?"],
            "요청": ["해줘", "해주세요", "만들어", "생성", "실행", "보여줘", "알려줘"],
            "피드백": ["좋아", "나빠", "잘못", "틀렸", "맞아", "고마워", "감사"],
            "설명": ["설명", "알려", "가르쳐", "이해", "모르겠"],
            "명령": ["시작", "중지", "실행", "정지", "빌드", "테스트"],
            "기능_추가": ["기능 추가", "추가해줘", "넣어줘"],
            "게임_수정": ["수정해줘", "바꿔줘", "변경해줘"],
            "대화": ["안녕", "반가워", "고마워", "미안", "괜찮아"],
            "pytorch_학습": ["pytorch", "파이토치", "텐서", "tensor", "autograd", "신경망", "딥러닝", "학습", "모델"]
        }
        
        # 엔티티 추출 패턴
        self.entity_patterns = {
            "godot": ["고도", "godot", "고돗", "엔진"],
            "csharp": ["씨샵", "c#", "csharp", "시샵"],
            "network": ["네트워크", "멀티플레이어", "서버", "클라이언트", "동기화"],
            "nakama": ["나카마", "nakama", "백엔드"],
            "ai": ["ai", "인공지능", "모델", "학습"],
            "build": ["빌드", "컴파일", "생성", "만들기"],
            "error": ["오류", "에러", "버그", "문제", "안됨", "안돼"],
            "점프": ["점프", "높이", "점프력"],
            "속도": ["속도", "빠르게", "느리게"],
            "체력": ["체력", "hp", "생명력"],
            "사운드": ["사운드", "소리", "음악"],
            "그래픽": ["그래픽", "시각", "효과"],
            "UI": ["UI", "인터페이스", "화면"],
            "pytorch": ["pytorch", "파이토치", "토치"],
            "tensor": ["텐서", "tensor", "배열"],
            "autograd": ["autograd", "자동미분", "미분", "그래디언트"],
            "neural_network": ["신경망", "네트워크", "nn", "모델"],
            "training": ["학습", "훈련", "training", "train"],
            "optimizer": ["옵티마이저", "optimizer", "adam", "sgd"],
            "loss": ["손실", "loss", "로스", "손실함수"]
        }
        
        # AI 모델 컨트롤러 초기화
        try:
            from modules.ai_model_controller import AIModelController
            self.ai_controller = AIModelController()
            logger.info("AI 모델 컨트롤러가 한글 대화 시스템에 연결되었습니다.")
        except ImportError:
            self.ai_controller = None
            logger.warning("AI 모델 컨트롤러를 로드할 수 없습니다. 의도/엔티티 인식이 제한됩니다.")
        
        # PyTorch 튜터 초기화
        try:
            from modules.pytorch_tutor import PyTorchTutor
            self.pytorch_tutor = PyTorchTutor()
            logger.info("PyTorch 튜터가 한글 대화 시스템에 연결되었습니다.")
        except ImportError:
            self.pytorch_tutor = None
            logger.warning("PyTorch 튜터를 로드할 수 없습니다.")
        
        # 연속 학습 시스템 연동
        try:
            from core_system.continuous_learning_system import ContinuousLearningSystem
            self.learning_system = ContinuousLearningSystem()
            self.knowledge_base = self.learning_system.knowledge_base
            logger.info("연속 학습 시스템과 연동되었습니다.")
        except Exception as e:
            self.learning_system = None
            logger.warning(f"연속 학습 시스템을 로드할 수 없습니다: {e}")
            
            # 지식 베이스 직접 로드 시도
            try:
                kb_path = Path(__file__).parent.parent / "continuous_learning" / "knowledge_base" / "knowledge_base.json"
                if kb_path.exists():
                    with open(kb_path, 'r', encoding='utf-8') as f:
                        self.knowledge_base = json.load(f)
                    logger.info(f"지식 베이스를 직접 로드했습니다: {len(self.knowledge_base)} 카테고리")
                else:
                    self.knowledge_base = {}
                    logger.warning("지식 베이스 파일을 찾을 수 없습니다.")
            except Exception as kb_error:
                self.knowledge_base = {}
                logger.error(f"지식 베이스 로드 실패: {kb_error}")
        
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
            ],
            "game_modification_success": [
                "✅ 게임에 {feature_or_aspect}을(를) 성공적으로 {action}했습니다!",
                "🎮 {feature_or_aspect} 변경 요청을 처리했습니다. 게임에서 확인해보세요!",
                "👍 {feature_or_aspect}에 대한 {action} 작업이 완료되었습니다."
            ],
            "game_modification_fail": [
                "❌ {feature_or_aspect}을(를) {action}하는 데 실패했습니다. 다시 시도해주세요.",
                "⚠️ 죄송합니다. {feature_or_aspect} 변경 요청을 처리할 수 없습니다.",
                "🤔 {feature_or_aspect}에 대한 {action} 작업 중 문제가 발생했습니다."
            ]
        }
        
        logger.info("한글 대화 시스템이 초기화되었습니다.")
    
    async def process_user_input(self, user_input: str, 
                               evolution_system=None) -> str:
        """사용자 입력 처리 및 응답 생성"""
        # 의도 분류
        intent = await self._classify_intent(user_input) # await 추가
        
        # 엔티티 추출
        entities = await self._extract_entities(user_input) # await 추가
        
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
    
    async def _classify_intent(self, text: str) -> str: # async 추가
        """AI 모델을 사용하여 의도 분류"""
        if self.ai_controller:
            try:
                prompt = f"""
다음 사용자 입력의 의도를 가장 잘 나타내는 단일 키워드를 선택하세요. 가능한 의도는 '질문', '요청', '피드백', '설명', '명령', '기능_추가', '게임_수정', '대화', 'pytorch_학습' 입니다.

사용자 입력: {text}
의도:
"""
                response = await self.ai_controller.generate_response(prompt, model_name="deepseek-coder-7b") # await 추가
                if response and response.get('response'):
                    intent = response['response'].strip().lower()
                    if intent in ['질문', '요청', '피드백', '설명', '명령', '기능_추가', '게임_수정', '대화', 'pytorch_학습']:
                        return intent
            except Exception as e:
                logger.warning(f"AI 기반 의도 분류 실패: {e}. 기본 분류로 폴백합니다.")
        
        # AI 모델을 사용할 수 없거나 실패한 경우 기존 방식 폴백
        text_lower = text.lower()
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return intent
        
        return "대화"  # 기본값
    
    async def _extract_entities(self, text: str) -> List[str]: # async 추가
        """AI 모델을 사용하여 엔티티 추출"""
        if self.ai_controller:
            try:
                prompt = f"""
다음 사용자 입력에서 게임 개발 및 PyTorch와 관련된 엔티티를 쉼표로 구분하여 나열하세요. 가능한 엔티티는 'godot', 'csharp', 'network', 'nakama', 'ai', 'build', 'error', '점프', '속도', '체력', '사운드', '그래픽', 'UI', 'pytorch', 'tensor', 'autograd', 'neural_network', 'training', 'optimizer', 'loss' 입니다.

사용자 입력: {text}
엔티티:
"""
                response = await self.ai_controller.generate_response(prompt, model_name="deepseek-coder-7b") # await 추가
                if response and response.get('response'):
                    entities_str = response['response'].strip().lower()
                    return [e.strip() for e in entities_str.split(',') if e.strip() in self.entity_patterns]
            except Exception as e:
                logger.warning(f"AI 기반 엔티티 추출 실패: {e}. 기본 추출로 폴백합니다.")

        # AI 모델을 사용할 수 없거나 실패한 경우 기존 방식 폴백
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
        
        elif intent == "pytorch_학습":
            # PyTorch 학습 모드 처리
            if self.pytorch_tutor:
                return await self._handle_pytorch_learning(user_input, entities)
            else:
                return "PyTorch 튜터가 설치되지 않았습니다. 모듈을 확인해주세요."
        
        elif intent == "질문":
            topic = entities[0] if entities else "그것"
            
            # 먼저 지식 베이스에서 검색
            knowledge_results = self._search_knowledge_base(user_input, entities)
            if knowledge_results:
                response = "📚 **학습한 지식 기반 답변**\n\n"
                
                # 가장 관련성 높은 답변 사용
                best_result = knowledge_results[0]
                response += f"**{best_result['category']}** 카테고리에서 찾은 정보:\n\n"
                response += best_result['item'].get('answer', '답변 없음')
                
                # 추가 관련 정보가 있으면 표시
                if len(knowledge_results) > 1:
                    response += "\n\n**관련 정보:**\n"
                    for result in knowledge_results[1:3]:  # 최대 2개 추가
                        response += f"- {result['category']}: {result['item'].get('question', '')[:50]}...\n"
                
                return response
            
            # PyTorch 관련 질문 처리
            if self.pytorch_tutor and any(e in ["pytorch", "tensor", "autograd", "neural_network", "training", "optimizer", "loss"] for e in entities):
                return await self._handle_pytorch_question(user_input, entities)
            
            # 기존 템플릿 응답
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
            # AI 통합 요청 처리
            if any(keyword in user_input.lower() for keyword in ["ai 추가", "ai 넣어", "인공지능", "pytorch"]):
                return await self._handle_ai_integration_request(user_input, entities)
            
            action = self._extract_action(user_input)
            response = self._select_template("command_acknowledge", action=action)
            
            # 실제 명령 실행 (여기서는 시뮬레이션)
            result = await self._execute_command(action)
            return f"{response}\n\n{result}"
        
        elif intent == "명령":
            action = self._extract_action(user_input)
            response = self._select_template("command_acknowledge", action=action)
            
            # 실제 명령 실행 (여기서는 시뮬레이션)
            result = await self._execute_command(action)
            return f"{response}\n\n{result}"
        
        elif intent == "기능_추가" or intent == "게임_수정":
            feature_or_aspect = self._extract_feature_or_aspect(user_input)
            action = "추가" if intent == "기능_추가" else "수정"
            
            # 학습한 지식을 활용한 게임 수정 제안
            knowledge_results = self._search_knowledge_base(user_input, entities)
            
            if self.game_factory and self.game_factory.current_project:
                # 지식 베이스의 정보를 활용하여 더 나은 수정 제안
                if knowledge_results:
                    logger.info(f"관련 지식 {len(knowledge_results)}개 발견, 게임 수정에 활용")
                    
                    # 가장 관련성 높은 지식을 프롬프트에 포함
                    best_knowledge = knowledge_results[0]['item'].get('answer', '')
                    enhanced_input = f"{user_input}\n\n참고 지식: {best_knowledge[:500]}"
                else:
                    enhanced_input = user_input
                
                success = await self._handle_game_modification(feature_or_aspect, action, enhanced_input)
                
                if success:
                    response = self._select_template("game_modification_success", 
                                                 feature_or_aspect=feature_or_aspect, action=action)
                    if knowledge_results:
                        response += "\n\n💡 학습한 지식을 활용하여 더 나은 수정을 적용했습니다!"
                    return response
                else:
                    return self._select_template("game_modification_fail", 
                                                 feature_or_aspect=feature_or_aspect, action=action)
            else:
                return "현재 개발 중인 게임 프로젝트가 없습니다. 먼저 게임을 생성해주세요."

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
    
    def _extract_feature_or_aspect(self, text: str) -> str:
        """사용자 입력에서 기능 또는 측면 추출"""
        text_lower = text.lower()
        for entity_type, patterns in self.entity_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return entity_type
        return "요청하신 내용"

    async def _handle_game_modification(self, feature_or_aspect: str, action: str, user_input: str) -> bool:
        """게임 수정 또는 기능 추가 처리"""
        if not self.game_factory or not self.game_factory.current_project:
            return False

        project_path = self.game_factory.current_project
        
        # AI 모델 컨트롤러 임포트
        try:
            from modules.ai_model_controller import AIModelController
            ai_controller = AIModelController()
        except ImportError:
            logger.error("AI 모델 컨트롤러를 찾을 수 없습니다.")
            return False

        # Godot 프로젝트 파일 읽기
        player_script_path = project_path / "scripts" / "Player.gd"
        if not player_script_path.exists():
            logger.error(f"Player.gd 스크립트를 찾을 수 없습니다: {player_script_path}")
            return False

        original_player_script_content = player_script_path.read_text()

        # AI에게 수정 요청 및 확인 과정 추가
        prompt = f"""
당신은 Godot 게임 개발 전문가 AI입니다. 현재 Godot 프로젝트의 Player.gd 스크립트 내용은 다음과 같습니다:

```gdscript
{original_player_script_content}
```

사용자가 "{user_input}"라고 요청했습니다. 이 요청에 따라 Player.gd 스크립트를 수정해주세요.

당신의 사고 과정 (Chain of Thought):
1. 문제 분석: 사용자의 요청은 무엇이며, Player.gd 스크립트에서 어떤 부분을 수정해야 할까요?
2. 해결 계획: 어떤 단계로 스크립트를 수정할 것인가요? (최소 2단계 이상)
3. 예상 결과: 수정된 스크립트가 어떤 기능을 할 것으로 예상되나요?
4. 최종 수정된 Player.gd 스크립트:
```gdscript
# 여기에 수정된 Player.gd 스크립트 전체 내용
```
"""
        
        try:
            ai_response = await self.ai_controller.generate_response(prompt, model_name="deepseek-coder-7b")
            if not ai_response or not ai_response.get('response'):
                logger.error("AI로부터 유효한 응답을 받지 못했습니다.")
                return False
            
            full_response_text = ai_response['response']
            logger.info(f"AI 응답: {full_response_text[:200]}...")

            # 최종 수정된 스크립트 추출
            import re
            script_match = re.search(r"```gdscript\n(.*?)```", full_response_text, re.DOTALL)
            
            if not script_match:
                logger.warning("AI 응답에서 수정된 Player.gd 스크립트를 찾을 수 없습니다.")
                return False

            modified_script_content = script_match.group(1).strip()

            # AI가 해석한 내용을 사용자에게 확인
            confirmation_prompt = f"""
AI가 사용자의 요청을 다음과 같이 해석했습니다:

{full_response_text}

이대로 게임을 수정하시겠습니까? (예/아니오): 
"""
            print(confirmation_prompt, end="", flush=True)
            user_confirmation = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            user_confirmation = user_confirmation.strip().lower()

            if user_confirmation == "예":
                player_script_path.write_text(modified_script_content)
                logger.info(f"Player.gd 스크립트가 성공적으로 {action}되었습니다.")
                return True
            else:
                logger.info("사용자가 게임 수정을 취소했습니다.")
                return False

        except Exception as e:
            logger.error(f"게임 수정 중 AI 요청 오류: {str(e)}")
            return False

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
    
    async def _handle_pytorch_learning(self, user_input: str, entities: List[str]) -> str:
        """PyTorch 학습 요청 처리"""
        # 특정 주제 검색
        search_results = self.pytorch_tutor.search_topic(user_input)
        
        if search_results:
            # 검색 결과가 있으면 첫 번째 주제 설명
            topic = search_results[0]
            response = f"🎓 **PyTorch 학습 모드**\n\n"
            response += self.pytorch_tutor.format_response(topic['topic_id'], style="detailed")
            
            # 실습 코드 제공
            practice_code = self.pytorch_tutor.generate_practice_code(topic['topic_id'])
            if practice_code:
                response += f"\n\n**💡 실습해보기**\n```python\n{practice_code}\n```"
            
            return response
        
        # 학습 경로 제공
        if "시작" in user_input or "기초" in user_input:
            learning_path = self.pytorch_tutor.get_learning_path("beginner")
            response = "🎓 **PyTorch 학습을 시작하겠습니다!**\n\n"
            response += "추천 학습 경로:\n"
            for i, topic_id in enumerate(learning_path[:5], 1):
                topic_info = self.pytorch_tutor.get_topic_explanation(topic_id)
                if topic_info:
                    response += f"{i}. {topic_info['title']} (`{topic_id}`)\n"
            
            response += "\n원하는 주제를 선택하거나 'pytorch 텐서 알려줘'와 같이 질문해주세요!"
            return response
        
        # 기본 응답
        return """🎓 **PyTorch 학습 도우미**

PyTorch의 다양한 주제를 학습할 수 있습니다:

📚 **기초 과정**
- PyTorch 소개
- 텐서(Tensor) 기초
- 자동 미분(Autograd)

🔧 **심화 과정**
- 신경망 구축 (nn.Module)
- 손실 함수와 옵티마이저
- 모델 학습 루프

💬 **사용 예시**
- "pytorch 텐서 기초 알려줘"
- "autograd 설명해줘"
- "신경망 만드는 방법 보여줘"

어떤 주제부터 시작하시겠습니까?"""
    
    async def _handle_pytorch_question(self, user_input: str, entities: List[str]) -> str:
        """PyTorch 관련 질문 처리"""
        # 관련 주제 검색
        for entity in entities:
            if entity in ["tensor", "autograd", "neural_network", "training", "optimizer", "loss"]:
                search_results = self.pytorch_tutor.search_topic(entity)
                if search_results:
                    topic = search_results[0]
                    return self.pytorch_tutor.format_response(topic['topic_id'], style="detailed")
        
        # AI를 통한 답변 생성
        if self.ai_controller:
            try:
                prompt = f"""
PyTorch에 대한 다음 질문에 한글로 친절하게 답변해주세요:

질문: {user_input}

답변은 다음 형식으로 작성해주세요:
1. 간단한 설명
2. 예제 코드 (있다면)
3. 추가 학습 자료 (있다면)
"""
                response = await self.ai_controller.generate_response(prompt, model_name="deepseek-coder-7b")
                if response and response.get('response'):
                    return f"🤖 **PyTorch 질문 답변**\n\n{response['response']}"
            except Exception as e:
                logger.error(f"PyTorch 질문 처리 중 오류: {e}")
        
        return "PyTorch 관련 질문을 더 구체적으로 설명해주시면 도움을 드릴 수 있습니다."
    
    def _search_knowledge_base(self, query: str, entities: List[str]) -> List[Dict[str, Any]]:
        """지식 베이스에서 관련 정보 검색"""
        results = []
        query_lower = query.lower()
        
        # 모든 카테고리 검색
        for category, items in self.knowledge_base.items():
            if isinstance(items, dict):
                # 하위 카테고리가 있는 경우
                for subcategory, subitems in items.items():
                    if isinstance(subitems, list):
                        for item in subitems:
                            if self._is_relevant_knowledge(item, query_lower, entities):
                                results.append({
                                    "category": f"{category}/{subcategory}",
                                    "item": item
                                })
            elif isinstance(items, list):
                # 직접 아이템 리스트인 경우
                for item in items:
                    if self._is_relevant_knowledge(item, query_lower, entities):
                        results.append({
                            "category": category,
                            "item": item
                        })
        
        # 품질 점수로 정렬
        results.sort(key=lambda x: x['item'].get('quality_score', 0), reverse=True)
        return results[:5]  # 상위 5개만 반환
    
    def _is_relevant_knowledge(self, item: Dict[str, Any], query: str, entities: List[str]) -> bool:
        """지식 항목이 쿼리와 관련있는지 확인"""
        # 질문이나 답변에 쿼리가 포함되어 있는지
        if query in item.get('question', '').lower() or query in item.get('answer', '').lower():
            return True
        
        # 키워드 매칭
        item_keywords = [kw.lower() for kw in item.get('keywords', [])]
        for entity in entities:
            if entity in item_keywords:
                return True
        
        # 쿼리의 단어들이 키워드에 있는지
        query_words = query.split()
        for word in query_words:
            if any(word in kw for kw in item_keywords):
                return True
        
        return False
    
    async def _handle_ai_integration_request(self, user_input: str, entities: List[str]) -> str:
        """게임에 AI 통합 요청 처리"""
        try:
            from modules.pytorch_game_ai import integrate_pytorch_with_godot
            
            # 현재 게임 프로젝트 확인
            if not self.game_factory or not self.game_factory.current_project:
                return "먼저 게임 프로젝트를 생성해주세요. AI를 추가하려면 게임이 있어야 합니다."
            
            game_type = self.game_factory.current_project.get("type", "platformer")
            project_path = self.game_factory.current_project.get("path", "")
            
            # PyTorch AI 통합
            integration_result = integrate_pytorch_with_godot(game_type, project_path)
            
            response = f"🤖 **PyTorch AI가 {game_type} 게임에 통합되었습니다!**\n\n"
            
            # 학습한 지식 활용
            knowledge_results = self._search_knowledge_base("게임 ai", ["ai", "godot", "neural_network"])
            if knowledge_results:
                response += "📚 **관련 학습 지식:**\n"
                response += knowledge_results[0]['item'].get('answer', '')[:300] + "...\n\n"
            
            response += "**생성된 AI 기능:**\n"
            response += f"- 게임 타입: {game_type}\n"
            response += f"- AI 스크립트: {len(integration_result['ai_script'])} 줄\n"
            response += f"- 모델 경로: {integration_result['model_path']}\n\n"
            
            response += "**통합 방법:**\n"
            response += "1. 생성된 AI 스크립트를 게임에 추가\n"
            response += "2. 캐릭터나 적에 AI 컴포넌트 연결\n"
            response += "3. 게임 플레이로 AI 학습 데이터 수집\n\n"
            
            response += "💡 AI가 게임을 플레이하면서 점점 더 똑똑해집니다!"
            
            return response
            
        except ImportError:
            return "PyTorch 게임 AI 모듈을 로드할 수 없습니다."
        except Exception as e:
            logger.error(f"AI 통합 중 오류: {e}")
            return "AI 통합 중 오류가 발생했습니다."


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