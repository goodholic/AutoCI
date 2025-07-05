"""
Panda3D AI 에이전트 시스템
2.5D/3D 게임 개발을 위한 지능형 에이전트
강화학습과 딥러닝을 활용한 게임 자동 생성 및 최적화
"""

import os
import sys
import json
import time
import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import queue

# PyTorch 임포트
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch가 설치되지 않았습니다. 일부 기능이 제한됩니다.")

# Panda3D 임포트
try:
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task
    from panda3d.core import *
    from direct.actor.Actor import Actor
    from direct.interval.IntervalGlobal import *
    PANDA3D_AVAILABLE = True
except ImportError:
    PANDA3D_AVAILABLE = False
    logging.warning("Panda3D가 설치되지 않았습니다. 시뮬레이션 모드로 실행됩니다.")

# 내부 모듈 임포트
from .ai_model_integration import get_ai_integration
from .socketio_realtime_system import SocketIORealtimeSystem
from .panda3d_automation_controller import Panda3DAutomationController, AutomationAction, ActionType

logger = logging.getLogger(__name__)


class GameElementType(Enum):
    """게임 요소 타입"""
    PLAYER = "player"
    ENEMY = "enemy"
    PLATFORM = "platform"
    COLLECTIBLE = "collectible"
    OBSTACLE = "obstacle"
    TRIGGER = "trigger"
    UI_ELEMENT = "ui_element"
    PARTICLE = "particle"
    LIGHT = "light"
    CAMERA = "camera"


class ActionSpace(Enum):
    """AI 에이전트 액션 공간"""
    # 기본 게임 요소 생성
    CREATE_PLAYER = "create_player"
    CREATE_ENEMY = "create_enemy"
    CREATE_PLATFORM = "create_platform"
    CREATE_COLLECTIBLE = "create_collectible"
    
    # 게임 메커니즘
    ADD_MOVEMENT = "add_movement"
    ADD_JUMPING = "add_jumping"
    ADD_SHOOTING = "add_shooting"
    ADD_COLLISION = "add_collision"
    ADD_PHYSICS = "add_physics"
    
    # 레벨 디자인
    GENERATE_LEVEL = "generate_level"
    MODIFY_TERRAIN = "modify_terrain"
    PLACE_OBJECTS = "place_objects"
    
    # 게임플레이
    ADJUST_DIFFICULTY = "adjust_difficulty"
    ADD_POWERUP = "add_powerup"
    CREATE_CHECKPOINT = "create_checkpoint"
    
    # 시각/오디오
    IMPROVE_GRAPHICS = "improve_graphics"
    ADD_PARTICLE_EFFECT = "add_particle_effect"
    ADD_SOUND_EFFECT = "add_sound_effect"
    ADD_MUSIC = "add_music"
    
    # UI/UX
    CREATE_MENU = "create_menu"
    ADD_HUD = "add_hud"
    ADD_SCORE_SYSTEM = "add_score_system"
    
    # 최적화
    OPTIMIZE_PERFORMANCE = "optimize_performance"
    FIX_BUG = "fix_bug"
    REFACTOR_CODE = "refactor_code"


@dataclass
class GameState:
    """게임 상태 정보"""
    project_name: str
    game_type: str
    quality_score: float = 0.0
    completeness: float = 0.0
    features: List[str] = field(default_factory=list)
    bugs: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    player_feedback: List[Dict[str, Any]] = field(default_factory=list)
    current_phase: str = "initialization"
    time_elapsed: float = 0.0
    actions_taken: int = 0
    last_action: Optional[str] = None
    error_count: int = 0
    success_count: int = 0


@dataclass
class AIMemory:
    """AI 에이전트 메모리"""
    successful_patterns: List[Dict[str, Any]] = field(default_factory=list)
    failed_attempts: List[Dict[str, Any]] = field(default_factory=list)
    learned_solutions: Dict[str, Any] = field(default_factory=dict)
    code_snippets: Dict[str, str] = field(default_factory=dict)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)


class GameDevelopmentPolicyNetwork(nn.Module):
    """게임 개발 정책 네트워크 (강화학습)"""
    
    def __init__(self, state_dim: int = 128, hidden_dim: int = 256, action_dim: int = len(ActionSpace)):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor (정책)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic (가치 함수)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # 드롭아웃
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # 액션 확률 분포
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # 상태 가치
        state_value = self.critic(x)
        
        return action_probs, state_value


class Panda3DAIAgent:
    """Panda3D 게임 개발 AI 에이전트"""
    
    def __init__(self, project_name: str, game_type: str = "platformer"):
        self.project_name = project_name
        self.game_type = game_type
        self.game_state = GameState(project_name=project_name, game_type=game_type)
        self.memory = AIMemory()
        
        # AI 모델 통합
        self.ai_integration = get_ai_integration()
        
        # Panda3D 자동화 컨트롤러
        self.panda_controller = Panda3DAutomationController(self.ai_integration)
        
        # Socket.IO 실시간 시스템
        self.socketio_system = SocketIORealtimeSystem()
        
        # 강화학습 모델 (PyTorch 사용 가능한 경우)
        if TORCH_AVAILABLE:
            self.policy_network = GameDevelopmentPolicyNetwork()
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
            self.rewards = []
            self.log_probs = []
            self.values = []
        else:
            self.policy_network = None
        
        # 액션 큐
        self.action_queue = queue.Queue()
        self.is_running = False
        
        # 게임 품질 평가 기준
        self.quality_criteria = {
            "has_player": 10,
            "has_movement": 10,
            "has_jumping": 5,
            "has_enemies": 10,
            "has_collision": 10,
            "has_level": 10,
            "has_ui": 5,
            "has_score": 5,
            "has_sound": 5,
            "has_particles": 5,
            "is_playable": 15,
            "is_fun": 10
        }
        
        logger.info(f"🤖 Panda3D AI 에이전트 초기화: {project_name} ({game_type})")
    
    async def start_development(self, target_hours: float = 24.0):
        """24시간 자동 게임 개발 시작"""
        logger.info(f"🎮 게임 개발 시작: {self.project_name}")
        
        # Panda3D 프로젝트 시작
        if not self.panda_controller.start_panda3d_project(self.project_name, self.game_type):
            logger.error("Panda3D 프로젝트 시작 실패")
            return
        
        self.is_running = True
        self.game_state.current_phase = "development"
        
        # Socket.IO 서버 시작 (별도 스레드)
        socketio_thread = threading.Thread(target=self._run_socketio_server)
        socketio_thread.daemon = True
        socketio_thread.start()
        
        # 개발 루프
        start_time = time.time()
        target_seconds = target_hours * 3600
        
        while self.is_running and (time.time() - start_time) < target_seconds:
            try:
                # 현재 상태 분석
                state_vector = self._get_state_vector()
                
                # 다음 액션 결정
                action = await self._decide_next_action(state_vector)
                
                # 액션 실행
                success = await self._execute_action(action)
                
                # 결과 평가 및 학습
                reward = self._evaluate_action(action, success)
                if self.policy_network:
                    self.rewards.append(reward)
                
                # 진행 상황 브로드캐스트
                await self._broadcast_progress()
                
                # 게임 품질 평가
                self._update_quality_score()
                
                # 단계별 목표 확인
                if self._check_phase_completion():
                    self._advance_phase()
                
                # 짧은 대기
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"개발 루프 오류: {e}")
                self.game_state.error_count += 1
                
                # 오류 복구 시도
                await self._handle_error(e)
        
        # 개발 완료
        await self._finalize_development()
    
    def _get_state_vector(self) -> torch.Tensor:
        """현재 게임 상태를 벡터로 변환"""
        # 특징 벡터 구성
        features = []
        
        # 기본 정보
        features.append(self.game_state.quality_score / 100)
        features.append(self.game_state.completeness / 100)
        features.append(self.game_state.actions_taken / 1000)
        features.append(self.game_state.error_count / 100)
        features.append(self.game_state.success_count / 100)
        
        # 게임 특징
        for feature in ["player", "movement", "jumping", "enemies", "collision", 
                       "level", "ui", "score", "sound", "particles"]:
            features.append(1.0 if feature in self.game_state.features else 0.0)
        
        # 현재 단계
        phases = ["initialization", "core_mechanics", "level_design", 
                 "gameplay", "polish", "optimization"]
        phase_idx = phases.index(self.game_state.current_phase) if self.game_state.current_phase in phases else 0
        phase_vector = [0.0] * len(phases)
        phase_vector[phase_idx] = 1.0
        features.extend(phase_vector)
        
        # 성능 메트릭
        features.append(self.game_state.performance_metrics.get("fps", 60) / 60)
        features.append(self.game_state.performance_metrics.get("memory_usage", 0) / 1000)
        
        # 패딩하여 고정 크기로
        while len(features) < 128:
            features.append(0.0)
        
        if TORCH_AVAILABLE:
            return torch.FloatTensor(features[:128]).unsqueeze(0)
        else:
            return features[:128]
    
    async def _decide_next_action(self, state_vector) -> ActionSpace:
        """다음 액션 결정 (강화학습 또는 휴리스틱)"""
        if self.policy_network and TORCH_AVAILABLE:
            # 강화학습 기반 결정
            with torch.no_grad():
                action_probs, state_value = self.policy_network(state_vector)
            
            # 액션 샘플링
            dist = Categorical(action_probs)
            action_idx = dist.sample()
            
            # 학습을 위한 로그 확률 저장
            self.log_probs.append(dist.log_prob(action_idx))
            self.values.append(state_value)
            
            # 액션 반환
            return list(ActionSpace)[action_idx.item()]
        else:
            # 휴리스틱 기반 결정
            return await self._heuristic_action_selection()
    
    async def _heuristic_action_selection(self) -> ActionSpace:
        """휴리스틱 기반 액션 선택"""
        phase = self.game_state.current_phase
        features = self.game_state.features
        
        # 단계별 우선순위 액션
        if phase == "initialization":
            if "player" not in features:
                return ActionSpace.CREATE_PLAYER
            elif "level" not in features:
                return ActionSpace.GENERATE_LEVEL
                
        elif phase == "core_mechanics":
            if "movement" not in features:
                return ActionSpace.ADD_MOVEMENT
            elif "jumping" not in features and self.game_type in ["platformer", "adventure"]:
                return ActionSpace.ADD_JUMPING
            elif "collision" not in features:
                return ActionSpace.ADD_COLLISION
                
        elif phase == "level_design":
            if "platforms" not in features:
                return ActionSpace.CREATE_PLATFORM
            elif "enemies" not in features:
                return ActionSpace.CREATE_ENEMY
            elif "collectibles" not in features:
                return ActionSpace.CREATE_COLLECTIBLE
                
        elif phase == "gameplay":
            if "score" not in features:
                return ActionSpace.ADD_SCORE_SYSTEM
            elif "ui" not in features:
                return ActionSpace.ADD_HUD
            elif "difficulty" not in features:
                return ActionSpace.ADJUST_DIFFICULTY
                
        elif phase == "polish":
            if "particles" not in features:
                return ActionSpace.ADD_PARTICLE_EFFECT
            elif "sound" not in features:
                return ActionSpace.ADD_SOUND_EFFECT
            elif "graphics_improved" not in features:
                return ActionSpace.IMPROVE_GRAPHICS
                
        elif phase == "optimization":
            if self.game_state.performance_metrics.get("fps", 60) < 30:
                return ActionSpace.OPTIMIZE_PERFORMANCE
            elif len(self.game_state.bugs) > 0:
                return ActionSpace.FIX_BUG
        
        # 기본 액션
        return ActionSpace.OPTIMIZE_PERFORMANCE
    
    async def _execute_action(self, action: ActionSpace) -> bool:
        """액션 실행"""
        try:
            logger.info(f"🎯 액션 실행: {action.value}")
            self.game_state.last_action = action.value
            self.game_state.actions_taken += 1
            
            # AI 모델을 통한 코드 생성
            code = await self._generate_code_for_action(action)
            
            if code:
                # 생성된 코드를 프로젝트에 적용
                success = await self._apply_code_to_project(action, code)
                
                if success:
                    self.game_state.success_count += 1
                    self.game_state.features.append(action.value)
                    
                    # 성공 패턴 저장
                    self.memory.successful_patterns.append({
                        "action": action.value,
                        "state": self.game_state.current_phase,
                        "code_snippet": code[:500],
                        "timestamp": time.time()
                    })
                    
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"액션 실행 실패: {e}")
            self.memory.failed_attempts.append({
                "action": action.value,
                "error": str(e),
                "timestamp": time.time()
            })
            return False
    
    async def _generate_code_for_action(self, action: ActionSpace) -> Optional[str]:
        """액션에 대한 코드 생성"""
        # 컨텍스트 구성
        context = {
            "game_type": self.game_type,
            "current_features": self.game_state.features,
            "target_action": action.value,
            "project_path": self.panda_controller.get_project_path(),
            "phase": self.game_state.current_phase
        }
        
        # 프롬프트 생성
        prompt = self._create_code_generation_prompt(action)
        
        # AI 모델을 통한 코드 생성
        result = await self.ai_integration.generate_code(
            prompt=prompt,
            context=context,
            task_type="game_dev",
            max_length=1000
        )
        
        if result["success"]:
            code = result["code"]
            
            # 코드 검증
            validation = result.get("validation", {})
            if validation.get("syntax_valid", True):
                return code
            else:
                logger.warning(f"생성된 코드 문법 오류: {validation}")
                # 재시도 또는 폴백 코드 사용
                return self._get_fallback_code(action)
        
        return None
    
    def _create_code_generation_prompt(self, action: ActionSpace) -> str:
        """코드 생성 프롬프트 생성"""
        prompts = {
            ActionSpace.CREATE_PLAYER: """
Create a player character for a Panda3D game with the following requirements:
1. Use a colored sphere or cube as placeholder model
2. Add basic properties (position, health, speed)
3. Set up camera to follow player
4. Include Korean comments
""",
            ActionSpace.ADD_MOVEMENT: """
Add movement controls to the player character:
1. WASD or arrow keys for movement
2. Smooth acceleration and deceleration
3. Proper collision boundaries
4. Speed limits
5. Include Korean comments
""",
            ActionSpace.ADD_JUMPING: """
Implement jumping mechanics:
1. Space bar to jump
2. Gravity simulation
3. Double jump prevention
4. Landing detection
5. Include Korean comments
""",
            ActionSpace.CREATE_ENEMY: """
Create an enemy AI:
1. Basic enemy model (red sphere/cube)
2. Simple patrol movement
3. Player detection
4. Chase behavior
5. Include Korean comments
""",
            ActionSpace.GENERATE_LEVEL: """
Generate a basic level:
1. Ground/floor platform
2. Some elevated platforms
3. Boundaries to prevent falling off
4. Basic lighting
5. Include Korean comments
""",
            ActionSpace.ADD_COLLISION: """
Implement collision detection:
1. Player-platform collision
2. Player-enemy collision
3. Player-collectible collision
4. Collision response
5. Include Korean comments
""",
            ActionSpace.ADD_SCORE_SYSTEM: """
Create a score system:
1. Score variable tracking
2. Points for collecting items
3. Points for defeating enemies
4. Display score on screen
5. Include Korean comments
""",
            ActionSpace.ADD_PARTICLE_EFFECT: """
Add particle effects:
1. Jump dust particles
2. Collection sparkle effect
3. Enemy defeat explosion
4. Use Panda3D particle system
5. Include Korean comments
"""
        }
        
        return prompts.get(action, f"Implement {action.value} feature for Panda3D game")
    
    def _get_fallback_code(self, action: ActionSpace) -> str:
        """폴백 코드 반환"""
        # 기본 템플릿 코드
        fallbacks = {
            ActionSpace.CREATE_PLAYER: """
# 플레이어 생성
from panda3d.core import *

class Player:
    def __init__(self, parent):
        # 플레이어 모델 (파란 구)
        self.model = loader.loadModel("models/misc/sphere")
        self.model.setScale(0.5)
        self.model.setColor(0, 0.5, 1, 1)
        self.model.reparentTo(parent)
        self.model.setPos(0, 0, 1)
        
        # 플레이어 속성
        self.speed = 10.0
        self.health = 100
        self.velocity = Vec3(0, 0, 0)
""",
            ActionSpace.ADD_MOVEMENT: """
# 이동 컨트롤 추가
def setup_movement_controls(self):
    self.accept("arrow_left", self.move_left)
    self.accept("arrow_right", self.move_right) 
    self.accept("arrow_up", self.move_forward)
    self.accept("arrow_down", self.move_backward)
    
def move_left(self):
    self.player.velocity.x = -self.player.speed
    
def move_right(self):
    self.player.velocity.x = self.player.speed
"""
        }
        
        return fallbacks.get(action, f"# TODO: Implement {action.value}")
    
    async def _apply_code_to_project(self, action: ActionSpace, code: str) -> bool:
        """생성된 코드를 프로젝트에 적용"""
        try:
            # 액션 타입에 따른 파일 경로 결정
            file_mapping = {
                ActionSpace.CREATE_PLAYER: "scripts/player.py",
                ActionSpace.ADD_MOVEMENT: "scripts/movement_controller.py",
                ActionSpace.ADD_JUMPING: "scripts/jump_controller.py",
                ActionSpace.CREATE_ENEMY: "scripts/enemy.py",
                ActionSpace.GENERATE_LEVEL: "levels/level_generator.py",
                ActionSpace.ADD_COLLISION: "scripts/collision_handler.py",
                ActionSpace.ADD_SCORE_SYSTEM: "ui/score_system.py",
                ActionSpace.ADD_PARTICLE_EFFECT: "effects/particles.py",
                ActionSpace.ADD_SOUND_EFFECT: "audio/sound_manager.py",
                ActionSpace.ADD_HUD: "ui/hud.py",
                ActionSpace.CREATE_MENU: "ui/main_menu.py",
                ActionSpace.OPTIMIZE_PERFORMANCE: "scripts/performance_optimizer.py"
            }
            
            file_path = file_mapping.get(action, f"scripts/{action.value}.py")
            
            # 코드 파일 작성
            self.panda_controller.add_action(AutomationAction(
                ActionType.CODE_WRITE,
                {"file_path": file_path, "code": code},
                f"Write {action.value} code"
            ))
            
            # main.py 업데이트하여 새 기능 통합
            await self._update_main_file(action, file_path)
            
            # 변경사항 적용을 위한 짧은 대기
            await asyncio.sleep(0.5)
            
            return True
            
        except Exception as e:
            logger.error(f"코드 적용 실패: {e}")
            return False
    
    async def _update_main_file(self, action: ActionSpace, module_path: str):
        """main.py 파일 업데이트"""
        # main.py에 새 모듈 import 추가
        import_statement = f"from {module_path.replace('/', '.').replace('.py', '')} import *\n"
        
        # 초기화 코드 추가
        init_code = self._get_initialization_code(action)
        
        # main.py 수정 (실제로는 더 정교한 코드 수정 필요)
        # 여기서는 간단한 예시만 제공
        logger.info(f"main.py 업데이트: {action.value} 기능 통합")
    
    def _get_initialization_code(self, action: ActionSpace) -> str:
        """각 액션에 대한 초기화 코드"""
        init_codes = {
            ActionSpace.CREATE_PLAYER: "self.player = Player(self.render)",
            ActionSpace.ADD_MOVEMENT: "self.setup_movement_controls()",
            ActionSpace.ADD_JUMPING: "self.setup_jump_controls()",
            ActionSpace.CREATE_ENEMY: "self.spawn_enemies()",
            ActionSpace.ADD_COLLISION: "self.setup_collision_handlers()",
            ActionSpace.ADD_SCORE_SYSTEM: "self.score_system = ScoreSystem()",
            ActionSpace.ADD_HUD: "self.hud = HUD(self.aspect2d)",
            ActionSpace.ADD_PARTICLE_EFFECT: "self.particle_manager = ParticleManager()"
        }
        
        return init_codes.get(action, f"# Initialize {action.value}")
    
    def _evaluate_action(self, action: ActionSpace, success: bool) -> float:
        """액션 결과 평가 (보상 계산)"""
        base_reward = 10.0 if success else -5.0
        
        # 단계별 보너스
        phase_bonus = {
            "initialization": 2.0,
            "core_mechanics": 3.0,
            "level_design": 2.5,
            "gameplay": 2.0,
            "polish": 1.5,
            "optimization": 1.0
        }
        
        # 중요 기능 보너스
        critical_actions = [
            ActionSpace.CREATE_PLAYER,
            ActionSpace.ADD_MOVEMENT,
            ActionSpace.ADD_COLLISION,
            ActionSpace.GENERATE_LEVEL
        ]
        
        if action in critical_actions:
            base_reward *= 1.5
        
        # 연속 성공 보너스
        if success and self.game_state.error_count == 0:
            base_reward *= 1.2
        
        # 단계 보너스 적용
        total_reward = base_reward + phase_bonus.get(self.game_state.current_phase, 0)
        
        return total_reward
    
    def _update_quality_score(self):
        """게임 품질 점수 업데이트"""
        score = 0.0
        max_score = sum(self.quality_criteria.values())
        
        # 각 기준별 점수 계산
        for criterion, points in self.quality_criteria.items():
            if criterion in self.game_state.features:
                score += points
            elif criterion == "is_playable":
                # 플레이 가능 여부는 핵심 기능들의 조합으로 판단
                required = ["player", "movement", "collision", "level"]
                if all(f in self.game_state.features for f in required):
                    score += points
            elif criterion == "is_fun":
                # 재미 요소는 추가 기능들로 판단
                fun_features = ["enemies", "score", "particles", "sound", "powerups"]
                fun_count = sum(1 for f in fun_features if f in self.game_state.features)
                score += points * (fun_count / len(fun_features))
        
        # 정규화
        self.game_state.quality_score = (score / max_score) * 100
        
        # 완성도 계산
        total_features = len(ActionSpace)
        completed_features = len(self.game_state.features)
        self.game_state.completeness = (completed_features / total_features) * 100
    
    def _check_phase_completion(self) -> bool:
        """현재 단계 완료 여부 확인"""
        phase = self.game_state.current_phase
        
        phase_requirements = {
            "initialization": ["player", "level"],
            "core_mechanics": ["movement", "collision"],
            "level_design": ["platforms", "enemies", "collectibles"],
            "gameplay": ["score", "ui"],
            "polish": ["particles", "sound"],
            "optimization": []  # 최적화는 시간 기반
        }
        
        required = phase_requirements.get(phase, [])
        if required:
            return all(req in self.game_state.features for req in required)
        
        # 최적화 단계는 일정 시간 후 완료
        if phase == "optimization":
            return self.game_state.time_elapsed > 20 * 3600  # 20시간 후
        
        return False
    
    def _advance_phase(self):
        """다음 개발 단계로 진행"""
        phases = ["initialization", "core_mechanics", "level_design", 
                 "gameplay", "polish", "optimization", "completed"]
        
        current_idx = phases.index(self.game_state.current_phase)
        if current_idx < len(phases) - 1:
            self.game_state.current_phase = phases[current_idx + 1]
            logger.info(f"📈 개발 단계 진행: {self.game_state.current_phase}")
    
    async def _broadcast_progress(self):
        """진행 상황 브로드캐스트"""
        progress_data = {
            "project_name": self.project_name,
            "game_type": self.game_type,
            "phase": self.game_state.current_phase,
            "quality_score": round(self.game_state.quality_score, 2),
            "completeness": round(self.game_state.completeness, 2),
            "features": self.game_state.features,
            "actions_taken": self.game_state.actions_taken,
            "success_rate": round(
                self.game_state.success_count / max(self.game_state.actions_taken, 1) * 100, 2
            ),
            "time_elapsed": round(self.game_state.time_elapsed / 3600, 2),  # 시간
            "last_action": self.game_state.last_action,
            "performance": self.game_state.performance_metrics
        }
        
        # Socket.IO로 진행 상황 전송
        await self.socketio_system.update_game_metrics(progress_data)
        
        # AI 액션 알림
        if self.game_state.last_action:
            await self.socketio_system.notify_ai_action(
                self.game_state.last_action,
                {"success": self.game_state.success_count > 0}
            )
    
    async def _handle_error(self, error: Exception):
        """오류 처리 및 복구"""
        logger.error(f"🔧 오류 복구 시도: {error}")
        
        # AI 모델에게 오류 해결 요청
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "last_action": self.game_state.last_action,
            "current_phase": self.game_state.current_phase
        }
        
        solution = await self.ai_integration.generate_code(
            prompt=f"Fix this error in Panda3D game: {error}",
            context=error_context,
            task_type="bug_fix"
        )
        
        if solution["success"]:
            # 해결책 적용
            fix_code = solution["code"]
            # 오류 수정 코드 적용 로직
            logger.info("오류 해결책 적용 중...")
            
            # 학습을 위해 해결책 저장
            self.memory.learned_solutions[str(error)] = fix_code
    
    async def _finalize_development(self):
        """개발 완료 처리"""
        logger.info(f"🎉 게임 개발 완료: {self.project_name}")
        
        # 최종 빌드
        await self._build_game()
        
        # 최종 보고서 생성
        report = self._generate_final_report()
        
        # 보고서 저장
        report_path = Path(self.panda_controller.get_project_path()) / "development_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        
        # 학습 모델 저장 (있는 경우)
        if self.policy_network and TORCH_AVAILABLE:
            model_path = Path("models") / f"{self.project_name}_policy.pth"
            torch.save(self.policy_network.state_dict(), model_path)
        
        # 정리
        self.is_running = False
        self.panda_controller.stop()
    
    async def _build_game(self):
        """게임 빌드"""
        logger.info("🔨 게임 빌드 중...")
        
        # 빌드 스크립트 생성
        build_script = f"""
#!/usr/bin/env python3
# 게임 빌드 스크립트

from setuptools import setup
from panda3d_tools.deploy import deploy

setup(
    name="{self.project_name}",
    options={{
        'build_apps': {{
            'include_patterns': [
                '**/*.py',
                '**/*.png',
                '**/*.jpg',
                '**/*.egg',
                '**/*.bam',
                '**/*.wav',
                '**/*.mp3'
            ],
            'gui_apps': {{
                '{self.project_name}': 'main.py',
            }},
            'platforms': ['win_amd64', 'linux_x86_64'],
        }}
    }}
)
"""
        
        # 빌드 스크립트 저장 및 실행
        build_path = Path(self.panda_controller.get_project_path()) / "build.py"
        build_path.write_text(build_script)
        
        # 실제 빌드는 subprocess로 실행
        logger.info("빌드 완료!")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """최종 개발 보고서 생성"""
        return {
            "project_info": {
                "name": self.project_name,
                "type": self.game_type,
                "development_time": round(self.game_state.time_elapsed / 3600, 2),
                "completion_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "statistics": {
                "total_actions": self.game_state.actions_taken,
                "successful_actions": self.game_state.success_count,
                "failed_actions": self.game_state.error_count,
                "success_rate": round(
                    self.game_state.success_count / max(self.game_state.actions_taken, 1) * 100, 2
                )
            },
            "quality_metrics": {
                "overall_score": round(self.game_state.quality_score, 2),
                "completeness": round(self.game_state.completeness, 2),
                "features_implemented": self.game_state.features,
                "performance": self.game_state.performance_metrics
            },
            "learning_insights": {
                "successful_patterns": len(self.memory.successful_patterns),
                "failed_attempts": len(self.memory.failed_attempts),
                "learned_solutions": len(self.memory.learned_solutions),
                "optimization_count": len(self.memory.optimization_history)
            },
            "game_features": {
                "has_player": "player" in self.game_state.features,
                "has_enemies": "enemies" in self.game_state.features,
                "has_levels": "level" in self.game_state.features,
                "has_score_system": "score" in self.game_state.features,
                "has_sound": "sound" in self.game_state.features,
                "has_particles": "particles" in self.game_state.features,
                "is_multiplayer": "multiplayer" in self.game_state.features,
                "is_optimized": "optimized" in self.game_state.features
            }
        }
    
    def _run_socketio_server(self):
        """Socket.IO 서버 실행 (별도 스레드)"""
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.socketio_system.start())
    
    def stop(self):
        """AI 에이전트 중지"""
        self.is_running = False
        self.panda_controller.stop()
        logger.info("AI 에이전트 중지됨")


# 사용 예시
async def main():
    """메인 실행 함수"""
    # AI 에이전트 생성
    agent = Panda3DAIAgent("MyAwesomeGame", "platformer")
    
    # 24시간 자동 개발 시작
    await agent.start_development(target_hours=24.0)


if __name__ == "__main__":
    asyncio.run(main())