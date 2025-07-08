#!/usr/bin/env python3
"""
PyTorch를 활용한 게임 AI 모듈
학습한 PyTorch 지식을 실제 게임 개발에 적용하는 예제
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)

class GameAIAgent(nn.Module):
    """게임 AI 에이전트를 위한 신경망"""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 128, output_size: int = 4):
        super(GameAIAgent, self).__init__()
        # 입력: 게임 상태 (플레이어 위치, 적 위치, 체력, 아이템 등)
        # 출력: 행동 (상, 하, 좌, 우)
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class GameAITrainer:
    """게임 AI 학습 관리자"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = GameAIAgent().to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        if model_path:
            self.load_model(model_path)
        
        # 경험 리플레이 버퍼
        self.experience_buffer = []
        self.max_buffer_size = 10000
        
    def collect_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """게임 플레이 경험 수집"""
        if len(self.experience_buffer) >= self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        })
    
    def train_batch(self, batch_size: int = 32) -> float:
        """배치 학습"""
        if len(self.experience_buffer) < batch_size:
            return 0.0
        
        # 랜덤 배치 샘플링
        indices = np.random.choice(len(self.experience_buffer), batch_size)
        batch = [self.experience_buffer[i] for i in indices]
        
        states = torch.tensor([exp['state'] for exp in batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32).to(self.device)
        
        # 순전파
        outputs = self.agent(states)
        loss = self.criterion(outputs, actions)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_action(self, state: np.ndarray) -> int:
        """현재 상태에서 최적의 행동 예측"""
        self.agent.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_probs = self.agent(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()
        return action
    
    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'experience_buffer': self.experience_buffer[-1000:]  # 최근 1000개만 저장
        }, path)
        logger.info(f"모델이 {path}에 저장되었습니다")
    
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.experience_buffer = checkpoint.get('experience_buffer', [])
        logger.info(f"모델이 {path}에서 로드되었습니다")

class GameAIIntegration:
    """Godot 게임과 PyTorch AI 통합"""
    
    def __init__(self):
        self.trainer = GameAITrainer()
        self.game_state = {}
        
    def generate_ai_script(self, game_type: str) -> str:
        """게임 타입에 맞는 AI 스크립트 생성"""
        
        if game_type == "platformer":
            return self._generate_platformer_ai()
        elif game_type == "rpg":
            return self._generate_rpg_ai()
        elif game_type == "puzzle":
            return self._generate_puzzle_ai()
        else:
            return self._generate_basic_ai()
    
    def _generate_platformer_ai(self) -> str:
        """플랫포머 게임용 AI 스크립트"""
        return '''extends CharacterBody2D

# PyTorch AI 통합
var ai_enabled = true
var state_buffer = []
var action_map = {
    0: "move_left",
    1: "move_right",
    2: "jump",
    3: "idle"
}

func _ready():
    # AI 상태 초기화
    state_buffer.resize(8)
    
func _physics_process(delta):
    if ai_enabled:
        var action = get_ai_action()
        execute_action(action)
    
func get_game_state():
    # 게임 상태 수집
    var state = []
    state.append(position.x / 1000.0)  # 정규화된 X 위치
    state.append(position.y / 1000.0)  # 정규화된 Y 위치
    state.append(velocity.x / 100.0)   # 정규화된 X 속도
    state.append(velocity.y / 100.0)   # 정규화된 Y 속도
    
    # 주변 환경 정보 (레이캐스트 사용)
    state.append(1.0 if is_on_floor() else 0.0)
    state.append(get_nearest_enemy_distance() / 1000.0)
    state.append(get_nearest_platform_height() / 100.0)
    state.append(get_player_health() / 100.0)
    
    return state

func get_ai_action():
    var state = get_game_state()
    
    # PyTorch 모델 호출 (실제로는 HTTP API나 GDExtension 사용)
    # 여기서는 간단한 휴리스틱으로 대체
    var action_probabilities = simple_ai_logic(state)
    
    # 확률적 행동 선택
    var action = weighted_random_choice(action_probabilities)
    return action

func simple_ai_logic(state):
    # 간단한 AI 로직 (실제 PyTorch 모델로 대체 가능)
    var probs = [0.25, 0.25, 0.25, 0.25]
    
    # 바닥에 있고 적이 가까우면 점프 확률 증가
    if state[4] > 0.5 and state[5] < 0.3:
        probs[2] = 0.6  # 점프
        probs[3] = 0.1  # 대기 감소
    
    return probs

func execute_action(action_index):
    var action = action_map[action_index]
    
    match action:
        "move_left":
            velocity.x = -SPEED
        "move_right":
            velocity.x = SPEED
        "jump":
            if is_on_floor():
                velocity.y = JUMP_VELOCITY
        "idle":
            velocity.x = 0

func weighted_random_choice(probabilities):
    var sum = 0.0
    for p in probabilities:
        sum += p
    
    var random_value = randf() * sum
    var cumulative = 0.0
    
    for i in range(probabilities.size()):
        cumulative += probabilities[i]
        if random_value <= cumulative:
            return i
    
    return probabilities.size() - 1
'''
    
    def _generate_rpg_ai(self) -> str:
        """RPG 게임용 AI 스크립트"""
        return '''extends CharacterBody2D

# RPG AI 시스템
class_name RPGAICharacter

enum AIState {
    IDLE,
    PATROL,
    CHASE,
    ATTACK,
    FLEE
}

var current_state = AIState.IDLE
var ai_decision_timer = 0.0
var decision_interval = 0.5  # 0.5초마다 AI 결정

# PyTorch 모델 정보
var model_weights = {}
var hidden_layer_size = 128

func _ready():
    # AI 모델 가중치 로드 (실제로는 파일에서 로드)
    initialize_ai_model()

func _physics_process(delta):
    ai_decision_timer += delta
    
    if ai_decision_timer >= decision_interval:
        ai_decision_timer = 0.0
        make_ai_decision()
    
    execute_current_state()

func get_state_vector():
    # RPG용 상태 벡터 생성
    var state = []
    
    # 자신의 상태
    state.append(health / max_health)
    state.append(mana / max_mana)
    state.append(position.x / 1000.0)
    state.append(position.y / 1000.0)
    
    # 플레이어 정보
    var player = get_player()
    if player:
        var distance = position.distance_to(player.position)
        state.append(distance / 1000.0)
        state.append((player.position.x - position.x) / 1000.0)
        state.append((player.position.y - position.y) / 1000.0)
        state.append(player.health / player.max_health)
    else:
        state.extend([1.0, 0.0, 0.0, 1.0])
    
    # 주변 상황
    state.append(get_allies_nearby())
    state.append(get_enemies_nearby())
    state.append(float(has_line_of_sight_to_player()))
    state.append(float(is_in_combat()))
    
    return state

func make_ai_decision():
    var state = get_state_vector()
    var action_probs = neural_network_forward(state)
    
    # 가장 높은 확률의 행동 선택
    var max_prob = 0.0
    var selected_action = 0
    
    for i in range(action_probs.size()):
        if action_probs[i] > max_prob:
            max_prob = action_probs[i]
            selected_action = i
    
    # 행동을 상태로 변환
    match selected_action:
        0: current_state = AIState.IDLE
        1: current_state = AIState.PATROL
        2: current_state = AIState.CHASE
        3: current_state = AIState.ATTACK
        4: current_state = AIState.FLEE

func neural_network_forward(input_vector):
    # 간단한 신경망 순전파 (실제 PyTorch 모델 흉내)
    # 실제로는 GDExtension이나 HTTP API로 PyTorch 모델 호출
    
    var hidden = []
    var output = []
    
    # 은닉층 계산 (ReLU 활성화)
    for i in range(hidden_layer_size):
        var sum = 0.0
        for j in range(input_vector.size()):
            # 실제 가중치 사용 (여기서는 임의값)
            sum += input_vector[j] * randf_range(-0.1, 0.1)
        hidden.append(max(0.0, sum))  # ReLU
    
    # 출력층 계산 (Softmax)
    var output_raw = []
    for i in range(5):  # 5개 행동
        var sum = 0.0
        for j in range(hidden.size()):
            sum += hidden[j] * randf_range(-0.1, 0.1)
        output_raw.append(sum)
    
    # Softmax 적용
    var max_val = output_raw.max()
    var sum_exp = 0.0
    
    for val in output_raw:
        sum_exp += exp(val - max_val)
    
    for val in output_raw:
        output.append(exp(val - max_val) / sum_exp)
    
    return output
'''
    
    def _generate_puzzle_ai(self) -> str:
        """퍼즐 게임용 AI 스크립트"""
        return '''extends Node2D

# 퍼즐 AI 솔버
class_name PuzzleAISolver

var board_state = []
var board_size = Vector2i(8, 8)
var ai_thinking = false

# 신경망 파라미터
var conv_filters = []  # CNN 필터들

func _ready():
    initialize_cnn_model()

func analyze_board():
    # 보드 상태를 CNN 입력으로 변환
    var input_tensor = board_to_tensor()
    var best_move = cnn_forward(input_tensor)
    return best_move

func board_to_tensor():
    # 2D 보드를 3D 텐서로 변환 (채널 포함)
    var tensor = []
    
    for y in range(board_size.y):
        var row = []
        for x in range(board_size.x):
            var cell = get_cell(x, y)
            # 원-핫 인코딩 (예: 6가지 색상)
            var encoded = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if cell >= 0 and cell < 6:
                encoded[cell] = 1.0
            row.append(encoded)
        tensor.append(row)
    
    return tensor

func cnn_forward(input_tensor):
    # 간단한 CNN 순전파 구현
    # 실제로는 PyTorch 모델 사용
    
    # Conv2D 레이어 (3x3 필터)
    var feature_maps = apply_convolution(input_tensor)
    
    # MaxPooling
    var pooled = apply_max_pooling(feature_maps)
    
    # Flatten & Dense
    var flattened = flatten_tensor(pooled)
    var output = dense_layer(flattened)
    
    # 최적의 움직임 선택
    return argmax(output)

func apply_convolution(input):
    # 3x3 컨볼루션 적용
    var output = []
    var filter_size = 3
    var stride = 1
    
    for y in range(0, board_size.y - filter_size + 1, stride):
        var row = []
        for x in range(0, board_size.x - filter_size + 1, stride):
            var conv_result = 0.0
            
            # 필터 적용
            for fy in range(filter_size):
                for fx in range(filter_size):
                    # 간단한 엣지 검출 필터
                    var filter_val = 1.0 if fy == 1 and fx == 1 else -0.125
                    conv_result += get_tensor_value(input, x + fx, y + fy) * filter_val
            
            row.append(max(0.0, conv_result))  # ReLU
        output.append(row)
    
    return output

func get_ai_hint():
    # 플레이어에게 힌트 제공
    if ai_thinking:
        return
    
    ai_thinking = true
    var best_move = analyze_board()
    
    # 힌트 시각화
    highlight_suggestion(best_move)
    ai_thinking = false
'''
    
    def _generate_basic_ai(self) -> str:
        """기본 AI 스크립트"""
        return '''extends Node

# 기본 AI 시스템
class_name BasicAI

signal decision_made(action)

var input_size = 4
var hidden_size = 16
var output_size = 4

# 간단한 신경망 가중치
var weights_ih = []  # input to hidden
var weights_ho = []  # hidden to output
var bias_h = []
var bias_o = []

func _ready():
    initialize_network()

func initialize_network():
    # 가중치 초기화 (Xavier 초기화)
    for i in range(input_size):
        var row = []
        for j in range(hidden_size):
            row.append(randf_range(-1.0, 1.0) * sqrt(2.0 / input_size))
        weights_ih.append(row)
    
    for i in range(hidden_size):
        var row = []
        for j in range(output_size):
            row.append(randf_range(-1.0, 1.0) * sqrt(2.0 / hidden_size))
        weights_ho.append(row)
    
    for i in range(hidden_size):
        bias_h.append(0.0)
    
    for i in range(output_size):
        bias_o.append(0.0)

func forward(input_vector):
    # 순전파
    var hidden = []
    
    # 은닉층
    for j in range(hidden_size):
        var sum = bias_h[j]
        for i in range(input_size):
            sum += input_vector[i] * weights_ih[i][j]
        hidden.append(tanh(sum))  # tanh 활성화
    
    # 출력층
    var output = []
    for j in range(output_size):
        var sum = bias_o[j]
        for i in range(hidden_size):
            sum += hidden[i] * weights_ho[i][j]
        output.append(sum)
    
    # Softmax
    return softmax(output)

func softmax(values):
    var max_val = values.max()
    var exp_values = []
    var sum_exp = 0.0
    
    for val in values:
        var exp_val = exp(val - max_val)
        exp_values.append(exp_val)
        sum_exp += exp_val
    
    var result = []
    for exp_val in exp_values:
        result.append(exp_val / sum_exp)
    
    return result

func make_decision(game_state):
    var action_probs = forward(game_state)
    var action = select_action(action_probs)
    emit_signal("decision_made", action)
    return action

func select_action(probabilities):
    # 확률적 선택 또는 최대값 선택
    if randf() < 0.1:  # 10% 탐험
        return randi() % probabilities.size()
    else:  # 90% 활용
        return probabilities.find(probabilities.max())
'''

def integrate_pytorch_with_godot(game_type: str, project_path: str) -> Dict[str, str]:
    """PyTorch AI를 Godot 게임에 통합"""
    integration = GameAIIntegration()
    
    # AI 스크립트 생성
    ai_script = integration.generate_ai_script(game_type)
    
    # 통합 가이드 생성
    guide = f"""
# PyTorch AI 통합 가이드

## 1. AI 스크립트 설치
생성된 AI 스크립트를 프로젝트의 적절한 위치에 저장하세요:
- 플랫포머: `res://scripts/ai/platformer_ai.gd`
- RPG: `res://scripts/ai/rpg_ai.gd`
- 퍼즐: `res://scripts/ai/puzzle_ai.gd`

## 2. PyTorch 모델 연동 방법

### 방법 1: HTTP API (권장)
```python
# Python 서버 (FastAPI)
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
async def predict(state: List[float]):
    action = trainer.predict_action(np.array(state))
    return {{"action": action}}
```

### 방법 2: GDExtension
C++로 PyTorch 모델을 래핑하여 직접 연동

### 방법 3: 모델 변환
PyTorch 모델을 ONNX로 변환 후 Godot에서 사용

## 3. 학습 데이터 수집
게임 플레이 중 상태와 행동을 기록하여 AI 개선에 활용

## 4. 실시간 학습
게임 실행 중에도 AI가 계속 학습하도록 설정 가능
"""
    
    return {
        "ai_script": ai_script,
        "integration_guide": guide,
        "model_path": f"models/{game_type}_ai_model.pth"
    }