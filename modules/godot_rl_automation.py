#!/usr/bin/env python3
"""
Godot 강화학습 자동화 시스템
화면 인식과 가상 입력을 통한 지능형 게임 개발
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import deque
import threading
import queue

# PyTorch 강화학습
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# 고급 컨트롤러
sys.path.insert(0, str(Path(__file__).parent))
from advanced_godot_controller import (
    GodotAutomationController, 
    GodotUIElement,
    UIElementInfo
)

logger = logging.getLogger(__name__)

class GodotEnvironment:
    """Godot 에디터를 강화학습 환경으로 래핑"""
    
    def __init__(self):
        self.controller = GodotAutomationController()
        self.action_space = self._define_action_space()
        self.observation_space = 224 * 224 * 3  # 이미지 크기
        
        # 상태 기록
        self.previous_state = None
        self.current_state = None
        self.episode_steps = 0
        self.max_steps_per_episode = 1000
        
        # 보상 설정
        self.reward_config = {
            "node_created": 10.0,
            "property_set": 5.0,
            "script_attached": 15.0,
            "scene_saved": 20.0,
            "error_occurred": -10.0,
            "idle_penalty": -0.1,
            "invalid_action": -5.0,
            "task_completed": 100.0
        }
        
    def _define_action_space(self) -> List[Dict]:
        """가능한 액션 정의"""
        return [
            # 마우스 액션
            {"type": "click", "params": {"x_offset": 0, "y_offset": 0}},
            {"type": "double_click", "params": {}},
            {"type": "right_click", "params": {}},
            {"type": "drag", "params": {"dx": 100, "dy": 0}},
            
            # 키보드 액션
            {"type": "key", "params": {"key": "ctrl+a"}},  # Add Node
            {"type": "key", "params": {"key": "ctrl+s"}},  # Save
            {"type": "key", "params": {"key": "f5"}},       # Play
            {"type": "key", "params": {"key": "f6"}},       # Play Scene
            {"type": "key", "params": {"key": "delete"}},   # Delete
            
            # 텍스트 입력
            {"type": "type", "params": {"text": "Node2D"}},
            {"type": "type", "params": {"text": "Sprite2D"}},
            {"type": "type", "params": {"text": "CollisionShape2D"}},
            {"type": "type", "params": {"text": "CharacterBody2D"}},
            
            # 메뉴 네비게이션
            {"type": "menu", "params": {"path": ["Scene", "New Scene"]}},
            {"type": "menu", "params": {"path": ["Scene", "Save Scene"]}},
            {"type": "menu", "params": {"path": ["Project", "Project Settings"]}},
            
            # UI 요소 클릭
            {"type": "click_ui", "params": {"element": "scene_dock"}},
            {"type": "click_ui", "params": {"element": "inspector"}},
            {"type": "click_ui", "params": {"element": "file_system"}},
            
            # 특수 액션
            {"type": "wait", "params": {"duration": 0.5}},
            {"type": "scan_ui", "params": {}}
        ]
        
    def reset(self) -> np.ndarray:
        """환경 리셋"""
        self.episode_steps = 0
        
        # UI 상태 스캔
        self.controller.scan_ui_state()
        
        # 초기 상태 캡처
        state = self._get_current_state()
        self.current_state = state
        self.previous_state = state
        
        return state
        
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """액션 수행 및 결과 반환"""
        self.episode_steps += 1
        
        # 이전 상태 저장
        self.previous_state = self.current_state
        
        # 액션 실행
        action = self.action_space[action_idx]
        reward = self._execute_action(action)
        
        # 새로운 상태 관찰
        self.current_state = self._get_current_state()
        
        # 에피소드 종료 조건
        done = (
            self.episode_steps >= self.max_steps_per_episode or
            self._is_task_completed() or
            self._is_error_state()
        )
        
        # 추가 정보
        info = {
            "action": action,
            "ui_state": self.controller.ui_state,
            "episode_steps": self.episode_steps
        }
        
        return self.current_state, reward, done, info
        
    def _execute_action(self, action: Dict) -> float:
        """액션 실행 및 보상 계산"""
        reward = 0.0
        
        try:
            if action["type"] == "click":
                # 현재 마우스 위치에서 오프셋만큼 이동 후 클릭
                x, y = self.controller.mouse.get_cursor_pos()
                x += action["params"]["x_offset"]
                y += action["params"]["y_offset"]
                self.controller.mouse.move_mouse_natural(x, y)
                self.controller.mouse.click_precise()
                reward = 0.1
                
            elif action["type"] == "key":
                keys = action["params"]["key"].split("+")
                self.controller.keyboard.send_key_combination(*keys)
                reward = 0.5
                
            elif action["type"] == "type":
                text = action["params"]["text"]
                self.controller.keyboard.type_with_timing(text)
                reward = 1.0
                
            elif action["type"] == "menu":
                path = action["params"]["path"]
                if self.controller.navigate_to_menu(path):
                    reward = 2.0
                else:
                    reward = self.reward_config["invalid_action"]
                    
            elif action["type"] == "click_ui":
                element_name = action["params"]["element"]
                element = self.controller.ui_state.get(element_name)
                if element:
                    x = element.position[0] + element.size[0] // 2
                    y = element.position[1] + element.size[1] // 2
                    self.controller.mouse.move_mouse_natural(x, y)
                    self.controller.mouse.click_precise()
                    reward = 1.0
                else:
                    reward = self.reward_config["invalid_action"]
                    
            elif action["type"] == "scan_ui":
                self.controller.scan_ui_state()
                reward = 0.1
                
            elif action["type"] == "wait":
                time.sleep(action["params"]["duration"])
                reward = self.reward_config["idle_penalty"]
                
            # 상태 변화에 따른 추가 보상
            reward += self._calculate_state_change_reward()
            
        except Exception as e:
            logger.error(f"액션 실행 오류: {e}")
            reward = self.reward_config["error_occurred"]
            
        return reward
        
    def _get_current_state(self) -> np.ndarray:
        """현재 상태 (화면) 캡처"""
        screenshot = self.controller.vision.capture_screen_fast()
        
        # 크기 조정 및 정규화
        import cv2
        screenshot = cv2.resize(screenshot, (224, 224))
        screenshot = screenshot.astype(np.float32) / 255.0
        
        return screenshot.flatten()
        
    def _calculate_state_change_reward(self) -> float:
        """상태 변화에 따른 보상 계산"""
        reward = 0.0
        
        # UI 요소 변화 감지
        current_elements = set(self.controller.ui_state.keys())
        
        # 노드가 추가되었는지 확인
        scene_dock = self.controller.ui_state.get(GodotUIElement.SCENE_DOCK.value)
        if scene_dock and "Node2D" in scene_dock.text_content:
            reward += self.reward_config["node_created"] * 0.1
            
        return reward
        
    def _is_task_completed(self) -> bool:
        """작업 완료 여부 확인"""
        # 예: 특정 씬이 생성되고 저장되었는지
        return False
        
    def _is_error_state(self) -> bool:
        """에러 상태 확인"""
        # 에러 다이얼로그 감지
        return False

class A3CNetwork(nn.Module):
    """Actor-Critic 네트워크"""
    
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        
        # 공유 레이어
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 계산된 특징 크기
        conv_out_size = self._get_conv_out_size((3, 224, 224))
        
        self.fc = nn.Linear(conv_out_size, 512)
        
        # Actor (정책)
        self.actor = nn.Linear(512, num_actions)
        
        # Critic (가치)
        self.critic = nn.Linear(512, 1)
        
    def _get_conv_out_size(self, shape):
        """Conv 레이어 출력 크기 계산"""
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
        
    def forward(self, x):
        # 입력 reshape (flatten -> image)
        x = x.view(-1, 3, 224, 224)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        # Shared FC
        x = F.relu(self.fc(x))
        
        # Actor-Critic outputs
        policy = F.softmax(self.actor(x), dim=1)
        value = self.critic(x)
        
        return policy, value

class GodotRLAgent:
    """강화학습 에이전트"""
    
    def __init__(self, env: GodotEnvironment):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 네트워크
        self.network = A3CNetwork(
            input_dim=env.observation_space,
            num_actions=len(env.action_space)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.0001)
        
        # 경험 버퍼
        self.memory = []
        self.gamma = 0.99
        self.tau = 0.95
        
        # 학습 통계
        self.episode_rewards = []
        self.episode_lengths = []
        
    def select_action(self, state: np.ndarray) -> int:
        """액션 선택"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, _ = self.network(state_tensor)
            
        # 확률적 액션 선택
        dist = Categorical(policy)
        action = dist.sample()
        
        return action.item()
        
    def train_episode(self) -> float:
        """한 에피소드 학습"""
        state = self.env.reset()
        episode_reward = 0
        
        while True:
            # 액션 선택
            action = self.select_action(state)
            
            # 환경에서 실행
            next_state, reward, done, info = self.env.step(action)
            
            # 메모리에 저장
            self.memory.append((state, action, reward, next_state, done))
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
                
        # 에피소드 종료 후 학습
        self.update()
        
        # 통계 기록
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(self.env.episode_steps)
        
        return episode_reward
        
    def update(self):
        """네트워크 업데이트"""
        if len(self.memory) == 0:
            return
            
        # 메모리에서 데이터 추출
        states, actions, rewards, next_states, dones = zip(*self.memory)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 현재 가치 계산
        policies, values = self.network(states)
        _, next_values = self.network(next_states)
        
        # TD 타겟
        targets = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        
        # 손실 계산
        value_loss = F.mse_loss(values.squeeze(), targets.detach())
        
        # 정책 손실 (REINFORCE with baseline)
        advantages = targets - values.squeeze()
        log_probs = torch.log(policies.gather(1, actions.unsqueeze(1)) + 1e-8)
        policy_loss = -(log_probs.squeeze() * advantages.detach()).mean()
        
        # 엔트로피 보너스 (탐색 촉진)
        entropy = -(policies * torch.log(policies + 1e-8)).sum(dim=1).mean()
        
        # 전체 손실
        total_loss = value_loss + policy_loss - 0.01 * entropy
        
        # 역전파
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        # 메모리 초기화
        self.memory = []
        
    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, path)
        
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])

class GodotAutomationOrchestrator:
    """자동화 오케스트레이터"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.env = GodotEnvironment()
        self.agent = GodotRLAgent(self.env)
        
        # 작업 큐
        self.task_queue = queue.Queue()
        self.current_task = None
        
        # 학습 스레드
        self.training_thread = None
        self.is_training = False
        
    def add_task(self, task_type: str, params: Dict = None):
        """작업 추가"""
        task = {
            "type": task_type,
            "params": params or {},
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        self.task_queue.put(task)
        
    def start_training(self, num_episodes: int = 1000):
        """학습 시작"""
        self.is_training = True
        self.training_thread = threading.Thread(
            target=self._training_loop,
            args=(num_episodes,)
        )
        self.training_thread.start()
        
    def _training_loop(self, num_episodes: int):
        """학습 루프"""
        logger.info(f"🚀 강화학습 시작 ({num_episodes} 에피소드)")
        
        for episode in range(num_episodes):
            if not self.is_training:
                break
                
            # 작업이 있으면 처리
            if not self.task_queue.empty():
                self.current_task = self.task_queue.get()
                logger.info(f"작업 시작: {self.current_task['type']}")
                
            # 에피소드 학습
            reward = self.agent.train_episode()
            
            # 진행 상황 출력
            if episode % 10 == 0:
                avg_reward = np.mean(self.agent.episode_rewards[-10:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
                
            # 주기적 저장
            if episode % 100 == 0:
                self.agent.save_model(f"godot_rl_model_ep{episode}.pth")
                
        logger.info("✅ 학습 완료")
        
    def stop_training(self):
        """학습 중지"""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join()
            
    def execute_learned_task(self, task_type: str):
        """학습된 작업 실행"""
        logger.info(f"🤖 학습된 작업 실행: {task_type}")
        
        state = self.env.reset()
        total_reward = 0
        
        for step in range(self.env.max_steps_per_episode):
            # 최적 액션 선택 (탐욕적)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                policy, _ = self.agent.network(state_tensor)
                action = policy.argmax().item()
                
            # 실행
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if done:
                break
                
        logger.info(f"✅ 작업 완료. 총 보상: {total_reward:.2f}")
        return total_reward

# 사용 예시
if __name__ == "__main__":
    # 프로젝트 경로
    project_path = "D:/MyGodotProject"
    
    # 오케스트레이터 생성
    orchestrator = GodotAutomationOrchestrator(project_path)
    
    # 작업 추가
    orchestrator.add_task("create_2d_platformer")
    orchestrator.add_task("create_ui_system")
    orchestrator.add_task("setup_player_controller")
    
    # 학습 시작
    orchestrator.start_training(num_episodes=500)
    
    # 학습 완료 대기
    input("학습 중... Enter를 눌러 중지: ")
    orchestrator.stop_training()
    
    # 학습된 모델로 작업 실행
    orchestrator.agent.load_model("godot_rl_model_ep400.pth")
    orchestrator.execute_learned_task("create_2d_platformer")