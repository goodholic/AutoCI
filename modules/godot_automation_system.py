#!/usr/bin/env python3
"""
AutoCI Godot 자동화 시스템
화면 인식과 가상 입력을 통한 게임 개발 자동화
PyTorch 강화학습으로 지속적인 개선
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

# 화면 캡처 및 인식
try:
    import cv2
    import pyautogui
    import pytesseract
    from PIL import Image, ImageGrab
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV 또는 pyautogui가 설치되지 않았습니다. pip install opencv-python pyautogui pillow pytesseract")

# PyTorch 강화학습
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# Windows 전용 모듈
if sys.platform == "win32":
    try:
        import win32api
        import win32con
        import win32gui
        WIN32_AVAILABLE = True
    except ImportError:
        WIN32_AVAILABLE = False
        print("⚠️ pywin32가 설치되지 않았습니다. pip install pywin32")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GodotScreenRecognizer:
    """Godot 에디터 화면 인식 시스템"""
    
    def __init__(self):
        self.template_dir = Path(__file__).parent / "templates"
        self.template_dir.mkdir(exist_ok=True)
        
        # 인식할 UI 요소 템플릿
        self.ui_elements = {
            "file_menu": "file_menu.png",
            "scene_panel": "scene_panel.png",
            "inspector": "inspector.png",
            "node_button": "node_button.png",
            "script_editor": "script_editor.png",
            "play_button": "play_button.png",
            "save_button": "save_button.png"
        }
        
        # 화면 영역 정의
        self.regions = {
            "menu_bar": (0, 0, 1920, 50),
            "scene_panel": (0, 50, 400, 600),
            "viewport": (400, 50, 1520, 900),
            "inspector": (1520, 50, 400, 900),
            "bottom_panel": (0, 900, 1920, 180)
        }
        
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """화면 캡처"""
        if not CV2_AVAILABLE:
            return None
            
        if region:
            screenshot = ImageGrab.grab(bbox=region)
        else:
            screenshot = ImageGrab.grab()
            
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
    def find_element(self, template_name: str, threshold: float = 0.8) -> Optional[Tuple[int, int]]:
        """템플릿 매칭으로 UI 요소 찾기"""
        if not CV2_AVAILABLE:
            return None
            
        screen = self.capture_screen()
        template_path = self.template_dir / self.ui_elements.get(template_name, template_name)
        
        if not template_path.exists():
            logger.warning(f"템플릿 파일을 찾을 수 없습니다: {template_path}")
            return None
            
        template = cv2.imread(str(template_path))
        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            # 템플릿 중심점 반환
            h, w = template.shape[:2]
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return (center_x, center_y)
            
        return None
        
    def read_text_from_region(self, region: Tuple[int, int, int, int]) -> str:
        """특정 영역의 텍스트 읽기"""
        if not CV2_AVAILABLE:
            return ""
            
        screen = self.capture_screen(region)
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        try:
            text = pytesseract.image_to_string(gray)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR 오류: {str(e)}")
            return ""
            
    def get_scene_tree_state(self) -> Dict[str, Any]:
        """씬 트리 상태 파악"""
        # 씬 패널 영역 캡처
        scene_region = self.regions["scene_panel"]
        scene_text = self.read_text_from_region(scene_region)
        
        # 간단한 파싱 (실제로는 더 정교한 처리 필요)
        nodes = []
        for line in scene_text.split('\n'):
            if line.strip():
                nodes.append(line.strip())
                
        return {
            "nodes": nodes,
            "node_count": len(nodes),
            "timestamp": datetime.now().isoformat()
        }

class GodotAutomationAgent(nn.Module):
    """Godot 자동화를 위한 강화학습 에이전트"""
    
    def __init__(self, state_size: int = 512, action_size: int = 50):
        super().__init__()
        
        # 상태 인코더 (CNN for screen)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 상태 처리
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # 액션 출력
        self.action_head = nn.Linear(256, action_size)
        self.value_head = nn.Linear(256, 1)
        
    def forward(self, x):
        # CNN 특징 추출
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        # 완전 연결 레이어
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 액션과 가치 출력
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        
        return action_probs, state_value

class VirtualInputController:
    """가상 마우스/키보드 컨트롤러"""
    
    def __init__(self):
        pyautogui.FAILSAFE = True  # 안전 장치
        pyautogui.PAUSE = 0.1  # 각 액션 사이 대기 시간
        
    def move_mouse(self, x: int, y: int, duration: float = 0.5):
        """마우스 이동"""
        pyautogui.moveTo(x, y, duration=duration)
        
    def click(self, x: int = None, y: int = None, button: str = 'left'):
        """마우스 클릭"""
        if x and y:
            pyautogui.click(x, y, button=button)
        else:
            pyautogui.click(button=button)
            
    def double_click(self, x: int = None, y: int = None):
        """더블 클릭"""
        if x and y:
            pyautogui.doubleClick(x, y)
        else:
            pyautogui.doubleClick()
            
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 1.0):
        """드래그"""
        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, duration=duration)
        
    def type_text(self, text: str, interval: float = 0.05):
        """텍스트 입력"""
        pyautogui.typewrite(text, interval=interval)
        
    def hotkey(self, *keys):
        """단축키 입력"""
        pyautogui.hotkey(*keys)
        
    def scroll(self, clicks: int, x: int = None, y: int = None):
        """스크롤"""
        if x and y:
            pyautogui.moveTo(x, y)
        pyautogui.scroll(clicks)

class ExperienceMemory:
    """강화학습용 경험 메모리"""
    
    def __init__(self, capacity: int = 10000):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        """배치 샘플링"""
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

class GodotAutomationSystem:
    """통합 Godot 자동화 시스템"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.recognizer = GodotScreenRecognizer()
        self.controller = VirtualInputController()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 강화학습 설정
        self.agent = GodotAutomationAgent().to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.0003)
        self.memory = ExperienceMemory()
        
        # 액션 정의
        self.actions = self._define_actions()
        
        # 사용자 요청 자원 폴더
        self.user_assets_dir = self.project_path / "user_requested_assets"
        self._setup_asset_directories()
        
        # 학습 기록
        self.learning_history = []
        self.success_count = 0
        self.total_attempts = 0
        
    def _setup_asset_directories(self):
        """사용자 요청 자원 폴더 구조 설정"""
        directories = [
            "images/sprites",
            "images/textures",
            "images/ui",
            "animations/2d",
            "animations/3d",
            "models/characters",
            "models/environment",
            "models/props",
            "audio/music",
            "audio/sfx",
            "audio/voice",
            "materials",
            "shaders",
            "fonts",
            "scenes/templates"
        ]
        
        for dir_path in directories:
            full_path = self.user_assets_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
            # README 파일 생성
            readme_path = full_path / "README.md"
            if not readme_path.exists():
                readme_content = f"""# {dir_path.replace('/', ' > ')}

이 폴더에 필요한 자원을 넣어주세요.

## 지원 형식
"""
                if "images" in dir_path:
                    readme_content += "- PNG, JPG, JPEG, WEBP\n- 권장 해상도: 2의 제곱수 (512x512, 1024x1024 등)\n"
                elif "animations" in dir_path:
                    readme_content += "- GIF, WEBM, MP4\n- 스프라이트 시트 지원\n"
                elif "models" in dir_path:
                    readme_content += "- GLTF, GLB, OBJ, FBX\n- 텍스처 포함 권장\n"
                elif "audio" in dir_path:
                    readme_content += "- OGG, MP3, WAV\n- 루프 가능한 형식 권장\n"
                    
                readme_path.write_text(readme_content, encoding='utf-8')
                
        # 요청 목록 파일
        request_file = self.user_assets_dir / "asset_requests.json"
        if not request_file.exists():
            request_file.write_text(json.dumps({
                "requests": [],
                "completed": [],
                "pending": []
            }, indent=2), encoding='utf-8')
            
    def _define_actions(self) -> List[Dict[str, Any]]:
        """가능한 액션 정의"""
        return [
            # 기본 마우스 액션
            {"type": "click", "params": {"button": "left"}},
            {"type": "click", "params": {"button": "right"}},
            {"type": "double_click", "params": {}},
            
            # 메뉴 액션
            {"type": "menu", "params": {"menu": "File", "item": "New Scene"}},
            {"type": "menu", "params": {"menu": "File", "item": "Save Scene"}},
            {"type": "menu", "params": {"menu": "Scene", "item": "New Node"}},
            
            # 노드 생성
            {"type": "create_node", "params": {"node_type": "Node2D"}},
            {"type": "create_node", "params": {"node_type": "Sprite2D"}},
            {"type": "create_node", "params": {"node_type": "CollisionShape2D"}},
            {"type": "create_node", "params": {"node_type": "AnimationPlayer"}},
            
            # 속성 편집
            {"type": "edit_property", "params": {"property": "position"}},
            {"type": "edit_property", "params": {"property": "scale"}},
            {"type": "edit_property", "params": {"property": "rotation"}},
            
            # 스크립트 액션
            {"type": "attach_script", "params": {}},
            {"type": "edit_script", "params": {}},
            
            # 자원 관리
            {"type": "import_asset", "params": {"asset_type": "texture"}},
            {"type": "import_asset", "params": {"asset_type": "model"}},
            
            # 테스트
            {"type": "play_scene", "params": {}},
            {"type": "stop_scene", "params": {}},
        ]
        
    def capture_state(self) -> torch.Tensor:
        """현재 상태 캡처"""
        screen = self.recognizer.capture_screen()
        if screen is None:
            return torch.zeros((3, 224, 224))
            
        # 크기 조정 및 정규화
        screen = cv2.resize(screen, (224, 224))
        screen = screen.transpose(2, 0, 1)  # HWC -> CHW
        screen = screen.astype(np.float32) / 255.0
        
        return torch.from_numpy(screen)
        
    def execute_action(self, action_idx: int) -> float:
        """액션 실행 및 보상 계산"""
        action = self.actions[action_idx]
        reward = 0.0
        
        try:
            if action["type"] == "click":
                self.controller.click()
                reward = 0.1
                
            elif action["type"] == "create_node":
                # 노드 생성 시퀀스
                self.controller.hotkey('ctrl', 'a')  # Add Node
                time.sleep(0.5)
                self.controller.type_text(action["params"]["node_type"])
                time.sleep(0.5)
                self.controller.hotkey('enter')
                reward = 1.0
                
            elif action["type"] == "play_scene":
                play_pos = self.recognizer.find_element("play_button")
                if play_pos:
                    self.controller.click(*play_pos)
                    reward = 0.5
                    
            # 성공적인 액션 기록
            self.success_count += 1
            
        except Exception as e:
            logger.error(f"액션 실행 오류: {str(e)}")
            reward = -1.0
            
        self.total_attempts += 1
        return reward
        
    def train_step(self, batch_size: int = 32):
        """강화학습 훈련 스텝"""
        if len(self.memory) < batch_size:
            return
            
        # 배치 샘플링
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        
        # 현재 Q값 계산
        action_probs, state_values = self.agent(states)
        
        # 다음 상태의 가치 계산
        with torch.no_grad():
            _, next_values = self.agent(next_states)
            next_values[dones] = 0.0
            
        # TD 타겟
        targets = rewards + 0.99 * next_values.squeeze()
        
        # 손실 계산
        value_loss = F.mse_loss(state_values.squeeze(), targets)
        
        # 정책 손실 (간단한 버전)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1))
        policy_loss = -torch.log(selected_probs + 1e-8).mean()
        
        # 전체 손실
        total_loss = value_loss + 0.5 * policy_loss
        
        # 역전파
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss.item()
        
    def automated_workflow(self, task_description: str):
        """자동화된 작업 수행"""
        logger.info(f"작업 시작: {task_description}")
        
        # 작업을 액션 시퀀스로 변환 (간단한 예시)
        if "sprite" in task_description.lower():
            action_sequence = [
                self.actions.index({"type": "create_node", "params": {"node_type": "Sprite2D"}}),
                self.actions.index({"type": "import_asset", "params": {"asset_type": "texture"}})
            ]
        elif "collision" in task_description.lower():
            action_sequence = [
                self.actions.index({"type": "create_node", "params": {"node_type": "CollisionShape2D"}})
            ]
        else:
            # 강화학습으로 액션 선택
            state = self.capture_state().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs, _ = self.agent(state)
            action_sequence = [action_probs.argmax().item()]
            
        # 액션 실행
        total_reward = 0
        for action_idx in action_sequence:
            reward = self.execute_action(action_idx)
            total_reward += reward
            time.sleep(1)  # 액션 간 대기
            
        logger.info(f"작업 완료. 총 보상: {total_reward}")
        
        # 학습 기록
        self.learning_history.append({
            "task": task_description,
            "reward": total_reward,
            "success_rate": self.success_count / max(1, self.total_attempts),
            "timestamp": datetime.now().isoformat()
        })
        
    def request_user_asset(self, asset_type: str, description: str) -> str:
        """사용자에게 자원 요청"""
        request_file = self.user_assets_dir / "asset_requests.json"
        
        # 기존 요청 로드
        with open(request_file, 'r', encoding='utf-8') as f:
            requests_data = json.load(f)
            
        # 새 요청 추가
        new_request = {
            "id": f"req_{int(time.time())}",
            "type": asset_type,
            "description": description,
            "status": "pending",
            "requested_at": datetime.now().isoformat(),
            "folder": self._get_asset_folder(asset_type)
        }
        
        requests_data["requests"].append(new_request)
        requests_data["pending"].append(new_request["id"])
        
        # 저장
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(requests_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"자원 요청 생성: {asset_type} - {description}")
        return new_request["folder"]
        
    def _get_asset_folder(self, asset_type: str) -> str:
        """자원 타입에 따른 폴더 경로 반환"""
        type_map = {
            "sprite": "images/sprites",
            "texture": "images/textures",
            "ui": "images/ui",
            "animation": "animations/2d",
            "model": "models/props",
            "character": "models/characters",
            "environment": "models/environment",
            "music": "audio/music",
            "sfx": "audio/sfx",
            "voice": "audio/voice"
        }
        
        return str(self.user_assets_dir / type_map.get(asset_type, "misc"))
        
    def save_model(self, path: str):
        """학습된 모델 저장"""
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_history': self.learning_history,
            'success_rate': self.success_count / max(1, self.total_attempts)
        }, path)
        
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_history = checkpoint.get('learning_history', [])
        
        logger.info(f"모델 로드 완료. 성공률: {checkpoint.get('success_rate', 0):.2%}")

# 사용 예시
if __name__ == "__main__":
    # 프로젝트 경로 설정
    project_path = "D:/MyGodotProject"
    
    # 자동화 시스템 초기화
    automation = GodotAutomationSystem(project_path)
    
    # 자원 요청 예시
    sprite_folder = automation.request_user_asset(
        "sprite", 
        "주인공 캐릭터 스프라이트 (idle, walk, jump 애니메이션 포함)"
    )
    print(f"스프라이트를 다음 폴더에 넣어주세요: {sprite_folder}")
    
    # 자동화 작업 실행
    automation.automated_workflow("Create a player sprite with collision")
    
    # 학습 루프 (실제로는 더 복잡한 로직 필요)
    for episode in range(100):
        state = automation.capture_state()
        done = False
        
        while not done:
            # 액션 선택
            with torch.no_grad():
                action_probs, _ = automation.agent(state.unsqueeze(0).to(automation.device))
            action = action_probs.argmax().item()
            
            # 액션 실행
            reward = automation.execute_action(action)
            next_state = automation.capture_state()
            
            # 메모리에 저장
            automation.memory.push(state, action, reward, next_state, done)
            
            # 학습
            if len(automation.memory) > 32:
                loss = automation.train_step()
                
            state = next_state
            
            # 종료 조건 (예시)
            if reward > 0.9 or automation.total_attempts > 1000:
                done = True
                
    # 모델 저장
    automation.save_model("godot_automation_model.pth")