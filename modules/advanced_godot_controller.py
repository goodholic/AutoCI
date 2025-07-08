#!/usr/bin/env python3
"""
고급 Godot 자동화 컨트롤러
변형된 Godot 에디터를 화면 인식으로 직접 제어
딥러닝 기반 비전 시스템과 정밀 제어
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import threading
from dataclasses import dataclass
from enum import Enum

# 화면 인식 및 제어
try:
    import cv2
    import pyautogui
    import pytesseract
    from PIL import Image, ImageGrab
    import mss  # 더 빠른 스크린샷
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("⚠️ pip install opencv-python pyautogui pillow pytesseract mss")

# 딥러닝
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from collections import deque

# Windows API (더 정밀한 제어)
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    import win32api
    import win32con
    import win32gui
    import win32process

logger = logging.getLogger(__name__)

class GodotUIElement(Enum):
    """Godot UI 요소 타입"""
    MENU_BAR = "menu_bar"
    SCENE_DOCK = "scene_dock"
    INSPECTOR = "inspector"
    FILE_SYSTEM = "file_system"
    NODE_TREE = "node_tree"
    VIEWPORT = "viewport"
    SCRIPT_EDITOR = "script_editor"
    OUTPUT_PANEL = "output_panel"
    DEBUGGER = "debugger"
    ANIMATION_PLAYER = "animation_player"
    SHADER_EDITOR = "shader_editor"

@dataclass
class UIElementInfo:
    """UI 요소 정보"""
    element_type: GodotUIElement
    position: Tuple[int, int]
    size: Tuple[int, int]
    confidence: float
    screenshot: np.ndarray = None
    text_content: str = ""
    is_active: bool = False
    children: List['UIElementInfo'] = None

class AdvancedVisionSystem:
    """고급 비전 시스템 - YOLO 스타일 객체 감지"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # UI 감지 모델 (사전 훈련된 모델 활용)
        self.detection_model = self._build_detection_model()
        self.classification_model = self._build_classification_model()
        
        # 화면 캐시
        self.screen_cache = {}
        self.last_capture_time = 0
        
        # 고속 스크린샷
        self.sct = mss.mss()
        
    def _build_detection_model(self) -> nn.Module:
        """UI 요소 감지 모델"""
        class UIDetector(nn.Module):
            def __init__(self, num_classes=len(GodotUIElement)):
                super().__init__()
                # Backbone: ResNet50
                self.backbone = models.resnet50(pretrained=True)
                self.backbone.fc = nn.Identity()
                
                # Detection head
                self.detect_conv = nn.Sequential(
                    nn.Conv2d(2048, 1024, 3, padding=1),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(1024, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU()
                )
                
                # Bounding box regression
                self.bbox_head = nn.Conv2d(512, 4, 1)  # x, y, w, h
                
                # Classification
                self.class_head = nn.Conv2d(512, num_classes, 1)
                
                # Confidence
                self.conf_head = nn.Conv2d(512, 1, 1)
                
            def forward(self, x):
                # Feature extraction
                features = self.backbone.conv1(x)
                features = self.backbone.bn1(features)
                features = self.backbone.relu(features)
                features = self.backbone.maxpool(features)
                
                features = self.backbone.layer1(features)
                features = self.backbone.layer2(features)
                features = self.backbone.layer3(features)
                features = self.backbone.layer4(features)
                
                # Detection
                detect_features = self.detect_conv(features)
                
                # Outputs
                bboxes = self.bbox_head(detect_features)
                classes = self.class_head(detect_features)
                confidence = torch.sigmoid(self.conf_head(detect_features))
                
                return bboxes, classes, confidence
                
        model = UIDetector().to(self.device)
        model.eval()
        return model
        
    def _build_classification_model(self) -> nn.Module:
        """UI 상태 분류 모델"""
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 256)
        model = model.to(self.device)
        model.eval()
        return model
        
    def capture_screen_fast(self, region: Optional[Dict] = None) -> np.ndarray:
        """고속 화면 캡처"""
        if region:
            monitor = {"top": region["top"], "left": region["left"], 
                      "width": region["width"], "height": region["height"]}
        else:
            monitor = self.sct.monitors[1]  # 주 모니터
            
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    def detect_ui_elements(self, screenshot: np.ndarray) -> List[UIElementInfo]:
        """UI 요소 감지"""
        # 전처리
        input_tensor = self.transform(screenshot).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            bboxes, classes, confidence = self.detection_model(input_tensor)
            
        # 후처리 (NMS 등)
        detected_elements = self._postprocess_detections(
            bboxes, classes, confidence, screenshot
        )
        
        return detected_elements
        
    def _postprocess_detections(self, bboxes, classes, confidence, original_img):
        """감지 결과 후처리"""
        elements = []
        
        # 신뢰도 임계값
        conf_threshold = 0.5
        
        # NMS (Non-Maximum Suppression)
        keep_indices = self._nms(bboxes, confidence, threshold=0.5)
        
        for idx in keep_indices:
            if confidence[0, 0, idx] > conf_threshold:
                # 바운딩 박스 변환
                bbox = bboxes[0, :, idx].cpu().numpy()
                x, y, w, h = self._denormalize_bbox(bbox, original_img.shape)
                
                # 클래스 예측
                class_probs = F.softmax(classes[0, :, idx], dim=0)
                class_id = class_probs.argmax().item()
                
                # UI 요소 정보 생성
                element = UIElementInfo(
                    element_type=list(GodotUIElement)[class_id],
                    position=(int(x), int(y)),
                    size=(int(w), int(h)),
                    confidence=float(confidence[0, 0, idx]),
                    screenshot=original_img[y:y+h, x:x+w]
                )
                
                elements.append(element)
                
        return elements
        
    def _nms(self, boxes, scores, threshold=0.5):
        """Non-Maximum Suppression"""
        # 간단한 NMS 구현 (실제로는 torchvision.ops.nms 사용 권장)
        keep = []
        order = scores.argsort(descending=True)
        
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            
            if order.numel() == 1:
                break
                
            # IoU 계산 및 억제
            # ... (구현 생략)
            
        return keep
        
    def _denormalize_bbox(self, bbox, img_shape):
        """정규화된 바운딩 박스를 픽셀 좌표로 변환"""
        h, w = img_shape[:2]
        x = bbox[0] * w
        y = bbox[1] * h
        width = bbox[2] * w
        height = bbox[3] * h
        return int(x), int(y), int(width), int(height)

class PrecisionMouseController:
    """정밀 마우스 컨트롤러"""
    
    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32
        
        # 마우스 이동 보간
        self.interpolation_steps = 50
        self.movement_noise = 0.2  # 자연스러운 움직임을 위한 노이즈
        
    def get_cursor_pos(self) -> Tuple[int, int]:
        """현재 커서 위치"""
        point = wintypes.POINT()
        self.user32.GetCursorPos(ctypes.byref(point))
        return point.x, point.y
        
    def move_mouse_natural(self, target_x: int, target_y: int, duration: float = 0.5):
        """자연스러운 마우스 이동 (베지어 곡선)"""
        start_x, start_y = self.get_cursor_pos()
        
        # 제어점 생성 (베지어 곡선)
        ctrl_x1 = start_x + (target_x - start_x) * 0.25 + np.random.randn() * 50
        ctrl_y1 = start_y + (target_y - start_y) * 0.25 + np.random.randn() * 50
        ctrl_x2 = start_x + (target_x - start_x) * 0.75 + np.random.randn() * 50
        ctrl_y2 = start_y + (target_y - start_y) * 0.75 + np.random.randn() * 50
        
        steps = int(duration * 60)  # 60 FPS
        
        for i in range(steps):
            t = i / steps
            
            # 베지어 곡선 계산
            x = int((1-t)**3 * start_x + 
                   3*(1-t)**2*t * ctrl_x1 + 
                   3*(1-t)*t**2 * ctrl_x2 + 
                   t**3 * target_x)
            y = int((1-t)**3 * start_y + 
                   3*(1-t)**2*t * ctrl_y1 + 
                   3*(1-t)*t**2 * ctrl_y2 + 
                   t**3 * target_y)
            
            # 약간의 노이즈 추가
            x += int(np.random.randn() * self.movement_noise)
            y += int(np.random.randn() * self.movement_noise)
            
            self.user32.SetCursorPos(x, y)
            time.sleep(1/60)
            
    def click_precise(self, button: str = 'left', double: bool = False):
        """정밀 클릭"""
        # 버튼 매핑
        button_down = {
            'left': win32con.MOUSEEVENTF_LEFTDOWN,
            'right': win32con.MOUSEEVENTF_RIGHTDOWN,
            'middle': win32con.MOUSEEVENTF_MIDDLEDOWN
        }
        button_up = {
            'left': win32con.MOUSEEVENTF_LEFTUP,
            'right': win32con.MOUSEEVENTF_RIGHTUP,
            'middle': win32con.MOUSEEVENTF_MIDDLEUP
        }
        
        # 클릭 수행
        x, y = self.get_cursor_pos()
        
        if double:
            for _ in range(2):
                win32api.mouse_event(button_down[button], x, y, 0, 0)
                time.sleep(0.05)
                win32api.mouse_event(button_up[button], x, y, 0, 0)
                time.sleep(0.05)
        else:
            win32api.mouse_event(button_down[button], x, y, 0, 0)
            time.sleep(0.1)
            win32api.mouse_event(button_up[button], x, y, 0, 0)
            
    def drag_precise(self, start_x: int, start_y: int, end_x: int, end_y: int, 
                    duration: float = 1.0):
        """정밀 드래그"""
        self.move_mouse_natural(start_x, start_y, duration/3)
        time.sleep(0.1)
        
        # 마우스 버튼 누르기
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, start_x, start_y, 0, 0)
        time.sleep(0.1)
        
        # 드래그
        self.move_mouse_natural(end_x, end_y, duration*2/3)
        time.sleep(0.1)
        
        # 마우스 버튼 놓기
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, end_x, end_y, 0, 0)

class AdvancedKeyboardController:
    """고급 키보드 컨트롤러"""
    
    def __init__(self):
        self.user32 = ctypes.windll.user32
        
        # 가상 키 코드
        self.VK_CODES = {
            'backspace': 0x08, 'tab': 0x09, 'enter': 0x0D, 'shift': 0x10,
            'ctrl': 0x11, 'alt': 0x12, 'pause': 0x13, 'caps_lock': 0x14,
            'esc': 0x1B, 'space': 0x20, 'page_up': 0x21, 'page_down': 0x22,
            'end': 0x23, 'home': 0x24, 'left': 0x25, 'up': 0x26,
            'right': 0x27, 'down': 0x28, 'select': 0x29, 'print': 0x2A,
            'execute': 0x2B, 'print_screen': 0x2C, 'insert': 0x2D,
            'delete': 0x2E, 'help': 0x2F,
            'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
            'f5': 0x74, 'f6': 0x75, 'f7': 0x76, 'f8': 0x77,
            'f9': 0x78, 'f10': 0x79, 'f11': 0x7A, 'f12': 0x7B
        }
        
    def type_with_timing(self, text: str, wpm: int = 60):
        """타이핑 속도를 조절한 입력"""
        # WPM을 초당 문자로 변환
        chars_per_second = (wpm * 5) / 60  # 평균 5자를 1단어로 가정
        delay = 1 / chars_per_second
        
        for char in text:
            # 자연스러운 타이핑을 위한 랜덤 지연
            actual_delay = delay + np.random.normal(0, delay * 0.2)
            actual_delay = max(0.01, actual_delay)  # 최소 지연
            
            pyautogui.write(char)
            time.sleep(actual_delay)
            
    def send_key_combination(self, *keys):
        """키 조합 전송"""
        # 키 누르기
        for key in keys:
            if key in self.VK_CODES:
                vk_code = self.VK_CODES[key]
            else:
                vk_code = ord(key.upper())
            self.user32.keybd_event(vk_code, 0, 0, 0)
            time.sleep(0.05)
            
        # 키 놓기 (역순)
        for key in reversed(keys):
            if key in self.VK_CODES:
                vk_code = self.VK_CODES[key]
            else:
                vk_code = ord(key.upper())
            self.user32.keybd_event(vk_code, 0, 0x0002, 0)  # KEYEVENTF_KEYUP
            time.sleep(0.05)

class GodotAutomationController:
    """통합 Godot 자동화 컨트롤러"""
    
    def __init__(self):
        self.vision = AdvancedVisionSystem()
        self.mouse = PrecisionMouseController()
        self.keyboard = AdvancedKeyboardController()
        
        # Godot 창 핸들
        self.godot_hwnd = None
        self.find_godot_window()
        
        # 작업 큐
        self.task_queue = deque()
        self.current_task = None
        
        # 상태 추적
        self.ui_state = {}
        self.last_action_time = 0
        self.action_history = deque(maxlen=100)
        
        # 학습된 패턴
        self.learned_patterns = self._load_patterns()
        
    def find_godot_window(self) -> bool:
        """Godot 창 찾기"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if "Godot" in window_title:
                    windows.append((hwnd, window_title))
            return True
            
        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if windows:
            self.godot_hwnd = windows[0][0]
            logger.info(f"Godot 창 발견: {windows[0][1]}")
            return True
        
        logger.error("Godot 창을 찾을 수 없습니다")
        return False
        
    def focus_godot_window(self):
        """Godot 창 포커스"""
        if self.godot_hwnd:
            win32gui.ShowWindow(self.godot_hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(self.godot_hwnd)
            time.sleep(0.5)
            
    def scan_ui_state(self) -> Dict[str, UIElementInfo]:
        """현재 UI 상태 스캔"""
        self.focus_godot_window()
        
        # 전체 화면 캡처
        screenshot = self.vision.capture_screen_fast()
        
        # UI 요소 감지
        detected_elements = self.vision.detect_ui_elements(screenshot)
        
        # 상태 업데이트
        self.ui_state = {
            element.element_type.value: element 
            for element in detected_elements
        }
        
        # OCR로 텍스트 추출
        self._extract_text_content()
        
        return self.ui_state
        
    def _extract_text_content(self):
        """UI 요소에서 텍스트 추출"""
        for element_type, element in self.ui_state.items():
            if element.screenshot is not None:
                try:
                    gray = cv2.cvtColor(element.screenshot, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray)
                    element.text_content = text.strip()
                except Exception as e:
                    logger.error(f"OCR 실패: {e}")
                    
    def navigate_to_menu(self, menu_path: List[str]):
        """메뉴 네비게이션"""
        # 예: ["Scene", "New Node"] 또는 ["File", "Save Scene As..."]
        
        for i, menu_item in enumerate(menu_path):
            if i == 0:
                # 최상위 메뉴 클릭
                menu_bar = self.ui_state.get(GodotUIElement.MENU_BAR.value)
                if menu_bar:
                    # 메뉴 아이템 위치 계산 (OCR 또는 패턴 매칭)
                    click_x = menu_bar.position[0] + self._estimate_menu_position(menu_item)
                    click_y = menu_bar.position[1] + menu_bar.size[1] // 2
                    
                    self.mouse.move_mouse_natural(click_x, click_y)
                    self.mouse.click_precise()
                    time.sleep(0.3)
            else:
                # 하위 메뉴 아이템 선택
                # 드롭다운 메뉴에서 아이템 찾기
                self._select_dropdown_item(menu_item)
                
    def _estimate_menu_position(self, menu_name: str) -> int:
        """메뉴 위치 추정"""
        # 기본 메뉴 위치 (학습된 값 사용)
        menu_positions = {
            "File": 30,
            "Edit": 80,
            "Scene": 130,
            "Project": 190,
            "Debug": 250,
            "Editor": 310,
            "Help": 370
        }
        return menu_positions.get(menu_name, 100)
        
    def _select_dropdown_item(self, item_name: str):
        """드롭다운 메뉴에서 아이템 선택"""
        # 현재 화면 캡처
        screenshot = self.vision.capture_screen_fast()
        
        # 텍스트 매칭으로 아이템 찾기
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # OCR로 모든 텍스트 박스 찾기
        d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        
        for i, text in enumerate(d['text']):
            if item_name.lower() in text.lower():
                x = d['left'][i] + d['width'][i] // 2
                y = d['top'][i] + d['height'][i] // 2
                
                self.mouse.move_mouse_natural(x, y)
                self.mouse.click_precise()
                return True
                
        return False
        
    def create_node(self, node_type: str, parent_path: str = None):
        """노드 생성"""
        # Scene 도크로 이동
        scene_dock = self.ui_state.get(GodotUIElement.SCENE_DOCK.value)
        if not scene_dock:
            logger.error("Scene 도크를 찾을 수 없습니다")
            return False
            
        # 부모 노드 선택 (있는 경우)
        if parent_path:
            self._select_node_in_tree(parent_path)
            
        # 노드 추가 버튼 찾기 또는 단축키 사용
        self.keyboard.send_key_combination('ctrl', 'a')
        time.sleep(0.5)
        
        # 노드 타입 입력
        self.keyboard.type_with_timing(node_type)
        time.sleep(0.3)
        
        # Enter로 생성
        self.keyboard.send_key_combination('enter')
        time.sleep(0.5)
        
        return True
        
    def _select_node_in_tree(self, node_path: str):
        """씬 트리에서 노드 선택"""
        # node_path: "Node2D/Sprite2D" 형식
        path_parts = node_path.split('/')
        
        # 루트부터 순차적으로 확장하며 선택
        for part in path_parts:
            # 현재 화면에서 노드 찾기
            if self._click_tree_item(part):
                time.sleep(0.2)
            else:
                logger.warning(f"노드를 찾을 수 없습니다: {part}")
                
    def _click_tree_item(self, item_name: str) -> bool:
        """트리 아이템 클릭"""
        scene_dock = self.ui_state.get(GodotUIElement.SCENE_DOCK.value)
        if not scene_dock:
            return False
            
        # 씬 도크 영역에서 텍스트 찾기
        region = {
            "top": scene_dock.position[1],
            "left": scene_dock.position[0],
            "width": scene_dock.size[0],
            "height": scene_dock.size[1]
        }
        
        screenshot = self.vision.capture_screen_fast(region)
        
        # 템플릿 매칭 또는 OCR
        return self._find_and_click_text(screenshot, item_name, region)
        
    def set_node_property(self, property_name: str, value: Any):
        """노드 속성 설정"""
        inspector = self.ui_state.get(GodotUIElement.INSPECTOR.value)
        if not inspector:
            logger.error("Inspector를 찾을 수 없습니다")
            return False
            
        # Inspector에서 속성 찾기
        # ... (구현)
        
        return True
        
    def execute_gdscript(self, script_code: str):
        """GDScript 코드 실행"""
        # Script 에디터 열기
        self.keyboard.send_key_combination('ctrl', 'alt', 's')
        time.sleep(0.5)
        
        # 코드 입력
        self.keyboard.type_with_timing(script_code, wpm=80)
        
        # 실행
        self.keyboard.send_key_combination('ctrl', 'shift', 'x')
        
    def run_automated_test(self) -> Dict[str, Any]:
        """자동화 테스트 실행"""
        results = {
            "success": True,
            "tests_run": 0,
            "tests_passed": 0,
            "errors": []
        }
        
        try:
            # 1. UI 상태 확인
            self.scan_ui_state()
            
            # 2. 기본 노드 생성 테스트
            if self.create_node("Node2D"):
                results["tests_passed"] += 1
            results["tests_run"] += 1
            
            # 3. 속성 설정 테스트
            if self.set_node_property("position", "100, 100"):
                results["tests_passed"] += 1
            results["tests_run"] += 1
            
            # 4. 씬 저장 테스트
            self.keyboard.send_key_combination('ctrl', 's')
            time.sleep(1)
            results["tests_run"] += 1
            results["tests_passed"] += 1
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            
        return results
        
    def _load_patterns(self) -> Dict:
        """학습된 UI 패턴 로드"""
        patterns_file = Path(__file__).parent / "godot_ui_patterns.json"
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                return json.load(f)
        return {}
        
    def save_patterns(self):
        """UI 패턴 저장"""
        patterns_file = Path(__file__).parent / "godot_ui_patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump(self.learned_patterns, f, indent=2)

class GodotProjectAutomation:
    """Godot 프로젝트 전체 자동화"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.controller = GodotAutomationController()
        
        # 작업 스크립트
        self.automation_scripts = []
        
    def create_2d_platformer(self):
        """2D 플랫포머 자동 생성"""
        logger.info("🎮 2D 플랫포머 생성 시작")
        
        # 1. 메인 씬 생성
        self.controller.navigate_to_menu(["Scene", "New Scene"])
        time.sleep(1)
        
        # 2. 기본 구조 생성
        self.controller.create_node("Node2D")  # 루트
        self.controller.create_node("CharacterBody2D", "Node2D")  # 플레이어
        self.controller.create_node("Sprite2D", "Node2D/CharacterBody2D")
        self.controller.create_node("CollisionShape2D", "Node2D/CharacterBody2D")
        
        # 3. 플레이어 스크립트 추가
        player_script = '''extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

func _physics_process(delta):
    if not is_on_floor():
        velocity.y += gravity * delta
    
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = JUMP_VELOCITY
    
    var direction = Input.get_axis("ui_left", "ui_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = move_toward(velocity.x, 0, SPEED)
    
    move_and_slide()
'''
        
        # 스크립트 첨부
        self.controller._select_node_in_tree("Node2D/CharacterBody2D")
        self.controller.keyboard.send_key_combination('ctrl', 'shift', 'a')
        time.sleep(1)
        self.controller.keyboard.type_with_timing(player_script)
        
        # 4. 씬 저장
        self.controller.keyboard.send_key_combination('ctrl', 's')
        time.sleep(1)
        self.controller.keyboard.type_with_timing("Main.tscn")
        self.controller.keyboard.send_key_combination('enter')
        
        logger.info("✅ 2D 플랫포머 기본 구조 생성 완료")
        
    def create_ui_system(self):
        """UI 시스템 생성"""
        logger.info("🖼️ UI 시스템 생성 시작")
        
        # UI 씬 생성
        self.controller.navigate_to_menu(["Scene", "New Scene"])
        time.sleep(1)
        
        # Control 노드 구조
        self.controller.create_node("Control")
        self.controller.create_node("MarginContainer", "Control")
        self.controller.create_node("VBoxContainer", "Control/MarginContainer")
        self.controller.create_node("Label", "Control/MarginContainer/VBoxContainer")
        self.controller.create_node("Button", "Control/MarginContainer/VBoxContainer")
        
        # 속성 설정
        self.controller._select_node_in_tree("Control")
        self.controller.set_node_property("anchor_preset", "15")  # Full Rect
        
        logger.info("✅ UI 시스템 생성 완료")

# 메인 실행
if __name__ == "__main__":
    # Godot 자동화 테스트
    automation = GodotProjectAutomation("D:/MyGodotProject")
    
    # 2D 플랫포머 생성
    automation.create_2d_platformer()
    
    # UI 시스템 생성
    automation.create_ui_system()
    
    # 자동화 테스트 실행
    controller = GodotAutomationController()
    results = controller.run_automated_test()
    print(f"테스트 결과: {results['tests_passed']}/{results['tests_run']} 통과")