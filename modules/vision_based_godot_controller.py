#!/usr/bin/env python3
"""
AutoCI Create - 비전 기반 Godot 컨트롤러
화면을 실시간으로 분석하고 정밀하게 조작하는 시스템
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import cv2
import pytesseract
from PIL import Image, ImageGrab, ImageDraw, ImageFont
import pyautogui
import mss
from collections import defaultdict
import threading
import queue

# Windows 전용 기능
if sys.platform == "win32":
    import win32gui
    import win32api
    import win32con
    import ctypes
    from ctypes import wintypes

# 딥러닝
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models

logger = logging.getLogger(__name__)

class GodotElementType(Enum):
    """Godot UI 요소 타입 (구체적)"""
    # 메인 UI
    MENU_BAR = auto()
    TOOLBAR = auto()
    SCENE_DOCK = auto()
    INSPECTOR_DOCK = auto()
    FILE_SYSTEM_DOCK = auto()
    OUTPUT_PANEL = auto()
    
    # 에디터 뷰
    VIEWPORT_2D = auto()
    VIEWPORT_3D = auto()
    SCRIPT_EDITOR = auto()
    SHADER_EDITOR = auto()
    ANIMATION_EDITOR = auto()
    
    # 다이얼로그
    NODE_CREATE_DIALOG = auto()
    FILE_DIALOG = auto()
    PROJECT_SETTINGS = auto()
    EXPORT_DIALOG = auto()
    
    # 특수 요소
    SCENE_TREE_ITEM = auto()
    PROPERTY_FIELD = auto()
    BUTTON = auto()
    TAB = auto()
    DROPDOWN = auto()
    CHECKBOX = auto()
    SLIDER = auto()
    TEXT_INPUT = auto()

@dataclass
class ScreenRegion:
    """화면 영역 정의"""
    x: int
    y: int
    width: int
    height: int
    name: str
    element_type: Optional[GodotElementType] = None
    
    def to_bbox(self) -> Tuple[int, int, int, int]:
        """바운딩 박스로 변환"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
        
    def contains_point(self, x: int, y: int) -> bool:
        """점이 영역 내에 있는지 확인"""
        return (self.x <= x < self.x + self.width and 
                self.y <= y < self.y + self.height)
                
    def center(self) -> Tuple[int, int]:
        """중심점 반환"""
        return (self.x + self.width // 2, self.y + self.height // 2)

@dataclass
class DetectedElement:
    """감지된 UI 요소"""
    element_type: GodotElementType
    region: ScreenRegion
    confidence: float
    text: str = ""
    is_active: bool = False
    is_visible: bool = True
    properties: Dict[str, Any] = field(default_factory=dict)
    screenshot: Optional[np.ndarray] = None
    
class AdvancedScreenAnalyzer:
    """고급 화면 분석기"""
    
    def __init__(self):
        self.sct = mss.mss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Godot UI 템플릿 (실제 UI 캡처 필요)
        self.ui_templates = self._load_ui_templates()
        
        # 색상 범위 (Godot 테마)
        self.color_ranges = {
            "dark_theme": {
                "background": (30, 30, 35),
                "panel": (40, 40, 45),
                "button": (55, 55, 60),
                "button_hover": (65, 65, 70),
                "text": (200, 200, 200),
                "selection": (70, 120, 180)
            }
        }
        
        # 캐시
        self.cache = {
            "last_screenshot": None,
            "last_analysis": None,
            "last_capture_time": 0
        }
        
        # UI 요소 감지 모델
        self._init_detection_models()
        
    def _init_detection_models(self):
        """감지 모델 초기화"""
        # YOLO 스타일 객체 감지기
        self.element_detector = self._create_element_detector()
        
        # 텍스트 감지기
        self.text_detector = self._create_text_detector()
        
    def _create_element_detector(self) -> nn.Module:
        """UI 요소 감지 모델"""
        # 사전 훈련된 모델 활용 (실제로는 Godot UI로 fine-tuning 필요)
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        return model.to(self.device)
        
    def _create_text_detector(self) -> nn.Module:
        """텍스트 영역 감지 모델"""
        # EAST 텍스트 감지기 스타일
        class TextDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = models.vgg16(pretrained=True).features
                self.text_conv = nn.Conv2d(512, 1, kernel_size=1)
                
            def forward(self, x):
                features = self.features(x)
                text_map = torch.sigmoid(self.text_conv(features))
                return text_map
                
        return TextDetector().to(self.device)
        
    def capture_screen(self, region: Optional[ScreenRegion] = None) -> np.ndarray:
        """화면 캡처 (고속)"""
        # 캐시 확인
        current_time = time.time()
        if (self.cache["last_screenshot"] is not None and 
            current_time - self.cache["last_capture_time"] < 0.1):
            return self.cache["last_screenshot"]
            
        # 캡처
        if region:
            monitor = {
                "left": region.x,
                "top": region.y,
                "width": region.width,
                "height": region.height
            }
        else:
            monitor = self.sct.monitors[1]
            
        screenshot = np.array(self.sct.grab(monitor))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        # 캐시 업데이트
        self.cache["last_screenshot"] = screenshot
        self.cache["last_capture_time"] = current_time
        
        return screenshot
        
    def analyze_screen(self, screenshot: Optional[np.ndarray] = None) -> Dict[str, List[DetectedElement]]:
        """전체 화면 분석"""
        if screenshot is None:
            screenshot = self.capture_screen()
            
        results = {
            "panels": [],
            "buttons": [],
            "text_fields": [],
            "tree_items": [],
            "other": []
        }
        
        # 1. 패널 감지 (색상 기반)
        panels = self._detect_panels(screenshot)
        results["panels"] = panels
        
        # 2. 버튼 감지
        buttons = self._detect_buttons(screenshot)
        results["buttons"] = buttons
        
        # 3. 텍스트 필드 감지
        text_fields = self._detect_text_fields(screenshot)
        results["text_fields"] = text_fields
        
        # 4. 트리 아이템 감지 (씬 도크)
        tree_items = self._detect_tree_items(screenshot)
        results["tree_items"] = tree_items
        
        # 5. OCR로 텍스트 추출
        self._extract_text_for_elements(results, screenshot)
        
        # 캐시 업데이트
        self.cache["last_analysis"] = results
        
        return results
        
    def _detect_panels(self, screenshot: np.ndarray) -> List[DetectedElement]:
        """패널 감지 (Godot의 어두운 테마 기반)"""
        panels = []
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # 너무 작은 영역 제외
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # 패널인지 확인 (색상 체크)
            roi = screenshot[y:y+h, x:x+w]
            avg_color = np.mean(roi, axis=(0, 1))
            
            # Godot 패널 색상과 비교
            panel_color = self.color_ranges["dark_theme"]["panel"]
            if np.allclose(avg_color, panel_color, atol=20):
                region = ScreenRegion(x, y, w, h, f"panel_{len(panels)}")
                element = DetectedElement(
                    element_type=GodotElementType.SCENE_DOCK,
                    region=region,
                    confidence=0.8,
                    screenshot=roi
                )
                panels.append(element)
                
        return panels
        
    def _detect_buttons(self, screenshot: np.ndarray) -> List[DetectedElement]:
        """버튼 감지"""
        buttons = []
        
        # 템플릿 매칭으로 버튼 찾기
        button_templates = [
            "add_node_button.png",
            "play_button.png",
            "save_button.png"
        ]
        
        for template_name in button_templates:
            template_path = Path(__file__).parent / "templates" / template_name
            if not template_path.exists():
                continue
                
            template = cv2.imread(str(template_path))
            result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
            
            # 임계값 이상인 위치 찾기
            locations = np.where(result >= 0.8)
            
            for pt in zip(*locations[::-1]):
                x, y = pt
                h, w = template.shape[:2]
                
                region = ScreenRegion(x, y, w, h, template_name.split('.')[0])
                element = DetectedElement(
                    element_type=GodotElementType.BUTTON,
                    region=region,
                    confidence=float(result[y, x]),
                    properties={"template": template_name}
                )
                buttons.append(element)
                
        return buttons
        
    def _detect_text_fields(self, screenshot: np.ndarray) -> List[DetectedElement]:
        """텍스트 입력 필드 감지"""
        text_fields = []
        
        # 흰색/밝은 회색 영역 찾기 (텍스트 필드는 보통 배경보다 밝음)
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((3, 20), np.uint8)  # 가로로 긴 커널
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 텍스트 필드 크기 조건
            if 50 < w < 500 and 15 < h < 40:
                region = ScreenRegion(x, y, w, h, f"text_field_{len(text_fields)}")
                element = DetectedElement(
                    element_type=GodotElementType.TEXT_INPUT,
                    region=region,
                    confidence=0.7
                )
                text_fields.append(element)
                
        return text_fields
        
    def _detect_tree_items(self, screenshot: np.ndarray) -> List[DetectedElement]:
        """씬 트리 아이템 감지"""
        tree_items = []
        
        # 씬 도크 영역 찾기 (왼쪽 패널)
        scene_dock_region = self._find_scene_dock(screenshot)
        if not scene_dock_region:
            return tree_items
            
        # 씬 도크 영역만 분석
        dock_screenshot = screenshot[
            scene_dock_region.y:scene_dock_region.y + scene_dock_region.height,
            scene_dock_region.x:scene_dock_region.x + scene_dock_region.width
        ]
        
        # 들여쓰기 레벨 감지 (트리 구조)
        gray = cv2.cvtColor(dock_screenshot, cv2.COLOR_BGR2GRAY)
        
        # 수평선 감지 (각 아이템의 경계)
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            # Y 좌표로 정렬
            lines = sorted(lines, key=lambda x: x[0][1])
            
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 5:  # 수평선만
                    # 들여쓰기 레벨 계산
                    indent_level = self._calculate_indent_level(dock_screenshot, y1)
                    
                    region = ScreenRegion(
                        scene_dock_region.x + x1,
                        scene_dock_region.y + y1,
                        x2 - x1,
                        20,  # 예상 높이
                        f"tree_item_{i}"
                    )
                    
                    element = DetectedElement(
                        element_type=GodotElementType.SCENE_TREE_ITEM,
                        region=region,
                        confidence=0.6,
                        properties={"indent_level": indent_level}
                    )
                    tree_items.append(element)
                    
        return tree_items
        
    def _find_scene_dock(self, screenshot: np.ndarray) -> Optional[ScreenRegion]:
        """씬 도크 위치 찾기"""
        # 일반적으로 왼쪽에 위치
        height, width = screenshot.shape[:2]
        
        # 왼쪽 1/4 영역 확인
        left_region = screenshot[:, :width//4]
        
        # "Scene" 텍스트 찾기
        gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        
        if "Scene" in text:
            # OCR 데이터로 정확한 위치 찾기
            d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            for i, word in enumerate(d['text']):
                if "Scene" in word:
                    x = d['left'][i]
                    y = d['top'][i]
                    return ScreenRegion(x, y, width//4, height-y, "scene_dock")
                    
        # 기본값
        return ScreenRegion(0, 50, width//4, height-100, "scene_dock")
        
    def _calculate_indent_level(self, image: np.ndarray, y: int) -> int:
        """들여쓰기 레벨 계산"""
        # 해당 라인의 첫 번째 픽셀 위치 찾기
        row = image[y:y+1, :]
        gray_row = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        
        # 첫 번째 밝은 픽셀 찾기
        for x in range(gray_row.shape[1]):
            if gray_row[0, x] > 100:
                return x // 20  # 20픽셀당 1레벨
                
        return 0
        
    def _extract_text_for_elements(self, results: Dict, screenshot: np.ndarray):
        """각 요소에서 텍스트 추출"""
        for category, elements in results.items():
            for element in elements:
                if element.screenshot is not None:
                    roi = element.screenshot
                else:
                    # 스크린샷에서 영역 추출
                    region = element.region
                    roi = screenshot[
                        region.y:region.y + region.height,
                        region.x:region.x + region.width
                    ]
                    
                # OCR
                try:
                    text = pytesseract.image_to_string(roi).strip()
                    element.text = text
                except Exception as e:
                    logger.error(f"OCR 오류: {e}")
                    
    def find_element_by_text(self, text: str, element_type: Optional[GodotElementType] = None) -> Optional[DetectedElement]:
        """텍스트로 요소 찾기"""
        # 최근 분석 결과 사용
        if not self.cache["last_analysis"]:
            self.analyze_screen()
            
        results = self.cache["last_analysis"]
        
        for category, elements in results.items():
            for element in elements:
                if text.lower() in element.text.lower():
                    if element_type is None or element.element_type == element_type:
                        return element
                        
        return None
        
    def find_element_by_image(self, template_path: str, threshold: float = 0.8) -> Optional[DetectedElement]:
        """이미지 템플릿으로 요소 찾기"""
        screenshot = self.capture_screen()
        template = cv2.imread(template_path)
        
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            h, w = template.shape[:2]
            region = ScreenRegion(max_loc[0], max_loc[1], w, h, "template_match")
            
            return DetectedElement(
                element_type=GodotElementType.BUTTON,
                region=region,
                confidence=max_val
            )
            
        return None
        
    def _load_ui_templates(self) -> Dict[str, np.ndarray]:
        """UI 템플릿 로드"""
        templates = {}
        template_dir = Path(__file__).parent / "templates"
        
        if template_dir.exists():
            for template_file in template_dir.glob("*.png"):
                template = cv2.imread(str(template_file))
                templates[template_file.stem] = template
                
        return templates

class PrecisionInputController:
    """정밀 입력 컨트롤러"""
    
    def __init__(self):
        # 안전 설정
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
        
        # 마우스 설정
        self.mouse_speed = 0.5  # 기본 이동 속도
        self.click_delay = 0.1  # 클릭 간 지연
        
        # 키보드 설정
        self.typing_speed = 60  # WPM
        self.key_delay = 0.05   # 키 입력 간 지연
        
        # Windows API
        if sys.platform == "win32":
            self.user32 = ctypes.windll.user32
            self.kernel32 = ctypes.windll.kernel32
            
    def move_to_element(self, element: DetectedElement, offset: Tuple[int, int] = (0, 0)):
        """요소로 마우스 이동"""
        center = element.region.center()
        target_x = center[0] + offset[0]
        target_y = center[1] + offset[1]
        
        self.move_mouse_smooth(target_x, target_y)
        
    def move_mouse_smooth(self, x: int, y: int, duration: Optional[float] = None):
        """부드러운 마우스 이동 (베지어 곡선)"""
        if duration is None:
            duration = self.mouse_speed
            
        start_x, start_y = pyautogui.position()
        
        # 제어점 계산 (3차 베지어 곡선)
        cp1_x = start_x + (x - start_x) * 0.25 + np.random.randint(-20, 20)
        cp1_y = start_y + (y - start_y) * 0.25 + np.random.randint(-20, 20)
        cp2_x = start_x + (x - start_x) * 0.75 + np.random.randint(-20, 20)
        cp2_y = start_y + (y - start_y) * 0.75 + np.random.randint(-20, 20)
        
        # 이동 경로 생성
        steps = int(duration * 60)  # 60 FPS
        for i in range(steps + 1):
            t = i / steps
            
            # 베지어 곡선 공식
            pos_x = (1-t)**3 * start_x + 3*(1-t)**2*t * cp1_x + 3*(1-t)*t**2 * cp2_x + t**3 * x
            pos_y = (1-t)**3 * start_y + 3*(1-t)**2*t * cp1_y + 3*(1-t)*t**2 * cp2_y + t**3 * y
            
            # 약간의 지터 추가 (자연스러움)
            jitter_x = np.random.normal(0, 1)
            jitter_y = np.random.normal(0, 1)
            
            pyautogui.moveTo(int(pos_x + jitter_x), int(pos_y + jitter_y))
            time.sleep(1/60)
            
    def click_element(self, element: DetectedElement, button: str = 'left', double: bool = False):
        """요소 클릭"""
        self.move_to_element(element)
        time.sleep(0.1)
        
        if double:
            pyautogui.doubleClick(button=button)
        else:
            pyautogui.click(button=button)
            
    def drag_element(self, from_element: DetectedElement, to_element: DetectedElement):
        """요소 드래그"""
        self.move_to_element(from_element)
        time.sleep(0.1)
        
        from_center = from_element.region.center()
        to_center = to_element.region.center()
        
        pyautogui.dragTo(to_center[0], to_center[1], duration=1.0, button='left')
        
    def type_text_natural(self, text: str, field: Optional[DetectedElement] = None):
        """자연스러운 텍스트 입력"""
        if field:
            self.click_element(field)
            time.sleep(0.2)
            
        # 기존 텍스트 선택
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.1)
        
        # 타이핑
        chars_per_second = (self.typing_speed * 5) / 60
        base_delay = 1 / chars_per_second
        
        for char in text:
            # 타이핑 리듬 변화
            delay = base_delay + np.random.normal(0, base_delay * 0.2)
            delay = max(0.02, delay)
            
            pyautogui.write(char)
            time.sleep(delay)
            
            # 때때로 짧은 휴식
            if np.random.random() < 0.1:
                time.sleep(np.random.uniform(0.1, 0.3))
                
    def send_hotkey(self, *keys):
        """단축키 전송"""
        pyautogui.hotkey(*keys)
        
    def scroll_in_element(self, element: DetectedElement, clicks: int):
        """요소 내에서 스크롤"""
        self.move_to_element(element)
        time.sleep(0.1)
        pyautogui.scroll(clicks)

class GodotAutomationEngine:
    """Godot 자동화 엔진"""
    
    def __init__(self):
        self.analyzer = AdvancedScreenAnalyzer()
        self.controller = PrecisionInputController()
        
        # 상태 추적
        self.current_scene = None
        self.selected_node = None
        self.open_dialogs = []
        
        # 작업 기록
        self.action_history = []
        self.undo_stack = []
        
        # Godot 창 핸들
        self.godot_window = None
        self._find_godot_window()
        
    def _find_godot_window(self):
        """Godot 창 찾기"""
        if sys.platform == "win32":
            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if "Godot" in title:
                        windows.append(hwnd)
                return True
                
            windows = []
            win32gui.EnumWindows(callback, windows)
            
            if windows:
                self.godot_window = windows[0]
                self.focus_godot()
                
    def focus_godot(self):
        """Godot 창에 포커스"""
        if self.godot_window and sys.platform == "win32":
            win32gui.SetForegroundWindow(self.godot_window)
            time.sleep(0.2)
            
    def create_new_scene(self, root_type: str = "Node2D") -> bool:
        """새 씬 생성"""
        logger.info(f"새 씬 생성: {root_type}")
        
        # 1. Scene 메뉴 클릭
        self.controller.send_hotkey('alt', 's')
        time.sleep(0.3)
        
        # 2. New Scene 선택
        self.controller.type_text_natural("New Scene")
        time.sleep(0.2)
        self.controller.send_hotkey('enter')
        time.sleep(0.5)
        
        # 3. 루트 노드 타입 선택
        self.controller.type_text_natural(root_type)
        time.sleep(0.2)
        self.controller.send_hotkey('enter')
        time.sleep(0.5)
        
        self.current_scene = root_type
        return True
        
    def add_node(self, node_type: str, parent: Optional[str] = None) -> bool:
        """노드 추가"""
        logger.info(f"노드 추가: {node_type} (부모: {parent})")
        
        # 1. 부모 노드 선택
        if parent:
            if not self.select_node(parent):
                return False
                
        # 2. Add Node 단축키
        self.controller.send_hotkey('ctrl', 'a')
        time.sleep(0.5)
        
        # 3. 노드 타입 검색
        self.controller.type_text_natural(node_type)
        time.sleep(0.3)
        
        # 4. 첫 번째 결과 선택 후 생성
        self.controller.send_hotkey('enter')
        time.sleep(0.5)
        
        return True
        
    def select_node(self, node_path: str) -> bool:
        """씬 트리에서 노드 선택"""
        logger.info(f"노드 선택: {node_path}")
        
        # 화면 분석
        analysis = self.analyzer.analyze_screen()
        
        # 씬 도크에서 노드 찾기
        for tree_item in analysis.get("tree_items", []):
            if node_path in tree_item.text:
                self.controller.click_element(tree_item)
                self.selected_node = node_path
                return True
                
        # 못 찾은 경우 스크롤하면서 찾기
        scene_dock = self._find_scene_dock(analysis)
        if scene_dock:
            # 위로 스크롤
            self.controller.scroll_in_element(scene_dock, 10)
            time.sleep(0.5)
            
            # 다시 분석
            analysis = self.analyzer.analyze_screen()
            for tree_item in analysis.get("tree_items", []):
                if node_path in tree_item.text:
                    self.controller.click_element(tree_item)
                    self.selected_node = node_path
                    return True
                    
        return False
        
    def set_property(self, property_name: str, value: Any) -> bool:
        """노드 속성 설정"""
        logger.info(f"속성 설정: {property_name} = {value}")
        
        # Inspector에서 속성 찾기
        analysis = self.analyzer.analyze_screen()
        
        # 속성 필드 찾기
        property_field = self.analyzer.find_element_by_text(property_name)
        if not property_field:
            # Inspector 스크롤
            inspector = self._find_inspector(analysis)
            if inspector:
                self.controller.scroll_in_element(inspector, -5)
                time.sleep(0.3)
                property_field = self.analyzer.find_element_by_text(property_name)
                
        if property_field:
            # 값 입력 필드 찾기 (보통 라벨 옆)
            value_field_x = property_field.region.x + property_field.region.width + 10
            value_field_y = property_field.region.y
            
            # 클릭하고 값 입력
            pyautogui.click(value_field_x, value_field_y)
            time.sleep(0.1)
            
            # 기존 값 선택
            self.controller.send_hotkey('ctrl', 'a')
            time.sleep(0.1)
            
            # 새 값 입력
            self.controller.type_text_natural(str(value))
            time.sleep(0.1)
            self.controller.send_hotkey('enter')
            
            return True
            
        return False
        
    def attach_script(self, script_content: Optional[str] = None) -> bool:
        """스크립트 첨부"""
        logger.info("스크립트 첨부")
        
        # 스크립트 아이콘 클릭 또는 단축키
        self.controller.send_hotkey('ctrl', 'shift', 'a')
        time.sleep(0.5)
        
        if script_content:
            # 기본 템플릿 대체
            time.sleep(1)  # 에디터 로드 대기
            self.controller.send_hotkey('ctrl', 'a')
            time.sleep(0.1)
            self.controller.type_text_natural(script_content)
            
        return True
        
    def save_scene(self, filename: str) -> bool:
        """씬 저장"""
        logger.info(f"씬 저장: {filename}")
        
        # Ctrl+S
        self.controller.send_hotkey('ctrl', 's')
        time.sleep(0.5)
        
        # 파일명 입력
        self.controller.type_text_natural(filename)
        time.sleep(0.2)
        self.controller.send_hotkey('enter')
        time.sleep(0.5)
        
        return True
        
    def run_scene(self) -> bool:
        """씬 실행"""
        logger.info("씬 실행")
        
        # F6 (현재 씬 실행)
        self.controller.send_hotkey('f6')
        return True
        
    def _find_scene_dock(self, analysis: Dict) -> Optional[DetectedElement]:
        """씬 도크 찾기"""
        for panel in analysis.get("panels", []):
            if panel.element_type == GodotElementType.SCENE_DOCK:
                return panel
        return None
        
    def _find_inspector(self, analysis: Dict) -> Optional[DetectedElement]:
        """Inspector 찾기"""
        for panel in analysis.get("panels", []):
            if "Inspector" in panel.text or panel.region.x > 1000:  # 보통 오른쪽
                return panel
        return None
        
    def create_complete_scene(self, scene_config: Dict) -> bool:
        """완전한 씬 생성 (설정 기반)"""
        logger.info("완전한 씬 생성 시작")
        
        # 1. 새 씬 생성
        root_type = scene_config.get("root_type", "Node2D")
        self.create_new_scene(root_type)
        
        # 2. 노드 추가
        for node_config in scene_config.get("nodes", []):
            node_type = node_config["type"]
            parent = node_config.get("parent")
            
            if self.add_node(node_type, parent):
                # 속성 설정
                for prop, value in node_config.get("properties", {}).items():
                    self.set_property(prop, value)
                    
                # 스크립트 첨부
                if "script" in node_config:
                    self.attach_script(node_config["script"])
                    
        # 3. 씬 저장
        filename = scene_config.get("filename", "NewScene.tscn")
        self.save_scene(filename)
        
        logger.info("✅ 씬 생성 완료")
        return True

# 사용 예시
def demo_godot_automation():
    """Godot 자동화 데모"""
    
    # 엔진 초기화
    engine = GodotAutomationEngine()
    
    # 2D 플랫포머 씬 설정
    platformer_config = {
        "root_type": "Node2D",
        "filename": "Player.tscn",
        "nodes": [
            {
                "type": "CharacterBody2D",
                "parent": None,
                "properties": {
                    "position": "0, 0"
                },
                "script": """extends CharacterBody2D

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
"""
            },
            {
                "type": "Sprite2D",
                "parent": "CharacterBody2D",
                "properties": {
                    "texture": "res://player.png",
                    "hframes": 4,
                    "vframes": 1
                }
            },
            {
                "type": "CollisionShape2D",
                "parent": "CharacterBody2D",
                "properties": {
                    "shape": "RectangleShape2D"
                }
            },
            {
                "type": "AnimationPlayer",
                "parent": "CharacterBody2D"
            }
        ]
    }
    
    # 씬 생성
    engine.create_complete_scene(platformer_config)
    
    # 실행
    engine.run_scene()

if __name__ == "__main__":
    demo_godot_automation()