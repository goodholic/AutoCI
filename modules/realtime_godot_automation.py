#!/usr/bin/env python3
"""
실시간 Godot 자동화 시스템
화면을 실시간으로 분석하고 지능적으로 조작
"""

import os
import sys
import json
import time
import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
import threading
import queue
from collections import deque

# 비전 기반 컨트롤러
sys.path.insert(0, str(Path(__file__).parent))
from vision_based_godot_controller import (
    GodotAutomationEngine,
    AdvancedScreenAnalyzer,
    PrecisionInputController,
    DetectedElement,
    GodotElementType,
    ScreenRegion
)

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass
class AutomationTask:
    """자동화 작업"""
    task_id: str
    task_type: str
    description: str
    steps: List[Dict[str, Any]]
    status: str = "pending"
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    
@dataclass
class ScreenState:
    """화면 상태"""
    timestamp: float
    screenshot: np.ndarray
    detected_elements: Dict[str, List[DetectedElement]]
    active_window: str
    mouse_position: Tuple[int, int]
    focused_element: Optional[DetectedElement] = None
    
class RealtimeScreenMonitor:
    """실시간 화면 모니터"""
    
    def __init__(self, fps: int = 10):
        self.analyzer = AdvancedScreenAnalyzer()
        self.fps = fps
        self.frame_time = 1.0 / fps
        
        # 상태 버퍼
        self.state_buffer = deque(maxlen=fps * 10)  # 10초 분량
        self.current_state = None
        
        # 변화 감지
        self.change_callbacks = []
        self.element_trackers = {}
        
        # 모니터링 스레드
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """모니터링 시작"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
        logger.info(f"화면 모니터링 시작 (FPS: {self.fps})")
        
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("화면 모니터링 중지")
        
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            start_time = time.time()
            
            # 화면 캡처 및 분석
            state = self._capture_current_state()
            
            # 상태 저장
            self.current_state = state
            self.state_buffer.append(state)
            
            # 변화 감지
            self._detect_changes(state)
            
            # 프레임 타이밍 조절
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_time - elapsed)
            time.sleep(sleep_time)
            
    def _capture_current_state(self) -> ScreenState:
        """현재 상태 캡처"""
        screenshot = self.analyzer.capture_screen()
        detected_elements = self.analyzer.analyze_screen(screenshot)
        
        import pyautogui
        mouse_pos = pyautogui.position()
        
        # 활성 창 정보
        active_window = "Godot"  # 실제로는 창 제목 가져오기
        
        # 포커스된 요소 찾기
        focused_element = None
        for elements in detected_elements.values():
            for element in elements:
                if element.region.contains_point(*mouse_pos):
                    focused_element = element
                    break
                    
        return ScreenState(
            timestamp=time.time(),
            screenshot=screenshot,
            detected_elements=detected_elements,
            active_window=active_window,
            mouse_position=mouse_pos,
            focused_element=focused_element
        )
        
    def _detect_changes(self, current_state: ScreenState):
        """상태 변화 감지"""
        if len(self.state_buffer) < 2:
            return
            
        previous_state = self.state_buffer[-2]
        
        # 요소 변화 감지
        for element_type, current_elements in current_state.detected_elements.items():
            previous_elements = previous_state.detected_elements.get(element_type, [])
            
            # 새로 나타난 요소
            new_elements = self._find_new_elements(current_elements, previous_elements)
            if new_elements:
                self._trigger_callbacks("element_appeared", new_elements)
                
            # 사라진 요소
            disappeared = self._find_disappeared_elements(current_elements, previous_elements)
            if disappeared:
                self._trigger_callbacks("element_disappeared", disappeared)
                
        # 마우스 이동 감지
        if current_state.mouse_position != previous_state.mouse_position:
            self._trigger_callbacks("mouse_moved", current_state.mouse_position)
            
    def _find_new_elements(self, current: List[DetectedElement], 
                          previous: List[DetectedElement]) -> List[DetectedElement]:
        """새로 나타난 요소 찾기"""
        new_elements = []
        
        for curr_elem in current:
            found = False
            for prev_elem in previous:
                if self._elements_match(curr_elem, prev_elem):
                    found = True
                    break
            if not found:
                new_elements.append(curr_elem)
                
        return new_elements
        
    def _find_disappeared_elements(self, current: List[DetectedElement], 
                                  previous: List[DetectedElement]) -> List[DetectedElement]:
        """사라진 요소 찾기"""
        disappeared = []
        
        for prev_elem in previous:
            found = False
            for curr_elem in current:
                if self._elements_match(curr_elem, prev_elem):
                    found = True
                    break
            if not found:
                disappeared.append(prev_elem)
                
        return disappeared
        
    def _elements_match(self, elem1: DetectedElement, elem2: DetectedElement) -> bool:
        """두 요소가 같은지 확인"""
        # 위치와 타입이 비슷하면 같은 요소로 간주
        pos_diff = abs(elem1.region.x - elem2.region.x) + abs(elem1.region.y - elem2.region.y)
        return (elem1.element_type == elem2.element_type and 
                pos_diff < 10 and
                abs(elem1.region.width - elem2.region.width) < 5)
                
    def _trigger_callbacks(self, event_type: str, data: Any):
        """콜백 트리거"""
        for callback in self.change_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"콜백 오류: {e}")
                
    def register_callback(self, callback: Callable):
        """변화 감지 콜백 등록"""
        self.change_callbacks.append(callback)
        
    def track_element(self, element_id: str, element: DetectedElement):
        """특정 요소 추적"""
        self.element_trackers[element_id] = {
            "element": element,
            "history": deque(maxlen=100),
            "last_seen": time.time()
        }
        
    def get_element_trajectory(self, element_id: str) -> List[Tuple[int, int]]:
        """요소의 이동 궤적"""
        if element_id in self.element_trackers:
            return list(self.element_trackers[element_id]["history"])
        return []
        
    def wait_for_element(self, element_type: GodotElementType, 
                        text: Optional[str] = None, 
                        timeout: float = 10.0) -> Optional[DetectedElement]:
        """특정 요소가 나타날 때까지 대기"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.current_state:
                for elements in self.current_state.detected_elements.values():
                    for element in elements:
                        if element.element_type == element_type:
                            if text is None or text in element.text:
                                return element
            time.sleep(0.1)
            
        return None
        
    def wait_for_stable_screen(self, stability_time: float = 1.0) -> bool:
        """화면이 안정될 때까지 대기"""
        if len(self.state_buffer) < int(self.fps * stability_time):
            time.sleep(stability_time)
            
        # 최근 프레임들의 변화량 계산
        recent_states = list(self.state_buffer)[-int(self.fps * stability_time):]
        
        for i in range(1, len(recent_states)):
            diff = cv2.absdiff(recent_states[i].screenshot, recent_states[i-1].screenshot)
            if np.mean(diff) > 5:  # 임계값
                return False
                
        return True

class IntelligentTaskExecutor:
    """지능형 작업 실행기"""
    
    def __init__(self):
        self.engine = GodotAutomationEngine()
        self.monitor = RealtimeScreenMonitor()
        
        # 작업 큐
        self.task_queue = queue.Queue()
        self.current_task = None
        
        # 학습 데이터
        self.execution_history = []
        self.success_patterns = []
        self.failure_patterns = []
        
        # 실행 전략
        self.strategies = self._load_strategies()
        
        # 모니터 콜백 등록
        self.monitor.register_callback(self._on_screen_change)
        
    def _load_strategies(self) -> Dict[str, Callable]:
        """실행 전략 로드"""
        return {
            "create_node": self._strategy_create_node,
            "set_property": self._strategy_set_property,
            "attach_script": self._strategy_attach_script,
            "save_scene": self._strategy_save_scene,
            "navigate_menu": self._strategy_navigate_menu
        }
        
    def start(self):
        """실행기 시작"""
        self.monitor.start_monitoring()
        self.execution_thread = threading.Thread(target=self._execution_loop)
        self.execution_thread.start()
        
    def stop(self):
        """실행기 중지"""
        self.monitor.stop_monitoring()
        
    def _execution_loop(self):
        """작업 실행 루프"""
        while True:
            try:
                # 작업 가져오기
                task = self.task_queue.get(timeout=1)
                self.current_task = task
                
                # 작업 실행
                logger.info(f"작업 실행 시작: {task.task_id} - {task.description}")
                self._execute_task(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"작업 실행 오류: {e}")
                if self.current_task:
                    self.current_task.status = "failed"
                    self.current_task.error = str(e)
                    
    def _execute_task(self, task: AutomationTask):
        """작업 실행"""
        task.status = "running"
        
        try:
            for i, step in enumerate(task.steps):
                # 진행률 업데이트
                task.progress = (i / len(task.steps)) * 100
                
                # 단계 실행
                step_type = step["type"]
                if step_type in self.strategies:
                    strategy = self.strategies[step_type]
                    result = strategy(step)
                    
                    if not result:
                        raise Exception(f"단계 실패: {step}")
                        
                else:
                    logger.warning(f"알 수 없는 단계 타입: {step_type}")
                    
                # 화면 안정화 대기
                self.monitor.wait_for_stable_screen()
                
            # 작업 완료
            task.status = "completed"
            task.progress = 100.0
            
            # 성공 패턴 기록
            self._record_success_pattern(task)
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            
            # 실패 패턴 기록
            self._record_failure_pattern(task, e)
            
    def _strategy_create_node(self, step: Dict) -> bool:
        """노드 생성 전략"""
        node_type = step["node_type"]
        parent = step.get("parent")
        
        # Add Node 버튼 찾기
        add_button = self.monitor.wait_for_element(
            GodotElementType.BUTTON, 
            text="Add", 
            timeout=5.0
        )
        
        if add_button:
            self.engine.controller.click_element(add_button)
        else:
            # 단축키 사용
            self.engine.controller.send_hotkey('ctrl', 'a')
            
        time.sleep(0.5)
        
        # 노드 타입 입력
        search_field = self.monitor.wait_for_element(
            GodotElementType.TEXT_INPUT,
            timeout=3.0
        )
        
        if search_field:
            self.engine.controller.type_text_natural(node_type, search_field)
            time.sleep(0.3)
            self.engine.controller.send_hotkey('enter')
            return True
            
        return False
        
    def _strategy_set_property(self, step: Dict) -> bool:
        """속성 설정 전략"""
        property_name = step["property"]
        value = step["value"]
        
        # Inspector에서 속성 찾기
        prop_element = self._find_property_in_inspector(property_name)
        
        if prop_element:
            # 값 필드 클릭
            value_region = ScreenRegion(
                prop_element.region.x + prop_element.region.width + 10,
                prop_element.region.y,
                100,
                prop_element.region.height,
                "value_field"
            )
            
            value_element = DetectedElement(
                element_type=GodotElementType.TEXT_INPUT,
                region=value_region,
                confidence=0.8
            )
            
            self.engine.controller.click_element(value_element)
            time.sleep(0.1)
            
            # 값 입력
            self.engine.controller.send_hotkey('ctrl', 'a')
            self.engine.controller.type_text_natural(str(value))
            self.engine.controller.send_hotkey('enter')
            
            return True
            
        return False
        
    def _strategy_attach_script(self, step: Dict) -> bool:
        """스크립트 첨부 전략"""
        script_content = step.get("content", "")
        
        # 스크립트 버튼 찾기
        script_button = self._find_script_button()
        
        if script_button:
            self.engine.controller.click_element(script_button)
        else:
            self.engine.controller.send_hotkey('ctrl', 'shift', 'a')
            
        time.sleep(1)
        
        # 스크립트 에디터 대기
        editor = self.monitor.wait_for_element(
            GodotElementType.SCRIPT_EDITOR,
            timeout=5.0
        )
        
        if editor and script_content:
            # 전체 선택 후 교체
            self.engine.controller.send_hotkey('ctrl', 'a')
            time.sleep(0.1)
            self.engine.controller.type_text_natural(script_content)
            
        return True
        
    def _strategy_save_scene(self, step: Dict) -> bool:
        """씬 저장 전략"""
        filename = step["filename"]
        
        # Ctrl+S
        self.engine.controller.send_hotkey('ctrl', 's')
        time.sleep(0.5)
        
        # 파일 다이얼로그 대기
        file_dialog = self.monitor.wait_for_element(
            GodotElementType.FILE_DIALOG,
            timeout=3.0
        )
        
        if file_dialog:
            # 파일명 입력
            self.engine.controller.type_text_natural(filename)
            time.sleep(0.2)
            self.engine.controller.send_hotkey('enter')
            return True
            
        return False
        
    def _strategy_navigate_menu(self, step: Dict) -> bool:
        """메뉴 네비게이션 전략"""
        menu_path = step["path"]
        
        for i, menu_item in enumerate(menu_path):
            if i == 0:
                # 최상위 메뉴
                menu_element = self.monitor.wait_for_element(
                    GodotElementType.MENU_BAR,
                    text=menu_item,
                    timeout=3.0
                )
                
                if menu_element:
                    self.engine.controller.click_element(menu_element)
                else:
                    # Alt + 첫 글자
                    self.engine.controller.send_hotkey('alt', menu_item[0].lower())
                    
            else:
                # 하위 메뉴
                time.sleep(0.3)
                submenu = self.monitor.wait_for_element(
                    GodotElementType.DROPDOWN,
                    text=menu_item,
                    timeout=2.0
                )
                
                if submenu:
                    self.engine.controller.click_element(submenu)
                else:
                    # 텍스트 입력으로 찾기
                    self.engine.controller.type_text_natural(menu_item[:3])
                    time.sleep(0.1)
                    self.engine.controller.send_hotkey('enter')
                    
        return True
        
    def _find_property_in_inspector(self, property_name: str) -> Optional[DetectedElement]:
        """Inspector에서 속성 찾기"""
        # 현재 화면에서 찾기
        element = self.engine.analyzer.find_element_by_text(property_name)
        if element:
            return element
            
        # 스크롤하면서 찾기
        inspector = self._find_inspector()
        if inspector:
            for _ in range(5):  # 최대 5번 스크롤
                self.engine.controller.scroll_in_element(inspector, -3)
                time.sleep(0.3)
                
                element = self.engine.analyzer.find_element_by_text(property_name)
                if element:
                    return element
                    
        return None
        
    def _find_inspector(self) -> Optional[DetectedElement]:
        """Inspector 찾기"""
        if self.monitor.current_state:
            for panel in self.monitor.current_state.detected_elements.get("panels", []):
                if panel.region.x > 1000:  # 보통 오른쪽에 위치
                    return panel
        return None
        
    def _find_script_button(self) -> Optional[DetectedElement]:
        """스크립트 버튼 찾기"""
        # 템플릿 매칭으로 찾기
        template_path = Path(__file__).parent / "templates" / "script_button.png"
        if template_path.exists():
            return self.engine.analyzer.find_element_by_image(str(template_path))
        return None
        
    def _on_screen_change(self, event_type: str, data: Any):
        """화면 변화 콜백"""
        if event_type == "element_appeared":
            logger.debug(f"요소 나타남: {data}")
        elif event_type == "element_disappeared":
            logger.debug(f"요소 사라짐: {data}")
            
    def _record_success_pattern(self, task: AutomationTask):
        """성공 패턴 기록"""
        pattern = {
            "task_type": task.task_type,
            "steps": task.steps,
            "screen_states": list(self.monitor.state_buffer)[-10:],  # 마지막 10프레임
            "timestamp": datetime.now().isoformat()
        }
        self.success_patterns.append(pattern)
        
    def _record_failure_pattern(self, task: AutomationTask, error: Exception):
        """실패 패턴 기록"""
        pattern = {
            "task_type": task.task_type,
            "steps": task.steps,
            "error": str(error),
            "failed_step": task.progress,
            "screen_states": list(self.monitor.state_buffer)[-10:],
            "timestamp": datetime.now().isoformat()
        }
        self.failure_patterns.append(pattern)
        
    def add_task(self, task: AutomationTask):
        """작업 추가"""
        self.task_queue.put(task)
        logger.info(f"작업 추가됨: {task.task_id}")

class AutoCICreateController:
    """AutoCI Create 메인 컨트롤러"""
    
    def __init__(self):
        self.executor = IntelligentTaskExecutor()
        self.task_id_counter = 0
        
    def start(self):
        """시스템 시작"""
        self.executor.start()
        logger.info("AutoCI Create 시스템 시작")
        
    def stop(self):
        """시스템 중지"""
        self.executor.stop()
        logger.info("AutoCI Create 시스템 중지")
        
    def create_2d_platformer_player(self) -> str:
        """2D 플랫포머 플레이어 생성"""
        task_id = f"task_{self.task_id_counter}"
        self.task_id_counter += 1
        
        task = AutomationTask(
            task_id=task_id,
            task_type="create_scene",
            description="2D 플랫포머 플레이어 씬 생성",
            steps=[
                {
                    "type": "navigate_menu",
                    "path": ["Scene", "New Scene"]
                },
                {
                    "type": "create_node",
                    "node_type": "CharacterBody2D"
                },
                {
                    "type": "create_node",
                    "node_type": "Sprite2D",
                    "parent": "CharacterBody2D"
                },
                {
                    "type": "create_node",
                    "node_type": "CollisionShape2D",
                    "parent": "CharacterBody2D"
                },
                {
                    "type": "create_node",
                    "node_type": "AnimationPlayer",
                    "parent": "CharacterBody2D"
                },
                {
                    "type": "set_property",
                    "property": "position",
                    "value": "0, 0"
                },
                {
                    "type": "attach_script",
                    "content": """extends CharacterBody2D

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
                    "type": "save_scene",
                    "filename": "Player.tscn"
                }
            ]
        )
        
        self.executor.add_task(task)
        return task_id
        
    def create_ui_menu(self) -> str:
        """UI 메뉴 생성"""
        task_id = f"task_{self.task_id_counter}"
        self.task_id_counter += 1
        
        task = AutomationTask(
            task_id=task_id,
            task_type="create_ui",
            description="메인 메뉴 UI 생성",
            steps=[
                {
                    "type": "navigate_menu",
                    "path": ["Scene", "New Scene"]
                },
                {
                    "type": "create_node",
                    "node_type": "Control"
                },
                {
                    "type": "set_property",
                    "property": "anchor_preset",
                    "value": "15"  # Full Rect
                },
                {
                    "type": "create_node",
                    "node_type": "VBoxContainer",
                    "parent": "Control"
                },
                {
                    "type": "create_node",
                    "node_type": "Label",
                    "parent": "Control/VBoxContainer"
                },
                {
                    "type": "set_property",
                    "property": "text",
                    "value": "Main Menu"
                },
                {
                    "type": "create_node",
                    "node_type": "Button",
                    "parent": "Control/VBoxContainer"
                },
                {
                    "type": "set_property",
                    "property": "text",
                    "value": "Play"
                },
                {
                    "type": "create_node",
                    "node_type": "Button",
                    "parent": "Control/VBoxContainer"
                },
                {
                    "type": "set_property",
                    "property": "text",
                    "value": "Options"
                },
                {
                    "type": "create_node",
                    "node_type": "Button",
                    "parent": "Control/VBoxContainer"
                },
                {
                    "type": "set_property",
                    "property": "text",
                    "value": "Exit"
                },
                {
                    "type": "save_scene",
                    "filename": "MainMenu.tscn"
                }
            ]
        )
        
        self.executor.add_task(task)
        return task_id
        
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """작업 상태 조회"""
        # 실제로는 작업 저장소에서 조회
        return {
            "task_id": task_id,
            "status": "running",
            "progress": 50.0
        }

# 사용 예시
def main():
    """메인 함수"""
    controller = AutoCICreateController()
    
    try:
        # 시스템 시작
        controller.start()
        
        # 플레이어 생성
        player_task_id = controller.create_2d_platformer_player()
        print(f"플레이어 생성 작업 시작: {player_task_id}")
        
        # 잠시 대기
        time.sleep(30)
        
        # UI 생성
        ui_task_id = controller.create_ui_menu()
        print(f"UI 생성 작업 시작: {ui_task_id}")
        
        # 작업 완료 대기
        time.sleep(30)
        
    finally:
        # 시스템 중지
        controller.stop()

if __name__ == "__main__":
    main()