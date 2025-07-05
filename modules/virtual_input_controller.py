"""
AutoCI 가상 입력 컨트롤러
독립적인 마우스/키보드 입력을 시뮬레이션하여 AI가 자율적으로 조작
"""

import asyncio
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
import json
from pathlib import Path

# 플랫폼별 입력 라이브러리
try:
    import pyautogui
    pyautogui.FAILSAFE = False  # 안전 모드 비활성화 (AI 용)
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    import pynput
    from pynput.mouse import Button, Controller as MouseController
    from pynput.keyboard import Key, Controller as KeyboardController
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

logger = logging.getLogger(__name__)


class InputMode(Enum):
    """입력 모드"""
    GAME_DEVELOPMENT = "game_development"  # 게임 개발 모드
    CODE_EDITING = "code_editing"  # 코드 편집 모드
    TESTING = "testing"  # 테스트 모드
    GODOT_EDITOR = "godot_editor"  # Godot 에디터 조작 모드


@dataclass
class VirtualScreen:
    """가상 스크린 정보"""
    width: int = 1920
    height: int = 1080
    scale: float = 1.0
    
    def to_real_coords(self, x: int, y: int) -> Tuple[int, int]:
        """가상 좌표를 실제 좌표로 변환"""
        return int(x * self.scale), int(y * self.scale)


class VirtualInputController:
    """AutoCI 전용 가상 입력 컨트롤러"""
    
    def __init__(self):
        self.mode = InputMode.GAME_DEVELOPMENT
        self.virtual_screen = VirtualScreen()
        self.is_active = False
        self.action_history: List[Dict[str, Any]] = []
        self.macro_library: Dict[str, List[Dict[str, Any]]] = {}
        
        # 입력 컨트롤러 초기화
        if PYNPUT_AVAILABLE:
            self.mouse = MouseController()
            self.keyboard = KeyboardController()
            logger.info("✅ pynput 기반 가상 입력 시스템 활성화")
        elif PYAUTOGUI_AVAILABLE:
            logger.info("✅ pyautogui 기반 가상 입력 시스템 활성화")
        else:
            logger.warning("⚠️ 가상 입력 라이브러리가 설치되지 않음")
            
        self._load_macros()
        
    def _load_macros(self):
        """사전 정의된 매크로 로드"""
        self.macro_library = {
            "godot_new_scene": [
                {"type": "key", "keys": ["ctrl", "n"]},
                {"type": "wait", "duration": 0.5}
            ],
            "godot_save": [
                {"type": "key", "keys": ["ctrl", "s"]},
                {"type": "wait", "duration": 0.3}
            ],
            "godot_run_game": [
                {"type": "key", "key": "F5"},
                {"type": "wait", "duration": 1.0}
            ],
            "code_format": [
                {"type": "key", "keys": ["ctrl", "shift", "f"]},
                {"type": "wait", "duration": 0.2}
            ],
            "code_comment": [
                {"type": "key", "keys": ["ctrl", "/"]},
                {"type": "wait", "duration": 0.1}
            ]
        }
        
    async def activate(self):
        """가상 입력 시스템 활성화"""
        self.is_active = True
        logger.info("🎮 가상 입력 컨트롤러 활성화")
        
    async def deactivate(self):
        """가상 입력 시스템 비활성화"""
        self.is_active = False
        logger.info("🛑 가상 입력 컨트롤러 비활성화")
        
    def set_mode(self, mode: InputMode):
        """입력 모드 설정"""
        self.mode = mode
        logger.info(f"📋 입력 모드 변경: {mode.value}")
        
    async def move_mouse(self, x: int, y: int, duration: float = 0.5):
        """마우스를 지정된 위치로 이동"""
        if not self.is_active:
            return
            
        real_x, real_y = self.virtual_screen.to_real_coords(x, y)
        
        if PYNPUT_AVAILABLE:
            # 부드러운 이동을 위한 애니메이션
            start_x, start_y = self.mouse.position
            steps = int(duration * 60)  # 60 FPS
            
            for i in range(steps):
                progress = (i + 1) / steps
                # 이징 함수 (ease-in-out)
                t = progress * progress * (3.0 - 2.0 * progress)
                
                current_x = start_x + (real_x - start_x) * t
                current_y = start_y + (real_y - start_y) * t
                
                self.mouse.position = (current_x, current_y)
                await asyncio.sleep(duration / steps)
                
        elif PYAUTOGUI_AVAILABLE:
            pyautogui.moveTo(real_x, real_y, duration=duration)
            
        self._record_action("mouse_move", {"x": x, "y": y, "duration": duration})
        
    async def click(self, button: str = "left", clicks: int = 1):
        """마우스 클릭"""
        if not self.is_active:
            return
            
        if PYNPUT_AVAILABLE:
            btn = Button.left if button == "left" else Button.right
            for _ in range(clicks):
                self.mouse.click(btn)
                await asyncio.sleep(0.1)
        elif PYAUTOGUI_AVAILABLE:
            pyautogui.click(button=button, clicks=clicks)
            
        self._record_action("mouse_click", {"button": button, "clicks": clicks})
        
    async def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 1.0):
        """마우스 드래그"""
        if not self.is_active:
            return
            
        await self.move_mouse(start_x, start_y, duration=0.2)
        
        if PYNPUT_AVAILABLE:
            self.mouse.press(Button.left)
            await self.move_mouse(end_x, end_y, duration=duration)
            self.mouse.release(Button.left)
        elif PYAUTOGUI_AVAILABLE:
            pyautogui.dragTo(end_x, end_y, duration=duration)
            
        self._record_action("mouse_drag", {
            "start": (start_x, start_y),
            "end": (end_x, end_y),
            "duration": duration
        })
        
    async def type_text(self, text: str, interval: float = 0.05):
        """텍스트 입력"""
        if not self.is_active:
            return
            
        if PYNPUT_AVAILABLE:
            for char in text:
                self.keyboard.type(char)
                await asyncio.sleep(interval)
        elif PYAUTOGUI_AVAILABLE:
            pyautogui.typewrite(text, interval=interval)
            
        self._record_action("type_text", {"text": text, "interval": interval})
        
    async def press_key(self, key: str):
        """단일 키 입력"""
        if not self.is_active:
            return
            
        if PYNPUT_AVAILABLE:
            # 특수 키 매핑
            special_keys = {
                "enter": Key.enter,
                "tab": Key.tab,
                "esc": Key.esc,
                "space": Key.space,
                "backspace": Key.backspace,
                "delete": Key.delete,
                "up": Key.up,
                "down": Key.down,
                "left": Key.left,
                "right": Key.right,
                "F1": Key.f1,
                "F5": Key.f5,
                "F11": Key.f11
            }
            
            if key in special_keys:
                self.keyboard.press(special_keys[key])
                self.keyboard.release(special_keys[key])
            else:
                self.keyboard.press(key)
                self.keyboard.release(key)
        elif PYAUTOGUI_AVAILABLE:
            pyautogui.press(key)
            
        self._record_action("press_key", {"key": key})
        
    async def hotkey(self, *keys):
        """단축키 조합 입력"""
        if not self.is_active:
            return
            
        if PYNPUT_AVAILABLE:
            # 모든 키 누르기
            pressed_keys = []
            for key in keys:
                if key in ["ctrl", "alt", "shift"]:
                    k = getattr(Key, key)
                elif key == "win":
                    # Windows 키 처리
                    try:
                        k = Key.cmd  # Mac/Linux
                    except AttributeError:
                        k = Key.cmd_l  # 왼쪽 Windows 키
                else:
                    k = key
                self.keyboard.press(k)
                pressed_keys.append(k)
                await asyncio.sleep(0.05)
                
            # 역순으로 키 놓기
            for k in reversed(pressed_keys):
                self.keyboard.release(k)
                await asyncio.sleep(0.05)
        elif PYAUTOGUI_AVAILABLE:
            pyautogui.hotkey(*keys)
            
        self._record_action("hotkey", {"keys": keys})
        
    async def execute_macro(self, macro_name: str):
        """사전 정의된 매크로 실행"""
        if macro_name not in self.macro_library:
            logger.warning(f"매크로를 찾을 수 없음: {macro_name}")
            return
            
        logger.info(f"📝 매크로 실행: {macro_name}")
        
        for action in self.macro_library[macro_name]:
            if action["type"] == "key":
                if "keys" in action:
                    await self.hotkey(*action["keys"])
                else:
                    await self.press_key(action["key"])
            elif action["type"] == "wait":
                await asyncio.sleep(action["duration"])
            elif action["type"] == "click":
                await self.click(action.get("button", "left"))
            elif action["type"] == "move":
                await self.move_mouse(action["x"], action["y"])
                
    async def godot_create_node(self, node_type: str, name: str):
        """Godot에서 노드 생성"""
        logger.info(f"🎮 Godot 노드 생성: {node_type} ({name})")
        
        # Scene 패널에서 우클릭
        await self.move_mouse(200, 300)
        await self.click("right")
        await asyncio.sleep(0.3)
        
        # "Add Child Node" 선택
        await self.type_text("Add Child")
        await self.press_key("enter")
        await asyncio.sleep(0.5)
        
        # 노드 타입 검색
        await self.type_text(node_type)
        await asyncio.sleep(0.3)
        await self.press_key("enter")
        
        # 노드 이름 변경
        await asyncio.sleep(0.5)
        await self.press_key("F2")
        await self.type_text(name)
        await self.press_key("enter")
        
    async def godot_add_script(self, script_content: str):
        """Godot에서 스크립트 추가"""
        logger.info("📝 Godot 스크립트 추가")
        
        # 스크립트 버튼 클릭
        await self.hotkey("ctrl", "alt", "s")
        await asyncio.sleep(0.5)
        
        # 스크립트 내용 입력
        await self.type_text(script_content)
        
        # 저장
        await self.execute_macro("godot_save")
        
    def _record_action(self, action_type: str, data: Dict[str, Any]):
        """액션 기록"""
        self.action_history.append({
            "type": action_type,
            "data": data,
            "timestamp": time.time(),
            "mode": self.mode.value
        })
        
        # 최근 100개만 유지
        if len(self.action_history) > 100:
            self.action_history.pop(0)
            
    def get_action_history(self) -> List[Dict[str, Any]]:
        """액션 히스토리 반환"""
        return self.action_history.copy()
        
    async def replay_actions(self, actions: List[Dict[str, Any]]):
        """녹화된 액션 재생"""
        logger.info(f"🔄 {len(actions)}개 액션 재생 시작")
        
        for action in actions:
            action_type = action["type"]
            data = action["data"]
            
            if action_type == "mouse_move":
                await self.move_mouse(data["x"], data["y"], data["duration"])
            elif action_type == "mouse_click":
                await self.click(data["button"], data["clicks"])
            elif action_type == "mouse_drag":
                await self.drag(*data["start"], *data["end"], data["duration"])
            elif action_type == "type_text":
                await self.type_text(data["text"], data["interval"])
            elif action_type == "press_key":
                await self.press_key(data["key"])
            elif action_type == "hotkey":
                await self.hotkey(*data["keys"])
                
            await asyncio.sleep(0.1)  # 액션 간 간격
            
    def get_pattern_statistics(self) -> Dict[str, int]:
        """입력 패턴 통계 반환"""
        stats = {
            "total_patterns": len(self.action_history),
            "sequences": len(self.macro_library),
            "mouse_moves": sum(1 for a in self.action_history if a["type"] == "mouse_move"),
            "clicks": sum(1 for a in self.action_history if a["type"] == "mouse_click"),
            "keyboard_inputs": sum(1 for a in self.action_history if a["type"] in ["type_text", "press_key", "hotkey"]),
            "godot_operations": sum(1 for a in self.action_history if a.get("mode") == InputMode.GODOT_EDITOR.value)
        }
        return stats


# 싱글톤 인스턴스
_virtual_input = None


def get_virtual_input() -> VirtualInputController:
    """가상 입력 컨트롤러 싱글톤 반환"""
    global _virtual_input
    if _virtual_input is None:
        _virtual_input = VirtualInputController()
    return _virtual_input