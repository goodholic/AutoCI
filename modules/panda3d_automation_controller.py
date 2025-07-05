"""
Panda3D 자동화 컨트롤러
AI가 Panda3D 엔진을 직접 조작하여 게임을 개발하는 시스템
"""

import os
import sys
import json
import time
import threading
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging
import numpy as np

# GUI 자동화 라이브러리는 선택적으로 임포트
try:
    import pyautogui
    import keyboard
    import mouse
    GUI_AVAILABLE = True
except (ImportError, Exception) as e:
    # WSL이나 headless 환경에서는 GUI 자동화 비활성화
    GUI_AVAILABLE = False
    logging.warning(f"GUI 자동화 라이브러리를 로드할 수 없습니다: {e}")
    logging.warning("GUI 자동화 기능이 비활성화됩니다.")

# Panda3D는 항상 임포트 시도
try:
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task
    from panda3d.core import *
    PANDA3D_AVAILABLE = True
except ImportError:
    PANDA3D_AVAILABLE = False
    logging.warning("Panda3D를 로드할 수 없습니다. 일부 기능이 제한됩니다.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionType(Enum):
    """자동화 액션 타입"""
    MOUSE_CLICK = "mouse_click"
    MOUSE_DRAG = "mouse_drag"
    MOUSE_MOVE = "mouse_move"
    KEY_PRESS = "key_press"
    KEY_HOLD = "key_hold"
    KEY_RELEASE = "key_release"
    TYPE_TEXT = "type_text"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    MENU_NAVIGATE = "menu_navigate"
    CODE_WRITE = "code_write"
    FILE_OPERATION = "file_operation"


@dataclass
class AutomationAction:
    """자동화 액션 정의"""
    action_type: ActionType
    parameters: Dict[str, Any]
    description: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class Panda3DAutomationController:
    """Panda3D 엔진 자동화 컨트롤러"""
    
    def __init__(self, ai_model_controller=None):
        self.ai_controller = ai_model_controller
        self.action_history: List[AutomationAction] = []
        self.current_project: Optional[str] = None
        self.panda3d_process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.action_queue: List[AutomationAction] = []
        self.execution_thread: Optional[threading.Thread] = None
        
        # Panda3D 윈도우 정보
        self.window_info = {
            "title": "Panda3D Game Development",
            "position": None,
            "size": None
        }
        
        # 안전 설정 (GUI가 사용 가능한 경우만)
        if GUI_AVAILABLE:
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.1
        
    def start_panda3d_project(self, project_name: str, project_type: str = "platformer") -> bool:
        """새 Panda3D 프로젝트 시작"""
        try:
            project_path = Path(f"game_projects/{project_name}")
            project_path.mkdir(parents=True, exist_ok=True)
            
            # 기본 프로젝트 구조 생성
            self._create_project_structure(project_path, project_type)
            
            # Panda3D 실행
            main_file = project_path / "main.py"
            self.panda3d_process = subprocess.Popen(
                [sys.executable, str(main_file)],
                cwd=str(project_path)
            )
            
            self.current_project = project_name
            self.is_running = True
            
            # 실행 스레드 시작
            self.execution_thread = threading.Thread(target=self._execution_loop)
            self.execution_thread.daemon = True
            self.execution_thread.start()
            
            logger.info(f"Panda3D 프로젝트 '{project_name}' 시작됨")
            return True
            
        except Exception as e:
            logger.error(f"Panda3D 프로젝트 시작 실패: {e}")
            return False
    
    def _create_project_structure(self, project_path: Path, project_type: str):
        """프로젝트 기본 구조 생성"""
        # main.py 생성
        main_content = self._get_template_main(project_type)
        (project_path / "main.py").write_text(main_content)
        
        # 디렉토리 구조
        directories = ["models", "textures", "sounds", "scripts", "levels", "ui"]
        for dir_name in directories:
            (project_path / dir_name).mkdir(exist_ok=True)
        
        # config.json
        config = {
            "project_name": project_path.name,
            "project_type": project_type,
            "version": "0.1.0",
            "panda3d_version": "1.10.13",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        (project_path / "config.json").write_text(json.dumps(config, indent=2))
    
    def _get_template_main(self, project_type: str) -> str:
        """프로젝트 타입별 템플릿 main.py 반환"""
        templates = {
            "platformer": """
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from direct.task import Task
from direct.actor.Actor import Actor
import sys

class GameApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # 기본 설정
        self.setBackgroundColor(0.1, 0.1, 0.1)
        self.disableMouse()
        
        # 카메라 설정
        self.camera.setPos(0, -20, 5)
        self.camera.lookAt(0, 0, 0)
        
        # 조명 설정
        self.setup_lights()
        
        # 기본 레벨 생성
        self.setup_level()
        
        # 플레이어 생성
        self.setup_player()
        
        # 입력 설정
        self.setup_controls()
        
        # 업데이트 태스크
        self.taskMgr.add(self.update, "update")
        
    def setup_lights(self):
        # 앰비언트 라이트
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
        # 디렉셔널 라이트
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(-45, -45, 0)
        self.render.setLight(dlnp)
    
    def setup_level(self):
        # 바닥 생성
        self.floor = self.loader.loadModel("models/environment")
        if not self.floor:
            # 기본 큐브로 바닥 생성
            self.floor = self.loader.loadModel("models/misc/sphere")
            self.floor.setScale(20, 20, 0.1)
            self.floor.setColor(0.3, 0.3, 0.3)
        self.floor.reparentTo(self.render)
        self.floor.setPos(0, 0, -1)
    
    def setup_player(self):
        # 플레이어 모델
        self.player = self.loader.loadModel("models/misc/sphere")
        self.player.setScale(0.5)
        self.player.setColor(0, 0.5, 1)
        self.player.reparentTo(self.render)
        self.player.setPos(0, 0, 1)
        
        self.player_velocity = Vec3(0, 0, 0)
        self.is_jumping = False
    
    def setup_controls(self):
        self.accept("arrow_left", self.move_left)
        self.accept("arrow_right", self.move_right)
        self.accept("space", self.jump)
        self.accept("escape", sys.exit)
    
    def move_left(self):
        self.player_velocity.x = -5
    
    def move_right(self):
        self.player_velocity.x = 5
    
    def jump(self):
        if not self.is_jumping:
            self.player_velocity.z = 10
            self.is_jumping = True
    
    def update(self, task):
        dt = globalClock.getDt()
        
        # 중력
        if self.player.getZ() > 1:
            self.player_velocity.z -= 20 * dt
        else:
            self.player_velocity.z = 0
            self.is_jumping = False
            self.player.setZ(1)
        
        # 마찰
        self.player_velocity.x *= 0.9
        
        # 위치 업데이트
        self.player.setPos(
            self.player.getX() + self.player_velocity.x * dt,
            self.player.getY() + self.player_velocity.y * dt,
            self.player.getZ() + self.player_velocity.z * dt
        )
        
        return Task.cont

app = GameApp()
app.run()
""",
            "racing": """
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from direct.task import Task
import sys

class RacingGame(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        self.setBackgroundColor(0.5, 0.8, 1)
        self.disableMouse()
        
        # 카메라 설정
        self.camera.setPos(0, -30, 10)
        self.camera.lookAt(0, 0, 0)
        
        # 조명
        self.setup_lights()
        
        # 트랙 생성
        self.setup_track()
        
        # 차량 생성
        self.setup_car()
        
        # 컨트롤
        self.setup_controls()
        
        self.taskMgr.add(self.update, "update")
    
    def setup_lights(self):
        alight = AmbientLight('alight')
        alight.setColor((0.3, 0.3, 0.3, 1))
        self.render.attachNewNode(alight)
        
        sun = DirectionalLight('sun')
        sun.setColor((1, 1, 0.8, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-45, -60, 0)
        self.render.setLight(sun_np)
    
    def setup_track(self):
        # 간단한 트랙
        self.track = self.loader.loadModel("models/misc/sphere")
        self.track.setScale(50, 50, 0.1)
        self.track.setColor(0.2, 0.2, 0.2)
        self.track.reparentTo(self.render)
        self.track.setPos(0, 0, 0)
    
    def setup_car(self):
        self.car = self.loader.loadModel("models/misc/sphere")
        self.car.setScale(1, 2, 0.5)
        self.car.setColor(1, 0, 0)
        self.car.reparentTo(self.render)
        self.car.setPos(0, 0, 0.5)
        
        self.car_speed = 0
        self.car_rotation = 0
    
    def setup_controls(self):
        self.accept("arrow_up", self.accelerate)
        self.accept("arrow_down", self.brake)
        self.accept("arrow_left", self.turn_left)
        self.accept("arrow_right", self.turn_right)
        self.accept("escape", sys.exit)
        
        self.keys = {"up": False, "down": False, "left": False, "right": False}
        
        self.accept("arrow_up-repeat", self.set_key, ["up", True])
        self.accept("arrow_up-up", self.set_key, ["up", False])
        self.accept("arrow_down-repeat", self.set_key, ["down", True])
        self.accept("arrow_down-up", self.set_key, ["down", False])
        self.accept("arrow_left-repeat", self.set_key, ["left", True])
        self.accept("arrow_left-up", self.set_key, ["left", False])
        self.accept("arrow_right-repeat", self.set_key, ["right", True])
        self.accept("arrow_right-up", self.set_key, ["right", False])
    
    def set_key(self, key, value):
        self.keys[key] = value
    
    def accelerate(self):
        self.car_speed = min(self.car_speed + 0.5, 20)
    
    def brake(self):
        self.car_speed = max(self.car_speed - 1, -5)
    
    def turn_left(self):
        if self.car_speed != 0:
            self.car_rotation += 2
    
    def turn_right(self):
        if self.car_speed != 0:
            self.car_rotation -= 2
    
    def update(self, task):
        dt = globalClock.getDt()
        
        # 입력 처리
        if self.keys["up"]:
            self.accelerate()
        if self.keys["down"]:
            self.brake()
        if self.keys["left"]:
            self.turn_left()
        if self.keys["right"]:
            self.turn_right()
        
        # 속도 감속
        self.car_speed *= 0.95
        
        # 회전
        self.car.setH(self.car_rotation)
        
        # 이동
        heading_rad = self.car_rotation * (3.14159 / 180)
        dx = -self.car_speed * dt * np.sin(heading_rad)
        dy = self.car_speed * dt * np.cos(heading_rad)
        
        self.car.setPos(
            self.car.getX() + dx,
            self.car.getY() + dy,
            self.car.getZ()
        )
        
        # 카메라 추적
        self.camera.setPos(
            self.car.getX() - 15 * np.sin(heading_rad),
            self.car.getY() - 15 * np.cos(heading_rad),
            10
        )
        self.camera.lookAt(self.car)
        
        return Task.cont

app = RacingGame()
app.run()
"""
        }
        return templates.get(project_type, templates["platformer"])
    
    def add_action(self, action: AutomationAction):
        """액션을 큐에 추가"""
        self.action_queue.append(action)
        logger.info(f"액션 추가됨: {action.action_type.value} - {action.description}")
    
    def _execution_loop(self):
        """액션 실행 루프"""
        while self.is_running:
            if self.action_queue:
                action = self.action_queue.pop(0)
                self._execute_action(action)
                self.action_history.append(action)
            time.sleep(0.1)
    
    def _execute_action(self, action: AutomationAction) -> bool:
        """단일 액션 실행"""
        try:
            params = action.parameters
            
            # GUI 자동화 액션들은 GUI_AVAILABLE일 때만 실행
            if not GUI_AVAILABLE and action.action_type in [
                ActionType.MOUSE_CLICK, ActionType.MOUSE_DRAG, ActionType.MOUSE_MOVE,
                ActionType.KEY_PRESS, ActionType.KEY_HOLD, ActionType.KEY_RELEASE,
                ActionType.TYPE_TEXT, ActionType.SCREENSHOT
            ]:
                logger.warning(f"GUI 자동화를 사용할 수 없어 액션을 건너뜁니다: {action.description}")
                return True  # 에러로 처리하지 않고 계속 진행
            
            if action.action_type == ActionType.MOUSE_CLICK:
                pyautogui.click(params["x"], params["y"], button=params.get("button", "left"))
                
            elif action.action_type == ActionType.MOUSE_DRAG:
                pyautogui.dragTo(params["x"], params["y"], duration=params.get("duration", 0.5))
                
            elif action.action_type == ActionType.MOUSE_MOVE:
                pyautogui.moveTo(params["x"], params["y"], duration=params.get("duration", 0.2))
                
            elif action.action_type == ActionType.KEY_PRESS:
                pyautogui.press(params["key"])
                
            elif action.action_type == ActionType.KEY_HOLD:
                pyautogui.keyDown(params["key"])
                
            elif action.action_type == ActionType.KEY_RELEASE:
                pyautogui.keyUp(params["key"])
                
            elif action.action_type == ActionType.TYPE_TEXT:
                pyautogui.typewrite(params["text"], interval=params.get("interval", 0.05))
                
            elif action.action_type == ActionType.WAIT:
                time.sleep(params["duration"])
                
            elif action.action_type == ActionType.SCREENSHOT:
                screenshot = pyautogui.screenshot()
                screenshot.save(params["filename"])
                
            elif action.action_type == ActionType.CODE_WRITE:
                self._write_code(params["file_path"], params["code"])
                
            elif action.action_type == ActionType.FILE_OPERATION:
                self._file_operation(params["operation"], params["path"], params.get("content"))
                
            logger.info(f"액션 실행 완료: {action.description}")
            return True
            
        except Exception as e:
            logger.error(f"액션 실행 실패: {action.description} - {e}")
            return False
    
    def _write_code(self, file_path: str, code: str):
        """코드 파일 작성"""
        path = Path(self.get_project_path()) / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
    
    def _file_operation(self, operation: str, path: str, content: Optional[str] = None):
        """파일 작업 수행"""
        full_path = Path(self.get_project_path()) / path
        
        if operation == "create":
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.touch()
            if content:
                full_path.write_text(content)
        elif operation == "delete":
            if full_path.exists():
                full_path.unlink()
        elif operation == "mkdir":
            full_path.mkdir(parents=True, exist_ok=True)
    
    def get_project_path(self) -> str:
        """현재 프로젝트 경로 반환"""
        if self.current_project:
            return f"game_projects/{self.current_project}"
        return ""
    
    def analyze_screen(self) -> Dict[str, Any]:
        """현재 화면 분석"""
        analysis = {
            "timestamp": time.time(),
            "screen_size": (1920, 1080),  # 기본값
            "detected_elements": [],
            "game_state": "unknown",
            "suggestions": []
        }
        
        if GUI_AVAILABLE:
            try:
                screenshot = pyautogui.screenshot()
                analysis["screen_size"] = screenshot.size
            except Exception as e:
                logger.warning(f"스크린샷 캡처 실패: {e}")
        
        # AI 모델이 있다면 화면 분석 요청
        if self.ai_controller:
            # AI에게 스크린샷 분석 요청
            pass
        
        return analysis
    
    def create_game_element(self, element_type: str, properties: Dict[str, Any]):
        """게임 요소 생성을 위한 액션 시퀀스 생성"""
        actions = []
        
        if element_type == "model":
            # 모델 생성 코드 작성
            code = self._generate_model_code(properties)
            actions.append(AutomationAction(
                ActionType.CODE_WRITE,
                {"file_path": f"scripts/{properties['name']}.py", "code": code},
                f"Create {properties['name']} model script"
            ))
            
        elif element_type == "level":
            # 레벨 생성 코드
            code = self._generate_level_code(properties)
            actions.append(AutomationAction(
                ActionType.CODE_WRITE,
                {"file_path": f"levels/{properties['name']}.py", "code": code},
                f"Create {properties['name']} level"
            ))
            
        # 액션들을 큐에 추가
        for action in actions:
            self.add_action(action)
    
    def _generate_model_code(self, properties: Dict[str, Any]) -> str:
        """모델 생성 코드 생성"""
        template = f"""
from panda3d.core import *

class {properties['name']}:
    def __init__(self, parent):
        self.model = loader.loadModel("models/misc/sphere")
        self.model.setScale({properties.get('scale', 1)})
        self.model.setColor({properties.get('color', '(1, 1, 1)')})
        self.model.reparentTo(parent)
        self.model.setPos({properties.get('position', '(0, 0, 0)')})
"""
        return template
    
    def _generate_level_code(self, properties: Dict[str, Any]) -> str:
        """레벨 생성 코드 생성"""
        template = f"""
from panda3d.core import *

class {properties['name']}Level:
    def __init__(self, parent):
        self.root = parent.attachNewNode("{properties['name']}")
        
        # 레벨 요소들 생성
        self.setup_terrain()
        self.setup_objects()
        self.setup_lighting()
    
    def setup_terrain(self):
        # 지형 생성
        pass
    
    def setup_objects(self):
        # 오브젝트 배치
        pass
    
    def setup_lighting(self):
        # 조명 설정
        pass
"""
        return template
    
    def stop(self):
        """자동화 중지"""
        self.is_running = False
        if self.panda3d_process:
            self.panda3d_process.terminate()
        logger.info("Panda3D 자동화 컨트롤러 중지됨")


# 테스트 및 예제 사용
if __name__ == "__main__":
    controller = Panda3DAutomationController()
    
    # 새 프로젝트 시작
    controller.start_panda3d_project("test_platformer", "platformer")
    
    # 몇 가지 액션 추가
    controller.add_action(AutomationAction(
        ActionType.WAIT,
        {"duration": 2},
        "Wait for Panda3D to load"
    ))
    
    controller.add_action(AutomationAction(
        ActionType.SCREENSHOT,
        {"filename": "game_screenshots/initial_state.png"},
        "Take initial screenshot"
    ))
    
    # 게임 요소 생성
    controller.create_game_element("model", {
        "name": "Enemy",
        "scale": 0.8,
        "color": "(1, 0, 0)",
        "position": "(5, 0, 1)"
    })
    
    # 실행
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        controller.stop()