"""
Panda3D 엔진 통합 모듈
AI가 Panda3D 엔진을 직접 조작하여 게임을 제작하는 핵심 모듈
"""

import os
import sys
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

# Panda3D imports
try:
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task
    from panda3d.core import (
        NodePath, Vec3, Point3, VBase4,
        DirectionalLight, AmbientLight, PointLight,
        CollisionNode, CollisionSphere, CollisionBox,
        CollisionTraverser, CollisionHandlerQueue,
        TextNode, CardMaker, Texture,
        PandaNode, ModelNode, GeomNode
    )
    from direct.gui.DirectGui import (
        DirectFrame, DirectButton, DirectLabel,
        DirectEntry, DirectScrolledList, DirectCheckButton
    )
    from direct.interval.IntervalGlobal import Sequence, Parallel, Func, Wait
    from direct.actor.Actor import Actor
    PANDA3D_AVAILABLE = True
except ImportError:
    PANDA3D_AVAILABLE = False
    logger.warning("Panda3D not available. Install with: pip install panda3d")

logger = logging.getLogger(__name__)


class Panda3DEngineIntegration:
    """Panda3D 엔진 통합 - AI가 직접 조작 가능한 인터페이스"""
    
    def __init__(self, project_path: str):
        """
        초기화
        
        Args:
            project_path: 게임 프로젝트 경로
        """
        self.project_path = Path(project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        self.app = None
        self.is_running = False
        self.game_objects = {}
        self.scenes = {}
        self.current_scene = None
        
        # 게임 설정
        self.config = {
            "window_title": "AutoCI Panda3D Game",
            "window_size": (1280, 720),
            "fullscreen": False,
            "show_fps": True,
            "background_color": (0.1, 0.1, 0.1, 1.0)
        }
        
        # 물리 설정
        self.physics_enabled = False
        self.collision_traverser = None
        
        logger.info(f"Panda3D 엔진 통합 초기화: {project_path}")
    
    def create_game_structure(self) -> Dict[str, str]:
        """게임 프로젝트 구조 생성"""
        structure = {
            "main.py": self._generate_main_file(),
            "config.json": json.dumps(self.config, indent=2),
            "assets/models/": "",
            "assets/textures/": "",
            "assets/sounds/": "",
            "assets/fonts/": "",
            "scripts/player.py": self._generate_player_script(),
            "scripts/enemy.py": self._generate_enemy_script(),
            "scripts/level.py": self._generate_level_script(),
            "scripts/ui.py": self._generate_ui_script()
        }
        
        # 파일 생성
        for file_path, content in structure.items():
            full_path = self.project_path / file_path
            if file_path.endswith('/'):
                full_path.mkdir(parents=True, exist_ok=True)
            else:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                if content:
                    full_path.write_text(content)
        
        logger.info("게임 프로젝트 구조 생성 완료")
        return {"status": "success", "path": str(self.project_path)}
    
    def _generate_main_file(self) -> str:
        """메인 게임 파일 생성"""
        return '''#!/usr/bin/env python3
"""
AutoCI Generated Panda3D Game
Generated at: {timestamp}
"""

from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import sys
sys.path.append('scripts')

from player import Player
from level import Level
from ui import GameUI


class Game(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # 기본 설정
        self.setBackgroundColor(0.1, 0.1, 0.1)
        self.disableMouse()
        
        # 카메라 설정
        self.camera.setPos(0, -20, 10)
        self.camera.lookAt(0, 0, 0)
        
        # 조명 설정
        self.setup_lights()
        
        # 게임 요소 초기화
        self.player = Player(self)
        self.level = Level(self)
        self.ui = GameUI(self)
        
        # 입력 설정
        self.setup_input()
        
        # 게임 루프
        self.taskMgr.add(self.update, "update")
    
    def setup_lights(self):
        """조명 설정"""
        # 방향성 조명
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.7, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(-45, -45, 0)
        self.render.setLight(dlnp)
        
        # 주변광
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
    
    def setup_input(self):
        """입력 설정"""
        self.accept("escape", sys.exit)
        self.accept("f1", self.toggle_debug)
    
    def toggle_debug(self):
        """디버그 모드 토글"""
        if self.render.isHidden():
            self.render.show()
        else:
            self.render.hide()
    
    def update(self, task):
        """게임 업데이트"""
        dt = globalClock.getDt()
        
        # 플레이어 업데이트
        self.player.update(dt)
        
        # 레벨 업데이트
        self.level.update(dt)
        
        return task.cont


if __name__ == "__main__":
    game = Game()
    game.run()
'''.format(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    
    def _generate_player_script(self) -> str:
        """플레이어 스크립트 생성"""
        return '''"""
Player Controller
"""

from panda3d.core import *
from direct.actor.Actor import Actor


class Player:
    def __init__(self, base):
        self.base = base
        
        # 플레이어 모델 (임시 큐브)
        self.model = self.base.loader.loadModel("models/environment")
        if not self.model:
            # 기본 큐브 생성
            self.model = self.create_cube()
        
        self.model.setScale(1, 1, 1)
        self.model.setPos(0, 0, 0)
        self.model.reparentTo(self.base.render)
        
        # 플레이어 속성
        self.speed = 10.0
        self.jump_speed = 15.0
        self.velocity = Vec3(0, 0, 0)
        self.is_jumping = False
        
        # 입력 설정
        self.setup_input()
        
        # 키 상태
        self.key_map = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "jump": False
        }
    
    def create_cube(self):
        """기본 큐브 생성"""
        from panda3d.core import CardMaker
        cm = CardMaker("player")
        cm.setFrame(-0.5, 0.5, -0.5, 0.5)
        
        node = NodePath("player")
        
        # 6면 생성
        for i in range(6):
            face = node.attachNewNode(cm.generate())
            if i == 0: face.setY(0.5)
            elif i == 1: face.setY(-0.5)
            elif i == 2: face.setX(0.5)
            elif i == 3: face.setX(-0.5)
            elif i == 4: face.setZ(0.5)
            elif i == 5: face.setZ(-0.5)
            
            if i < 2:
                face.setH(180 if i == 0 else 0)
            elif i < 4:
                face.setH(90 if i == 2 else -90)
            else:
                face.setP(90 if i == 4 else -90)
        
        node.setColor(0.2, 0.5, 1.0, 1.0)
        return node
    
    def setup_input(self):
        """입력 설정"""
        # 키보드 입력
        self.base.accept("w", self.set_key, ["forward", True])
        self.base.accept("w-up", self.set_key, ["forward", False])
        self.base.accept("s", self.set_key, ["backward", True])
        self.base.accept("s-up", self.set_key, ["backward", False])
        self.base.accept("a", self.set_key, ["left", True])
        self.base.accept("a-up", self.set_key, ["left", False])
        self.base.accept("d", self.set_key, ["right", True])
        self.base.accept("d-up", self.set_key, ["right", False])
        self.base.accept("space", self.set_key, ["jump", True])
        self.base.accept("space-up", self.set_key, ["jump", False])
    
    def set_key(self, key, value):
        """키 상태 설정"""
        self.key_map[key] = value
    
    def update(self, dt):
        """플레이어 업데이트"""
        # 이동 처리
        move_vec = Vec3(0, 0, 0)
        
        if self.key_map["forward"]:
            move_vec.y += 1
        if self.key_map["backward"]:
            move_vec.y -= 1
        if self.key_map["left"]:
            move_vec.x -= 1
        if self.key_map["right"]:
            move_vec.x += 1
        
        # 정규화 및 속도 적용
        if move_vec.length() > 0:
            move_vec.normalize()
            move_vec *= self.speed * dt
            
            # 위치 업데이트
            new_pos = self.model.getPos() + move_vec
            self.model.setPos(new_pos)
            
            # 카메라 추적
            cam_pos = new_pos + Vec3(0, -20, 10)
            self.base.camera.setPos(cam_pos)
            self.base.camera.lookAt(self.model)
        
        # 점프 처리
        if self.key_map["jump"] and not self.is_jumping:
            self.velocity.z = self.jump_speed
            self.is_jumping = True
        
        # 중력
        if self.is_jumping:
            self.velocity.z -= 30 * dt  # 중력 가속도
            new_z = self.model.getZ() + self.velocity.z * dt
            
            # 바닥 체크
            if new_z <= 0:
                new_z = 0
                self.velocity.z = 0
                self.is_jumping = False
            
            self.model.setZ(new_z)
'''
    
    def _generate_enemy_script(self) -> str:
        """적 스크립트 생성"""
        return '''"""
Enemy AI Controller
"""

from panda3d.core import *
import random


class Enemy:
    def __init__(self, base, pos):
        self.base = base
        
        # 적 모델 (임시 큐브)
        self.model = self.create_cube()
        self.model.setPos(pos)
        self.model.setColor(1.0, 0.2, 0.2, 1.0)
        self.model.reparentTo(self.base.render)
        
        # AI 속성
        self.speed = 5.0
        self.detection_range = 10.0
        self.attack_range = 2.0
        self.health = 100
        
        # AI 상태
        self.state = "idle"  # idle, patrol, chase, attack
        self.target = None
        self.patrol_points = []
        self.current_patrol = 0
    
    def create_cube(self):
        """기본 큐브 생성"""
        from panda3d.core import CardMaker
        cm = CardMaker("enemy")
        cm.setFrame(-0.5, 0.5, -0.5, 0.5)
        
        node = NodePath("enemy")
        
        # 6면 생성
        for i in range(6):
            face = node.attachNewNode(cm.generate())
            if i == 0: face.setY(0.5)
            elif i == 1: face.setY(-0.5)
            elif i == 2: face.setX(0.5)
            elif i == 3: face.setX(-0.5)
            elif i == 4: face.setZ(0.5)
            elif i == 5: face.setZ(-0.5)
            
            if i < 2:
                face.setH(180 if i == 0 else 0)
            elif i < 4:
                face.setH(90 if i == 2 else -90)
            else:
                face.setP(90 if i == 4 else -90)
        
        return node
    
    def update(self, dt, player_pos):
        """적 AI 업데이트"""
        # 플레이어와의 거리 계산
        distance = (player_pos - self.model.getPos()).length()
        
        # 상태 전환
        if distance < self.attack_range:
            self.state = "attack"
        elif distance < self.detection_range:
            self.state = "chase"
            self.target = player_pos
        else:
            self.state = "patrol"
        
        # 상태별 행동
        if self.state == "chase" and self.target:
            # 플레이어 추적
            direction = self.target - self.model.getPos()
            if direction.length() > 0:
                direction.normalize()
                move = direction * self.speed * dt
                self.model.setPos(self.model.getPos() + move)
                
                # 방향 전환
                self.model.lookAt(self.target)
        
        elif self.state == "patrol":
            # 순찰
            if len(self.patrol_points) > 0:
                target = self.patrol_points[self.current_patrol]
                direction = target - self.model.getPos()
                
                if direction.length() < 1.0:
                    # 다음 순찰 지점
                    self.current_patrol = (self.current_patrol + 1) % len(self.patrol_points)
                else:
                    direction.normalize()
                    move = direction * self.speed * 0.5 * dt
                    self.model.setPos(self.model.getPos() + move)
        
        elif self.state == "attack":
            # 공격 애니메이션 (간단한 흔들림)
            shake = random.uniform(-0.1, 0.1)
            self.model.setX(self.model.getX() + shake)
'''
    
    def _generate_level_script(self) -> str:
        """레벨 스크립트 생성"""
        return '''"""
Level Manager
"""

from panda3d.core import *
from enemy import Enemy
import random


class Level:
    def __init__(self, base):
        self.base = base
        
        # 레벨 요소들
        self.platforms = []
        self.enemies = []
        self.collectibles = []
        
        # 레벨 생성
        self.create_level()
    
    def create_level(self):
        """레벨 생성"""
        # 바닥 플랫폼
        ground = self.create_platform(Vec3(0, 0, -1), Vec3(50, 50, 1))
        ground.setColor(0.3, 0.3, 0.3, 1.0)
        
        # 플랫폼들
        platform_positions = [
            (Vec3(5, 5, 2), Vec3(3, 3, 0.5)),
            (Vec3(-5, 5, 4), Vec3(3, 3, 0.5)),
            (Vec3(0, 10, 6), Vec3(5, 3, 0.5)),
            (Vec3(10, 0, 3), Vec3(3, 3, 0.5)),
            (Vec3(-10, -5, 5), Vec3(4, 4, 0.5))
        ]
        
        for pos, scale in platform_positions:
            platform = self.create_platform(pos, scale)
            platform.setColor(0.5, 0.5, 0.5, 1.0)
        
        # 적 생성
        enemy_positions = [
            Vec3(10, 10, 0),
            Vec3(-10, 10, 0),
            Vec3(0, 15, 0),
            Vec3(15, -5, 0)
        ]
        
        for pos in enemy_positions:
            enemy = Enemy(self.base, pos)
            enemy.patrol_points = [
                pos,
                pos + Vec3(5, 0, 0),
                pos + Vec3(5, 5, 0),
                pos + Vec3(0, 5, 0)
            ]
            self.enemies.append(enemy)
        
        # 수집품 생성
        for i in range(10):
            collectible = self.create_collectible(
                Vec3(
                    random.uniform(-20, 20),
                    random.uniform(-20, 20),
                    random.uniform(1, 8)
                )
            )
            self.collectibles.append(collectible)
    
    def create_platform(self, pos, scale):
        """플랫폼 생성"""
        from panda3d.core import CardMaker
        cm = CardMaker("platform")
        cm.setFrame(-0.5, 0.5, -0.5, 0.5)
        
        platform = NodePath("platform")
        
        # 상단면
        top = platform.attachNewNode(cm.generate())
        top.setP(-90)
        top.setScale(scale.x, scale.y, 1)
        
        platform.setPos(pos)
        platform.reparentTo(self.base.render)
        self.platforms.append(platform)
        
        return platform
    
    def create_collectible(self, pos):
        """수집품 생성"""
        collectible = self.base.loader.loadModel("models/environment")
        if not collectible:
            # 기본 구체 생성
            from panda3d.core import CardMaker
            collectible = NodePath("collectible")
            
            cm = CardMaker("collectible")
            cm.setFrame(-0.3, 0.3, -0.3, 0.3)
            
            for i in range(8):
                face = collectible.attachNewNode(cm.generate())
                face.setH(i * 45)
        
        collectible.setScale(0.5, 0.5, 0.5)
        collectible.setPos(pos)
        collectible.setColor(1.0, 1.0, 0.0, 1.0)
        collectible.reparentTo(self.base.render)
        
        # 회전 애니메이션
        from direct.interval.IntervalGlobal import LerpHprInterval
        rotation = LerpHprInterval(collectible, 2.0, Vec3(360, 0, 0))
        rotation.loop()
        
        return collectible
    
    def update(self, dt):
        """레벨 업데이트"""
        # 적 업데이트
        if hasattr(self.base, 'player'):
            player_pos = self.base.player.model.getPos()
            for enemy in self.enemies:
                enemy.update(dt, player_pos)
        
        # 수집품 체크
        if hasattr(self.base, 'player'):
            player_pos = self.base.player.model.getPos()
            for collectible in self.collectibles[:]:
                distance = (collectible.getPos() - player_pos).length()
                if distance < 1.0:
                    # 수집
                    collectible.removeNode()
                    self.collectibles.remove(collectible)
                    
                    # 점수 증가 (UI에 알림)
                    if hasattr(self.base, 'ui'):
                        self.base.ui.add_score(10)
'''
    
    def _generate_ui_script(self) -> str:
        """UI 스크립트 생성"""
        return '''"""
Game UI Manager
"""

from direct.gui.DirectGui import *
from panda3d.core import *


class GameUI:
    def __init__(self, base):
        self.base = base
        
        # UI 요소들
        self.score = 0
        self.health = 100
        
        # UI 생성
        self.create_ui()
    
    def create_ui(self):
        """UI 요소 생성"""
        # 점수 표시
        self.score_label = DirectLabel(
            text=f"Score: {self.score}",
            scale=0.07,
            pos=(-1.2, 0, 0.9),
            text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0.5),
            text_fg=(1, 1, 1, 1)
        )
        
        # 체력 표시
        self.health_label = DirectLabel(
            text=f"Health: {self.health}",
            scale=0.07,
            pos=(-1.2, 0, 0.8),
            text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0.5),
            text_fg=(1, 0.2, 0.2, 1)
        )
        
        # 게임 제목
        self.title_label = DirectLabel(
            text="AutoCI Panda3D Game",
            scale=0.1,
            pos=(0, 0, 0.9),
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            text_fg=(1, 1, 1, 1)
        )
        
        # 컨트롤 안내
        self.help_text = DirectLabel(
            text="WASD: Move | Space: Jump | ESC: Exit",
            scale=0.05,
            pos=(0, 0, -0.9),
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0.3),
            text_fg=(0.8, 0.8, 0.8, 1)
        )
        
        # 게임 오버 화면 (숨김)
        self.game_over_screen = DirectFrame(
            frameSize=(-2, 2, -2, 2),
            frameColor=(0, 0, 0, 0.8)
        )
        self.game_over_screen.hide()
        
        self.game_over_text = DirectLabel(
            text="GAME OVER",
            scale=0.2,
            pos=(0, 0, 0),
            parent=self.game_over_screen,
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            text_fg=(1, 0.2, 0.2, 1)
        )
        
        self.restart_button = DirectButton(
            text="Restart",
            scale=0.1,
            pos=(0, 0, -0.3),
            parent=self.game_over_screen,
            command=self.restart_game
        )
    
    def add_score(self, points):
        """점수 추가"""
        self.score += points
        self.score_label["text"] = f"Score: {self.score}"
    
    def set_health(self, health):
        """체력 설정"""
        self.health = max(0, min(100, health))
        self.health_label["text"] = f"Health: {self.health}"
        
        if self.health <= 0:
            self.show_game_over()
    
    def show_game_over(self):
        """게임 오버 화면 표시"""
        self.game_over_screen.show()
        
        # 게임 일시정지
        if hasattr(self.base, 'taskMgr'):
            self.base.taskMgr.remove("update")
    
    def restart_game(self):
        """게임 재시작"""
        # 게임 리셋 로직
        self.score = 0
        self.health = 100
        self.score_label["text"] = f"Score: {self.score}"
        self.health_label["text"] = f"Health: {self.health}"
        self.game_over_screen.hide()
        
        # 게임 업데이트 재개
        if hasattr(self.base, 'taskMgr'):
            self.base.taskMgr.add(self.base.update, "update")
'''
    
    def start_engine(self, config: Optional[Dict[str, Any]] = None):
        """Panda3D 엔진 시작"""
        if not PANDA3D_AVAILABLE:
            logger.error("Panda3D not installed!")
            return False
        
        if config:
            self.config.update(config)
        
        # 별도 스레드에서 실행
        def run_panda():
            self.app = ShowBase()
            self.setup_engine()
            self.app.run()
        
        self.engine_thread = threading.Thread(target=run_panda)
        self.engine_thread.daemon = True
        self.engine_thread.start()
        
        self.is_running = True
        logger.info("Panda3D 엔진 시작됨")
        return True
    
    def setup_engine(self):
        """엔진 기본 설정"""
        if not self.app:
            return
        
        # 윈도우 설정
        props = WindowProperties()
        props.setTitle(self.config["window_title"])
        props.setSize(*self.config["window_size"])
        props.setFullscreen(self.config["fullscreen"])
        self.app.win.requestProperties(props)
        
        # 배경색
        self.app.setBackgroundColor(*self.config["background_color"])
        
        # FPS 표시
        if self.config["show_fps"]:
            self.app.setFrameRateMeter(True)
        
        # 기본 카메라 설정
        self.app.disableMouse()
        self.app.camera.setPos(0, -20, 10)
        self.app.camera.lookAt(0, 0, 0)
        
        # 기본 조명
        self.setup_default_lighting()
    
    def setup_default_lighting(self):
        """기본 조명 설정"""
        if not self.app:
            return
        
        # 방향성 조명
        dlight = DirectionalLight('default_dlight')
        dlight.setColor((0.8, 0.8, 0.7, 1))
        dlnp = self.app.render.attachNewNode(dlight)
        dlnp.setHpr(-45, -45, 0)
        self.app.render.setLight(dlnp)
        
        # 주변광
        alight = AmbientLight('default_alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.app.render.attachNewNode(alight)
        self.app.render.setLight(alnp)
    
    def create_object(self, obj_type: str, name: str, **kwargs) -> Optional[NodePath]:
        """게임 오브젝트 생성"""
        if not self.app:
            return None
        
        obj = None
        
        if obj_type == "cube":
            obj = self._create_cube(name, **kwargs)
        elif obj_type == "sphere":
            obj = self._create_sphere(name, **kwargs)
        elif obj_type == "plane":
            obj = self._create_plane(name, **kwargs)
        elif obj_type == "model":
            obj = self._load_model(name, **kwargs)
        elif obj_type == "actor":
            obj = self._create_actor(name, **kwargs)
        
        if obj:
            self.game_objects[name] = obj
            logger.info(f"오브젝트 생성: {name} ({obj_type})")
        
        return obj
    
    def _create_cube(self, name: str, size: float = 1.0, 
                     color: Tuple[float, float, float, float] = (1, 1, 1, 1)) -> NodePath:
        """큐브 생성"""
        cm = CardMaker(name)
        cm.setFrame(-size/2, size/2, -size/2, size/2)
        
        cube = NodePath(name)
        
        # 6면 생성
        faces = [
            (Vec3(0, size/2, 0), Vec3(0, 0, 0)),      # 앞
            (Vec3(0, -size/2, 0), Vec3(0, 180, 0)),  # 뒤
            (Vec3(size/2, 0, 0), Vec3(0, 90, 0)),    # 오른쪽
            (Vec3(-size/2, 0, 0), Vec3(0, -90, 0)),  # 왼쪽
            (Vec3(0, 0, size/2), Vec3(-90, 0, 0)),   # 위
            (Vec3(0, 0, -size/2), Vec3(90, 0, 0))    # 아래
        ]
        
        for pos, rot in faces:
            face = cube.attachNewNode(cm.generate())
            face.setPos(pos)
            face.setHpr(rot)
        
        cube.setColor(*color)
        cube.reparentTo(self.app.render)
        
        return cube
    
    def _create_sphere(self, name: str, radius: float = 1.0,
                      color: Tuple[float, float, float, float] = (1, 1, 1, 1)) -> NodePath:
        """구체 생성 (간단한 다면체)"""
        # 실제 구체 모델이 없으므로 8면체로 대체
        sphere = NodePath(name)
        
        # 정점 정의
        vertices = [
            Vec3(0, 0, radius),     # 위
            Vec3(0, 0, -radius),    # 아래
            Vec3(radius, 0, 0),     # 오른쪽
            Vec3(-radius, 0, 0),    # 왼쪽
            Vec3(0, radius, 0),     # 앞
            Vec3(0, -radius, 0)     # 뒤
        ]
        
        # 면 생성 (삼각형)
        cm = CardMaker(f"{name}_face")
        cm.setFrame(-0.5, 0.5, -0.5, 0.5)
        
        sphere.setColor(*color)
        sphere.reparentTo(self.app.render)
        
        return sphere
    
    def _create_plane(self, name: str, width: float = 10.0, height: float = 10.0,
                     color: Tuple[float, float, float, float] = (1, 1, 1, 1)) -> NodePath:
        """평면 생성"""
        cm = CardMaker(name)
        cm.setFrame(-width/2, width/2, -height/2, height/2)
        
        plane = self.app.render.attachNewNode(cm.generate())
        plane.setP(-90)  # 수평으로 회전
        plane.setColor(*color)
        
        return plane
    
    def _load_model(self, name: str, model_path: str, **kwargs) -> Optional[NodePath]:
        """모델 로드"""
        try:
            model = self.app.loader.loadModel(model_path)
            if model:
                model.reparentTo(self.app.render)
                
                # 추가 설정 적용
                if "pos" in kwargs:
                    model.setPos(*kwargs["pos"])
                if "scale" in kwargs:
                    model.setScale(*kwargs["scale"])
                if "color" in kwargs:
                    model.setColor(*kwargs["color"])
                
                return model
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return None
    
    def _create_actor(self, name: str, model_path: str, 
                     anims: Optional[Dict[str, str]] = None, **kwargs) -> Optional[Actor]:
        """애니메이션 액터 생성"""
        try:
            actor = Actor(model_path, anims)
            actor.reparentTo(self.app.render)
            
            # 추가 설정 적용
            if "pos" in kwargs:
                actor.setPos(*kwargs["pos"])
            if "scale" in kwargs:
                actor.setScale(*kwargs["scale"])
            
            return actor
        except Exception as e:
            logger.error(f"액터 생성 실패: {e}")
            return None
    
    def move_object(self, name: str, position: Tuple[float, float, float]):
        """오브젝트 이동"""
        if name in self.game_objects:
            self.game_objects[name].setPos(*position)
    
    def rotate_object(self, name: str, rotation: Tuple[float, float, float]):
        """오브젝트 회전"""
        if name in self.game_objects:
            self.game_objects[name].setHpr(*rotation)
    
    def scale_object(self, name: str, scale: Union[float, Tuple[float, float, float]]):
        """오브젝트 크기 조정"""
        if name in self.game_objects:
            if isinstance(scale, (int, float)):
                self.game_objects[name].setScale(scale)
            else:
                self.game_objects[name].setScale(*scale)
    
    def set_object_color(self, name: str, color: Tuple[float, float, float, float]):
        """오브젝트 색상 설정"""
        if name in self.game_objects:
            self.game_objects[name].setColor(*color)
    
    def create_animation(self, obj_name: str, anim_type: str, **kwargs):
        """애니메이션 생성"""
        if obj_name not in self.game_objects:
            return
        
        obj = self.game_objects[obj_name]
        
        if anim_type == "rotate":
            # 회전 애니메이션
            from direct.interval.IntervalGlobal import LerpHprInterval
            duration = kwargs.get("duration", 2.0)
            hpr = kwargs.get("hpr", Vec3(360, 0, 0))
            
            interval = LerpHprInterval(obj, duration, hpr)
            if kwargs.get("loop", False):
                interval.loop()
            else:
                interval.start()
        
        elif anim_type == "move":
            # 이동 애니메이션
            from direct.interval.IntervalGlobal import LerpPosInterval
            duration = kwargs.get("duration", 2.0)
            pos = kwargs.get("pos", Vec3(0, 0, 0))
            
            interval = LerpPosInterval(obj, duration, pos)
            if kwargs.get("loop", False):
                interval.loop()
            else:
                interval.start()
        
        elif anim_type == "scale":
            # 크기 애니메이션
            from direct.interval.IntervalGlobal import LerpScaleInterval
            duration = kwargs.get("duration", 2.0)
            scale = kwargs.get("scale", 2.0)
            
            interval = LerpScaleInterval(obj, duration, scale)
            if kwargs.get("loop", False):
                interval.loop()
            else:
                interval.start()
    
    def setup_collision(self, obj_name: str, shape: str = "sphere", **kwargs):
        """충돌 감지 설정"""
        if obj_name not in self.game_objects:
            return
        
        obj = self.game_objects[obj_name]
        
        # 충돌 노드 생성
        cnode = CollisionNode(f"{obj_name}_collision")
        
        if shape == "sphere":
            radius = kwargs.get("radius", 1.0)
            cnode.addSolid(CollisionSphere(0, 0, 0, radius))
        elif shape == "box":
            min_point = kwargs.get("min", Point3(-1, -1, -1))
            max_point = kwargs.get("max", Point3(1, 1, 1))
            cnode.addSolid(CollisionBox(min_point, max_point))
        
        cnodepath = obj.attachNewNode(cnode)
        
        # 충돌 감지 활성화
        if not self.collision_traverser:
            self.collision_traverser = CollisionTraverser()
            self.app.taskMgr.add(self._collision_task, "collision_task")
        
        # 충돌 핸들러 설정
        handler = CollisionHandlerQueue()
        self.collision_traverser.addCollider(cnodepath, handler)
    
    def _collision_task(self, task):
        """충돌 감지 태스크"""
        if self.collision_traverser:
            self.collision_traverser.traverse(self.app.render)
        return task.cont
    
    def create_ui_element(self, ui_type: str, **kwargs):
        """UI 요소 생성"""
        if not self.app:
            return None
        
        if ui_type == "label":
            return DirectLabel(**kwargs)
        elif ui_type == "button":
            return DirectButton(**kwargs)
        elif ui_type == "entry":
            return DirectEntry(**kwargs)
        elif ui_type == "frame":
            return DirectFrame(**kwargs)
    
    def take_screenshot(self, filename: str = None):
        """스크린샷 촬영"""
        if not self.app:
            return
        
        if not filename:
            filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        
        self.app.screenshot(filename)
        logger.info(f"스크린샷 저장: {filename}")
    
    def stop_engine(self):
        """엔진 중지"""
        self.is_running = False
        if self.app:
            self.app.finalizeExit()
        logger.info("Panda3D 엔진 중지")


# 테스트 코드
if __name__ == "__main__":
    # 엔진 테스트
    engine = Panda3DEngineIntegration("test_game")
    
    # 게임 구조 생성
    engine.create_game_structure()
    
    # 엔진 시작
    if engine.start_engine():
        print("Panda3D 엔진이 시작되었습니다!")
        
        # 테스트 오브젝트 생성
        time.sleep(2)  # 엔진 초기화 대기
        
        cube = engine.create_object("cube", "test_cube", size=2.0, color=(1, 0, 0, 1))
        if cube:
            engine.move_object("test_cube", (0, 0, 2))
            engine.create_animation("test_cube", "rotate", duration=4.0, loop=True)
        
        # 계속 실행
        try:
            while engine.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            engine.stop_engine()