
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
