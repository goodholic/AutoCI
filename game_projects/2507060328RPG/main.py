
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *

class 2507060328RPGGame(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.setBackgroundColor(0.1, 0.1, 0.1, 1)
        
        # 기본 카메라 설정
        self.camera.setPos(0, -20, 10)
        self.camera.lookAt(0, 0, 0)
        
        # 기본 조명
        dlight = DirectionalLight('dlight')
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(-45, -45, 0)
        self.render.setLight(dlnp)
        
        # 게임 상태
        self.is_running = True
        self.score = 0
        
        # 게임 업데이트 태스크
        self.taskMgr.add(self.update, "update")
        
    def update(self, task):
        # 게임 로직 업데이트
        return task.cont

if __name__ == "__main__":
    game = 2507060328RPGGame()
    game.run()
