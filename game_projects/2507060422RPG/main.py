#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-generated game main file"""

from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import sys

class 2507060422RPGGame(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.setBackgroundColor(0.1, 0.1, 0.1, 1)
        
        # 기본 카메라 설정
        self.camera.setPos(0, -20, 10)
        self.camera.lookAt(0, 0, 0)
        
        # 기본 조명
        self._setup_lighting()
        
        # 게임 초기화
        self._init_game()
        
    def _setup_lighting(self):
        """Setup basic lighting"""
        dlight = DirectionalLight('main_light')
        dlight_node = self.render.attachNewNode(dlight)
        dlight_node.setHpr(-45, -45, 0)
        self.render.setLight(dlight_node)
        
    def _init_game(self):
        """Initialize game components"""
        pass

if __name__ == "__main__":
    app = 2507060422RPGGame()
    app.run()
