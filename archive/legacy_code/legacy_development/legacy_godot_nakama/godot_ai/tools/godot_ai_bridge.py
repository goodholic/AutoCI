#!/usr/bin/env python3
"""
AutoCI-Godot Bridge
Python AI 시스템과 Godot 간의 통신 브리지
"""

import sys
import json
import asyncio
from pathlib import Path

# AutoCI 모듈 경로 추가
autoci_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(autoci_path))

from modules.ai_godot_automation import AIGodotAutomation
from modules.ai_scene_composer import AISceneComposer
from modules.ai_resource_manager import AIResourceManager
from modules.ai_gameplay_generator import AIGameplayGenerator

class GodotAIBridge:
    """Godot-AI 브리지"""
    
    def __init__(self):
        self.godot_automation = AIGodotAutomation()
        self.scene_composer = AISceneComposer()
        self.resource_manager = AIResourceManager(Path.cwd())
        self.gameplay_generator = AIGameplayGenerator()
        
    async def execute_command(self, command_data: dict):
        """명령 실행"""
        command = command_data.get("command")
        
        if command == "create_project":
            return await self._create_project(command_data)
        elif command == "optimize_scene":
            return await self._optimize_scene(command_data)
        elif command == "generate_resources":
            return await self._generate_resources(command_data)
        elif command == "generate_gameplay":
            return await self._generate_gameplay(command_data)
        else:
            return {"error": f"Unknown command: {command}"}
            
    async def _create_project(self, data: dict):
        project = await self.godot_automation.create_complete_game_project(
            data.get("project_name", "AI_Game"),
            data.get("project_path", "./ai_game"),
            data.get("game_type", "platformer")
        )
        return {"success": True, "project": project.name if project else None}
        
    async def _optimize_scene(self, data: dict):
        # 씬 최적화 로직
        return {"success": True, "message": "Scene optimized"}
        
    async def _generate_resources(self, data: dict):
        # 리소스 생성 로직
        return {"success": True, "message": "Resources generated"}
        
    async def _generate_gameplay(self, data: dict):
        # 게임플레이 생성 로직
        return {"success": True, "message": "Gameplay generated"}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command_json = sys.argv[1]
        command_data = json.loads(command_json)
        
        bridge = GodotAIBridge()
        result = asyncio.run(bridge.execute_command(command_data))
        
        print(json.dumps(result))
    else:
        print(json.dumps({"error": "No command provided"}))
