"""
Godot Engine Controller for AutoCI
Godot Editor를 제어하는 모듈
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class GodotController:
    """
    Godot 컨트롤러 - Godot Engine과 상호작용
    """
    
    def __init__(self, godot_path: str = "./godot_engine/godot", 
                 projects_dir: str = "./godot_projects"):
        self.godot_path = godot_path
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(exist_ok=True)
        logger.info(f"GodotController 초기화 - Godot: {godot_path}")
        
    async def create_project(self, name: str, path: str = None) -> bool:
        """새 Godot 프로젝트 생성"""
        if not path:
            path = self.projects_dir / name
        
        project_path = Path(path)
        project_path.mkdir(parents=True, exist_ok=True)
        
        # project.godot 파일 생성
        project_file = project_path / "project.godot"
        project_content = f"""[application]
config/name="{name}"
config/features=PackedStringArray("4.2", "GL Compatibility")

[rendering]
renderer/rendering_method="gl_compatibility"
"""
        project_file.write_text(project_content)
        
        logger.info(f"프로젝트 생성됨: {name} at {project_path}")
        return True
        
    def add_script(self, project_path: str, script_name: str, code: str) -> Dict:
        """프로젝트에 스크립트 추가"""
        try:
            script_path = Path(project_path) / f"{script_name}.gd"
            script_path.write_text(code)
            return {"success": True, "path": str(script_path)}
        except Exception as e:
            logger.error(f"스크립트 추가 실패: {e}")
            return {"success": False, "error": str(e)}
            
    def create_scene(self, project_path: str, scene_name: str) -> Dict:
        """씬 파일 생성"""
        try:
            scene_path = Path(project_path) / f"{scene_name}.tscn"
            scene_content = """[gd_scene format=3]

[node name="Root" type="Node2D"]
"""
            scene_path.write_text(scene_content)
            return {"success": True, "path": str(scene_path)}
        except Exception as e:
            logger.error(f"씬 생성 실패: {e}")
            return {"success": False, "error": str(e)}