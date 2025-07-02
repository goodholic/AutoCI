#!/usr/bin/env python3
"""
게임 제작 오류 복구 시스템
게임 제작 중 발생하는 다양한 오류를 감지하고 자동으로 복구합니다.
"""

import os
import sys
import json
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re

class GameErrorRecovery:
    """게임 제작 오류 복구 시스템"""
    
    def __init__(self):
        self.error_log = []
        self.recovery_attempts = {}
        self.common_errors = {
            "missing_file": self._fix_missing_file,
            "invalid_scene": self._fix_invalid_scene,
            "script_error": self._fix_script_error,
            "project_config": self._fix_project_config,
            "resource_path": self._fix_resource_path,
            "node_reference": self._fix_node_reference
        }
    
    async def check_and_fix_project(self, project_path: Path) -> Dict[str, Any]:
        """프로젝트 전체 검사 및 자동 수정"""
        print("\n🔧 프로젝트 오류 검사 및 자동 복구 시작...")
        
        results = {
            "success": True,
            "errors_found": 0,
            "errors_fixed": 0,
            "unfixed_errors": [],
            "details": []
        }
        
        # 1. 프로젝트 구조 검사
        structure_check = await self._check_project_structure(project_path)
        if not structure_check["valid"]:
            fixed = await self._fix_project_structure(project_path, structure_check["missing"])
            results["errors_found"] += len(structure_check["missing"])
            results["errors_fixed"] += fixed
            results["details"].append(f"프로젝트 구조: {fixed}/{len(structure_check['missing'])} 수정됨")
        
        # 2. project.godot 검사
        godot_check = await self._check_project_godot(project_path)
        if not godot_check["valid"]:
            if await self._fix_project_godot(project_path, godot_check["issues"]):
                results["errors_fixed"] += 1
                results["details"].append("project.godot 수정됨")
            else:
                results["unfixed_errors"].append("project.godot 수정 실패")
            results["errors_found"] += 1
        
        # 3. 씬 파일 검사
        scene_check = await self._check_scene_files(project_path)
        if scene_check["errors"]:
            fixed_scenes = 0
            for scene_error in scene_check["errors"]:
                if await self._fix_scene_file(project_path, scene_error):
                    fixed_scenes += 1
            results["errors_found"] += len(scene_check["errors"])
            results["errors_fixed"] += fixed_scenes
            results["details"].append(f"씬 파일: {fixed_scenes}/{len(scene_check['errors'])} 수정됨")
        
        # 4. 스크립트 검사
        script_check = await self._check_scripts(project_path)
        if script_check["errors"]:
            fixed_scripts = 0
            for script_error in script_check["errors"]:
                if await self._fix_script_file(project_path, script_error):
                    fixed_scripts += 1
            results["errors_found"] += len(script_check["errors"])
            results["errors_fixed"] += fixed_scripts
            results["details"].append(f"스크립트: {fixed_scripts}/{len(script_check['errors'])} 수정됨")
        
        # 5. 리소스 경로 검사
        resource_check = await self._check_resource_paths(project_path)
        if resource_check["broken_paths"]:
            fixed_paths = 0
            for broken_path in resource_check["broken_paths"]:
                if await self._fix_resource_path(project_path, broken_path):
                    fixed_paths += 1
            results["errors_found"] += len(resource_check["broken_paths"])
            results["errors_fixed"] += fixed_paths
            results["details"].append(f"리소스 경로: {fixed_paths}/{len(resource_check['broken_paths'])} 수정됨")
        
        # 결과 요약
        if results["errors_found"] == 0:
            print("✅ 오류가 발견되지 않았습니다!")
        else:
            print(f"\n📊 오류 복구 결과:")
            print(f"  - 발견된 오류: {results['errors_found']}개")
            print(f"  - 수정된 오류: {results['errors_fixed']}개")
            print(f"  - 미해결 오류: {len(results['unfixed_errors'])}개")
            
            for detail in results["details"]:
                print(f"  - {detail}")
        
        results["success"] = len(results["unfixed_errors"]) == 0
        return results
    
    async def _check_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """프로젝트 구조 검사"""
        required_dirs = ["scenes", "scripts", "assets"]
        required_files = ["project.godot", "icon.svg"]
        
        missing = []
        
        for dir_name in required_dirs:
            if not (project_path / dir_name).exists():
                missing.append(("dir", dir_name))
        
        for file_name in required_files:
            if not (project_path / file_name).exists():
                missing.append(("file", file_name))
        
        return {
            "valid": len(missing) == 0,
            "missing": missing
        }
    
    async def _fix_project_structure(self, project_path: Path, missing: List[Tuple[str, str]]) -> int:
        """프로젝트 구조 수정"""
        fixed = 0
        
        for item_type, item_name in missing:
            try:
                if item_type == "dir":
                    (project_path / item_name).mkdir(parents=True, exist_ok=True)
                    print(f"  ✅ 디렉토리 생성: {item_name}")
                    fixed += 1
                elif item_type == "file" and item_name == "icon.svg":
                    # 기본 아이콘 생성
                    icon_content = """<svg height="128" width="128" xmlns="http://www.w3.org/2000/svg">
<rect x="0" y="0" width="128" height="128" fill="#1a5fb4"/>
<text x="64" y="75" font-family="Arial" font-size="48" fill="white" text-anchor="middle">G</text>
</svg>"""
                    with open(project_path / "icon.svg", "w") as f:
                        f.write(icon_content)
                    print(f"  ✅ 아이콘 생성: {item_name}")
                    fixed += 1
            except Exception as e:
                print(f"  ❌ {item_name} 생성 실패: {e}")
        
        return fixed
    
    async def _check_project_godot(self, project_path: Path) -> Dict[str, Any]:
        """project.godot 파일 검사"""
        godot_file = project_path / "project.godot"
        
        if not godot_file.exists():
            return {"valid": False, "issues": ["파일이 없음"]}
        
        issues = []
        
        try:
            content = godot_file.read_text(encoding='utf-8')
            
            # 필수 섹션 확인
            required_sections = ["[application]", "[display]", "[rendering]"]
            for section in required_sections:
                if section not in content:
                    issues.append(f"{section} 섹션 누락")
            
            # 메인 씬 확인
            if 'run/main_scene=' not in content:
                issues.append("메인 씬 설정 누락")
            
            # 버전 확인
            if 'config/features=' not in content:
                issues.append("Godot 버전 설정 누락")
            
        except Exception as e:
            issues.append(f"파일 읽기 오류: {e}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    async def _fix_project_godot(self, project_path: Path, issues: List[str]) -> bool:
        """project.godot 파일 수정"""
        try:
            # 기본 project.godot 생성
            default_content = """[application]

config/name="Fixed Game"
run/main_scene="res://scenes/Main.tscn"
config/features=PackedStringArray("4.3")
config/icon="res://icon.svg"

[display]

window/size/viewport_width=1024
window/size/viewport_height=600

[rendering]

renderer/rendering_method="mobile"
"""
            
            with open(project_path / "project.godot", "w", encoding="utf-8") as f:
                f.write(default_content)
            
            print("  ✅ project.godot 파일 재생성됨")
            return True
            
        except Exception as e:
            print(f"  ❌ project.godot 수정 실패: {e}")
            return False
    
    async def _check_scene_files(self, project_path: Path) -> Dict[str, Any]:
        """씬 파일 검사"""
        scene_dir = project_path / "scenes"
        errors = []
        
        if not scene_dir.exists():
            return {"errors": ["scenes 디렉토리가 없음"]}
        
        # 메인 씬 확인
        main_scene = scene_dir / "Main.tscn"
        if not main_scene.exists():
            errors.append({"file": "Main.tscn", "error": "메인 씬이 없음"})
        else:
            # 씬 파일 유효성 검사
            try:
                content = main_scene.read_text()
                if not content.startswith("[gd_scene"):
                    errors.append({"file": "Main.tscn", "error": "잘못된 씬 파일 형식"})
            except:
                errors.append({"file": "Main.tscn", "error": "씬 파일 읽기 실패"})
        
        return {"errors": errors}
    
    async def _fix_scene_file(self, project_path: Path, scene_error: Dict[str, str]) -> bool:
        """씬 파일 수정"""
        try:
            scene_path = project_path / "scenes" / scene_error["file"]
            
            if "없음" in scene_error["error"]:
                # 기본 씬 생성
                basic_scene = """[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://scripts/Main.gd" id="1"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

[node name="Label" type="Label" parent="."]
offset_right = 200.0
offset_bottom = 50.0
text = "Game Running!"
"""
                with open(scene_path, "w") as f:
                    f.write(basic_scene)
                
                # 스크립트도 생성
                script_path = project_path / "scripts" / "Main.gd"
                if not script_path.exists():
                    script_path.parent.mkdir(exist_ok=True)
                    with open(script_path, "w") as f:
                        f.write("""extends Node2D

func _ready():
    print("Game started!")
""")
                
                print(f"  ✅ {scene_error['file']} 생성됨")
                return True
            
        except Exception as e:
            print(f"  ❌ {scene_error['file']} 수정 실패: {e}")
            
        return False
    
    async def _check_scripts(self, project_path: Path) -> Dict[str, Any]:
        """스크립트 파일 검사"""
        script_dir = project_path / "scripts"
        errors = []
        
        if script_dir.exists():
            for script_file in script_dir.glob("*.gd"):
                try:
                    content = script_file.read_text(encoding='utf-8')
                    
                    # 기본 문법 검사
                    if not content.strip():
                        errors.append({"file": script_file.name, "error": "빈 스크립트"})
                    elif not any(content.startswith(x) for x in ["extends", "class_name", "#"]):
                        errors.append({"file": script_file.name, "error": "잘못된 스크립트 시작"})
                    
                except Exception as e:
                    errors.append({"file": script_file.name, "error": f"읽기 오류: {e}"})
        
        return {"errors": errors}
    
    async def _fix_script_file(self, project_path: Path, script_error: Dict[str, str]) -> bool:
        """스크립트 파일 수정"""
        try:
            script_path = project_path / "scripts" / script_error["file"]
            
            if "빈 스크립트" in script_error["error"]:
                # 기본 스크립트 생성
                with open(script_path, "w") as f:
                    f.write("""extends Node

func _ready():
    pass

func _process(delta):
    pass
""")
                print(f"  ✅ {script_error['file']} 수정됨")
                return True
                
        except Exception as e:
            print(f"  ❌ {script_error['file']} 수정 실패: {e}")
            
        return False
    
    async def _check_resource_paths(self, project_path: Path) -> Dict[str, Any]:
        """리소스 경로 검사"""
        broken_paths = []
        
        # 모든 씬과 스크립트 파일에서 리소스 경로 확인
        for file_path in project_path.rglob("*.tscn"):
            try:
                content = file_path.read_text()
                # res:// 경로 추출
                paths = re.findall(r'path="(res://[^"]+)"', content)
                for res_path in paths:
                    # res://를 실제 경로로 변환
                    actual_path = project_path / res_path.replace("res://", "")
                    if not actual_path.exists():
                        broken_paths.append({
                            "file": str(file_path.relative_to(project_path)),
                            "broken_path": res_path,
                            "type": "resource"
                        })
            except:
                pass
        
        return {"broken_paths": broken_paths}
    
    async def _fix_resource_path(self, project_path: Path, broken_path_info: Dict[str, str]) -> bool:
        """리소스 경로 수정"""
        try:
            # 깨진 경로가 스크립트인 경우 빈 스크립트 생성
            if broken_path_info["broken_path"].endswith(".gd"):
                script_path = project_path / broken_path_info["broken_path"].replace("res://", "")
                script_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(script_path, "w") as f:
                    f.write("""extends Node

func _ready():
    pass
""")
                print(f"  ✅ 누락된 스크립트 생성: {broken_path_info['broken_path']}")
                return True
                
        except Exception as e:
            print(f"  ❌ 경로 수정 실패: {e}")
            
        return False
    
    def _fix_missing_file(self, project_path: Path, file_path: str) -> bool:
        """누락된 파일 생성"""
        # 구현...
        pass
    
    def _fix_invalid_scene(self, project_path: Path, scene_path: str) -> bool:
        """잘못된 씬 파일 수정"""
        # 구현...
        pass
    
    def _fix_script_error(self, project_path: Path, script_path: str, error: str) -> bool:
        """스크립트 오류 수정"""
        # 구현...
        pass
    
    def _fix_project_config(self, project_path: Path, config_error: str) -> bool:
        """프로젝트 설정 오류 수정"""
        # 구현...
        pass
    
    def _fix_node_reference(self, project_path: Path, node_error: str) -> bool:
        """노드 참조 오류 수정"""
        # 구현...
        pass

# 싱글톤 인스턴스
_recovery_instance = None

def get_error_recovery():
    """오류 복구 시스템 인스턴스 반환"""
    global _recovery_instance
    if _recovery_instance is None:
        _recovery_instance = GameErrorRecovery()
    return _recovery_instance