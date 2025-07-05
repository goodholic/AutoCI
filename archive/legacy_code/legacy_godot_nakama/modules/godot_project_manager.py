#!/usr/bin/env python3
"""
Godot 프로젝트 관리자
Git 프로젝트를 스캔하여 Godot 프로젝트를 찾고 선택/생성
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import json
import asyncio

class GodotProjectManager:
    """Godot 프로젝트 관리자"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.game_projects_dir = self.project_root / "game_projects"
        self.game_projects_dir.mkdir(exist_ok=True)
        
    def find_godot_projects(self, search_paths: List[Path] = None) -> List[Dict[str, str]]:
        """Godot 프로젝트 찾기"""
        if search_paths is None:
            # 기본 검색 경로들
            search_paths = [
                self.game_projects_dir,
                Path.home(),
                Path.cwd(),
                Path("/mnt/d"),  # WSL에서 D 드라이브
                Path("/mnt/c"),  # WSL에서 C 드라이브
            ]
        
        godot_projects = []
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            # project.godot 파일 찾기
            try:
                for project_file in search_path.rglob("project.godot"):
                    # 너무 깊은 경로는 건너뛰기
                    if len(project_file.parts) - len(search_path.parts) > 5:
                        continue
                        
                    project_dir = project_file.parent
                    
                    # Git 프로젝트인지 확인
                    is_git = (project_dir / ".git").exists()
                    
                    # 프로젝트 정보 읽기
                    project_info = self._read_project_info(project_file)
                    
                    godot_projects.append({
                        "name": project_info.get("name", project_dir.name),
                        "path": str(project_dir),
                        "project_file": str(project_file),
                        "is_git": is_git,
                        "description": project_info.get("description", ""),
                        "last_modified": project_file.stat().st_mtime
                    })
                    
            except PermissionError:
                continue
            except Exception as e:
                print(f"검색 중 오류 {search_path}: {e}")
                
        # 최근 수정 순으로 정렬
        godot_projects.sort(key=lambda x: x["last_modified"], reverse=True)
        
        return godot_projects[:20]  # 최대 20개만 반환
        
    def _read_project_info(self, project_file: Path) -> Dict[str, str]:
        """project.godot 파일에서 정보 읽기"""
        info = {}
        try:
            with open(project_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('config/name='):
                        info["name"] = line.split('=', 1)[1].strip().strip('"')
                    elif line.startswith('config/description='):
                        info["description"] = line.split('=', 1)[1].strip().strip('"')
                    elif line.startswith('config/version='):
                        info["version"] = line.split('=', 1)[1].strip().strip('"')
        except:
            pass
        return info
        
    def create_new_godot_project(self, name: str, project_type: str = "2d") -> Tuple[bool, str]:
        """새 Godot 프로젝트 생성"""
        # 프로젝트 이름 정리
        safe_name = name.replace(" ", "_").lower()
        project_dir = self.game_projects_dir / f"{safe_name}_{project_type}"
        
        if project_dir.exists():
            # 이미 존재하면 번호 추가
            i = 1
            while (self.game_projects_dir / f"{safe_name}_{project_type}_{i}").exists():
                i += 1
            project_dir = self.game_projects_dir / f"{safe_name}_{project_type}_{i}"
            
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # project.godot 파일 생성
        project_config = f"""
; Engine configuration file.
; It's best edited using the editor UI and not directly,
; since the properties are organized in sections here.

[application]

config/name="{name}"
config/description="AI가 자동으로 생성한 {project_type} 게임 프로젝트"
run/main_scene="res://Main.tscn"
config/features=PackedStringArray("4.3", "GL Compatibility")
config/icon="res://icon.svg"

[display]

window/size/viewport_width=1280
window/size/viewport_height=720

[rendering]

renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"
"""
        
        project_file = project_dir / "project.godot"
        project_file.write_text(project_config.strip())
        
        # 기본 씬 생성
        main_scene = """[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://Main.gd" id="1"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

[node name="Label" type="Label" parent="."]
offset_left = 500.0
offset_top = 300.0
offset_right = 780.0
offset_bottom = 340.0
text = "AutoCI AI 게임 개발 시작!"
theme_override_font_sizes/font_size = 24
"""
        
        (project_dir / "Main.tscn").write_text(main_scene)
        
        # 기본 스크립트 생성
        main_script = """extends Node2D

func _ready():
    print("AutoCI AI 게임 개발 프로젝트가 시작되었습니다!")
    print("이제 AI가 자동으로 게임을 개발합니다.")
    
func _process(delta):
    # AI가 여기에 게임 로직을 추가합니다
    pass
"""
        
        (project_dir / "Main.gd").write_text(main_script)
        
        # 아이콘 생성
        icon_svg = """<svg height="128" width="128" xmlns="http://www.w3.org/2000/svg">
<rect x="10" y="10" width="108" height="108" rx="20" fill="#4a90e2"/>
<text x="64" y="75" font-size="48" text-anchor="middle" fill="white">AI</text>
</svg>"""
        
        (project_dir / "icon.svg").write_text(icon_svg)
        
        # Git 초기화
        try:
            subprocess.run(["git", "init"], cwd=project_dir, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=project_dir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial commit by AutoCI"], cwd=project_dir, capture_output=True)
        except:
            pass
            
        return True, str(project_dir)
        
    async def select_or_create_project(self) -> Optional[str]:
        """프로젝트 선택 또는 생성 대화형 인터페이스"""
        print("\n🎮 Godot 프로젝트 선택")
        print("=" * 60)
        
        # 기존 프로젝트 찾기
        print("🔍 Godot 프로젝트를 검색 중...")
        projects = self.find_godot_projects()
        
        if projects:
            print(f"\n📁 {len(projects)}개의 Godot 프로젝트를 찾았습니다:\n")
            
            for i, project in enumerate(projects, 1):
                git_status = "🔗 Git" if project["is_git"] else "📁"
                print(f"{i}. {git_status} {project['name']}")
                print(f"   경로: {project['path']}")
                if project['description']:
                    print(f"   설명: {project['description']}")
                print()
                
            print(f"{len(projects) + 1}. 🆕 새 프로젝트 생성")
            print("0. ❌ 취소 (Godot 없이 진행)")
            
            while True:
                try:
                    choice = input("\n선택하세요 (0-{}): ".format(len(projects) + 1))
                    choice = int(choice)
                    
                    if choice == 0:
                        return None
                    elif 1 <= choice <= len(projects):
                        selected = projects[choice - 1]
                        print(f"\n✅ '{selected['name']}' 프로젝트를 선택했습니다.")
                        return selected['path']
                    elif choice == len(projects) + 1:
                        break
                    else:
                        print("❌ 잘못된 선택입니다.")
                except ValueError:
                    print("❌ 숫자를 입력해주세요.")
        else:
            print("\n📭 Godot 프로젝트를 찾을 수 없습니다.")
            
        # 새 프로젝트 생성
        print("\n🆕 새 Godot 프로젝트 생성")
        print("-" * 40)
        
        name = input("프로젝트 이름: ").strip()
        if not name:
            name = "AutoCI_Game"
            
        print("\n프로젝트 타입:")
        print("1. 2D 게임")
        print("2. 3D 게임")
        
        project_type = "2d"
        try:
            type_choice = int(input("선택 (1-2) [기본: 1]: ") or "1")
            if type_choice == 2:
                project_type = "3d"
        except:
            pass
            
        print(f"\n🔨 '{name}' {project_type.upper()} 프로젝트를 생성 중...")
        success, project_path = self.create_new_godot_project(name, project_type)
        
        if success:
            print(f"✅ 프로젝트가 생성되었습니다: {project_path}")
            return project_path
        else:
            print("❌ 프로젝트 생성에 실패했습니다.")
            return None
            
    def get_windows_path(self, wsl_path: str) -> str:
        """WSL 경로를 Windows 경로로 변환"""
        path = Path(wsl_path)
        path_str = str(path)
        
        if path_str.startswith("/mnt/"):
            # /mnt/c/... -> C:\...
            drive = path_str[5]
            rest = path_str[7:]
            return f"{drive.upper()}:\\{rest.replace('/', '\\')}"
        else:
            # WSL 내부 경로는 \\wsl$\Ubuntu\... 형태로 변환
            return f"\\\\wsl$\\Ubuntu{path_str.replace('/', '\\')}"

# 테스트
async def test_manager():
    """프로젝트 매니저 테스트"""
    manager = GodotProjectManager()
    
    # 프로젝트 선택 또는 생성
    project_path = await manager.select_or_create_project()
    
    if project_path:
        print(f"\n선택된 프로젝트: {project_path}")
        win_path = manager.get_windows_path(project_path)
        print(f"Windows 경로: {win_path}")
    else:
        print("\n프로젝트가 선택되지 않았습니다.")

if __name__ == "__main__":
    asyncio.run(test_manager())