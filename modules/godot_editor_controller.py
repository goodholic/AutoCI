#!/usr/bin/env python3
"""
Godot Editor 직접 제어 모듈
AI가 Godot Editor를 직접 조작하여 씬을 편집하고 오브젝트를 배치
"""

import os
import json
import subprocess
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

class GodotEditorController:
    def __init__(self, godot_path: str = None):
        self.godot_path = godot_path or self._find_godot()
        self.project_path = None
        self.current_scene = None
        self.editor_process = None
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _find_godot(self) -> str:
        """Godot 실행 파일 찾기"""
        possible_paths = [
            "/usr/local/bin/godot",
            "/usr/bin/godot",
            "~/godot/godot",
            "./godot",
            "godot"
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path) or subprocess.run(["which", path], capture_output=True).returncode == 0:
                return expanded_path
                
        raise FileNotFoundError("Godot을 찾을 수 없습니다. Godot을 설치하거나 경로를 지정하세요.")
    
    async def create_project(self, project_name: str, project_path: str) -> bool:
        """새 Godot 프로젝트 생성"""
        try:
            project_dir = Path(project_path) / project_name
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # project.godot 파일 생성
            project_config = {
                "application": {
                    "config/name": project_name,
                    "run/main_scene": "res://scenes/Main.tscn",
                    "config/icon": "res://icon.png"
                },
                "display": {
                    "window/size/width": 1280,
                    "window/size/height": 720
                },
                "rendering": {
                    "environment/default_environment": "res://default_env.tres"
                }
            }
            
            # project.godot 작성
            with open(project_dir / "project.godot", 'w') as f:
                f.write("; Engine configuration file.\n")
                f.write("; It's best edited using the editor UI and not directly,\n")
                f.write("; since the properties are organized in sections here.\n\n")
                
                for section, properties in project_config.items():
                    f.write(f"[{section}]\n\n")
                    for key, value in properties.items():
                        if isinstance(value, str):
                            f.write(f'{key}="{value}"\n')
                        else:
                            f.write(f'{key}={value}\n')
                    f.write("\n")
            
            # 기본 디렉토리 구조 생성
            for dir_name in ["scenes", "scripts", "assets", "resources"]:
                (project_dir / dir_name).mkdir(exist_ok=True)
            
            self.project_path = str(project_dir)
            self.logger.info(f"프로젝트 생성 완료: {self.project_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"프로젝트 생성 실패: {e}")
            return False
    
    async def open_project(self, project_path: str) -> bool:
        """Godot 프로젝트 열기 (헤드리스 모드)"""
        try:
            self.project_path = project_path
            
            # Godot을 헤드리스 모드로 실행 (GUI 없이)
            cmd = [self.godot_path, "--headless", "--path", project_path]
            
            self.logger.info(f"Godot 프로젝트 열기: {project_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"프로젝트 열기 실패: {e}")
            return False
    
    async def create_scene(self, scene_name: str, scene_type: str = "2D") -> Dict:
        """새 씬 생성"""
        scene_data = {
            "name": scene_name,
            "type": scene_type,
            "nodes": []
        }
        
        # 루트 노드 생성
        if scene_type == "2D":
            root_node = self.create_node("Node2D", scene_name)
        else:
            root_node = self.create_node("Node3D", scene_name)
            
        scene_data["nodes"].append(root_node)
        scene_data["root_id"] = root_node["id"]
        
        self.current_scene = scene_data
        
        # .tscn 파일로 저장
        await self.save_scene(f"scenes/{scene_name}.tscn")
        
        return scene_data
    
    def create_node(self, node_type: str, node_name: str, parent_id: Optional[str] = None) -> Dict:
        """노드 생성"""
        import uuid
        
        node = {
            "id": str(uuid.uuid4()),
            "type": node_type,
            "name": node_name,
            "parent": parent_id,
            "transform": {
                "position": {"x": 0, "y": 0},
                "rotation": 0,
                "scale": {"x": 1, "y": 1}
            },
            "properties": {},
            "children": []
        }
        
        # 노드 타입별 기본 속성 설정
        if node_type == "Sprite2D":
            node["properties"]["texture"] = None
            node["properties"]["centered"] = True
        elif node_type == "CollisionShape2D":
            node["properties"]["shape"] = None
        elif node_type == "RigidBody2D":
            node["properties"]["mass"] = 1.0
            node["properties"]["gravity_scale"] = 1.0
        elif node_type == "Area2D":
            node["properties"]["monitoring"] = True
        elif node_type == "Label":
            node["properties"]["text"] = "Label"
            node["properties"]["font_size"] = 16
        elif node_type == "Button":
            node["properties"]["text"] = "Button"
            
        return node
    
    async def add_node_to_scene(self, node_type: str, node_name: str, 
                                parent_path: Optional[str] = None,
                                position: Optional[Tuple[float, float]] = None) -> Dict:
        """씬에 노드 추가"""
        if not self.current_scene:
            raise ValueError("현재 열린 씬이 없습니다")
        
        # 부모 노드 찾기
        parent_id = None
        if parent_path:
            parent_node = self._find_node_by_path(parent_path)
            if parent_node:
                parent_id = parent_node["id"]
        else:
            parent_id = self.current_scene["root_id"]
        
        # 노드 생성
        new_node = self.create_node(node_type, node_name, parent_id)
        
        # 위치 설정
        if position:
            new_node["transform"]["position"] = {"x": position[0], "y": position[1]}
        
        # 씬에 추가
        self.current_scene["nodes"].append(new_node)
        
        # 부모 노드의 children 업데이트
        if parent_id:
            parent_node = self._find_node_by_id(parent_id)
            if parent_node:
                parent_node["children"].append(new_node["id"])
        
        self.logger.info(f"노드 추가: {node_name} ({node_type}) at {position}")
        return new_node
    
    async def create_sprite(self, name: str, texture_path: str, 
                           position: Tuple[float, float] = (0, 0),
                           scale: Tuple[float, float] = (1, 1)) -> Dict:
        """스프라이트 생성"""
        sprite = await self.add_node_to_scene("Sprite2D", name, position=position)
        sprite["properties"]["texture"] = texture_path
        sprite["transform"]["scale"] = {"x": scale[0], "y": scale[1]}
        return sprite
    
    async def create_collision_body(self, name: str, body_type: str = "RigidBody2D",
                                   position: Tuple[float, float] = (0, 0),
                                   shape_type: str = "RectangleShape2D") -> Dict:
        """충돌 바디 생성"""
        # 바디 생성
        body = await self.add_node_to_scene(body_type, name, position=position)
        
        # 충돌 모양 추가
        collision_shape = await self.add_node_to_scene(
            "CollisionShape2D", 
            f"{name}_CollisionShape",
            parent_path=name
        )
        
        # 모양 설정
        collision_shape["properties"]["shape"] = {
            "type": shape_type,
            "size": {"x": 64, "y": 64} if shape_type == "RectangleShape2D" else {"radius": 32}
        }
        
        return body
    
    async def create_player_character(self, position: Tuple[float, float] = (640, 360)) -> Dict:
        """플레이어 캐릭터 생성"""
        # CharacterBody2D 생성
        player = await self.add_node_to_scene("CharacterBody2D", "Player", position=position)
        
        # 스프라이트 추가
        sprite = await self.add_node_to_scene("Sprite2D", "Sprite", parent_path="Player")
        sprite["properties"]["texture"] = "res://assets/player.png"
        
        # 충돌 모양 추가
        collision = await self.add_node_to_scene("CollisionShape2D", "CollisionShape", parent_path="Player")
        collision["properties"]["shape"] = {
            "type": "CapsuleShape2D",
            "radius": 16,
            "height": 32
        }
        
        # 스크립트 연결
        player["properties"]["script"] = "res://scripts/Player.gd"
        
        return player
    
    async def create_enemy(self, enemy_type: str, position: Tuple[float, float]) -> Dict:
        """적 캐릭터 생성"""
        enemy_name = f"Enemy_{enemy_type}_{int(time.time())}"
        
        # 적 타입에 따른 설정
        if enemy_type == "walker":
            enemy = await self.add_node_to_scene("CharacterBody2D", enemy_name, position=position)
        elif enemy_type == "flyer":
            enemy = await self.add_node_to_scene("Area2D", enemy_name, position=position)
        else:
            enemy = await self.add_node_to_scene("RigidBody2D", enemy_name, position=position)
        
        # 스프라이트 추가
        sprite = await self.add_node_to_scene("Sprite2D", "Sprite", parent_path=enemy_name)
        sprite["properties"]["texture"] = f"res://assets/enemy_{enemy_type}.png"
        
        # 충돌 추가
        collision = await self.add_node_to_scene("CollisionShape2D", "CollisionShape", parent_path=enemy_name)
        
        # 스크립트 연결
        enemy["properties"]["script"] = f"res://scripts/Enemy_{enemy_type}.gd"
        
        return enemy
    
    async def create_platform(self, position: Tuple[float, float], 
                             size: Tuple[float, float] = (200, 20)) -> Dict:
        """플랫폼 생성"""
        platform_name = f"Platform_{int(position[0])}_{int(position[1])}"
        
        # StaticBody2D로 플랫폼 생성
        platform = await self.add_node_to_scene("StaticBody2D", platform_name, position=position)
        
        # 시각적 표현
        sprite = await self.add_node_to_scene("NinePatchRect", "Visual", parent_path=platform_name)
        sprite["properties"]["size"] = {"x": size[0], "y": size[1]}
        sprite["properties"]["texture"] = "res://assets/platform.png"
        
        # 충돌 모양
        collision = await self.add_node_to_scene("CollisionShape2D", "CollisionShape", parent_path=platform_name)
        collision["properties"]["shape"] = {
            "type": "RectangleShape2D",
            "size": {"x": size[0], "y": size[1]}
        }
        
        return platform
    
    async def create_collectible(self, item_type: str, position: Tuple[float, float]) -> Dict:
        """수집 아이템 생성"""
        item_name = f"{item_type}_{int(time.time())}"
        
        # Area2D로 아이템 생성
        item = await self.add_node_to_scene("Area2D", item_name, position=position)
        
        # 스프라이트
        sprite = await self.add_node_to_scene("Sprite2D", "Sprite", parent_path=item_name)
        sprite["properties"]["texture"] = f"res://assets/{item_type}.png"
        
        # 충돌 영역
        collision = await self.add_node_to_scene("CollisionShape2D", "CollisionShape", parent_path=item_name)
        collision["properties"]["shape"] = {
            "type": "CircleShape2D",
            "radius": 16
        }
        
        # 스크립트
        item["properties"]["script"] = "res://scripts/Collectible.gd"
        item["properties"]["item_type"] = item_type
        
        return item
    
    async def create_level_layout(self, level_data: Dict) -> None:
        """레벨 레이아웃 생성"""
        # 플랫폼 생성
        if "platforms" in level_data:
            for platform in level_data["platforms"]:
                await self.create_platform(
                    position=(platform["x"], platform["y"]),
                    size=(platform.get("width", 200), platform.get("height", 20))
                )
        
        # 적 배치
        if "enemies" in level_data:
            for enemy in level_data["enemies"]:
                await self.create_enemy(
                    enemy_type=enemy["type"],
                    position=(enemy["x"], enemy["y"])
                )
        
        # 아이템 배치
        if "items" in level_data:
            for item in level_data["items"]:
                await self.create_collectible(
                    item_type=item["type"],
                    position=(item["x"], item["y"])
                )
        
        # 플레이어 시작 위치
        if "player_start" in level_data:
            await self.create_player_character(
                position=(level_data["player_start"]["x"], level_data["player_start"]["y"])
            )
    
    async def create_ui_element(self, ui_type: str, name: str, 
                               position: Tuple[float, float],
                               size: Optional[Tuple[float, float]] = None) -> Dict:
        """UI 요소 생성"""
        # UI 컨테이너가 없으면 생성
        ui_root = self._find_node_by_path("UI")
        if not ui_root:
            ui_root = await self.add_node_to_scene("CanvasLayer", "UI")
        
        # UI 요소 생성
        ui_element = await self.add_node_to_scene(ui_type, name, parent_path="UI", position=position)
        
        if size:
            ui_element["properties"]["size"] = {"x": size[0], "y": size[1]}
        
        # UI 타입별 기본 설정
        if ui_type == "Label":
            ui_element["properties"]["text"] = name
        elif ui_type == "Button":
            ui_element["properties"]["text"] = name
        elif ui_type == "ProgressBar":
            ui_element["properties"]["value"] = 100
            ui_element["properties"]["max_value"] = 100
        
        return ui_element
    
    async def setup_game_ui(self) -> None:
        """게임 UI 설정"""
        # 체력바
        await self.create_ui_element("ProgressBar", "HealthBar", position=(20, 20), size=(200, 20))
        
        # 점수 표시
        await self.create_ui_element("Label", "ScoreLabel", position=(1100, 20))
        
        # 게임 오버 메시지
        game_over = await self.create_ui_element("Label", "GameOverLabel", position=(640, 360))
        game_over["properties"]["visible"] = False
        game_over["properties"]["text"] = "Game Over"
        game_over["properties"]["font_size"] = 48
    
    async def save_scene(self, file_path: str) -> bool:
        """씬을 .tscn 파일로 저장"""
        if not self.current_scene or not self.project_path:
            return False
        
        try:
            scene_path = Path(self.project_path) / file_path
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            
            # .tscn 형식으로 변환
            tscn_content = self._convert_to_tscn_format()
            
            with open(scene_path, 'w') as f:
                f.write(tscn_content)
            
            self.logger.info(f"씬 저장 완료: {scene_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"씬 저장 실패: {e}")
            return False
    
    def _convert_to_tscn_format(self) -> str:
        """씬 데이터를 .tscn 형식으로 변환"""
        lines = []
        lines.append('[gd_scene load_steps=2 format=3]\n')
        
        # 노드 변환
        for i, node in enumerate(self.current_scene["nodes"]):
            node_str = f'[node name="{node["name"]}" type="{node["type"]}"'
            
            if node["parent"]:
                parent_node = self._find_node_by_id(node["parent"])
                if parent_node:
                    node_str += f' parent="."'
                    
            lines.append(node_str + ']')
            
            # 변환 속성
            if node["transform"]["position"]["x"] != 0 or node["transform"]["position"]["y"] != 0:
                lines.append(f'position = Vector2({node["transform"]["position"]["x"]}, {node["transform"]["position"]["y"]})')
            
            if node["transform"]["scale"]["x"] != 1 or node["transform"]["scale"]["y"] != 1:
                lines.append(f'scale = Vector2({node["transform"]["scale"]["x"]}, {node["transform"]["scale"]["y"]})')
            
            # 추가 속성
            for key, value in node["properties"].items():
                if value is not None:
                    if isinstance(value, str):
                        lines.append(f'{key} = "{value}"')
                    elif isinstance(value, bool):
                        lines.append(f'{key} = {"true" if value else "false"}')
                    else:
                        lines.append(f'{key} = {value}')
            
            lines.append('')
        
        return '\n'.join(lines)
    
    def _find_node_by_id(self, node_id: str) -> Optional[Dict]:
        """ID로 노드 찾기"""
        if not self.current_scene:
            return None
            
        for node in self.current_scene["nodes"]:
            if node["id"] == node_id:
                return node
        return None
    
    def _find_node_by_path(self, path: str) -> Optional[Dict]:
        """경로로 노드 찾기"""
        if not self.current_scene:
            return None
            
        for node in self.current_scene["nodes"]:
            if node["name"] == path:
                return node
        return None
    
    async def generate_script_for_node(self, node_name: str, script_template: str) -> bool:
        """노드용 스크립트 생성"""
        node = self._find_node_by_path(node_name)
        if not node:
            return False
        
        script_path = Path(self.project_path) / "scripts" / f"{node_name}.gd"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 스크립트 템플릿 적용
        script_content = script_template.replace("{{NODE_TYPE}}", node["type"])
        script_content = script_content.replace("{{NODE_NAME}}", node_name)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 노드에 스크립트 연결
        node["properties"]["script"] = f"res://scripts/{node_name}.gd"
        
        return True
    
    async def run_scene(self, scene_path: str) -> None:
        """씬 실행 (테스트)"""
        if not self.project_path:
            return
        
        # Godot으로 씬 실행
        cmd = [self.godot_path, "--path", self.project_path, scene_path]
        subprocess.Popen(cmd)
        
        self.logger.info(f"씬 실행: {scene_path}")
    
    async def create_moving_platform(self, name: str, position: Tuple[float, float]) -> Dict:
        """이동하는 플랫폼 생성"""
        platform_name = f"MovingPlatform_{name}"
        
        # AnimatableBody2D로 플랫폼 생성
        platform = await self.add_node_to_scene("AnimatableBody2D", platform_name, position=position)
        
        # 시각적 표현
        sprite = await self.add_node_to_scene("Sprite2D", "Sprite", parent_path=platform_name)
        sprite["properties"]["texture"] = "res://assets/platform_moving.png"
        
        # 충돌 모양
        collision = await self.add_node_to_scene("CollisionShape2D", "CollisionShape", parent_path=platform_name)
        collision["properties"]["shape"] = {
            "type": "RectangleShape2D",
            "size": {"x": 150, "y": 20}
        }
        
        # 이동 경로 설정
        path = await self.add_node_to_scene("Path2D", f"{platform_name}_Path")
        path["properties"]["curve"] = {
            "points": [
                {"x": position[0], "y": position[1]},
                {"x": position[0] + 200, "y": position[1]},
                {"x": position[0] + 200, "y": position[1] - 100},
                {"x": position[0], "y": position[1] - 100},
                {"x": position[0], "y": position[1]}
            ]
        }
        
        # PathFollow2D 추가
        follow = await self.add_node_to_scene("PathFollow2D", "PathFollow", parent_path=f"{platform_name}_Path")
        follow["properties"]["loop"] = True
        
        # 플랫폼 이동 스크립트
        platform["properties"]["script"] = "res://scripts/MovingPlatform.gd"
        
        # 이동 스크립트 생성
        await self._create_moving_platform_script(platform_name)
        
        return platform
    
    async def create_obstacle(self, position: Tuple[float, float]) -> Dict:
        """장애물 생성"""
        obstacle_types = ["spike", "saw", "barrel", "cone"]
        obstacle_type = obstacle_types[int(time.time() * 1000) % len(obstacle_types)]
        obstacle_name = f"Obstacle_{obstacle_type}_{int(position[0])}_{int(position[1])}"
        
        # 장애물 타입별 설정
        if obstacle_type == "spike":
            # 가시 장애물
            obstacle = await self.add_node_to_scene("Area2D", obstacle_name, position=position)
            sprite = await self.add_node_to_scene("Sprite2D", "Sprite", parent_path=obstacle_name)
            sprite["properties"]["texture"] = "res://assets/spike.png"
            
            collision = await self.add_node_to_scene("CollisionShape2D", "CollisionShape", parent_path=obstacle_name)
            collision["properties"]["shape"] = {
                "type": "ConvexPolygonShape2D",
                "points": [{"x": -16, "y": 16}, {"x": 0, "y": -16}, {"x": 16, "y": 16}]
            }
            
        elif obstacle_type == "saw":
            # 회전 톱날
            obstacle = await self.add_node_to_scene("Area2D", obstacle_name, position=position)
            sprite = await self.add_node_to_scene("AnimatedSprite2D", "AnimatedSprite", parent_path=obstacle_name)
            sprite["properties"]["animation"] = "rotating"
            sprite["properties"]["playing"] = True
            
            collision = await self.add_node_to_scene("CollisionShape2D", "CollisionShape", parent_path=obstacle_name)
            collision["properties"]["shape"] = {
                "type": "CircleShape2D",
                "radius": 24
            }
            
            # 회전 애니메이션
            obstacle["properties"]["rotation_speed"] = 360.0
            
        elif obstacle_type == "barrel":
            # 구르는 통
            obstacle = await self.add_node_to_scene("RigidBody2D", obstacle_name, position=position)
            obstacle["properties"]["mass"] = 5.0
            obstacle["properties"]["linear_velocity"] = {"x": -200, "y": 0}
            
            sprite = await self.add_node_to_scene("Sprite2D", "Sprite", parent_path=obstacle_name)
            sprite["properties"]["texture"] = "res://assets/barrel.png"
            
            collision = await self.add_node_to_scene("CollisionShape2D", "CollisionShape", parent_path=obstacle_name)
            collision["properties"]["shape"] = {
                "type": "CircleShape2D",
                "radius": 32
            }
            
        else:  # cone
            # 교통 콘
            obstacle = await self.add_node_to_scene("StaticBody2D", obstacle_name, position=position)
            sprite = await self.add_node_to_scene("Sprite2D", "Sprite", parent_path=obstacle_name)
            sprite["properties"]["texture"] = "res://assets/cone.png"
            
            collision = await self.add_node_to_scene("CollisionShape2D", "CollisionShape", parent_path=obstacle_name)
            collision["properties"]["shape"] = {
                "type": "ConvexPolygonShape2D",
                "points": [{"x": -20, "y": 32}, {"x": -8, "y": -32}, {"x": 8, "y": -32}, {"x": 20, "y": 32}]
            }
        
        # 공통 속성
        obstacle["properties"]["groups"] = ["obstacles"]
        obstacle["properties"]["collision_layer"] = 4
        obstacle["properties"]["collision_mask"] = 1
        
        return obstacle
    
    async def _create_moving_platform_script(self, platform_name: str):
        """이동 플랫폼 스크립트 생성"""
        script_content = f'''extends AnimatableBody2D

@export var speed = 50.0
@export var wait_time = 1.0

var path_follow: PathFollow2D
var moving = true
var wait_timer = 0.0

func _ready():
    # Path2D의 PathFollow2D 찾기
    var path = get_parent().get_node("{platform_name}_Path")
    if path:
        path_follow = path.get_node("PathFollow")
        if path_follow:
            path_follow.progress = 0.0

func _physics_process(delta):
    if not path_follow:
        return
        
    if moving:
        path_follow.progress += speed * delta
        global_position = path_follow.global_position
        
        # 경로 끝에 도달했는지 확인
        if path_follow.progress_ratio >= 0.99 or path_follow.progress_ratio <= 0.01:
            moving = false
            wait_timer = wait_time
    else:
        wait_timer -= delta
        if wait_timer <= 0:
            moving = true
            speed = -speed  # 방향 전환
'''
        
        script_path = Path(self.project_path) / "scripts" / "MovingPlatform.gd"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_content)


class GodotSceneBuilder:
    """AI가 사용하기 쉬운 고수준 씬 빌더"""
    
    def __init__(self, controller: GodotEditorController):
        self.controller = controller
    
    async def create_platformer_level(self, level_name: str, difficulty: str = "easy") -> bool:
        """플랫포머 레벨 생성"""
        # 씬 생성
        await self.controller.create_scene(level_name, "2D")
        
        # 난이도별 레벨 데이터
        level_layouts = {
            "easy": {
                "platforms": [
                    {"x": 200, "y": 500, "width": 300, "height": 20},
                    {"x": 600, "y": 400, "width": 200, "height": 20},
                    {"x": 900, "y": 300, "width": 250, "height": 20},
                ],
                "enemies": [
                    {"type": "walker", "x": 350, "y": 470},
                    {"type": "walker", "x": 700, "y": 370},
                ],
                "items": [
                    {"type": "coin", "x": 250, "y": 450},
                    {"type": "coin", "x": 650, "y": 350},
                    {"type": "coin", "x": 950, "y": 250},
                ],
                "player_start": {"x": 100, "y": 450}
            },
            "medium": {
                "platforms": [
                    {"x": 200, "y": 550, "width": 200, "height": 20},
                    {"x": 500, "y": 450, "width": 150, "height": 20},
                    {"x": 750, "y": 350, "width": 180, "height": 20},
                    {"x": 1000, "y": 250, "width": 200, "height": 20},
                ],
                "enemies": [
                    {"type": "walker", "x": 300, "y": 520},
                    {"type": "flyer", "x": 600, "y": 300},
                    {"type": "walker", "x": 850, "y": 320},
                ],
                "items": [
                    {"type": "coin", "x": 250, "y": 500},
                    {"type": "powerup", "x": 550, "y": 400},
                    {"type": "coin", "x": 800, "y": 300},
                    {"type": "coin", "x": 1050, "y": 200},
                ],
                "player_start": {"x": 100, "y": 500}
            }
        }
        
        # 레벨 생성
        level_data = level_layouts.get(difficulty, level_layouts["easy"])
        await self.controller.create_level_layout(level_data)
        
        # UI 설정
        await self.controller.setup_game_ui()
        
        # 씬 저장
        return await self.controller.save_scene(f"scenes/{level_name}.tscn")
    
    async def create_rpg_dungeon(self, dungeon_name: str, room_count: int = 5) -> bool:
        """RPG 던전 생성"""
        await self.controller.create_scene(dungeon_name, "2D")
        
        # 던전 벽 생성
        wall_positions = [
            {"pos": (640, 50), "size": (1280, 20)},   # 상단
            {"pos": (640, 670), "size": (1280, 20)},  # 하단
            {"pos": (50, 360), "size": (20, 720)},    # 좌측
            {"pos": (1230, 360), "size": (20, 720)},  # 우측
        ]
        
        for wall in wall_positions:
            await self.controller.create_platform(wall["pos"], wall["size"])
        
        # 플레이어 생성
        await self.controller.create_player_character((640, 360))
        
        # 몬스터 배치
        import random
        for i in range(room_count):
            x = random.randint(200, 1080)
            y = random.randint(200, 520)
            enemy_type = random.choice(["goblin", "skeleton", "orc"])
            await self.controller.create_enemy(enemy_type, (x, y))
        
        # 보물 상자 배치
        for i in range(3):
            x = random.randint(200, 1080)
            y = random.randint(200, 520)
            await self.controller.create_collectible("treasure", (x, y))
        
        await self.controller.setup_game_ui()
        return await self.controller.save_scene(f"scenes/{dungeon_name}.tscn")
    
    async def create_puzzle_level(self, level_name: str, grid_size: int = 8) -> bool:
        """퍼즐 레벨 생성"""
        await self.controller.create_scene(level_name, "2D")
        
        # 그리드 생성
        cell_size = 64
        start_x = 640 - (grid_size * cell_size) // 2
        start_y = 360 - (grid_size * cell_size) // 2
        
        for row in range(grid_size):
            for col in range(grid_size):
                x = start_x + col * cell_size + cell_size // 2
                y = start_y + row * cell_size + cell_size // 2
                
                # 타일 생성
                tile_name = f"Tile_{row}_{col}"
                tile = await self.controller.add_node_to_scene("Area2D", tile_name, position=(x, y))
                
                # 타일 스프라이트
                sprite = await self.controller.add_node_to_scene("Sprite2D", "Sprite", parent_path=tile_name)
                sprite["properties"]["texture"] = "res://assets/puzzle_tile.png"
                
                # 클릭 영역
                collision = await self.controller.add_node_to_scene("CollisionShape2D", "CollisionShape", parent_path=tile_name)
                collision["properties"]["shape"] = {
                    "type": "RectangleShape2D",
                    "size": {"x": cell_size - 4, "y": cell_size - 4}
                }
        
        await self.controller.setup_game_ui()
        return await self.controller.save_scene(f"scenes/{level_name}.tscn")
    
    async def create_rpg_world(self, world_name: str, world_size: str = "medium") -> bool:
        """RPG 월드 생성"""
        await self.controller.create_scene(world_name, "2D")
        
        # 월드 크기별 설정
        world_configs = {
            "small": {"width": 50, "height": 50, "towns": 2, "dungeons": 1},
            "medium": {"width": 100, "height": 100, "towns": 4, "dungeons": 3},
            "large": {"width": 200, "height": 200, "towns": 8, "dungeons": 6}
        }
        
        config = world_configs.get(world_size, world_configs["medium"])
        
        # 타일맵 생성
        tilemap = await self.controller.add_node_to_scene("TileMap", "WorldTileMap")
        tilemap["properties"]["cell_size"] = {"x": 32, "y": 32}
        tilemap["properties"]["tile_set"] = "res://assets/rpg_tileset.tres"
        
        # 지형 생성 (간단한 노이즈 기반)
        import random
        for x in range(config["width"]):
            for y in range(config["height"]):
                # 기본 지형 타입 결정
                noise = random.random()
                if noise < 0.3:
                    tile_type = "grass"
                elif noise < 0.5:
                    tile_type = "dirt"
                elif noise < 0.7:
                    tile_type = "stone"
                else:
                    tile_type = "water"
                    
                # 타일 설정 (실제로는 TileMap API 사용)
                # tilemap.set_cell(x, y, tile_type)
        
        # 마을 생성
        for i in range(config["towns"]):
            town_x = random.randint(10, config["width"] - 10) * 32
            town_y = random.randint(10, config["height"] - 10) * 32
            await self._create_rpg_town(f"Town_{i}", (town_x, town_y))
        
        # 던전 입구 생성
        for i in range(config["dungeons"]):
            dungeon_x = random.randint(5, config["width"] - 5) * 32
            dungeon_y = random.randint(5, config["height"] - 5) * 32
            await self._create_dungeon_entrance(f"Dungeon_{i}", (dungeon_x, dungeon_y))
        
        # 플레이어 시작 위치
        await self.controller.create_player_character((config["width"] * 16, config["height"] * 16))
        
        # RPG UI 설정
        await self._setup_rpg_ui()
        
        return await self.controller.save_scene(f"scenes/{world_name}.tscn")
    
    async def _create_rpg_town(self, town_name: str, position: Tuple[float, float]):
        """RPG 마을 생성"""
        town = await self.controller.add_node_to_scene("Node2D", town_name, position=position)
        
        # 건물들
        buildings = [
            {"type": "house", "offset": (-100, -50)},
            {"type": "shop", "offset": (0, -50)},
            {"type": "inn", "offset": (100, -50)},
            {"type": "house", "offset": (-100, 50)},
            {"type": "blacksmith", "offset": (100, 50)}
        ]
        
        for building in buildings:
            building_name = f"{town_name}_{building['type']}"
            building_pos = (position[0] + building['offset'][0], position[1] + building['offset'][1])
            
            # 건물 생성
            building_node = await self.controller.add_node_to_scene("StaticBody2D", building_name, position=building_pos)
            
            # 건물 스프라이트
            sprite = await self.controller.add_node_to_scene("Sprite2D", "Sprite", parent_path=building_name)
            sprite["properties"]["texture"] = f"res://assets/buildings/{building['type']}.png"
            
            # 충돌 영역
            collision = await self.controller.add_node_to_scene("CollisionShape2D", "CollisionShape", parent_path=building_name)
            collision["properties"]["shape"] = {
                "type": "RectangleShape2D",
                "size": {"x": 64, "y": 64}
            }
            
            # 상호작용 영역
            if building['type'] in ['shop', 'inn', 'blacksmith']:
                interaction = await self.controller.add_node_to_scene("Area2D", "InteractionArea", parent_path=building_name)
                interaction["properties"]["groups"] = ["interactable"]
                interaction["properties"]["interaction_type"] = building['type']
                
                interaction_collision = await self.controller.add_node_to_scene(
                    "CollisionShape2D", 
                    "CollisionShape", 
                    parent_path=f"{building_name}/InteractionArea"
                )
                interaction_collision["properties"]["shape"] = {
                    "type": "CircleShape2D",
                    "radius": 48
                }
        
        # NPC 배치
        import random
        npc_count = random.randint(3, 6)
        for i in range(npc_count):
            npc_x = position[0] + random.randint(-150, 150)
            npc_y = position[1] + random.randint(-100, 100)
            await self._create_rpg_npc(f"{town_name}_NPC_{i}", (npc_x, npc_y))
    
    async def _create_dungeon_entrance(self, dungeon_name: str, position: Tuple[float, float]):
        """던전 입구 생성"""
        entrance = await self.controller.add_node_to_scene("Area2D", dungeon_name, position=position)
        entrance["properties"]["groups"] = ["dungeon_entrance"]
        
        # 입구 스프라이트
        sprite = await self.controller.add_node_to_scene("Sprite2D", "Sprite", parent_path=dungeon_name)
        sprite["properties"]["texture"] = "res://assets/dungeon_entrance.png"
        
        # 충돌 영역
        collision = await self.controller.add_node_to_scene("CollisionShape2D", "CollisionShape", parent_path=dungeon_name)
        collision["properties"]["shape"] = {
            "type": "RectangleShape2D",
            "size": {"x": 64, "y": 64}
        }
        
        # 던전 정보
        import random
        entrance["properties"]["dungeon_level"] = random.randint(1, 10)
        entrance["properties"]["dungeon_type"] = random.choice(["cave", "ruins", "crypt"])
    
    async def _create_rpg_npc(self, npc_name: str, position: Tuple[float, float]):
        """RPG NPC 생성"""
        npc = await self.controller.add_node_to_scene("CharacterBody2D", npc_name, position=position)
        
        # NPC 스프라이트
        sprite = await self.controller.add_node_to_scene("AnimatedSprite2D", "AnimatedSprite", parent_path=npc_name)
        import random
        npc_types = ["villager", "merchant", "guard", "elder"]
        npc_type = random.choice(npc_types)
        sprite["properties"]["sprite_frames"] = f"res://assets/npcs/{npc_type}_frames.tres"
        sprite["properties"]["animation"] = "idle"
        
        # 충돌
        collision = await self.controller.add_node_to_scene("CollisionShape2D", "CollisionShape", parent_path=npc_name)
        collision["properties"]["shape"] = {
            "type": "CapsuleShape2D",
            "radius": 12,
            "height": 24
        }
        
        # 상호작용 영역
        interaction = await self.controller.add_node_to_scene("Area2D", "InteractionArea", parent_path=npc_name)
        interaction["properties"]["groups"] = ["npc_interaction"]
        
        interaction_collision = await self.controller.add_node_to_scene(
            "CollisionShape2D", 
            "CollisionShape", 
            parent_path=f"{npc_name}/InteractionArea"
        )
        interaction_collision["properties"]["shape"] = {
            "type": "CircleShape2D",
            "radius": 40
        }
        
        # NPC 데이터
        npc["properties"]["npc_type"] = npc_type
        npc["properties"]["dialogue"] = self._get_npc_dialogue(npc_type)
        npc["properties"]["script"] = "res://scripts/RPG_NPC.gd"
    
    def _get_npc_dialogue(self, npc_type: str) -> List[str]:
        """NPC 대화 내용 생성"""
        dialogues = {
            "villager": [
                "안녕하세요, 모험가님!",
                "오늘 날씨가 좋네요.",
                "북쪽 던전에는 위험한 몬스터가 있다고 들었어요."
            ],
            "merchant": [
                "좋은 물건이 많이 있습니다!",
                "오늘만 특별 할인입니다.",
                "포션이 필요하신가요?"
            ],
            "guard": [
                "마을의 평화를 지키고 있습니다.",
                "밤에는 마을 밖으로 나가지 마세요.",
                "수상한 사람을 본 적 있나요?"
            ],
            "elder": [
                "오래전 이 마을에는...",
                "전설의 검이 어딘가에 숨겨져 있다고 합니다.",
                "젊은이여, 운명이 당신을 이곳으로 인도했군요."
            ]
        }
        return dialogues.get(npc_type, ["..."])
    
    async def _setup_rpg_ui(self):
        """RPG UI 설정"""
        # 체력/마나 바
        await self.controller.create_ui_element("ProgressBar", "HealthBar", position=(20, 20), size=(200, 20))
        await self.controller.create_ui_element("ProgressBar", "ManaBar", position=(20, 50), size=(200, 20))
        
        # 경험치 바
        await self.controller.create_ui_element("ProgressBar", "ExpBar", position=(20, 80), size=(200, 10))
        
        # 인벤토리 버튼
        await self.controller.create_ui_element("Button", "InventoryButton", position=(1200, 20), size=(60, 60))
        
        # 퀘스트 로그
        await self.controller.create_ui_element("RichTextLabel", "QuestLog", position=(1000, 100), size=(260, 200))
        
        # 대화 상자
        dialogue_box = await self.controller.create_ui_element("PanelContainer", "DialogueBox", position=(200, 500), size=(880, 150))
        dialogue_box["properties"]["visible"] = False
        
        # 대화 텍스트
        dialogue_text = await self.controller.add_node_to_scene(
            "RichTextLabel", 
            "DialogueText", 
            parent_path="UI/DialogueBox",
            position=(20, 20)
        )
        dialogue_text["properties"]["size"] = {"x": 840, "y": 110}
        dialogue_text["properties"]["bbcode_enabled"] = True