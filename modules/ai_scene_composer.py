#!/usr/bin/env python3
"""
AI 씬 자동 구성 및 편집 시스템
AI가 게임 요구사항에 따라 씬을 지능적으로 구성하고 편집
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import xml.etree.ElementTree as ET
import math
import random

@dataclass
class NodeTemplate:
    """노드 템플릿"""
    node_type: str
    name: str
    position: Optional[Tuple[float, float]] = None
    scale: Optional[Tuple[float, float]] = None
    rotation: Optional[float] = None
    properties: Dict[str, Any] = None
    children: List['NodeTemplate'] = None
    script_path: Optional[str] = None
    groups: List[str] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.children is None:
            self.children = []
        if self.groups is None:
            self.groups = []

@dataclass
class SceneLayout:
    """씬 레이아웃 정의"""
    name: str
    root_type: str
    layout_type: str  # "platformer", "racing", "puzzle", "rpg", "menu"
    dimensions: Tuple[int, int]
    camera_settings: Dict[str, Any]
    lighting: Dict[str, Any]
    physics_settings: Dict[str, Any]
    ui_layout: Dict[str, Any]
    gameplay_elements: List[NodeTemplate]

class AISceneComposer:
    """AI 씬 자동 구성 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger("AISceneComposer")
        
        # 게임 타입별 씬 템플릿
        self.scene_templates = {
            "platformer": self._get_platformer_templates(),
            "racing": self._get_racing_templates(),
            "puzzle": self._get_puzzle_templates(),
            "rpg": self._get_rpg_templates(),
            "menu": self._get_menu_templates()
        }
        
        # 노드 타입별 기본 설정
        self.node_defaults = self._load_node_defaults()
        
        # AI 기반 배치 알고리즘
        self.placement_algorithms = {
            "random": self._random_placement,
            "grid": self._grid_placement,
            "organic": self._organic_placement,
            "balanced": self._balanced_placement,
            "guided": self._guided_placement
        }
        
    async def compose_scene_intelligently(self, game_type: str, scene_purpose: str, 
                                        requirements: Dict[str, Any]) -> SceneLayout:
        """게임 타입과 목적에 따른 지능적 씬 구성"""
        self.logger.info(f"AI 씬 구성 시작: {game_type} - {scene_purpose}")
        
        # 기본 씬 구조 결정
        base_layout = await self._determine_base_layout(game_type, scene_purpose)
        
        # 요구사항에 따른 커스터마이징
        customized_layout = await self._customize_layout(base_layout, requirements)
        
        # AI 기반 요소 배치
        final_layout = await self._place_elements_intelligently(customized_layout, requirements)
        
        # 씬 최적화
        optimized_layout = await self._optimize_scene_layout(final_layout)
        
        return optimized_layout
        
    async def _determine_base_layout(self, game_type: str, scene_purpose: str) -> SceneLayout:
        """기본 레이아웃 결정"""
        templates = self.scene_templates.get(game_type, {})
        template = templates.get(scene_purpose, templates.get("default", {}))
        
        if not template:
            # 기본 템플릿 생성
            template = self._create_default_template(game_type, scene_purpose)
            
        return SceneLayout(
            name=f"{game_type}_{scene_purpose}",
            root_type=template.get("root_type", "Node2D"),
            layout_type=game_type,
            dimensions=template.get("dimensions", (1920, 1080)),
            camera_settings=template.get("camera", {}),
            lighting=template.get("lighting", {}),
            physics_settings=template.get("physics", {}),
            ui_layout=template.get("ui", {}),
            gameplay_elements=[]
        )
        
    async def _customize_layout(self, base_layout: SceneLayout, 
                              requirements: Dict[str, Any]) -> SceneLayout:
        """요구사항에 따른 레이아웃 커스터마이징"""
        customized = base_layout
        
        # 카메라 설정 조정
        if "camera" in requirements:
            customized.camera_settings.update(requirements["camera"])
            
        # 물리 설정 조정
        if "physics" in requirements:
            customized.physics_settings.update(requirements["physics"])
            
        # UI 레이아웃 조정
        if "ui" in requirements:
            customized.ui_layout.update(requirements["ui"])
            
        # 크기 조정
        if "dimensions" in requirements:
            customized.dimensions = requirements["dimensions"]
            
        return customized
        
    async def _place_elements_intelligently(self, layout: SceneLayout, 
                                          requirements: Dict[str, Any]) -> SceneLayout:
        """AI 기반 요소 배치"""
        elements_to_place = requirements.get("elements", [])
        placement_strategy = requirements.get("placement_strategy", "balanced")
        
        algorithm = self.placement_algorithms.get(placement_strategy, self._balanced_placement)
        
        for element_spec in elements_to_place:
            element = await self._create_element_from_spec(element_spec)
            positioned_element = await algorithm(element, layout, requirements)
            layout.gameplay_elements.append(positioned_element)
            
        return layout
        
    async def _create_element_from_spec(self, spec: Dict[str, Any]) -> NodeTemplate:
        """사양으로부터 요소 생성"""
        element_type = spec.get("type", "Node2D")
        element_name = spec.get("name", f"Element_{random.randint(1000, 9999)}")
        
        # 기본 속성 로드
        defaults = self.node_defaults.get(element_type, {})
        properties = {**defaults, **spec.get("properties", {})}
        
        # 자식 노드들 생성
        children = []
        for child_spec in spec.get("children", []):
            child = await self._create_element_from_spec(child_spec)
            children.append(child)
            
        return NodeTemplate(
            node_type=element_type,
            name=element_name,
            properties=properties,
            children=children,
            script_path=spec.get("script"),
            groups=spec.get("groups", [])
        )
        
    async def _random_placement(self, element: NodeTemplate, layout: SceneLayout, 
                              requirements: Dict[str, Any]) -> NodeTemplate:
        """무작위 배치"""
        width, height = layout.dimensions
        margin = requirements.get("margin", 50)
        
        element.position = (
            random.uniform(margin, width - margin),
            random.uniform(margin, height - margin)
        )
        
        return element
        
    async def _grid_placement(self, element: NodeTemplate, layout: SceneLayout, 
                            requirements: Dict[str, Any]) -> NodeTemplate:
        """격자 배치"""
        grid_size = requirements.get("grid_size", (10, 10))
        width, height = layout.dimensions
        
        grid_x = width / grid_size[0]
        grid_y = height / grid_size[1]
        
        # 기존 요소들의 위치 분석
        occupied_cells = set()
        for existing in layout.gameplay_elements:
            if existing.position:
                cell_x = int(existing.position[0] / grid_x)
                cell_y = int(existing.position[1] / grid_y)
                occupied_cells.add((cell_x, cell_y))
                
        # 빈 격자 찾기
        for y in range(grid_size[1]):
            for x in range(grid_size[0]):
                if (x, y) not in occupied_cells:
                    element.position = (x * grid_x + grid_x/2, y * grid_y + grid_y/2)
                    return element
                    
        # 모든 격자가 차면 겹치게 배치
        element.position = (grid_x/2, grid_y/2)
        return element
        
    async def _organic_placement(self, element: NodeTemplate, layout: SceneLayout, 
                               requirements: Dict[str, Any]) -> NodeTemplate:
        """자연스러운 배치"""
        width, height = layout.dimensions
        
        # 기존 요소들과의 관계 고려
        min_distance = requirements.get("min_distance", 100)
        max_attempts = requirements.get("max_attempts", 50)
        
        for attempt in range(max_attempts):
            # 클러스터링을 고려한 위치 생성
            if layout.gameplay_elements:
                # 기존 요소 근처에 배치 (클러스터링)
                base_element = random.choice(layout.gameplay_elements)
                if base_element.position:
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(min_distance, min_distance * 2)
                    
                    new_x = base_element.position[0] + math.cos(angle) * distance
                    new_y = base_element.position[1] + math.sin(angle) * distance
                    
                    # 경계 확인
                    new_x = max(50, min(width - 50, new_x))
                    new_y = max(50, min(height - 50, new_y))
                else:
                    new_x = random.uniform(50, width - 50)
                    new_y = random.uniform(50, height - 50)
            else:
                # 첫 번째 요소는 중앙 근처에
                center_x, center_y = width / 2, height / 2
                offset_x = random.uniform(-width/4, width/4)
                offset_y = random.uniform(-height/4, height/4)
                
                new_x = center_x + offset_x
                new_y = center_y + offset_y
                
            # 다른 요소들과의 최소 거리 확인
            valid_position = True
            for existing in layout.gameplay_elements:
                if existing.position:
                    distance = math.sqrt(
                        (new_x - existing.position[0])**2 + 
                        (new_y - existing.position[1])**2
                    )
                    if distance < min_distance:
                        valid_position = False
                        break
                        
            if valid_position:
                element.position = (new_x, new_y)
                break
        else:
            # 최대 시도 후에도 위치를 찾지 못하면 무작위 배치
            element.position = (
                random.uniform(50, width - 50),
                random.uniform(50, height - 50)
            )
            
        return element
        
    async def _balanced_placement(self, element: NodeTemplate, layout: SceneLayout, 
                                requirements: Dict[str, Any]) -> NodeTemplate:
        """균형 잡힌 배치"""
        width, height = layout.dimensions
        
        # 씬을 구역으로 나누어 배치
        zones = requirements.get("zones", 4)  # 4개 구역으로 나눔
        zone_width = width / 2
        zone_height = height / 2
        
        # 각 구역의 요소 수 계산
        zone_counts = [0, 0, 0, 0]
        for existing in layout.gameplay_elements:
            if existing.position:
                zone_x = 0 if existing.position[0] < zone_width else 1
                zone_y = 0 if existing.position[1] < zone_height else 1
                zone_index = zone_y * 2 + zone_x
                zone_counts[zone_index] += 1
                
        # 가장 적은 요소가 있는 구역 선택
        target_zone = zone_counts.index(min(zone_counts))
        zone_x = target_zone % 2
        zone_y = target_zone // 2
        
        # 해당 구역 내에서 위치 선정
        margin = 50
        min_x = zone_x * zone_width + margin
        max_x = (zone_x + 1) * zone_width - margin
        min_y = zone_y * zone_height + margin
        max_y = (zone_y + 1) * zone_height - margin
        
        element.position = (
            random.uniform(min_x, max_x),
            random.uniform(min_y, max_y)
        )
        
        return element
        
    async def _guided_placement(self, element: NodeTemplate, layout: SceneLayout, 
                              requirements: Dict[str, Any]) -> NodeTemplate:
        """가이드된 배치 (특정 규칙 기반)"""
        width, height = layout.dimensions
        element_type = element.node_type
        
        # 요소 타입별 배치 규칙
        placement_rules = {
            "Player": {"zone": "bottom_center", "priority": "high"},
            "Enemy": {"zone": "top_sides", "priority": "medium"},
            "Collectible": {"zone": "scattered", "priority": "low"},
            "Platform": {"zone": "horizontal_lines", "priority": "high"},
            "Checkpoint": {"zone": "path_points", "priority": "medium"},
            "UI": {"zone": "screen_edges", "priority": "high"}
        }
        
        rule = placement_rules.get(element_type, {"zone": "center", "priority": "medium"})
        zone = rule["zone"]
        
        if zone == "bottom_center":
            element.position = (width / 2, height * 0.8)
        elif zone == "top_sides":
            side = random.choice([0.2, 0.8])
            element.position = (width * side, height * 0.2)
        elif zone == "scattered":
            element.position = (
                random.uniform(width * 0.1, width * 0.9),
                random.uniform(height * 0.1, height * 0.9)
            )
        elif zone == "horizontal_lines":
            y_levels = [height * 0.3, height * 0.5, height * 0.7]
            y_pos = random.choice(y_levels)
            element.position = (random.uniform(0, width), y_pos)
        elif zone == "path_points":
            # 경로를 따라 배치
            path_points = requirements.get("path_points", [(width*0.1, height*0.5), (width*0.9, height*0.5)])
            element.position = random.choice(path_points)
        elif zone == "screen_edges":
            edges = ["top", "bottom", "left", "right"]
            edge = random.choice(edges)
            margin = 30
            
            if edge == "top":
                element.position = (random.uniform(0, width), margin)
            elif edge == "bottom":
                element.position = (random.uniform(0, width), height - margin)
            elif edge == "left":
                element.position = (margin, random.uniform(0, height))
            else:  # right
                element.position = (width - margin, random.uniform(0, height))
        else:  # center
            element.position = (width / 2, height / 2)
            
        return element
        
    async def _optimize_scene_layout(self, layout: SceneLayout) -> SceneLayout:
        """씬 레이아웃 최적화"""
        # 겹침 해결
        layout = await self._resolve_overlaps(layout)
        
        # 성능 최적화
        layout = await self._optimize_performance(layout)
        
        # 접근성 확인
        layout = await self._ensure_accessibility(layout)
        
        return layout
        
    async def _resolve_overlaps(self, layout: SceneLayout) -> SceneLayout:
        """요소 겹침 해결"""
        min_distance = 30  # 최소 거리
        
        for i, element in enumerate(layout.gameplay_elements):
            if not element.position:
                continue
                
            for j, other in enumerate(layout.gameplay_elements[i+1:], i+1):
                if not other.position:
                    continue
                    
                distance = math.sqrt(
                    (element.position[0] - other.position[0])**2 + 
                    (element.position[1] - other.position[1])**2
                )
                
                if distance < min_distance:
                    # 겹침 해결 - 한 요소를 이동
                    angle = random.uniform(0, 2 * math.pi)
                    new_x = other.position[0] + math.cos(angle) * min_distance
                    new_y = other.position[1] + math.sin(angle) * min_distance
                    
                    # 경계 확인
                    width, height = layout.dimensions
                    new_x = max(30, min(width - 30, new_x))
                    new_y = max(30, min(height - 30, new_y))
                    
                    layout.gameplay_elements[j].position = (new_x, new_y)
                    
        return layout
        
    async def _optimize_performance(self, layout: SceneLayout) -> SceneLayout:
        """성능 최적화"""
        # 너무 많은 요소가 있으면 일부 제거 또는 그룹화
        max_elements = 100
        
        if len(layout.gameplay_elements) > max_elements:
            # 우선순위가 낮은 요소들 제거
            layout.gameplay_elements.sort(key=lambda x: x.properties.get("priority", 5))
            layout.gameplay_elements = layout.gameplay_elements[:max_elements]
            
        return layout
        
    async def _ensure_accessibility(self, layout: SceneLayout) -> SceneLayout:
        """접근성 확인"""
        # 플레이어가 모든 중요한 요소에 접근할 수 있는지 확인
        # 경로 분석 및 조정
        
        # 간단한 구현: 너무 멀리 떨어진 요소들을 중앙으로 이동
        width, height = layout.dimensions
        center_x, center_y = width / 2, height / 2
        max_distance = min(width, height) / 2
        
        for element in layout.gameplay_elements:
            if element.position:
                distance = math.sqrt(
                    (element.position[0] - center_x)**2 + 
                    (element.position[1] - center_y)**2
                )
                
                if distance > max_distance:
                    # 중앙으로 가깝게 이동
                    ratio = max_distance / distance
                    new_x = center_x + (element.position[0] - center_x) * ratio
                    new_y = center_y + (element.position[1] - center_y) * ratio
                    element.position = (new_x, new_y)
                    
        return layout
        
    def _get_platformer_templates(self) -> Dict[str, Any]:
        """플랫포머 씬 템플릿"""
        return {
            "level": {
                "root_type": "Node2D",
                "dimensions": (2000, 1200),
                "camera": {
                    "type": "Camera2D",
                    "zoom": [2.0, 2.0],
                    "follow_player": True,
                    "smoothing": True
                },
                "physics": {
                    "gravity": 980,
                    "default_friction": 1.0
                },
                "ui": {
                    "hud_position": "top",
                    "health_bar": True,
                    "score_display": True
                }
            },
            "menu": {
                "root_type": "Control",
                "dimensions": (1920, 1080),
                "camera": {},
                "ui": {
                    "layout": "vertical_center",
                    "background": True,
                    "buttons": ["Start", "Options", "Quit"]
                }
            }
        }
        
    def _get_racing_templates(self) -> Dict[str, Any]:
        """레이싱 씬 템플릿"""
        return {
            "track": {
                "root_type": "Node3D",
                "dimensions": (5000, 3000),
                "camera": {
                    "type": "Camera3D",
                    "follow_target": True,
                    "position_offset": [0, 5, 10],
                    "rotation_offset": [-15, 0, 0]
                },
                "physics": {
                    "gravity": 9.8,
                    "vehicle_physics": True
                },
                "lighting": {
                    "environment": "outdoor",
                    "sun_angle": 45,
                    "shadows": True
                }
            }
        }
        
    def _get_puzzle_templates(self) -> Dict[str, Any]:
        """퍼즐 씬 템플릿"""
        return {
            "puzzle": {
                "root_type": "Control",
                "dimensions": (1280, 720),
                "camera": {},
                "ui": {
                    "layout": "centered",
                    "grid_based": True,
                    "moves_counter": True,
                    "timer": True
                }
            }
        }
        
    def _get_rpg_templates(self) -> Dict[str, Any]:
        """RPG 씬 템플릿"""
        return {
            "world": {
                "root_type": "Node2D",
                "dimensions": (3000, 3000),
                "camera": {
                    "type": "Camera2D",
                    "zoom": [1.5, 1.5],
                    "follow_player": True,
                    "bounds": True
                },
                "physics": {
                    "gravity": 0,
                    "collision_layers": ["player", "npc", "environment", "items"]
                },
                "ui": {
                    "hud_elements": ["health", "mana", "inventory", "minimap"]
                }
            }
        }
        
    def _get_menu_templates(self) -> Dict[str, Any]:
        """메뉴 씬 템플릿"""
        return {
            "main_menu": {
                "root_type": "Control",
                "dimensions": (1920, 1080),
                "ui": {
                    "layout": "vertical_center",
                    "background_image": True,
                    "title": True,
                    "buttons": ["New Game", "Continue", "Settings", "Exit"]
                }
            },
            "settings": {
                "root_type": "Control", 
                "dimensions": (1920, 1080),
                "ui": {
                    "layout": "tabbed",
                    "tabs": ["Graphics", "Audio", "Controls"],
                    "apply_button": True
                }
            }
        }
        
    def _load_node_defaults(self) -> Dict[str, Dict[str, Any]]:
        """노드 타입별 기본 설정"""
        return {
            "Player": {
                "priority": 10,
                "physics": True,
                "script_required": True,
                "collision": True
            },
            "Enemy": {
                "priority": 7,
                "physics": True,
                "ai_required": True,
                "collision": True
            },
            "Collectible": {
                "priority": 3,
                "physics": False,
                "auto_collect": True,
                "animation": True
            },
            "Platform": {
                "priority": 8,
                "physics": True,
                "static": True,
                "collision": True
            },
            "Background": {
                "priority": 1,
                "physics": False,
                "parallax": True,
                "z_index": -10
            },
            "UI": {
                "priority": 9,
                "canvas_layer": True,
                "mouse_filter": "pass"
            }
        }
        
    def _create_default_template(self, game_type: str, scene_purpose: str) -> Dict[str, Any]:
        """기본 템플릿 생성"""
        return {
            "root_type": "Node2D",
            "dimensions": (1920, 1080),
            "camera": {"type": "Camera2D"},
            "physics": {},
            "ui": {},
            "lighting": {}
        }
        
    async def generate_scene_file(self, layout: SceneLayout, output_path: Path) -> bool:
        """씬 파일 생성"""
        try:
            scene_content = await self._generate_tscn_content(layout)
            
            with open(output_path, 'w') as f:
                f.write(scene_content)
                
            self.logger.info(f"씬 파일 생성 완료: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"씬 파일 생성 실패: {e}")
            return False
            
    async def _generate_tscn_content(self, layout: SceneLayout) -> str:
        """TSCN 파일 내용 생성"""
        content_lines = []
        
        # 헤더
        content_lines.append(f'[gd_scene load_steps=1 format=3 uid="uid://ai_generated_{layout.name}"]')
        content_lines.append('')
        
        # 루트 노드
        content_lines.append(f'[node name="{layout.name}" type="{layout.root_type}"]')
        
        # 카메라 추가
        if layout.camera_settings:
            camera_type = layout.camera_settings.get("type", "Camera2D")
            content_lines.append(f'[node name="Camera" type="{camera_type}" parent="."]')
            
            if "zoom" in layout.camera_settings:
                zoom = layout.camera_settings["zoom"]
                content_lines.append(f'zoom = Vector2({zoom[0]}, {zoom[1]})')
                
        # 게임플레이 요소들 추가
        for i, element in enumerate(layout.gameplay_elements):
            content_lines.extend(self._generate_node_content(element, f"Element{i}"))
            
        return '\n'.join(content_lines)
        
    def _generate_node_content(self, node: NodeTemplate, node_id: str) -> List[str]:
        """개별 노드 내용 생성"""
        lines = []
        
        # 노드 선언
        lines.append(f'[node name="{node.name}" type="{node.node_type}" parent="."]')
        
        # 위치 설정
        if node.position:
            if node.node_type in ["Node2D", "CharacterBody2D", "RigidBody2D", "Area2D", "StaticBody2D"]:
                lines.append(f'position = Vector2({node.position[0]}, {node.position[1]})')
            elif node.node_type in ["Node3D", "CharacterBody3D", "RigidBody3D", "Area3D", "StaticBody3D"]:
                lines.append(f'position = Vector3({node.position[0]}, 0, {node.position[1]})')
                
        # 스케일 설정
        if node.scale:
            lines.append(f'scale = Vector2({node.scale[0]}, {node.scale[1]})')
            
        # 회전 설정
        if node.rotation:
            lines.append(f'rotation = {node.rotation}')
            
        # 속성 설정
        for prop, value in node.properties.items():
            if isinstance(value, str):
                lines.append(f'{prop} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f'{prop} = {str(value).lower()}')
            else:
                lines.append(f'{prop} = {value}')
                
        # 스크립트 첨부
        if node.script_path:
            lines.append(f'script = preload("{node.script_path}")')
            
        # 그룹 설정 (주석으로 표시)
        if node.groups:
            lines.append(f'# Groups: {", ".join(node.groups)}')
            
        lines.append('')
        
        # 자식 노드들
        for child in node.children:
            child_lines = self._generate_node_content(child, f"{node_id}_child")
            # 부모 경로 수정
            for line in child_lines:
                if 'parent="."' in line:
                    line = line.replace('parent="."', f'parent="{node.name}"')
                lines.append(line)
                
        return lines

# 사용 예제 및 테스트
async def main():
    """메인 실행 함수"""
    print("🎭 AI 씬 자동 구성 시스템")
    print("=" * 60)
    
    composer = AISceneComposer()
    
    # 플랫포머 레벨 씬 생성 예제
    requirements = {
        "elements": [
            {
                "type": "CharacterBody2D",
                "name": "Player",
                "properties": {"priority": 10},
                "script": "res://scripts/Player.gd",
                "children": [
                    {"type": "Sprite2D", "name": "Sprite"},
                    {"type": "CollisionShape2D", "name": "Collision"}
                ]
            },
            {
                "type": "StaticBody2D", 
                "name": "Platform",
                "properties": {"priority": 8},
                "children": [
                    {"type": "Sprite2D", "name": "Sprite"},
                    {"type": "CollisionShape2D", "name": "Collision"}
                ]
            },
            {
                "type": "Area2D",
                "name": "Collectible",
                "properties": {"priority": 3},
                "groups": ["collectibles"],
                "children": [
                    {"type": "Sprite2D", "name": "Sprite"},
                    {"type": "CollisionShape2D", "name": "Collision"}
                ]
            }
        ],
        "placement_strategy": "guided",
        "camera": {"zoom": [2.0, 2.0]},
        "dimensions": (2000, 1200)
    }
    
    # 씬 구성
    layout = await composer.compose_scene_intelligently("platformer", "level", requirements)
    
    print(f"씬 구성 완료: {layout.name}")
    print(f"요소 수: {len(layout.gameplay_elements)}")
    print(f"크기: {layout.dimensions}")
    
    # 씬 파일 생성
    output_path = Path("/tmp/ai_generated_scene.tscn")
    success = await composer.generate_scene_file(layout, output_path)
    
    if success:
        print(f"씬 파일 생성 성공: {output_path}")
    else:
        print("씬 파일 생성 실패")

if __name__ == "__main__":
    asyncio.run(main())