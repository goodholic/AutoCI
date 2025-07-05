#!/usr/bin/env python3
"""
Godot Editor API 자동화 스크립트
AI가 Godot Editor를 직접 제어하여 모든 작업을 자동화
"""

import os
import sys
import json
import asyncio
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import xml.etree.ElementTree as ET
import re

class GodotEditorAPI:
    """Godot Editor API 자동화 인터페이스"""
    
    def __init__(self, godot_executable: str = None):
        self.godot_path = godot_executable or self._find_godot()
        self.logger = logging.getLogger("GodotEditorAPI")
        
        # EditorScript 템플릿들
        self.editor_scripts = {
            "scene_creator": self._get_scene_creator_script(),
            "node_manipulator": self._get_node_manipulator_script(),
            "resource_manager": self._get_resource_manager_script(),
            "project_analyzer": self._get_project_analyzer_script(),
            "automated_builder": self._get_automated_builder_script()
        }
        
    def _find_godot(self) -> str:
        """Godot 실행 파일 찾기"""
        possible_paths = [
            "/usr/local/bin/godot",
            "/usr/bin/godot",
            "/opt/godot/godot",
            "godot",
            "Godot_v4.3-stable_linux.x86_64"
        ]
        
        for path in possible_paths:
            if os.system(f"which {path} > /dev/null 2>&1") == 0:
                return path
                
        return "godot"
        
    async def execute_editor_script(self, project_path: str, script_name: str, 
                                  parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Editor 스크립트 실행"""
        if script_name not in self.editor_scripts:
            raise ValueError(f"Unknown script: {script_name}")
            
        # 임시 스크립트 파일 생성
        script_content = self.editor_scripts[script_name]
        if parameters:
            script_content = self._inject_parameters(script_content, parameters)
            
        script_path = await self._create_temp_script(script_content)
        
        try:
            # Godot Editor에서 스크립트 실행
            result = await self._run_editor_script(project_path, script_path)
            return result
        finally:
            # 임시 파일 정리
            if script_path.exists():
                script_path.unlink()
                
    async def create_scene_automatically(self, project_path: str, scene_config: Dict[str, Any]) -> bool:
        """씬 자동 생성"""
        parameters = {
            "scene_name": scene_config.get("name", "NewScene"),
            "scene_type": scene_config.get("type", "Node2D"),
            "nodes": json.dumps(scene_config.get("nodes", [])),
            "save_path": scene_config.get("path", "res://scenes/")
        }
        
        result = await self.execute_editor_script(
            project_path, "scene_creator", parameters
        )
        
        return result.get("success", False)
        
    async def manipulate_nodes_batch(self, project_path: str, operations: List[Dict[str, Any]]) -> bool:
        """노드 일괄 조작"""
        parameters = {
            "operations": json.dumps(operations)
        }
        
        result = await self.execute_editor_script(
            project_path, "node_manipulator", parameters
        )
        
        return result.get("success", False)
        
    async def manage_resources_automatically(self, project_path: str, 
                                           resource_operations: List[Dict[str, Any]]) -> bool:
        """리소스 자동 관리"""
        parameters = {
            "operations": json.dumps(resource_operations)
        }
        
        result = await self.execute_editor_script(
            project_path, "resource_manager", parameters
        )
        
        return result.get("success", False)
        
    async def analyze_project_structure(self, project_path: str) -> Dict[str, Any]:
        """프로젝트 구조 분석"""
        result = await self.execute_editor_script(
            project_path, "project_analyzer"
        )
        
        return result.get("analysis", {})
        
    async def build_project_automatically(self, project_path: str, 
                                        build_config: Dict[str, Any]) -> bool:
        """프로젝트 자동 빌드"""
        parameters = {
            "export_preset": build_config.get("preset", "Linux/X11"),
            "output_path": build_config.get("output", "build/"),
            "debug_mode": build_config.get("debug", False)
        }
        
        result = await self.execute_editor_script(
            project_path, "automated_builder", parameters
        )
        
        return result.get("success", False)
        
    def _get_scene_creator_script(self) -> str:
        """씬 생성 자동화 스크립트"""
        return '''@tool
extends EditorScript

# 매개변수 (실행 시 주입됨)
var scene_name: String = "{scene_name}"
var scene_type: String = "{scene_type}"
var nodes_data: String = "{nodes}"
var save_path: String = "{save_path}"

func _run():
    print("AI Scene Creator 실행 시작...")
    
    # 새 씬 생성
    var scene = PackedScene.new()
    var root_node = _create_node_by_type(scene_type)
    root_node.name = scene_name
    
    # 노드 구조 생성
    var nodes = JSON.parse_string(nodes_data)
    if nodes != null and nodes is Array:
        for node_data in nodes:
            _add_node_recursive(root_node, node_data)
    
    # 씬 패킹 및 저장
    scene.pack(root_node)
    var full_path = save_path + scene_name + ".tscn"
    var result = ResourceSaver.save(scene, full_path)
    
    if result == OK:
        print("씬 생성 성공: ", full_path)
        # 에디터에서 씬 열기
        EditorInterface.open_scene_from_path(full_path)
        _output_result({"success": true, "path": full_path})
    else:
        print("씬 생성 실패")
        _output_result({"success": false, "error": "Failed to save scene"})

func _create_node_by_type(type_name: String) -> Node:
    match type_name:
        "Node2D":
            return Node2D.new()
        "Node3D":
            return Node3D.new()
        "Control":
            return Control.new()
        "CharacterBody2D":
            return CharacterBody2D.new()
        "CharacterBody3D":
            return CharacterBody3D.new()
        "RigidBody2D":
            return RigidBody2D.new()
        "RigidBody3D":
            return RigidBody3D.new()
        "Area2D":
            return Area2D.new()
        "Area3D":
            return Area3D.new()
        "StaticBody2D":
            return StaticBody2D.new()
        "StaticBody3D":
            return StaticBody3D.new()
        "Sprite2D":
            return Sprite2D.new()
        "Sprite3D":
            return Sprite3D.new()
        "Label":
            return Label.new()
        "Button":
            return Button.new()
        "Timer":
            return Timer.new()
        "AudioStreamPlayer":
            return AudioStreamPlayer.new()
        "AudioStreamPlayer2D":
            return AudioStreamPlayer2D.new()
        "AudioStreamPlayer3D":
            return AudioStreamPlayer3D.new()
        "AnimationPlayer":
            return AnimationPlayer.new()
        "Camera2D":
            return Camera2D.new()
        "Camera3D":
            return Camera3D.new()
        "CollisionShape2D":
            return CollisionShape2D.new()
        "CollisionShape3D":
            return CollisionShape3D.new()
        _:
            return Node.new()

func _add_node_recursive(parent: Node, node_data: Dictionary):
    var node_type = node_data.get("type", "Node")
    var node_name = node_data.get("name", "NewNode")
    var node_position = node_data.get("position", Vector2.ZERO)
    var node_properties = node_data.get("properties", {})
    var child_nodes = node_data.get("children", [])
    
    var new_node = _create_node_by_type(node_type)
    new_node.name = node_name
    
    # 노드 속성 설정
    _apply_node_properties(new_node, node_properties)
    
    # 위치 설정 (Node2D인 경우)
    if new_node is Node2D and node_position != Vector2.ZERO:
        new_node.position = node_position
    
    parent.add_child(new_node)
    new_node.owner = parent.get_tree().current_scene if parent.get_tree() else parent
    
    # 자식 노드들 재귀적 추가
    for child_data in child_nodes:
        _add_node_recursive(new_node, child_data)

func _apply_node_properties(node: Node, properties: Dictionary):
    for property_name in properties:
        var property_value = properties[property_name]
        
        if node.has_method("set_" + property_name):
            node.call("set_" + property_name, property_value)
        elif property_name in node:
            node[property_name] = property_value

func _output_result(result: Dictionary):
    # 결과를 임시 파일에 저장
    var temp_file = FileAccess.open("user://editor_script_result.json", FileAccess.WRITE)
    if temp_file:
        temp_file.store_string(JSON.stringify(result))
        temp_file.close()
'''

    def _get_node_manipulator_script(self) -> str:
        """노드 조작 자동화 스크립트"""
        return '''@tool
extends EditorScript

var operations_data: String = "{operations}"

func _run():
    print("AI Node Manipulator 실행 시작...")
    
    var operations = JSON.parse_string(operations_data)
    if operations == null or not operations is Array:
        _output_result({"success": false, "error": "Invalid operations data"})
        return
    
    var current_scene = EditorInterface.get_edited_scene_root()
    if current_scene == null:
        _output_result({"success": false, "error": "No scene is currently open"})
        return
    
    var success_count = 0
    var errors = []
    
    for operation in operations:
        var result = _execute_operation(current_scene, operation)
        if result.success:
            success_count += 1
        else:
            errors.append(result.error)
    
    var final_result = {
        "success": errors.is_empty(),
        "operations_executed": success_count,
        "total_operations": operations.size(),
        "errors": errors
    }
    
    _output_result(final_result)

func _execute_operation(scene_root: Node, operation: Dictionary) -> Dictionary:
    var op_type = operation.get("type", "")
    var target_path = operation.get("target", "")
    var parameters = operation.get("parameters", {})
    
    var target_node = scene_root.get_node_or_null(target_path) if target_path else scene_root
    
    if target_node == null and op_type != "create_node":
        return {"success": false, "error": "Target node not found: " + target_path}
    
    match op_type:
        "create_node":
            return _create_node_operation(scene_root, parameters)
        "delete_node":
            return _delete_node_operation(target_node)
        "move_node":
            return _move_node_operation(target_node, parameters)
        "set_property":
            return _set_property_operation(target_node, parameters)
        "attach_script":
            return _attach_script_operation(target_node, parameters)
        "add_child":
            return _add_child_operation(target_node, parameters)
        "change_parent":
            return _change_parent_operation(scene_root, target_node, parameters)
        _:
            return {"success": false, "error": "Unknown operation type: " + op_type}

func _create_node_operation(parent: Node, params: Dictionary) -> Dictionary:
    var node_type = params.get("node_type", "Node")
    var node_name = params.get("name", "NewNode")
    var position = params.get("position", Vector2.ZERO)
    
    var new_node = _create_node_by_type(node_type)
    new_node.name = node_name
    
    if new_node is Node2D and position != Vector2.ZERO:
        new_node.position = position
    
    parent.add_child(new_node)
    new_node.owner = parent.get_tree().current_scene
    
    return {"success": true}

func _delete_node_operation(node: Node) -> Dictionary:
    node.queue_free()
    return {"success": true}

func _move_node_operation(node: Node, params: Dictionary) -> Dictionary:
    if node is Node2D:
        var new_position = params.get("position", Vector2.ZERO)
        node.position = new_position
    elif node is Node3D:
        var new_position = params.get("position", Vector3.ZERO)
        node.position = new_position
    
    return {"success": true}

func _set_property_operation(node: Node, params: Dictionary) -> Dictionary:
    var property_name = params.get("property", "")
    var property_value = params.get("value", null)
    
    if property_name.is_empty():
        return {"success": false, "error": "Property name is required"}
    
    if node.has_method("set_" + property_name):
        node.call("set_" + property_name, property_value)
    elif property_name in node:
        node[property_name] = property_value
    else:
        return {"success": false, "error": "Property not found: " + property_name}
    
    return {"success": true}

func _attach_script_operation(node: Node, params: Dictionary) -> Dictionary:
    var script_path = params.get("script_path", "")
    
    if script_path.is_empty():
        return {"success": false, "error": "Script path is required"}
    
    var script_resource = load(script_path)
    if script_resource == null:
        return {"success": false, "error": "Failed to load script: " + script_path}
    
    node.set_script(script_resource)
    return {"success": true}

func _add_child_operation(parent: Node, params: Dictionary) -> Dictionary:
    var child_type = params.get("child_type", "Node")
    var child_name = params.get("child_name", "NewChild")
    
    var child_node = _create_node_by_type(child_type)
    child_node.name = child_name
    
    parent.add_child(child_node)
    child_node.owner = parent.get_tree().current_scene
    
    return {"success": true}

func _change_parent_operation(scene_root: Node, node: Node, params: Dictionary) -> Dictionary:
    var new_parent_path = params.get("new_parent", "")
    
    if new_parent_path.is_empty():
        return {"success": false, "error": "New parent path is required"}
    
    var new_parent = scene_root.get_node_or_null(new_parent_path)
    if new_parent == null:
        return {"success": false, "error": "New parent not found: " + new_parent_path}
    
    var old_parent = node.get_parent()
    if old_parent:
        old_parent.remove_child(node)
    
    new_parent.add_child(node)
    node.owner = scene_root
    
    return {"success": true}

func _create_node_by_type(type_name: String) -> Node:
    # scene_creator_script와 동일한 구현
    match type_name:
        "Node2D":
            return Node2D.new()
        "Node3D":
            return Node3D.new()
        "Control":
            return Control.new()
        "CharacterBody2D":
            return CharacterBody2D.new()
        "CharacterBody3D":
            return CharacterBody3D.new()
        "RigidBody2D":
            return RigidBody2D.new()
        "RigidBody3D":
            return RigidBody3D.new()
        "Area2D":
            return Area2D.new()
        "Area3D":
            return Area3D.new()
        "StaticBody2D":
            return StaticBody2D.new()
        "StaticBody3D":
            return StaticBody3D.new()
        "Sprite2D":
            return Sprite2D.new()
        "Sprite3D":
            return Sprite3D.new()
        "Label":
            return Label.new()
        "Button":
            return Button.new()
        "Timer":
            return Timer.new()
        "AudioStreamPlayer":
            return AudioStreamPlayer.new()
        "AudioStreamPlayer2D":
            return AudioStreamPlayer2D.new()
        "AudioStreamPlayer3D":
            return AudioStreamPlayer3D.new()
        "AnimationPlayer":
            return AnimationPlayer.new()
        "Camera2D":
            return Camera2D.new()
        "Camera3D":
            return Camera3D.new()
        "CollisionShape2D":
            return CollisionShape2D.new()
        "CollisionShape3D":
            return CollisionShape3D.new()
        _:
            return Node.new()

func _output_result(result: Dictionary):
    var temp_file = FileAccess.open("user://editor_script_result.json", FileAccess.WRITE)
    if temp_file:
        temp_file.store_string(JSON.stringify(result))
        temp_file.close()
'''

    def _get_resource_manager_script(self) -> str:
        """리소스 관리 자동화 스크립트"""
        return '''@tool
extends EditorScript

var operations_data: String = "{operations}"

func _run():
    print("AI Resource Manager 실행 시작...")
    
    var operations = JSON.parse_string(operations_data)
    if operations == null or not operations is Array:
        _output_result({"success": false, "error": "Invalid operations data"})
        return
    
    var success_count = 0
    var errors = []
    
    for operation in operations:
        var result = _execute_resource_operation(operation)
        if result.success:
            success_count += 1
        else:
            errors.append(result.error)
    
    var final_result = {
        "success": errors.is_empty(),
        "operations_executed": success_count,
        "total_operations": operations.size(),
        "errors": errors
    }
    
    _output_result(final_result)

func _execute_resource_operation(operation: Dictionary) -> Dictionary:
    var op_type = operation.get("type", "")
    var parameters = operation.get("parameters", {})
    
    match op_type:
        "create_texture":
            return _create_texture_resource(parameters)
        "create_material":
            return _create_material_resource(parameters)
        "create_audio":
            return _create_audio_resource(parameters)
        "optimize_textures":
            return _optimize_textures(parameters)
        "batch_import":
            return _batch_import_resources(parameters)
        "create_animation":
            return _create_animation_resource(parameters)
        "generate_atlas":
            return _generate_texture_atlas(parameters)
        _:
            return {"success": false, "error": "Unknown resource operation: " + op_type}

func _create_texture_resource(params: Dictionary) -> Dictionary:
    var width = params.get("width", 64)
    var height = params.get("height", 64)
    var color = params.get("color", Color.WHITE)
    var save_path = params.get("save_path", "res://assets/generated_texture.png")
    
    var image = Image.create(width, height, false, Image.FORMAT_RGBA8)
    image.fill(color)
    
    var texture = ImageTexture.new()
    texture.create_from_image(image)
    
    var result = ResourceSaver.save(texture, save_path)
    if result == OK:
        EditorInterface.get_resource_filesystem().reimport_files([save_path])
        return {"success": true, "path": save_path}
    else:
        return {"success": false, "error": "Failed to save texture"}

func _create_material_resource(params: Dictionary) -> Dictionary:
    var material_type = params.get("material_type", "StandardMaterial3D")
    var properties = params.get("properties", {})
    var save_path = params.get("save_path", "res://assets/generated_material.tres")
    
    var material
    match material_type:
        "StandardMaterial3D":
            material = StandardMaterial3D.new()
        "CanvasItemMaterial":
            material = CanvasItemMaterial.new()
        "ShaderMaterial":
            material = ShaderMaterial.new()
        _:
            return {"success": false, "error": "Unknown material type: " + material_type}
    
    # 속성 적용
    for property_name in properties:
        var property_value = properties[property_name]
        if material.has_method("set_" + property_name):
            material.call("set_" + property_name, property_value)
        elif property_name in material:
            material[property_name] = property_value
    
    var result = ResourceSaver.save(material, save_path)
    if result == OK:
        return {"success": true, "path": save_path}
    else:
        return {"success": false, "error": "Failed to save material"}

func _create_audio_resource(params: Dictionary) -> Dictionary:
    var audio_type = params.get("audio_type", "sine_wave")
    var frequency = params.get("frequency", 440.0)
    var duration = params.get("duration", 1.0)
    var sample_rate = params.get("sample_rate", 44100)
    var save_path = params.get("save_path", "res://assets/generated_audio.ogg")
    
    # 간단한 사인파 생성
    var frames = int(duration * sample_rate)
    var audio_stream = AudioStreamGenerator.new()
    
    # 실제로는 더 복잡한 오디오 생성 로직 필요
    # 여기서는 기본 구조만 제공
    
    var result = ResourceSaver.save(audio_stream, save_path)
    if result == OK:
        return {"success": true, "path": save_path}
    else:
        return {"success": false, "error": "Failed to save audio"}

func _optimize_textures(params: Dictionary) -> Dictionary:
    var directory = params.get("directory", "res://assets/textures/")
    var max_size = params.get("max_size", 1024)
    var compression = params.get("compression", "lossless")
    
    var dir = DirAccess.open(directory)
    if dir == null:
        return {"success": false, "error": "Directory not found: " + directory}
    
    var files_processed = 0
    dir.list_dir_begin()
    var file_name = dir.get_next()
    
    while file_name != "":
        if file_name.ends_with(".png") or file_name.ends_with(".jpg"):
            var file_path = directory + "/" + file_name
            _optimize_single_texture(file_path, max_size, compression)
            files_processed += 1
        
        file_name = dir.get_next()
    
    return {"success": true, "files_processed": files_processed}

func _optimize_single_texture(file_path: String, max_size: int, compression: String):
    var image = Image.load_from_file(file_path)
    if image == null:
        return
    
    # 크기 조정
    if image.get_width() > max_size or image.get_height() > max_size:
        var aspect_ratio = float(image.get_width()) / float(image.get_height())
        var new_width = max_size
        var new_height = int(max_size / aspect_ratio)
        
        if new_height > max_size:
            new_height = max_size
            new_width = int(max_size * aspect_ratio)
        
        image.resize(new_width, new_height)
    
    # 압축 적용 및 저장
    image.save_png(file_path)

func _batch_import_resources(params: Dictionary) -> Dictionary:
    var source_directory = params.get("source_directory", "")
    var target_directory = params.get("target_directory", "res://assets/imported/")
    var file_types = params.get("file_types", ["png", "jpg", "ogg", "wav"])
    
    if source_directory.is_empty():
        return {"success": false, "error": "Source directory is required"}
    
    var source_dir = DirAccess.open(source_directory)
    if source_dir == null:
        return {"success": false, "error": "Source directory not found"}
    
    # 대상 디렉토리 생성
    DirAccess.make_dir_recursive_absolute(target_directory)
    
    var files_imported = 0
    source_dir.list_dir_begin()
    var file_name = source_dir.get_next()
    
    while file_name != "":
        var file_extension = file_name.get_extension().to_lower()
        if file_extension in file_types:
            var source_path = source_directory + "/" + file_name
            var target_path = target_directory + "/" + file_name
            
            if source_dir.copy(source_path, target_path) == OK:
                files_imported += 1
        
        file_name = source_dir.get_next()
    
    # 임포트 갱신
    EditorInterface.get_resource_filesystem().scan()
    
    return {"success": true, "files_imported": files_imported}

func _create_animation_resource(params: Dictionary) -> Dictionary:
    var animation_name = params.get("name", "new_animation")
    var duration = params.get("duration", 1.0)
    var loop_mode = params.get("loop_mode", Animation.LOOP_LINEAR)
    var save_path = params.get("save_path", "res://assets/animations/" + animation_name + ".tres")
    
    var animation = Animation.new()
    animation.resource_name = animation_name
    animation.length = duration
    animation.loop_mode = loop_mode
    
    # 기본 트랙 추가 (위치 애니메이션)
    var position_track = animation.add_track(Animation.TYPE_POSITION_3D)
    animation.track_set_path(position_track, NodePath("."))
    animation.track_insert_key(position_track, 0.0, Vector3.ZERO)
    animation.track_insert_key(position_track, duration, Vector3(1, 0, 0))
    
    var result = ResourceSaver.save(animation, save_path)
    if result == OK:
        return {"success": true, "path": save_path}
    else:
        return {"success": false, "error": "Failed to save animation"}

func _generate_texture_atlas(params: Dictionary) -> Dictionary:
    var source_textures = params.get("source_textures", [])
    var atlas_size = params.get("atlas_size", 512)
    var save_path = params.get("save_path", "res://assets/atlas.png")
    
    if source_textures.is_empty():
        return {"success": false, "error": "No source textures provided"}
    
    # 간단한 아틀라스 생성 로직
    var atlas_image = Image.create(atlas_size, atlas_size, false, Image.FORMAT_RGBA8)
    atlas_image.fill(Color.TRANSPARENT)
    
    var x_offset = 0
    var y_offset = 0
    var max_height = 0
    
    for texture_path in source_textures:
        var image = Image.load_from_file(texture_path)
        if image == null:
            continue
        
        # 아틀라스에 이미지 배치
        if x_offset + image.get_width() > atlas_size:
            x_offset = 0
            y_offset += max_height
            max_height = 0
        
        if y_offset + image.get_height() > atlas_size:
            break  # 아틀라스 크기 초과
        
        atlas_image.blit_rect(image, Rect2i(0, 0, image.get_width(), image.get_height()), Vector2i(x_offset, y_offset))
        
        x_offset += image.get_width()
        max_height = max(max_height, image.get_height())
    
    var result = atlas_image.save_png(save_path)
    if result == OK:
        return {"success": true, "path": save_path}
    else:
        return {"success": false, "error": "Failed to save atlas"}

func _output_result(result: Dictionary):
    var temp_file = FileAccess.open("user://editor_script_result.json", FileAccess.WRITE)
    if temp_file:
        temp_file.store_string(JSON.stringify(result))
        temp_file.close()
'''

    def _get_project_analyzer_script(self) -> str:
        """프로젝트 분석 자동화 스크립트"""
        return '''@tool
extends EditorScript

func _run():
    print("AI Project Analyzer 실행 시작...")
    
    var analysis = {
        "project_info": _analyze_project_info(),
        "scenes": _analyze_scenes(),
        "scripts": _analyze_scripts(),
        "resources": _analyze_resources(),
        "dependencies": _analyze_dependencies(),
        "performance": _analyze_performance(),
        "quality_metrics": _analyze_code_quality()
    }
    
    _output_result({"success": true, "analysis": analysis})

func _analyze_project_info() -> Dictionary:
    var project_settings = ProjectSettings
    return {
        "name": project_settings.get_setting("application/config/name", "Unknown"),
        "version": project_settings.get_setting("application/config/version", "1.0"),
        "main_scene": project_settings.get_setting("application/run/main_scene", ""),
        "engine_version": Engine.get_version_info(),
        "platform": OS.get_name(),
        "debug_mode": OS.is_debug_build()
    }

func _analyze_scenes() -> Dictionary:
    var scenes = []
    var scene_files = _find_files_by_extension("res://", "tscn")
    
    for scene_path in scene_files:
        var scene_data = _analyze_single_scene(scene_path)
        scenes.append(scene_data)
    
    return {
        "total_scenes": scenes.size(),
        "scenes": scenes
    }

func _analyze_single_scene(scene_path: String) -> Dictionary:
    var scene = load(scene_path)
    if scene == null:
        return {"path": scene_path, "error": "Failed to load scene"}
    
    var instance = scene.instantiate()
    if instance == null:
        return {"path": scene_path, "error": "Failed to instantiate scene"}
    
    var analysis = {
        "path": scene_path,
        "name": instance.name,
        "type": instance.get_class(),
        "node_count": _count_nodes_recursive(instance),
        "script_attached": instance.get_script() != null,
        "children": _analyze_node_structure(instance)
    }
    
    instance.queue_free()
    return analysis

func _count_nodes_recursive(node: Node) -> int:
    var count = 1
    for child in node.get_children():
        count += _count_nodes_recursive(child)
    return count

func _analyze_node_structure(node: Node) -> Array:
    var children = []
    for child in node.get_children():
        children.append({
            "name": child.name,
            "type": child.get_class(),
            "script": child.get_script() != null,
            "child_count": child.get_child_count()
        })
    return children

func _analyze_scripts() -> Dictionary:
    var scripts = []
    var script_files = _find_files_by_extension("res://", "gd")
    
    for script_path in script_files:
        var script_data = _analyze_single_script(script_path)
        scripts.append(script_data)
    
    return {
        "total_scripts": scripts.size(),
        "scripts": scripts
    }

func _analyze_single_script(script_path: String) -> Dictionary:
    var file = FileAccess.open(script_path, FileAccess.READ)
    if file == null:
        return {"path": script_path, "error": "Failed to open script"}
    
    var content = file.get_as_text()
    file.close()
    
    var lines = content.split("\\n")
    var analysis = {
        "path": script_path,
        "line_count": lines.size(),
        "function_count": _count_functions(content),
        "class_name": _extract_class_name(content),
        "extends": _extract_extends(content),
        "exports": _count_exports(content),
        "signals": _count_signals(content),
        "complexity_score": _calculate_complexity(content)
    }
    
    return analysis

func _count_functions(content: String) -> int:
    var func_regex = RegEx.new()
    func_regex.compile("func\\s+\\w+")
    var matches = func_regex.search_all(content)
    return matches.size()

func _extract_class_name(content: String) -> String:
    var class_regex = RegEx.new()
    class_regex.compile("class_name\\s+(\\w+)")
    var result = class_regex.search(content)
    return result.get_string(1) if result else ""

func _extract_extends(content: String) -> String:
    var extends_regex = RegEx.new()
    extends_regex.compile("extends\\s+(\\w+)")
    var result = extends_regex.search(content)
    return result.get_string(1) if result else ""

func _count_exports(content: String) -> int:
    var export_regex = RegEx.new()
    export_regex.compile("@export")
    var matches = export_regex.search_all(content)
    return matches.size()

func _count_signals(content: String) -> int:
    var signal_regex = RegEx.new()
    signal_regex.compile("signal\\s+\\w+")
    var matches = signal_regex.search_all(content)
    return matches.size()

func _calculate_complexity(content: String) -> int:
    # 간단한 복잡도 계산 (if, for, while, match 문 개수)
    var complexity = 0
    var keywords = ["if", "for", "while", "match", "elif"]
    
    for keyword in keywords:
        var regex = RegEx.new()
        regex.compile("\\b" + keyword + "\\b")
        var matches = regex.search_all(content)
        complexity += matches.size()
    
    return complexity

func _analyze_resources() -> Dictionary:
    var resources = {
        "textures": _find_files_by_extension("res://", "png").size() + _find_files_by_extension("res://", "jpg").size(),
        "audio": _find_files_by_extension("res://", "ogg").size() + _find_files_by_extension("res://", "wav").size(),
        "materials": _find_files_by_extension("res://", "tres").size(),
        "animations": _find_files_by_extension("res://", "anim").size(),
        "fonts": _find_files_by_extension("res://", "ttf").size() + _find_files_by_extension("res://", "otf").size()
    }
    
    resources["total"] = resources.textures + resources.audio + resources.materials + resources.animations + resources.fonts
    
    return resources

func _analyze_dependencies() -> Dictionary:
    # 프로젝트 의존성 분석
    var autoloads = ProjectSettings.get_setting("autoload", {})
    var plugins = []
    
    # 플러그인 폴더 확인
    var plugin_dir = DirAccess.open("res://addons/")
    if plugin_dir:
        plugin_dir.list_dir_begin()
        var dir_name = plugin_dir.get_next()
        while dir_name != "":
            if plugin_dir.current_is_dir():
                plugins.append(dir_name)
            dir_name = plugin_dir.get_next()
    
    return {
        "autoloads": autoloads.size(),
        "plugins": plugins.size(),
        "plugin_list": plugins
    }

func _analyze_performance() -> Dictionary:
    # 성능 관련 메트릭
    return {
        "estimated_memory_usage": _estimate_memory_usage(),
        "texture_memory": _estimate_texture_memory(),
        "audio_memory": _estimate_audio_memory(),
        "scene_complexity": _calculate_scene_complexity()
    }

func _estimate_memory_usage() -> int:
    # 대략적인 메모리 사용량 추정 (MB)
    var texture_files = _find_files_by_extension("res://", "png") + _find_files_by_extension("res://", "jpg")
    var audio_files = _find_files_by_extension("res://", "ogg") + _find_files_by_extension("res://", "wav")
    
    # 간단한 추정식
    return texture_files.size() * 2 + audio_files.size() * 5  # MB

func _estimate_texture_memory() -> int:
    var texture_files = _find_files_by_extension("res://", "png") + _find_files_by_extension("res://", "jpg")
    return texture_files.size() * 2  # MB per texture (rough estimate)

func _estimate_audio_memory() -> int:
    var audio_files = _find_files_by_extension("res://", "ogg") + _find_files_by_extension("res://", "wav")
    return audio_files.size() * 5  # MB per audio file (rough estimate)

func _calculate_scene_complexity() -> float:
    var scene_files = _find_files_by_extension("res://", "tscn")
    var total_complexity = 0.0
    
    for scene_path in scene_files:
        var scene = load(scene_path)
        if scene:
            var instance = scene.instantiate()
            if instance:
                total_complexity += _count_nodes_recursive(instance)
                instance.queue_free()
    
    return total_complexity / scene_files.size() if scene_files.size() > 0 else 0.0

func _analyze_code_quality() -> Dictionary:
    var script_files = _find_files_by_extension("res://", "gd")
    var total_lines = 0
    var total_functions = 0
    var total_complexity = 0
    
    for script_path in script_files:
        var file = FileAccess.open(script_path, FileAccess.READ)
        if file:
            var content = file.get_as_text()
            file.close()
            
            total_lines += content.split("\\n").size()
            total_functions += _count_functions(content)
            total_complexity += _calculate_complexity(content)
    
    return {
        "total_lines": total_lines,
        "total_functions": total_functions,
        "average_complexity": float(total_complexity) / script_files.size() if script_files.size() > 0 else 0.0,
        "lines_per_script": float(total_lines) / script_files.size() if script_files.size() > 0 else 0.0,
        "functions_per_script": float(total_functions) / script_files.size() if script_files.size() > 0 else 0.0
    }

func _find_files_by_extension(path: String, extension: String) -> Array:
    var files = []
    var dir = DirAccess.open(path)
    
    if dir:
        dir.list_dir_begin()
        var file_name = dir.get_next()
        
        while file_name != "":
            var full_path = path + "/" + file_name if path != "res://" else path + file_name
            
            if dir.current_is_dir() and not file_name.begins_with("."):
                files.append_array(_find_files_by_extension(full_path, extension))
            elif file_name.ends_with("." + extension):
                files.append(full_path)
            
            file_name = dir.get_next()
    
    return files

func _output_result(result: Dictionary):
    var temp_file = FileAccess.open("user://editor_script_result.json", FileAccess.WRITE)
    if temp_file:
        temp_file.store_string(JSON.stringify(result))
        temp_file.close()
'''

    def _get_automated_builder_script(self) -> str:
        """자동 빌드 스크립트"""
        return '''@tool
extends EditorScript

var export_preset: String = "{export_preset}"
var output_path: String = "{output_path}"
var debug_mode: bool = {debug_mode}

func _run():
    print("AI Automated Builder 실행 시작...")
    
    var result = _build_project()
    _output_result(result)

func _build_project() -> Dictionary:
    var export_presets = EditorInterface.get_export_presets()
    var target_preset = null
    
    # 지정된 프리셋 찾기
    for preset in export_presets:
        if preset.get_name() == export_preset:
            target_preset = preset
            break
    
    if target_preset == null:
        return {"success": false, "error": "Export preset not found: " + export_preset}
    
    # 출력 디렉토리 생성
    DirAccess.make_dir_recursive_absolute(output_path)
    
    # 빌드 실행
    var export_path = output_path + "/" + ProjectSettings.get_setting("application/config/name", "game")
    
    # 플랫폼별 확장자 추가
    match export_preset:
        "Windows Desktop":
            export_path += ".exe"
        "Linux/X11":
            export_path += ".x86_64"
        "macOS":
            export_path += ".app"
        "Android":
            export_path += ".apk"
        "iOS":
            export_path += ".ipa"
        "Web":
            export_path += ".html"
    
    # 실제 export 실행 (Godot 4.x API 사용)
    var error = EditorInterface.export_project(target_preset, export_path, debug_mode)
    
    if error == OK:
        return {
            "success": true,
            "export_path": export_path,
            "preset": export_preset,
            "debug_mode": debug_mode,
            "file_size": _get_file_size(export_path)
        }
    else:
        return {
            "success": false,
            "error": "Export failed with error code: " + str(error),
            "preset": export_preset
        }

func _get_file_size(file_path: String) -> int:
    var file = FileAccess.open(file_path, FileAccess.READ)
    if file:
        var size = file.get_length()
        file.close()
        return size
    return 0

func _output_result(result: Dictionary):
    var temp_file = FileAccess.open("user://editor_script_result.json", FileAccess.WRITE)
    if temp_file:
        temp_file.store_string(JSON.stringify(result))
        temp_file.close()
'''

    def _inject_parameters(self, script_content: str, parameters: Dict[str, Any]) -> str:
        """스크립트에 매개변수 주입"""
        injected_script = script_content
        
        for key, value in parameters.items():
            placeholder = "{" + key + "}"
            if isinstance(value, str):
                injected_script = injected_script.replace(placeholder, value)
            elif isinstance(value, bool):
                injected_script = injected_script.replace(placeholder, "true" if value else "false")
            else:
                injected_script = injected_script.replace(placeholder, str(value))
                
        return injected_script
        
    async def _create_temp_script(self, script_content: str) -> Path:
        """임시 스크립트 파일 생성"""
        temp_dir = Path(tempfile.gettempdir())
        script_path = temp_dir / f"godot_ai_script_{os.getpid()}.gd"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        return script_path
        
    async def _run_editor_script(self, project_path: str, script_path: Path) -> Dict[str, Any]:
        """Editor 스크립트 실행"""
        cmd = [
            self.godot_path,
            "--headless",
            "--path", project_path,
            "--script", str(script_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # 결과 파일에서 읽기
            result_file = Path(project_path) / ".godot" / "editor_script_result.json"
            if not result_file.exists():
                # user:// 경로에서 시도
                user_data_dir = Path.home() / ".local" / "share" / "godot" / "app_userdata"
                result_file = user_data_dir / "editor_script_result.json"
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result = json.load(f)
                result_file.unlink()  # 임시 파일 삭제
                return result
            else:
                return {
                    "success": False,
                    "error": "No result file found",
                    "stdout": stdout.decode() if stdout else "",
                    "stderr": stderr.decode() if stderr else ""
                }
                
        except Exception as e:
            self.logger.error(f"Editor 스크립트 실행 실패: {e}")
            return {"success": False, "error": str(e)}

# 추가 유틸리티 함수들
class GodotProjectAutomator:
    """Godot 프로젝트 완전 자동화 클래스"""
    
    def __init__(self, godot_path: str = None):
        self.api = GodotEditorAPI(godot_path)
        self.logger = logging.getLogger("GodotProjectAutomator")
        
    async def fully_automate_project(self, project_path: str, automation_config: Dict[str, Any]) -> bool:
        """프로젝트 완전 자동화"""
        try:
            self.logger.info("Godot 프로젝트 완전 자동화 시작")
            
            # 1. 프로젝트 분석
            analysis = await self.api.analyze_project_structure(project_path)
            self.logger.info(f"프로젝트 분석 완료: {analysis.get('project_info', {}).get('name', 'Unknown')}")
            
            # 2. 씬 자동 생성
            scenes_config = automation_config.get("scenes", [])
            for scene_config in scenes_config:
                success = await self.api.create_scene_automatically(project_path, scene_config)
                if success:
                    self.logger.info(f"씬 생성 성공: {scene_config.get('name')}")
                    
            # 3. 노드 자동 조작
            node_operations = automation_config.get("node_operations", [])
            if node_operations:
                success = await self.api.manipulate_nodes_batch(project_path, node_operations)
                if success:
                    self.logger.info("노드 조작 완료")
                    
            # 4. 리소스 자동 관리
            resource_operations = automation_config.get("resource_operations", [])
            if resource_operations:
                success = await self.api.manage_resources_automatically(project_path, resource_operations)
                if success:
                    self.logger.info("리소스 관리 완료")
                    
            # 5. 자동 빌드
            build_config = automation_config.get("build", {})
            if build_config:
                success = await self.api.build_project_automatically(project_path, build_config)
                if success:
                    self.logger.info("프로젝트 빌드 완료")
                    
            self.logger.info("Godot 프로젝트 완전 자동화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"프로젝트 자동화 실패: {e}")
            return False

def main():
    """메인 실행 함수"""
    print("🤖 Godot Editor API 자동화 시스템")
    print("=" * 60)
    
    # 사용 예제
    api = GodotEditorAPI()
    automator = GodotProjectAutomator()
    
    async def test_automation():
        project_path = "/tmp/test_project"
        
        # 씬 자동 생성 테스트
        scene_config = {
            "name": "TestScene",
            "type": "Node2D",
            "path": "res://scenes/",
            "nodes": [
                {
                    "type": "CharacterBody2D",
                    "name": "Player",
                    "position": [100, 100],
                    "children": [
                        {"type": "Sprite2D", "name": "Sprite"},
                        {"type": "CollisionShape2D", "name": "Collision"}
                    ]
                }
            ]
        }
        
        success = await api.create_scene_automatically(project_path, scene_config)
        print(f"씬 생성 결과: {success}")
        
        # 노드 조작 테스트
        operations = [
            {
                "type": "create_node",
                "parameters": {
                    "node_type": "Label",
                    "name": "ScoreLabel",
                    "position": [10, 10]
                }
            },
            {
                "type": "set_property",
                "target": "ScoreLabel",
                "parameters": {
                    "property": "text",
                    "value": "Score: 0"
                }
            }
        ]
        
        success = await api.manipulate_nodes_batch(project_path, operations)
        print(f"노드 조작 결과: {success}")
        
    # 비동기 실행
    asyncio.run(test_automation())

if __name__ == "__main__":
    main()