@tool
extends Node

# AutoCI AI Controller Core
# Python AI 시스템과 Godot 간의 핵심 제어 인터페이스

signal ai_command_received(command: String, params: Dictionary)
signal automation_status_changed(status: String)

var python_bridge
var automation_active = false
var project_analyzer

func _ready():
    print("AI Controller 초기화 중...")
    setup_python_bridge()
    setup_project_analyzer()

func setup_python_bridge():
    """Python 브리지 설정"""
    python_bridge = preload("res://addons/autoci_ai/python_bridge.gd").new()
    add_child(python_bridge)
    python_bridge.connect("command_received", _on_python_command)

func setup_project_analyzer():
    """프로젝트 분석기 설정"""
    project_analyzer = preload("res://addons/autoci_ai/project_analyzer.gd").new()
    add_child(project_analyzer)

func start_automation():
    """자동화 시작"""
    automation_active = true
    automation_status_changed.emit("started")
    print("AI 자동화 시작됨")

func stop_automation():
    """자동화 중지"""
    automation_active = false
    automation_status_changed.emit("stopped")
    print("AI 자동화 중지됨")

func create_ai_project():
    """AI 프로젝트 생성"""
    var project_data = {
        "game_type": "platformer",
        "ai_optimized": true,
        "auto_generation": true
    }
    
    if python_bridge:
        python_bridge.send_command("create_project", project_data)

func optimize_current_scene():
    """현재 씬 최적화"""
    var current_scene = EditorInterface.get_edited_scene_root()
    if current_scene:
        var scene_data = project_analyzer.analyze_scene(current_scene)
        python_bridge.send_command("optimize_scene", scene_data)

func generate_game_resources():
    """게임 리소스 생성"""
    var resource_request = {
        "type": "complete_set",
        "game_type": "platformer",
        "quality": "high"
    }
    
    python_bridge.send_command("generate_resources", resource_request)

func _on_python_command(command: String, params: Dictionary):
    """Python 명령 처리"""
    match command:
        "create_scene":
            _create_scene_from_data(params)
        "add_node":
            _add_node_to_scene(params)
        "modify_properties":
            _modify_node_properties(params)
        "generate_script":
            _generate_node_script(params)

func _create_scene_from_data(data: Dictionary):
    """데이터로부터 씬 생성"""
    var scene = PackedScene.new()
    var root_node = _create_node_from_data(data.get("root", {}))
    
    scene.pack(root_node)
    
    var scene_path = "res://scenes/" + data.get("name", "ai_generated") + ".tscn"
    ResourceSaver.save(scene, scene_path)
    
    print("AI 씬 생성 완료: ", scene_path)

func _create_node_from_data(node_data: Dictionary) -> Node:
    """데이터로부터 노드 생성"""
    var node_type = node_data.get("type", "Node2D")
    var node = ClassDB.instantiate(node_type)
    
    # 기본 속성 설정
    node.name = node_data.get("name", "AINode")
    
    # 위치, 스케일, 회전 설정
    if node.has_method("set_position") and node_data.has("position"):
        var pos = node_data["position"]
        node.set_position(Vector2(pos[0], pos[1]))
    
    # 자식 노드들 생성
    for child_data in node_data.get("children", []):
        var child = _create_node_from_data(child_data)
        node.add_child(child)
    
    return node

func _add_node_to_scene(params: Dictionary):
    """씬에 노드 추가"""
    var current_scene = EditorInterface.get_edited_scene_root()
    if current_scene:
        var new_node = _create_node_from_data(params)
        current_scene.add_child(new_node)
        new_node.set_owner(current_scene)

func _modify_node_properties(params: Dictionary):
    """노드 속성 수정"""
    var node_path = params.get("path", "")
    var properties = params.get("properties", {})
    
    var current_scene = EditorInterface.get_edited_scene_root()
    if current_scene:
        var target_node = current_scene.get_node(node_path)
        if target_node:
            for prop in properties:
                target_node.set(prop, properties[prop])

func _generate_node_script(params: Dictionary):
    """노드 스크립트 생성"""
    var script_content = params.get("content", "")
    var script_path = params.get("path", "res://scripts/ai_generated.gd")
    
    var script = GDScript.new()
    script.source_code = script_content
    
    ResourceSaver.save(script, script_path)
    print("AI 스크립트 생성 완료: ", script_path)
