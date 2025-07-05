@tool
extends EditorPlugin

# AutoCI Scene Automation Plugin
# 씬 자동화 전용 플러그인

var scene_composer
var layout_optimizer

func _enter_tree():
    print("AutoCI Scene Automation 플러그인 활성화")
    
    # 씬 컴포저 초기화
    scene_composer = preload("res://addons/scene_automation/auto_composer.gd").new()
    layout_optimizer = preload("res://addons/scene_automation/layout_optimizer.gd").new()
    
    add_child(scene_composer)
    add_child(layout_optimizer)
    
    # 컨텍스트 메뉴 추가
    add_tool_submenu_item("AI Scene Tools", _create_scene_menu())

func _exit_tree():
    print("AutoCI Scene Automation 플러그인 비활성화")
    
    if scene_composer:
        scene_composer.queue_free()
    if layout_optimizer:
        layout_optimizer.queue_free()

func _create_scene_menu():
    var menu = PopupMenu.new()
    menu.add_item("AI 씬 구성", 0)
    menu.add_item("레이아웃 최적화", 1)
    menu.add_item("노드 자동 배치", 2)
    
    menu.id_pressed.connect(_on_scene_menu_pressed)
    return menu

func _on_scene_menu_pressed(id: int):
    match id:
        0:
            scene_composer.compose_intelligent_scene()
        1:
            layout_optimizer.optimize_current_layout()
        2:
            scene_composer.auto_place_nodes()

func get_plugin_name():
    return "AutoCI Scene Automation"
