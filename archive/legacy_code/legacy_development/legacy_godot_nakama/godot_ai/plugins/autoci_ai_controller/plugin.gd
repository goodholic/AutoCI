@tool
extends EditorPlugin

# AutoCI AI Controller Plugin
# AI 자동 제어 핵심 플러그인

var ai_controller
var automation_dock

func _enter_tree():
    print("AutoCI AI Controller 플러그인 활성화")
    
    # AI 컨트롤러 초기화
    ai_controller = preload("res://addons/autoci_ai/ai_controller.gd").new()
    add_child(ai_controller)
    
    # 자동화 독 추가
    automation_dock = preload("res://addons/autoci_ai/automation_dock.tscn").instantiate()
    add_control_to_dock(DOCK_SLOT_LEFT_UR, automation_dock)
    
    # 메뉴 항목 추가
    add_tool_menu_item("AI 프로젝트 생성", _on_ai_create_project)
    add_tool_menu_item("AI 씬 최적화", _on_ai_optimize_scene)
    add_tool_menu_item("AI 리소스 생성", _on_ai_generate_resources)
    
    # 자동화 시작
    ai_controller.start_automation()

func _exit_tree():
    print("AutoCI AI Controller 플러그인 비활성화")
    
    # 정리
    if automation_dock:
        remove_control_from_docks(automation_dock)
        automation_dock.queue_free()
        
    remove_tool_menu_item("AI 프로젝트 생성")
    remove_tool_menu_item("AI 씬 최적화")
    remove_tool_menu_item("AI 리소스 생성")
    
    if ai_controller:
        ai_controller.stop_automation()
        ai_controller.queue_free()

func _on_ai_create_project():
    """AI 프로젝트 생성"""
    if ai_controller:
        ai_controller.create_ai_project()

func _on_ai_optimize_scene():
    """AI 씬 최적화"""
    if ai_controller:
        ai_controller.optimize_current_scene()

func _on_ai_generate_resources():
    """AI 리소스 생성"""
    if ai_controller:
        ai_controller.generate_game_resources()

func has_main_screen():
    return false

func get_plugin_name():
    return "AutoCI AI Controller"
