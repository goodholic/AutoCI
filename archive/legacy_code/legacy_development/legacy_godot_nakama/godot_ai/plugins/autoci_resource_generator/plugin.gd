@tool
extends EditorPlugin

# AutoCI Resource Generator Plugin
# 리소스 자동 생성 플러그인

var texture_gen
var audio_gen
var material_gen

func _enter_tree():
    print("AutoCI Resource Generator 플러그인 활성화")
    
    # 생성기 초기화
    texture_gen = preload("res://addons/resource_gen/texture_generator.gd").new()
    audio_gen = preload("res://addons/resource_gen/audio_generator.gd").new()
    material_gen = preload("res://addons/resource_gen/material_generator.gd").new()
    
    add_child(texture_gen)
    add_child(audio_gen)
    add_child(material_gen)
    
    # 리소스 생성 메뉴 추가
    add_tool_submenu_item("AI Resource Generator", _create_resource_menu())

func _exit_tree():
    print("AutoCI Resource Generator 플러그인 비활성화")
    
    if texture_gen:
        texture_gen.queue_free()
    if audio_gen:
        audio_gen.queue_free()
    if material_gen:
        material_gen.queue_free()

func _create_resource_menu():
    var menu = PopupMenu.new()
    menu.add_item("AI 텍스처 생성", 0)
    menu.add_item("AI 오디오 생성", 1)
    menu.add_item("AI 머티리얼 생성", 2)
    menu.add_item("배치 리소스 생성", 3)
    
    menu.id_pressed.connect(_on_resource_menu_pressed)
    return menu

func _on_resource_menu_pressed(id: int):
    match id:
        0:
            texture_gen.generate_procedural_textures()
        1:
            audio_gen.generate_game_audio()
        2:
            material_gen.generate_materials()
        3:
            _batch_generate_resources()

func _batch_generate_resources():
    """배치 리소스 생성"""
    texture_gen.generate_procedural_textures()
    audio_gen.generate_game_audio()
    material_gen.generate_materials()

func get_plugin_name():
    return "AutoCI Resource Generator"
