@tool
extends Node

# AutoCI Scene Generator
# AI 기반 씬 자동 생성

func generate_platformer_scene(scene_name: String) -> PackedScene:
    """플랫포머 씬 생성"""
    var scene = PackedScene.new()
    var root = Node2D.new()
    root.name = scene_name
    
    # 플레이어 추가
    var player = CharacterBody2D.new()
    player.name = "Player"
    player.position = Vector2(100, 400)
    root.add_child(player)
    
    # 플랫폼 추가
    for i in range(5):
        var platform = StaticBody2D.new()
        platform.name = "Platform" + str(i)
        platform.position = Vector2(200 + i * 300, 500 + randf_range(-50, 50))
        root.add_child(platform)
    
    scene.pack(root)
    return scene

func generate_racing_scene(scene_name: String) -> PackedScene:
    """레이싱 씬 생성"""
    var scene = PackedScene.new()
    var root = Node3D.new()
    root.name = scene_name
    
    # 차량 추가
    var vehicle = RigidBody3D.new()
    vehicle.name = "Vehicle"
    vehicle.position = Vector3(0, 1, 0)
    root.add_child(vehicle)
    
    # 트랙 요소들 추가
    for i in range(10):
        var track_piece = StaticBody3D.new()
        track_piece.name = "TrackPiece" + str(i)
        track_piece.position = Vector3(i * 10, 0, 0)
        root.add_child(track_piece)
    
    scene.pack(root)
    return scene
