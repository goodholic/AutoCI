@tool
extends Node

# AutoCI Auto Composer
# 지능형 씬 자동 구성

enum PlacementStrategy {
    RANDOM,
    GRID,
    ORGANIC,
    BALANCED,
    GUIDED
}

func compose_scene_intelligently(game_type: String, elements: Array, strategy: PlacementStrategy) -> Node:
    """지능형 씬 구성"""
    var root = Node2D.new()
    root.name = "AI_ComposedScene"
    
    match strategy:
        PlacementStrategy.RANDOM:
            _place_elements_randomly(root, elements)
        PlacementStrategy.GRID:
            _place_elements_in_grid(root, elements)
        PlacementStrategy.ORGANIC:
            _place_elements_organically(root, elements)
        PlacementStrategy.BALANCED:
            _place_elements_balanced(root, elements)
        PlacementStrategy.GUIDED:
            _place_elements_guided(root, elements, game_type)
    
    return root

func _place_elements_randomly(root: Node, elements: Array):
    """무작위 배치"""
    for element_data in elements:
        var node = _create_element(element_data)
        node.position = Vector2(
            randf_range(0, 1920),
            randf_range(0, 1080)
        )
        root.add_child(node)

func _place_elements_in_grid(root: Node, elements: Array):
    """격자 배치"""
    var grid_size = int(sqrt(elements.size())) + 1
    var cell_width = 1920.0 / grid_size
    var cell_height = 1080.0 / grid_size
    
    for i in range(elements.size()):
        var element_data = elements[i]
        var node = _create_element(element_data)
        
        var grid_x = i % grid_size
        var grid_y = i / grid_size
        
        node.position = Vector2(
            grid_x * cell_width + cell_width / 2,
            grid_y * cell_height + cell_height / 2
        )
        root.add_child(node)

func _place_elements_organically(root: Node, elements: Array):
    """유기적 배치"""
    var placed_positions = []
    var min_distance = 100.0
    
    for element_data in elements:
        var node = _create_element(element_data)
        var position = _find_organic_position(placed_positions, min_distance)
        
        node.position = position
        placed_positions.append(position)
        root.add_child(node)

func _place_elements_balanced(root: Node, elements: Array):
    """균형 배치"""
    # 화면을 4개 구역으로 나누어 균등 배치
    var zones = [
        Rect2(0, 0, 960, 540),      # 좌상
        Rect2(960, 0, 960, 540),    # 우상
        Rect2(0, 540, 960, 540),    # 좌하
        Rect2(960, 540, 960, 540)   # 우하
    ]
    
    for i in range(elements.size()):
        var element_data = elements[i]
        var node = _create_element(element_data)
        var zone = zones[i % 4]
        
        node.position = Vector2(
            zone.position.x + randf() * zone.size.x,
            zone.position.y + randf() * zone.size.y
        )
        root.add_child(node)

func _place_elements_guided(root: Node, elements: Array, game_type: String):
    """가이드된 배치"""
    for element_data in elements:
        var node = _create_element(element_data)
        var position = _get_guided_position(element_data, game_type)
        
        node.position = position
        root.add_child(node)

func _create_element(element_data: Dictionary) -> Node:
    """요소 생성"""
    var node_type = element_data.get("type", "Node2D")
    var node = ClassDB.instantiate(node_type)
    node.name = element_data.get("name", "Element")
    return node

func _find_organic_position(existing_positions: Array, min_distance: float) -> Vector2:
    """유기적 위치 찾기"""
    var max_attempts = 50
    
    for attempt in range(max_attempts):
        var pos = Vector2(randf_range(50, 1870), randf_range(50, 1030))
        var valid = true
        
        for existing_pos in existing_positions:
            if pos.distance_to(existing_pos) < min_distance:
                valid = false
                break
        
        if valid:
            return pos
    
    # 실패시 랜덤 위치 반환
    return Vector2(randf_range(50, 1870), randf_range(50, 1030))

func _get_guided_position(element_data: Dictionary, game_type: String) -> Vector2:
    """가이드된 위치 계산"""
    var element_type = element_data.get("type", "")
    
    match game_type:
        "platformer":
            return _get_platformer_position(element_type)
        "racing":
            return _get_racing_position(element_type)
        "puzzle":
            return _get_puzzle_position(element_type)
        _:
            return Vector2(960, 540)  # 중앙

func _get_platformer_position(element_type: String) -> Vector2:
    """플랫포머 요소 위치"""
    match element_type:
        "Player":
            return Vector2(100, 400)
        "Enemy":
            return Vector2(randf_range(500, 1500), randf_range(300, 600))
        "Platform":
            return Vector2(randf_range(200, 1800), randf_range(400, 800))
        _:
            return Vector2(randf_range(100, 1820), randf_range(100, 980))

func _get_racing_position(element_type: String) -> Vector2:
    """레이싱 요소 위치"""
    match element_type:
        "Vehicle":
            return Vector2(100, 500)
        "Checkpoint":
            return Vector2(randf_range(300, 1600), randf_range(400, 600))
        _:
            return Vector2(randf_range(100, 1820), randf_range(100, 980))

func _get_puzzle_position(element_type: String) -> Vector2:
    """퍼즐 요소 위치"""
    # 격자 기반 배치
    var grid_x = randi() % 8
    var grid_y = randi() % 6
    return Vector2(grid_x * 240 + 120, grid_y * 180 + 90)
