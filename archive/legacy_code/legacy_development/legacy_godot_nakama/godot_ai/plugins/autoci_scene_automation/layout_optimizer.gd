@tool
extends Node

# AutoCI Layout Optimizer
# 레이아웃 최적화 시스템

func optimize_scene_layout(scene_root: Node) -> void:
    """씬 레이아웃 최적화"""
    if not scene_root:
        return
    
    _resolve_overlaps(scene_root)
    _optimize_performance(scene_root)
    _ensure_accessibility(scene_root)

func _resolve_overlaps(scene_root: Node) -> void:
    """겹침 해결"""
    var nodes_with_positions = []
    _collect_positioned_nodes(scene_root, nodes_with_positions)
    
    var min_distance = 30.0
    
    for i in range(nodes_with_positions.size()):
        for j in range(i + 1, nodes_with_positions.size()):
            var node1 = nodes_with_positions[i]
            var node2 = nodes_with_positions[j]
            
            if not node1.has_method("get_position") or not node2.has_method("get_position"):
                continue
                
            var distance = node1.get_position().distance_to(node2.get_position())
            
            if distance < min_distance:
                _separate_nodes(node1, node2, min_distance)

func _optimize_performance(scene_root: Node) -> void:
    """성능 최적화"""
    var node_count = _count_all_nodes(scene_root)
    var max_nodes = 100
    
    if node_count > max_nodes:
        print("경고: 노드 수가 많습니다 (", node_count, "/", max_nodes, ")")
        _suggest_optimizations(scene_root)

func _ensure_accessibility(scene_root: Node) -> void:
    """접근성 확인"""
    var positioned_nodes = []
    _collect_positioned_nodes(scene_root, positioned_nodes)
    
    var center = Vector2(960, 540)
    var max_distance = 800
    
    for node in positioned_nodes:
        if node.has_method("get_position"):
            var distance = node.get_position().distance_to(center)
            if distance > max_distance:
                _move_node_closer(node, center, max_distance)

func _collect_positioned_nodes(node: Node, collection: Array) -> void:
    """위치가 있는 노드들 수집"""
    if node.has_method("get_position"):
        collection.append(node)
    
    for child in node.get_children():
        _collect_positioned_nodes(child, collection)

func _count_all_nodes(node: Node) -> int:
    """모든 노드 수 계산"""
    var count = 1
    for child in node.get_children():
        count += _count_all_nodes(child)
    return count

func _separate_nodes(node1: Node, node2: Node, min_distance: float) -> void:
    """노드 분리"""
    var pos1 = node1.get_position()
    var pos2 = node2.get_position()
    
    var direction = (pos2 - pos1).normalized()
    var new_pos2 = pos1 + direction * min_distance
    
    if node2.has_method("set_position"):
        node2.set_position(new_pos2)

func _move_node_closer(node: Node, center: Vector2, max_distance: float) -> void:
    """노드를 중앙에 가깝게 이동"""
    var current_pos = node.get_position()
    var direction = (current_pos - center).normalized()
    var new_pos = center + direction * max_distance
    
    if node.has_method("set_position"):
        node.set_position(new_pos)

func _suggest_optimizations(scene_root: Node) -> void:
    """최적화 제안"""
    print("최적화 제안:")
    print("- 불필요한 노드 제거")
    print("- 노드 그룹화 고려")
    print("- 오브젝트 풀링 사용")
