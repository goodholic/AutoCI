extends Node2D

# RPG 월드 관리자
var current_map: String = "town"
var spawn_points: Dictionary = {}

func _ready() -> void:
	load_map(current_map)

func load_map(map_name: String) -> void:
	print("🗺️ 맵 로드: ", map_name)
	current_map = map_name
	
	# 맵별 설정
	match map_name:
		"town":
			setup_town()
		"dungeon":
			setup_dungeon()
		"forest":
			setup_forest()

func setup_town() -> void:
	# 마을 설정
	spawn_npcs([
		{"name": "상인", "pos": Vector2(300, 400)},
		{"name": "대장장이", "pos": Vector2(500, 400)},
		{"name": "마법사", "pos": Vector2(700, 400)}
	])

func setup_dungeon() -> void:
	# 던전 설정
	spawn_enemies([
		{"type": "슬라임", "pos": Vector2(200, 300)},
		{"type": "고블린", "pos": Vector2(400, 300)},
		{"type": "스켈레톤", "pos": Vector2(600, 300)}
	])

func setup_forest() -> void:
	# 숲 설정
	spawn_items([
		{"type": "포션", "pos": Vector2(250, 350)},
		{"type": "금화", "pos": Vector2(450, 350)}
	])

func spawn_npcs(npc_list: Array) -> void:
	for npc_data in npc_list:
		print("👤 NPC 생성: ", npc_data.name)

func spawn_enemies(enemy_list: Array) -> void:
	for enemy_data in enemy_list:
		print("👹 적 생성: ", enemy_data.type)

func spawn_items(item_list: Array) -> void:
	for item_data in item_list:
		print("💎 아이템 생성: ", item_data.type)
