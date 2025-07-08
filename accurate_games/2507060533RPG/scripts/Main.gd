extends Node2D

# RPG 게임 메인 스크립트
# AI가 정확하게 제작한 게임입니다

signal game_started
signal game_over
signal game_paused

var game_started: bool = false
var score: int = 0
var game_time: float = 0.0
var is_paused: bool = false

func _ready() -> void:
	print("🎮 게임이 준비되었습니다!")
	_initialize_game()
	
func _initialize_game() -> void:
	# 게임 초기화
	game_started = false
	score = 0
	game_time = 0.0
	is_paused = false
	
	# 시그널 연결
	_connect_signals()
	
	# 게임 시작
	call_deferred("start_game")

func _connect_signals() -> void:
	# 플레이어 시그널 연결
	if has_node("Player"):
		var player = $Player
		if player.has_signal("died"):
			player.died.connect(_on_player_died)
	
func start_game() -> void:
	game_started = true
	emit_signal("game_started")
	print("🎮 게임 시작!")

func _process(delta: float) -> void:
	if game_started and not is_paused:
		game_time += delta
		_update_game_logic(delta)

func _update_game_logic(delta: float) -> void:
	# 게임 로직 업데이트
	pass

func _input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		toggle_pause()

func toggle_pause() -> void:
	is_paused = !is_paused
	get_tree().paused = is_paused
	emit_signal("game_paused")

func _on_player_died() -> void:
	game_started = false
	emit_signal("game_over")
	print("💀 게임 오버!")
	
	# 3초 후 재시작
	await get_tree().create_timer(3.0).timeout
	restart_game()

func restart_game() -> void:
	get_tree().reload_current_scene()

func add_score(points: int) -> void:
	score += points
	print("점수: ", score)
