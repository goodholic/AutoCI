extends Control

# UI 관리자
@onready var health_bar = $HealthBar
@onready var score_label = $ScoreLabel
@onready var game_over_panel = $GameOverPanel
@onready var restart_button = $GameOverPanel/RestartButton

var score: int = 0

func _ready() -> void:
	# 시그널 연결
	restart_button.pressed.connect(_on_restart_pressed)
	
	# 메인 씬 시그널 연결
	var main = get_node_or_null("/root/Main")
	if main:
		if main.has_signal("game_over"):
			main.game_over.connect(_on_game_over)
	
	# 플레이어 시그널 연결
	var player = get_node_or_null("/root/Main/Player")
	if player:
		if player.has_signal("health_changed"):
			player.health_changed.connect(update_health)
	
	# 초기화
	game_over_panel.visible = false
	update_score(0)

func update_health(value: int) -> void:
	health_bar.value = value
	
	# 체력바 색상 변경
	if value < 30:
		health_bar.modulate = Color.RED
	elif value < 60:
		health_bar.modulate = Color.YELLOW
	else:
		health_bar.modulate = Color.GREEN

func update_score(value: int) -> void:
	score = value
	score_label.text = "Score: " + str(score)

func add_score(points: int) -> void:
	update_score(score + points)

func _on_game_over() -> void:
	game_over_panel.visible = true
	print("💀 게임 오버 UI 표시")

func _on_restart_pressed() -> void:
	get_tree().reload_current_scene()

func show_message(text: String, duration: float = 2.0) -> void:
	# 임시 메시지 표시
	var label = Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 32)
	label.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
	add_child(label)
	
	await get_tree().create_timer(duration).timeout
	label.queue_free()
