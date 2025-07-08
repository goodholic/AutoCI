extends CharacterBody2D

# RPG 플레이어 스크립트
signal died
signal health_changed(new_health)
signal level_up(new_level)

# 스탯
var max_health: int = 100
var current_health: int = 100
var attack_power: int = 10
var defense: int = 5
var level: int = 1
var experience: int = 0
var exp_to_next_level: int = 100

# 이동
const SPEED = 200.0

# 상태
var is_attacking: bool = false
var is_dead: bool = false
var invulnerable: bool = false

func _ready() -> void:
	current_health = max_health
	add_to_group("player")

func _physics_process(delta: float) -> void:
	if is_dead:
		return
		
	handle_movement()
	handle_actions()
	
func handle_movement() -> void:
	var input_vector = Vector2.ZERO
	
	input_vector.x = Input.get_axis("ui_left", "ui_right")
	input_vector.y = Input.get_axis("ui_up", "ui_down")
	
	if input_vector.length() > 0:
		velocity = input_vector.normalized() * SPEED
	else:
		velocity = velocity.move_toward(Vector2.ZERO, SPEED * 0.1)
	
	move_and_slide()

func handle_actions() -> void:
	if Input.is_action_just_pressed("ui_accept") and not is_attacking:
		attack()

func attack() -> void:
	is_attacking = true
	print("⚔️ 공격!")
	
	# 공격 애니메이션과 판정
	await get_tree().create_timer(0.3).timeout
	
	# 주변 적에게 데미지
	var enemies = get_tree().get_nodes_in_group("enemies")
	for enemy in enemies:
		if enemy.global_position.distance_to(global_position) < 50:
			if enemy.has_method("take_damage"):
				enemy.take_damage(attack_power)
	
	is_attacking = false

func take_damage(amount: int) -> void:
	if invulnerable or is_dead:
		return
		
	current_health -= max(0, amount - defense)
	emit_signal("health_changed", current_health)
	
	print("💔 데미지: ", amount, " 현재 체력: ", current_health)
	
	if current_health <= 0:
		die()
	else:
		# 무적 시간
		invulnerable = true
		modulate = Color(1, 0.5, 0.5, 0.5)
		await get_tree().create_timer(1.0).timeout
		modulate = Color.WHITE
		invulnerable = false

func die() -> void:
	is_dead = true
	emit_signal("died")
	print("💀 플레이어 사망!")
	
	# 사망 애니메이션
	var tween = create_tween()
	tween.tween_property(self, "modulate:a", 0, 1.0)
	tween.tween_callback(queue_free)

func heal(amount: int) -> void:
	current_health = min(current_health + amount, max_health)
	emit_signal("health_changed", current_health)
	print("💚 회복: ", amount)

func gain_experience(amount: int) -> void:
	experience += amount
	print("✨ 경험치 획득: ", amount)
	
	while experience >= exp_to_next_level:
		level_up()

func level_up() -> void:
	level += 1
	experience -= exp_to_next_level
	exp_to_next_level = int(exp_to_next_level * 1.5)
	
	# 스탯 상승
	max_health += 20
	current_health = max_health
	attack_power += 5
	defense += 2
	
	emit_signal("level_up", level)
	emit_signal("health_changed", current_health)
	print("🎉 레벨 업! 레벨: ", level)
