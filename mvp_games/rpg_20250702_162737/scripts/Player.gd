extends CharacterBody2D

# 플레이어 설정
const SPEED = 300.0
const JUMP_VELOCITY = -400.0

# 중력 설정
var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

func _physics_process(delta):
    # 중력 적용
    if not is_on_floor():
        velocity.y += gravity * delta
    
    # 점프 처리
    if Input.is_action_just_pressed("jump") and is_on_floor():
        velocity.y = JUMP_VELOCITY
    
    # 좌우 이동 처리
    var direction = Input.get_axis("move_left", "move_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = move_toward(velocity.x, 0, SPEED)
    
    # 물리 이동 적용
    move_and_slide()
    
    # 화면 밖으로 나가지 않도록
    position.x = clamp(position.x, 32, 992)
    
    # 떨어지면 리스폰
    if position.y > 700:
        position = Vector2(100, 300)
        velocity = Vector2.ZERO

func _ready():
    print("Player ready!")
