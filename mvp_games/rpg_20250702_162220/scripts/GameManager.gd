extends Node2D

# 게임 상태
var game_started = false
var game_paused = false

func _ready():
    print("===== MVP Game Started =====")
    print("Game Manager initialized")
    game_started = true
    
    # 게임 정보 출력
    print("Game: Simple Platform Game")
    print("Controls: A/D to move, Space to jump")
    print("Objective: Survive and collect score!")

func _process(_delta):
    # ESC로 종료
    if Input.is_action_just_pressed("ui_cancel"):
        get_tree().quit()
    
    # P로 일시정지 (옵션)
    if Input.is_key_pressed(KEY_P):
        game_paused = !game_paused
        get_tree().paused = game_paused

func _notification(what):
    if what == NOTIFICATION_WM_CLOSE_REQUEST:
        print("Game closed. Final score: ", get_node("UI").score)
        get_tree().quit()
