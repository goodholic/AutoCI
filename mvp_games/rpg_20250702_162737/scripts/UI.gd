extends CanvasLayer

var score = 0
var start_time = 0

func _ready():
    start_time = Time.get_ticks_msec()
    print("UI ready!")

func _process(_delta):
    # 시간 기반 점수 시스템
    var elapsed_time = (Time.get_ticks_msec() - start_time) / 1000
    score = int(elapsed_time * 10)
    $ScoreLabel.text = "Score: " + str(score)
