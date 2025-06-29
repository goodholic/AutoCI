extends Control

var socket: StreamPeerTCP
var is_connected := false
var reconnect_timer := 0.0

func _ready():
    print("AutoCI Dashboard 시작")
    _connect_to_autoci()
    
func _connect_to_autoci():
    socket = StreamPeerTCP.new()
    var result = socket.connect_to_host("127.0.0.1", 9999)
    if result == OK:
        is_connected = true
        print("AutoCI에 연결됨")
    else:
        print("AutoCI 연결 실패, 재시도 예정...")

func _process(delta):
    if not is_connected:
        reconnect_timer += delta
        if reconnect_timer > 3.0:
            reconnect_timer = 0.0
            _connect_to_autoci()
        return
    
    # 데이터 수신
    if socket.get_available_bytes() > 0:
        var data = socket.get_string(socket.get_available_bytes())
        _process_data(data)

func _process_data(data: String):
    try:
        var json_data = JSON.parse_string(data)
        if json_data:
            _update_dashboard(json_data)
    except:
        pass

func _update_dashboard(data: Dictionary):
    # 현재 작업 업데이트
    if data.has("current_task"):
        $StatsContainer/CurrentTask.text = "[b]현재 작업:[/b] " + data.current_task
    
    # 진행률 업데이트
    if data.has("progress"):
        $StatsContainer/Progress.value = data.progress
    
    # 통계 업데이트
    if data.has("stats"):
        var stats_text = "[b]📊 시스템 통계[/b]\n\n"
        stats_text += "⏱️ 실행 시간: " + data.stats.get("uptime", "0") + "\n"
        stats_text += "🎮 생성된 게임: " + str(data.stats.get("games_created", 0)) + "개\n"
        stats_text += "📚 학습한 주제: " + str(data.stats.get("topics_learned", 0)) + "개\n"
        stats_text += "🔧 완료된 작업: " + str(data.stats.get("tasks_completed", 0)) + "개\n"
        stats_text += "🤖 AI 모델: " + data.stats.get("ai_model", "미선택") + "\n"
        $StatsContainer/Stats.text = stats_text
    
    # 로그 추가
    if data.has("log"):
        var log_entry = "[color=#" + data.get("color", "ffffff") + "]"
        log_entry += "[" + data.get("time", "") + "] "
        log_entry += data.log + "[/color]\n"
        $LogContainer/LogScroll/LogText.append_text(log_entry)
