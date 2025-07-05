extends Control

var socket : StreamPeerTCP
var update_timer : Timer
var start_time : float
var log_lines : Array = []
const MAX_LOG_LINES = 100

func _ready():
	start_time = Time.get_ticks_msec() / 1000.0
	
	# 타이머 설정
	update_timer = Timer.new()
	update_timer.wait_time = 0.1
	update_timer.timeout.connect(_update_dashboard)
	add_child(update_timer)
	update_timer.start()
	
	# 소켓 연결
	_connect_to_autoci()
	
func _connect_to_autoci():
	socket = StreamPeerTCP.new()
	var error = socket.connect_to_host("127.0.0.1", 12345)
	if error != OK:
		add_log("[color=red]AutoCI 연결 실패[/color]")
	else:
		add_log("[color=green]AutoCI 연결 성공[/color]")

func _update_dashboard():
	# 실행 시간 업데이트
	var elapsed = Time.get_ticks_msec() / 1000.0 - start_time
	var hours = int(elapsed / 3600)
	var minutes = int((elapsed % 3600) / 60)
	var seconds = int(elapsed % 60)
	$MainPanel/StatusContainer/StatsGrid/TimeValue.text = "%02d:%02d:%02d" % [hours, minutes, seconds]
	
	# 소켓에서 데이터 읽기
	if socket and socket.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		var available = socket.get_available_bytes()
		if available > 0:
			var data = socket.get_string(available)
			_process_data(data)

func _process_data(data: String):
	# JSON 데이터 파싱
	var json = JSON.new()
	var parse_result = json.parse(data)
	if parse_result != OK:
		return
		
	var msg = json.data
	
	match msg.get("type", ""):
		"status":
			_update_status(msg)
		"log":
			add_log(msg.get("message", ""))
		"error":
			_show_error(msg.get("message", ""))
		"progress":
			$MainPanel/StatusContainer/ProgressBar.value = msg.get("value", 0)

func _update_status(status: Dictionary):
	$MainPanel/StatusContainer/CurrentTask.text = "[b]현재 작업:[/b] " + status.get("current_task", "")
	$MainPanel/StatusContainer/StatsGrid/TasksValue.text = str(status.get("tasks_completed", 0))
	$MainPanel/StatusContainer/StatsGrid/ErrorsValue.text = str(status.get("errors_count", 0))
	$MainPanel/StatusContainer/StatsGrid/AIStatusValue.text = status.get("ai_status", "")
	
	if status.has("progress"):
		$MainPanel/StatusContainer/ProgressBar.value = status.get("progress", 0)

func add_log(message: String):
	var timestamp = Time.get_time_string_from_system()
	log_lines.append("[color=gray]%s[/color] %s" % [timestamp, message])
	
	# 최대 라인 수 제한
	if log_lines.size() > MAX_LOG_LINES:
		log_lines.pop_front()
	
	# 로그 텍스트 업데이트
	$MainPanel/StatusContainer/LogScroll/LogText.text = "\n".join(log_lines)

func _show_error(error_message: String):
	$ErrorPanel.visible = true
	$ErrorPanel/ErrorText.text = "[color=red]" + error_message + "[/color]"
	add_log("[color=red]오류: " + error_message + "[/color]")

func _on_error_close():
	$ErrorPanel.visible = false

func _exit_tree():
	if socket:
		socket.disconnect_from_host()
