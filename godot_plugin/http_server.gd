@tool
extends Node

class_name GodotCIHTTPServer

var server: TCPServer
var api_handler
var port: int = 8080
var running: bool = false
var thread: Thread

func start(p: int):
	port = p
	server = TCPServer.new()
	var err = server.listen(port)
	if err != OK:
		push_error("Failed to start HTTP server on port " + str(port))
		return
		
	running = true
	thread = Thread.new()
	thread.start(_server_thread)
	print("HTTP Server started on port " + str(port))

func stop():
	running = false
	if server:
		server.stop()
	if thread:
		thread.wait_to_finish()

func _server_thread():
	while running:
		if server.is_connection_available():
			var client = server.take_connection()
			_handle_client(client)
		OS.delay_msec(10)

func _handle_client(client: StreamPeerTCP):
	if not client.is_connected_to_host():
		return
		
	# Read request
	var request = _read_http_request(client)
	if request.is_empty():
		client.disconnect_from_host()
		return
		
	# Parse request
	var parts = request.split(" ")
	if parts.size() < 3:
		_send_error(client, 400, "Bad Request")
		return
		
	var method = parts[0]
	var path = parts[1]
	var body = _extract_body(request)
	
	# Handle CORS preflight
	if method == "OPTIONS":
		_send_cors_response(client)
		return
		
	# Route request to API handler
	var response = api_handler.handle_request(method, path, body)
	
	# Send response
	_send_response(client, response)
	client.disconnect_from_host()

func _read_http_request(client: StreamPeerTCP) -> String:
	var request = ""
	var timeout = OS.get_ticks_msec() + 5000  # 5 second timeout
	
	while OS.get_ticks_msec() < timeout:
		if client.get_available_bytes() > 0:
			var data = client.get_string(client.get_available_bytes())
			request += data
			if request.contains("\r\n\r\n"):
				break
		OS.delay_msec(1)
		
	return request

func _extract_body(request: String) -> String:
	var parts = request.split("\r\n\r\n")
	if parts.size() > 1:
		return parts[1]
	return ""

func _send_response(client: StreamPeerTCP, response: Dictionary):
	var status = response.get("status", 200)
	var body = JSON.stringify(response.get("data", {}))
	
	var headers = [
		"HTTP/1.1 " + str(status) + " " + _get_status_text(status),
		"Content-Type: application/json",
		"Content-Length: " + str(body.length()),
		"Access-Control-Allow-Origin: *",
		"Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS",
		"Access-Control-Allow-Headers: Content-Type",
		"Connection: close"
	]
	
	var response_text = headers.join("\r\n") + "\r\n\r\n" + body
	client.put_data(response_text.to_utf8_buffer())

func _send_error(client: StreamPeerTCP, code: int, message: String):
	_send_response(client, {
		"status": code,
		"data": {"error": message}
	})

func _send_cors_response(client: StreamPeerTCP):
	var headers = [
		"HTTP/1.1 200 OK",
		"Access-Control-Allow-Origin: *",
		"Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS",
		"Access-Control-Allow-Headers: Content-Type",
		"Content-Length: 0",
		"Connection: close"
	]
	
	var response = headers.join("\r\n") + "\r\n\r\n"
	client.put_data(response.to_utf8_buffer())
	client.disconnect_from_host()

func _get_status_text(code: int) -> String:
	match code:
		200: return "OK"
		201: return "Created"
		400: return "Bad Request"
		404: return "Not Found"
		500: return "Internal Server Error"
		_: return "Unknown"