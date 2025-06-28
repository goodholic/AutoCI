@tool
extends EditorPlugin

const HTTPServer = preload("res://addons/godot_ci/http_server.gd")
const APIHandler = preload("res://addons/godot_ci/api_handler.gd")
const WebSocketServer = preload("res://addons/godot_ci/websocket_server.gd")

var http_server: HTTPServer
var api_handler: APIHandler
var ws_server: WebSocketServer
var dock

func _enter_tree():
	print("GodotCI Plugin: Starting...")
	
	# Initialize API handler
	api_handler = APIHandler.new()
	api_handler.plugin = self
	
	# Start HTTP server
	http_server = HTTPServer.new()
	http_server.api_handler = api_handler
	http_server.start(8080)
	
	# Start WebSocket server
	ws_server = WebSocketServer.new()
	ws_server.api_handler = api_handler
	ws_server.start(8081)
	
	# Add custom dock
	dock = preload("res://addons/godot_ci/dock.tscn").instantiate()
	add_control_to_dock(DOCK_SLOT_LEFT_UL, dock)
	
	print("GodotCI Plugin: Ready on port 8080 (HTTP) and 8081 (WebSocket)")

func _exit_tree():
	print("GodotCI Plugin: Stopping...")
	
	if http_server:
		http_server.stop()
		http_server.queue_free()
		
	if ws_server:
		ws_server.stop()
		ws_server.queue_free()
		
	if api_handler:
		api_handler.queue_free()
		
	remove_control_from_docks(dock)
	if dock:
		dock.queue_free()
		
	print("GodotCI Plugin: Stopped")

func _has_main_screen():
	return false

func _get_plugin_name():
	return "GodotCI"