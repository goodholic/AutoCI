@tool
extends EditorPlugin

const PORT = 8080
var http_server: HTTPServer
var command_processor: CommandProcessor

func _enter_tree():
	print("GodotCI Plugin activated")
	
	# Initialize HTTP server
	http_server = preload("res://addons/godot_ci/http_server.gd").new()
	http_server.port = PORT
	
	# Initialize command processor
	command_processor = preload("res://addons/godot_ci/command_processor.gd").new()
	command_processor.editor_plugin = self
	
	# Start server
	http_server.command_processor = command_processor
	http_server.start()
	
	print("GodotCI API running on port %d" % PORT)

func _exit_tree():
	print("GodotCI Plugin deactivated")
	
	if http_server:
		http_server.stop()
		http_server.queue_free()
	
	if command_processor:
		command_processor.queue_free()

func _has_main_screen():
	return false

func _get_plugin_name():
	return "GodotCI"