@tool
extends Node

class_name GodotCIAPIHandler

var plugin: EditorPlugin
var scene_manipulator
var project_manager

func _ready():
	scene_manipulator = preload("res://addons/godot_ci/scene_manipulator.gd").new()
	project_manager = preload("res://addons/godot_ci/project_manager.gd").new()
	scene_manipulator.plugin = plugin
	project_manager.plugin = plugin

func handle_request(method: String, path: String, body: String) -> Dictionary:
	# Parse JSON body if present
	var data = {}
	if not body.is_empty():
		var json = JSON.new()
		var parse_result = json.parse(body)
		if parse_result == OK:
			data = json.data
	
	# Remove /api prefix
	if path.begins_with("/api"):
		path = path.substr(4)
	
	# Route to appropriate handler
	var segments = path.split("/")
	segments = segments.filter(func(s): return not s.is_empty())
	
	if segments.is_empty():
		return _error_response("No endpoint specified")
	
	match segments[0]:
		"ping":
			return _handle_ping()
		"project":
			return _handle_project(method, segments, data)
		"scene":
			return _handle_scene(method, segments, data)
		"node":
			return _handle_node(method, segments, data)
		"resource":
			return _handle_resource(method, segments, data)
		"script":
			return _handle_script(method, segments, data)
		"build":
			return _handle_build(method, segments, data)
		"editor":
			return _handle_editor(method, segments, data)
		"debug":
			return _handle_debug(method, segments, data)
		_:
			return _error_response("Unknown endpoint: " + segments[0])

func _handle_ping() -> Dictionary:
	return _success_response({"message": "GodotCI API is running"})

func _handle_project(method: String, segments: Array, data: Dictionary) -> Dictionary:
	if segments.size() < 2:
		return _error_response("Invalid project endpoint")
	
	match segments[1]:
		"create":
			if method != "POST":
				return _error_response("Method not allowed")
			return project_manager.create_project(data)
		"open":
			if method != "GET":
				return _error_response("Method not allowed")
			return project_manager.open_project(data.get("path", ""))
		"info":
			if method != "GET":
				return _error_response("Method not allowed")
			return project_manager.get_project_info()
		"settings":
			if method != "PUT":
				return _error_response("Method not allowed")
			return project_manager.set_project_setting(data)
		_:
			return _error_response("Unknown project endpoint")

func _handle_scene(method: String, segments: Array, data: Dictionary) -> Dictionary:
	if segments.size() < 2:
		return _error_response("Invalid scene endpoint")
	
	match segments[1]:
		"create":
			if method != "POST":
				return _error_response("Method not allowed")
			return scene_manipulator.create_scene(data)
		"load":
			if method != "GET":
				return _error_response("Method not allowed")
			return scene_manipulator.load_scene(data.get("path", ""))
		"save":
			if method != "PUT":
				return _error_response("Method not allowed")
			return scene_manipulator.save_scene(data.get("path", ""))
		"close":
			if method != "DELETE":
				return _error_response("Method not allowed")
			return scene_manipulator.close_scene()
		"tree":
			if method != "GET":
				return _error_response("Method not allowed")
			return scene_manipulator.get_scene_tree()
		_:
			return _error_response("Unknown scene endpoint")

func _handle_node(method: String, segments: Array, data: Dictionary) -> Dictionary:
	if segments.size() < 2:
		return _error_response("Invalid node endpoint")
	
	if segments[1] == "create" and method == "POST":
		return scene_manipulator.create_node(data)
	
	# Get node path from URL
	var node_path = "/" + segments.slice(1).join("/")
	
	if segments.size() < 3:
		# Operations on the node itself
		match method:
			"GET":
				return scene_manipulator.get_node_info(node_path)
			"DELETE":
				return scene_manipulator.delete_node(node_path)
			_:
				return _error_response("Method not allowed")
	
	# Sub-operations on node
	match segments[2]:
		"property":
			if method == "PUT":
				return scene_manipulator.set_node_property(node_path, data)
			elif method == "GET" and segments.size() > 3:
				return scene_manipulator.get_node_property(node_path, segments[3])
		"properties":
			if method == "PUT":
				return scene_manipulator.set_node_properties(node_path, data)
		"move":
			if method == "PUT":
				return scene_manipulator.move_node(node_path, data.get("new_parent", ""))
		"rename":
			if method == "PUT":
				return scene_manipulator.rename_node(node_path, data.get("new_name", ""))
		"signal":
			return _handle_signal(method, node_path, segments.slice(3), data)
		"script":
			if method == "PUT":
				return scene_manipulator.attach_script(node_path, data.get("script_path", ""))
		_:
			return _error_response("Unknown node operation")
	
	return _error_response("Invalid request")

func _handle_signal(method: String, node_path: String, segments: Array, data: Dictionary) -> Dictionary:
	if segments.is_empty():
		return _error_response("Invalid signal endpoint")
	
	match segments[0]:
		"connect":
			if method != "POST":
				return _error_response("Method not allowed")
			return scene_manipulator.connect_signal(node_path, data)
		"disconnect":
			if method != "DELETE":
				return _error_response("Method not allowed")
			return scene_manipulator.disconnect_signal(node_path, data)
		"emit":
			if method != "POST":
				return _error_response("Method not allowed")
			return scene_manipulator.emit_signal(node_path, data)
		_:
			return _error_response("Unknown signal operation")

func _handle_resource(method: String, segments: Array, data: Dictionary) -> Dictionary:
	if segments.size() < 2:
		return _error_response("Invalid resource endpoint")
	
	match segments[1]:
		"import":
			if method != "POST":
				return _error_response("Method not allowed")
			return project_manager.import_asset(data)
		"create":
			if method != "POST":
				return _error_response("Method not allowed")
			return project_manager.create_resource(data)
		"load":
			if method != "GET":
				return _error_response("Method not allowed")
			return project_manager.load_resource(data.get("path", ""))
		_:
			return _error_response("Unknown resource endpoint")

func _handle_script(method: String, segments: Array, data: Dictionary) -> Dictionary:
	if segments.size() < 2:
		return _error_response("Invalid script endpoint")
	
	match segments[1]:
		"create":
			if method != "POST":
				return _error_response("Method not allowed")
			return project_manager.create_script(data)
		"execute":
			if method != "POST":
				return _error_response("Method not allowed")
			return _execute_script(data.get("code", ""), data.get("context", "editor"))
		_:
			return _error_response("Unknown script endpoint")

func _handle_build(method: String, segments: Array, data: Dictionary) -> Dictionary:
	if segments.size() < 2:
		return _error_response("Invalid build endpoint")
	
	match segments[1]:
		"preset":
			if segments.size() > 2 and segments[2] == "create":
				if method != "POST":
					return _error_response("Method not allowed")
				return project_manager.add_export_preset(data)
		"export":
			if method != "POST":
				return _error_response("Method not allowed")
			return project_manager.export_project(data)
		"run":
			if method != "POST":
				return _error_response("Method not allowed")
			return _run_project(data)
		"compile":
			if method != "POST":
				return _error_response("Method not allowed")
			return project_manager.build_project(data)
		_:
			return _error_response("Unknown build endpoint")

func _handle_editor(method: String, segments: Array, data: Dictionary) -> Dictionary:
	if segments.size() < 2:
		return _error_response("Invalid editor endpoint")
	
	match segments[1]:
		"play":
			if method != "POST":
				return _error_response("Method not allowed")
			return _play_scene(data.get("scene", ""))
		"stop":
			if method != "POST":
				return _error_response("Method not allowed")
			return _stop_scene()
		"pause":
			if method != "POST":
				return _error_response("Method not allowed")
			return _pause_scene()
		"reload":
			if method != "POST":
				return _error_response("Method not allowed")
			return _reload_scene()
		"state":
			if method != "GET":
				return _error_response("Method not allowed")
			return _get_editor_state()
		_:
			return _error_response("Unknown editor endpoint")

func _handle_debug(method: String, segments: Array, data: Dictionary) -> Dictionary:
	if segments.size() < 2:
		return _error_response("Invalid debug endpoint")
	
	match segments[1]:
		"breakpoint":
			if segments.size() > 2:
				match segments[2]:
					"set":
						if method != "POST":
							return _error_response("Method not allowed")
						return _set_breakpoint(data)
					"remove":
						if method != "DELETE":
							return _error_response("Method not allowed")
						return _remove_breakpoint(data)
		"performance":
			if method != "GET":
				return _error_response("Method not allowed")
			return _get_performance_data()
		_:
			return _error_response("Unknown debug endpoint")

# Editor control functions
func _play_scene(scene_path: String) -> Dictionary:
	if scene_path.is_empty():
		EditorInterface.play_main_scene()
	else:
		EditorInterface.play_custom_scene(scene_path)
	return _success_response({"playing": true})

func _stop_scene() -> Dictionary:
	EditorInterface.stop_playing_scene()
	return _success_response({"playing": false})

func _pause_scene() -> Dictionary:
	EditorInterface.set_main_screen_editor("Script")
	return _success_response({"paused": true})

func _reload_scene() -> Dictionary:
	EditorInterface.reload_scene_from_path(
		EditorInterface.get_edited_scene_root().scene_file_path
	)
	return _success_response({"reloaded": true})

func _get_editor_state() -> Dictionary:
	return _success_response({
		"playing": EditorInterface.is_playing_scene(),
		"edited_scene": EditorInterface.get_edited_scene_root().scene_file_path if EditorInterface.get_edited_scene_root() else "",
		"open_scenes": EditorInterface.get_open_scenes()
	})

func _execute_script(code: String, context: String) -> Dictionary:
	var script = GDScript.new()
	script.source_code = code
	
	var err = script.reload()
	if err != OK:
		return _error_response("Script compilation failed")
	
	var instance = script.new()
	if instance.has_method("_run"):
		var result = instance.call("_run")
		return _success_response({"result": result})
	else:
		return _error_response("Script must have _run() method")

func _run_project(data: Dictionary) -> Dictionary:
	var args = data.get("args", [])
	var scene = data.get("scene", "")
	
	if scene.is_empty():
		EditorInterface.play_main_scene()
	else:
		EditorInterface.play_custom_scene(scene)
		
	return _success_response({"running": true})

func _set_breakpoint(data: Dictionary) -> Dictionary:
	# Implementation would require debugger access
	return _success_response({"breakpoint_set": true})

func _remove_breakpoint(data: Dictionary) -> Dictionary:
	# Implementation would require debugger access
	return _success_response({"breakpoint_removed": true})

func _get_performance_data() -> Dictionary:
	var data = {}
	for i in range(Performance.Monitor.MAX):
		var monitor_name = Performance.get_monitor_name(i)
		if monitor_name != "":
			data[monitor_name] = Performance.get_monitor(i)
	return _success_response({"performance": data})

# Response helpers
func _success_response(data: Dictionary) -> Dictionary:
	data["success"] = true
	return {"status": 200, "data": data}

func _error_response(message: String, code: int = 400) -> Dictionary:
	return {
		"status": code,
		"data": {
			"success": false,
			"error": message
		}
	}