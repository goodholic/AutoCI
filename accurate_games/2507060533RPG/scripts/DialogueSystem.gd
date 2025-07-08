extends Node

# 대화 시스템
class_name DialogueSystem

signal dialogue_started
signal dialogue_finished

var current_dialogue: Array = []
var dialogue_index: int = 0
var is_active: bool = false

func start_dialogue(dialogue_data: Array) -> void:
	current_dialogue = dialogue_data
	dialogue_index = 0
	is_active = true
	emit_signal("dialogue_started")
	show_next_line()

func show_next_line() -> void:
	if dialogue_index < current_dialogue.size():
		var line = current_dialogue[dialogue_index]
		print("💬 ", line.speaker, ": ", line.text)
		dialogue_index += 1
	else:
		finish_dialogue()

func finish_dialogue() -> void:
	is_active = false
	current_dialogue.clear()
	dialogue_index = 0
	emit_signal("dialogue_finished")

func _input(event: InputEvent) -> void:
	if is_active and event.is_action_pressed("ui_accept"):
		show_next_line()
