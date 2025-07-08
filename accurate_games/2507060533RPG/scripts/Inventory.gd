extends Node

# 인벤토리 시스템
class_name Inventory

signal item_added(item)
signal item_removed(item)

var items: Array = []
var max_size: int = 20

func add_item(item: Dictionary) -> bool:
	if items.size() >= max_size:
		print("인벤토리가 가득 찼습니다!")
		return false
	
	items.append(item)
	emit_signal("item_added", item)
	return true

func remove_item(index: int) -> void:
	if index >= 0 and index < items.size():
		var item = items[index]
		items.remove_at(index)
		emit_signal("item_removed", item)

func get_item_count() -> int:
	return items.size()
