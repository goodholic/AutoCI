#!/usr/bin/env python3
"""
AutoCI Godot Automation Example
Demonstrates how to use AutoCI to control Godot Engine
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.godot_controller import GodotController
from modules.godot_commands import GodotCommands


def example_create_2d_platformer():
    """Example: Create a simple 2D platformer game"""
    print("=== Creating 2D Platformer Game ===")
    
    # Initialize controller
    controller = GodotController()
    commands = GodotCommands()
    
    try:
        # Start Godot engine
        print("Starting Godot Engine...")
        controller.start_godot_engine(headless=False)
        commands.controller = controller
        
        # Create new project
        print("Creating new project...")
        project_path = "/tmp/my_platformer"
        commands.cmd_create_project("MyPlatformer", project_path, "2d_platformer")
        
        # Create main scene
        print("Creating main scene...")
        commands.cmd_create_scene("Main", "2d_game")
        
        # Setup physics
        print("Setting up physics...")
        commands.cmd_setup_physics("2d", 980.0)
        
        # Add player
        print("Adding player character...")
        commands.cmd_add_player("Player", "2d")
        controller.set_property("/root/Player", "position", [400, 300])
        
        # Add platforms
        print("Adding platforms...")
        for i, (x, y) in enumerate([(200, 400), (400, 450), (600, 400)]):
            controller.create_node("StaticBody2D", f"Platform{i}")
            platform_path = f"/root/Platform{i}"
            
            # Add sprite
            controller.create_node("Sprite2D", "Sprite", platform_path)
            controller.set_property(f"{platform_path}/Sprite", "texture", "res://platform.png")
            
            # Add collision
            controller.create_node("CollisionShape2D", "Collision", platform_path)
            shape = controller.create_resource(
                "RectangleShape2D",
                f"res://shapes/platform{i}_shape.tres",
                {"size": [200, 20]}
            )
            controller.set_property(f"{platform_path}/Collision", "shape", shape["path"])
            
            # Position platform
            controller.set_property(platform_path, "position", [x, y])
        
        # Add enemies
        print("Adding enemies...")
        for i in range(3):
            commands.cmd_add_enemy(f"Enemy{i}", "basic")
            controller.set_property(f"/root/Enemy{i}", "position", [200 + i * 200, 200])
        
        # Add UI
        print("Adding UI elements...")
        commands.cmd_add_ui_element("label", "ScoreLabel", {
            "text": "Score: 0",
            "position": [10, 10],
            "add_theme_font_size_override": 24
        })
        
        commands.cmd_add_ui_element("label", "LivesLabel", {
            "text": "Lives: 3",
            "position": [10, 50],
            "add_theme_font_size_override": 24
        })
        
        # Add collectibles
        print("Adding collectibles...")
        for i in range(5):
            controller.create_node("Area2D", f"Coin{i}")
            coin_path = f"/root/Coin{i}"
            
            # Add sprite
            controller.create_node("Sprite2D", "Sprite", coin_path)
            controller.set_property(f"{coin_path}/Sprite", "texture", "res://coin.png")
            
            # Add collision
            controller.create_node("CollisionShape2D", "Collision", coin_path)
            shape = controller.create_resource(
                "CircleShape2D",
                f"res://shapes/coin{i}_shape.tres",
                {"radius": 16}
            )
            controller.set_property(f"{coin_path}/Collision", "shape", shape["path"])
            
            # Position coin
            controller.set_property(coin_path, "position", [100 + i * 150, 150])
            
            # Connect signal
            controller.connect_signal(coin_path, "body_entered", "/root/Player", "_on_coin_collected")
        
        # Save scene
        print("Saving scene...")
        commands.cmd_save_scene("res://scenes/main.tscn")
        
        # Setup export
        print("Setting up export presets...")
        commands.cmd_setup_export(["windows", "linux", "web"])
        
        # Take screenshot
        print("Taking screenshot...")
        commands.cmd_take_screenshot("/tmp/platformer_screenshot.png")
        
        # Run the game
        print("Running the game...")
        commands.cmd_run_game()
        
        print("\n✅ 2D Platformer created successfully!")
        print(f"Project location: {project_path}")
        
        # Wait a bit before closing
        time.sleep(5)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Stop Godot
        controller.stop_godot_engine()


def example_batch_scene_generation():
    """Example: Generate multiple test scenes"""
    print("\n=== Batch Scene Generation ===")
    
    controller = GodotController()
    
    try:
        # Connect to running Godot instance
        if not controller.ping():
            print("Starting Godot...")
            controller.start_godot_engine()
        
        # Generate test scenes
        test_scenarios = [
            ("stress_test_1000_nodes", 1000),
            ("stress_test_5000_nodes", 5000),
            ("stress_test_10000_nodes", 10000)
        ]
        
        for scene_name, node_count in test_scenarios:
            print(f"\nGenerating scene: {scene_name} with {node_count} nodes...")
            
            # Create scene
            controller.create_scene(scene_name, "Node2D")
            
            # Add nodes
            for i in range(node_count):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{node_count} nodes...")
                    
                controller.create_node("Sprite2D", f"TestSprite{i}")
                controller.set_properties(f"/root/TestSprite{i}", {
                    "position": [i % 100 * 10, (i // 100) * 10],
                    "modulate": [1.0, i / node_count, 0.5, 1.0]
                })
            
            # Save scene
            controller.save_scene(f"res://tests/{scene_name}.tscn")
            print(f"✅ Scene '{scene_name}' generated!")
            
            # Profile performance
            perf_data = controller.get_performance_data()
            print(f"  Memory usage: {perf_data.get('memory', 0) / 1024 / 1024:.2f} MB")
            print(f"  Object count: {perf_data.get('object_count', 0)}")
            
            # Close scene
            controller.close_scene()
            
    except Exception as e:
        print(f"❌ Error: {e}")


def example_automated_testing():
    """Example: Automated testing workflow"""
    print("\n=== Automated Testing Workflow ===")
    
    commands = GodotCommands()
    
    try:
        # Connect to Godot
        commands.connect()
        
        # Create test scene
        print("Creating test scene...")
        commands.cmd_create_test_scene("player_movement", "unit")
        
        # Add test player
        commands.cmd_add_player("TestPlayer", "2d")
        
        # Execute test script
        test_code = """
extends Node

func _run():
    var player = get_node("/root/TestPlayer")
    var initial_pos = player.position
    
    # Simulate input
    Input.action_press("ui_right")
    
    # Wait a frame
    await get_tree().process_frame
    
    # Check movement
    var moved = player.position.x > initial_pos.x
    
    return {
        "test": "player_movement",
        "passed": moved,
        "message": "Player moved right" if moved else "Player did not move"
    }
"""
        
        result = commands.controller.execute_script(test_code)
        print(f"Test result: {result}")
        
        # Run performance test
        print("\nRunning performance profile...")
        perf_result = commands.cmd_profile_performance(3.0)
        
        avg_fps = sum(d.get("fps", 60) for d in perf_result["data"]) / len(perf_result["data"])
        print(f"Average FPS: {avg_fps:.1f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_procedural_level():
    """Example: Generate a procedural level"""
    print("\n=== Procedural Level Generation ===")
    
    import random
    
    controller = GodotController()
    commands = GodotCommands()
    
    try:
        # Start Godot if needed
        if not controller.ping():
            controller.start_godot_engine()
        commands.controller = controller
        
        # Create scene
        print("Creating procedural level...")
        commands.cmd_create_scene("ProceduralLevel", "2d_game")
        
        # Generate terrain
        print("Generating terrain...")
        terrain_tiles = []
        width, height = 50, 30
        
        for y in range(height):
            for x in range(width):
                # Simple noise-based terrain
                if y > height * 0.7 + random.randint(-2, 2):
                    tile_type = "ground"
                elif y > height * 0.5 and random.random() > 0.8:
                    tile_type = "platform"
                else:
                    continue
                    
                # Create tile
                tile_name = f"Tile_{x}_{y}"
                controller.create_node("StaticBody2D", tile_name)
                tile_path = f"/root/{tile_name}"
                
                # Add sprite
                controller.create_node("Sprite2D", "Sprite", tile_path)
                controller.set_property(f"{tile_path}/Sprite", "texture", f"res://{tile_type}.png")
                
                # Position
                controller.set_property(tile_path, "position", [x * 32, y * 32])
                
                terrain_tiles.append((x, y, tile_type))
        
        # Add random enemies
        print("Placing enemies...")
        enemy_count = random.randint(10, 20)
        for i in range(enemy_count):
            x = random.randint(5, width - 5) * 32
            y = random.randint(5, height // 2) * 32
            
            commands.cmd_add_enemy(f"Enemy{i}", "basic")
            controller.set_property(f"/root/Enemy{i}", "position", [x, y])
        
        # Add collectibles
        print("Placing collectibles...")
        for i in range(30):
            x = random.randint(2, width - 2) * 32
            y = random.randint(2, height - 5) * 32
            
            controller.create_node("Area2D", f"Pickup{i}")
            controller.set_property(f"/root/Pickup{i}", "position", [x, y])
        
        # Add player spawn
        commands.cmd_add_player("Player", "2d")
        controller.set_property("/root/Player", "position", [100, 100])
        
        # Save level
        print("Saving procedural level...")
        controller.save_scene("res://levels/procedural_level.tscn")
        
        print("✅ Procedural level generated!")
        
        # Take screenshot
        controller.execute_script("""
            get_viewport().get_texture().get_image().save_png("res://procedural_level.png")
        """)
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("AutoCI Godot Automation Examples")
    print("=" * 40)
    
    # Run examples
    example_create_2d_platformer()
    example_batch_scene_generation()
    example_automated_testing()
    example_procedural_level()
    
    print("\n✅ All examples completed!")