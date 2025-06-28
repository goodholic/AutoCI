"""
AutoCI Godot Command System
Provides high-level commands for Godot automation
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from .godot_controller import GodotController


class GodotCommands:
    """High-level command interface for Godot automation"""
    
    def __init__(self):
        self.controller = None
        self.current_project = None
        self.current_scene = None
        
    def connect(self, host: str = "localhost", port: int = 8080) -> bool:
        """Connect to Godot engine"""
        self.controller = GodotController(host, port)
        return self.controller.ping()
        
    def start_engine(self, project_path: Optional[str] = None,
                    headless: bool = False) -> bool:
        """Start Godot engine"""
        if not self.controller:
            self.controller = GodotController()
            
        try:
            self.controller.start_godot_engine(project_path, headless)
            self.current_project = project_path
            return True
        except Exception as e:
            print(f"Failed to start Godot: {e}")
            return False
            
    def stop_engine(self):
        """Stop Godot engine"""
        if self.controller:
            self.controller.stop_godot_engine()
            
    # Project Commands
    def cmd_create_project(self, name: str, path: str, 
                          template: Optional[str] = None) -> Dict:
        """Create a new Godot project with optional template"""
        result = self.controller.create_project(name, path)
        
        if result["success"] and template:
            # Apply project template
            self._apply_project_template(path, template)
            
        self.current_project = path
        return result
        
    def cmd_open_project(self, path: str) -> Dict:
        """Open an existing project"""
        result = self.controller.open_project(path)
        if result["success"]:
            self.current_project = path
        return result
        
    def cmd_configure_project(self, settings: Dict[str, Any]) -> Dict:
        """Configure project settings"""
        results = []
        for setting, value in settings.items():
            result = self.controller.set_project_setting(setting, value)
            results.append(result)
        return {"success": all(r["success"] for r in results), "results": results}
        
    # Scene Commands
    def cmd_create_scene(self, name: str, template: Optional[str] = None) -> Dict:
        """Create a scene with optional template"""
        if template == "2d_game":
            result = self.controller.create_scene(name, "Node2D")
        elif template == "3d_game":
            result = self.controller.create_scene(name, "Node3D")
        elif template == "ui":
            result = self.controller.create_scene(name, "Control")
        else:
            result = self.controller.create_scene(name)
            
        if result["success"]:
            self.current_scene = name
            
            # Apply scene template
            if template:
                self._apply_scene_template(template)
                
        return result
        
    def cmd_save_scene(self, path: Optional[str] = None) -> Dict:
        """Save current scene"""
        if not path and self.current_scene:
            path = f"res://{self.current_scene}.tscn"
        return self.controller.save_scene(path)
        
    # Node Commands
    def cmd_add_player(self, name: str = "Player", 
                      player_type: str = "2d") -> Dict:
        """Add a player character to the scene"""
        if player_type == "2d":
            return self._create_2d_player(name)
        elif player_type == "3d":
            return self._create_3d_player(name)
        else:
            return {"success": False, "error": "Invalid player type"}
            
    def cmd_add_enemy(self, name: str = "Enemy", 
                     enemy_type: str = "basic") -> Dict:
        """Add an enemy to the scene"""
        if enemy_type == "basic":
            return self._create_basic_enemy(name)
        elif enemy_type == "boss":
            return self._create_boss_enemy(name)
        else:
            return {"success": False, "error": "Invalid enemy type"}
            
    def cmd_add_ui_element(self, element_type: str, name: str,
                          properties: Dict = None) -> Dict:
        """Add UI element to the scene"""
        ui_types = {
            "button": "Button",
            "label": "Label",
            "text_input": "LineEdit",
            "health_bar": "ProgressBar",
            "menu": "MenuBar",
            "dialog": "AcceptDialog"
        }
        
        if element_type not in ui_types:
            return {"success": False, "error": "Invalid UI element type"}
            
        # Create UI layer if not exists
        self._ensure_ui_layer()
        
        # Create element
        result = self.controller.create_node(
            ui_types[element_type], 
            name, 
            "/root/UI"
        )
        
        # Apply properties
        if result["success"] and properties:
            self.controller.set_properties(f"/root/UI/{name}", properties)
            
        return result
        
    def cmd_add_environment(self, env_type: str) -> Dict:
        """Add environment to the scene"""
        environments = {
            "outdoor": self._create_outdoor_environment,
            "indoor": self._create_indoor_environment,
            "space": self._create_space_environment,
            "underwater": self._create_underwater_environment
        }
        
        if env_type not in environments:
            return {"success": False, "error": "Invalid environment type"}
            
        return environments[env_type]()
        
    # Animation Commands
    def cmd_add_animation(self, node_path: str, animation_name: str,
                         animation_data: Dict) -> Dict:
        """Add animation to a node"""
        # Create AnimationPlayer if not exists
        anim_player_path = f"{node_path}/AnimationPlayer"
        if not self._node_exists(anim_player_path):
            self.controller.create_node("AnimationPlayer", "AnimationPlayer", node_path)
            
        # Create animation resource
        anim_resource = self.controller.create_resource(
            "Animation",
            f"res://animations/{animation_name}.tres",
            animation_data
        )
        
        # Add to animation player
        return self.controller.execute_script(f"""
            var player = get_node("{anim_player_path}")
            var anim = load("{anim_resource['path']}")
            player.add_animation("{animation_name}", anim)
        """)
        
    # Physics Commands
    def cmd_setup_physics(self, physics_type: str = "2d",
                         gravity: float = 980.0) -> Dict:
        """Setup physics for the project"""
        settings = {}
        
        if physics_type == "2d":
            settings["physics/2d/default_gravity"] = gravity
            settings["physics/2d/default_gravity_vector"] = [0, 1]
        else:
            settings["physics/3d/default_gravity"] = gravity
            settings["physics/3d/default_gravity_vector"] = [0, -1, 0]
            
        return self.cmd_configure_project(settings)
        
    # Build Commands
    def cmd_setup_export(self, platforms: List[str]) -> Dict:
        """Setup export presets for platforms"""
        results = []
        
        platform_configs = {
            "windows": {"platform": "Windows Desktop", "extension": ".exe"},
            "linux": {"platform": "Linux", "extension": ""},
            "macos": {"platform": "macOS", "extension": ".app"},
            "android": {"platform": "Android", "extension": ".apk"},
            "ios": {"platform": "iOS", "extension": ".ipa"},
            "web": {"platform": "Web", "extension": ".html"}
        }
        
        for platform in platforms:
            if platform in platform_configs:
                config = platform_configs[platform]
                result = self.controller.add_export_preset(
                    f"{platform}_export",
                    config["platform"]
                )
                results.append(result)
                
        return {"success": all(r["success"] for r in results), "results": results}
        
    def cmd_build_game(self, platform: str, output_path: str,
                      debug: bool = False) -> Dict:
        """Build the game for a platform"""
        preset_name = f"{platform}_export"
        return self.controller.export_project(preset_name, output_path, debug)
        
    def cmd_run_game(self, scene: Optional[str] = None) -> Dict:
        """Run the game"""
        return self.controller.run_project(scene)
        
    # Testing Commands
    def cmd_create_test_scene(self, test_name: str, test_type: str) -> Dict:
        """Create a test scene"""
        scene_name = f"test_{test_name}"
        result = self.controller.create_scene(scene_name)
        
        if result["success"]:
            # Add test framework
            self._setup_test_framework(test_type)
            
        return result
        
    def cmd_run_tests(self, test_pattern: str = "*") -> Dict:
        """Run automated tests"""
        return self.controller.execute_script(f"""
            var test_runner = preload("res://tests/test_runner.gd").new()
            test_runner.run_tests("{test_pattern}")
        """)
        
    # Utility Commands
    def cmd_take_screenshot(self, output_path: str) -> Dict:
        """Take a screenshot of the current scene"""
        return self.controller.execute_script(f"""
            get_viewport().get_texture().get_image().save_png("{output_path}")
        """)
        
    def cmd_profile_performance(self, duration: float = 5.0) -> Dict:
        """Profile performance for a duration"""
        import time
        
        # Start profiling
        self.controller.execute_script("Performance.set_monitor_enabled(true)")
        
        # Collect data
        start_time = time.time()
        performance_data = []
        
        while time.time() - start_time < duration:
            data = self.controller.get_performance_data()
            performance_data.append(data)
            time.sleep(0.1)
            
        return {
            "success": True,
            "data": performance_data,
            "duration": duration
        }
        
    # Helper Methods
    def _create_2d_player(self, name: str) -> Dict:
        """Create a 2D player character"""
        # Create player body
        self.controller.create_node("CharacterBody2D", name)
        player_path = f"/root/{name}"
        
        # Add sprite
        self.controller.create_node("Sprite2D", "Sprite", player_path)
        
        # Add collision
        self.controller.create_node("CollisionShape2D", "Collision", player_path)
        shape = self.controller.create_resource(
            "RectangleShape2D",
            f"res://shapes/{name}_shape.tres"
        )
        self.controller.set_property(
            f"{player_path}/Collision",
            "shape",
            shape["path"]
        )
        
        # Add basic player script
        script_content = """extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

func _physics_process(delta):
    if not is_on_floor():
        velocity += get_gravity() * delta
        
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = JUMP_VELOCITY
        
    var direction = Input.get_axis("ui_left", "ui_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = move_toward(velocity.x, 0, SPEED)
        
    move_and_slide()
"""
        
        self.controller.create_script(
            f"res://scripts/{name}.gd",
            script_content,
            "CharacterBody2D"
        )
        self.controller.attach_script(player_path, f"res://scripts/{name}.gd")
        
        return {"success": True, "path": player_path}
        
    def _create_3d_player(self, name: str) -> Dict:
        """Create a 3D player character"""
        # Create player body
        self.controller.create_node("CharacterBody3D", name)
        player_path = f"/root/{name}"
        
        # Add mesh
        self.controller.create_node("MeshInstance3D", "Mesh", player_path)
        mesh = self.controller.create_resource(
            "CapsuleMesh",
            f"res://meshes/{name}_mesh.tres"
        )
        self.controller.set_property(
            f"{player_path}/Mesh",
            "mesh",
            mesh["path"]
        )
        
        # Add collision
        self.controller.create_node("CollisionShape3D", "Collision", player_path)
        shape = self.controller.create_resource(
            "CapsuleShape3D",
            f"res://shapes/{name}_shape.tres"
        )
        self.controller.set_property(
            f"{player_path}/Collision",
            "shape",
            shape["path"]
        )
        
        return {"success": True, "path": player_path}
        
    def _create_basic_enemy(self, name: str) -> Dict:
        """Create a basic enemy"""
        # Similar to player but with enemy AI
        self.controller.create_node("CharacterBody2D", name)
        enemy_path = f"/root/{name}"
        
        # Add components
        self.controller.create_node("Sprite2D", "Sprite", enemy_path)
        self.controller.create_node("CollisionShape2D", "Collision", enemy_path)
        self.controller.create_node("Area2D", "DetectionArea", enemy_path)
        
        # Add enemy script
        script_content = """extends CharacterBody2D

var speed = 100.0
var player = null

func _ready():
    player = get_node("/root/Player")
    
func _physics_process(delta):
    if player:
        var direction = (player.global_position - global_position).normalized()
        velocity = direction * speed
        move_and_slide()
"""
        
        self.controller.create_script(
            f"res://scripts/{name}.gd",
            script_content,
            "CharacterBody2D"
        )
        self.controller.attach_script(enemy_path, f"res://scripts/{name}.gd")
        
        return {"success": True, "path": enemy_path}
        
    def _ensure_ui_layer(self):
        """Ensure UI layer exists"""
        if not self._node_exists("/root/UI"):
            self.controller.create_node("CanvasLayer", "UI")
            
    def _node_exists(self, path: str) -> bool:
        """Check if node exists"""
        try:
            self.controller.get_node(path)
            return True
        except:
            return False
            
    def _apply_project_template(self, project_path: str, template: str):
        """Apply project template"""
        templates = {
            "2d_platformer": self._setup_2d_platformer,
            "3d_fps": self._setup_3d_fps,
            "puzzle": self._setup_puzzle_game,
            "rpg": self._setup_rpg_game
        }
        
        if template in templates:
            templates[template](project_path)
            
    def _apply_scene_template(self, template: str):
        """Apply scene template"""
        templates = {
            "2d_game": self._setup_2d_game_scene,
            "3d_game": self._setup_3d_game_scene,
            "ui": self._setup_ui_scene
        }
        
        if template in templates:
            templates[template]()
            
    def _setup_2d_game_scene(self):
        """Setup basic 2D game scene"""
        # Add camera
        self.controller.create_node("Camera2D", "MainCamera")
        self.controller.set_property("/root/MainCamera", "enabled", True)
        
        # Add UI layer
        self.controller.create_node("CanvasLayer", "UI")
        
    def _create_outdoor_environment(self) -> Dict:
        """Create outdoor environment"""
        # Add directional light (sun)
        self.controller.create_node("DirectionalLight3D", "Sun")
        self.controller.set_properties("/root/Sun", {
            "light_energy": 1.0,
            "shadow_enabled": True,
            "rotation": [-0.785, 0.785, 0]  # 45 degree angle
        })
        
        # Add environment
        self.controller.create_node("WorldEnvironment", "Environment")
        env_resource = self.controller.create_resource(
            "Environment",
            "res://environments/outdoor.tres",
            {
                "background_mode": "sky",
                "ambient_light_source": "sky",
                "fog_enabled": True
            }
        )
        self.controller.set_property(
            "/root/Environment",
            "environment",
            env_resource["path"]
        )
        
        return {"success": True}