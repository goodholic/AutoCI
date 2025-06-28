"""
Godot Engine Controller for AutoCI
Provides comprehensive control over Godot Engine through HTTP/WebSocket APIs
"""

import json
import time
import subprocess
import requests
import websocket
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import threading
import queue
from datetime import datetime


class GodotController:
    """Main controller for Godot Engine automation"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/api"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.session = requests.Session()
        self.websocket = None
        self.ws_thread = None
        self.event_queue = queue.Queue()
        self.godot_process = None
        
    def start_godot_engine(self, project_path: Optional[str] = None, 
                          headless: bool = False,
                          editor: bool = True) -> subprocess.Popen:
        """Start Godot engine with specified parameters"""
        cmd = ["godot"]
        
        if headless:
            cmd.append("--headless")
            
        if editor:
            cmd.append("--editor")
        else:
            cmd.append("--no-window")
            
        if project_path:
            cmd.append("--path")
            cmd.append(str(project_path))
            
        # Enable our custom plugin
        cmd.extend(["--enable-plugin", "GodotCI"])
        
        self.godot_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for Godot to initialize
        time.sleep(3)
        
        # Check if API is available
        if not self.ping():
            raise RuntimeError("Failed to connect to Godot API")
            
        return self.godot_process
        
    def stop_godot_engine(self):
        """Stop the Godot engine process"""
        if self.godot_process:
            self.godot_process.terminate()
            self.godot_process.wait()
            self.godot_process = None
            
    def ping(self) -> bool:
        """Check if Godot API is available"""
        try:
            response = self.session.get(f"{self.base_url}/ping", timeout=2)
            return response.status_code == 200
        except:
            return False
            
    # Project Management
    def create_project(self, name: str, path: str, 
                      renderer: str = "forward_plus") -> Dict:
        """Create a new Godot project"""
        return self._post("/project/create", {
            "name": name,
            "path": path,
            "renderer": renderer
        })
        
    def open_project(self, path: str) -> Dict:
        """Open an existing Godot project"""
        return self._get("/project/open", params={"path": path})
        
    def get_project_info(self) -> Dict:
        """Get current project information"""
        return self._get("/project/info")
        
    def set_project_setting(self, setting: str, value: Any) -> Dict:
        """Set a project setting"""
        return self._put("/project/settings", {
            "setting": setting,
            "value": value
        })
        
    # Scene Management
    def create_scene(self, name: str, root_type: str = "Node") -> Dict:
        """Create a new scene"""
        return self._post("/scene/create", {
            "name": name,
            "root_type": root_type
        })
        
    def load_scene(self, path: str) -> Dict:
        """Load a scene from file"""
        return self._get("/scene/load", params={"path": path})
        
    def save_scene(self, path: Optional[str] = None) -> Dict:
        """Save the current scene"""
        data = {"path": path} if path else {}
        return self._put("/scene/save", data)
        
    def close_scene(self) -> Dict:
        """Close the current scene"""
        return self._delete("/scene/close")
        
    def get_scene_tree(self) -> Dict:
        """Get the entire scene tree structure"""
        return self._get("/scene/tree")
        
    # Node Operations
    def create_node(self, node_type: str, name: str, 
                   parent: str = "/root") -> Dict:
        """Create a new node in the scene"""
        return self._post("/node/create", {
            "type": node_type,
            "name": name,
            "parent": parent
        })
        
    def get_node(self, path: str) -> Dict:
        """Get node information"""
        return self._get(f"/node/{path}")
        
    def delete_node(self, path: str) -> Dict:
        """Delete a node from the scene"""
        return self._delete(f"/node/{path}")
        
    def move_node(self, path: str, new_parent: str) -> Dict:
        """Move a node to a new parent"""
        return self._put(f"/node/{path}/move", {
            "new_parent": new_parent
        })
        
    def rename_node(self, path: str, new_name: str) -> Dict:
        """Rename a node"""
        return self._put(f"/node/{path}/rename", {
            "new_name": new_name
        })
        
    # Property Management
    def set_property(self, node_path: str, property: str, value: Any) -> Dict:
        """Set a node property"""
        return self._put(f"/node/{node_path}/property", {
            "property": property,
            "value": value
        })
        
    def get_property(self, node_path: str, property: str) -> Any:
        """Get a node property value"""
        response = self._get(f"/node/{node_path}/property/{property}")
        return response.get("value")
        
    def set_properties(self, node_path: str, properties: Dict[str, Any]) -> Dict:
        """Set multiple properties at once"""
        return self._put(f"/node/{node_path}/properties", properties)
        
    # Signal Management
    def connect_signal(self, source_path: str, signal: str, 
                      target_path: str, method: str) -> Dict:
        """Connect a signal between nodes"""
        return self._post(f"/node/{source_path}/signal/connect", {
            "signal": signal,
            "target": target_path,
            "method": method
        })
        
    def disconnect_signal(self, source_path: str, signal: str, 
                         target_path: str, method: str) -> Dict:
        """Disconnect a signal"""
        return self._delete(f"/node/{source_path}/signal/disconnect", {
            "signal": signal,
            "target": target_path,
            "method": method
        })
        
    def emit_signal(self, node_path: str, signal: str, args: List[Any] = None) -> Dict:
        """Emit a signal from a node"""
        return self._post(f"/node/{node_path}/signal/emit", {
            "signal": signal,
            "args": args or []
        })
        
    # Resource Management
    def import_asset(self, file_path: str, import_settings: Dict = None) -> Dict:
        """Import an asset into the project"""
        return self._post("/resource/import", {
            "path": file_path,
            "settings": import_settings or {}
        })
        
    def create_resource(self, resource_type: str, save_path: str,
                       properties: Dict = None) -> Dict:
        """Create a new resource"""
        return self._post("/resource/create", {
            "type": resource_type,
            "path": save_path,
            "properties": properties or {}
        })
        
    def load_resource(self, path: str) -> Dict:
        """Load a resource"""
        return self._get("/resource/load", params={"path": path})
        
    # Script Operations
    def attach_script(self, node_path: str, script_path: str) -> Dict:
        """Attach a script to a node"""
        return self._put(f"/node/{node_path}/script", {
            "script_path": script_path
        })
        
    def create_script(self, path: str, content: str, 
                     base_type: str = "Node") -> Dict:
        """Create a new script file"""
        return self._post("/script/create", {
            "path": path,
            "content": content,
            "base_type": base_type
        })
        
    def execute_script(self, code: str, context: str = "editor") -> Dict:
        """Execute GDScript code in the editor"""
        return self._post("/script/execute", {
            "code": code,
            "context": context
        })
        
    # Build and Export
    def add_export_preset(self, name: str, platform: str, 
                         settings: Dict = None) -> Dict:
        """Add an export preset"""
        return self._post("/build/preset/create", {
            "name": name,
            "platform": platform,
            "settings": settings or {}
        })
        
    def export_project(self, preset: str, output_path: str,
                      debug: bool = False) -> Dict:
        """Export the project using a preset"""
        return self._post("/build/export", {
            "preset": preset,
            "output": output_path,
            "debug": debug
        })
        
    def run_project(self, scene: Optional[str] = None,
                   args: List[str] = None) -> Dict:
        """Run the project"""
        return self._post("/build/run", {
            "scene": scene,
            "args": args or []
        })
        
    def build_project(self, target: str = "debug") -> Dict:
        """Build the project"""
        return self._post("/build/compile", {
            "target": target
        })
        
    # Editor Control
    def play_scene(self, scene_path: Optional[str] = None) -> Dict:
        """Play a scene in the editor"""
        data = {"scene": scene_path} if scene_path else {}
        return self._post("/editor/play", data)
        
    def stop_scene(self) -> Dict:
        """Stop the running scene"""
        return self._post("/editor/stop")
        
    def pause_scene(self) -> Dict:
        """Pause the running scene"""
        return self._post("/editor/pause")
        
    def reload_scene(self) -> Dict:
        """Reload the current scene"""
        return self._post("/editor/reload")
        
    def get_editor_state(self) -> Dict:
        """Get current editor state"""
        return self._get("/editor/state")
        
    # Debugging and Profiling
    def set_breakpoint(self, script_path: str, line: int) -> Dict:
        """Set a breakpoint in a script"""
        return self._post("/debug/breakpoint/set", {
            "script": script_path,
            "line": line
        })
        
    def remove_breakpoint(self, script_path: str, line: int) -> Dict:
        """Remove a breakpoint"""
        return self._delete("/debug/breakpoint/remove", {
            "script": script_path,
            "line": line
        })
        
    def get_performance_data(self) -> Dict:
        """Get performance profiling data"""
        return self._get("/debug/performance")
        
    # WebSocket Real-time Events
    def connect_websocket(self):
        """Connect to WebSocket for real-time events"""
        def on_message(ws, message):
            event = json.loads(message)
            self.event_queue.put(event)
            
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
            
        def on_close(ws):
            print("WebSocket closed")
            
        self.websocket = websocket.WebSocketApp(
            self.ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        self.ws_thread = threading.Thread(
            target=self.websocket.run_forever
        )
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
    def get_events(self, timeout: float = 0) -> List[Dict]:
        """Get queued events from WebSocket"""
        events = []
        deadline = time.time() + timeout
        
        while True:
            try:
                remaining = deadline - time.time()
                if remaining <= 0 and timeout > 0:
                    break
                    
                event = self.event_queue.get(
                    timeout=remaining if timeout > 0 else 0
                )
                events.append(event)
            except queue.Empty:
                break
                
        return events
        
    # Helper Methods
    def _get(self, endpoint: str, params: Dict = None) -> Dict:
        """Make GET request"""
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params
        )
        response.raise_for_status()
        return response.json()
        
    def _post(self, endpoint: str, data: Dict = None) -> Dict:
        """Make POST request"""
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json=data
        )
        response.raise_for_status()
        return response.json()
        
    def _put(self, endpoint: str, data: Dict = None) -> Dict:
        """Make PUT request"""
        response = self.session.put(
            f"{self.base_url}{endpoint}",
            json=data
        )
        response.raise_for_status()
        return response.json()
        
    def _delete(self, endpoint: str, data: Dict = None) -> Dict:
        """Make DELETE request"""
        response = self.session.delete(
            f"{self.base_url}{endpoint}",
            json=data
        )
        response.raise_for_status()
        return response.json()
        
    # High-level Operations
    def create_2d_sprite(self, name: str, texture_path: str,
                        position: Tuple[float, float] = (0, 0)) -> str:
        """Create a 2D sprite with texture"""
        # Create Sprite2D node
        self.create_node("Sprite2D", name)
        node_path = f"/root/{name}"
        
        # Set texture and position
        self.set_properties(node_path, {
            "texture": texture_path,
            "position": position
        })
        
        return node_path
        
    def create_ui_button(self, name: str, text: str,
                        position: Tuple[float, float] = (0, 0),
                        size: Tuple[float, float] = (100, 50)) -> str:
        """Create a UI button"""
        # Create Button node
        self.create_node("Button", name)
        node_path = f"/root/{name}"
        
        # Configure button
        self.set_properties(node_path, {
            "text": text,
            "position": position,
            "size": size
        })
        
        return node_path
        
    def create_3d_mesh(self, name: str, mesh_type: str = "BoxMesh",
                      position: Tuple[float, float, float] = (0, 0, 0)) -> str:
        """Create a 3D mesh instance"""
        # Create MeshInstance3D
        self.create_node("MeshInstance3D", name)
        node_path = f"/root/{name}"
        
        # Create and assign mesh
        mesh_resource = self.create_resource(mesh_type, f"res://{name}_mesh.tres")
        self.set_property(node_path, "mesh", mesh_resource["path"])
        self.set_property(node_path, "position", position)
        
        return node_path
        
    def setup_physics_body_2d(self, name: str, 
                             shape_type: str = "RectangleShape2D") -> str:
        """Create a physics body with collision shape"""
        # Create CharacterBody2D
        self.create_node("CharacterBody2D", name)
        body_path = f"/root/{name}"
        
        # Add CollisionShape2D
        self.create_node("CollisionShape2D", "CollisionShape", body_path)
        shape_path = f"{body_path}/CollisionShape"
        
        # Create and assign shape
        shape_resource = self.create_resource(
            shape_type, 
            f"res://{name}_shape.tres"
        )
        self.set_property(shape_path, "shape", shape_resource["path"])
        
        return body_path