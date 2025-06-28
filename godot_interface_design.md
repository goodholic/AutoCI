# AutoCI - Godot Engine Integration Design

## Overview
This document outlines the design for integrating AutoCI with the Godot Engine, enabling AutoCI to control and automate Godot operations.

## Architecture

### 1. Integration Approaches

#### A. Language Server Protocol Extension
- **Pros**: 
  - Existing TCP infrastructure
  - JSON-RPC protocol
  - Already supports external communication
- **Implementation**: Extend LSP with custom commands for scene manipulation

#### B. Custom Editor Plugin with HTTP/WebSocket API
- **Pros**:
  - RESTful API design
  - Easy to test and debug
  - Platform independent
- **Implementation**: Create editor plugin that exposes HTTP endpoints

#### C. GDExtension Native Module
- **Pros**:
  - Direct engine access
  - Maximum performance
  - Full API access
- **Implementation**: C++ extension with IPC or network interface

#### D. Remote Debugger Protocol Extension
- **Pros**:
  - Bidirectional communication
  - Existing infrastructure
- **Implementation**: Extend debugger protocol with automation commands

### 2. Recommended Architecture: Hybrid Approach

```
┌─────────────────┐     ┌──────────────────────────────┐
│                 │     │       Godot Engine           │
│                 │     ├──────────────────────────────┤
│                 │     │  GodotCI Plugin              │
│    AutoCI       │     │  ┌────────────────────────┐ │
│                 │────▶│  │ HTTP/WebSocket Server  │ │
│                 │     │  └────────────────────────┘ │
│                 │     │  ┌────────────────────────┐ │
│                 │     │  │ Command Processor      │ │
│                 │     │  └────────────────────────┘ │
│                 │     │  ┌────────────────────────┐ │
│                 │     │  │ Scene Manipulator      │ │
│                 │     │  └────────────────────────┘ │
└─────────────────┘     └──────────────────────────────┘
```

### 3. Core Components

#### A. GodotCI Editor Plugin
```gdscript
# Main plugin controller
tool
extends EditorPlugin

var server: HTTPServer
var command_processor: CommandProcessor
var scene_manipulator: SceneManipulator

func _enter_tree():
    # Initialize HTTP server
    server = HTTPServer.new()
    server.listen(8080)
    
    # Initialize processors
    command_processor = CommandProcessor.new()
    scene_manipulator = SceneManipulator.new()
    
    # Register handlers
    server.register_handler("/api/scene", scene_handler)
    server.register_handler("/api/project", project_handler)
    server.register_handler("/api/build", build_handler)
    server.register_handler("/api/run", run_handler)
```

#### B. Command Categories

1. **Project Management**
   - Create new project
   - Open existing project
   - Configure project settings
   - Manage assets

2. **Scene Operations**
   - Create/load/save scenes
   - Add/remove/modify nodes
   - Set node properties
   - Connect signals

3. **Resource Management**
   - Import assets
   - Create materials/shaders
   - Manage textures/meshes

4. **Build & Export**
   - Configure export presets
   - Build for platforms
   - Run automated tests

5. **Editor Control**
   - Play/stop scenes
   - Debug operations
   - Performance profiling

### 4. API Design

#### REST API Endpoints

```yaml
# Project Management
POST   /api/project/create
GET    /api/project/open
PUT    /api/project/settings
GET    /api/project/info

# Scene Management
POST   /api/scene/create
GET    /api/scene/load
PUT    /api/scene/save
DELETE /api/scene/close

# Node Operations
POST   /api/node/create
GET    /api/node/{path}
PUT    /api/node/{path}/property
DELETE /api/node/{path}
POST   /api/node/{path}/signal/connect

# Build Operations
POST   /api/build/export
GET    /api/build/status
POST   /api/build/run

# Editor Control
POST   /api/editor/play
POST   /api/editor/stop
GET    /api/editor/status
```

#### WebSocket Commands

```json
// Scene manipulation
{
  "command": "scene.node.create",
  "params": {
    "type": "Node2D",
    "name": "Player",
    "parent": "/root/Main",
    "properties": {
      "position": [100, 200],
      "scale": [2, 2]
    }
  }
}

// Real-time updates
{
  "event": "node.property_changed",
  "data": {
    "path": "/root/Main/Player",
    "property": "position",
    "value": [150, 250]
  }
}
```

### 5. AutoCI Integration Module

```python
# autoci_godot.py
class GodotController:
    def __init__(self, host="localhost", port=8080):
        self.base_url = f"http://{host}:{port}/api"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.session = requests.Session()
        self.websocket = None
    
    def create_project(self, name, path):
        """Create a new Godot project"""
        return self.session.post(
            f"{self.base_url}/project/create",
            json={"name": name, "path": path}
        )
    
    def create_scene(self, name):
        """Create a new scene"""
        return self.session.post(
            f"{self.base_url}/scene/create",
            json={"name": name}
        )
    
    def add_node(self, node_type, name, parent="/root"):
        """Add a node to the scene"""
        return self.session.post(
            f"{self.base_url}/node/create",
            json={
                "type": node_type,
                "name": name,
                "parent": parent
            }
        )
    
    def set_property(self, node_path, property, value):
        """Set a node property"""
        return self.session.put(
            f"{self.base_url}/node/{node_path}/property",
            json={"property": property, "value": value}
        )
    
    def export_project(self, preset, output_path):
        """Export the project"""
        return self.session.post(
            f"{self.base_url}/build/export",
            json={
                "preset": preset,
                "output": output_path
            }
        )
```

### 6. Command Examples for AutoCI

```python
# Example: Create a simple 2D game
godot = GodotController()

# Create project
godot.create_project("MyGame", "/path/to/project")

# Create main scene
godot.create_scene("Main")

# Add player
godot.add_node("CharacterBody2D", "Player", "/root")
godot.add_node("Sprite2D", "Sprite", "/root/Player")
godot.add_node("CollisionShape2D", "Collision", "/root/Player")

# Configure player
godot.set_property("/root/Player", "position", [400, 300])
godot.set_property("/root/Player/Sprite", "texture", "res://player.png")

# Add UI
godot.add_node("CanvasLayer", "UI", "/root")
godot.add_node("Label", "Score", "/root/UI")
godot.set_property("/root/UI/Score", "text", "Score: 0")

# Export game
godot.export_project("Windows Desktop", "./builds/mygame.exe")
```

### 7. Implementation Phases

#### Phase 1: Basic Infrastructure
- HTTP server in editor plugin
- Basic project/scene commands
- Node creation and property setting

#### Phase 2: Advanced Features
- WebSocket for real-time updates
- Signal connections
- Resource management

#### Phase 3: Build Automation
- Export preset management
- Automated testing
- CI/CD integration

#### Phase 4: Advanced Control
- Performance profiling
- Debug automation
- Custom script execution

### 8. Security Considerations

1. **Authentication**: API key or token-based auth
2. **Authorization**: Role-based access control
3. **Validation**: Input sanitization and validation
4. **Rate Limiting**: Prevent abuse
5. **Sandbox Mode**: Restricted operations for safety

### 9. Error Handling

```json
{
  "error": {
    "code": "NODE_NOT_FOUND",
    "message": "Node at path '/root/Invalid' does not exist",
    "details": {
      "path": "/root/Invalid",
      "suggestion": "Check if the parent node exists"
    }
  }
}
```

### 10. Benefits

1. **Automation**: Full scene creation and modification
2. **Testing**: Automated test scene generation
3. **CI/CD**: Integrated build pipelines
4. **Procedural Generation**: Dynamic content creation
5. **Remote Control**: Control Godot from any platform
6. **Batch Operations**: Process multiple projects/scenes

This design provides a comprehensive framework for AutoCI to control Godot Engine, enabling powerful automation capabilities while maintaining security and stability.