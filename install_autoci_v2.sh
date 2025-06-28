#!/bin/bash

# AutoCI v2.0 Unified Installer
# Installs Gemini CLI, Llama 7B, and Godot integration

echo "🚀 AutoCI v2.0 Unified Installer"
echo "================================"
echo ""

# Check system requirements
echo "📋 Checking system requirements..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi
echo "✅ Python version: $python_version"

# Check Node.js
node_version=$(node --version 2>&1)
if [ $? -ne 0 ]; then
    echo "❌ Node.js not found. Please install Node.js 18 or higher."
    exit 1
fi
echo "✅ Node.js version: $node_version"

# Check .NET SDK
dotnet_version=$(dotnet --version 2>&1)
if [ $? -ne 0 ]; then
    echo "❌ .NET SDK not found. Please install .NET SDK 8.0 or higher."
    exit 1
fi
echo "✅ .NET SDK version: $dotnet_version"

# Create virtual environment
echo ""
echo "🔧 Setting up Python environment..."
python3 -m venv autoci_venv
source autoci_venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip

# Create requirements file for unified system
cat > requirements_unified.txt << EOF
# Core dependencies
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
sentencepiece>=0.1.99
protobuf>=3.20.0
huggingface-hub>=0.19.0

# API and server
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0
aiohttp>=3.9.0

# Gemini CLI dependencies
pyyaml>=6.0
rich>=13.0
click>=8.1.0
prompt_toolkit>=3.0.0

# Monitoring and utilities
psutil>=5.9.0
watchdog>=3.0.0
python-dotenv>=1.0.0

# Development tools
black>=23.0.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
EOF

pip install -r requirements_unified.txt

# Setup Gemini CLI tools
echo ""
echo "🛠️ Setting up Gemini CLI integration..."

# Build Gemini CLI tools
cd gemini-cli
npm install
npm run build
cd ..

# Download Llama model if not exists
echo ""
echo "🧠 Checking Llama 7B model..."
if [ ! -d "CodeLlama-7b-Instruct-hf" ]; then
    echo "Downloading Code Llama 7B model..."
    python download_model.py
else
    echo "✅ Llama model already exists"
fi

# Setup Godot plugin
echo ""
echo "🎮 Setting up Godot plugin..."

# Create Godot plugin directory structure
mkdir -p godot_plugin/addons/godot_ci
cp godot_plugin/*.gd godot_plugin/addons/godot_ci/
cp godot_plugin/plugin.cfg godot_plugin/addons/godot_ci/

# Create additional plugin files
cat > godot_plugin/addons/godot_ci/scene_manipulator.gd << 'EOF'
@tool
extends Node

class_name SceneManipulator

var plugin: EditorPlugin

func create_scene(data: Dictionary) -> Dictionary:
	var scene_name = data.get("name", "NewScene")
	var root_type = data.get("root_type", "Node")
	
	# Create root node based on type
	var root_node
	match root_type:
		"Node2D": root_node = Node2D.new()
		"Node3D": root_node = Node3D.new()
		"Control": root_node = Control.new()
		_: root_node = Node.new()
	
	root_node.name = scene_name
	
	# Set as edited scene
	var scene = PackedScene.new()
	scene.pack(root_node)
	EditorInterface.open_scene_from_path("")
	EditorInterface.get_edited_scene_root().queue_free()
	
	return {"success": true, "scene_name": scene_name}

func create_node(data: Dictionary) -> Dictionary:
	var node_type = data.get("type", "Node")
	var node_name = data.get("name", "NewNode")
	var parent_path = data.get("parent", "/root")
	
	var parent = get_node(parent_path)
	if not parent:
		return {"success": false, "error": "Parent node not found"}
	
	# Create node instance
	var new_node = ClassDB.instantiate(node_type)
	if not new_node:
		return {"success": false, "error": "Invalid node type"}
	
	new_node.name = node_name
	parent.add_child(new_node)
	new_node.owner = EditorInterface.get_edited_scene_root()
	
	return {"success": true, "node_path": new_node.get_path()}
EOF

cat > godot_plugin/addons/godot_ci/project_manager.gd << 'EOF'
@tool
extends Node

class_name ProjectManager

var plugin: EditorPlugin

func create_project(data: Dictionary) -> Dictionary:
	var project_name = data.get("name", "NewProject")
	var project_path = data.get("path", "")
	var renderer = data.get("renderer", "forward_plus")
	
	# Create project configuration
	var config = ConfigFile.new()
	config.set_value("application", "config/name", project_name)
	config.set_value("application", "config/features", PackedStringArray(["4.2", renderer]))
	config.set_value("rendering", "renderer/rendering_method", renderer)
	
	# Save project.godot
	var err = config.save(project_path + "/project.godot")
	
	return {"success": err == OK, "path": project_path}

func import_asset(data: Dictionary) -> Dictionary:
	var file_path = data.get("path", "")
	var import_settings = data.get("settings", {})
	
	# Trigger reimport
	EditorInterface.get_resource_filesystem().scan()
	
	return {"success": true, "imported": file_path}
EOF

# Create AutoCI command wrapper
echo ""
echo "📝 Creating AutoCI command..."

cat > autoci << 'EOF'
#!/bin/bash
# AutoCI v2.0 Command Wrapper

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$SCRIPT_DIR/autoci_venv/bin/activate"

# Set environment variables
export AUTOCI_HOME="$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Parse command
case "$1" in
    start)
        # Start all services
        python "$SCRIPT_DIR/start_unified.py"
        ;;
    "")
        # Interactive mode
        python -m modules.unified_autoci
        ;;
    generate)
        shift
        python -m modules.unified_autoci generate "$@"
        ;;
    analyze)
        shift
        python -m modules.unified_autoci analyze "$@"
        ;;
    monitor)
        # Open monitoring dashboard
        python "$SCRIPT_DIR/monitoring/dashboard.py"
        ;;
    godot)
        shift
        python -m modules.godot_commands "$@"
        ;;
    help)
        echo "AutoCI v2.0 - Unified AI Game Development System"
        echo ""
        echo "Usage: autoci [command] [options]"
        echo ""
        echo "Commands:"
        echo "  (none)     Start interactive mode"
        echo "  start      Start all AutoCI services"
        echo "  generate   Generate code from description"
        echo "  analyze    Analyze project or code"
        echo "  monitor    Open monitoring dashboard"
        echo "  godot      Direct Godot control commands"
        echo "  help       Show this help message"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run 'autoci help' for usage information"
        exit 1
        ;;
esac
EOF

chmod +x autoci

# Create start script for unified system
cat > start_unified.py << 'EOF'
#!/usr/bin/env python3
"""
Start all AutoCI v2.0 services
"""

import subprocess
import time
import os
import sys
from pathlib import Path

def start_service(name, command, cwd=None):
    """Start a service in the background"""
    print(f"Starting {name}...")
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path.cwd())
    
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(2)  # Give service time to start
    
    if process.poll() is None:
        print(f"✅ {name} started (PID: {process.pid})")
        return process
    else:
        print(f"❌ Failed to start {name}")
        stdout, stderr = process.communicate()
        print(f"Error: {stderr.decode()}")
        return None

def main():
    print("🚀 Starting AutoCI v2.0 Services")
    print("================================\n")
    
    services = []
    
    # Start Llama API server
    llama_process = start_service(
        "Llama API Server",
        "cd MyAIWebApp/Models && uvicorn enhanced_server:app --host 0.0.0.0 --port 8000"
    )
    if llama_process:
        services.append(llama_process)
    
    # Start monitoring API
    monitor_process = start_service(
        "Monitoring API",
        "python expert_learning_api.py"
    )
    if monitor_process:
        services.append(monitor_process)
    
    # Start C# Backend
    backend_process = start_service(
        "C# Backend",
        "cd MyAIWebApp/Backend && dotnet run"
    )
    if backend_process:
        services.append(backend_process)
    
    # Start Frontend
    frontend_process = start_service(
        "Blazor Frontend",
        "cd MyAIWebApp/Frontend && dotnet run"
    )
    if frontend_process:
        services.append(frontend_process)
    
    print("\n✨ All services started!")
    print("\nAccess points:")
    print("  • AI API: http://localhost:8000")
    print("  • Backend API: http://localhost:5049")
    print("  • Frontend: https://localhost:7100")
    print("  • Monitoring: http://localhost:8080")
    print("\nPress Ctrl+C to stop all services\n")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping all services...")
        for process in services:
            process.terminate()
        print("Goodbye!")

if __name__ == "__main__":
    main()
EOF

# Add to PATH
echo ""
echo "🔧 Adding AutoCI to PATH..."
echo "export PATH=\"$PWD:\$PATH\"" >> ~/.bashrc
echo "export AUTOCI_HOME=\"$PWD\"" >> ~/.bashrc

# Create desktop entry (for Linux desktop environments)
if [ -d "$HOME/.local/share/applications" ]; then
    cat > "$HOME/.local/share/applications/autoci.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=AutoCI v2.0
Comment=Unified AI Game Development System
Exec=$PWD/autoci
Icon=$PWD/assets/autoci-icon.png
Terminal=true
Categories=Development;IDE;
EOF
fi

# Final setup
echo ""
echo "🎉 Installation complete!"
echo ""
echo "To start using AutoCI v2.0:"
echo "  1. Restart your terminal or run: source ~/.bashrc"
echo "  2. Run: autoci"
echo ""
echo "First time setup:"
echo "  • The Llama model will be downloaded automatically if needed"
echo "  • Godot plugin should be copied to your Godot project"
echo "  • All services will start automatically"
echo ""
echo "For help, run: autoci help"
echo ""
echo "Happy game development! 🎮"
EOF

chmod +x install_autoci_v2.sh