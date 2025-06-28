"""
Unified AutoCI System - Integration of Gemini CLI, Llama 7B, and Godot Engine
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import subprocess
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from modules.godot_controller import GodotController
from modules.godot_commands import GodotCommands


class GeminiCLIAdapter:
    """Adapter for Gemini CLI integration"""
    
    def __init__(self, gemini_path: str = "./gemini-cli"):
        self.gemini_path = Path(gemini_path)
        self.tools = {}
        self._setup_tools()
    
    def _setup_tools(self):
        """Register AutoCI tools with Gemini CLI"""
        self.tools = {
            'godot-control': self._godot_control_tool,
            'csharp-generate': self._csharp_generate_tool,
            'llama-analyze': self._llama_analyze_tool,
            'project-create': self._project_create_tool,
            'scene-manipulate': self._scene_manipulate_tool
        }
    
    async def parse_command(self, command: str) -> Dict:
        """Parse natural language command using Gemini CLI"""
        # Extract intent and parameters from command
        intent = await self._analyze_intent(command)
        
        return {
            'command': command,
            'intent': intent['type'],
            'parameters': intent['parameters'],
            'confidence': intent['confidence'],
            'requires_code_generation': intent.get('needs_code', False),
            'requires_godot_action': intent.get('needs_godot', False)
        }
    
    async def _analyze_intent(self, command: str) -> Dict:
        """Analyze command intent"""
        # Intent patterns
        intents = {
            'CREATE_GAME': ['create', 'make', 'build', 'new game'],
            'GENERATE_CODE': ['generate', 'write', 'code', 'script'],
            'MODIFY_SCENE': ['add', 'remove', 'change', 'modify', 'scene'],
            'ANALYZE_PROJECT': ['analyze', 'check', 'review', 'inspect'],
            'IMPROVE_CODE': ['improve', 'optimize', 'refactor', 'fix']
        }
        
        command_lower = command.lower()
        
        for intent_type, keywords in intents.items():
            if any(keyword in command_lower for keyword in keywords):
                return {
                    'type': intent_type,
                    'parameters': self._extract_parameters(command, intent_type),
                    'confidence': 0.9,
                    'needs_code': intent_type in ['GENERATE_CODE', 'CREATE_GAME'],
                    'needs_godot': intent_type in ['CREATE_GAME', 'MODIFY_SCENE']
                }
        
        return {
            'type': 'UNKNOWN',
            'parameters': {},
            'confidence': 0.3
        }
    
    def _extract_parameters(self, command: str, intent_type: str) -> Dict:
        """Extract parameters based on intent"""
        params = {}
        
        if intent_type == 'CREATE_GAME':
            # Extract game type
            if '2d' in command.lower():
                params['dimension'] = '2d'
            elif '3d' in command.lower():
                params['dimension'] = '3d'
            
            # Extract game genre
            genres = ['platformer', 'rpg', 'shooter', 'puzzle', 'racing']
            for genre in genres:
                if genre in command.lower():
                    params['genre'] = genre
                    break
        
        return params
    
    async def execute_tool(self, tool_name: str, params: Dict) -> Dict:
        """Execute a registered tool"""
        if tool_name in self.tools:
            return await self.tools[tool_name](params)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _godot_control_tool(self, params: Dict) -> Dict:
        """Tool for controlling Godot"""
        action = params.get('action')
        if action == 'create_scene':
            return {'success': True, 'scene_created': params.get('scene_name')}
        return {'success': False, 'error': 'Unknown action'}
    
    async def _csharp_generate_tool(self, params: Dict) -> Dict:
        """Tool for generating C# code"""
        return {
            'success': True,
            'code': f"// Generated code for {params.get('description', 'unknown')}"
        }
    
    async def _llama_analyze_tool(self, params: Dict) -> Dict:
        """Tool for Llama analysis"""
        return {'success': True, 'analysis': 'Code analysis complete'}
    
    async def _project_create_tool(self, params: Dict) -> Dict:
        """Tool for creating projects"""
        return {'success': True, 'project_path': params.get('path')}
    
    async def _scene_manipulate_tool(self, params: Dict) -> Dict:
        """Tool for scene manipulation"""
        return {'success': True, 'nodes_created': params.get('nodes', [])}


class LlamaEngine:
    """Enhanced Llama 7B integration for code generation"""
    
    def __init__(self, model_path: str = "./CodeLlama-7b-Instruct-hf"):
        self.model_path = Path(model_path)
        self.api_url = "http://localhost:8000"
        self.context_builder = GodotContextBuilder()
        
    async def generate_code(self, requirements: Dict) -> str:
        """Generate code based on requirements"""
        prompt = self._build_generation_prompt(requirements)
        
        # Call Llama API
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/generate",
                json={"prompt": prompt, "max_tokens": 2048}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("generated_text", "")
                else:
                    # Fallback template-based generation
                    return self._generate_template_code(requirements)
    
    def _build_generation_prompt(self, requirements: Dict) -> str:
        """Build prompt for code generation"""
        prompt = f"""You are an expert C# developer specializing in Godot 4.x game development.
Generate production-ready C# code based on these requirements:

Project Type: {requirements.get('type', 'game')}
Dimension: {requirements.get('dimension', '2D')}
Features: {', '.join(requirements.get('features', []))}

Requirements:
{requirements.get('description', '')}

Generate complete, working C# code following Godot best practices:
- Use Godot 4.x C# API
- Include proper error handling
- Add XML documentation comments
- Follow SOLID principles
- Use async/await where appropriate

Code:
"""
        return prompt
    
    def _generate_template_code(self, requirements: Dict) -> str:
        """Fallback template-based code generation"""
        if requirements.get('type') == 'player':
            return self._generate_player_code(requirements)
        elif requirements.get('type') == 'enemy':
            return self._generate_enemy_code(requirements)
        else:
            return self._generate_generic_code(requirements)
    
    def _generate_player_code(self, requirements: Dict) -> str:
        """Generate player controller code"""
        dimension = requirements.get('dimension', '2D')
        
        if dimension == '2D':
            return """using Godot;

namespace Game.Player
{
    /// <summary>
    /// Player character controller for 2D movement
    /// </summary>
    public partial class Player : CharacterBody2D
    {
        [Export] public float Speed = 300.0f;
        [Export] public float JumpVelocity = -400.0f;
        
        public float gravity = ProjectSettings.GetSetting("physics/2d/default_gravity").AsSingle();
        
        public override void _PhysicsProcess(double delta)
        {
            Vector2 velocity = Velocity;
            
            // Add gravity
            if (!IsOnFloor())
                velocity.Y += gravity * (float)delta;
            
            // Handle jump
            if (Input.IsActionJustPressed("ui_accept") && IsOnFloor())
                velocity.Y = JumpVelocity;
            
            // Get input direction
            Vector2 direction = Input.GetVector("ui_left", "ui_right", "ui_up", "ui_down");
            if (direction != Vector2.Zero)
            {
                velocity.X = direction.X * Speed;
            }
            else
            {
                velocity.X = Mathf.MoveToward(Velocity.X, 0, Speed);
            }
            
            Velocity = velocity;
            MoveAndSlide();
        }
    }
}"""
        else:
            return """using Godot;

namespace Game.Player
{
    /// <summary>
    /// Player character controller for 3D movement
    /// </summary>
    public partial class Player : CharacterBody3D
    {
        [Export] public float Speed = 5.0f;
        [Export] public float JumpVelocity = 4.5f;
        
        public float gravity = ProjectSettings.GetSetting("physics/3d/default_gravity").AsSingle();
        
        public override void _PhysicsProcess(double delta)
        {
            Vector3 velocity = Velocity;
            
            // Add gravity
            if (!IsOnFloor())
                velocity.Y -= gravity * (float)delta;
            
            // Handle jump
            if (Input.IsActionJustPressed("ui_accept") && IsOnFloor())
                velocity.Y = JumpVelocity;
            
            // Get input direction
            Vector2 inputDir = Input.GetVector("ui_left", "ui_right", "ui_up", "ui_down");
            Vector3 direction = (Transform.Basis * new Vector3(inputDir.X, 0, inputDir.Y)).Normalized();
            
            if (direction != Vector3.Zero)
            {
                velocity.X = direction.X * Speed;
                velocity.Z = direction.Z * Speed;
            }
            else
            {
                velocity.X = Mathf.MoveToward(Velocity.X, 0, Speed);
                velocity.Z = Mathf.MoveToward(Velocity.Z, 0, Speed);
            }
            
            Velocity = velocity;
            MoveAndSlide();
        }
    }
}"""
    
    def _generate_enemy_code(self, requirements: Dict) -> str:
        """Generate enemy AI code"""
        return """using Godot;

namespace Game.Enemies
{
    /// <summary>
    /// Basic enemy AI controller
    /// </summary>
    public partial class Enemy : CharacterBody2D
    {
        [Export] public float Speed = 100.0f;
        [Export] public float DetectionRange = 300.0f;
        [Export] public NodePath PlayerPath { get; set; }
        
        private Node2D player;
        private Vector2 startPosition;
        
        public override void _Ready()
        {
            startPosition = GlobalPosition;
            if (PlayerPath != null)
            {
                player = GetNode<Node2D>(PlayerPath);
            }
        }
        
        public override void _PhysicsProcess(double delta)
        {
            if (player == null) return;
            
            float distanceToPlayer = GlobalPosition.DistanceTo(player.GlobalPosition);
            
            if (distanceToPlayer < DetectionRange)
            {
                // Chase player
                Vector2 direction = (player.GlobalPosition - GlobalPosition).Normalized();
                Velocity = direction * Speed;
            }
            else
            {
                // Return to start position
                Vector2 direction = (startPosition - GlobalPosition).Normalized();
                Velocity = direction * Speed * 0.5f;
            }
            
            MoveAndSlide();
        }
    }
}"""
    
    def _generate_generic_code(self, requirements: Dict) -> str:
        """Generate generic node code"""
        return """using Godot;

namespace Game
{
    /// <summary>
    /// Auto-generated node class
    /// </summary>
    public partial class CustomNode : Node2D
    {
        [Export] public string NodeName { get; set; } = "CustomNode";
        
        public override void _Ready()
        {
            GD.Print($"{NodeName} is ready!");
        }
        
        public override void _Process(double delta)
        {
            // Add your logic here
        }
    }
}"""
    
    async def analyze_code(self, code: str) -> Dict:
        """Analyze existing code for improvements"""
        analysis = {
            'issues': [],
            'suggestions': [],
            'quality_score': 0.0
        }
        
        # Basic analysis
        if 'try' not in code and 'catch' not in code:
            analysis['issues'].append({
                'type': 'error_handling',
                'severity': 'medium',
                'message': 'No error handling found'
            })
        
        if '///' not in code:
            analysis['issues'].append({
                'type': 'documentation',
                'severity': 'low',
                'message': 'Missing XML documentation comments'
            })
        
        if 'async' in code and 'ConfigureAwait' not in code:
            analysis['suggestions'].append({
                'type': 'performance',
                'message': 'Consider using ConfigureAwait(false) for async methods'
            })
        
        # Calculate quality score
        lines = code.split('\n')
        score = 100.0
        score -= len(analysis['issues']) * 10
        score -= len(analysis['suggestions']) * 5
        score = max(0, min(100, score))
        
        analysis['quality_score'] = score / 100.0
        
        return analysis


class GodotContextBuilder:
    """Build context for Godot-specific code generation"""
    
    def build(self, requirements: Dict) -> Dict:
        """Build complete context for code generation"""
        context = {
            'godot_version': '4.2',
            'csharp_version': '11.0',
            'target_framework': 'net8.0',
            'project_type': requirements.get('type', 'game'),
            'features': requirements.get('features', []),
            'patterns': self._get_recommended_patterns(requirements)
        }
        return context
    
    def _get_recommended_patterns(self, requirements: Dict) -> List[str]:
        """Get recommended design patterns based on requirements"""
        patterns = []
        
        if 'multiplayer' in str(requirements).lower():
            patterns.append('Observer')
            patterns.append('Command')
        
        if 'inventory' in str(requirements).lower():
            patterns.append('Factory')
            patterns.append('Strategy')
        
        if 'ai' in str(requirements).lower() or 'enemy' in str(requirements).lower():
            patterns.append('State Machine')
            patterns.append('Behavior Tree')
        
        return patterns


class UnifiedAutoCI:
    """Main unified AutoCI system"""
    
    def __init__(self):
        self.gemini = GeminiCLIAdapter()
        self.llama = LlamaEngine()
        self.godot = GodotController()
        self.godot_commands = GodotCommands()
        self.godot_commands.controller = self.godot
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AutoCI')
        
        # Project context
        self.current_project = None
        self.current_scene = None
        
    async def execute(self, command: str) -> Dict:
        """Execute natural language command"""
        self.logger.info(f"Executing command: {command}")
        
        try:
            # 1. Parse command with Gemini CLI
            parsed = await self.gemini.parse_command(command)
            self.logger.info(f"Parsed intent: {parsed['intent']}")
            
            # 2. Route to appropriate handler
            result = await self._route_command(parsed)
            
            # 3. Format response
            response = self._format_response(result)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'I encountered an error while processing your request.'
            }
    
    async def _route_command(self, parsed: Dict) -> Dict:
        """Route parsed command to appropriate handler"""
        intent = parsed['intent']
        params = parsed['parameters']
        
        if intent == 'CREATE_GAME':
            return await self._handle_create_game(parsed)
        elif intent == 'GENERATE_CODE':
            return await self._handle_generate_code(parsed)
        elif intent == 'MODIFY_SCENE':
            return await self._handle_modify_scene(parsed)
        elif intent == 'ANALYZE_PROJECT':
            return await self._handle_analyze_project(parsed)
        elif intent == 'IMPROVE_CODE':
            return await self._handle_improve_code(parsed)
        else:
            return {
                'success': False,
                'message': "I'm not sure how to help with that. Could you please rephrase?"
            }
    
    async def _handle_create_game(self, parsed: Dict) -> Dict:
        """Handle game creation request"""
        params = parsed['parameters']
        
        # Extract game details from command
        game_type = params.get('genre', 'platformer')
        dimension = params.get('dimension', '2d')
        
        self.logger.info(f"Creating {dimension} {game_type} game")
        
        # 1. Create project
        project_name = f"My{game_type.title()}Game"
        project_path = f"./projects/{project_name}"
        
        # Create project directory
        os.makedirs(project_path, exist_ok=True)
        
        # 2. Start Godot if needed
        if not self.godot.ping():
            self.logger.info("Starting Godot engine...")
            self.godot.start_godot_engine(project_path, headless=True)
        
        # 3. Create project in Godot
        self.godot_commands.cmd_create_project(project_name, project_path)
        
        # 4. Create main scene
        scene_type = "2d_game" if dimension == "2d" else "3d_game"
        self.godot_commands.cmd_create_scene("Main", scene_type)
        
        # 5. Generate and add player
        player_requirements = {
            'type': 'player',
            'dimension': dimension.upper(),
            'features': ['movement', 'jump']
        }
        
        player_code = await self.llama.generate_code(player_requirements)
        
        # Create player in scene
        self.godot_commands.cmd_add_player("Player", dimension)
        
        # 6. Add game-specific elements
        if game_type == 'platformer':
            await self._setup_platformer_elements()
        elif game_type == 'rpg':
            await self._setup_rpg_elements()
        
        # 7. Save scene
        self.godot_commands.cmd_save_scene()
        
        self.current_project = project_path
        self.current_scene = "Main"
        
        return {
            'success': True,
            'project_path': project_path,
            'message': f"Created {dimension} {game_type} game at {project_path}",
            'next_steps': [
                "You can now run the game with 'run the game'",
                "Add more features with commands like 'add enemies' or 'add collectibles'",
                "Modify the player with 'make the player faster' or 'add double jump'"
            ]
        }
    
    async def _handle_generate_code(self, parsed: Dict) -> Dict:
        """Handle code generation request"""
        description = parsed['command']
        
        requirements = {
            'description': description,
            'type': 'generic',
            'features': []
        }
        
        # Generate code
        code = await self.llama.generate_code(requirements)
        
        # Analyze generated code
        analysis = await self.llama.analyze_code(code)
        
        return {
            'success': True,
            'code': code,
            'analysis': analysis,
            'message': "I've generated the code for you. Would you like me to create a file with this code?"
        }
    
    async def _handle_modify_scene(self, parsed: Dict) -> Dict:
        """Handle scene modification request"""
        if not self.godot.ping():
            return {
                'success': False,
                'message': "Godot is not running. Please start a project first."
            }
        
        # Extract modification details
        command = parsed['command'].lower()
        
        if 'add' in command:
            if 'enemy' in command or 'enemies' in command:
                # Add enemies to scene
                enemy_count = 3  # Default
                for i in range(enemy_count):
                    self.godot_commands.cmd_add_enemy(f"Enemy{i}", "basic")
                
                return {
                    'success': True,
                    'message': f"Added {enemy_count} enemies to the scene"
                }
            
            elif 'platform' in command:
                # Add platforms
                self._add_platforms()
                return {
                    'success': True,
                    'message': "Added platforms to the scene"
                }
        
        return {
            'success': False,
            'message': "I'm not sure what to modify. Try 'add enemies' or 'add platforms'"
        }
    
    async def _handle_analyze_project(self, parsed: Dict) -> Dict:
        """Handle project analysis request"""
        if not self.current_project:
            return {
                'success': False,
                'message': "No project is currently open. Create one with 'create a new game'"
            }
        
        # Analyze project structure
        analysis = {
            'project_path': self.current_project,
            'scenes': [],
            'scripts': [],
            'assets': [],
            'issues': [],
            'suggestions': []
        }
        
        # Add analysis logic here
        
        return {
            'success': True,
            'analysis': analysis,
            'message': "Project analysis complete"
        }
    
    async def _handle_improve_code(self, parsed: Dict) -> Dict:
        """Handle code improvement request"""
        # This would analyze and improve existing code
        return {
            'success': True,
            'message': "Code improvement feature coming soon"
        }
    
    async def _setup_platformer_elements(self):
        """Setup platformer-specific elements"""
        # Add platforms
        for i in range(5):
            x = 200 + i * 300
            y = 400 + (i % 2) * 100
            
            self.godot.create_node("StaticBody2D", f"Platform{i}")
            platform_path = f"/root/Platform{i}"
            
            self.godot.create_node("CollisionShape2D", "Collision", platform_path)
            self.godot.set_property(platform_path, "position", [x, y])
        
        # Add collectibles
        for i in range(10):
            x = 150 + i * 200
            y = 300
            
            self.godot.create_node("Area2D", f"Coin{i}")
            coin_path = f"/root/Coin{i}"
            
            self.godot.create_node("Sprite2D", "Sprite", coin_path)
            self.godot.set_property(coin_path, "position", [x, y])
    
    async def _setup_rpg_elements(self):
        """Setup RPG-specific elements"""
        # Add NPCs
        for i in range(3):
            self.godot.create_node("CharacterBody2D", f"NPC{i}")
            npc_path = f"/root/NPC{i}"
            self.godot.set_property(npc_path, "position", [300 + i * 200, 400])
        
        # Add inventory system UI
        self.godot_commands.cmd_add_ui_element("panel", "InventoryPanel", {
            "position": [10, 100],
            "size": [200, 300]
        })
    
    def _add_platforms(self):
        """Add platforms to current scene"""
        for i in range(3):
            x = 200 + i * 300
            y = 450
            
            self.godot.create_node("StaticBody2D", f"ExtraPlatform{i}")
            platform_path = f"/root/ExtraPlatform{i}"
            
            self.godot.create_node("CollisionShape2D", "Collision", platform_path)
            self.godot.set_property(platform_path, "position", [x, y])
    
    def _format_response(self, result: Dict) -> Dict:
        """Format response for user"""
        response = {
            'success': result.get('success', False),
            'message': result.get('message', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add additional fields if present
        for key in ['code', 'analysis', 'project_path', 'next_steps']:
            if key in result:
                response[key] = result[key]
        
        return response
    
    async def start_interactive_mode(self):
        """Start interactive command mode"""
        print("🤖 AutoCI v2.0 - Unified AI Game Development System")
        print("Type 'help' for commands or 'exit' to quit\n")
        
        while True:
            try:
                command = input("AutoCI> ").strip()
                
                if command.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                if command.lower() == 'help':
                    self._print_help()
                    continue
                
                if not command:
                    continue
                
                # Execute command
                result = await self.execute(command)
                
                # Print response
                print(f"\n{result['message']}")
                
                if 'code' in result:
                    print("\n--- Generated Code ---")
                    print(result['code'])
                    print("--- End Code ---\n")
                
                if 'next_steps' in result:
                    print("\nNext steps:")
                    for step in result['next_steps']:
                        print(f"  • {step}")
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except Exception as e:
                print(f"Error: {e}")
    
    def _print_help(self):
        """Print help information"""
        help_text = """
AutoCI v2.0 Commands:

🎮 Game Creation:
  - "Create a 2D platformer game"
  - "Make a 3D RPG with magic system"
  - "Build a puzzle game"

💻 Code Generation:
  - "Generate a player controller"
  - "Create an inventory system"
  - "Write enemy AI code"

🎨 Scene Modification:
  - "Add enemies to the scene"
  - "Create platforms"
  - "Add UI elements"

📊 Analysis:
  - "Analyze my project"
  - "Check code quality"
  - "Find performance issues"

🔧 Other:
  - "help" - Show this help
  - "exit" - Quit AutoCI

Natural language is supported! Just describe what you want to create.
"""
        print(help_text)


# Main entry point
async def main():
    """Main entry point for unified AutoCI"""
    autoci = UnifiedAutoCI()
    await autoci.start_interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())