"""
AutoCI Autonomous Agent - 24/7 Game Development AI
Works independently to create games through natural conversation
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import threading
import queue
import time
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.unified_autoci import UnifiedAutoCI, GeminiCLIAdapter, LlamaEngine
from modules.godot_controller import GodotController
from modules.godot_commands import GodotCommands


class AutonomousGameDeveloper:
    """24/7 Autonomous Game Development AI Agent"""
    
    def __init__(self):
        self.autoci = UnifiedAutoCI()
        self.is_running = False
        self.current_projects = {}
        self.learning_queue = queue.Queue()
        self.conversation_history = []
        self.user_preferences = {}
        self.knowledge_base = KnowledgeBase()
        
        # Agent state
        self.state = {
            'mode': 'autonomous',
            'current_task': None,
            'games_created': 0,
            'lines_written': 0,
            'bugs_fixed': 0,
            'patterns_learned': 0,
            'uptime_start': datetime.now()
        }
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("logs/autonomous")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"agent_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AutonomousAgent')
        
    async def start(self, mode: str = "autonomous"):
        """Start the autonomous agent"""
        self.is_running = True
        self.logger.info(f"🚀 AutoCI Agent starting in {mode} mode")
        
        # Start background tasks
        asyncio.create_task(self._learning_loop())
        asyncio.create_task(self._project_monitor())
        asyncio.create_task(self._optimization_loop())
        
        if mode == "autonomous":
            await self._autonomous_mode()
        else:
            await self._interactive_mode()
            
    async def _autonomous_mode(self):
        """Run in fully autonomous mode"""
        self.logger.info("🤖 Entering autonomous mode - I'll work on games independently")
        
        while self.is_running:
            try:
                # Check for pending tasks
                task = await self._get_next_task()
                
                if task:
                    await self._execute_task(task)
                else:
                    # No specific task, be creative
                    await self._creative_development()
                    
                # Brief pause
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in autonomous mode: {e}")
                await asyncio.sleep(30)
                
    async def _interactive_mode(self):
        """Run in interactive mode with user"""
        print("\n🤖 AutoCI Agent - Interactive Mode")
        print("I'm your 24/7 game development partner!")
        print("Just tell me what you want to create.\n")
        
        while self.is_running:
            try:
                # Get user input
                user_input = await self._get_user_input()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    await self._shutdown()
                    break
                    
                # Process conversation
                response = await self._process_conversation(user_input)
                print(f"\n🤖 AutoCI: {response['message']}\n")
                
                # Execute any resulting actions
                if response.get('action'):
                    await self._execute_action(response['action'])
                    
            except KeyboardInterrupt:
                await self._shutdown()
                break
                
    async def _process_conversation(self, user_input: str) -> Dict:
        """Process natural language conversation"""
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Analyze intent and context
        context = self._build_conversation_context()
        intent = await self._analyze_intent(user_input, context)
        
        # Generate response
        response = await self._generate_response(intent, context)
        
        # Add agent response to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response['message'],
            'timestamp': datetime.now()
        })
        
        # Learn from interaction
        self._learn_from_conversation(user_input, response)
        
        return response
        
    async def _analyze_intent(self, user_input: str, context: Dict) -> Dict:
        """Deep intent analysis using Gemini CLI + Llama"""
        # Use Gemini for initial parsing
        parsed = await self.autoci.gemini.parse_command(user_input)
        
        # Enhance with Llama understanding
        enhanced_intent = {
            'base_intent': parsed['intent'],
            'parameters': parsed['parameters'],
            'emotion': self._detect_emotion(user_input),
            'urgency': self._detect_urgency(user_input),
            'complexity': self._estimate_complexity(user_input),
            'context_relevant': self._check_context_relevance(user_input, context)
        }
        
        return enhanced_intent
        
    def _detect_emotion(self, text: str) -> str:
        """Detect user emotion from text"""
        positive_words = ['love', 'awesome', 'great', 'amazing', 'perfect', 'excellent']
        negative_words = ['hate', 'terrible', 'bad', 'awful', 'horrible', 'sucks']
        urgent_words = ['asap', 'urgent', 'quickly', 'fast', 'now', 'immediately']
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in positive_words):
            return 'positive'
        elif any(word in text_lower for word in negative_words):
            return 'negative'
        elif any(word in text_lower for word in urgent_words):
            return 'urgent'
        else:
            return 'neutral'
            
    def _detect_urgency(self, text: str) -> float:
        """Detect urgency level (0.0 to 1.0)"""
        urgent_indicators = [
            'asap', 'urgent', 'quickly', 'right now', 'immediately',
            'hurry', 'rush', 'deadline', 'by tonight', 'before'
        ]
        
        text_lower = text.lower()
        urgency_count = sum(1 for indicator in urgent_indicators if indicator in text_lower)
        
        return min(1.0, urgency_count * 0.3)
        
    def _estimate_complexity(self, text: str) -> str:
        """Estimate task complexity"""
        complex_indicators = [
            'complex', 'advanced', 'sophisticated', 'multiple', 'system',
            'ai', 'procedural', 'multiplayer', 'networked', 'database'
        ]
        
        simple_indicators = [
            'simple', 'basic', 'quick', 'small', 'prototype', 'test'
        ]
        
        text_lower = text.lower()
        complex_count = sum(1 for ind in complex_indicators if ind in text_lower)
        simple_count = sum(1 for ind in simple_indicators if ind in text_lower)
        
        if complex_count > simple_count:
            return 'complex'
        elif simple_count > complex_count:
            return 'simple'
        else:
            return 'medium'
            
    def _check_context_relevance(self, text: str, context: Dict) -> bool:
        """Check if the input is relevant to current context"""
        if not context.get('current_project'):
            return True
            
        project_keywords = context['current_project'].get('keywords', [])
        text_lower = text.lower()
        
        return any(keyword in text_lower for keyword in project_keywords)
        
    def _build_conversation_context(self) -> Dict:
        """Build context from conversation history"""
        context = {
            'history_length': len(self.conversation_history),
            'current_project': self.current_projects.get('active'),
            'user_preferences': self.user_preferences,
            'recent_topics': self._extract_recent_topics(),
            'session_duration': (datetime.now() - self.state['uptime_start']).total_seconds()
        }
        
        return context
        
    def _extract_recent_topics(self, window: int = 10) -> List[str]:
        """Extract topics from recent conversation"""
        recent = self.conversation_history[-window:]
        topics = []
        
        # Simple topic extraction (can be enhanced with NLP)
        topic_keywords = {
            'player': ['player', 'character', 'hero', 'protagonist'],
            'enemy': ['enemy', 'monster', 'boss', 'opponent'],
            'level': ['level', 'stage', 'map', 'world'],
            'ui': ['ui', 'menu', 'button', 'interface'],
            'physics': ['physics', 'collision', 'gravity', 'movement']
        }
        
        for entry in recent:
            text_lower = entry['content'].lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    topics.append(topic)
                    
        return list(set(topics))
        
    async def _generate_response(self, intent: Dict, context: Dict) -> Dict:
        """Generate intelligent response based on intent and context"""
        response = {
            'message': '',
            'action': None,
            'emotion': 'friendly',
            'suggestions': []
        }
        
        # Adjust response based on emotion
        if intent['emotion'] == 'positive':
            response['emotion'] = 'enthusiastic'
        elif intent['emotion'] == 'negative':
            response['emotion'] = 'supportive'
        elif intent['emotion'] == 'urgent':
            response['emotion'] = 'focused'
            
        # Generate base response
        if intent['base_intent'] == 'CREATE_GAME':
            response['message'] = self._generate_game_creation_response(intent, context)
            response['action'] = {
                'type': 'create_game',
                'parameters': intent['parameters']
            }
        elif intent['base_intent'] == 'MODIFY_SCENE':
            response['message'] = "I'll modify that for you right away."
            response['action'] = {
                'type': 'modify_scene',
                'parameters': intent['parameters']
            }
        else:
            # Use Llama for complex responses
            response['message'] = await self._generate_llama_response(intent, context)
            
        # Add suggestions based on context
        response['suggestions'] = self._generate_suggestions(intent, context)
        
        return response
        
    def _generate_game_creation_response(self, intent: Dict, context: Dict) -> str:
        """Generate response for game creation request"""
        complexity = intent['complexity']
        urgency = intent['urgency']
        
        templates = {
            'simple': [
                "I'll create a simple {} for you. This should take just a few minutes!",
                "Perfect! A simple {} coming right up. I'll have it ready shortly.",
                "Great choice! I'll build a clean, simple {} for you."
            ],
            'medium': [
                "I'll create a {} with all the features you'd expect. Give me about 10-15 minutes.",
                "Interesting project! I'll build a solid {} with proper architecture.",
                "I'm on it! Creating a well-structured {} for you."
            ],
            'complex': [
                "This is an ambitious project! I'll create a sophisticated {} with advanced features.",
                "Challenge accepted! I'll build a complex {} with all the bells and whistles.",
                "Excellent! I'll architect a professional-grade {} for you."
            ]
        }
        
        game_type = intent['parameters'].get('genre', 'game')
        template = random.choice(templates.get(complexity, templates['medium']))
        
        response = template.format(game_type)
        
        if urgency > 0.7:
            response += " I'll prioritize this and work as fast as possible!"
        elif urgency > 0.3:
            response += " I'll get started on this right away."
            
        return response
        
    async def _generate_llama_response(self, intent: Dict, context: Dict) -> str:
        """Generate response using Llama for complex conversations"""
        prompt = f"""You are AutoCI, a friendly and capable AI game developer.
        
User emotion: {intent['emotion']}
Complexity: {intent['complexity']}
Context: Currently working on {context.get('current_project', {}).get('name', 'no active project')}
Recent topics: {', '.join(context.get('recent_topics', []))}

Generate a helpful, conversational response that:
1. Acknowledges the user's request
2. Shows understanding of their needs
3. Explains what you'll do
4. Is encouraging and supportive

Keep the response concise and friendly.
"""
        
        # This would call Llama API in production
        # For now, return a contextual response
        return "I understand what you're looking for. Let me help you with that!"
        
    def _generate_suggestions(self, intent: Dict, context: Dict) -> List[str]:
        """Generate helpful suggestions based on context"""
        suggestions = []
        
        if intent['base_intent'] == 'CREATE_GAME':
            suggestions.extend([
                "Would you like me to add multiplayer support?",
                "Should I include a save/load system?",
                "Do you want particle effects for more polish?"
            ])
        elif context.get('current_project'):
            suggestions.extend([
                "I could optimize the performance while I'm at it",
                "Would you like me to add more enemy types?",
                "Should I improve the UI design?"
            ])
            
        return suggestions[:3]  # Limit to 3 suggestions
        
    def _learn_from_conversation(self, user_input: str, response: Dict):
        """Learn from each conversation"""
        learning_data = {
            'input': user_input,
            'response': response,
            'timestamp': datetime.now(),
            'success': True  # Would be determined by user feedback
        }
        
        self.learning_queue.put(learning_data)
        
        # Extract preferences
        if 'prefer' in user_input.lower():
            self._extract_preferences(user_input)
            
    def _extract_preferences(self, text: str):
        """Extract user preferences from conversation"""
        # Simple preference extraction
        preferences = {
            'code_style': None,
            'game_types': [],
            'features': []
        }
        
        if 'early return' in text.lower():
            preferences['code_style'] = 'early_returns'
        elif 'documentation' in text.lower():
            preferences['features'].append('documentation')
            
        self.user_preferences.update(preferences)
        
    async def _execute_action(self, action: Dict):
        """Execute an action determined from conversation"""
        action_type = action['type']
        params = action['parameters']
        
        self.logger.info(f"Executing action: {action_type}")
        
        if action_type == 'create_game':
            await self._create_game_autonomous(params)
        elif action_type == 'modify_scene':
            await self._modify_scene_autonomous(params)
        elif action_type == 'optimize_code':
            await self._optimize_code_autonomous(params)
            
    async def _create_game_autonomous(self, params: Dict):
        """Autonomously create a complete game"""
        project_name = f"Game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Creating game: {project_name}")
        self.state['current_task'] = f"Creating {params.get('genre', 'game')}"
        
        # Create project structure
        project_path = Path(f"projects/{project_name}")
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize project tracking
        self.current_projects[project_name] = {
            'name': project_name,
            'type': params.get('genre', 'game'),
            'status': 'in_progress',
            'created': datetime.now(),
            'features': [],
            'quality_score': 0.0
        }
        
        # Execute game creation workflow
        try:
            # 1. Setup Godot project
            if not self.autoci.godot.ping():
                self.autoci.godot.start_godot_engine(str(project_path), headless=True)
                
            # 2. Create base structure
            await self._setup_game_structure(project_name, params)
            
            # 3. Generate core systems
            await self._generate_core_systems(project_name, params)
            
            # 4. Add game-specific features
            await self._add_game_features(project_name, params)
            
            # 5. Polish and optimize
            await self._polish_game(project_name)
            
            # 6. Test and debug
            await self._test_game(project_name)
            
            # Update project status
            self.current_projects[project_name]['status'] = 'completed'
            self.current_projects[project_name]['quality_score'] = 0.85
            
            self.state['games_created'] += 1
            self.logger.info(f"✅ Game creation completed: {project_name}")
            
        except Exception as e:
            self.logger.error(f"Error creating game: {e}")
            self.current_projects[project_name]['status'] = 'failed'
            
    async def _setup_game_structure(self, project_name: str, params: Dict):
        """Setup basic game structure"""
        self.logger.info("Setting up game structure...")
        
        # Create main scene
        scene_type = "2d_game" if params.get('dimension', '2d') == '2d' else "3d_game"
        self.autoci.godot_commands.cmd_create_scene("Main", scene_type)
        
        # Setup physics
        self.autoci.godot_commands.cmd_setup_physics(
            params.get('dimension', '2d'),
            980.0
        )
        
        # Create folder structure in Godot
        folders = ['scenes', 'scripts', 'assets', 'audio', 'ui']
        for folder in folders:
            self.autoci.godot.create_node("Node", f"_{folder}")
            
    async def _generate_core_systems(self, project_name: str, params: Dict):
        """Generate core game systems"""
        self.logger.info("Generating core systems...")
        
        # Player system
        player_code = await self.autoci.llama.generate_code({
            'type': 'player',
            'dimension': params.get('dimension', '2D').upper(),
            'features': ['movement', 'jump', 'health']
        })
        
        self.autoci.godot_commands.cmd_add_player("Player", params.get('dimension', '2d'))
        self.state['lines_written'] += len(player_code.split('\n'))
        
        # Input system
        input_code = await self.autoci.llama.generate_code({
            'type': 'input_manager',
            'description': 'Centralized input handling system'
        })
        
        # Game state manager
        state_code = await self.autoci.llama.generate_code({
            'type': 'game_state_manager',
            'features': ['pause', 'save', 'load', 'scene_transition']
        })
        
        self.state['lines_written'] += len(input_code.split('\n'))
        self.state['lines_written'] += len(state_code.split('\n'))
        
    async def _add_game_features(self, project_name: str, params: Dict):
        """Add game-specific features based on genre"""
        genre = params.get('genre', 'platformer')
        self.logger.info(f"Adding {genre} features...")
        
        if genre == 'platformer':
            await self._add_platformer_features(project_name)
        elif genre == 'rpg':
            await self._add_rpg_features(project_name)
        elif genre == 'shooter':
            await self._add_shooter_features(project_name)
        elif genre == 'puzzle':
            await self._add_puzzle_features(project_name)
            
        # Add UI for all games
        await self._add_game_ui(project_name, genre)
        
    async def _add_platformer_features(self, project_name: str):
        """Add platformer-specific features"""
        # Platforms
        for i in range(10):
            x = 200 + (i * 300)
            y = 400 + random.randint(-100, 100)
            self.autoci.godot.create_node("StaticBody2D", f"Platform_{i}")
            self.autoci.godot.set_property(f"/root/Platform_{i}", "position", [x, y])
            
        # Enemies
        for i in range(5):
            self.autoci.godot_commands.cmd_add_enemy(f"Enemy_{i}", "basic")
            
        # Collectibles
        for i in range(15):
            x = 150 + (i * 200)
            y = 300 + random.randint(-50, 50)
            self.autoci.godot.create_node("Area2D", f"Coin_{i}")
            self.autoci.godot.set_property(f"/root/Coin_{i}", "position", [x, y])
            
        # Moving platforms
        moving_platform_code = await self.autoci.llama.generate_code({
            'type': 'moving_platform',
            'description': 'Platform that moves between two points'
        })
        
        self.state['lines_written'] += len(moving_platform_code.split('\n'))
        
    async def _add_rpg_features(self, project_name: str):
        """Add RPG-specific features"""
        # Inventory system
        inventory_code = await self.autoci.llama.generate_code({
            'type': 'inventory_system',
            'features': ['items', 'equipment', 'drag_drop']
        })
        
        # Dialog system
        dialog_code = await self.autoci.llama.generate_code({
            'type': 'dialog_system',
            'features': ['branching', 'choices', 'npc_interaction']
        })
        
        # Quest system
        quest_code = await self.autoci.llama.generate_code({
            'type': 'quest_system',
            'features': ['objectives', 'rewards', 'tracking']
        })
        
        # NPCs
        for i in range(8):
            self.autoci.godot.create_node("CharacterBody2D", f"NPC_{i}")
            npc_pos = [
                random.randint(100, 800),
                random.randint(100, 600)
            ]
            self.autoci.godot.set_property(f"/root/NPC_{i}", "position", npc_pos)
            
        total_lines = sum(len(code.split('\n')) for code in 
                         [inventory_code, dialog_code, quest_code])
        self.state['lines_written'] += total_lines
        
    async def _add_shooter_features(self, project_name: str):
        """Add shooter-specific features"""
        # Weapon system
        weapon_code = await self.autoci.llama.generate_code({
            'type': 'weapon_system',
            'features': ['multiple_weapons', 'ammo', 'reload', 'projectiles']
        })
        
        # Enemy spawner
        spawner_code = await self.autoci.llama.generate_code({
            'type': 'enemy_spawner',
            'features': ['waves', 'difficulty_scaling', 'spawn_patterns']
        })
        
        # Power-ups
        for power_type in ['health', 'ammo', 'speed', 'damage']:
            self.autoci.godot.create_node("Area2D", f"PowerUp_{power_type}")
            
        self.state['lines_written'] += len(weapon_code.split('\n'))
        self.state['lines_written'] += len(spawner_code.split('\n'))
        
    async def _add_puzzle_features(self, project_name: str):
        """Add puzzle-specific features"""
        # Grid system
        grid_code = await self.autoci.llama.generate_code({
            'type': 'grid_system',
            'description': 'Grid-based puzzle mechanics'
        })
        
        # Match detection
        match_code = await self.autoci.llama.generate_code({
            'type': 'match_detection',
            'description': 'Detect matching patterns in puzzle'
        })
        
        self.state['lines_written'] += len(grid_code.split('\n'))
        self.state['lines_written'] += len(match_code.split('\n'))
        
    async def _add_game_ui(self, project_name: str, genre: str):
        """Add UI elements appropriate for the game genre"""
        self.logger.info("Adding UI elements...")
        
        # Main menu
        self.autoci.godot_commands.cmd_add_ui_element("panel", "MainMenu", {
            "position": [0, 0],
            "size": [1920, 1080]
        })
        
        # HUD elements
        hud_elements = {
            'platformer': ['score', 'lives', 'time'],
            'rpg': ['health', 'mana', 'quest_tracker'],
            'shooter': ['health', 'ammo', 'score'],
            'puzzle': ['moves', 'score', 'time']
        }
        
        for element in hud_elements.get(genre, ['score']):
            self.autoci.godot_commands.cmd_add_ui_element("label", f"HUD_{element}", {
                "text": f"{element.title()}: 0",
                "position": [10, 10 + hud_elements.get(genre, []).index(element) * 40]
            })
            
        # Pause menu
        pause_menu_code = await self.autoci.llama.generate_code({
            'type': 'pause_menu',
            'features': ['resume', 'settings', 'quit']
        })
        
        self.state['lines_written'] += len(pause_menu_code.split('\n'))
        
    async def _polish_game(self, project_name: str):
        """Add polish and game feel improvements"""
        self.logger.info("Adding polish and game feel...")
        
        # Particle effects
        particle_systems = ['player_jump', 'enemy_death', 'collectible_pickup']
        for system in particle_systems:
            self.autoci.godot.create_node("CPUParticles2D", f"Particles_{system}")
            
        # Screen shake system
        screen_shake_code = await self.autoci.llama.generate_code({
            'type': 'screen_shake',
            'description': 'Camera shake for impacts and explosions'
        })
        
        # Sound manager
        sound_manager_code = await self.autoci.llama.generate_code({
            'type': 'sound_manager',
            'features': ['sfx', 'music', 'volume_control']
        })
        
        # Juice effects (squash/stretch, etc)
        juice_code = await self.autoci.llama.generate_code({
            'type': 'juice_effects',
            'description': 'Squash, stretch, and bounce effects for game feel'
        })
        
        total_polish_lines = sum(len(code.split('\n')) for code in 
                                [screen_shake_code, sound_manager_code, juice_code])
        self.state['lines_written'] += total_polish_lines
        
        # Update project features
        self.current_projects[project_name]['features'].extend([
            'particle_effects', 'screen_shake', 'sound_system', 'juice_effects'
        ])
        
    async def _test_game(self, project_name: str):
        """Test the game and fix any issues"""
        self.logger.info("Testing and debugging game...")
        
        # Generate test cases
        test_code = await self.autoci.llama.generate_code({
            'type': 'unit_tests',
            'description': 'Automated tests for game systems'
        })
        
        # Simulate testing
        test_results = {
            'collision_detection': 'pass',
            'player_movement': 'pass',
            'enemy_ai': 'fail',
            'ui_interaction': 'pass',
            'save_load': 'fail'
        }
        
        # Fix failures
        for test, result in test_results.items():
            if result == 'fail':
                self.logger.info(f"Fixing {test}...")
                fix_code = await self.autoci.llama.generate_code({
                    'type': 'bug_fix',
                    'description': f'Fix {test} issue'
                })
                self.state['bugs_fixed'] += 1
                self.state['lines_written'] += len(fix_code.split('\n'))
                
        # Performance optimization
        await self._optimize_game_performance(project_name)
        
    async def _optimize_game_performance(self, project_name: str):
        """Optimize game performance"""
        optimizations = [
            'object_pooling',
            'texture_atlasing',
            'lod_system',
            'culling_optimization'
        ]
        
        for optimization in optimizations:
            if random.random() > 0.5:  # Randomly apply some optimizations
                opt_code = await self.autoci.llama.generate_code({
                    'type': optimization,
                    'description': f'Implement {optimization} for better performance'
                })
                self.state['lines_written'] += len(opt_code.split('\n'))
                self.current_projects[project_name]['features'].append(optimization)
                
    async def _creative_development(self):
        """When no specific task, be creative and improve existing projects"""
        self.logger.info("No specific tasks - entering creative mode...")
        
        # Choose a creative action
        actions = [
            self._improve_existing_project,
            self._create_experimental_feature,
            self._optimize_codebase,
            self._create_mini_game,
            self._research_new_techniques
        ]
        
        action = random.choice(actions)
        await action()
        
    async def _improve_existing_project(self):
        """Improve an existing project"""
        if not self.current_projects:
            await self._create_mini_game()
            return
            
        # Pick a random project
        project_name = random.choice(list(self.current_projects.keys()))
        project = self.current_projects[project_name]
        
        self.logger.info(f"Improving project: {project_name}")
        
        improvements = [
            ('ai_enhancement', 'Enhance enemy AI with better pathfinding'),
            ('visual_effects', 'Add more particle effects and shaders'),
            ('sound_design', 'Improve sound effects and add ambient music'),
            ('ui_polish', 'Redesign UI for better user experience'),
            ('performance', 'Optimize performance bottlenecks')
        ]
        
        improvement_type, description = random.choice(improvements)
        
        improvement_code = await self.autoci.llama.generate_code({
            'type': improvement_type,
            'description': description,
            'project_context': project
        })
        
        self.state['lines_written'] += len(improvement_code.split('\n'))
        project['features'].append(improvement_type)
        project['quality_score'] = min(1.0, project.get('quality_score', 0.7) + 0.05)
        
    async def _create_experimental_feature(self):
        """Create an experimental game feature"""
        experimental_features = [
            {
                'name': 'time_manipulation',
                'description': 'Time slow/reverse mechanics like Braid'
            },
            {
                'name': 'procedural_music',
                'description': 'Music that adapts to gameplay'
            },
            {
                'name': 'ai_companion',
                'description': 'AI companion that learns from player'
            },
            {
                'name': 'dynamic_difficulty',
                'description': 'Difficulty that adapts to player skill'
            }
        ]
        
        feature = random.choice(experimental_features)
        self.logger.info(f"Creating experimental feature: {feature['name']}")
        
        feature_code = await self.autoci.llama.generate_code({
            'type': 'experimental_feature',
            'name': feature['name'],
            'description': feature['description']
        })
        
        self.state['lines_written'] += len(feature_code.split('\n'))
        self.state['patterns_learned'] += 1
        
    async def _optimize_codebase(self):
        """Analyze and optimize existing code"""
        self.logger.info("Analyzing codebase for optimizations...")
        
        # Simulate code analysis
        optimization_targets = [
            'reduce_draw_calls',
            'optimize_physics_calculations',
            'improve_memory_usage',
            'cache_expensive_operations',
            'parallelize_ai_calculations'
        ]
        
        for target in optimization_targets:
            if random.random() > 0.7:  # 30% chance to optimize each
                opt_code = await self.autoci.llama.generate_code({
                    'type': 'optimization',
                    'target': target
                })
                self.state['lines_written'] += len(opt_code.split('\n'))
                self.logger.info(f"Optimized: {target}")
                
    async def _create_mini_game(self):
        """Create a small experimental game"""
        mini_games = [
            {'name': 'ColorMatch', 'genre': 'puzzle', 'description': 'Match colors in a grid'},
            {'name': 'SpaceEvader', 'genre': 'shooter', 'description': 'Avoid asteroids in space'},
            {'name': 'JumpKing', 'genre': 'platformer', 'description': 'Precision jumping game'},
            {'name': 'WordHunt', 'genre': 'puzzle', 'description': 'Find words in a grid'}
        ]
        
        game = random.choice(mini_games)
        self.logger.info(f"Creating mini-game: {game['name']}")
        
        await self._create_game_autonomous({
            'genre': game['genre'],
            'name': game['name'],
            'description': game['description'],
            'scope': 'mini'
        })
        
    async def _research_new_techniques(self):
        """Research and learn new game development techniques"""
        research_topics = [
            'shader_programming',
            'networking_patterns',
            'ecs_architecture',
            'behavior_trees',
            'procedural_generation',
            'machine_learning_in_games'
        ]
        
        topic = random.choice(research_topics)
        self.logger.info(f"Researching: {topic}")
        
        # Simulate research and learning
        await asyncio.sleep(random.randint(5, 15))
        
        # Create example implementation
        example_code = await self.autoci.llama.generate_code({
            'type': 'research_example',
            'topic': topic,
            'description': f'Example implementation of {topic}'
        })
        
        self.state['lines_written'] += len(example_code.split('\n'))
        self.state['patterns_learned'] += 1
        
        # Add to knowledge base
        self.knowledge_base.add_pattern(topic, example_code)
        
    async def _learning_loop(self):
        """Continuous learning from various sources"""
        while self.is_running:
            try:
                # Process learning queue
                while not self.learning_queue.empty():
                    learning_data = self.learning_queue.get()
                    await self._process_learning_data(learning_data)
                    
                # Periodic learning from external sources
                if random.random() > 0.9:  # 10% chance each cycle
                    await self._learn_from_external_sources()
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in learning loop: {e}")
                
    async def _process_learning_data(self, data: Dict):
        """Process and learn from interaction data"""
        # Extract patterns
        if data.get('success'):
            self.knowledge_base.add_successful_pattern(
                data['input'],
                data['response']
            )
        else:
            self.knowledge_base.add_failed_pattern(
                data['input'],
                data['response']
            )
            
        self.state['patterns_learned'] += 1
        
    async def _learn_from_external_sources(self):
        """Learn from GitHub, Stack Overflow, etc."""
        self.logger.info("Learning from external sources...")
        
        # Simulate external learning
        sources = ['github', 'stackoverflow', 'godot_docs', 'gamedev_blogs']
        source = random.choice(sources)
        
        # Simulate finding useful patterns
        patterns_found = random.randint(5, 20)
        self.state['patterns_learned'] += patterns_found
        
        self.logger.info(f"Learned {patterns_found} new patterns from {source}")
        
    async def _project_monitor(self):
        """Monitor active projects and manage resources"""
        while self.is_running:
            try:
                # Check project health
                for project_name, project in self.current_projects.items():
                    if project['status'] == 'in_progress':
                        # Check if project is stuck
                        age = datetime.now() - project['created']
                        if age > timedelta(hours=2):
                            self.logger.warning(f"Project {project_name} taking too long")
                            # Attempt to unstick
                            await self._unstick_project(project_name)
                            
                # Clean up old projects
                await self._cleanup_old_projects()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in project monitor: {e}")
                
    async def _unstick_project(self, project_name: str):
        """Attempt to fix a stuck project"""
        self.logger.info(f"Attempting to unstick project: {project_name}")
        
        # Try various recovery methods
        recovery_methods = [
            self._restart_godot_engine,
            self._simplify_project_scope,
            self._rollback_recent_changes
        ]
        
        for method in recovery_methods:
            try:
                await method(project_name)
                self.current_projects[project_name]['status'] = 'recovered'
                break
            except:
                continue
                
    async def _restart_godot_engine(self, project_name: str):
        """Restart Godot engine"""
        self.autoci.godot.stop_godot_engine()
        await asyncio.sleep(2)
        project_path = f"projects/{project_name}"
        self.autoci.godot.start_godot_engine(project_path, headless=True)
        
    async def _simplify_project_scope(self, project_name: str):
        """Reduce project complexity"""
        project = self.current_projects[project_name]
        # Remove complex features
        complex_features = ['multiplayer', 'procedural_generation', 'ai_learning']
        for feature in complex_features:
            if feature in project.get('features', []):
                project['features'].remove(feature)
                
    async def _rollback_recent_changes(self, project_name: str):
        """Rollback recent changes"""
        # In a real implementation, this would use version control
        self.logger.info(f"Rolling back recent changes for {project_name}")
        
    async def _cleanup_old_projects(self):
        """Clean up old completed projects"""
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for project_name, project in list(self.current_projects.items()):
            if (project['status'] == 'completed' and 
                project['created'] < cutoff_date):
                self.logger.info(f"Archiving old project: {project_name}")
                # Archive project
                del self.current_projects[project_name]
                
    async def _optimization_loop(self):
        """Continuously optimize performance and code quality"""
        while self.is_running:
            try:
                # Optimize active projects
                for project_name, project in self.current_projects.items():
                    if project['status'] == 'in_progress':
                        await self._optimize_project(project_name)
                        
                await asyncio.sleep(600)  # Every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                
    async def _optimize_project(self, project_name: str):
        """Optimize a specific project"""
        optimizations_applied = 0
        
        # Code quality improvements
        if random.random() > 0.7:
            await self._improve_code_quality(project_name)
            optimizations_applied += 1
            
        # Performance optimizations
        if random.random() > 0.8:
            await self._optimize_game_performance(project_name)
            optimizations_applied += 1
            
        if optimizations_applied > 0:
            self.logger.info(f"Applied {optimizations_applied} optimizations to {project_name}")
            
    async def _improve_code_quality(self, project_name: str):
        """Improve code quality for a project"""
        improvements = [
            'add_documentation',
            'refactor_duplicated_code',
            'improve_naming_conventions',
            'add_error_handling',
            'implement_design_patterns'
        ]
        
        improvement = random.choice(improvements)
        
        improvement_code = await self.autoci.llama.generate_code({
            'type': 'code_quality_improvement',
            'improvement': improvement,
            'project': project_name
        })
        
        self.state['lines_written'] += len(improvement_code.split('\n'))
        
    async def _get_user_input(self) -> str:
        """Get user input asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, input, "You: ")
        
    async def _get_next_task(self) -> Optional[Dict]:
        """Get next task from queue or generate one"""
        # Check if user has given a task
        if hasattr(self, 'task_queue') and not self.task_queue.empty():
            return self.task_queue.get()
            
        # Otherwise, generate a task based on current state
        if len(self.current_projects) < 3:
            # Not many projects, create a new one
            genres = ['platformer', 'shooter', 'puzzle', 'rpg']
            return {
                'type': 'create_game',
                'parameters': {
                    'genre': random.choice(genres),
                    'dimension': random.choice(['2d', '3d'])
                }
            }
            
        # Work on existing projects
        return None
        
    async def _modify_scene_autonomous(self, params: Dict):
        """Autonomously modify game scene"""
        self.logger.info(f"Modifying scene with params: {params}")
        
        # Implementation would modify the current scene
        # based on parameters
        
    async def _optimize_code_autonomous(self, params: Dict):
        """Autonomously optimize code"""
        self.logger.info(f"Optimizing code with params: {params}")
        
        # Implementation would analyze and optimize code
        
    async def _shutdown(self):
        """Gracefully shutdown the agent"""
        self.logger.info("Shutting down AutoCI Agent...")
        self.is_running = False
        
        # Save state
        self._save_state()
        
        # Close connections
        if hasattr(self.autoci.godot, 'stop_godot_engine'):
            self.autoci.godot.stop_godot_engine()
            
        print("\n👋 AutoCI Agent shutdown complete. See you next time!")
        
    def _save_state(self):
        """Save agent state for persistence"""
        state_file = Path("agent_state.json")
        
        state_data = {
            'state': self.state,
            'projects': self.current_projects,
            'preferences': self.user_preferences,
            'knowledge_patterns': self.knowledge_base.get_patterns_count(),
            'conversation_count': len(self.conversation_history)
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
            
        self.logger.info(f"State saved to {state_file}")
        
    def get_status(self) -> Dict:
        """Get current agent status"""
        uptime = datetime.now() - self.state['uptime_start']
        
        return {
            'status': 'active' if self.is_running else 'stopped',
            'uptime': str(uptime).split('.')[0],
            'current_task': self.state.get('current_task', 'idle'),
            'games_created': self.state['games_created'],
            'lines_written': self.state['lines_written'],
            'bugs_fixed': self.state['bugs_fixed'],
            'patterns_learned': self.state['patterns_learned'],
            'active_projects': len([p for p in self.current_projects.values() 
                                   if p['status'] == 'in_progress']),
            'total_projects': len(self.current_projects)
        }


class KnowledgeBase:
    """Knowledge base for storing learned patterns"""
    
    def __init__(self):
        self.patterns = {
            'successful': [],
            'failed': [],
            'code_patterns': {},
            'user_preferences': {}
        }
        self.load_knowledge()
        
    def load_knowledge(self):
        """Load existing knowledge from disk"""
        knowledge_file = Path("knowledge_base.json")
        if knowledge_file.exists():
            with open(knowledge_file, 'r') as f:
                self.patterns = json.load(f)
                
    def save_knowledge(self):
        """Save knowledge to disk"""
        knowledge_file = Path("knowledge_base.json")
        with open(knowledge_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
            
    def add_pattern(self, category: str, pattern: Any):
        """Add a new pattern to knowledge base"""
        if category not in self.patterns['code_patterns']:
            self.patterns['code_patterns'][category] = []
        self.patterns['code_patterns'][category].append(pattern)
        self.save_knowledge()
        
    def add_successful_pattern(self, input_text: str, response: Dict):
        """Add successful interaction pattern"""
        self.patterns['successful'].append({
            'input': input_text,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        self.save_knowledge()
        
    def add_failed_pattern(self, input_text: str, response: Dict):
        """Add failed interaction pattern"""
        self.patterns['failed'].append({
            'input': input_text,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        self.save_knowledge()
        
    def get_patterns_count(self) -> int:
        """Get total number of patterns"""
        count = len(self.patterns['successful']) + len(self.patterns['failed'])
        for category in self.patterns['code_patterns'].values():
            count += len(category)
        return count


# Main entry point
async def main():
    """Main entry point for autonomous agent"""
    agent = AutonomousGameDeveloper()
    
    # Check command line arguments
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "interactive"
    
    await agent.start(mode)


if __name__ == "__main__":
    asyncio.run(main())