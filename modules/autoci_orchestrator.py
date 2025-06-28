"""
AutoCI Orchestrator - Main integration layer for Llama, Gemini, and Godot
Coordinates all AI agents and game engine control for 24/7 autonomous operation
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue

from .llama_integration import LlamaIntegration, LlamaConfig
from .gemini_integration import GeminiIntegration, GeminiConfig
from .godot_controller import GodotController

class TaskType(Enum):
    """Types of tasks the orchestrator can handle"""
    CREATE_GAME = "create_game"
    MODIFY_SCENE = "modify_scene"
    GENERATE_CODE = "generate_code"
    SOLVE_PROBLEM = "solve_problem"
    OPTIMIZE_PERFORMANCE = "optimize_performance"
    DESIGN_AI = "design_ai"
    CREATE_CONTENT = "create_content"
    REVIEW_CODE = "review_code"
    ANALYZE_DESIGN = "analyze_design"
    CONTINUOUS_IMPROVEMENT = "continuous_improvement"

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class Task:
    """Represents a task for the orchestrator"""
    id: str
    type: TaskType
    description: str
    parameters: Dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None

class AutoCIOrchestrator:
    """Main orchestrator for AutoCI system"""
    
    def __init__(self, 
                 llama_config: Optional[LlamaConfig] = None,
                 gemini_config: Optional[GeminiConfig] = None,
                 godot_host: str = "localhost",
                 godot_port: int = 8080):
        
        # Initialize components
        self.llama = LlamaIntegration(llama_config)
        self.gemini = GeminiIntegration(gemini_config)
        self.godot = GodotController(godot_host, godot_port)
        
        # Task management
        self.task_queue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        
        # Learning system
        self.learning_database = Path("autoci_learning.db")
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # 24/7 operation
        self.running = False
        self.worker_threads = []
        self.main_loop = None
        
    async def start(self):
        """Start the orchestrator for 24/7 operation"""
        self.running = True
        self.logger.info("AutoCI Orchestrator starting...")
        
        # Start Godot engine if needed
        if not self.godot.ping():
            self.logger.info("Starting Godot engine...")
            self.godot.start_godot_engine(editor=True)
            
        # Connect WebSocket for real-time events
        self.godot.connect_websocket()
        
        # Start worker tasks
        self.main_loop = asyncio.create_task(self._main_loop())
        
        self.logger.info("AutoCI Orchestrator started successfully")
        
    async def stop(self):
        """Stop the orchestrator"""
        self.running = False
        
        if self.main_loop:
            self.main_loop.cancel()
            
        # Cleanup
        self.llama.cleanup()
        self.gemini.cleanup()
        
        self.logger.info("AutoCI Orchestrator stopped")
        
    async def process_user_input(self, input_text: str, context: Dict[str, Any] = None) -> str:
        """Process natural language input from user"""
        context = context or {}
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "user_input",
            "content": input_text,
            "context": context
        })
        
        # Analyze intent
        intent = await self._analyze_intent(input_text, context)
        
        # Create and queue appropriate task
        task = await self._create_task_from_intent(intent, input_text, context)
        
        if task:
            await self.queue_task(task)
            response = f"알겠습니다! {task.description}을(를) 시작하겠습니다."
        else:
            response = "죄송합니다. 요청을 이해하지 못했습니다. 다시 설명해 주시겠습니까?"
            
        # Add response to history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "assistant_response",
            "content": response,
            "task_id": task.id if task else None
        })
        
        return response
        
    async def queue_task(self, task: Task):
        """Queue a task for processing"""
        await self.task_queue.put((task.priority.value, task.created_at, task))
        self.logger.info(f"Task queued: {task.id} - {task.description}")
        
    async def _main_loop(self):
        """Main processing loop for 24/7 operation"""
        while self.running:
            try:
                # Get next task
                priority, created_at, task = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                
                # Process task
                self.active_tasks[task.id] = task
                task.status = "processing"
                
                try:
                    result = await self._process_task(task)
                    task.result = result
                    task.status = "completed"
                    self.completed_tasks.append(task)
                    
                    # Learn from successful task
                    await self._learn_from_task(task, success=True)
                    
                except Exception as e:
                    task.error = str(e)
                    task.status = "failed"
                    self.logger.error(f"Task failed: {task.id} - {e}")
                    
                    # Learn from failure
                    await self._learn_from_task(task, success=False)
                    
                finally:
                    del self.active_tasks[task.id]
                    
            except asyncio.TimeoutError:
                # No tasks, check for background work
                await self._perform_background_tasks()
                
            except Exception as e:
                self.logger.error(f"Main loop error: {e}")
                
    async def _analyze_intent(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user intent using both Llama and Gemini"""
        # Quick intent analysis with Llama
        llama_intent = self.llama.translate_intent(input_text)
        
        # Complex intent analysis with Gemini for ambiguous cases
        if llama_intent.get("action") == "unknown":
            gemini_prompt = f"""Analyze this user input and determine their intent:

User Input: "{input_text}"
Context: {json.dumps(context, indent=2)}

Identify:
1. Primary action (create, modify, delete, query, etc.)
2. Target object or system
3. Required parameters
4. Complexity level (simple/complex)"""

            gemini_response = await self.gemini._run_gemini_cli_async(
                gemini_prompt, 
                {"temperature": 0.3}
            )
            
            # Merge insights
            return self._merge_intent_analysis(llama_intent, gemini_response)
            
        return llama_intent
        
    async def _create_task_from_intent(self, intent: Dict[str, Any], 
                                     original_input: str,
                                     context: Dict[str, Any]) -> Optional[Task]:
        """Create appropriate task based on analyzed intent"""
        action = intent.get("action", "").lower()
        
        task_mappings = {
            "create": self._create_creation_task,
            "modify": self._create_modification_task,
            "delete": self._create_deletion_task,
            "optimize": self._create_optimization_task,
            "analyze": self._create_analysis_task,
            "generate": self._create_generation_task,
        }
        
        creator = task_mappings.get(action)
        if creator:
            return await creator(intent, original_input, context)
            
        return None
        
    async def _create_creation_task(self, intent: Dict[str, Any], 
                                   original_input: str,
                                   context: Dict[str, Any]) -> Task:
        """Create a creation task"""
        target = intent.get("target", "")
        
        if "game" in target or "프로젝트" in original_input:
            return Task(
                id=self._generate_task_id(),
                type=TaskType.CREATE_GAME,
                description=f"새로운 게임 프로젝트 생성: {original_input}",
                parameters={
                    "intent": intent,
                    "original_input": original_input,
                    "context": context
                },
                priority=TaskPriority.HIGH
            )
        elif "player" in target or "character" in target or "플레이어" in original_input:
            return Task(
                id=self._generate_task_id(),
                type=TaskType.CREATE_CONTENT,
                description=f"플레이어 캐릭터 생성",
                parameters={
                    "content_type": "player_character",
                    "specifications": intent.get("parameters", {}),
                    "original_input": original_input
                },
                priority=TaskPriority.HIGH
            )
        else:
            return Task(
                id=self._generate_task_id(),
                type=TaskType.CREATE_CONTENT,
                description=f"콘텐츠 생성: {target}",
                parameters=intent,
                priority=TaskPriority.MEDIUM
            )
            
    async def _process_task(self, task: Task) -> Any:
        """Process a single task using appropriate AI models and Godot"""
        self.logger.info(f"Processing task: {task.id} - {task.type.value}")
        
        handlers = {
            TaskType.CREATE_GAME: self._handle_create_game,
            TaskType.MODIFY_SCENE: self._handle_modify_scene,
            TaskType.GENERATE_CODE: self._handle_generate_code,
            TaskType.SOLVE_PROBLEM: self._handle_solve_problem,
            TaskType.OPTIMIZE_PERFORMANCE: self._handle_optimize_performance,
            TaskType.DESIGN_AI: self._handle_design_ai,
            TaskType.CREATE_CONTENT: self._handle_create_content,
            TaskType.REVIEW_CODE: self._handle_review_code,
            TaskType.ANALYZE_DESIGN: self._handle_analyze_design,
            TaskType.CONTINUOUS_IMPROVEMENT: self._handle_continuous_improvement,
        }
        
        handler = handlers.get(task.type)
        if handler:
            return await handler(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
            
    async def _handle_create_game(self, task: Task) -> Dict[str, Any]:
        """Handle game creation task"""
        params = task.parameters
        original_input = params.get("original_input", "")
        
        # 1. Use Gemini to design the game structure
        design = self.gemini.analyze_game_design(original_input)
        
        # 2. Create project structure in Godot
        project_name = design.get("overview", "NewGame").replace(" ", "_")
        project_path = f"./projects/{project_name}"
        
        self.godot.create_project(project_name, project_path)
        
        # 3. Create main scenes based on design
        for scene_name in design.get("scenes", ["Main"]):
            self.godot.create_scene(scene_name)
            
            # 4. Use Llama to generate scene scripts
            script_prompt = f"Create a GDScript for {scene_name} scene in a {design.get('overview', 'game')}"
            script_code = self.llama.generate_code(script_prompt, "gdscript")
            
            if script_code:
                script_path = f"res://{scene_name}.gd"
                self.godot.create_script(script_path, script_code)
                self.godot.attach_script(f"/root", script_path)
                
        # 5. Save the main scene
        self.godot.save_scene(f"res://{design.get('scenes', ['Main'])[0]}.tscn")
        
        return {
            "status": "success",
            "project_name": project_name,
            "project_path": project_path,
            "design": design,
            "scenes_created": design.get("scenes", [])
        }
        
    async def _handle_create_content(self, task: Task) -> Dict[str, Any]:
        """Handle content creation task"""
        params = task.parameters
        content_type = params.get("content_type", "")
        
        if content_type == "player_character":
            return await self._create_player_character(params)
        elif "enemy" in content_type:
            return await self._create_enemy(params)
        elif "item" in content_type:
            return await self._create_item(params)
        else:
            # Generic content creation
            creative_params = {
                "type": content_type,
                "specifications": params.get("specifications", {})
            }
            
            creative_content = self.gemini.generate_creative_content(
                content_type, 
                creative_params
            )
            
            return {
                "status": "success",
                "content": creative_content
            }
            
    async def _create_player_character(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a player character with full implementation"""
        # 1. Design character concept with Gemini
        character_design = self.gemini.generate_creative_content(
            "player_character",
            {
                "game_type": params.get("game_type", "2D platformer"),
                "style": params.get("style", "classic"),
                "abilities": params.get("abilities", ["move", "jump"])
            }
        )
        
        # 2. Create character scene structure
        self.godot.create_node("CharacterBody2D", "Player", "/root")
        
        # Add visual components
        self.godot.create_node("Sprite2D", "Sprite", "/root/Player")
        self.godot.create_node("CollisionShape2D", "CollisionShape", "/root/Player")
        self.godot.create_node("AnimationPlayer", "AnimationPlayer", "/root/Player")
        
        # 3. Generate player controller script with Llama
        script_prompt = f"""Create a player controller script with these features:
        - Movement: {character_design.get('concept', 'basic movement')}
        - Abilities: {json.dumps(params.get('abilities', ['move', 'jump']))}
        - Input handling for keyboard and gamepad"""
        
        player_script = self.llama.generate_code(script_prompt, "gdscript")
        
        # 4. Apply script to player
        if player_script:
            script_path = "res://Player.gd"
            self.godot.create_script(script_path, player_script)
            self.godot.attach_script("/root/Player", script_path)
            
        # 5. Configure collision shape
        shape_resource = self.godot.create_resource(
            "RectangleShape2D",
            "res://player_collision.tres",
            {"size": [32, 64]}
        )
        self.godot.set_property("/root/Player/CollisionShape", "shape", shape_resource["path"])
        
        # 6. Set initial position
        self.godot.set_property("/root/Player", "position", [400, 300])
        
        return {
            "status": "success",
            "character_path": "/root/Player",
            "design": character_design,
            "script_path": script_path if player_script else None,
            "components": ["Sprite2D", "CollisionShape2D", "AnimationPlayer"]
        }
        
    async def _handle_generate_code(self, task: Task) -> Dict[str, Any]:
        """Handle code generation task"""
        params = task.parameters
        
        # Determine which AI to use based on complexity
        if params.get("complexity", "simple") == "complex":
            # Use Gemini for complex logic
            code = await self.gemini._run_gemini_cli_async(
                params.get("prompt", ""),
                {"temperature": 0.7}
            )
        else:
            # Use Llama for quick code generation
            code = self.llama.generate_code(
                params.get("prompt", ""),
                params.get("language", "gdscript")
            )
            
        # Optionally review the code
        if params.get("review", False):
            review = self.gemini.review_code_quality(code)
            
            # Apply improvements if score is low
            if review.get("score", 10) < 7:
                improved_code = self.llama.optimize_code(code, "quality")
                return {
                    "original_code": code,
                    "improved_code": improved_code,
                    "review": review
                }
                
        return {
            "code": code,
            "language": params.get("language", "gdscript")
        }
        
    async def _handle_solve_problem(self, task: Task) -> Dict[str, Any]:
        """Handle problem solving task"""
        params = task.parameters
        
        # Use Gemini for complex problem solving
        solution = self.gemini.solve_complex_problem(
            params.get("problem", ""),
            params.get("context", {})
        )
        
        # If solution includes code, generate it with Llama
        if solution.get("implementation"):
            implementation_code = self.llama.generate_code(
                solution["implementation"],
                "gdscript"
            )
            solution["generated_code"] = implementation_code
            
        return solution
        
    async def _handle_optimize_performance(self, task: Task) -> Dict[str, Any]:
        """Handle performance optimization task"""
        params = task.parameters
        
        # Get current performance data from Godot
        perf_data = self.godot.get_performance_data()
        
        # Analyze with Gemini
        optimization_plan = self.gemini.solve_complex_problem(
            f"Optimize performance based on metrics: {json.dumps(perf_data)}",
            {"current_code": params.get("code", "")}
        )
        
        # Apply optimizations
        if params.get("code"):
            optimized_code = self.gemini.optimize_game_logic(
                params["code"],
                ["performance", "memory", "readability"]
            )
            
            return {
                "performance_data": perf_data,
                "optimization_plan": optimization_plan,
                "optimized_code": optimized_code
            }
            
        return {
            "performance_data": perf_data,
            "optimization_plan": optimization_plan
        }
        
    async def _handle_design_ai(self, task: Task) -> Dict[str, Any]:
        """Handle AI behavior design task"""
        params = task.parameters
        
        # Design AI with Gemini
        ai_design = self.gemini.design_ai_behavior(
            params.get("entity_type", "enemy"),
            params.get("behavior", "basic patrol and chase")
        )
        
        # Generate implementation with Llama
        if ai_design.get("implementation"):
            ai_script = self.llama.generate_gdscript_class(
                f"{params.get('entity_type', 'Enemy')}AI",
                "CharacterBody2D",
                [
                    {"name": "speed", "default": ai_design["parameters"].get("speed", 100)},
                    {"name": "detection_range", "default": ai_design["parameters"].get("detection_range", 200)},
                    {"name": "state", "default": "'idle'"}
                ],
                [
                    {"name": "_ready", "description": "Initialize AI"},
                    {"name": "_physics_process", "params": "delta", "description": "AI update loop"},
                    {"name": "change_state", "params": "new_state", "description": "State transition"}
                ]
            )
            
            ai_design["generated_script"] = ai_script
            
        return ai_design
        
    async def _perform_background_tasks(self):
        """Perform background tasks during idle time"""
        # Check for Godot events
        events = self.godot.get_events(timeout=0.1)
        for event in events:
            await self._handle_godot_event(event)
            
        # Periodic learning optimization
        if len(self.completed_tasks) % 10 == 0 and self.completed_tasks:
            await self._optimize_from_experience()
            
    async def _handle_godot_event(self, event: Dict[str, Any]):
        """Handle real-time events from Godot"""
        event_type = event.get("event", "")
        
        if event_type == "error":
            # Create problem-solving task
            task = Task(
                id=self._generate_task_id(),
                type=TaskType.SOLVE_PROBLEM,
                description=f"자동 오류 해결: {event.get('message', '')}",
                parameters={"problem": event.get("message", ""), "context": event},
                priority=TaskPriority.HIGH
            )
            await self.queue_task(task)
            
    async def _learn_from_task(self, task: Task, success: bool):
        """Learn from completed tasks to improve future performance"""
        learning_entry = {
            "task_id": task.id,
            "task_type": task.type.value,
            "parameters": task.parameters,
            "success": success,
            "result": task.result if success else task.error,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in learning database
        # In production, this would use a proper database
        self.logger.info(f"Learning from task: {task.id} - Success: {success}")
        
    async def _optimize_from_experience(self):
        """Optimize behavior based on accumulated experience"""
        # Analyze patterns in completed tasks
        success_rate = sum(1 for t in self.completed_tasks if t.status == "completed") / len(self.completed_tasks)
        
        self.logger.info(f"Current success rate: {success_rate:.2%}")
        
        # Adjust strategies based on performance
        if success_rate < 0.8:
            # Need to be more careful
            self.logger.info("Adjusting strategies for better success rate")
            
    def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        import uuid
        return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
    def _merge_intent_analysis(self, llama_intent: Dict[str, Any], 
                              gemini_response: str) -> Dict[str, Any]:
        """Merge intent analysis from both models"""
        # Parse Gemini response and merge with Llama
        merged = llama_intent.copy()
        
        # Simple parsing - in production, use more sophisticated NLP
        if "create" in gemini_response.lower():
            merged["action"] = "create"
        elif "modify" in gemini_response.lower():
            merged["action"] = "modify"
            
        return merged
        
    async def _create_modification_task(self, intent: Dict[str, Any],
                                       original_input: str,
                                       context: Dict[str, Any]) -> Task:
        """Create a modification task"""
        return Task(
            id=self._generate_task_id(),
            type=TaskType.MODIFY_SCENE,
            description=f"씬 수정: {original_input}",
            parameters={
                "intent": intent,
                "original_input": original_input,
                "context": context
            },
            priority=TaskPriority.MEDIUM
        )
        
    async def _create_deletion_task(self, intent: Dict[str, Any],
                                   original_input: str,
                                   context: Dict[str, Any]) -> Task:
        """Create a deletion task"""
        return Task(
            id=self._generate_task_id(),
            type=TaskType.MODIFY_SCENE,
            description=f"삭제 작업: {original_input}",
            parameters={
                "action": "delete",
                "intent": intent,
                "original_input": original_input
            },
            priority=TaskPriority.MEDIUM
        )
        
    async def _create_optimization_task(self, intent: Dict[str, Any],
                                       original_input: str,
                                       context: Dict[str, Any]) -> Task:
        """Create an optimization task"""
        return Task(
            id=self._generate_task_id(),
            type=TaskType.OPTIMIZE_PERFORMANCE,
            description=f"최적화: {original_input}",
            parameters={
                "intent": intent,
                "original_input": original_input,
                "context": context
            },
            priority=TaskPriority.MEDIUM
        )
        
    async def _create_analysis_task(self, intent: Dict[str, Any],
                                   original_input: str,
                                   context: Dict[str, Any]) -> Task:
        """Create an analysis task"""
        return Task(
            id=self._generate_task_id(),
            type=TaskType.ANALYZE_DESIGN,
            description=f"분석: {original_input}",
            parameters={
                "intent": intent,
                "original_input": original_input,
                "context": context
            },
            priority=TaskPriority.LOW
        )
        
    async def _create_generation_task(self, intent: Dict[str, Any],
                                     original_input: str,
                                     context: Dict[str, Any]) -> Task:
        """Create a generation task"""
        return Task(
            id=self._generate_task_id(),
            type=TaskType.GENERATE_CODE,
            description=f"코드 생성: {original_input}",
            parameters={
                "intent": intent,
                "original_input": original_input,
                "context": context
            },
            priority=TaskPriority.MEDIUM
        )
        
    # Additional handlers
    async def _handle_modify_scene(self, task: Task) -> Dict[str, Any]:
        """Handle scene modification task"""
        # Implementation for scene modification
        return {"status": "success", "modifications": []}
        
    async def _handle_review_code(self, task: Task) -> Dict[str, Any]:
        """Handle code review task"""
        params = task.parameters
        code = params.get("code", "")
        
        review = self.gemini.review_code_quality(code, params.get("language", "gdscript"))
        
        if review["score"] < 7:
            improved_code = self.llama.optimize_code(code, "quality")
            review["improved_code"] = improved_code
            
        return review
        
    async def _handle_analyze_design(self, task: Task) -> Dict[str, Any]:
        """Handle design analysis task"""
        params = task.parameters
        return self.gemini.analyze_game_design(params.get("description", ""))
        
    async def _handle_continuous_improvement(self, task: Task) -> Dict[str, Any]:
        """Handle continuous improvement task"""
        # Analyze current project state and suggest improvements
        return {
            "status": "success",
            "improvements": ["Performance optimization", "Code refactoring", "UI enhancement"]
        }
        
    async def _create_enemy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create an enemy character"""
        # Similar to player creation but with AI
        enemy_name = params.get("name", "Enemy")
        
        # Create enemy structure
        self.godot.create_node("CharacterBody2D", enemy_name, "/root")
        self.godot.create_node("Sprite2D", "Sprite", f"/root/{enemy_name}")
        self.godot.create_node("CollisionShape2D", "CollisionShape", f"/root/{enemy_name}")
        
        # Design and implement AI
        ai_design = self.gemini.design_ai_behavior(
            enemy_name,
            params.get("behavior", "patrol and chase player")
        )
        
        # Generate AI script
        ai_script = self.llama.generate_code(
            f"Create enemy AI script: {ai_design.get('implementation', '')}",
            "gdscript"
        )
        
        if ai_script:
            script_path = f"res://{enemy_name}AI.gd"
            self.godot.create_script(script_path, ai_script)
            self.godot.attach_script(f"/root/{enemy_name}", script_path)
            
        return {
            "status": "success",
            "enemy_path": f"/root/{enemy_name}",
            "ai_design": ai_design,
            "script_path": script_path if ai_script else None
        }
        
    async def _create_item(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a game item"""
        item_name = params.get("name", "Item")
        item_type = params.get("type", "pickup")
        
        # Create item structure based on type
        if item_type == "pickup":
            self.godot.create_node("Area2D", item_name, "/root")
            self.godot.create_node("Sprite2D", "Sprite", f"/root/{item_name}")
            self.godot.create_node("CollisionShape2D", "CollisionShape", f"/root/{item_name}")
            
            # Generate pickup script
            pickup_script = self.llama.generate_code(
                f"Create pickup item script for {item_name} with effect: {params.get('effect', 'heal')}",
                "gdscript"
            )
            
            if pickup_script:
                script_path = f"res://{item_name}.gd"
                self.godot.create_script(script_path, pickup_script)
                self.godot.attach_script(f"/root/{item_name}", script_path)
                
        return {
            "status": "success",
            "item_path": f"/root/{item_name}",
            "item_type": item_type
        }