"""
Llama 7B Integration Module for AutoCI
Provides local LLM capabilities for code generation and understanding
"""

import os
import sys
import json
import torch
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for llama module
sys.path.append(str(Path(__file__).parent.parent))

try:
    from llama import Llama
except ImportError:
    # Fallback for development
    Llama = None

@dataclass
class LlamaConfig:
    """Configuration for Llama model"""
    model_path: str = "CodeLlama-7b-Instruct-hf"
    max_seq_len: int = 4096
    max_batch_size: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
class LlamaIntegration:
    """Llama 7B integration for AutoCI"""
    
    def __init__(self, config: Optional[LlamaConfig] = None):
        self.config = config or LlamaConfig()
        self.model = None
        self.tokenizer = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.logger = logging.getLogger(__name__)
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize Llama model"""
        try:
            if Llama is None:
                self.logger.warning("Llama module not available, using mock mode")
                return
                
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                self.logger.error(f"Model path {model_path} does not exist")
                return
                
            self.logger.info("Loading Llama 7B model...")
            self.model = Llama.build(
                ckpt_dir=str(model_path),
                tokenizer_path=str(model_path / "tokenizer.model"),
                max_seq_len=self.config.max_seq_len,
                max_batch_size=self.config.max_batch_size,
            )
            self.logger.info("Llama model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Llama model: {e}")
            
    def generate_code(self, prompt: str, language: str = "gdscript") -> str:
        """Generate code based on prompt"""
        if self.model is None:
            return self._mock_generate_code(prompt, language)
            
        # Create specialized prompt for code generation
        system_prompt = self._get_system_prompt(language)
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        try:
            response = self.model.generate(
                prompts=[full_prompt],
                max_gen_len=1024,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            
            if response and len(response) > 0:
                return self._extract_code(response[0]['generation'])
                
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            
        return ""
        
    def analyze_code(self, code: str, language: str = "gdscript") -> Dict[str, Any]:
        """Analyze code and provide insights"""
        if self.model is None:
            return self._mock_analyze_code(code, language)
            
        prompt = f"""Analyze the following {language} code and provide:
1. Summary of functionality
2. Potential issues or improvements
3. Performance considerations

Code:
```{language}
{code}
```

Analysis:"""
        
        try:
            response = self.model.generate(
                prompts=[prompt],
                max_gen_len=512,
                temperature=0.3,  # Lower temperature for analysis
                top_p=0.9,
            )
            
            if response and len(response) > 0:
                analysis_text = response[0]['generation']
                return self._parse_analysis(analysis_text)
                
        except Exception as e:
            self.logger.error(f"Code analysis failed: {e}")
            
        return {"error": "Analysis failed"}
        
    def complete_code(self, partial_code: str, context: str = "") -> str:
        """Complete partial code"""
        if self.model is None:
            return self._mock_complete_code(partial_code, context)
            
        prompt = f"""Complete the following GDScript code:

Context: {context}

Code:
```gdscript
{partial_code}
```

Completed code:"""
        
        try:
            response = self.model.generate(
                prompts=[prompt],
                max_gen_len=256,
                temperature=0.5,
                top_p=0.9,
            )
            
            if response and len(response) > 0:
                return self._extract_code(response[0]['generation'])
                
        except Exception as e:
            self.logger.error(f"Code completion failed: {e}")
            
        return partial_code
        
    def translate_intent(self, user_intent: str) -> Dict[str, Any]:
        """Translate user intent to actionable commands"""
        if self.model is None:
            return self._mock_translate_intent(user_intent)
            
        prompt = f"""Translate the following user intent into Godot Engine commands:

User Intent: "{user_intent}"

Provide:
1. Action type (create, modify, delete, etc.)
2. Target object/node
3. Required parameters
4. GDScript code if needed

Response format: JSON"""
        
        try:
            response = self.model.generate(
                prompts=[prompt],
                max_gen_len=512,
                temperature=0.3,
                top_p=0.9,
            )
            
            if response and len(response) > 0:
                return self._parse_json_response(response[0]['generation'])
                
        except Exception as e:
            self.logger.error(f"Intent translation failed: {e}")
            
        return {"error": "Translation failed"}
        
    def generate_gdscript_class(self, class_name: str, 
                               base_class: str = "Node",
                               properties: List[Dict] = None,
                               methods: List[Dict] = None) -> str:
        """Generate a complete GDScript class"""
        properties = properties or []
        methods = methods or []
        
        prompt = f"""Generate a complete GDScript class with the following specifications:

Class Name: {class_name}
Base Class: {base_class}
Properties: {json.dumps(properties, indent=2)}
Methods: {json.dumps(methods, indent=2)}

Generate production-ready GDScript code with proper comments and error handling."""
        
        if self.model is None:
            return self._mock_generate_gdscript_class(
                class_name, base_class, properties, methods
            )
            
        try:
            response = self.model.generate(
                prompts=[prompt],
                max_gen_len=1024,
                temperature=0.7,
                top_p=0.9,
            )
            
            if response and len(response) > 0:
                return self._extract_code(response[0]['generation'])
                
        except Exception as e:
            self.logger.error(f"GDScript class generation failed: {e}")
            
        return ""
        
    def optimize_code(self, code: str, optimization_goal: str = "performance") -> str:
        """Optimize existing code"""
        if self.model is None:
            return code
            
        prompt = f"""Optimize the following GDScript code for {optimization_goal}:

Original Code:
```gdscript
{code}
```

Provide optimized version with comments explaining changes:"""
        
        try:
            response = self.model.generate(
                prompts=[prompt],
                max_gen_len=1024,
                temperature=0.5,
                top_p=0.9,
            )
            
            if response and len(response) > 0:
                return self._extract_code(response[0]['generation'])
                
        except Exception as e:
            self.logger.error(f"Code optimization failed: {e}")
            
        return code
        
    async def generate_code_async(self, prompt: str, language: str = "gdscript") -> str:
        """Async version of code generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.generate_code, 
            prompt, 
            language
        )
        
    # Helper methods
    def _get_system_prompt(self, language: str) -> str:
        """Get system prompt for specific language"""
        prompts = {
            "gdscript": """You are an expert GDScript developer for Godot Engine.
Generate clean, efficient, and well-commented code following Godot best practices.
Use proper signal patterns, node references, and resource management.""",
            
            "python": """You are an expert Python developer.
Generate clean, efficient, and well-documented code following PEP 8 guidelines.""",
            
            "csharp": """You are an expert C# developer for game development.
Generate clean, efficient code following C# conventions and Unity/Godot patterns."""
        }
        
        return prompts.get(language, prompts["gdscript"])
        
    def _extract_code(self, response: str) -> str:
        """Extract code from model response"""
        # Look for code blocks
        import re
        
        # Try to find code within triple backticks
        code_match = re.search(r'```(?:gdscript|python|csharp)?\n(.*?)```', 
                              response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
            
        # If no code block, try to extract after "Assistant:" or similar
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith(('class ', 'func ', 'extends ', 'var ', 'const ')):
                in_code = True
            if in_code:
                code_lines.append(line)
                
        return '\n'.join(code_lines).strip()
        
    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse analysis text into structured format"""
        result = {
            "summary": "",
            "issues": [],
            "improvements": [],
            "performance": ""
        }
        
        sections = analysis_text.split('\n\n')
        for section in sections:
            lower_section = section.lower()
            if 'summary' in lower_section or 'functionality' in lower_section:
                result["summary"] = section
            elif 'issue' in lower_section or 'problem' in lower_section:
                result["issues"] = [line.strip() for line in section.split('\n') 
                                   if line.strip() and not line.startswith('#')]
            elif 'improvement' in lower_section or 'suggestion' in lower_section:
                result["improvements"] = [line.strip() for line in section.split('\n')
                                        if line.strip() and not line.startswith('#')]
            elif 'performance' in lower_section:
                result["performance"] = section
                
        return result
        
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from model response"""
        import re
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
                
        # Fallback: try to parse structured text
        result = {
            "action": "unknown",
            "target": "",
            "parameters": {},
            "code": ""
        }
        
        lines = response.split('\n')
        for line in lines:
            if 'action' in line.lower() and ':' in line:
                result["action"] = line.split(':')[1].strip()
            elif 'target' in line.lower() and ':' in line:
                result["target"] = line.split(':')[1].strip()
                
        return result
        
    # Mock methods for development/testing
    def _mock_generate_code(self, prompt: str, language: str) -> str:
        """Mock code generation for testing"""
        if "player" in prompt.lower() and language == "gdscript":
            return '''extends CharacterBody2D

@export var speed = 300.0
@export var jump_velocity = -400.0

var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

func _physics_process(delta):
    if not is_on_floor():
        velocity.y += gravity * delta
    
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = jump_velocity
    
    var direction = Input.get_axis("ui_left", "ui_right")
    if direction:
        velocity.x = direction * speed
    else:
        velocity.x = move_toward(velocity.x, 0, speed)
    
    move_and_slide()'''
        
        return f"# Generated {language} code for: {prompt}"
        
    def _mock_analyze_code(self, code: str, language: str) -> Dict[str, Any]:
        """Mock code analysis"""
        return {
            "summary": f"Analysis of {language} code with {len(code.split())} lines",
            "issues": ["Consider adding error handling", "Add input validation"],
            "improvements": ["Use signals for loose coupling", "Add documentation"],
            "performance": "Code appears efficient for typical use cases"
        }
        
    def _mock_complete_code(self, partial_code: str, context: str) -> str:
        """Mock code completion"""
        if "func" in partial_code and not partial_code.strip().endswith(":"):
            return partial_code + ":\n    pass"
        return partial_code + "\n    # TODO: Implement"
        
    def _mock_translate_intent(self, user_intent: str) -> Dict[str, Any]:
        """Mock intent translation"""
        intent_lower = user_intent.lower()
        
        if "create" in intent_lower or "make" in intent_lower:
            if "player" in intent_lower:
                return {
                    "action": "create",
                    "target": "player_character",
                    "parameters": {
                        "type": "CharacterBody2D",
                        "name": "Player",
                        "components": ["Sprite2D", "CollisionShape2D"]
                    },
                    "code": "# Player creation code"
                }
            elif "enemy" in intent_lower:
                return {
                    "action": "create",
                    "target": "enemy",
                    "parameters": {
                        "type": "CharacterBody2D",
                        "name": "Enemy",
                        "ai_type": "basic"
                    }
                }
                
        return {
            "action": "unknown",
            "target": "",
            "parameters": {},
            "code": ""
        }
        
    def _mock_generate_gdscript_class(self, class_name: str, base_class: str,
                                     properties: List[Dict], methods: List[Dict]) -> str:
        """Mock GDScript class generation"""
        code = f"extends {base_class}\n\n"
        code += f"class_name {class_name}\n\n"
        
        # Add properties
        for prop in properties:
            code += f"@export var {prop.get('name', 'property')} = {prop.get('default', 'null')}\n"
        
        code += "\n"
        
        # Add methods
        for method in methods:
            code += f"func {method.get('name', 'method')}({method.get('params', '')}):\n"
            code += f"    # TODO: Implement {method.get('description', '')}\n"
            code += "    pass\n\n"
            
        return code
        
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
            
    def __del__(self):
        """Destructor"""
        self.cleanup()