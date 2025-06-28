"""
Gemini CLI Integration Module for AutoCI
Provides advanced AI capabilities through Gemini API
"""

import os
import json
import asyncio
import aiohttp
import subprocess
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import tempfile
from concurrent.futures import ThreadPoolExecutor

@dataclass
class GeminiConfig:
    """Configuration for Gemini integration"""
    api_key: Optional[str] = None
    cli_path: str = "gemini-cli/packages/cli/dist/index.js"
    node_path: str = "node"
    model: str = "gemini-pro"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 30

class GeminiIntegration:
    """Gemini CLI integration for AutoCI"""
    
    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or GeminiConfig()
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._validate_setup()
        
    def _validate_setup(self):
        """Validate Gemini CLI setup"""
        cli_path = Path(self.config.cli_path)
        if not cli_path.exists():
            self.logger.warning(f"Gemini CLI not found at {cli_path}")
            
        # Check if node is available
        try:
            result = subprocess.run(
                [self.config.node_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                self.logger.warning("Node.js not available")
        except Exception as e:
            self.logger.error(f"Failed to check Node.js: {e}")
            
    def _run_gemini_cli(self, prompt: str, options: Dict[str, Any] = None) -> str:
        """Run Gemini CLI with given prompt"""
        options = options or {}
        
        # Prepare command
        cmd = [
            self.config.node_path,
            self.config.cli_path,
            "chat",
            "--model", self.config.model,
            "--temperature", str(options.get("temperature", self.config.temperature)),
        ]
        
        # Add API key if available
        env = os.environ.copy()
        if self.config.api_key:
            env["GEMINI_API_KEY"] = self.config.api_key
            
        try:
            # Run Gemini CLI
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                env=env
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                self.logger.error(f"Gemini CLI error: {result.stderr}")
                return ""
                
        except subprocess.TimeoutExpired:
            self.logger.error("Gemini CLI timeout")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to run Gemini CLI: {e}")
            return ""
            
    async def _run_gemini_cli_async(self, prompt: str, options: Dict[str, Any] = None) -> str:
        """Async version of Gemini CLI execution"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._run_gemini_cli,
            prompt,
            options
        )
        
    def analyze_game_design(self, description: str) -> Dict[str, Any]:
        """Analyze game design and provide structured plan"""
        prompt = f"""Analyze the following game description and provide a detailed implementation plan:

Game Description: {description}

Provide a structured response with:
1. Game Overview
2. Core Mechanics
3. Required Scenes and Nodes
4. Technical Requirements
5. Implementation Steps
6. Potential Challenges

Format the response as JSON for easy parsing."""

        response = self._run_gemini_cli(prompt, {"temperature": 0.3})
        
        if response:
            return self._parse_json_response(response)
        else:
            return self._mock_analyze_game_design(description)
            
    def solve_complex_problem(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Solve complex game development problems"""
        context = context or {}
        
        prompt = f"""Solve the following game development problem:

Problem: {problem}

Context:
{json.dumps(context, indent=2)}

Provide:
1. Root cause analysis
2. Multiple solution approaches
3. Recommended solution with implementation
4. Code examples if applicable

Format as JSON."""

        response = self._run_gemini_cli(prompt, {"temperature": 0.5})
        
        if response:
            return self._parse_json_response(response)
        else:
            return self._mock_solve_complex_problem(problem, context)
            
    def generate_creative_content(self, content_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate creative game content"""
        prompt = f"""Generate creative {content_type} for a game with these parameters:

{json.dumps(parameters, indent=2)}

Provide:
1. Creative concept
2. Implementation details
3. Variations or alternatives
4. Integration suggestions

Be creative and innovative!"""

        response = self._run_gemini_cli(prompt, {"temperature": 0.9})
        
        if response:
            return self._parse_creative_response(response, content_type)
        else:
            return self._mock_generate_creative_content(content_type, parameters)
            
    def optimize_game_logic(self, current_logic: str, optimization_goals: List[str]) -> str:
        """Optimize game logic for specified goals"""
        prompt = f"""Optimize the following game logic:

Current Logic:
```gdscript
{current_logic}
```

Optimization Goals:
{json.dumps(optimization_goals, indent=2)}

Provide optimized code with explanations for each optimization."""

        response = self._run_gemini_cli(prompt, {"temperature": 0.3})
        
        if response:
            return self._extract_code(response)
        else:
            return current_logic
            
    def design_ai_behavior(self, entity_type: str, behavior_description: str) -> Dict[str, Any]:
        """Design AI behavior for game entities"""
        prompt = f"""Design an AI behavior system for {entity_type}:

Behavior Description: {behavior_description}

Provide:
1. State machine design
2. Decision tree
3. GDScript implementation
4. Tuning parameters

Make it engaging and balanced for gameplay."""

        response = self._run_gemini_cli(prompt, {"temperature": 0.7})
        
        if response:
            return self._parse_ai_design(response)
        else:
            return self._mock_design_ai_behavior(entity_type, behavior_description)
            
    def review_code_quality(self, code: str, language: str = "gdscript") -> Dict[str, Any]:
        """Perform comprehensive code review"""
        prompt = f"""Review the following {language} code:

```{language}
{code}
```

Provide:
1. Code quality score (1-10)
2. Issues found (bugs, anti-patterns)
3. Performance concerns
4. Security considerations
5. Refactoring suggestions
6. Best practices violations

Be thorough and constructive."""

        response = self._run_gemini_cli(prompt, {"temperature": 0.2})
        
        if response:
            return self._parse_code_review(response)
        else:
            return self._mock_review_code_quality(code, language)
            
    def generate_game_narrative(self, genre: str, themes: List[str], length: str = "medium") -> Dict[str, Any]:
        """Generate game narrative and story"""
        prompt = f"""Create a compelling game narrative:

Genre: {genre}
Themes: {', '.join(themes)}
Length: {length}

Provide:
1. Main story arc
2. Character descriptions
3. Key plot points
4. Dialogue examples
5. Environmental storytelling ideas
6. Player choices and consequences"""

        response = self._run_gemini_cli(prompt, {"temperature": 0.8})
        
        if response:
            return self._parse_narrative(response)
        else:
            return self._mock_generate_game_narrative(genre, themes, length)
            
    def plan_game_features(self, game_type: str, target_audience: str, scope: str = "medium") -> List[Dict[str, Any]]:
        """Plan game features based on requirements"""
        prompt = f"""Plan features for a {game_type} game:

Target Audience: {target_audience}
Project Scope: {scope}

Provide a prioritized list of features with:
1. Feature name and description
2. Implementation complexity (1-5)
3. Player value (1-5)
4. Dependencies
5. Estimated implementation time

Consider industry best practices and current gaming trends."""

        response = self._run_gemini_cli(prompt, {"temperature": 0.5})
        
        if response:
            return self._parse_feature_list(response)
        else:
            return self._mock_plan_game_features(game_type, target_audience, scope)
            
    async def analyze_game_design_async(self, description: str) -> Dict[str, Any]:
        """Async version of game design analysis"""
        return await asyncio.create_task(
            self._run_async_analysis(description)
        )
        
    async def _run_async_analysis(self, description: str) -> Dict[str, Any]:
        """Helper for async analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.analyze_game_design,
            description
        )
        
    # Parsing methods
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response"""
        import re
        
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
                
        # Fallback: parse structured text
        return self._parse_structured_text(response)
        
    def _parse_structured_text(self, text: str) -> Dict[str, Any]:
        """Parse structured text into dictionary"""
        result = {}
        current_section = None
        current_content = []
        
        for line in text.split('\n'):
            # Check if line is a section header
            if line.strip() and (line.strip()[0].isdigit() or line.startswith('#')):
                if current_section:
                    result[current_section] = '\n'.join(current_content).strip()
                current_section = line.strip().lstrip('#').lstrip('0123456789.').strip()
                current_content = []
            else:
                current_content.append(line)
                
        if current_section:
            result[current_section] = '\n'.join(current_content).strip()
            
        return result
        
    def _extract_code(self, response: str) -> str:
        """Extract code from response"""
        import re
        
        # Look for code blocks
        code_match = re.search(r'```(?:gdscript|python|csharp)?\n(.*?)```', 
                              response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
            
        return response
        
    def _parse_creative_response(self, response: str, content_type: str) -> Dict[str, Any]:
        """Parse creative content response"""
        result = {
            "type": content_type,
            "concept": "",
            "implementation": "",
            "variations": [],
            "integration": ""
        }
        
        sections = self._parse_structured_text(response)
        
        # Map sections to result
        for key, value in sections.items():
            key_lower = key.lower()
            if 'concept' in key_lower:
                result["concept"] = value
            elif 'implementation' in key_lower:
                result["implementation"] = value
            elif 'variation' in key_lower or 'alternative' in key_lower:
                result["variations"] = [v.strip() for v in value.split('\n') if v.strip()]
            elif 'integration' in key_lower:
                result["integration"] = value
                
        return result
        
    def _parse_ai_design(self, response: str) -> Dict[str, Any]:
        """Parse AI behavior design"""
        result = {
            "state_machine": {},
            "decision_tree": {},
            "implementation": "",
            "parameters": {}
        }
        
        sections = self._parse_structured_text(response)
        
        for key, value in sections.items():
            key_lower = key.lower()
            if 'state' in key_lower:
                result["state_machine"] = self._parse_state_machine(value)
            elif 'decision' in key_lower:
                result["decision_tree"] = self._parse_decision_tree(value)
            elif 'implementation' in key_lower or 'code' in key_lower:
                result["implementation"] = self._extract_code(value)
            elif 'parameter' in key_lower or 'tuning' in key_lower:
                result["parameters"] = self._parse_parameters(value)
                
        return result
        
    def _parse_state_machine(self, text: str) -> Dict[str, Any]:
        """Parse state machine description"""
        states = {}
        current_state = None
        
        for line in text.split('\n'):
            if line.strip() and '->' in line:
                # Transition
                parts = line.split('->')
                if len(parts) == 2 and current_state:
                    if 'transitions' not in states[current_state]:
                        states[current_state]['transitions'] = []
                    states[current_state]['transitions'].append({
                        'to': parts[1].strip(),
                        'condition': parts[0].strip()
                    })
            elif line.strip() and ':' in line:
                # State definition
                state_name = line.split(':')[0].strip()
                states[state_name] = {'description': line.split(':')[1].strip()}
                current_state = state_name
                
        return states
        
    def _parse_decision_tree(self, text: str) -> Dict[str, Any]:
        """Parse decision tree description"""
        # Simple parsing - can be enhanced
        return {"raw": text}
        
    def _parse_parameters(self, text: str) -> Dict[str, Any]:
        """Parse parameters from text"""
        params = {}
        
        for line in text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                try:
                    # Try to parse as number
                    if '.' in value:
                        params[key] = float(value.strip())
                    else:
                        params[key] = int(value.strip())
                except ValueError:
                    params[key] = value.strip()
                    
        return params
        
    def _parse_code_review(self, response: str) -> Dict[str, Any]:
        """Parse code review response"""
        result = {
            "score": 0,
            "issues": [],
            "performance": [],
            "security": [],
            "refactoring": [],
            "best_practices": []
        }
        
        sections = self._parse_structured_text(response)
        
        for key, value in sections.items():
            key_lower = key.lower()
            if 'score' in key_lower:
                try:
                    result["score"] = float(value.split('/')[0].strip())
                except:
                    pass
            elif 'issue' in key_lower or 'bug' in key_lower:
                result["issues"] = [v.strip() for v in value.split('\n') if v.strip()]
            elif 'performance' in key_lower:
                result["performance"] = [v.strip() for v in value.split('\n') if v.strip()]
            elif 'security' in key_lower:
                result["security"] = [v.strip() for v in value.split('\n') if v.strip()]
            elif 'refactor' in key_lower:
                result["refactoring"] = [v.strip() for v in value.split('\n') if v.strip()]
            elif 'practice' in key_lower:
                result["best_practices"] = [v.strip() for v in value.split('\n') if v.strip()]
                
        return result
        
    def _parse_narrative(self, response: str) -> Dict[str, Any]:
        """Parse game narrative response"""
        return self._parse_structured_text(response)
        
    def _parse_feature_list(self, response: str) -> List[Dict[str, Any]]:
        """Parse feature planning response"""
        features = []
        sections = response.split('\n\n')
        
        for section in sections:
            if section.strip():
                feature = {
                    "name": "",
                    "description": "",
                    "complexity": 3,
                    "value": 3,
                    "dependencies": [],
                    "time_estimate": ""
                }
                
                lines = section.split('\n')
                if lines:
                    feature["name"] = lines[0].strip()
                    
                for line in lines[1:]:
                    line_lower = line.lower()
                    if 'description' in line_lower:
                        feature["description"] = line.split(':', 1)[1].strip()
                    elif 'complexity' in line_lower:
                        try:
                            feature["complexity"] = int(line.split(':')[1].strip()[0])
                        except:
                            pass
                    elif 'value' in line_lower:
                        try:
                            feature["value"] = int(line.split(':')[1].strip()[0])
                        except:
                            pass
                            
                features.append(feature)
                
        return features
        
    # Mock methods for testing
    def _mock_analyze_game_design(self, description: str) -> Dict[str, Any]:
        """Mock game design analysis"""
        return {
            "overview": f"Game based on: {description}",
            "mechanics": ["Movement", "Combat", "Progression"],
            "scenes": ["MainMenu", "GameLevel", "GameOver"],
            "requirements": ["Physics2D", "Audio", "SaveSystem"],
            "steps": ["Setup project", "Create player", "Add enemies", "Implement UI"],
            "challenges": ["Balancing difficulty", "Performance optimization"]
        }
        
    def _mock_solve_complex_problem(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock problem solving"""
        return {
            "analysis": f"Analyzing: {problem}",
            "solutions": [
                {"approach": "Solution A", "pros": ["Fast"], "cons": ["Complex"]},
                {"approach": "Solution B", "pros": ["Simple"], "cons": ["Slower"]}
            ],
            "recommended": "Solution A",
            "implementation": "# Implementation code here"
        }
        
    def _mock_generate_creative_content(self, content_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock creative content generation"""
        return {
            "type": content_type,
            "concept": f"Creative {content_type} concept",
            "implementation": "Implementation details",
            "variations": ["Variation 1", "Variation 2"],
            "integration": "Integration suggestions"
        }
        
    def _mock_design_ai_behavior(self, entity_type: str, behavior_description: str) -> Dict[str, Any]:
        """Mock AI behavior design"""
        return {
            "state_machine": {
                "idle": {"description": "Waiting state"},
                "patrol": {"description": "Moving around"},
                "chase": {"description": "Following player"}
            },
            "decision_tree": {"root": "Check player distance"},
            "implementation": f"# AI implementation for {entity_type}",
            "parameters": {"detection_range": 100, "speed": 50}
        }
        
    def _mock_review_code_quality(self, code: str, language: str) -> Dict[str, Any]:
        """Mock code review"""
        return {
            "score": 7.5,
            "issues": ["Missing error handling", "Unused variable"],
            "performance": ["Consider object pooling"],
            "security": ["Validate user input"],
            "refactoring": ["Extract method for clarity"],
            "best_practices": ["Add documentation"]
        }
        
    def _mock_generate_game_narrative(self, genre: str, themes: List[str], length: str) -> Dict[str, Any]:
        """Mock narrative generation"""
        return {
            "story_arc": f"A {genre} story about {', '.join(themes)}",
            "characters": ["Hero", "Villain", "Mentor"],
            "plot_points": ["Introduction", "Conflict", "Resolution"],
            "dialogue": ["Example dialogue here"],
            "environmental": ["Visual storytelling ideas"],
            "choices": ["Choice A leads to X", "Choice B leads to Y"]
        }
        
    def _mock_plan_game_features(self, game_type: str, target_audience: str, scope: str) -> List[Dict[str, Any]]:
        """Mock feature planning"""
        return [
            {
                "name": "Core Gameplay Loop",
                "description": f"Main {game_type} mechanics",
                "complexity": 3,
                "value": 5,
                "dependencies": [],
                "time_estimate": "1 week"
            },
            {
                "name": "Progress System",
                "description": "Player progression and rewards",
                "complexity": 2,
                "value": 4,
                "dependencies": ["Core Gameplay Loop"],
                "time_estimate": "3 days"
            }
        ]
        
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
            
    def __del__(self):
        """Destructor"""
        self.cleanup()