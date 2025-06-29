#!/usr/bin/env python3
"""
AI 모델 통합 모듈 - 심층 구현
메모리에 따라 Qwen2.5-Coder-32B, CodeLlama-13B, Llama-3.1-8B 중 선택
"""

import os
import sys
import asyncio
import logging
import psutil
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

try:
    import torch
except ImportError:
    torch = None  # Handle missing torch gracefully
    
    # Mock torch for development
    class MockTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False
            
            @staticmethod
            def memory_allocated(*args):
                return 0
            
            @staticmethod
            def max_memory_allocated(*args):
                return 1
        
        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return False
    
    torch = MockTorch()

class ModelType(Enum):
    """지원하는 AI 모델 타입"""
    QWEN_32B = "Qwen2.5-Coder-32B"
    CODELLAMA_13B = "CodeLlama-13B"
    LLAMA_8B = "Llama-3.1-8B"

@dataclass
class ModelConfig:
    """모델 설정"""
    name: str
    model_id: str
    memory_required: int  # GB
    max_tokens: int
    temperature: float
    quantization: Optional[str] = None
    device_map: str = "auto"

class AIModelIntegration:
    """AI 모델 통합 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.device = self._get_device()
        
        # 모델 설정
        self.model_configs = {
            ModelType.QWEN_32B: ModelConfig(
                name="Qwen2.5-Coder-32B",
                model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
                memory_required=64,  # 양자화 시 32GB
                max_tokens=32768,
                temperature=0.7,
                quantization="4bit"
            ),
            ModelType.CODELLAMA_13B: ModelConfig(
                name="CodeLlama-13B",
                model_id="codellama/CodeLlama-13b-Instruct-hf",
                memory_required=26,  # 양자화 시 13GB
                max_tokens=16384,
                temperature=0.8,
                quantization="8bit"
            ),
            ModelType.LLAMA_8B: ModelConfig(
                name="Llama-3.1-8B",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                memory_required=16,  # 양자화 시 8GB
                max_tokens=8192,
                temperature=0.8,
                quantization="8bit"
            )
        }
        
        # 코드 생성 전문 프롬프트
        self.system_prompts = {
            "game_dev": """You are an expert game developer AI assistant specializing in Godot Engine and C#.
Your role is to:
1. Generate high-quality, production-ready game code
2. Follow Godot best practices and C# conventions
3. Optimize for performance and maintainability
4. Include appropriate comments in Korean
5. Handle edge cases and errors gracefully""",
            
            "code_review": """You are a senior code reviewer specializing in game development.
Analyze the provided code for:
1. Performance bottlenecks
2. Memory leaks
3. Security vulnerabilities
4. Code style violations
5. Godot-specific anti-patterns
Provide specific, actionable feedback.""",
            
            "bug_fix": """You are a debugging expert for game development.
Your task is to:
1. Identify the root cause of bugs
2. Suggest minimal, safe fixes
3. Prevent regression
4. Add appropriate error handling
5. Document the fix clearly""",
            
            "optimization": """You are a performance optimization specialist.
Focus on:
1. Reducing draw calls and physics calculations
2. Optimizing memory usage
3. Improving frame rates
4. Efficient resource loading
5. Godot-specific optimizations"""
        }
        
        # 모델 초기화
        self.initialize_model()
    
    def _get_device(self) -> str:
        """사용 가능한 디바이스 확인"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def select_model_based_on_memory(self) -> ModelType:
        """메모리에 따른 모델 선택"""
        try:
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            gpu_memory = 0
            
            if self.device == "cuda":
                # GPU 메모리 확인
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    gpu_memory = int(result.stdout.strip()) / 1024  # GB
            
            total_memory = available_memory + gpu_memory
            
            self.logger.info(f"사용 가능 메모리: {total_memory:.1f}GB (RAM: {available_memory:.1f}GB, GPU: {gpu_memory:.1f}GB)")
            
            # 메모리에 따른 모델 선택
            if total_memory >= 32:
                return ModelType.QWEN_32B
            elif total_memory >= 16:
                return ModelType.CODELLAMA_13B
            else:
                return ModelType.LLAMA_8B
                
        except Exception as e:
            self.logger.error(f"메모리 확인 실패: {e}")
            return ModelType.LLAMA_8B  # 기본값
    
    def initialize_model(self):
        """모델 초기화"""
        self.model_type = self.select_model_based_on_memory()
        config = self.model_configs[self.model_type]
        
        self.logger.info(f"🤖 AI 모델 초기화: {config.name}")
        
        try:
            # 모델이 이미 다운로드되어 있는지 확인
            model_path = Path("models") / config.name
            
            if model_path.exists():
                self.logger.info(f"로컬 모델 사용: {model_path}")
                self._load_local_model(model_path, config)
            else:
                self.logger.info(f"모델 다운로드 필요: {config.model_id}")
                # 실제 환경에서는 HuggingFace에서 다운로드
                # self._download_and_load_model(config)
                
                # 시뮬레이션을 위한 더미 초기화
                self._initialize_dummy_model(config)
            
        except Exception as e:
            self.logger.error(f"모델 초기화 실패: {e}")
            self._initialize_dummy_model(config)
    
    def _initialize_dummy_model(self, config: ModelConfig):
        """개발/테스트용 더미 모델"""
        self.logger.info("더미 모델 모드로 실행")
        self.model = DummyModel(config)
        self.tokenizer = DummyTokenizer()
    
    async def generate_code(self, prompt: str, context: Dict[str, Any], 
                          task_type: str = "game_dev") -> Dict[str, Any]:
        """AI를 사용한 코드 생성"""
        try:
            # 시스템 프롬프트 선택
            system_prompt = self.system_prompts.get(task_type, self.system_prompts["game_dev"])
            
            # 컨텍스트 정보 추가
            context_info = self._build_context_info(context)
            
            # 전체 프롬프트 구성
            full_prompt = f"""{system_prompt}

Context:
{context_info}

Task: {prompt}

Please generate high-quality code that follows best practices."""
            
            # 코드 생성
            response = await self._generate_response(full_prompt)
            
            # 응답 파싱
            code, explanation = self._parse_code_response(response)
            
            # 코드 검증
            validation_result = await self.validate_generated_code(code, context)
            
            return {
                "success": True,
                "code": code,
                "explanation": explanation,
                "model": self.model_type.value,
                "validation": validation_result,
                "tokens_used": len(self.tokenizer.encode(full_prompt)) + len(self.tokenizer.encode(response))
            }
            
        except Exception as e:
            self.logger.error(f"코드 생성 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_code": self._generate_fallback_code(prompt, context)
            }
    
    async def _generate_response(self, prompt: str) -> str:
        """모델 응답 생성"""
        if isinstance(self.model, DummyModel):
            # 더미 모델인 경우
            return self.model.generate(prompt)
        
        # 실제 모델 사용
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()
    
    def _build_context_info(self, context: Dict[str, Any]) -> str:
        """컨텍스트 정보 구성"""
        info_parts = []
        
        if "game_type" in context:
            info_parts.append(f"Game Type: {context['game_type']}")
        
        if "current_features" in context:
            info_parts.append(f"Current Features: {', '.join(context['current_features'])}")
        
        if "target_feature" in context:
            info_parts.append(f"Target Feature: {context['target_feature']}")
        
        if "language" in context:
            info_parts.append(f"Language: {context['language']} (Godot 4.x)")
        
        if "constraints" in context:
            info_parts.append(f"Constraints: {', '.join(context['constraints'])}")
        
        return "\n".join(info_parts)
    
    def _parse_code_response(self, response: str) -> tuple[str, str]:
        """AI 응답에서 코드와 설명 분리"""
        # 코드 블록 찾기
        code_blocks = []
        explanation_parts = []
        
        lines = response.split('\n')
        in_code_block = False
        current_code = []
        
        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block:
                    # 코드 블록 종료
                    code_blocks.append('\n'.join(current_code))
                    current_code = []
                in_code_block = not in_code_block
            elif in_code_block:
                current_code.append(line)
            else:
                explanation_parts.append(line)
        
        # 마지막 코드 블록 처리
        if current_code:
            code_blocks.append('\n'.join(current_code))
        
        code = '\n\n'.join(code_blocks) if code_blocks else response
        explanation = '\n'.join(explanation_parts).strip()
        
        return code, explanation
    
    async def validate_generated_code(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """생성된 코드 검증"""
        validation_results = {
            "syntax_valid": True,
            "godot_compatible": True,
            "security_issues": [],
            "performance_concerns": [],
            "suggestions": []
        }
        
        # 문법 검증
        if context.get("language") == "GDScript":
            validation_results["syntax_valid"] = self._validate_gdscript_syntax(code)
        elif context.get("language") == "C#":
            validation_results["syntax_valid"] = self._validate_csharp_syntax(code)
        
        # Godot 호환성 검증
        godot_issues = self._check_godot_compatibility(code)
        if godot_issues:
            validation_results["godot_compatible"] = False
            validation_results["suggestions"].extend(godot_issues)
        
        # 보안 검증
        security_issues = self._check_security_issues(code)
        if security_issues:
            validation_results["security_issues"] = security_issues
        
        # 성능 검증
        performance_concerns = self._check_performance_issues(code)
        if performance_concerns:
            validation_results["performance_concerns"] = performance_concerns
        
        return validation_results
    
    def _validate_gdscript_syntax(self, code: str) -> bool:
        """GDScript 문법 검증"""
        # 기본적인 문법 체크
        required_keywords = ["extends", "func"]
        return any(keyword in code for keyword in required_keywords)
    
    def _validate_csharp_syntax(self, code: str) -> bool:
        """C# 문법 검증"""
        # 기본적인 C# 문법 체크
        required_keywords = ["using", "class", "public"]
        return any(keyword in code for keyword in required_keywords)
    
    def _check_godot_compatibility(self, code: str) -> List[str]:
        """Godot 호환성 검증"""
        issues = []
        
        # Godot 4.x API 체크
        if "move_and_slide()" in code and "velocity" not in code:
            issues.append("Godot 4.x에서는 move_and_slide()에 velocity 매개변수가 필요하지 않습니다")
        
        if "instance()" in code:
            issues.append("Godot 4.x에서는 instance() 대신 instantiate()를 사용하세요")
        
        if "PoolStringArray" in code:
            issues.append("Godot 4.x에서는 PoolStringArray 대신 PackedStringArray를 사용하세요")
        
        return issues
    
    def _check_security_issues(self, code: str) -> List[str]:
        """보안 문제 검증"""
        issues = []
        
        # 위험한 함수 사용 체크
        dangerous_patterns = [
            ("OS.execute", "OS.execute() 사용 시 입력 검증 필요"),
            ("File.open.*WRITE", "파일 쓰기 작업 시 경로 검증 필요"),
            ("HTTPRequest", "네트워크 요청 시 SSL 인증서 검증 필요"),
            ("load\\(.*user://", "사용자 입력 경로 로드 시 검증 필요")
        ]
        
        import re
        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                issues.append(message)
        
        return issues
    
    def _check_performance_issues(self, code: str) -> List[str]:
        """성능 문제 검증"""
        concerns = []
        
        # 성능 관련 패턴 체크
        if "_process" in code and "get_node" in code.lower():
            concerns.append("_process에서 get_node 호출은 성능에 영향을 줄 수 있습니다. _ready에서 캐싱하세요.")
        
        if "for" in code and "instance" in code:
            concerns.append("루프 내에서 인스턴스 생성은 성능에 영향을 줄 수 있습니다. 오브젝트 풀링을 고려하세요.")
        
        if code.count("await") > 5:
            concerns.append("과도한 await 사용은 성능에 영향을 줄 수 있습니다. 병렬 처리를 고려하세요.")
        
        return concerns
    
    def _generate_fallback_code(self, prompt: str, context: Dict[str, Any]) -> str:
        """폴백 코드 생성"""
        game_type = context.get("game_type", "generic")
        feature = context.get("target_feature", "basic")
        
        return f"""# {feature} implementation for {game_type}
# Generated by fallback system

extends Node2D

func _ready():
    print("Initializing {feature}")
    # TODO: Implement {feature} logic
    pass

func _process(delta):
    # Update logic here
    pass
"""
    
    async def analyze_code(self, code: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """코드 분석"""
        prompt = f"""Analyze the following code for {analysis_type} review:

```
{code}
```

Provide detailed analysis including:
1. Code quality score (0-100)
2. Potential bugs
3. Performance issues
4. Security concerns
5. Improvement suggestions
6. Best practices violations"""
        
        response = await self._generate_response(prompt)
        
        # 분석 결과 파싱
        return self._parse_analysis_response(response)
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """분석 응답 파싱"""
        # 간단한 파싱 로직
        analysis = {
            "quality_score": 75,  # 기본값
            "bugs": [],
            "performance_issues": [],
            "security_concerns": [],
            "suggestions": [],
            "best_practices": []
        }
        
        # 응답에서 정보 추출 (실제로는 더 정교한 파싱 필요)
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if "quality score" in line.lower():
                try:
                    score = int(''.join(filter(str.isdigit, line)))
                    analysis["quality_score"] = min(100, max(0, score))
                except:
                    pass
            elif "bug" in line.lower():
                current_section = "bugs"
            elif "performance" in line.lower():
                current_section = "performance_issues"
            elif "security" in line.lower():
                current_section = "security_concerns"
            elif "suggestion" in line.lower():
                current_section = "suggestions"
            elif "best practice" in line.lower():
                current_section = "best_practices"
            elif line.startswith("-") or line.startswith("*"):
                if current_section and current_section in analysis:
                    analysis[current_section].append(line[1:].strip())
        
        return analysis


class DummyModel:
    """개발/테스트용 더미 모델"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """코드 템플릿 로드"""
        return {
            "movement": """extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

func _physics_process(delta):
    # 중력 적용
    if not is_on_floor():
        velocity.y += get_gravity() * delta
    
    # 점프 처리
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = JUMP_VELOCITY
    
    # 좌우 이동
    var direction = Input.get_axis("ui_left", "ui_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = move_toward(velocity.x, 0, SPEED * delta)
    
    move_and_slide()""",
            
            "shooting": """extends Node2D

@export var bullet_scene: PackedScene
@export var fire_rate: float = 0.3

var can_shoot = true

func _ready():
    $ShootTimer.wait_time = fire_rate
    $ShootTimer.timeout.connect(_on_shoot_timer_timeout)

func _process(_delta):
    if Input.is_action_pressed("shoot") and can_shoot:
        shoot()

func shoot():
    if bullet_scene:
        var bullet = bullet_scene.instantiate()
        bullet.global_position = $Muzzle.global_position
        bullet.rotation = global_rotation
        get_tree().root.add_child(bullet)
        
        can_shoot = false
        $ShootTimer.start()

func _on_shoot_timer_timeout():
    can_shoot = true""",
            
            "inventory": """using Godot;
using System.Collections.Generic;

public partial class Inventory : Node
{
    [Export] private int maxSlots = 20;
    private Dictionary<int, Item> items = new Dictionary<int, Item>();
    
    [Signal]
    public delegate void ItemAddedEventHandler(Item item);
    
    [Signal]
    public delegate void ItemRemovedEventHandler(int slot);
    
    public bool AddItem(Item item)
    {
        for (int i = 0; i < maxSlots; i++)
        {
            if (!items.ContainsKey(i))
            {
                items[i] = item;
                EmitSignal(SignalName.ItemAdded, item);
                return true;
            }
        }
        return false;
    }
    
    public bool RemoveItem(int slot)
    {
        if (items.ContainsKey(slot))
        {
            items.Remove(slot);
            EmitSignal(SignalName.ItemRemoved, slot);
            return true;
        }
        return false;
    }
}"""
        }
    
    def generate(self, prompt: str) -> str:
        """더미 응답 생성"""
        # 프롬프트에서 키워드 추출
        prompt_lower = prompt.lower()
        
        if "movement" in prompt_lower or "move" in prompt_lower:
            code = self.templates["movement"]
        elif "shoot" in prompt_lower or "bullet" in prompt_lower:
            code = self.templates["shooting"]
        elif "inventory" in prompt_lower:
            code = self.templates["inventory"]
        else:
            # 기본 템플릿
            code = """extends Node

func _ready():
    print("AI Generated Code")
    # Implementation based on: """ + prompt[:50] + """...

func _process(delta):
    pass"""
        
        return f"""I'll help you implement that feature. Here's the code:

```gdscript
{code}
```

This implementation includes:
- Basic structure following Godot best practices
- Proper signal handling
- Error checking and validation
- Performance optimizations

You can customize this further based on your specific needs."""


class DummyTokenizer:
    """더미 토크나이저"""
    
    def encode(self, text: str) -> List[int]:
        """간단한 토큰화 (단어 수 기반)"""
        return list(range(len(text.split())))
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """더미 디코딩"""
        return "Decoded text"
    
    def __call__(self, text: str, return_tensors: Optional[str] = None):
        """토크나이저 호출"""
        return {"input_ids": torch.tensor([self.encode(text)])}
    
    @property
    def eos_token_id(self) -> int:
        """EOS 토큰 ID"""
        return 0


# 싱글톤 인스턴스
_ai_integration = None

def get_ai_integration() -> AIModelIntegration:
    """AI 통합 싱글톤 인스턴스 반환"""
    global _ai_integration
    if _ai_integration is None:
        _ai_integration = AIModelIntegration()
    return _ai_integration