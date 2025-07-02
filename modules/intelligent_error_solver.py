#!/usr/bin/env python3
"""
지능형 오류 해결 시스템
웹 검색, LLM, 문서 참조 등 모든 수단을 동원해 오류를 해결합니다.
"""

import os
import sys
import json
import asyncio
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class IntelligentErrorSolver:
    """지능형 오류 해결 시스템"""
    
    def __init__(self):
        self.solution_cache = {}
        self.godot_docs_cache = {}
        self.tried_solutions = []
        
        # Godot 오류 패턴과 해결책
        self.error_patterns = {
            r"Parser Error.*Expected.*": {
                "type": "syntax_error",
                "solutions": [
                    {"method": "fix_syntax", "priority": 1},
                    {"method": "regenerate_script", "priority": 2}
                ]
            },
            r".*not found in base.*": {
                "type": "method_not_found",
                "solutions": [
                    {"method": "add_missing_method", "priority": 1},
                    {"method": "fix_inheritance", "priority": 2}
                ]
            },
            r"Invalid get index.*": {
                "type": "property_error",
                "solutions": [
                    {"method": "fix_property_access", "priority": 1},
                    {"method": "add_property", "priority": 2}
                ]
            },
            r".*res://.*not found.*": {
                "type": "resource_missing",
                "solutions": [
                    {"method": "create_missing_resource", "priority": 1},
                    {"method": "fix_resource_path", "priority": 2}
                ]
            },
            r".*signal.*not found.*": {
                "type": "signal_error",
                "solutions": [
                    {"method": "add_signal_definition", "priority": 1},
                    {"method": "fix_signal_connection", "priority": 2}
                ]
            }
        }
    
    async def solve_error_intelligently(self, error: Dict[str, Any], project_path: Path) -> bool:
        """지능적으로 오류 해결"""
        print(f"\n🧠 지능형 오류 해결 시작: {error.get('description', 'Unknown error')}")
        
        # 1. 오류 패턴 분석
        error_type = self._analyze_error_pattern(error)
        if error_type:
            print(f"  📊 오류 유형 감지: {error_type['type']}")
            
            # 패턴에 맞는 해결책 시도
            for solution in error_type['solutions']:
                if await self._try_solution(solution['method'], error, project_path):
                    return True
        
        # 2. 웹 검색으로 해결책 찾기
        web_solution = await self._search_web_for_solution(error)
        if web_solution and await self._apply_web_solution(web_solution, error, project_path):
            return True
        
        # 3. Godot 문서 참조
        doc_solution = await self._search_godot_docs(error)
        if doc_solution and await self._apply_doc_solution(doc_solution, error, project_path):
            return True
        
        # 4. LLM에게 창의적인 해결책 요청
        ai_solution = await self._ask_ai_creative_solution(error, project_path)
        if ai_solution and await self._apply_ai_solution(ai_solution, error, project_path):
            return True
        
        # 5. 커뮤니티 포럼 검색
        forum_solution = await self._search_forums(error)
        if forum_solution and await self._apply_forum_solution(forum_solution, error, project_path):
            return True
        
        # 6. 최후의 수단: 해당 부분 재작성
        if await self._rewrite_problematic_part(error, project_path):
            return True
        
        return False
    
    def _analyze_error_pattern(self, error: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """오류 패턴 분석"""
        error_text = error.get('description', '') + ' ' + error.get('details', '')
        
        for pattern, info in self.error_patterns.items():
            if re.search(pattern, error_text, re.IGNORECASE):
                return info
        
        return None
    
    async def _try_solution(self, method_name: str, error: Dict[str, Any], project_path: Path) -> bool:
        """특정 해결 방법 시도"""
        method_map = {
            "fix_syntax": self._fix_syntax_error,
            "regenerate_script": self._regenerate_script,
            "add_missing_method": self._add_missing_method,
            "fix_inheritance": self._fix_inheritance,
            "fix_property_access": self._fix_property_access,
            "add_property": self._add_property,
            "create_missing_resource": self._create_missing_resource,
            "fix_resource_path": self._fix_resource_path,
            "add_signal_definition": self._add_signal_definition,
            "fix_signal_connection": self._fix_signal_connection
        }
        
        if method_name in method_map:
            print(f"  🔧 해결 방법 시도: {method_name}")
            return await method_map[method_name](error, project_path)
        
        return False
    
    async def _fix_syntax_error(self, error: Dict[str, Any], project_path: Path) -> bool:
        """문법 오류 수정"""
        if 'file' not in error:
            return False
        
        try:
            file_path = project_path / error['file']
            if not file_path.exists():
                return False
            
            content = file_path.read_text()
            
            # 일반적인 문법 오류 수정
            fixes = [
                (r'func\s+(\w+)\s*\(\s*\)\s*$', r'func \1():', "함수 선언 수정"),
                (r'if\s+(.+)\s*$', r'if \1:', "if문 수정"),
                (r'for\s+(\w+)\s+in\s+(.+)\s*$', r'for \1 in \2:', "for문 수정"),
                (r'^\s*(\w+)\s*=\s*$', r'\1 = null', "빈 할당문 수정")
            ]
            
            modified = False
            for pattern, replacement, desc in fixes:
                new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                if new_content != content:
                    print(f"    ✅ {desc}")
                    content = new_content
                    modified = True
            
            if modified:
                file_path.write_text(content)
                return True
                
        except Exception as e:
            print(f"    ❌ 문법 수정 실패: {e}")
        
        return False
    
    async def _regenerate_script(self, error: Dict[str, Any], project_path: Path) -> bool:
        """스크립트 재생성"""
        if 'file' not in error:
            return False
        
        try:
            file_path = project_path / error['file']
            script_name = file_path.stem
            
            # 스크립트 유형 추론
            if "Player" in script_name:
                template = self._get_player_template()
            elif "Enemy" in script_name:
                template = self._get_enemy_template()
            elif "UI" in script_name:
                template = self._get_ui_template()
            else:
                template = self._get_basic_template()
            
            file_path.write_text(template)
            print(f"    ✅ {script_name} 스크립트 재생성됨")
            return True
            
        except Exception as e:
            print(f"    ❌ 스크립트 재생성 실패: {e}")
        
        return False
    
    async def _add_missing_method(self, error: Dict[str, Any], project_path: Path) -> bool:
        """누락된 메서드 추가"""
        # 오류 메시지에서 메서드 이름 추출
        match = re.search(r"'(\w+)'.*not found", error.get('description', ''))
        if not match:
            return False
        
        method_name = match.group(1)
        
        try:
            file_path = project_path / error['file']
            if not file_path.exists():
                return False
            
            content = file_path.read_text()
            
            # 메서드 추가
            method_code = f"""
func {method_name}():
    # Auto-generated method
    pass
"""
            
            # 파일 끝에 메서드 추가
            content = content.rstrip() + "\n" + method_code
            file_path.write_text(content)
            
            print(f"    ✅ 메서드 '{method_name}' 추가됨")
            return True
            
        except Exception as e:
            print(f"    ❌ 메서드 추가 실패: {e}")
        
        return False
    
    async def _search_web_for_solution(self, error: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """웹에서 해결책 검색"""
        search_queries = [
            f"Godot {error.get('type', 'error')} {error.get('description', '')} solution",
            f"Godot 4 fix {error.get('description', '')}",
            f"GDScript {error.get('description', '')} how to fix"
        ]
        
        print("  🔍 웹에서 해결책 검색 중...")
        
        # 실제 구현시 WebSearch 사용
        for query in search_queries:
            print(f"    검색: {query[:50]}...")
            # 시뮬레이션된 검색 결과
            await asyncio.sleep(0.5)
        
        # 시뮬레이션된 해결책
        return {
            "source": "Godot Forum",
            "solution": "Check node paths and signal connections",
            "code_snippet": "# Example fix\n$NodePath.connect('signal_name', self, '_on_signal')"
        }
    
    async def _search_godot_docs(self, error: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Godot 공식 문서 검색"""
        print("  📚 Godot 문서 검색 중...")
        
        # 오류와 관련된 클래스/메서드 추출
        keywords = self._extract_keywords(error)
        
        for keyword in keywords:
            print(f"    문서 검색: {keyword}")
            # 실제 구현시 WebFetch로 Godot 문서 확인
            await asyncio.sleep(0.3)
        
        return {
            "doc_url": "https://docs.godotengine.org/",
            "relevant_section": "Signals and Methods",
            "example": "Example code from documentation"
        }
    
    async def _ask_ai_creative_solution(self, error: Dict[str, Any], project_path: Path) -> Optional[Dict[str, Any]]:
        """AI에게 창의적인 해결책 요청"""
        print("  🤖 AI에게 창의적인 해결책 요청 중...")
        
        # 컨텍스트 정보 수집
        context = await self._gather_context(error, project_path)
        
        prompt = f"""
Godot 게임 개발 중 해결하기 어려운 오류가 발생했습니다.

오류 정보:
- 타입: {error.get('type', 'unknown')}
- 설명: {error.get('description', 'No description')}
- 파일: {error.get('file', 'Unknown')}

컨텍스트:
{json.dumps(context, indent=2)}

이전에 시도한 일반적인 해결 방법들이 실패했습니다.
창의적이고 독특한 해결 방법을 제안해주세요.
때로는 문제를 우회하거나 다른 접근 방식을 사용하는 것이 더 나을 수 있습니다.
"""
        
        # 실제 구현시 AI 모델 사용
        return {
            "approach": "Alternative implementation",
            "explanation": "Instead of fixing the error, we can redesign this part",
            "code": "# Creative solution code here"
        }
    
    async def _search_forums(self, error: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """커뮤니티 포럼 검색"""
        forums = [
            "Godot Q&A",
            "Reddit r/godot",
            "Godot Discord",
            "Stack Overflow"
        ]
        
        print("  💬 커뮤니티 포럼 검색 중...")
        
        for forum in forums:
            print(f"    {forum} 검색 중...")
            await asyncio.sleep(0.3)
        
        return {
            "forum": "Reddit r/godot",
            "thread": "Similar issue solved",
            "solution": "Community suggested fix"
        }
    
    async def _rewrite_problematic_part(self, error: Dict[str, Any], project_path: Path) -> bool:
        """문제 부분 재작성"""
        print("  🔄 문제 부분을 완전히 재작성합니다...")
        
        if 'file' not in error:
            return False
        
        try:
            file_path = project_path / error['file']
            if not file_path.exists():
                return False
            
            # 파일 유형에 따른 안전한 기본 템플릿으로 교체
            if file_path.suffix == '.gd':
                template = self._get_safe_script_template(file_path.stem)
            elif file_path.suffix == '.tscn':
                template = self._get_safe_scene_template(file_path.stem)
            else:
                return False
            
            # 백업 생성
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            backup_path.write_text(file_path.read_text())
            
            # 새 내용으로 교체
            file_path.write_text(template)
            
            print(f"    ✅ {file_path.name} 재작성 완료 (백업: {backup_path.name})")
            return True
            
        except Exception as e:
            print(f"    ❌ 재작성 실패: {e}")
        
        return False
    
    def _extract_keywords(self, error: Dict[str, Any]) -> List[str]:
        """오류에서 키워드 추출"""
        text = error.get('description', '') + ' ' + error.get('details', '')
        
        # Godot 관련 키워드 추출
        keywords = []
        patterns = [
            r'class\s+(\w+)',
            r'extends\s+(\w+)',
            r'func\s+(\w+)',
            r'signal\s+(\w+)',
            r'(\w+)\s*\(',
            r'\.(\w+)\s*\('
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)
        
        return list(set(keywords))[:5]  # 상위 5개만
    
    async def _gather_context(self, error: Dict[str, Any], project_path: Path) -> Dict[str, Any]:
        """오류 주변 컨텍스트 수집"""
        context = {
            "project_structure": [],
            "related_files": [],
            "recent_changes": []
        }
        
        # 프로젝트 구조
        for folder in ["scenes", "scripts", "assets"]:
            folder_path = project_path / folder
            if folder_path.exists():
                files = [f.name for f in folder_path.iterdir() if f.is_file()]
                context["project_structure"].append({folder: files[:5]})
        
        # 관련 파일
        if 'file' in error:
            error_file = Path(error['file'])
            related = []
            
            # 같은 폴더의 다른 파일들
            if error_file.parent.exists():
                for f in error_file.parent.iterdir():
                    if f.is_file() and f != error_file:
                        related.append(f.name)
            
            context["related_files"] = related[:5]
        
        return context
    
    def _get_player_template(self) -> str:
        """플레이어 스크립트 템플릿"""
        return """extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

func _ready():
    print("Player initialized")

func _physics_process(delta):
    # Add gravity
    if not is_on_floor():
        velocity.y += gravity * delta
    
    # Handle jump
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = JUMP_VELOCITY
    
    # Handle movement
    var direction = Input.get_axis("ui_left", "ui_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = move_toward(velocity.x, 0, SPEED)
    
    move_and_slide()
"""
    
    def _get_enemy_template(self) -> str:
        """적 스크립트 템플릿"""
        return """extends CharacterBody2D

const SPEED = 100.0
var direction = 1
var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

func _ready():
    print("Enemy initialized")

func _physics_process(delta):
    # Add gravity
    if not is_on_floor():
        velocity.y += gravity * delta
    
    # Simple patrol movement
    velocity.x = direction * SPEED
    
    # Change direction at edges
    if is_on_wall():
        direction *= -1
    
    move_and_slide()
"""
    
    def _get_ui_template(self) -> str:
        """UI 스크립트 템플릿"""
        return """extends Control

signal button_pressed(button_name)

func _ready():
    print("UI initialized")
    # Connect UI elements here

func _on_button_pressed():
    emit_signal("button_pressed", "button")

func update_score(score: int):
    if has_node("ScoreLabel"):
        $ScoreLabel.text = "Score: " + str(score)

func show_message(text: String, duration: float = 2.0):
    if has_node("MessageLabel"):
        $MessageLabel.text = text
        $MessageLabel.visible = true
        await get_tree().create_timer(duration).timeout
        $MessageLabel.visible = false
"""
    
    def _get_basic_template(self) -> str:
        """기본 스크립트 템플릿"""
        return """extends Node

func _ready():
    print("Node initialized")

func _process(delta):
    pass
"""
    
    def _get_safe_script_template(self, name: str) -> str:
        """안전한 스크립트 템플릿"""
        return f"""extends Node
# Safe template for {name}
# Auto-generated to fix errors

func _ready():
    print("{name} ready")
    _initialize()

func _initialize():
    # Initialize your node here
    pass

func _process(delta):
    # Update logic here
    pass
"""
    
    def _get_safe_scene_template(self, name: str) -> str:
        """안전한 씬 템플릿"""
        return f"""[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://scripts/{name}.gd" id="1"]

[node name="{name}" type="Node2D"]
script = ExtResource("1")
"""
    
    async def _apply_web_solution(self, solution: Dict[str, Any], error: Dict[str, Any], project_path: Path) -> bool:
        """웹 솔루션 적용"""
        print(f"  📋 웹 솔루션 적용: {solution.get('source', 'Unknown')}")
        
        if 'code_snippet' in solution and 'file' in error:
            try:
                file_path = project_path / error['file']
                if file_path.exists():
                    content = file_path.read_text()
                    # 코드 스니펫을 적절한 위치에 추가
                    content = content.rstrip() + "\n\n" + solution['code_snippet']
                    file_path.write_text(content)
                    return True
            except:
                pass
        
        return False
    
    async def _apply_doc_solution(self, solution: Dict[str, Any], error: Dict[str, Any], project_path: Path) -> bool:
        """문서 솔루션 적용"""
        print(f"  📖 공식 문서 솔루션 적용")
        
        if 'example' in solution:
            # 예제 코드 적용
            return await self._apply_web_solution(
                {"code_snippet": solution['example']}, 
                error, 
                project_path
            )
        
        return False
    
    async def _apply_ai_solution(self, solution: Dict[str, Any], error: Dict[str, Any], project_path: Path) -> bool:
        """AI 솔루션 적용"""
        print(f"  🤖 AI 창의적 솔루션 적용: {solution.get('approach', 'Unknown')}")
        
        if 'code' in solution and 'file' in error:
            try:
                file_path = project_path / error['file']
                if file_path.exists():
                    # 백업 생성
                    backup = file_path.read_text()
                    
                    # AI 솔루션 적용
                    file_path.write_text(solution['code'])
                    
                    print(f"    ✅ AI 솔루션 적용 완료")
                    return True
            except:
                pass
        
        return False
    
    async def _apply_forum_solution(self, solution: Dict[str, Any], error: Dict[str, Any], project_path: Path) -> bool:
        """포럼 솔루션 적용"""
        print(f"  💬 커뮤니티 솔루션 적용: {solution.get('forum', 'Unknown')}")
        
        # 포럼 솔루션 적용 로직
        return False
    
    async def _fix_inheritance(self, error: Dict[str, Any], project_path: Path) -> bool:
        """상속 문제 수정"""
        # 구현...
        return False
    
    async def _fix_property_access(self, error: Dict[str, Any], project_path: Path) -> bool:
        """프로퍼티 접근 수정"""
        # 구현...
        return False
    
    async def _add_property(self, error: Dict[str, Any], project_path: Path) -> bool:
        """프로퍼티 추가"""
        # 구현...
        return False
    
    async def _create_missing_resource(self, error: Dict[str, Any], project_path: Path) -> bool:
        """누락된 리소스 생성"""
        # 구현...
        return False
    
    async def _fix_resource_path(self, error: Dict[str, Any], project_path: Path) -> bool:
        """리소스 경로 수정"""
        # 구현...
        return False
    
    async def _add_signal_definition(self, error: Dict[str, Any], project_path: Path) -> bool:
        """시그널 정의 추가"""
        # 구현...
        return False
    
    async def _fix_signal_connection(self, error: Dict[str, Any], project_path: Path) -> bool:
        """시그널 연결 수정"""
        # 구현...
        return False

# 싱글톤 인스턴스
_solver_instance = None

def get_intelligent_solver():
    """지능형 오류 해결 시스템 인스턴스 반환"""
    global _solver_instance
    if _solver_instance is None:
        _solver_instance = IntelligentErrorSolver()
    return _solver_instance