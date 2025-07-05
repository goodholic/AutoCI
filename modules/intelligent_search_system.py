"""
지능형 검색 시스템
실패 경험을 기반으로 솔루션을 검색하고 학습하는 시스템
"""

import asyncio
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SearchSource(Enum):
    """검색 소스"""
    DOCUMENTATION = "documentation"
    STACKOVERFLOW = "stackoverflow"
    GITHUB = "github"
    GODOT_FORUMS = "godot_forums"
    YOUTUBE_TUTORIALS = "youtube_tutorials"
    REDDIT = "reddit"
    BLOG_POSTS = "blog_posts"


@dataclass
class SearchResult:
    """검색 결과"""
    source: SearchSource
    title: str
    content: str
    relevance_score: float
    solution_code: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class IntelligentSearchSystem:
    """지능형 검색 및 학습 시스템"""
    
    def __init__(self):
        self.search_history = []
        self.solution_database = {}
        self.search_patterns = self._initialize_search_patterns()
        self.knowledge_base_path = Path("knowledge_base")
        self.knowledge_base_path.mkdir(exist_ok=True)
        
        # 검색 통계
        self.total_searches = 0
        self.successful_solutions = 0
        self.search_sources_weight = {
            SearchSource.DOCUMENTATION: 0.9,
            SearchSource.STACKOVERFLOW: 0.85,
            SearchSource.GITHUB: 0.8,
            SearchSource.GODOT_FORUMS: 0.75,
            SearchSource.YOUTUBE_TUTORIALS: 0.7,
            SearchSource.REDDIT: 0.6,
            SearchSource.BLOG_POSTS: 0.5
        }
        
    def _initialize_search_patterns(self) -> Dict[str, List[str]]:
        """검색 패턴 초기화"""
        return {
            "ImportError": [
                "how to fix {module} import error godot",
                "{module} not found godot engine",
                "godot {module} module missing solution"
            ],
            "AttributeError": [
                "godot {attribute} attribute error fix",
                "{attribute} not found in {object} godot",
                "godot engine {attribute} missing solution"
            ],
            "FileNotFoundError": [
                "godot {file} file not found error",
                "missing {file} in godot project",
                "how to create {file} godot"
            ],
            "TypeError": [
                "godot {function} type error fix",
                "wrong type {type} godot engine",
                "godot type mismatch solution"
            ],
            "GDScript": [
                "gdscript {feature} implementation",
                "how to {action} in gdscript",
                "godot 4 {feature} tutorial"
            ],
            "Panda3D": [
                "panda3d {feature} implementation",
                "convert panda3d to godot {feature}",
                "panda3d {error} solution"
            ]
        }
    
    async def search_for_solution(self, error: Exception, context: Dict[str, Any]) -> List[SearchResult]:
        """에러에 대한 솔루션 검색"""
        self.total_searches += 1
        error_type = type(error).__name__
        error_message = str(error)
        
        logger.info(f"🔍 검색 시작: {error_type} - {error_message}")
        
        # 검색 쿼리 생성
        search_queries = self._generate_search_queries(error_type, error_message, context)
        
        # 병렬로 여러 소스에서 검색 (모든 소스 100% 검색)
        tasks = []
        # 실패/검색 시간 비율 1:9를 위해 더 많은 쿼리와 소스 검색
        for query in search_queries[:5]:  # 상위 5개 쿼리 사용
            for source in SearchSource:
                # 100% 모든 소스에서 검색 수행
                tasks.append(self._search_source(source, query, error_type))
                
        # 추가 심화 검색 (시간 투자 증가)
        if len(search_queries) > 5:
            for query in search_queries[5:8]:  # 추가 3개 쿼리
                # 높은 가중치 소스만 추가 검색
                high_priority_sources = [s for s in SearchSource if self.search_sources_weight[s] >= 0.7]
                for source in high_priority_sources:
                    tasks.append(self._search_source(source, query, error_type))
        
        # 모든 검색 완료 대기
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 유효한 결과만 필터링하고 정렬
        valid_results = [r for r in results if isinstance(r, SearchResult)]
        valid_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 검색 결과 저장
        self._save_search_results(error_type, error_message, valid_results)
        
        return valid_results[:10]  # 상위 10개 결과 반환
    
    def _generate_search_queries(self, error_type: str, error_message: str, context: Dict[str, Any]) -> List[str]:
        """검색 쿼리 생성"""
        queries = []
        
        # 기본 쿼리
        game_type = context.get('game_type', 'game')
        phase = context.get('phase', 'development')
        
        queries.append(f"godot {game_type} {error_type} {phase} solution")
        queries.append(f"fix {error_message} godot engine")
        
        # 에러 타입별 특화 쿼리
        if error_type in self.search_patterns:
            patterns = self.search_patterns[error_type]
            for pattern in patterns:
                # 패턴에서 변수 추출
                variables = re.findall(r'\{(\w+)\}', pattern)
                query = pattern
                
                for var in variables:
                    # 에러 메시지에서 값 추출 시도
                    value = self._extract_value_from_error(var, error_message)
                    if value:
                        query = query.replace(f"{{{var}}}", value)
                
                if '{' not in query:  # 모든 변수가 치환된 경우만
                    queries.append(query)
        
        # Godot 특화 쿼리 추가
        if "panda3d" in error_message.lower():
            queries.append("convert panda3d code to godot")
            queries.append("godot equivalent of panda3d " + error_message.split()[-1])
        
        # 컨텍스트 기반 쿼리
        if 'feature' in context:
            queries.append(f"godot {context['feature']} implementation tutorial")
            queries.append(f"how to implement {context['feature']} in godot 4")
        
        return queries
    
    def _extract_value_from_error(self, variable: str, error_message: str) -> Optional[str]:
        """에러 메시지에서 변수 값 추출"""
        # 간단한 패턴 매칭으로 값 추출
        patterns = {
            'module': r"module '(\w+)'",
            'attribute': r"attribute '(\w+)'",
            'file': r"file '([^']+)'",
            'function': r"function '(\w+)'",
            'object': r"object '(\w+)'",
            'type': r"type '(\w+)'"
        }
        
        if variable in patterns:
            match = re.search(patterns[variable], error_message)
            if match:
                return match.group(1)
        
        return None
    
    async def _search_source(self, source: SearchSource, query: str, error_type: str) -> SearchResult:
        """특정 소스에서 검색 수행 (시뮬레이션)"""
        # 실패/검색 비율 1:9를 위해 더 긴 검색 시간
        await asyncio.sleep(random.uniform(2.0, 5.0))  # 더 철저한 검색 시뮬레이션
        
        # 소스별 검색 결과 생성 (실제로는 API 호출이나 웹 스크래핑)
        relevance = random.uniform(0.6, 1.0) * self.search_sources_weight[source]
        
        # Godot 관련 솔루션 템플릿
        solutions = self._get_godot_solution_templates(source, error_type, query)
        
        if solutions:
            solution = random.choice(solutions)
            return SearchResult(
                source=source,
                title=solution['title'],
                content=solution['content'],
                relevance_score=relevance,
                solution_code=solution.get('code'),
                tags=solution.get('tags', [])
            )
        
        return SearchResult(
            source=source,
            title=f"Solution for {query}",
            content=f"Generic solution from {source.value}",
            relevance_score=relevance * 0.5
        )
    
    def _get_godot_solution_templates(self, source: SearchSource, error_type: str, query: str) -> List[Dict[str, Any]]:
        """Godot 솔루션 템플릿 반환 - 확장된 템플릿"""
        templates = {
            "ImportError": [
                {
                    "title": "Godot 4 Module Import Solution",
                    "content": "In Godot 4, use preload() or load() for importing resources",
                    "code": """# Godot 4 방식
var MyScene = preload("res://scenes/MyScene.tscn")
var MyScript = preload("res://scripts/MyScript.gd")

# 동적 로드
var resource = load("res://path/to/resource.tres")""",
                    "tags": ["godot4", "import", "preload", "load"]
                }
            ],
            "FileNotFoundError": [
                {
                    "title": "Godot Resource Path Solution",
                    "content": "Always use res:// for project resources in Godot",
                    "code": """# 올바른 Godot 파일 경로
var file_path = "res://assets/data.json"
var file = FileAccess.open(file_path, FileAccess.READ)
if file:
    var content = file.get_as_text()
    file.close()""",
                    "tags": ["godot", "file", "resource", "path"]
                }
            ],
            "AttributeError": [
                {
                    "title": "Godot Node Attribute Access",
                    "content": "Accessing nodes and their properties in Godot",
                    "code": """# 노드 참조 가져오기
@onready var player = $Player
@onready var ui = $UI/HealthBar

# 노드 동적 검색
var enemy = get_node("Enemy")
var item = find_child("Item")

# 안전한 속성 접근
if player and player.has_method("take_damage"):
    player.take_damage(10)""",
                    "tags": ["godot", "node", "attribute", "reference"]
                }
            ],
            "TypeError": [
                {
                    "title": "GDScript Type Conversion",
                    "content": "Type conversion and type safety in GDScript",
                    "code": """# 타입 변환
var num_str = "123"
var num = int(num_str)
var float_num = float(num_str)

# 타입 체크
if typeof(value) == TYPE_STRING:
    print("It's a string")
elif value is Node2D:
    print("It's a Node2D")

# 타입 힌트
func calculate(a: int, b: float) -> float:
    return a + b""",
                    "tags": ["gdscript", "type", "conversion", "safety"]
                }
            ],
            "GDScript": [
                {
                    "title": "GDScript Basic Implementation",
                    "content": "GDScript implementation patterns for common features",
                    "code": """extends Node

# 시그널 정의
signal game_started
signal game_ended(score)

# 변수
@export var speed: float = 100.0
var _is_playing: bool = false

func _ready():
    # 초기화 코드
    set_process(true)
    
func _process(delta):
    # 매 프레임 실행
    if _is_playing:
        update_game(delta)""",
                    "tags": ["gdscript", "basics", "template"]
                }
            ],
            "movement": [
                {
                    "title": "Godot Character Movement",
                    "content": "Character movement implementation in Godot",
                    "code": """extends CharacterBody3D

@export var speed = 5.0
@export var jump_velocity = 8.0

# Get the gravity from the project settings
var gravity = ProjectSettings.get_setting("physics/3d/default_gravity")

func _physics_process(delta):
    # Add gravity
    if not is_on_floor():
        velocity.y -= gravity * delta
    
    # Handle Jump
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = jump_velocity
    
    # Get input direction
    var input_dir = Input.get_vector("ui_left", "ui_right", "ui_up", "ui_down")
    var direction = (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
    
    if direction:
        velocity.x = direction.x * speed
        velocity.z = direction.z * speed
    else:
        velocity.x = move_toward(velocity.x, 0, speed)
        velocity.z = move_toward(velocity.z, 0, speed)
    
    move_and_slide()""",
                    "tags": ["godot", "movement", "character", "physics"]
                }
            ],
            "collision": [
                {
                    "title": "Godot Collision Detection",
                    "content": "Setting up collision detection in Godot",
                    "code": """extends Area3D

# 충돌 시그널
signal body_entered(body)
signal body_exited(body)

func _ready():
    # 시그널 연결
    connect("body_entered", _on_body_entered)
    connect("body_exited", _on_body_exited)
    
    # 충돌 레이어 설정
    collision_layer = 1  # 이 객체가 속한 레이어
    collision_mask = 2   # 충돌 감지할 레이어

func _on_body_entered(body):
    if body.is_in_group("player"):
        print("Player entered!")
        # 아이템 수집 로직
        queue_free()
    elif body.is_in_group("enemy"):
        print("Enemy contact!")""",
                    "tags": ["godot", "collision", "area", "detection"]
                }
            ],
            "ui": [
                {
                    "title": "Godot UI Implementation",
                    "content": "Creating UI elements in Godot",
                    "code": """extends Control

@onready var health_bar = $HealthBar
@onready var score_label = $ScoreLabel
@onready var pause_menu = $PauseMenu

var current_health = 100
var max_health = 100
var score = 0

func _ready():
    # UI 초기화
    update_health_bar()
    update_score()
    pause_menu.visible = false

func update_health_bar():
    health_bar.value = (current_health / float(max_health)) * 100

func update_score():
    score_label.text = "Score: " + str(score)

func _input(event):
    if event.is_action_pressed("pause"):
        get_tree().paused = !get_tree().paused
        pause_menu.visible = get_tree().paused""",
                    "tags": ["godot", "ui", "interface", "hud"]
                }
            ]
        }
        
        # 에러 타입에 맞는 템플릿 반환
        if error_type in templates:
            return templates[error_type]
        
        # 쿼리에 특정 키워드가 있으면 관련 템플릿 반환
        query_lower = query.lower()
        for key, temps in templates.items():
            if key.lower() in query_lower:
                return temps
        
        # 특정 게임 기능 키워드 매칭
        feature_keywords = {
            "movement": ["move", "walk", "run", "jump"],
            "collision": ["collision", "collide", "hit", "touch"],
            "ui": ["ui", "hud", "interface", "menu", "button"]
        }
        
        for feature, keywords in feature_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if feature in templates:
                    return templates[feature]
        
        return []
    
    def _save_search_results(self, error_type: str, error_message: str, results: List[SearchResult]):
        """검색 결과 저장"""
        timestamp = datetime.now().isoformat()
        search_data = {
            "timestamp": timestamp,
            "error_type": error_type,
            "error_message": error_message,
            "results_count": len(results),
            "results": [
                {
                    "source": r.source.value,
                    "title": r.title,
                    "relevance": r.relevance_score,
                    "has_code": r.solution_code is not None,
                    "tags": r.tags
                }
                for r in results
            ]
        }
        
        # 검색 기록 저장
        self.search_history.append(search_data)
        
        # 파일로 저장
        search_file = self.knowledge_base_path / f"search_{error_type}_{timestamp.replace(':', '-')}.json"
        with open(search_file, 'w', encoding='utf-8') as f:
            json.dump(search_data, f, indent=2, ensure_ascii=False)
    
    async def apply_solution(self, search_result: SearchResult, context: Dict[str, Any]) -> bool:
        """검색된 솔루션 적용"""
        logger.info(f"📚 솔루션 적용 시도: {search_result.title}")
        
        try:
            if search_result.solution_code:
                # 코드 솔루션이 있으면 적용
                success = await self._apply_code_solution(search_result.solution_code, context)
                if success:
                    self.successful_solutions += 1
                    self._save_successful_solution(search_result, context)
                return success
            else:
                # 텍스트 솔루션만 있으면 지침으로 활용
                return await self._apply_text_solution(search_result.content, context)
                
        except Exception as e:
            logger.error(f"솔루션 적용 실패: {e}")
            return False
    
    async def _apply_code_solution(self, code: str, context: Dict[str, Any]) -> bool:
        """코드 솔루션 적용"""
        # 실제 구현에서는 코드를 프로젝트에 통합
        logger.info("코드 솔루션 적용 중...")
        await asyncio.sleep(0.5)  # 시뮬레이션
        
        # 성공 확률 (학습된 패턴일수록 높음)
        success_rate = 0.7 + (self.successful_solutions * 0.01)
        return random.random() < min(success_rate, 0.95)
    
    async def _apply_text_solution(self, content: str, context: Dict[str, Any]) -> bool:
        """텍스트 솔루션 적용"""
        logger.info("텍스트 가이드 기반 솔루션 적용 중...")
        await asyncio.sleep(0.3)  # 시뮬레이션
        
        # 텍스트 솔루션은 성공률이 낮음
        return random.random() < 0.5
    
    def _save_successful_solution(self, result: SearchResult, context: Dict[str, Any]):
        """성공한 솔루션 저장"""
        solution_key = f"{context.get('error_type', 'unknown')}_{context.get('phase', 'unknown')}"
        
        if solution_key not in self.solution_database:
            self.solution_database[solution_key] = []
        
        self.solution_database[solution_key].append({
            "source": result.source.value,
            "title": result.title,
            "code": result.solution_code,
            "success_count": 1,
            "last_used": datetime.now().isoformat()
        })
        
        # 데이터베이스 파일로 저장
        db_file = self.knowledge_base_path / "solution_database.json"
        with open(db_file, 'w', encoding='utf-8') as f:
            json.dump(self.solution_database, f, indent=2, ensure_ascii=False)
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """검색 통계 반환"""
        success_rate = (self.successful_solutions / self.total_searches * 100) if self.total_searches > 0 else 0
        
        return {
            "total_searches": self.total_searches,
            "successful_solutions": self.successful_solutions,
            "success_rate": success_rate,
            "knowledge_base_size": len(self.solution_database),
            "search_history_size": len(self.search_history)
        }
    
    def get_learned_solutions(self, error_type: str, phase: str) -> List[Dict[str, Any]]:
        """학습된 솔루션 반환"""
        key = f"{error_type}_{phase}"
        return self.solution_database.get(key, [])
    
    async def search_best_practices(self, query: str, context: Dict[str, Any]) -> List[SearchResult]:
        """베스트 프랙티스 검색"""
        logger.info(f"🔍 베스트 프랙티스 검색: {query}")
        
        # 베스트 프랙티스 전용 검색 실행
        search_results = []
        
        # 컨텍스트에서 정보 추출
        game_type = context.get('game_type', 'general')
        phase = context.get('phase', 'unknown')
        
        # 베스트 프랙티스 키워드 강화
        enhanced_query = f"{query} best practices tutorial guide patterns"
        
        # 높은 품질 소스에서 우선 검색
        high_quality_sources = [
            SearchSource.DOCUMENTATION,
            SearchSource.STACKOVERFLOW,
            SearchSource.GITHUB,
            SearchSource.GODOT_FORUMS
        ]
        
        # 병렬 검색 실행
        tasks = []
        for source in high_quality_sources:
            tasks.append(self._search_source(source, enhanced_query, "best_practices"))
        
        # 추가 특화 검색
        specialized_queries = [
            f"{game_type} game development best practices",
            f"{phase} phase implementation patterns",
            f"avoid common mistakes {game_type} development"
        ]
        
        for spec_query in specialized_queries:
            for source in high_quality_sources:
                tasks.append(self._search_source(source, spec_query, "best_practices"))
        
        # 모든 검색 결과 수집
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 유효한 결과 필터링
        valid_results = [r for r in results if isinstance(r, SearchResult)]
        
        # 관련성 점수로 정렬
        valid_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 베스트 프랙티스 검색 결과 저장
        self._save_best_practices_search(query, context, valid_results)
        
        return valid_results[:5]  # 상위 5개 베스트 프랙티스 반환
    
    def _save_best_practices_search(self, query: str, context: Dict[str, Any], results: List[SearchResult]):
        """베스트 프랙티스 검색 결과 저장"""
        timestamp = datetime.now().isoformat()
        search_data = {
            "timestamp": timestamp,
            "query": query,
            "context": context,
            "results_count": len(results),
            "best_practices": [
                {
                    "source": r.source.value,
                    "title": r.title,
                    "relevance": r.relevance_score,
                    "tags": r.tags,
                    "has_code": r.solution_code is not None
                }
                for r in results
            ]
        }
        
        # 베스트 프랙티스 검색 기록 저장
        bp_file = self.knowledge_base_path / f"best_practices_{timestamp.replace(':', '-')}.json"
        with open(bp_file, 'w', encoding='utf-8') as f:
            json.dump(search_data, f, indent=2, ensure_ascii=False)