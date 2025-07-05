#!/usr/bin/env python3
"""
Godot 엔진 세부 지식 학습 시스템

Godot 엔진의 모든 세부사항을 학습하고 관리하는 전문 시스템
상용 AI 모델 수준의 깊이 있는 지식 관리
"""

import os
import json
import time
import sqlite3
import hashlib
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import re
import ast


@dataclass
class GodotKnowledge:
    """Godot 지식 항목"""
    id: str
    category: str  # core, node, gdscript, physics, rendering, audio, networking 등
    subcategory: str
    topic: str
    content: Dict[str, Any]
    examples: List[Dict[str, Any]]
    best_practices: List[str]
    common_errors: List[Dict[str, Any]]
    performance_tips: List[str]
    version_info: Dict[str, Any]
    related_topics: List[str]
    difficulty_level: int  # 1-10
    usage_frequency: float  # 0-1
    last_updated: str
    confidence_score: float
    source_references: List[str]
    tags: List[str]


@dataclass
class GodotAPI:
    """Godot API 정보"""
    class_name: str
    inherits: str
    brief_description: str
    description: str
    methods: List[Dict[str, Any]]
    properties: List[Dict[str, Any]]
    signals: List[Dict[str, Any]]
    constants: List[Dict[str, Any]]
    theme_items: List[Dict[str, Any]]
    examples: List[str]
    version: str


class GodotKnowledgeSystem:
    """Godot 엔진 전문 지식 시스템"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path("/mnt/d/AutoCI/AutoCI/godot_knowledge")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 경로
        self.db_path = self.base_path / "godot_knowledge.db"
        self.api_db_path = self.base_path / "godot_api.db"
        
        # 지식 카테고리
        self.categories = {
            "core": ["engine", "project_settings", "input", "resources", "scenes"],
            "nodes": ["2d", "3d", "ui", "animation", "particles", "shaders"],
            "gdscript": ["syntax", "classes", "signals", "coroutines", "patterns"],
            "csharp": ["integration", "api_differences", "performance", "mono"],
            "physics": ["2d_physics", "3d_physics", "collision", "rigid_body", "areas"],
            "rendering": ["viewport", "lighting", "materials", "post_processing", "optimization"],
            "audio": ["audio_stream", "audio_bus", "3d_audio", "effects"],
            "networking": ["multiplayer", "rpc", "webrtc", "websocket", "http"],
            "platform": ["export", "mobile", "web", "console", "vr_ar"],
            "tools": ["editor", "debugger", "profiler", "version_control"],
            "best_practices": ["architecture", "performance", "organization", "patterns"],
            "advanced": ["custom_modules", "gdnative", "engine_compilation", "plugins"]
        }
        
        # 캐시
        self.knowledge_cache: Dict[str, GodotKnowledge] = {}
        self.api_cache: Dict[str, GodotAPI] = {}
        self.search_cache: Dict[str, List[Any]] = {}
        
        # 학습 통계
        self.stats = {
            "total_knowledge_items": 0,
            "total_api_items": 0,
            "categories_covered": set(),
            "last_update": None,
            "query_count": 0,
            "cache_hits": 0,
            "learning_sessions": 0
        }
        
        # 초기화
        self._initialize_database()
        self._load_core_knowledge()
        
        # 백그라운드 학습
        self.is_learning = True
        self.learning_thread = threading.Thread(target=self._continuous_learning)
        self.learning_thread.daemon = True
        self.learning_thread.start()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        # 지식 데이터베이스
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                category TEXT,
                subcategory TEXT,
                topic TEXT,
                content TEXT,
                examples TEXT,
                best_practices TEXT,
                common_errors TEXT,
                performance_tips TEXT,
                version_info TEXT,
                related_topics TEXT,
                difficulty_level INTEGER,
                usage_frequency REAL,
                last_updated TEXT,
                confidence_score REAL,
                source_references TEXT,
                tags TEXT,
                UNIQUE(category, subcategory, topic)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_index (
                term TEXT,
                knowledge_id TEXT,
                relevance REAL,
                FOREIGN KEY(knowledge_id) REFERENCES knowledge(id)
            )
        """)
        
        # 인덱스
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON knowledge(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic ON knowledge(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_search ON search_index(term)")
        
        conn.commit()
        conn.close()
        
        # API 데이터베이스
        conn = sqlite3.connect(self.api_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_classes (
                class_name TEXT PRIMARY KEY,
                inherits TEXT,
                brief_description TEXT,
                description TEXT,
                methods TEXT,
                properties TEXT,
                signals TEXT,
                constants TEXT,
                theme_items TEXT,
                examples TEXT,
                version TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_core_knowledge(self):
        """핵심 Godot 지식 로드"""
        # 기본 지식 항목들
        core_knowledge = [
            {
                "category": "core",
                "subcategory": "engine",
                "topic": "scene_tree",
                "content": {
                    "description": "SceneTree는 Godot의 핵심 구조로 모든 노드의 계층구조를 관리합니다.",
                    "key_concepts": [
                        "노드 계층구조",
                        "신호 시스템",
                        "그룹 관리",
                        "씬 전환"
                    ],
                    "important_methods": [
                        "get_tree()",
                        "add_child()",
                        "queue_free()",
                        "change_scene()"
                    ]
                },
                "examples": [
                    {
                        "title": "씬 전환",
                        "code": 'get_tree().change_scene("res://scenes/Level2.tscn")'
                    }
                ],
                "best_practices": [
                    "노드 이름은 명확하고 일관성 있게 지정",
                    "깊은 계층구조는 피하고 컴포지션 패턴 사용",
                    "queue_free()를 사용한 안전한 노드 제거"
                ],
                "difficulty_level": 3
            },
            {
                "category": "gdscript",
                "subcategory": "syntax",
                "topic": "signals",
                "content": {
                    "description": "시그널은 Godot의 이벤트 시스템으로 노드 간 통신을 담당합니다.",
                    "syntax": "signal signal_name(param1, param2)",
                    "emit_syntax": "emit_signal('signal_name', value1, value2)",
                    "connection_methods": ["connect()", "disconnect()", "is_connected()"]
                },
                "examples": [
                    {
                        "title": "시그널 정의 및 발송",
                        "code": """signal health_changed(new_health)

func take_damage(amount):
    health -= amount
    emit_signal('health_changed', health)"""
                    }
                ],
                "best_practices": [
                    "시그널 이름은 동사 과거형 사용 (clicked, died, collected)",
                    "너무 많은 매개변수 전달 피하기",
                    "약한 결합을 위해 시그널 사용"
                ],
                "difficulty_level": 4
            },
            {
                "category": "physics",
                "subcategory": "2d_physics",
                "topic": "collision_detection",
                "content": {
                    "description": "2D 충돌 감지 시스템",
                    "collision_shapes": ["CollisionShape2D", "CollisionPolygon2D"],
                    "layers_masks": {
                        "collision_layer": "이 객체가 속한 레이어",
                        "collision_mask": "이 객체가 충돌을 감지할 레이어"
                    }
                },
                "examples": [
                    {
                        "title": "Area2D 충돌 감지",
                        "code": """func _ready():
    area.connect('body_entered', self, '_on_body_entered')

func _on_body_entered(body):
    if body.is_in_group('enemies'):
        take_damage()"""
                    }
                ],
                "common_errors": [
                    {
                        "error": "충돌이 감지되지 않음",
                        "causes": ["레이어/마스크 설정 오류", "CollisionShape 없음"],
                        "solution": "Project Settings에서 레이어 이름 확인 및 설정"
                    }
                ],
                "difficulty_level": 5
            }
        ]
        
        # 지식 추가
        for knowledge in core_knowledge:
            self.add_knowledge(**knowledge)
    
    def add_knowledge(
        self,
        category: str,
        subcategory: str,
        topic: str,
        content: Dict[str, Any],
        examples: List[Dict[str, Any]] = None,
        best_practices: List[str] = None,
        common_errors: List[Dict[str, Any]] = None,
        performance_tips: List[str] = None,
        difficulty_level: int = 5,
        tags: List[str] = None
    ) -> str:
        """새로운 지식 추가"""
        knowledge_id = self._generate_id(category, subcategory, topic)
        
        knowledge = GodotKnowledge(
            id=knowledge_id,
            category=category,
            subcategory=subcategory,
            topic=topic,
            content=content,
            examples=examples or [],
            best_practices=best_practices or [],
            common_errors=common_errors or [],
            performance_tips=performance_tips or [],
            version_info={"godot_version": "4.0+", "last_verified": datetime.now().isoformat()},
            related_topics=[],
            difficulty_level=difficulty_level,
            usage_frequency=0.5,
            last_updated=datetime.now().isoformat(),
            confidence_score=0.8,
            source_references=[],
            tags=tags or [category, subcategory]
        )
        
        # 데이터베이스에 저장
        self._save_knowledge(knowledge)
        
        # 캐시 업데이트
        self.knowledge_cache[knowledge_id] = knowledge
        
        # 검색 인덱스 업데이트
        self._update_search_index(knowledge)
        
        # 통계 업데이트
        self.stats["total_knowledge_items"] += 1
        self.stats["categories_covered"].add(category)
        
        return knowledge_id
    
    def query_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        difficulty_range: Optional[Tuple[int, int]] = None
    ) -> List[GodotKnowledge]:
        """지식 검색"""
        self.stats["query_count"] += 1
        
        # 캐시 확인
        cache_key = f"{query}_{category}_{difficulty_range}"
        if cache_key in self.search_cache:
            self.stats["cache_hits"] += 1
            return self.search_cache[cache_key]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 기본 쿼리
        sql = """
            SELECT DISTINCT k.* FROM knowledge k
            LEFT JOIN search_index si ON k.id = si.knowledge_id
            WHERE (
                k.topic LIKE ? OR
                k.content LIKE ? OR
                k.tags LIKE ? OR
                si.term LIKE ?
            )
        """
        params = [f"%{query}%"] * 4
        
        # 카테고리 필터
        if category:
            sql += " AND k.category = ?"
            params.append(category)
        
        # 난이도 필터
        if difficulty_range:
            sql += " AND k.difficulty_level BETWEEN ? AND ?"
            params.extend(difficulty_range)
        
        sql += " ORDER BY si.relevance DESC, k.usage_frequency DESC LIMIT 20"
        
        cursor.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            knowledge = self._row_to_knowledge(row)
            results.append(knowledge)
        
        conn.close()
        
        # 캐시 저장
        self.search_cache[cache_key] = results
        
        return results
    
    def get_api_reference(self, class_name: str) -> Optional[GodotAPI]:
        """API 레퍼런스 가져오기"""
        # 캐시 확인
        if class_name in self.api_cache:
            return self.api_cache[class_name]
        
        conn = sqlite3.connect(self.api_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM api_classes WHERE class_name = ?
        """, (class_name,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            api = self._row_to_api(row)
            self.api_cache[class_name] = api
            return api
        
        return None
    
    def learn_from_code(self, code: str, context: Dict[str, Any]) -> List[str]:
        """코드에서 학습"""
        insights = []
        
        # GDScript 패턴 분석
        patterns = self._analyze_gdscript_patterns(code)
        
        for pattern in patterns:
            # 새로운 패턴 발견
            if pattern["confidence"] > 0.7:
                knowledge_id = self.add_knowledge(
                    category="gdscript",
                    subcategory="patterns",
                    topic=pattern["name"],
                    content={
                        "pattern": pattern["pattern"],
                        "usage": pattern["usage"],
                        "benefits": pattern.get("benefits", [])
                    },
                    examples=[{"code": pattern["example"]}],
                    difficulty_level=pattern.get("difficulty", 5)
                )
                insights.append(f"새로운 패턴 학습: {pattern['name']}")
        
        # API 사용 분석
        api_usage = self._analyze_api_usage(code)
        for api_call in api_usage:
            self._update_usage_frequency(api_call["class"], api_call["method"])
        
        return insights
    
    def get_best_practices(self, topic: str) -> List[Dict[str, Any]]:
        """특정 주제의 베스트 프랙티스"""
        practices = []
        
        # 관련 지식 검색
        knowledge_items = self.query_knowledge(topic, category="best_practices")
        
        for item in knowledge_items:
            practices.extend([
                {
                    "practice": practice,
                    "topic": item.topic,
                    "difficulty": item.difficulty_level,
                    "source": item.id
                }
                for practice in item.best_practices
            ])
        
        # 중요도순 정렬
        practices.sort(key=lambda x: x["difficulty"])
        
        return practices
    
    def get_common_errors(self, context: str) -> List[Dict[str, Any]]:
        """일반적인 오류 및 해결책"""
        errors = []
        
        # 컨텍스트 관련 지식 검색
        knowledge_items = self.query_knowledge(context)
        
        for item in knowledge_items:
            for error in item.common_errors:
                errors.append({
                    "error": error.get("error", ""),
                    "causes": error.get("causes", []),
                    "solution": error.get("solution", ""),
                    "prevention": error.get("prevention", ""),
                    "source": item.topic
                })
        
        return errors
    
    def suggest_optimization(self, code: str, target: str = "performance") -> List[Dict[str, Any]]:
        """최적화 제안"""
        suggestions = []
        
        # 코드 분석
        analysis = self._analyze_code_performance(code)
        
        # 최적화 가능 부분 찾기
        for issue in analysis["issues"]:
            # 관련 최적화 팁 검색
            knowledge_items = self.query_knowledge(f"{issue['type']} optimization")
            
            for item in knowledge_items:
                for tip in item.performance_tips:
                    if issue["type"] in tip.lower():
                        suggestions.append({
                            "issue": issue["description"],
                            "suggestion": tip,
                            "impact": issue.get("impact", "medium"),
                            "difficulty": item.difficulty_level,
                            "example": self._get_optimization_example(issue["type"])
                        })
        
        # 우선순위 정렬 (영향도 높고 난이도 낮은 순)
        suggestions.sort(key=lambda x: (-self._impact_score(x["impact"]), x["difficulty"]))
        
        return suggestions[:10]  # 상위 10개
    
    def generate_code_template(self, task: str, requirements: Dict[str, Any]) -> str:
        """코드 템플릿 생성"""
        # 작업 유형 분석
        task_type = self._analyze_task_type(task)
        
        # 관련 지식 검색
        knowledge_items = self.query_knowledge(task_type)
        
        # 템플릿 생성
        template = self._build_template(task_type, requirements, knowledge_items)
        
        return template
    
    def _analyze_gdscript_patterns(self, code: str) -> List[Dict[str, Any]]:
        """GDScript 패턴 분석"""
        patterns = []
        
        # 시그널 패턴
        signal_pattern = r'signal\s+(\w+)\s*\([^)]*\)'
        signals = re.findall(signal_pattern, code)
        if signals:
            patterns.append({
                "name": "signal_usage",
                "pattern": "Signal-based communication",
                "usage": f"Found {len(signals)} signals",
                "example": signals[0] if signals else "",
                "confidence": 0.9,
                "benefits": ["Loose coupling", "Event-driven architecture"]
            })
        
        # 노드 패턴
        node_pattern = r'(onready var|export var)\s+(\w+)'
        nodes = re.findall(node_pattern, code)
        if nodes:
            patterns.append({
                "name": "node_reference_pattern",
                "pattern": "Node reference caching",
                "usage": f"Found {len(nodes)} node references",
                "example": f"{nodes[0][0]} {nodes[0][1]}" if nodes else "",
                "confidence": 0.85
            })
        
        # 코루틴 패턴
        if 'yield(' in code or 'await ' in code:
            patterns.append({
                "name": "coroutine_pattern",
                "pattern": "Asynchronous programming",
                "usage": "Coroutines for async operations",
                "example": "await get_tree().create_timer(1.0).timeout",
                "confidence": 0.8,
                "difficulty": 6
            })
        
        return patterns
    
    def _analyze_api_usage(self, code: str) -> List[Dict[str, Any]]:
        """API 사용 분석"""
        api_usage = []
        
        # 메서드 호출 패턴
        method_pattern = r'(\w+)\.(\w+)\s*\('
        methods = re.findall(method_pattern, code)
        
        for obj, method in methods:
            api_usage.append({
                "class": obj,
                "method": method,
                "frequency": 1
            })
        
        # get_node 패턴
        get_node_pattern = r'get_node\s*\(\s*["\']([^"\']+)["\']\s*\)'
        node_paths = re.findall(get_node_pattern, code)
        if node_paths:
            api_usage.append({
                "class": "Node",
                "method": "get_node",
                "frequency": len(node_paths)
            })
        
        return api_usage
    
    def _analyze_code_performance(self, code: str) -> Dict[str, Any]:
        """코드 성능 분석"""
        issues = []
        
        # _process에서 무거운 작업
        if '_process(' in code or '_physics_process(' in code:
            process_content = self._extract_function_content(code, '_process')
            if 'for ' in process_content or 'while ' in process_content:
                issues.append({
                    "type": "heavy_process",
                    "description": "_process에서 반복문 사용",
                    "impact": "high"
                })
        
        # get_node 반복 호출
        get_node_count = code.count('get_node(')
        if get_node_count > 5:
            issues.append({
                "type": "repeated_get_node",
                "description": f"get_node가 {get_node_count}번 호출됨",
                "impact": "medium"
            })
        
        # 문자열 연결
        if '+' in code and ('"' in code or "'" in code):
            issues.append({
                "type": "string_concatenation",
                "description": "문자열 연결 사용",
                "impact": "low"
            })
        
        return {"issues": issues}
    
    def _build_template(self, task_type: str, requirements: Dict[str, Any], knowledge_items: List[GodotKnowledge]) -> str:
        """템플릿 빌드"""
        template = f"# {task_type.title()} Template\n"
        template += f"# Generated by Godot Knowledge System\n\n"
        
        # 기본 구조
        if "player" in task_type.lower():
            template += self._get_player_template(requirements)
        elif "enemy" in task_type.lower():
            template += self._get_enemy_template(requirements)
        elif "ui" in task_type.lower():
            template += self._get_ui_template(requirements)
        else:
            template += self._get_generic_template(task_type, requirements)
        
        # 지식 기반 추가
        for item in knowledge_items[:3]:  # 상위 3개
            if item.examples:
                template += f"\n# Example from {item.topic}:\n"
                template += f"# {item.examples[0].get('code', '')}\n"
        
        return template
    
    def _get_player_template(self, requirements: Dict[str, Any]) -> str:
        """플레이어 템플릿"""
        return """extends CharacterBody2D

# Player properties
@export var speed = 300.0
@export var jump_velocity = -400.0
@export var gravity = 980.0

# Get the gravity from the project settings to be synced with RigidBody nodes
var gravity_scale = ProjectSettings.get_setting("physics/2d/default_gravity")

func _ready():
    # Initialize player
    pass

func _physics_process(delta):
    # Add gravity
    if not is_on_floor():
        velocity.y += gravity * delta
    
    # Handle jump
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = jump_velocity
    
    # Get input direction
    var direction = Input.get_axis("ui_left", "ui_right")
    if direction:
        velocity.x = direction * speed
    else:
        velocity.x = move_toward(velocity.x, 0, speed)
    
    move_and_slide()
"""
    
    def _get_enemy_template(self, requirements: Dict[str, Any]) -> str:
        """적 템플릿"""
        return """extends CharacterBody2D

# Enemy properties
@export var speed = 150.0
@export var health = 100
@export var damage = 10
@export var detection_range = 300.0

@onready var sprite = $Sprite2D
@onready var collision = $CollisionShape2D

var player = null
var is_dead = false

signal died
signal damaged(amount)

func _ready():
    add_to_group("enemies")
    
func _physics_process(delta):
    if is_dead:
        return
        
    if player:
        # Move towards player
        var direction = (player.global_position - global_position).normalized()
        velocity = direction * speed
        move_and_slide()
        
        # Flip sprite
        if direction.x < 0:
            sprite.flip_h = true
        else:
            sprite.flip_h = false

func take_damage(amount):
    health -= amount
    emit_signal("damaged", amount)
    
    if health <= 0:
        die()

func die():
    is_dead = true
    emit_signal("died")
    queue_free()

func _on_detection_area_body_entered(body):
    if body.is_in_group("player"):
        player = body

func _on_detection_area_body_exited(body):
    if body == player:
        player = null
"""
    
    def _get_ui_template(self, requirements: Dict[str, Any]) -> str:
        """UI 템플릿"""
        return """extends Control

# UI elements
@onready var health_bar = $HealthBar
@onready var score_label = $ScoreLabel
@onready var pause_menu = $PauseMenu

var is_paused = false

signal pause_toggled(paused)

func _ready():
    pause_menu.visible = false
    
func _unhandled_input(event):
    if event.is_action_pressed("pause"):
        toggle_pause()

func toggle_pause():
    is_paused = !is_paused
    pause_menu.visible = is_paused
    get_tree().paused = is_paused
    emit_signal("pause_toggled", is_paused)

func update_health(current_health, max_health):
    health_bar.value = (current_health / max_health) * 100

func update_score(score):
    score_label.text = "Score: " + str(score)

func _on_resume_button_pressed():
    toggle_pause()

func _on_quit_button_pressed():
    get_tree().quit()
"""
    
    def _get_generic_template(self, task_type: str, requirements: Dict[str, Any]) -> str:
        """일반 템플릿"""
        return f"""extends Node

# {task_type} implementation
class_name {task_type.title().replace(' ', '')}

# Properties
@export var enabled = true

# Signals
signal task_completed
signal task_failed(reason)

func _ready():
    # Initialize
    pass

func execute():
    if not enabled:
        emit_signal("task_failed", "Task is disabled")
        return
    
    # Implementation here
    
    emit_signal("task_completed")
"""
    
    def _continuous_learning(self):
        """지속적 학습 프로세스"""
        while self.is_learning:
            try:
                # 주기적 지식 업데이트
                self._update_knowledge_confidence()
                
                # 사용 빈도 업데이트
                self._update_usage_frequencies()
                
                # 새로운 연결 발견
                self._discover_relationships()
                
                self.stats["learning_sessions"] += 1
                self.stats["last_update"] = datetime.now().isoformat()
                
                time.sleep(300)  # 5분마다
                
            except Exception as e:
                print(f"학습 오류: {e}")
                time.sleep(60)
    
    def _save_knowledge(self, knowledge: GodotKnowledge):
        """지식 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO knowledge VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            knowledge.id,
            knowledge.category,
            knowledge.subcategory,
            knowledge.topic,
            json.dumps(knowledge.content),
            json.dumps(knowledge.examples),
            json.dumps(knowledge.best_practices),
            json.dumps(knowledge.common_errors),
            json.dumps(knowledge.performance_tips),
            json.dumps(knowledge.version_info),
            json.dumps(knowledge.related_topics),
            knowledge.difficulty_level,
            knowledge.usage_frequency,
            knowledge.last_updated,
            knowledge.confidence_score,
            json.dumps(knowledge.source_references),
            json.dumps(knowledge.tags)
        ))
        
        conn.commit()
        conn.close()
    
    def _update_search_index(self, knowledge: GodotKnowledge):
        """검색 인덱스 업데이트"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 기존 인덱스 삭제
        cursor.execute("DELETE FROM search_index WHERE knowledge_id = ?", (knowledge.id,))
        
        # 검색 가능한 텀 추출
        terms = set()
        terms.add(knowledge.topic.lower())
        terms.update(knowledge.tags)
        terms.update(knowledge.category.split('_'))
        terms.update(knowledge.subcategory.split('_'))
        
        # 컨텐츠에서 키워드 추출
        if isinstance(knowledge.content, dict):
            for key, value in knowledge.content.items():
                if isinstance(value, str):
                    terms.update(value.lower().split()[:5])  # 처음 5단어
        
        # 인덱스 추가
        for term in terms:
            if len(term) > 2:  # 2글자 이상만
                cursor.execute("""
                    INSERT INTO search_index (term, knowledge_id, relevance)
                    VALUES (?, ?, ?)
                """, (term, knowledge.id, 1.0))
        
        conn.commit()
        conn.close()
    
    def _generate_id(self, *args) -> str:
        """ID 생성"""
        content = "_".join(str(arg) for arg in args)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _row_to_knowledge(self, row) -> GodotKnowledge:
        """데이터베이스 행을 지식 객체로 변환"""
        return GodotKnowledge(
            id=row[0],
            category=row[1],
            subcategory=row[2],
            topic=row[3],
            content=json.loads(row[4]),
            examples=json.loads(row[5]),
            best_practices=json.loads(row[6]),
            common_errors=json.loads(row[7]),
            performance_tips=json.loads(row[8]),
            version_info=json.loads(row[9]),
            related_topics=json.loads(row[10]),
            difficulty_level=row[11],
            usage_frequency=row[12],
            last_updated=row[13],
            confidence_score=row[14],
            source_references=json.loads(row[15]),
            tags=json.loads(row[16])
        )
    
    def _row_to_api(self, row) -> GodotAPI:
        """데이터베이스 행을 API 객체로 변환"""
        return GodotAPI(
            class_name=row[0],
            inherits=row[1],
            brief_description=row[2],
            description=row[3],
            methods=json.loads(row[4]),
            properties=json.loads(row[5]),
            signals=json.loads(row[6]),
            constants=json.loads(row[7]),
            theme_items=json.loads(row[8]),
            examples=json.loads(row[9]),
            version=row[10]
        )
    
    def _extract_function_content(self, code: str, function_name: str) -> str:
        """함수 내용 추출"""
        pattern = rf'func\s+{function_name}\s*\([^)]*\)\s*:\s*\n((?:\s+.*\n)*)'
        match = re.search(pattern, code)
        return match.group(1) if match else ""
    
    def _impact_score(self, impact: str) -> int:
        """영향도 점수"""
        scores = {"high": 3, "medium": 2, "low": 1}
        return scores.get(impact, 1)
    
    def _update_usage_frequency(self, class_name: str, method_name: str):
        """사용 빈도 업데이트"""
        # 구현 예정
        pass
    
    def _update_knowledge_confidence(self):
        """지식 신뢰도 업데이트"""
        # 구현 예정
        pass
    
    def _update_usage_frequencies(self):
        """사용 빈도 업데이트"""
        # 구현 예정
        pass
    
    def _discover_relationships(self):
        """관계 발견"""
        # 구현 예정
        pass
    
    def _get_optimization_example(self, optimization_type: str) -> str:
        """최적화 예제"""
        examples = {
            "heavy_process": """# Before
func _process(delta):
    for enemy in get_tree().get_nodes_in_group("enemies"):
        check_distance(enemy)

# After
var check_timer = 0.0
func _process(delta):
    check_timer += delta
    if check_timer > 0.1:  # Check every 0.1 seconds
        check_timer = 0.0
        for enemy in get_tree().get_nodes_in_group("enemies"):
            check_distance(enemy)""",
            
            "repeated_get_node": """# Before
func update_ui():
    get_node("UI/HealthBar").value = health
    get_node("UI/ManaBar").value = mana
    get_node("UI/ScoreLabel").text = str(score)

# After
@onready var health_bar = $UI/HealthBar
@onready var mana_bar = $UI/ManaBar
@onready var score_label = $UI/ScoreLabel

func update_ui():
    health_bar.value = health
    mana_bar.value = mana
    score_label.text = str(score)"""
        }
        return examples.get(optimization_type, "")
    
    def _analyze_task_type(self, task: str) -> str:
        """작업 유형 분석"""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["player", "character", "hero"]):
            return "player_controller"
        elif any(word in task_lower for word in ["enemy", "ai", "npc"]):
            return "enemy_ai"
        elif any(word in task_lower for word in ["ui", "menu", "hud"]):
            return "user_interface"
        elif any(word in task_lower for word in ["item", "pickup", "collectible"]):
            return "item_system"
        else:
            return "generic_system"
    
    def export_knowledge(self, output_path: str, format: str = "json"):
        """지식 내보내기"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM knowledge")
        all_knowledge = []
        
        for row in cursor.fetchall():
            knowledge = self._row_to_knowledge(row)
            all_knowledge.append(asdict(knowledge))
        
        conn.close()
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_knowledge, f, ensure_ascii=False, indent=2)
        
        return len(all_knowledge)
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            **self.stats,
            "cache_size": len(self.knowledge_cache),
            "api_cache_size": len(self.api_cache),
            "categories": list(self.stats["categories_covered"]),
            "total_categories": len(self.categories)
        }


def demo():
    """데모 실행"""
    print("Godot Knowledge System 데모")
    print("-" * 50)
    
    # 시스템 초기화
    knowledge_system = GodotKnowledgeSystem()
    
    # 지식 검색
    print("\n1. SceneTree 관련 지식 검색:")
    results = knowledge_system.query_knowledge("scene tree")
    for result in results[:3]:
        print(f"  - {result.topic}: {result.content.get('description', '')[:100]}...")
    
    # 베스트 프랙티스
    print("\n2. 시그널 베스트 프랙티스:")
    practices = knowledge_system.get_best_practices("signals")
    for practice in practices[:3]:
        print(f"  - {practice['practice']}")
    
    # 코드 템플릿 생성
    print("\n3. 플레이어 컨트롤러 템플릿 생성:")
    template = knowledge_system.generate_code_template(
        "player controller",
        {"movement": "2d", "features": ["jump", "dash"]}
    )
    print(template[:500] + "...")
    
    # 최적화 제안
    sample_code = """
func _process(delta):
    for enemy in get_tree().get_nodes_in_group("enemies"):
        var distance = position.distance_to(enemy.position)
        if distance < 100:
            attack(enemy)
    """
    
    print("\n4. 코드 최적화 제안:")
    suggestions = knowledge_system.suggest_optimization(sample_code)
    for suggestion in suggestions[:2]:
        print(f"  - {suggestion['issue']}: {suggestion['suggestion']}")
    
    # 통계
    print("\n5. 시스템 통계:")
    stats = knowledge_system.get_statistics()
    print(f"  - 총 지식 항목: {stats['total_knowledge_items']}")
    print(f"  - 카테고리: {len(stats['categories'])}/{stats['total_categories']}")
    print(f"  - 쿼리 수: {stats['query_count']}")
    print(f"  - 캐시 히트: {stats['cache_hits']}")


if __name__ == "__main__":
    demo()