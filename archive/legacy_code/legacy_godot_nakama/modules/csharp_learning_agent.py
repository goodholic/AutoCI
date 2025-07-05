#!/usr/bin/env python3
"""
C# 학습 전문 AI 에이전트
Godot 개발을 위한 C# 지식을 체계적으로 학습하고 적용
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import random

class CSharpLearningAgent:
    def __init__(self):
        self.knowledge_base = self._initialize_knowledge_base()
        self.learning_path = self._create_learning_path()
        self.code_patterns = {}
        self.godot_patterns = {}
        self.project_root = Path(__file__).parent.parent
        
    def _initialize_knowledge_base(self) -> Dict:
        """C# 지식 베이스 초기화"""
        return {
            "fundamentals": {
                "syntax": {
                    "variables": ["int", "string", "bool", "float", "double", "decimal", "char", "byte"],
                    "operators": ["arithmetic", "comparison", "logical", "bitwise", "assignment"],
                    "control_flow": ["if-else", "switch", "for", "while", "do-while", "foreach", "break", "continue"],
                    "methods": ["void", "return types", "parameters", "overloading", "optional parameters", "params"],
                    "error_handling": ["try-catch", "finally", "throw", "custom exceptions"]
                },
                "oop": {
                    "classes": ["constructors", "destructors", "fields", "properties", "methods"],
                    "inheritance": ["base class", "derived class", "virtual", "override", "sealed", "abstract"],
                    "interfaces": ["implementation", "multiple interfaces", "explicit implementation"],
                    "polymorphism": ["method overriding", "method overloading", "operator overloading"],
                    "encapsulation": ["access modifiers", "public", "private", "protected", "internal"],
                    "abstraction": ["abstract classes", "abstract methods", "interfaces"]
                },
                "advanced": {
                    "generics": ["generic classes", "generic methods", "constraints", "default values"],
                    "delegates": ["Action", "Func", "Predicate", "custom delegates", "multicast"],
                    "events": ["event declaration", "event handling", "EventHandler", "custom events"],
                    "linq": ["query syntax", "method syntax", "deferred execution", "immediate execution"],
                    "async": ["async/await", "Task", "Task<T>", "ConfigureAwait", "cancellation tokens"],
                    "reflection": ["Type", "Assembly", "attributes", "dynamic invocation"],
                    "memory": ["garbage collection", "IDisposable", "using statement", "weak references"]
                }
            },
            "godot_csharp": {
                "node_system": {
                    "basics": ["Node", "Node2D", "Node3D", "Control", "scene tree"],
                    "lifecycle": ["_Ready", "_Process", "_PhysicsProcess", "_Input", "_UnhandledInput"],
                    "signals": ["[Signal]", "EmitSignal", "Connect", "Disconnect", "custom signals"],
                    "groups": ["AddToGroup", "RemoveFromGroup", "GetTree().GetNodesInGroup"],
                    "scene_management": ["PackedScene", "Instance", "QueueFree", "ChangeScene"]
                },
                "gdscript_interop": {
                    "calling_gdscript": ["Call", "CallDeferred", "HasMethod"],
                    "properties": ["Export attribute", "GetNode<T>", "NodePath"],
                    "resources": ["Resource", "load", "preload", "custom resources"]
                },
                "godot_patterns": {
                    "autoload": "Global game systems and singletons",
                    "signals": "Decoupled communication between nodes",
                    "scene_inheritance": "Reusable scene templates",
                    "custom_resources": "Data containers for items, abilities"
                },
                "godot_specific": {
                    "physics": ["KinematicBody2D", "RigidBody2D", "Area2D", "CollisionShape2D"],
                    "rendering": ["Sprite", "AnimationPlayer", "Shader", "Viewport"],
                    "audio": ["AudioStreamPlayer", "AudioBus", "AudioEffect"],
                    "ui": ["Control", "Button", "Label", "LineEdit", "TextureRect"]
                }
            },
            "best_practices": {
                "code_organization": {
                    "namespaces": "Logical grouping of related classes",
                    "partial_classes": "Splitting large classes across files",
                    "regions": "Code folding for better navigation",
                    "xml_comments": "IntelliSense documentation"
                },
                "performance": {
                    "value_vs_reference": "struct vs class usage",
                    "string_optimization": "StringBuilder for concatenation",
                    "collection_choice": "List vs Array vs Dictionary",
                    "linq_performance": "When to avoid LINQ"
                },
                "patterns": {
                    "solid_principles": "Single Responsibility, Open/Closed, etc.",
                    "dependency_injection": "Loose coupling, testability",
                    "repository_pattern": "Data access abstraction",
                    "mvvm": "Model-View-ViewModel for UI"
                }
            }
        }
    
    def _create_learning_path(self) -> List[Dict]:
        """체계적인 학습 경로 생성"""
        return [
            # 기초
            {"level": 1, "topic": "C# 기본 문법", "subtopics": ["변수와 타입", "연산자", "제어문"], "duration": 2},
            {"level": 1, "topic": "메서드와 함수", "subtopics": ["메서드 정의", "매개변수", "반환값"], "duration": 2},
            {"level": 1, "topic": "클래스 기초", "subtopics": ["클래스 정의", "생성자", "속성"], "duration": 3},
            
            # OOP
            {"level": 2, "topic": "객체지향 프로그래밍", "subtopics": ["상속", "다형성", "캡슐화"], "duration": 4},
            {"level": 2, "topic": "인터페이스와 추상클래스", "subtopics": ["인터페이스 구현", "추상 클래스"], "duration": 3},
            {"level": 2, "topic": "예외 처리", "subtopics": ["try-catch", "사용자 정의 예외"], "duration": 2},
            
            # 고급
            {"level": 3, "topic": "제네릭", "subtopics": ["제네릭 클래스", "제네릭 메서드", "제약 조건"], "duration": 3},
            {"level": 3, "topic": "델리게이트와 이벤트", "subtopics": ["델리게이트", "이벤트", "람다식"], "duration": 4},
            {"level": 3, "topic": "LINQ", "subtopics": ["쿼리 구문", "메서드 구문", "지연 실행"], "duration": 3},
            {"level": 3, "topic": "비동기 프로그래밍", "subtopics": ["async/await", "Task", "병렬 처리"], "duration": 4},
            
            # Godot 특화
            {"level": 4, "topic": "Godot 노드 시스템", "subtopics": ["노드 구조", "생명주기", "씬 관리"], "duration": 3},
            {"level": 4, "topic": "Godot 시그널", "subtopics": ["시그널 정의", "연결", "커스텀 시그널"], "duration": 3},
            {"level": 4, "topic": "Godot 물리", "subtopics": ["KinematicBody", "RigidBody", "충돌 처리"], "duration": 3},
            {"level": 4, "topic": "Godot UI", "subtopics": ["Control 노드", "이벤트 처리", "레이아웃"], "duration": 3},
            
            # 게임 개발 패턴
            {"level": 5, "topic": "게임 디자인 패턴", "subtopics": ["싱글톤", "옵저버", "상태 머신"], "duration": 4},
            {"level": 5, "topic": "최적화 기법", "subtopics": ["오브젝트 풀링", "프로파일링", "메모리 관리"], "duration": 3}
        ]
    
    async def learn_concept(self, topic: str, context: Optional[str] = None) -> Dict:
        """특정 개념 학습"""
        # 지식 베이스에서 관련 정보 찾기
        knowledge = self._find_knowledge(topic)
        
        # 학습 내용 생성
        learning_content = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "knowledge": knowledge,
            "examples": await self._generate_examples(topic, context),
            "exercises": self._create_exercises(topic),
            "best_practices": self._get_best_practices(topic)
        }
        
        # 학습 기록 저장
        await self._save_learning_record(learning_content)
        
        return learning_content
    
    def _find_knowledge(self, topic: str) -> Dict:
        """지식 베이스에서 주제 검색"""
        topic_lower = topic.lower()
        results = {}
        
        def search_dict(d: Dict, path: str = ""):
            for key, value in d.items():
                current_path = f"{path}/{key}" if path else key
                if topic_lower in key.lower():
                    results[current_path] = value
                elif isinstance(value, dict):
                    search_dict(value, current_path)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and topic_lower in item.lower():
                            if current_path not in results:
                                results[current_path] = []
                            results[current_path].append(item)
        
        search_dict(self.knowledge_base)
        return results
    
    async def _generate_examples(self, topic: str, context: Optional[str] = None) -> List[Dict]:
        """주제별 예제 코드 생성"""
        examples = []
        
        # 기본 문법 예제
        if "variable" in topic.lower() or "변수" in topic:
            examples.append({
                "title": "변수 선언과 초기화",
                "code": """
// 기본 타입 변수
int playerHealth = 100;
string playerName = "Hero";
float moveSpeed = 5.5f;
bool isAlive = true;

// var 키워드 사용 (타입 추론)
var score = 0;  // int로 추론
var position = new Vector3(0, 0, 0);  // Vector3로 추론

// const와 readonly
const int MAX_LEVEL = 99;  // 컴파일 시간 상수
readonly string gameVersion = "1.0.0";  // 런타임 상수
"""
            })
        
        # 클래스 예제
        if "class" in topic.lower() or "클래스" in topic:
            examples.append({
                "title": "게임 캐릭터 클래스",
                "code": """
using Godot;

public class GameCharacter : Node2D
{
    // 필드
    private int health;
    private string characterName;
    
    // 속성
    public int Health 
    { 
        get => health;
        set => health = Mathf.Clamp(value, 0, MaxHealth);
    }
    
    public string CharacterName { get; private set; }
    public int MaxHealth { get; private set; } = 100;
    
    // 생성자
    public GameCharacter()
    {
        characterName = "Unknown";
        health = MaxHealth;
    }
    
    // 메서드
    public void TakeDamage(int damage)
    {
        Health -= damage;
        if (Health <= 0)
        {
            Die();
        }
    }
    
    protected virtual void Die()
    {
        GD.Print($"{CharacterName} has died!");
        QueueFree();
    }
}

// 상속 예제
public class Player : GameCharacter
{
    [Export] public int Level { get; private set; } = 1;
    
    public override void _Ready()
    {
        CharacterName = "Player";
        GD.Print($"{CharacterName} ready at level {Level}");
    }
    
    protected override void Die()
    {
        base.Die();
        // 플레이어 특별 죽음 처리
        GetTree().ReloadCurrentScene();
    }
}
"""
            })
        
        # Godot C# 예제
        if "godot" in topic.lower():
            examples.append({
                "title": "Godot 플레이어 컨트롤러",
                "code": """
using Godot;

public class PlayerController : KinematicBody2D
{
    [Export] private float speed = 200.0f;
    [Export] private float jumpSpeed = -400.0f;
    [Export] private float gravity = 1200.0f;
    
    private Vector2 velocity = Vector2.Zero;
    private AnimationPlayer animationPlayer;
    
    public override void _Ready()
    {
        animationPlayer = GetNode<AnimationPlayer>("AnimationPlayer");
        GD.Print("Player controller initialized");
    }
    
    public override void _PhysicsProcess(float delta)
    {
        // 중력 적용
        velocity.y += gravity * delta;
        
        // 입력 처리
        velocity.x = 0;
        
        if (Input.IsActionPressed("move_right"))
        {
            velocity.x += speed;
            animationPlayer.Play("walk");
        }
        else if (Input.IsActionPressed("move_left"))
        {
            velocity.x -= speed;
            animationPlayer.Play("walk");
        }
        else
        {
            animationPlayer.Play("idle");
        }
            
        // 점프
        if (IsOnFloor() && Input.IsActionJustPressed("jump"))
        {
            velocity.y = jumpSpeed;
            animationPlayer.Play("jump");
        }
        
        velocity = MoveAndSlide(velocity, Vector2.Up);
    }
    
    // 시그널 처리
    private void _on_Area2D_body_entered(Node body)
    {
        if (body.IsInGroup("enemies"))
        {
            EmitSignal("player_hit");
            TakeDamage(10);
        }
    }
}
"""
            })
        
        # 비동기 프로그래밍 예제
        if "async" in topic.lower() or "비동기" in topic:
            examples.append({
                "title": "비동기 리소스 로딩",
                "code": """
using System;
using System.Threading.Tasks;
using Godot;

public class ResourceLoader : Node
{
    private Dictionary<string, Resource> resourceCache = new Dictionary<string, Resource>();
    
    public async Task<T> LoadResourceAsync<T>(string path) where T : Resource
    {
        if (resourceCache.ContainsKey(path))
        {
            return (T)resourceCache[path];
        }
        
        try
        {
            // 비동기로 리소스 로드
            var resource = await Task.Run(() => GD.Load<T>(path));
            
            if (resource != null)
            {
                resourceCache[path] = resource;
                return resource;
            }
            
            throw new Exception($"Failed to load resource: {path}");
        }
        catch (Exception e)
        {
            GD.PrintErr($"Resource loading error: {e.Message}");
            throw;
        }
    }
    
    public async Task PreloadResourcesAsync(string[] paths)
    {
        var tasks = new List<Task>();
        
        foreach (var path in paths)
        {
            tasks.Add(LoadResourceAsync<Resource>(path));
        }
        
        await Task.WhenAll(tasks);
        GD.Print($"Preloaded {paths.Length} resources");
    }
}
"""
            })
        
        # LINQ 예제
        if "linq" in topic.lower():
            examples.append({
                "title": "LINQ를 사용한 게임 오브젝트 쿼리",
                "code": """
using System.Linq;
using System.Collections.Generic;
using Godot;

public class GameObjectManager : Node
{
    private List<GameObject> gameObjects = new List<GameObject>();
    
    public class GameObject
    {
        public string Name { get; set; }
        public string Type { get; set; }
        public int Level { get; set; }
        public Vector2 Position { get; set; }
        public bool IsActive { get; set; }
    }
    
    // 타입별 오브젝트 찾기
    public List<GameObject> GetObjectsByType(string type)
    {
        return gameObjects
            .Where(obj => obj.Type == type && obj.IsActive)
            .OrderBy(obj => obj.Level)
            .ToList();
    }
    
    // 범위 내 오브젝트 찾기
    public List<GameObject> GetObjectsInRange(Vector2 center, float radius)
    {
        return gameObjects
            .Where(obj => obj.IsActive)
            .Where(obj => obj.Position.DistanceTo(center) <= radius)
            .OrderBy(obj => obj.Position.DistanceTo(center))
            .ToList();
    }
    
    // 레벨별 그룹화
    public Dictionary<int, List<GameObject>> GroupByLevel()
    {
        return gameObjects
            .Where(obj => obj.IsActive)
            .GroupBy(obj => obj.Level)
            .ToDictionary(g => g.Key, g => g.ToList());
    }
    
    // 복잡한 쿼리 - 가장 가까운 적 찾기
    public GameObject FindNearestEnemy(Vector2 playerPos, int playerLevel)
    {
        return gameObjects
            .Where(obj => obj.Type == "Enemy" && obj.IsActive)
            .Where(obj => Math.Abs(obj.Level - playerLevel) <= 5) // 레벨 차이 5 이내
            .OrderBy(obj => obj.Position.DistanceTo(playerPos))
            .FirstOrDefault();
    }
}
"""
            })
        
        # 시그널 예제
        if "signal" in topic.lower() or "시그널" in topic:
            examples.append({
                "title": "Godot 시그널 시스템",
                "code": """
using Godot;

public class HealthSystem : Node
{
    [Signal]
    public delegate void HealthChanged(int newHealth, int maxHealth);
    
    [Signal]
    public delegate void Died();
    
    [Signal]
    public delegate void DamageTaken(int damage, string damageType);
    
    private int currentHealth;
    private int maxHealth = 100;
    
    public int Health 
    { 
        get => currentHealth;
        private set
        {
            var oldHealth = currentHealth;
            currentHealth = Mathf.Clamp(value, 0, maxHealth);
            
            if (oldHealth != currentHealth)
            {
                EmitSignal(nameof(HealthChanged), currentHealth, maxHealth);
                
                if (currentHealth == 0 && oldHealth > 0)
                {
                    EmitSignal(nameof(Died));
                }
            }
        }
    }
    
    public override void _Ready()
    {
        currentHealth = maxHealth;
        
        // 시그널 연결 예제
        Connect(nameof(Died), this, nameof(OnDeath));
    }
    
    public void TakeDamage(int damage, string damageType = "normal")
    {
        if (damage > 0)
        {
            Health -= damage;
            EmitSignal(nameof(DamageTaken), damage, damageType);
        }
    }
    
    public void Heal(int amount)
    {
        Health += amount;
    }
    
    private void OnDeath()
    {
        GD.Print("Character has died!");
        // 죽음 처리 로직
    }
}

// 시그널 사용 예제
public class HealthBar : Control
{
    private ProgressBar progressBar;
    private Label healthLabel;
    
    public override void _Ready()
    {
        progressBar = GetNode<ProgressBar>("ProgressBar");
        healthLabel = GetNode<Label>("HealthLabel");
        
        // HealthSystem의 시그널에 연결
        var healthSystem = GetNode<HealthSystem>("../HealthSystem");
        healthSystem.Connect("HealthChanged", this, nameof(OnHealthChanged));
        healthSystem.Connect("DamageTaken", this, nameof(OnDamageTaken));
    }
    
    private void OnHealthChanged(int newHealth, int maxHealth)
    {
        progressBar.Value = (float)newHealth / maxHealth * 100;
        healthLabel.Text = $"{newHealth}/{maxHealth}";
    }
    
    private void OnDamageTaken(int damage, string damageType)
    {
        // 데미지 텍스트 표시
        var damageText = new Label();
        damageText.Text = $"-{damage}";
        damageText.AddColorOverride("font_color", Colors.Red);
        AddChild(damageText);
        
        // 애니메이션 후 제거
        GetTree().CreateTimer(1.0f).Connect("timeout", damageText, "queue_free");
    }
}
"""
            })
        
        return examples
    
    def _create_exercises(self, topic: str) -> List[Dict]:
        """연습 문제 생성"""
        exercises = []
        
        if "class" in topic.lower():
            exercises.append({
                "title": "인벤토리 시스템 구현",
                "description": "아이템을 관리하는 인벤토리 클래스를 구현하세요.",
                "requirements": [
                    "아이템 추가/제거 기능",
                    "최대 용량 제한",
                    "아이템 검색 기능",
                    "아이템 정렬 기능"
                ],
                "hints": ["List<T> 사용", "LINQ 활용", "시그널로 변경 알림"]
            })
        
        if "async" in topic.lower():
            exercises.append({
                "title": "비동기 씬 로더",
                "description": "Godot 씬을 비동기로 로드하는 시스템을 만드세요.",
                "requirements": [
                    "여러 씬 동시 로드",
                    "진행률 표시",
                    "취소 기능",
                    "에러 처리"
                ],
                "hints": ["Task.WhenAll", "IProgress<T>", "CancellationToken"]
            })
        
        return exercises
    
    def _get_best_practices(self, topic: str) -> List[str]:
        """주제별 모범 사례"""
        practices = []
        
        # 일반적인 C# 모범 사례
        practices.extend([
            "명확하고 의미 있는 변수명 사용",
            "한 메서드는 하나의 책임만 가지도록 설계",
            "인터페이스를 통한 느슨한 결합",
            "예외는 예외적인 상황에만 사용"
        ])
        
        # Godot 특화 사례
        if "godot" in topic.lower():
            practices.extend([
                "_Ready에서 노드 참조 가져오기",
                "시그널로 노드 간 통신",
                "씬 상속으로 재사용성 향상",
                "Export 어트리뷰트로 인스펙터 노출",
                "Autoload로 전역 시스템 구현"
            ])
        
        return practices
    
    async def _save_learning_record(self, content: Dict):
        """학습 기록 저장"""
        record_path = self.project_root / "csharp_learning" / f"learn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        record_path.parent.mkdir(exist_ok=True)
        
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
    
    async def generate_project_template(self, project_type: str, framework: str = "godot") -> Dict:
        """프로젝트 템플릿 생성"""
        templates = {
            "godot": {
                "fps": self._generate_godot_fps_template(),
                "platformer": self._generate_godot_platformer_template(),
                "puzzle": self._generate_godot_puzzle_template(),
                "rpg": self._generate_godot_rpg_template()
            }
        }
        
        return templates.get(framework, {}).get(project_type, {})
    
    def _generate_godot_fps_template(self) -> Dict:
        """Godot FPS 템플릿"""
        return {
            "name": "Godot FPS Template",
            "structure": {
                "Scripts/Player": ["Player.cs", "WeaponManager.cs", "PlayerCamera.cs"],
                "Scripts/Weapons": ["Weapon.cs", "Projectile.cs", "WeaponPickup.cs"],
                "Scripts/Enemies": ["Enemy.cs", "EnemyAI.cs", "EnemySpawner.cs"],
                "Scripts/Systems": ["GameManager.cs", "AudioManager.cs", "SaveSystem.cs"]
            },
            "core_components": {
                "Player": """
using Godot;

public class Player : KinematicBody
{
    [Export] private float moveSpeed = 5.0f;
    [Export] private float runSpeed = 10.0f;
    [Export] private float jumpSpeed = 8.0f;
    [Export] private float gravity = -20.0f;
    [Export] private float mouseSensitivity = 0.002f;
    
    private Vector3 velocity = Vector3.Zero;
    private Camera camera;
    private bool isRunning = false;
    
    public override void _Ready()
    {
        camera = GetNode<Camera>("CameraMount/Camera");
        Input.SetMouseMode(Input.MouseMode.Captured);
    }
    
    public override void _Input(InputEvent @event)
    {
        if (@event is InputEventMouseMotion mouseMotion)
        {
            RotateY(-mouseMotion.Relative.x * mouseSensitivity);
            camera.RotateX(-mouseMotion.Relative.y * mouseSensitivity);
            
            // 카메라 각도 제한
            var cameraRotation = camera.RotationDegrees;
            cameraRotation.x = Mathf.Clamp(cameraRotation.x, -90, 90);
            camera.RotationDegrees = cameraRotation;
        }
    }
    
    public override void _PhysicsProcess(float delta)
    {
        HandleMovement(delta);
    }
    
    private void HandleMovement(float delta)
    {
        var direction = Vector3.Zero;
        
        if (Input.IsActionPressed("move_forward"))
            direction -= Transform.basis.z;
        if (Input.IsActionPressed("move_backward"))
            direction += Transform.basis.z;
        if (Input.IsActionPressed("move_left"))
            direction -= Transform.basis.x;
        if (Input.IsActionPressed("move_right"))
            direction += Transform.basis.x;
            
        direction = direction.Normalized();
        
        isRunning = Input.IsActionPressed("run");
        var currentSpeed = isRunning ? runSpeed : moveSpeed;
        
        velocity.x = direction.x * currentSpeed;
        velocity.z = direction.z * currentSpeed;
        
        if (!IsOnFloor())
        {
            velocity.y += gravity * delta;
        }
        else if (Input.IsActionJustPressed("jump"))
        {
            velocity.y = jumpSpeed;
        }
        
        velocity = MoveAndSlide(velocity, Vector3.Up);
    }
}
"""
            }
        }
    
    def _generate_godot_platformer_template(self) -> Dict:
        """Godot 플랫포머 템플릿"""
        return {
            "name": "Godot Platformer Template",
            "structure": {
                "Scripts/Player": ["PlatformerPlayer.cs", "PlayerStates.cs"],
                "Scripts/Enemies": ["PatrolEnemy.cs", "FlyingEnemy.cs"],
                "Scripts/Objects": ["Collectible.cs", "MovingPlatform.cs"],
                "Scripts/Systems": ["LevelManager.cs", "CheckpointSystem.cs"]
            }
        }
    
    def _generate_godot_puzzle_template(self) -> Dict:
        """Godot 퍼즐 템플릿"""
        return {
            "name": "Godot Puzzle Template",
            "structure": {
                "Scripts/Core": ["GameBoard.cs", "Piece.cs"],
                "Scripts/Logic": ["MatchFinder.cs", "BoardFiller.cs"],
                "Scripts/Effects": ["ComboSystem.cs", "ParticleManager.cs"]
            }
        }
    
    def _generate_godot_rpg_template(self) -> Dict:
        """Godot RPG 템플릿"""
        return {
            "name": "Godot RPG Template",
            "structure": {
                "Scripts/Player": ["PlayerCharacter.cs", "PlayerStats.cs"],
                "Scripts/Combat": ["BattleSystem.cs", "Skill.cs"],
                "Scripts/World": ["WorldMap.cs", "NPC.cs", "DialogueSystem.cs"],
                "Scripts/Inventory": ["Inventory.cs", "Item.cs", "Equipment.cs"]
            }
        }
    
    async def analyze_code(self, code: str) -> Dict:
        """코드 분석 및 개선 제안"""
        analysis = {
            "issues": [],
            "suggestions": [],
            "performance": [],
            "best_practices": []
        }
        
        # Godot 특화 분석
        
        # GetNode 캐싱 체크
        if "GetNode" in code and "_Ready" not in code:
            analysis["performance"].append({
                "issue": "GetNode 반복 호출",
                "suggestion": "_Ready에서 노드를 캐시하여 성능 향상",
                "severity": "medium"
            })
        
        # 시그널 사용 권장
        if "public void" in code and "On" in code:
            analysis["suggestions"].append({
                "suggestion": "시그널 사용을 고려해보세요",
                "reason": "느슨한 결합과 더 나은 유지보수성"
            })
        
        # Export 어트리뷰트 사용
        if "public float" in code or "public int" in code:
            if "[Export]" not in code:
                analysis["best_practices"].append({
                    "issue": "public 필드에 Export 어트리뷰트 누락",
                    "suggestion": "[Export] 어트리뷰트로 인스펙터에서 조정 가능하게 만들기"
                })
        
        # null 체크
        if "GetNode<" in code and "?" not in code and "if (" not in code:
            analysis["issues"].append({
                "issue": "null 체크 누락 가능성",
                "suggestion": "GetNode 결과에 대한 null 체크 추가"
            })
        
        return analysis
    
    def get_learning_progress(self) -> Dict:
        """학습 진행 상황 반환"""
        progress = {
            "completed_topics": [],
            "current_level": 1,
            "total_hours": 0,
            "exercises_completed": 0,
            "projects_created": 0
        }
        
        # 학습 기록에서 진행 상황 계산
        learning_files = list((self.project_root / "csharp_learning").glob("learn_*.json"))
        
        for file in learning_files:
            with open(file, 'r', encoding='utf-8') as f:
                record = json.load(f)
                progress["completed_topics"].append(record["topic"])
        
        # 레벨 계산
        topics_per_level = 5
        progress["current_level"] = min(5, len(progress["completed_topics"]) // topics_per_level + 1)
        
        return progress
    
    def get_example(self, topic: str) -> Optional[str]:
        """주제에 대한 심층 예제 코드 반환"""
        # Godot 특화 C# 예제 코드
        examples = {
            "async programming": """
// Godot C# Async Programming - 심층 예제
using Godot;
using System.Threading.Tasks;

public partial class ResourceLoader : Node
{
    // 비동기 리소스 로딩
    public async Task<Texture2D> LoadTextureAsync(string path)
    {
        // 메인 스레드에서 비동기 작업 실행
        return await Task.Run(() => 
        {
            return GD.Load<Texture2D>(path);
        });
    }
    
    // 여러 리소스 동시 로딩
    public async Task LoadAllResourcesAsync()
    {
        var tasks = new[]
        {
            LoadTextureAsync("res://player.png"),
            LoadTextureAsync("res://enemy.png"),
            LoadTextureAsync("res://background.png")
        };
        
        var textures = await Task.WhenAll(tasks);
        
        // 로드된 텍스처 적용
        GetNode<Sprite2D>("Player").Texture = textures[0];
        GetNode<Sprite2D>("Enemy").Texture = textures[1];
        GetNode<Sprite2D>("Background").Texture = textures[2];
    }
    
    // Godot 시그널과 async/await 결합
    public async Task WaitForSignalAsync()
    {
        // 시그널을 기다리는 비동기 작업
        await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
        
        // 타이머 시그널 대기
        var timer = GetNode<Timer>("Timer");
        await ToSignal(timer, Timer.SignalName.Timeout);
    }
}
""",
            "linq": """
// LINQ in Godot C# - 심층 예제
using Godot;
using System.Linq;
using System.Collections.Generic;

public partial class GameManager : Node
{
    // 노드 컬렉션 LINQ 처리
    public void ProcessEnemies()
    {
        // 모든 적 노드 가져오기
        var enemies = GetTree().GetNodesInGroup("enemies")
            .Cast<Enemy>()
            .Where(e => e.Health > 0 && e.IsInsideViewport())
            .OrderByDescending(e => e.ThreatLevel)
            .ThenBy(e => e.GlobalPosition.DistanceTo(PlayerPosition))
            .Take(10);
            
        foreach (var enemy in enemies)
        {
            enemy.UpdateAI();
        }
    }
    
    // LINQ로 게임 오브젝트 검색
    public T FindClosestObject<T>(Vector2 position, string group) where T : Node2D
    {
        return GetTree().GetNodesInGroup(group)
            .OfType<T>()
            .Where(obj => obj.Visible)
            .OrderBy(obj => obj.GlobalPosition.DistanceTo(position))
            .FirstOrDefault();
    }
    
    // LINQ로 인벤토리 관리
    public class Inventory
    {
        private List<Item> items = new();
        
        public IEnumerable<Item> GetItemsByType(ItemType type)
        {
            return items
                .Where(item => item.Type == type && !item.IsUsed)
                .OrderByDescending(item => item.Rarity)
                .ThenBy(item => item.Name);
        }
        
        public int GetTotalValue()
        {
            return items.Sum(item => item.Value * item.Quantity);
        }
        
        public Dictionary<ItemType, int> GetItemCountByType()
        {
            return items
                .GroupBy(item => item.Type)
                .ToDictionary(
                    group => group.Key,
                    group => group.Sum(item => item.Quantity)
                );
        }
    }
}
""",
            "delegates": """
// Delegates and Events in Godot C# - 심층 예제
using Godot;
using System;

public partial class Player : CharacterBody2D
{
    // 델리게이트 정의
    public delegate void HealthChangedEventHandler(int currentHealth, int maxHealth);
    public delegate void StateChangedEventHandler(PlayerState oldState, PlayerState newState);
    
    // C# 이벤트
    public event HealthChangedEventHandler HealthChanged;
    public event StateChangedEventHandler StateChanged;
    public event Action<Item> ItemCollected;
    public event Func<float, float> DamageModifier;
    
    // Godot 시그널
    [Signal]
    public delegate void DiedEventHandler();
    
    [Signal]
    public delegate void LevelUpEventHandler(int newLevel);
    
    private int health = 100;
    private PlayerState currentState = PlayerState.Idle;
    
    public void TakeDamage(float damage)
    {
        // 델이미지 모디파이어 체인
        if (DamageModifier != null)
        {
            foreach (Func<float, float> modifier in DamageModifier.GetInvocationList())
            {
                damage = modifier(damage);
            }
        }
        
        health -= (int)damage;
        health = Mathf.Max(0, health);
        
        // 이벤트 발생
        HealthChanged?.Invoke(health, 100);
        
        if (health <= 0)
        {
            EmitSignal(SignalName.Died);
            ChangeState(PlayerState.Dead);
        }
    }
    
    private void ChangeState(PlayerState newState)
    {
        if (currentState != newState)
        {
            var oldState = currentState;
            currentState = newState;
            StateChanged?.Invoke(oldState, newState);
        }
    }
    
    // 람다식과 Action/Func 사용
    public void PerformAction(Action<Player> action)
    {
        action?.Invoke(this);
    }
    
    public T Calculate<T>(Func<Player, T> calculation)
    {
        return calculation != null ? calculation(this) : default(T);
    }
}

public enum PlayerState
{
    Idle, Moving, Jumping, Attacking, Dead
}
"""
        }
        
        # 주제 찾기 - 더 세분화된 매칭
        topic_lower = topic.lower()
        for key in examples:
            if key in topic_lower or topic_lower in key:
                return examples[key]
        
        # 기본 예제 생성
        return self._generate_basic_example(topic)
    
    async def learn_topic(self, topic: str) -> Optional[str]:
        """비동기로 주제 학습 - 심층 학습"""
        # 학습 컨텍스트 분석
        context = await self._analyze_learning_context(topic)
        
        # 학습 내용 생성
        content = f"# {topic} 학습 내용\n\n"
        content += f"## 학습 목표\n{context['objectives']}\n\n"
        
        # 개념 설명
        concept_info = self._get_concept_explanation(topic)
        if concept_info:
            content += f"## 핵심 개념\n{concept_info}\n\n"
        
        # Godot에서의 활용
        godot_usage = self._get_godot_usage(topic)
        if godot_usage:
            content += f"## Godot에서의 활용\n{godot_usage}\n\n"
        
        # 예제 코드
        example = self.get_example(topic)
        if example:
            content += f"## 실습 예제\n```csharp\n{example}\n```\n\n"
        
        # 베스트 프랙티스
        best_practices = self._get_best_practices_for_topic(topic)
        if best_practices:
            content += f"## 베스트 프랙티스\n{best_practices}\n\n"
        
        # 실습 과제
        exercises = self._create_exercises(topic)
        if exercises:
            content += f"## 실습 과제\n{exercises}\n\n"
        
        # 학습 기록 저장
        await self._save_learning_progress(topic, content)
        
        # 학습 시뮬레이션
        await asyncio.sleep(1.0)
        
        return content
    
    async def _analyze_learning_context(self, topic: str) -> Dict:
        """학습 컨텍스트 분석"""
        # 학습 목표 설정
        objectives = {
            "async programming": "Godot에서 비동기 프로그래밍을 활용하여 게임 성능 향상",
            "linq": "LINQ를 사용하여 게임 데이터를 효율적으로 처리",
            "delegates": "델리게이트와 이벤트로 유연한 게임 시스템 구축"
        }
        
        return {
            "objectives": objectives.get(topic.lower(), f"{topic}의 핵심 개념을 이해하고 Godot에 적용")
        }
    
    def _get_concept_explanation(self, topic: str) -> str:
        """개념 설명 가져오기"""
        explanations = {
            "async programming": """
- **비동기 프로그래밍**: 작업을 동시에 수행하여 성능을 향상시키는 기법
- **async/await**: C#의 비동기 프로그래밍 키워드
- **Task**: 비동기 작업을 나타내는 객체
- **Godot에서의 활용**: 리소스 로딩, 네트워크 통신, 시그널 대기
""",
            "linq": """
- **LINQ(Language Integrated Query)**: C#에 통합된 쿼리 기능
- **쿼리 구문**: SQL과 유사한 구문 (from, where, select)
- **메서드 구문**: 체인 형태의 메서드 호출 (Where, Select, OrderBy)
- **Godot에서의 활용**: 노드 검색, 인벤토리 관리, 적 AI 처리
""",
            "delegates": """
- **델리게이트**: 메서드를 참조하는 타입
- **이벤트**: 델리게이트를 기반으로 한 알림 메커니즘
- **Action/Func**: 미리 정의된 제네릭 델리게이트
- **Godot에서의 활용**: C# 이벤트와 Godot 시그널의 조합
"""
        }
        
        return explanations.get(topic.lower(), "")
    
    def _get_godot_usage(self, topic: str) -> str:
        """게임 개발에서의 실제 활용 예시"""
        usage_examples = {
            "async programming": """
1. **레벨 로딩**: 비동기로 씨스템 파일 로드
2. **세이브 시스템**: 비동기로 게임 저장/불러오기
3. **네트워크 통신**: HTTP 요청, 멀티플레이어 데이터 송수신
4. **AI 처리**: 복잡한 AI 계산을 비동기로 처리
""",
            "linq": """
1. **적 관리**: 특정 조건의 적만 필터링
2. **아이템 정렬**: 인벤토리 아이템을 다양한 기준으로 정렬
3. **퀴스트 시스템**: 완료/진행 중인 퀴스트 필터링
4. **레벨 디자인**: 게임 오브젝트를 타입별로 그룹화
""",
            "delegates": """
1. **이벤트 시스템**: 체력 변화, 레벨업, 아이템 획득 알림
2. **커맨드 패턴**: 사용자 입력을 커맨드 객체로 처리
3. **콜백 시스템**: 애니메이션 종료, 타이머 완료 콜백
4. **UI 핸들러**: 버튼 클릭, 슬라이더 변경 이벤트
"""
        }
        
        return usage_examples.get(topic.lower(), "")
    
    def _get_best_practices_for_topic(self, topic: str) -> str:
        """주제별 베스트 프랙티스"""
        practices = {
            "async programming": """
- ConfigureAwait(false) 사용하여 컨텍스트 전환 오버헤드 감소
- CPU 집약적 작업은 Task.Run으로 분리
- CancellationToken으로 작업 취소 처리
- 비동기 메서드는 Async 접미사 사용
""",
            "linq": """
- 성능이 중요한 경우 LINQ 대신 for 루프 사용 고려
- ToList()나 ToArray()로 즉시 실행 강제
- 복잡한 쿼리는 여러 줄로 나누어 가독성 향상
- Any() 사용하여 존재 여부만 확인
""",
            "delegates": """
- 널 체크: delegate?.Invoke() 형태 사용
- 이벤트는 private 필드와 public 프로퍼티로 캉슐화
- 메모리 누수 방지를 위해 이벤트 구독 해제 확인
- Action/Func 사용하여 코드 간소화
"""
        }
        
        return practices.get(topic.lower(), "")
    
    def _create_exercises(self, topic: str) -> str:
        """실습 과제 생성"""
        exercises = {
            "async programming": """
1. 여러 텍스처를 비동기로 로드하고 프로그레스 바 표시하기
2. 세이브/로드 시스템을 비동기로 구현하기
3. HTTP API 호출을 비동기로 처리하는 리더보드 시스템 만들기
""",
            "linq": """
1. 적 그룹에서 가장 강력한 5명의 적을 찾아 처리하기
2. 인벤토리 아이템을 타입별로 그룹화하고 통계 표시하기
3. 퀴스트 시스템에서 조건에 맞는 퀴스트만 필터링하기
""",
            "delegates": """
1. 체력 변화, 레벨업, 아이템 획듍 이벤트 시스템 구현하기
2. 커맨드 패턴을 사용한 입력 처리 시스템 만들기
3. 델이미지 계산에 여러 모디파이어를 적용하는 시스템 구현하기
"""
        }
        
        return exercises.get(topic.lower(), f"1. {topic}의 기본 개념을 활용한 간단한 예제 만들기\n2. Godot 프로젝트에 적용하기")
    
    async def _save_learning_progress(self, topic: str, content: str):
        """학습 진행 상황 저장"""
        # 학습 기록 저장
        record = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "content_length": len(content),
            "completed": True
        }
        
        # 파일로 저장 (실제 구현에서)
        # self.save_to_file(record)
        
        await asyncio.sleep(0.1)  # 저장 시뮬레이션
    
    def _generate_basic_example(self, topic: str) -> str:
        """기본 예제 코드 생성"""
        return f"""
// {topic} - Godot C# 예제
using Godot;
using System;

public partial class {topic.replace(' ', '')}Example : Node
{{
    public override void _Ready()
    {{
        GD.Print($"Learning {{nameof({topic.replace(' ', '')}Example)}}");
        
        // TODO: {topic} 구현
        // 1. 기본 개념 이해
        // 2. Godot에 적용
        // 3. 실제 게임 기능 구현
    }}
    
    public override void _Process(double delta)
    {{
        // 프레임마다 업데이트
    }}
}}
"""
    
    async def analyze_and_improve_code(self, code: str, context: Dict) -> Dict:
        """AI가 C# 코드를 분석하고 개선점 제안"""
        improvements = []
        
        # 비동기 패턴 분석
        if "async" in code and "await" not in code:
            improvements.append({
                "type": "async_usage",
                "severity": "warning",
                "message": "async 메서드에서 await를 사용하지 않음",
                "suggestion": "비동기 작업이 없다면 async 키워드를 제거하거나, 실제 비동기 작업 추가"
            })
        
        # Godot 시그널 패턴 확인
        if "[Signal]" in code and "EmitSignal" not in code:
            improvements.append({
                "type": "signal_usage",
                "severity": "info",
                "message": "선언된 시그널이 사용되지 않음",
                "suggestion": "시그널을 발생시키는 코드 추가 필요"
            })
        
        # 메모리 누수 가능성 확인
        if "new Timer()" in code and "QueueFree" not in code:
            improvements.append({
                "type": "memory_leak",
                "severity": "error",
                "message": "동적으로 생성된 노드가 해제되지 않을 수 있음",
                "suggestion": "QueueFree() 호출 또는 using 패턴 사용"
            })
        
        # LINQ 최적화 기회
        if "foreach" in code and ".Where(" not in code:
            improvements.append({
                "type": "optimization",
                "severity": "info",
                "message": "LINQ를 사용한 더 간결한 코드 작성 가능",
                "suggestion": "Where, Select, Any 등의 LINQ 메서드 활용 고려"
            })
        
        # Godot 특화 패턴
        if "GetNode<" in code:
            if "$" not in code:
                improvements.append({
                    "type": "godot_pattern",
                    "severity": "info",
                    "message": "노드 경로에 $ 단축 문법 사용 가능",
                    "suggestion": 'GetNode<Label>("UI/Label") → GetNode<Label>("$UI/Label")'
                })
        
        # null 체크
        if "GetNode" in code and "?" not in code and "null" not in code:
            improvements.append({
                "type": "null_safety",
                "severity": "warning",
                "message": "GetNode 결과에 대한 null 체크 누락",
                "suggestion": "?.를 사용하거나 null 체크 추가"
            })
        
        # 성능 분석
        performance_score = 100
        if "_Process(" in code:
            if "GetNode" in code:
                performance_score -= 10
                improvements.append({
                    "type": "performance",
                    "severity": "warning",
                    "message": "_Process에서 매 프레임 GetNode 호출",
                    "suggestion": "_Ready에서 노드 참조를 캐시하여 성능 향상"
                })
        
        # 개선된 코드 생성
        improved_code = await self._generate_improved_code(code, improvements)
        
        return {
            "original_code": code,
            "improved_code": improved_code,
            "improvements": improvements,
            "performance_score": performance_score,
            "readability_score": self._calculate_readability_score(code),
            "godot_integration_score": self._calculate_godot_integration_score(code)
        }
    
    async def _generate_improved_code(self, code: str, improvements: List[Dict]) -> str:
        """개선사항을 적용한 코드 생성"""
        improved = code
        
        # 각 개선사항 적용
        for improvement in improvements:
            if improvement["type"] == "async_usage":
                # async/await 패턴 수정
                improved = improved.replace("async void", "void")
            
            elif improvement["type"] == "null_safety":
                # null 체크 추가
                improved = improved.replace("GetNode<", "GetNode<")
                improved = improved.replace(">().", ">()?.") 
            
            elif improvement["type"] == "performance" and "_Process" in improved:
                # 노드 캐싱 추가
                lines = improved.split('\n')
                for i, line in enumerate(lines):
                    if "GetNode<" in line and "_Process" in '\n'.join(lines[max(0, i-10):i+10]):
                        # 클래스 필드로 이동
                        node_type = line.split('GetNode<')[1].split('>')[0]
                        node_path = line.split('"')[1]
                        field_name = node_path.split('/')[-1].lower()
                        
                        # 필드 선언 추가
                        class_start = next(j for j, l in enumerate(lines) if "class" in l and "{" in l)
                        lines.insert(class_start + 1, f"    private {node_type} {field_name};")
                        
                        # _Ready에서 초기화
                        ready_idx = next((j for j, l in enumerate(lines) if "_Ready()" in l), -1)
                        if ready_idx > 0:
                            lines.insert(ready_idx + 2, f'        {field_name} = GetNode<{node_type}>("{node_path}");')
                        
                        # _Process에서 캐시된 변수 사용
                        lines[i] = line.replace(f'GetNode<{node_type}>("{node_path}")', field_name)
                
                improved = '\n'.join(lines)
        
        return improved
    
    def _calculate_readability_score(self, code: str) -> int:
        """코드 가독성 점수 계산"""
        score = 100
        
        # 줄 길이 체크
        lines = code.split('\n')
        long_lines = sum(1 for line in lines if len(line) > 120)
        score -= long_lines * 2
        
        # 중첩 깊이 체크
        max_nesting = 0
        current_nesting = 0
        for line in lines:
            current_nesting += line.count('{') - line.count('}')
            max_nesting = max(max_nesting, current_nesting)
        
        if max_nesting > 4:
            score -= (max_nesting - 4) * 5
        
        # 주석 비율
        comment_lines = sum(1 for line in lines if '//' in line or '/*' in line)
        if len(lines) > 0:
            comment_ratio = comment_lines / len(lines)
            if comment_ratio < 0.1:
                score -= 10
        
        return max(0, score)
    
    def _calculate_godot_integration_score(self, code: str) -> int:
        """Godot 통합 점수 계산"""
        score = 0
        
        # Godot 클래스 사용
        godot_classes = ["Node", "Node2D", "Node3D", "CharacterBody2D", "RigidBody2D", 
                        "Area2D", "Control", "Sprite2D", "AnimationPlayer"]
        for cls in godot_classes:
            if cls in code:
                score += 10
        
        # Godot 메서드 사용
        godot_methods = ["_Ready", "_Process", "_PhysicsProcess", "_Input", 
                        "MoveAndSlide", "GetNode", "AddChild", "QueueFree"]
        for method in godot_methods:
            if method in code:
                score += 5
        
        # 시그널 사용
        if "[Signal]" in code:
            score += 15
        if "EmitSignal" in code:
            score += 10
        
        # Export 속성 사용
        if "[Export]" in code:
            score += 10
        
        return min(100, score)


# 싱글톤 인스턴스
_csharp_agent = None

def get_csharp_learning_agent() -> CSharpLearningAgent:
    """C# 학습 에이전트 싱글톤 인스턴스 반환"""
    global _csharp_agent
    if _csharp_agent is None:
        _csharp_agent = CSharpLearningAgent()
    return _csharp_agent