"""
C# 통합 학습 모듈
가상 입력과 Godot 조작을 통한 C# 패턴 학습 및 적용
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class CSharpIntegratedLearning:
    """C# 통합 학습 시스템"""
    
    def __init__(self):
        self.patterns_dir = Path("experiences/csharp_patterns")
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        
        # C# 패턴 카테고리
        self.pattern_categories = {
            "godot_basics": "Godot 기본 패턴",
            "player_control": "플레이어 제어",
            "physics": "물리 시스템",
            "ui_system": "UI 시스템",
            "networking": "네트워킹",
            "optimization": "최적화"
        }
        
        # 학습된 패턴
        self.learned_patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """저장된 C# 패턴 로드"""
        patterns = {category: [] for category in self.pattern_categories}
        
        for pattern_file in self.patterns_dir.glob("*.json"):
            try:
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    category = data.get("category", "godot_basics")
                    if category in patterns:
                        patterns[category].append(data)
            except Exception as e:
                logger.error(f"패턴 로드 오류 {pattern_file}: {e}")
        
        return patterns
    
    async def learn_from_godot_action(self, action: str, code: str, success: bool):
        """Godot 조작에서 C# 패턴 학습"""
        pattern = {
            "action": action,
            "code": code,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "category": self._categorize_code(code),
            "quality_score": 0.0
        }
        
        # 품질 평가
        pattern["quality_score"] = self._evaluate_code_quality(code)
        
        # 성공한 고품질 패턴만 저장
        if success and pattern["quality_score"] > 0.7:
            filename = f"{pattern['category']}_{action}_{int(datetime.now().timestamp())}.json"
            filepath = self.patterns_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(pattern, f, indent=2, ensure_ascii=False)
            
            logger.info(f"C# 패턴 학습 완료: {action} (품질: {pattern['quality_score']:.2f})")
    
    def _categorize_code(self, code: str) -> str:
        """코드 카테고리 분류"""
        code_lower = code.lower()
        
        if "characterbody2d" in code_lower or "input" in code_lower:
            return "player_control"
        elif "rigidbody" in code_lower or "collision" in code_lower:
            return "physics"
        elif "control" in code_lower or "button" in code_lower:
            return "ui_system"
        elif "rpc" in code_lower or "multiplayer" in code_lower:
            return "networking"
        elif "pool" in code_lower or "cache" in code_lower:
            return "optimization"
        else:
            return "godot_basics"
    
    def _evaluate_code_quality(self, code: str) -> float:
        """코드 품질 평가"""
        score = 1.0
        
        # 기본 체크
        if not code.strip():
            return 0.0
        
        # Godot C# 규칙 체크
        checks = {
            "using Godot;": 0.2,
            "public partial class": 0.2,
            "_Ready()": 0.1,
            "_Process(": 0.1,
            "override": 0.1,
            "GD.Print": 0.05,
            "[Export]": 0.1,
            "GetNode": 0.1,
            "signal": 0.05
        }
        
        for pattern, weight in checks.items():
            if pattern in code:
                score = min(1.0, score + weight)
        
        # 코드 길이 페널티 (너무 짧거나 너무 길면 감점)
        lines = code.split('\n')
        if len(lines) < 5:
            score *= 0.7
        elif len(lines) > 200:
            score *= 0.8
        
        return min(1.0, max(0.0, score))
    
    def generate_code_from_pattern(self, action_type: str) -> Optional[str]:
        """학습된 패턴에서 코드 생성"""
        # 해당 액션에 맞는 카테고리 찾기
        category = self._get_category_for_action(action_type)
        
        if category in self.learned_patterns:
            patterns = self.learned_patterns[category]
            
            # 품질 점수가 높은 패턴 우선
            patterns.sort(key=lambda p: p.get("quality_score", 0), reverse=True)
            
            if patterns:
                best_pattern = patterns[0]
                logger.info(f"패턴 기반 코드 생성: {action_type} (품질: {best_pattern['quality_score']:.2f})")
                return self._adapt_code(best_pattern["code"], action_type)
        
        # 기본 템플릿 제공
        return self._get_default_template(action_type)
    
    def _get_category_for_action(self, action_type: str) -> str:
        """액션 타입에 맞는 카테고리 반환"""
        action_mapping = {
            "player_movement": "player_control",
            "enemy_ai": "player_control",
            "collision": "physics",
            "ui_button": "ui_system",
            "multiplayer": "networking"
        }
        
        for key, category in action_mapping.items():
            if key in action_type.lower():
                return category
        
        return "godot_basics"
    
    def _adapt_code(self, template_code: str, action_type: str) -> str:
        """템플릿 코드를 액션에 맞게 수정"""
        # 기본적인 치환만 수행
        adapted = template_code
        
        # 클래스명 변경
        if "Player" in adapted and "enemy" in action_type.lower():
            adapted = adapted.replace("Player", "Enemy")
        
        # 주석 추가
        header = f"// AutoCI Generated: {action_type}\n// Based on learned pattern\n\n"
        
        return header + adapted
    
    def _get_default_template(self, action_type: str) -> str:
        """기본 C# 템플릿 제공"""
        templates = {
            "player_movement": """using Godot;

public partial class Player : CharacterBody2D
{
    [Export] private float Speed = 300.0f;
    [Export] private float JumpVelocity = -400.0f;
    
    private float gravity = ProjectSettings.GetSetting("physics/2d/default_gravity").AsSingle();
    
    public override void _PhysicsProcess(double delta)
    {
        Vector2 velocity = Velocity;
        
        // Add gravity
        if (!IsOnFloor())
            velocity.Y += gravity * (float)delta;
        
        // Handle Jump
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
            velocity.X = Mathf.MoveToward(Velocity.X, 0, Speed * (float)delta);
        }
        
        Velocity = velocity;
        MoveAndSlide();
    }
}""",
            "ui_system": """using Godot;

public partial class UIManager : Control
{
    private Label scoreLabel;
    private Button startButton;
    
    public override void _Ready()
    {
        scoreLabel = GetNode<Label>("ScoreLabel");
        startButton = GetNode<Button>("StartButton");
        
        startButton.Pressed += OnStartButtonPressed;
    }
    
    private void OnStartButtonPressed()
    {
        GD.Print("Game Started!");
        // Add game start logic
    }
    
    public void UpdateScore(int score)
    {
        scoreLabel.Text = $"Score: {score}";
    }
}""",
            "default": """using Godot;

public partial class GameNode : Node2D
{
    public override void _Ready()
    {
        GD.Print("Node Ready!");
    }
    
    public override void _Process(double delta)
    {
        // Add frame logic here
    }
}"""
        }
        
        # 가장 적합한 템플릿 찾기
        for key, template in templates.items():
            if key in action_type.lower():
                return template
        
        return templates["default"]
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        stats = {
            "total_patterns": 0,
            "categories": {},
            "average_quality": 0.0,
            "recent_patterns": []
        }
        
        total_quality = 0.0
        
        for category, patterns in self.learned_patterns.items():
            stats["categories"][category] = len(patterns)
            stats["total_patterns"] += len(patterns)
            
            for pattern in patterns:
                total_quality += pattern.get("quality_score", 0)
        
        if stats["total_patterns"] > 0:
            stats["average_quality"] = total_quality / stats["total_patterns"]
        
        # 최근 패턴 5개
        all_patterns = []
        for patterns in self.learned_patterns.values():
            all_patterns.extend(patterns)
        
        all_patterns.sort(key=lambda p: p.get("timestamp", ""), reverse=True)
        stats["recent_patterns"] = all_patterns[:5]
        
        return stats
    
    async def apply_learned_patterns(self, target_file: Path) -> bool:
        """학습된 패턴을 실제 파일에 적용"""
        try:
            # 파일 읽기
            if not target_file.exists():
                logger.error(f"대상 파일이 없습니다: {target_file}")
                return False
            
            with open(target_file, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            # 코드 개선
            improved_code = self._improve_code_with_patterns(original_code)
            
            # 변경사항이 있으면 저장
            if improved_code != original_code:
                # 백업 생성
                backup_file = target_file.with_suffix('.bak')
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(original_code)
                
                # 개선된 코드 저장
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(improved_code)
                
                logger.info(f"패턴 적용 완료: {target_file}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"패턴 적용 오류: {e}")
            return False
    
    def _improve_code_with_patterns(self, code: str) -> str:
        """학습된 패턴으로 코드 개선"""
        improved = code
        
        # 간단한 개선 규칙들
        improvements = {
            # Godot 4.x 패턴
            "extends ": "public partial class ",
            "func _ready():": "public override void _Ready()",
            "func _process(delta):": "public override void _Process(double delta)",
            "var ": "",  # C#은 타입을 명시해야 함
            "print(": "GD.Print(",
            
            # 성능 개선
            "GetNode(": "GetNode<Node>(",  # 제네릭 사용
            "position.x": "Position.X",
            "position.y": "Position.Y"
        }
        
        for old_pattern, new_pattern in improvements.items():
            if old_pattern in improved:
                improved = improved.replace(old_pattern, new_pattern)
        
        return improved


# 통합 학습 실행
async def run_integrated_learning():
    """C# 통합 학습 실행"""
    learning = CSharpIntegratedLearning()
    
    print("🎓 C# 통합 학습 시작...")
    
    # 샘플 학습
    sample_actions = [
        ("create_player", """using Godot;

public partial class Player : CharacterBody2D
{
    [Export] private float Speed = 300.0f;
    
    public override void _Ready()
    {
        GD.Print("Player initialized");
    }
}""", True),
        ("add_enemy_ai", """using Godot;

public partial class Enemy : CharacterBody2D
{
    private Node2D target;
    
    public override void _Process(double delta)
    {
        if (target != null)
        {
            Vector2 direction = (target.GlobalPosition - GlobalPosition).Normalized();
            Velocity = direction * 200.0f;
            MoveAndSlide();
        }
    }
}""", True)
    ]
    
    for action, code, success in sample_actions:
        await learning.learn_from_godot_action(action, code, success)
    
    # 통계 출력
    stats = learning.get_learning_statistics()
    print(f"\n📊 학습 통계:")
    print(f"  - 총 패턴: {stats['total_patterns']}개")
    print(f"  - 평균 품질: {stats['average_quality']:.2f}")
    print(f"  - 카테고리별: {stats['categories']}")
    
    # 코드 생성 테스트
    generated = learning.generate_code_from_pattern("player_movement")
    if generated:
        print(f"\n🔧 생성된 코드:")
        print(generated[:200] + "..." if len(generated) > 200 else generated)


if __name__ == "__main__":
    asyncio.run(run_integrated_learning())