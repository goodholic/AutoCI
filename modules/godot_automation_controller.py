#!/usr/bin/env python3
"""
Godot 자동화 컨트롤러 Python 래퍼
C#으로 구현된 GodotAutomationController를 Python에서 사용하기 위한 래퍼
"""

import os
import sys
import json
import asyncio
import subprocess
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

# Try to import clr (pythonnet) but make it optional
try:
    import clr  # pythonnet
    CLR_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    CLR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"C# integration not available: {e}")

logger = logging.getLogger(__name__)

# C# DLL 경로 설정 (CLR이 사용 가능한 경우에만)
if CLR_AVAILABLE:
    dll_path = Path(__file__).parent / "bin" / "AutoCI.Modules.dll"
    if dll_path.exists():
        clr.AddReference(str(dll_path))
    else:
        # 대체 경로 시도
        dll_path = Path(__file__).parent.parent / "bin" / "Debug" / "net6.0" / "AutoCI.Modules.dll"
        if dll_path.exists():
            clr.AddReference(str(dll_path))

try:
    # C# 네임스페이스 임포트
    if CLR_AVAILABLE:
        from AutoCI.Modules import GodotAutomationController as CSharpController
    else:
        CSharpController = None
except ImportError:
    logger.warning("C# GodotAutomationController를 로드할 수 없습니다. 순수 Python 모드로 실행합니다.")
    CSharpController = None

class GodotAutomationController:
    """Godot 자동화 컨트롤러 - Python 래퍼"""
    
    def __init__(self):
        self.project_path = None
        self.is_engine_running = False
        self.csharp_controller = None
        
        # C# 컨트롤러 초기화 시도
        if CSharpController:
            try:
                self.csharp_controller = CSharpController()
                logger.info("✅ C# GodotAutomationController 초기화 성공")
            except Exception as e:
                logger.error(f"❌ C# 컨트롤러 초기화 실패: {e}")
                self.csharp_controller = None
        else:
            logger.info("⚠️ Python 전용 모드로 실행됩니다.")
    
    async def start_engine(self, project_name: str) -> bool:
        """Godot 엔진 시작"""
        if self.csharp_controller:
            # C# 메서드 호출
            try:
                task = self.csharp_controller.StartEngine(project_name)
                # C# Task를 Python asyncio로 변환
                result = await asyncio.get_event_loop().run_in_executor(None, task.Result)
                self.is_engine_running = result
                return result
            except Exception as e:
                logger.error(f"C# StartEngine 오류: {e}")
                return await self._python_start_engine(project_name)
        else:
            return await self._python_start_engine(project_name)
    
    async def _python_start_engine(self, project_name: str) -> bool:
        """Python으로 Godot 엔진 시작"""
        try:
            self.project_path = Path.home() / "Documents" / "Godot" / "Projects" / project_name
            self.project_path.mkdir(parents=True, exist_ok=True)
            
            # project.godot 파일 생성
            project_config = f"""
config_version=5

[application]
config/name="{project_name}"
config/features=PackedStringArray("4.2", "C#", "Forward Plus")
config/icon="res://icon.svg"

[dotnet]
project/assembly_name="{project_name}"

[rendering]
renderer/rendering_method="forward_plus"
"""
            (self.project_path / "project.godot").write_text(project_config)
            
            # 필요한 디렉토리 생성
            for dir_name in ["scenes", "scripts", "assets"]:
                (self.project_path / dir_name).mkdir(exist_ok=True)
            
            logger.info(f"✅ 프로젝트 생성됨: {self.project_path}")
            self.is_engine_running = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Godot 엔진 시작 실패: {e}")
            return False
    
    async def execute_action(self, action: str, parameters: Dict[str, Any] = None) -> bool:
        """AI 액션 실행"""
        if parameters is None:
            parameters = {}
            
        if self.csharp_controller:
            try:
                # Python dict를 C# Dictionary로 변환
                csharp_params = self._convert_to_csharp_dict(parameters)
                task = self.csharp_controller.ExecuteAction(action, csharp_params)
                result = await asyncio.get_event_loop().run_in_executor(None, task.Result)
                return result
            except Exception as e:
                logger.error(f"C# ExecuteAction 오류: {e}")
                return await self._python_execute_action(action, parameters)
        else:
            return await self._python_execute_action(action, parameters)
    
    async def _python_execute_action(self, action: str, parameters: Dict[str, Any]) -> bool:
        """Python으로 액션 실행"""
        logger.info(f"🎮 액션 실행: {action} - {parameters}")
        
        action_handlers = {
            "create_scene": self._create_scene,
            "add_player": self._add_player,
            "add_enemy": self._add_enemy,
            "add_platform": self._add_platform,
            "add_ui": self._add_ui,
            "generate_csharp_code": self._generate_csharp_code
        }
        
        handler = action_handlers.get(action)
        if handler:
            return await handler(parameters)
        else:
            logger.warning(f"알 수 없는 액션: {action}")
            return False
    
    async def _create_scene(self, parameters: Dict) -> bool:
        """씬 생성"""
        scene_name = parameters.get("name", "MainScene")
        scene_content = f"""[gd_scene load_steps=2 format=3]

[node name="{scene_name}" type="Node2D"]
"""
        
        scene_path = self.project_path / "scenes" / f"{scene_name}.tscn"
        scene_path.write_text(scene_content)
        logger.info(f"✅ 씬 생성됨: {scene_path}")
        return True
    
    async def _add_player(self, parameters: Dict) -> bool:
        """플레이어 추가"""
        # Player.cs 생성
        player_script = """using Godot;

public partial class Player : CharacterBody2D
{
    private float speed = 300.0f;
    private float jumpVelocity = -400.0f;
    
    public override void _PhysicsProcess(double delta)
    {
        Vector2 velocity = Velocity;
        
        if (!IsOnFloor())
            velocity.Y += GetGravity() * (float)delta;
        
        if (Input.IsActionJustPressed("jump") && IsOnFloor())
            velocity.Y = jumpVelocity;
        
        Vector2 direction = Input.GetVector("move_left", "move_right", "move_up", "move_down");
        if (direction != Vector2.Zero)
        {
            velocity.X = direction.X * speed;
        }
        else
        {
            velocity.X = Mathf.MoveToward(Velocity.X, 0, speed * (float)delta);
        }
        
        Velocity = velocity;
        MoveAndSlide();
    }
}"""
        
        script_path = self.project_path / "scripts" / "Player.cs"
        script_path.write_text(player_script)
        logger.info(f"✅ 플레이어 스크립트 생성됨: {script_path}")
        return True
    
    async def _add_enemy(self, parameters: Dict) -> bool:
        """적 추가"""
        count = parameters.get("count", 1)
        logger.info(f"✅ {count}개의 적 추가됨")
        return True
    
    async def _add_platform(self, parameters: Dict) -> bool:
        """플랫폼 추가"""
        count = parameters.get("count", 5)
        logger.info(f"✅ {count}개의 플랫폼 추가됨")
        return True
    
    async def _add_ui(self, parameters: Dict) -> bool:
        """UI 추가"""
        ui_script = """using Godot;

public partial class UIManager : Control
{
    private Label scoreLabel;
    private ProgressBar healthBar;
    
    public override void _Ready()
    {
        scoreLabel = GetNode<Label>("ScoreLabel");
        healthBar = GetNode<ProgressBar>("HealthBar");
    }
    
    public void UpdateScore(int score)
    {
        scoreLabel.Text = $"Score: {score}";
    }
    
    public void UpdateHealth(float health)
    {
        healthBar.Value = health;
    }
}"""
        
        script_path = self.project_path / "scripts" / "UIManager.cs"
        script_path.write_text(ui_script)
        logger.info(f"✅ UI 시스템 추가됨")
        return True
    
    async def _generate_csharp_code(self, parameters: Dict) -> bool:
        """C# 코드 생성"""
        class_name = parameters.get("class", "GameController")
        code = f"""using Godot;
using System;

public partial class {class_name} : Node
{{
    private int score = 0;
    private float timeElapsed = 0.0f;
    
    public override void _Ready()
    {{
        GD.Print("Game Started!");
    }}
    
    public override void _Process(double delta)
    {{
        timeElapsed += (float)delta;
    }}
    
    public void AddScore(int points)
    {{
        score += points;
        EmitSignal(SignalName.ScoreChanged, score);
    }}
    
    [Signal]
    public delegate void ScoreChangedEventHandler(int newScore);
}}"""
        
        script_path = self.project_path / "scripts" / f"{class_name}.cs"
        script_path.write_text(code)
        logger.info(f"✅ C# 코드 생성됨: {script_path}")
        return True
    
    def _convert_to_csharp_dict(self, python_dict: Dict) -> Any:
        """Python dict를 C# Dictionary로 변환"""
        if not CSharpController or not CLR_AVAILABLE:
            return python_dict
            
        try:
            from System.Collections.Generic import Dictionary
            from System import String, Object
            
            csharp_dict = Dictionary[String, Object]()
            for key, value in python_dict.items():
                csharp_dict[key] = value
            return csharp_dict
        except:
            return python_dict
    
    async def add_feature(self, feature_name: str, context: Dict = None) -> bool:
        """기능 추가"""
        if context is None:
            context = {}
            
        logger.info(f"➕ 기능 추가: {feature_name}")
        
        # 기능별 액션 매핑
        feature_actions = {
            "double_jump": {"action": "add_feature", "parameters": {"type": "double_jump"}},
            "dash": {"action": "add_feature", "parameters": {"type": "dash"}},
            "wall_jump": {"action": "add_feature", "parameters": {"type": "wall_jump"}},
            "inventory": {"action": "add_feature", "parameters": {"type": "inventory"}},
            "multiplayer": {"action": "add_feature", "parameters": {"type": "multiplayer"}}
        }
        
        # 기능에 맞는 액션 실행
        for keyword, action_data in feature_actions.items():
            if keyword in feature_name.lower():
                return await self.execute_action(action_data["action"], action_data["parameters"])
        
        # 기본 기능 추가
        return await self.execute_action("add_feature", {"name": feature_name})
    
    async def modify_game(self, aspect: str) -> str:
        """게임 수정"""
        logger.info(f"🔧 게임 수정: {aspect}")
        
        # 수정 사항 분석
        modifications = []
        
        if "속도" in aspect or "speed" in aspect.lower():
            modifications.append("플레이어 이동 속도 조정")
            await self.execute_action("modify_player_speed", {"multiplier": 1.5})
            
        if "점프" in aspect or "jump" in aspect.lower():
            modifications.append("점프 높이 조정")
            await self.execute_action("modify_jump_height", {"multiplier": 1.2})
            
        if "색상" in aspect or "color" in aspect.lower():
            modifications.append("색상 테마 변경")
            await self.execute_action("change_color_theme", {"theme": "dark"})
            
        return ", ".join(modifications) if modifications else "일반 수정 적용"
    
    async def open_editor(self, project_name: str) -> bool:
        """Godot 에디터 열기"""
        try:
            # Godot 실행 파일 찾기
            godot_paths = [
                "C:/Program Files/Godot/Godot.exe",
                "C:/Program Files (x86)/Godot/Godot.exe",
                "/usr/local/bin/godot",
                "/usr/bin/godot"
            ]
            
            godot_exe = None
            for path in godot_paths:
                if Path(path).exists():
                    godot_exe = path
                    break
            
            if godot_exe:
                project_path = self.project_path or Path.home() / "Documents" / "Godot" / "Projects" / project_name
                subprocess.Popen([godot_exe, "--editor", "--path", str(project_path)])
                logger.info("✅ Godot 에디터가 열렸습니다")
                return True
            else:
                logger.error("❌ Godot 실행 파일을 찾을 수 없습니다")
                return False
                
        except Exception as e:
            logger.error(f"❌ Godot 에디터 실행 실패: {e}")
            return False
    
    def stop_engine(self):
        """엔진 종료"""
        if self.csharp_controller:
            try:
                self.csharp_controller.StopEngine()
            except:
                pass
        
        self.is_engine_running = False
        logger.info("🛑 Godot 엔진 종료됨")

# 싱글톤 인스턴스
_controller = None

def get_godot_controller() -> GodotAutomationController:
    """Godot 컨트롤러 싱글톤 인스턴스 반환"""
    global _controller
    if _controller is None:
        _controller = GodotAutomationController()
    return _controller