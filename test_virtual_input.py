#!/usr/bin/env python3
"""
AutoCI 가상 입력 시스템 테스트
독립적인 마우스/키보드로 AI가 자율 조작하는 데모
"""

import asyncio
import math
from modules.virtual_input_controller import get_virtual_input, InputMode


async def demo_virtual_input():
    """가상 입력 시스템 데모"""
    print("🎮 AutoCI 가상 입력 시스템 데모")
    print("=" * 50)
    
    virtual_input = get_virtual_input()
    await virtual_input.activate()
    
    # 1. 마우스 이동 데모
    print("\n1️⃣ 부드러운 마우스 이동 시연")
    await virtual_input.move_mouse(960, 540, duration=1.5)
    await asyncio.sleep(1)
    
    # 2. 마우스 패턴 그리기
    print("\n2️⃣ 마우스로 패턴 그리기")
    center_x, center_y = 960, 540
    radius = 150
    
    # 원 그리기
    for angle in range(0, 361, 5):
        x = center_x + radius * math.cos(math.radians(angle))
        y = center_y + radius * math.sin(math.radians(angle))
        await virtual_input.move_mouse(int(x), int(y), duration=0.05)
    
    await asyncio.sleep(1)
    
    # 3. 텍스트 타이핑 데모
    print("\n3️⃣ AI가 코드를 작성합니다")
    code_text = """// AutoCI AI가 자동으로 작성하는 Godot C# 코드
using Godot;

public partial class Player : CharacterBody2D
{
    private const float Speed = 300.0f;
    private const float JumpVelocity = -400.0f;
    
    public override void _PhysicsProcess(double delta)
    {
        Vector2 velocity = Velocity;
        
        // Add gravity
        if (!IsOnFloor())
            velocity.Y += GetGravity().Y * (float)delta;
            
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
}
"""
    
    # WSL 환경 체크
    import platform
    if "microsoft" in platform.uname().release.lower():
        print("\n⚠️  WSL 환경 감지 - 텍스트 에디터 열기 건너뜀")
        print("   (WSL에서는 Windows 앱 직접 실행이 제한됩니다)")
    else:
        # 메모장 열기 (Windows) 또는 텍스트 에디터
        try:
            if platform.system() == "Windows":
                await virtual_input.hotkey("win", "r")
                await asyncio.sleep(0.5)
                await virtual_input.type_text("notepad")
                await virtual_input.press_key("enter")
            else:
                # Linux/Mac에서는 터미널에 출력
                print("텍스트 에디터 시뮬레이션...")
        except Exception as e:
            print(f"에디터 열기 실패: {e}")
        await asyncio.sleep(1)
    
    # 코드 타이핑 (빠르게)
    await virtual_input.type_text(code_text, interval=0.01)
    
    await asyncio.sleep(2)
    
    # 4. 단축키 조합 데모
    print("\n4️⃣ 단축키 조합 실행")
    await virtual_input.hotkey("ctrl", "a")  # 전체 선택
    await asyncio.sleep(0.5)
    await virtual_input.hotkey("ctrl", "c")  # 복사
    await asyncio.sleep(0.5)
    
    # 5. Godot 관련 매크로 시연
    print("\n5️⃣ Godot 편집기 매크로 시뮬레이션")
    
    # 새 파일
    await virtual_input.hotkey("ctrl", "n")
    await asyncio.sleep(0.5)
    
    # 붙여넣기
    await virtual_input.hotkey("ctrl", "v")
    await asyncio.sleep(0.5)
    
    # 저장
    await virtual_input.hotkey("ctrl", "s")
    await asyncio.sleep(0.5)
    await virtual_input.type_text("AI_Generated_Player.cs")
    await virtual_input.press_key("enter")
    
    # 6. 액션 히스토리 확인
    print("\n6️⃣ 수행한 액션 히스토리")
    history = virtual_input.get_action_history()
    print(f"총 {len(history)}개의 액션 기록됨")
    
    # 최근 5개 액션 표시
    for action in history[-5:]:
        print(f"  - {action['type']}: {action['data']}")
    
    await virtual_input.deactivate()
    print("\n✅ 가상 입력 데모 완료!")
    print("\n💡 이 시스템으로 AutoCI는 독립적으로 마우스와 키보드를 제어할 수 있습니다.")
    print("   실제 사용자 입력과 충돌하지 않고 백그라운드에서 작업 가능합니다.")


async def demo_godot_automation():
    """Godot 자동화 시나리오"""
    print("\n🎮 Godot 자동화 시나리오")
    print("=" * 50)
    
    virtual_input = get_virtual_input()
    await virtual_input.activate()
    virtual_input.set_mode(InputMode.GODOT_EDITOR)
    
    print("\n시나리오: 2D 플랫폼 게임 자동 생성")
    
    # 1. 프로젝트 생성
    print("\n1️⃣ 새 프로젝트 생성")
    await virtual_input.execute_macro("godot_new_scene")
    
    # 2. 노드 추가
    print("\n2️⃣ 게임 노드 구조 생성")
    await virtual_input.godot_create_node("Node2D", "Main")
    await asyncio.sleep(0.5)
    await virtual_input.godot_create_node("CharacterBody2D", "Player")
    await asyncio.sleep(0.5)
    await virtual_input.godot_create_node("TileMap", "World")
    
    # 3. 스크립트 추가
    print("\n3️⃣ AI가 스크립트 작성")
    player_script = """extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

func _physics_process(delta):
    if not is_on_floor():
        velocity.y += gravity * delta
    
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = JUMP_VELOCITY
    
    var direction = Input.get_axis("ui_left", "ui_right")
    if direction:
        velocity.x = direction * SPEED
    else:
        velocity.x = move_toward(velocity.x, 0, SPEED)
    
    move_and_slide()
"""
    
    await virtual_input.godot_add_script(player_script)
    
    # 4. 게임 실행
    print("\n4️⃣ 게임 테스트 실행")
    await virtual_input.execute_macro("godot_run_game")
    
    await virtual_input.deactivate()
    print("\n✅ Godot 자동화 시나리오 완료!")


async def main():
    """메인 실행 함수"""
    while True:
        print("\n🤖 AutoCI 가상 입력 시스템")
        print("=" * 50)
        print("1. 가상 입력 기본 데모")
        print("2. Godot 자동화 시나리오")
        print("3. 종료")
        
        choice = input("\n선택하세요 (1-3): ")
        
        if choice == "1":
            await demo_virtual_input()
        elif choice == "2":
            await demo_godot_automation()
        elif choice == "3":
            print("👋 프로그램을 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다.")
        
        input("\n계속하려면 Enter를 누르세요...")


if __name__ == "__main__":
    print("⚠️  주의: 이 프로그램은 마우스와 키보드를 자동으로 제어합니다.")
    print("   실행 중에는 마우스/키보드를 건드리지 마세요.")
    print("   긴급 중단: Ctrl+C")
    input("\n시작하려면 Enter를 누르세요...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⛔ 프로그램이 중단되었습니다.")