#!/usr/bin/env python3
"""
WSL 환경을 위한 가상 입력 데모
GUI 없이 터미널에서 동작 확인
"""

import asyncio
import sys
import time
from modules.virtual_input_controller import get_virtual_input, InputMode


async def test_basic_functions():
    """기본 기능 테스트 (GUI 없이)"""
    print("🎮 가상 입력 시스템 기본 테스트")
    print("=" * 50)
    
    virtual_input = get_virtual_input()
    await virtual_input.activate()
    
    print("\n1️⃣ 시스템 정보")
    print(f"   - 가상 스크린: {virtual_input.virtual_screen.width}x{virtual_input.virtual_screen.height}")
    print(f"   - 현재 모드: {virtual_input.mode.value}")
    print(f"   - 매크로 개수: {len(virtual_input.macro_library)}")
    
    # 액션 시뮬레이션 (실제로 실행하지 않고 기록만)
    print("\n2️⃣ 액션 시뮬레이션 (기록만)")
    
    actions = [
        ("마우스 이동", lambda: virtual_input.move_mouse(100, 100, 0.1)),
        ("클릭", lambda: virtual_input.click()),
        ("텍스트 입력", lambda: virtual_input.type_text("Hello AutoCI")),
        ("단축키", lambda: virtual_input.hotkey("ctrl", "s")),
    ]
    
    for name, action in actions:
        print(f"   - {name}...", end="")
        try:
            await action()
            print(" ✅ (시뮬레이션 성공)")
        except Exception as e:
            print(f" ❌ ({e})")
        await asyncio.sleep(0.1)
    
    # 통계 확인
    print("\n3️⃣ 입력 패턴 통계")
    stats = virtual_input.get_pattern_statistics()
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    # 액션 히스토리
    print("\n4️⃣ 최근 액션 히스토리")
    history = virtual_input.get_action_history()
    for i, action in enumerate(history[-5:], 1):
        print(f"   {i}. {action['type']}: {action.get('data', {})}")
    
    await virtual_input.deactivate()
    print("\n✅ 테스트 완료!")


async def test_godot_simulation():
    """Godot 조작 시뮬레이션"""
    print("\n🎮 Godot 조작 시뮬레이션")
    print("=" * 50)
    
    virtual_input = get_virtual_input()
    await virtual_input.activate()
    virtual_input.set_mode(InputMode.GODOT_EDITOR)
    
    print("Godot 에디터 모드로 전환됨")
    
    # Godot 조작 시나리오
    scenarios = [
        ("노드 생성", lambda: virtual_input.godot_create_node("CharacterBody2D", "Player")),
        ("스크립트 추가", lambda: virtual_input.godot_add_script("extends CharacterBody2D")),
        ("씬 저장", lambda: virtual_input.execute_macro("godot_save")),
        ("게임 실행", lambda: virtual_input.execute_macro("godot_run_game")),
    ]
    
    print("\n시뮬레이션 시나리오:")
    for i, (name, action) in enumerate(scenarios, 1):
        print(f"\n{i}. {name}")
        try:
            await action()
            print("   → 성공적으로 시뮬레이션됨")
            
            # 해당 액션이 어떤 입력을 생성하는지 표시
            recent_actions = virtual_input.get_action_history()[-3:]
            for action in recent_actions:
                print(f"     • {action['type']}: {list(action.get('data', {}).keys())}")
        except Exception as e:
            print(f"   → 오류: {e}")
        
        await asyncio.sleep(0.5)
    
    # 최종 통계
    print("\n📊 Godot 조작 통계:")
    stats = virtual_input.get_pattern_statistics()
    print(f"   - 총 액션: {stats['total_patterns']}")
    print(f"   - Godot 작업: {stats['godot_operations']}")
    print(f"   - 마우스 조작: {stats['mouse_moves'] + stats['clicks']}")
    print(f"   - 키보드 입력: {stats['keyboard_inputs']}")
    
    await virtual_input.deactivate()


async def test_complex_learning():
    """복합 학습 시스템 테스트"""
    print("\n🧠 복합 학습 시스템 연동 테스트")
    print("=" * 50)
    
    try:
        from modules.complex_learning_integration import get_complex_learning
        
        complex_learning = get_complex_learning()
        print("✅ 복합 학습 시스템 로드 성공")
        
        # 컴포넌트 확인
        components = {
            "가상 입력": hasattr(complex_learning, 'virtual_input'),
            "Godot 학습": hasattr(complex_learning, 'godot_learning'),
            "연속 학습": hasattr(complex_learning, 'continuous_learning'),
            "게임 파이프라인": hasattr(complex_learning, 'game_pipeline')
        }
        
        print("\n시스템 컴포넌트:")
        for name, exists in components.items():
            status = "✅" if exists else "❌"
            print(f"   {status} {name}")
        
        # 통합 지식 구조 확인
        print("\n통합 지식 구조:")
        for category in complex_learning.integrated_knowledge.keys():
            count = len(complex_learning.integrated_knowledge[category])
            print(f"   - {category}: {count}개 항목")
        
        # 학습 통계
        print("\n학습 통계:")
        for stat, value in complex_learning.stats.items():
            print(f"   - {stat}: {value}")
            
    except Exception as e:
        print(f"❌ 복합 학습 시스템 로드 실패: {e}")


async def main():
    """메인 메뉴"""
    print("🤖 AutoCI 가상 입력 시스템 (WSL 버전)")
    print("=" * 50)
    print("WSL 환경에서 안전하게 테스트합니다.")
    
    while True:
        print("\n메뉴:")
        print("1. 기본 기능 테스트")
        print("2. Godot 조작 시뮬레이션")
        print("3. 복합 학습 시스템 테스트")
        print("4. 종료")
        
        choice = input("\n선택 (1-4): ")
        
        if choice == "1":
            await test_basic_functions()
        elif choice == "2":
            await test_godot_simulation()
        elif choice == "3":
            await test_complex_learning()
        elif choice == "4":
            print("👋 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다.")
        
        input("\nEnter를 눌러 계속...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⛔ 프로그램이 중단되었습니다.")