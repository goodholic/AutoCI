#!/usr/bin/env python3
"""
개발 경험 수집기 테스트 및 통합 예제
"""

import asyncio
from pathlib import Path
from modules.development_experience_collector import get_experience_collector
from modules.persistent_game_improver import get_persistent_improver
from modules.extreme_persistence_engine import get_extreme_persistence_engine

async def test_experience_collector():
    """경험 수집기 테스트"""
    print("🧪 개발 경험 수집기 테스트 시작\n")
    
    # 수집기 인스턴스 가져오기
    collector = get_experience_collector()
    
    # 1. 오류 해결책 수집 테스트
    print("1️⃣ 오류 해결책 수집 테스트")
    test_error = {
        'type': 'script_error',
        'description': 'Player.gd - Invalid get index "velocity" on base: "Nil"',
        'file': 'scripts/Player.gd',
        'line': 42
    }
    
    test_solution = {
        'strategy': 'null_check_addition',
        'attempts': 3,
        'code': 'if self: velocity = Vector2.ZERO'
    }
    
    await collector.collect_error_solution(test_error, test_solution, True)
    print("✅ 오류 해결책 수집 완료\n")
    
    # 2. 게임 메카닉 수집 테스트
    print("2️⃣ 게임 메카닉 수집 테스트")
    dash_mechanic = {
        'code': """
func dash():
    if can_dash:
        velocity.x = DASH_SPEED * direction
        can_dash = false
        $DashTimer.start()
""",
        'description': '대시 메카닉 구현'
    }
    
    performance_metrics = {
        'fps_impact': -2,  # FPS 2 감소
        'response_time': 0.05  # 50ms 반응 시간
    }
    
    await collector.collect_game_mechanic("dash_system", dash_mechanic, performance_metrics)
    print("✅ 게임 메카닉 수집 완료\n")
    
    # 3. 코드 패턴 수집 테스트
    print("3️⃣ 코드 패턴 수집 테스트")
    singleton_pattern = """
# 싱글톤 패턴
var _instance = null

func get_instance():
    if _instance == null:
        _instance = self.new()
    return _instance
"""
    
    await collector.collect_code_pattern(
        "godot_singleton",
        singleton_pattern,
        "Godot에서 싱글톤 구현",
        effectiveness=0.9
    )
    print("✅ 코드 패턴 수집 완료\n")
    
    # 4. 성능 최적화 수집 테스트
    print("4️⃣ 성능 최적화 수집 테스트")
    optimization = {
        'type': 'physics_optimization',
        'before': {'fps': 45, 'physics_time': 12.5},
        'after': {'fps': 58, 'physics_time': 8.2},
        'method': 'Reduced physics tick rate from 60 to 30 for background objects',
        'code_changes': 'set_physics_process(false) for static objects'
    }
    
    await collector.collect_performance_optimization(optimization)
    print("✅ 성능 최적화 수집 완료\n")
    
    # 5. 리소스 패턴 수집 테스트
    print("5️⃣ 리소스 패턴 수집 테스트")
    texture_generation = {
        'method': 'procedural_gradient',
        'parameters': {'width': 256, 'height': 256, 'colors': ['#FF0000', '#00FF00']},
        'code': 'Image.create_from_data() with gradient algorithm',
        'success_rate': 0.95
    }
    
    await collector.collect_resource_pattern("texture", texture_generation)
    print("✅ 리소스 패턴 수집 완료\n")
    
    # 6. 커뮤니티 솔루션 수집 테스트
    print("6️⃣ 커뮤니티 솔루션 수집 테스트")
    community_solution = {
        'solution': 'Use Area2D instead of CharacterBody2D for triggers',
        'code': 'Replace CharacterBody2D with Area2D node',
        'votes': 42,
        'verified': True
    }
    
    await collector.collect_community_solution(
        "Character not detecting area triggers",
        community_solution,
        "Reddit r/godot"
    )
    print("✅ 커뮤니티 솔루션 수집 완료\n")
    
    # 7. AI 발견 수집 테스트
    print("7️⃣ AI 발견 수집 테스트")
    ai_discovery = {
        'type': 'creative_workaround',
        'description': 'Using shader to simulate physics instead of physics engine',
        'code': 'shader_type canvas_item; // Physics simulation in shader',
        'context': 'Performance optimization for 1000+ particles',
        'creativity_score': 8,
        'effectiveness': 0.85
    }
    
    await collector.collect_ai_discovery(ai_discovery)
    print("✅ AI 발견 수집 완료\n")
    
    # 8. 유사 문제 검색 테스트
    print("8️⃣ 유사 문제 검색 테스트")
    search_problem = {
        'type': 'script_error',
        'description': 'Player.gd - Cannot access velocity'
    }
    
    similar_solutions = collector.search_similar_problems(search_problem)
    print(f"발견된 유사 해결책: {len(similar_solutions)}개\n")
    
    # 9. 모범 사례 조회
    print("9️⃣ 모범 사례 조회")
    best_practices = collector.get_best_practices()
    print(f"모범 사례 {len(best_practices)}개:")
    for practice in best_practices[:3]:
        print(f"  - {practice}")
    print()
    
    # 10. 학습 인사이트 조회
    print("🔟 학습 인사이트")
    insights = collector.get_learning_insights()
    print(f"총 경험: {insights['total_experiences']}")
    print(f"성공률: {insights['success_rate']:.1%}")
    print(f"AI 창의성 점수: {insights['ai_creativity_score']:.1f}/10")
    print(f"커뮤니티 기여: {insights['community_contribution']}")

async def test_integration():
    """기존 시스템과의 통합 테스트"""
    print("\n\n🔗 시스템 통합 테스트\n")
    
    # 수집기와 개선 시스템 가져오기
    collector = get_experience_collector()
    improver = get_persistent_improver()
    extreme_engine = get_extreme_persistence_engine()
    
    # AI 모델 컨트롤러도 통합
    try:
        from modules.ai_model_controller import AIModelController
        ai_controller = AIModelController()
        await collector.integrate_with_ai_controller(ai_controller)
        ai_integrated = True
    except Exception as e:
        print(f"⚠️ AI 모델 컨트롤러 통합 스킵: {e}")
        ai_integrated = False
    
    # 통합
    await collector.integrate_with_improver(improver)
    await collector.integrate_with_extreme_engine(extreme_engine)
    
    print("✅ 시스템 통합 완료")
    print("- persistent_game_improver 통합 ✓")
    print("- extreme_persistence_engine 통합 ✓")
    if ai_integrated:
        print("- ai_model_controller 통합 ✓")
    print("\n이제 게임 개발 중 모든 학습 경험이 자동으로 수집됩니다!")
    
    # 모니터링 시작 (데모용으로 짧게)
    project_path = Path("test_project")
    monitoring_task = asyncio.create_task(collector.start_monitoring(project_path))
    
    print(f"\n📡 {project_path} 프로젝트 모니터링 시작...")
    await asyncio.sleep(5)  # 5초 동안 모니터링
    
    collector.stop_monitoring()
    monitoring_task.cancel()
    
    print("\n✅ 모니터링 중지 및 지식 저장 완료")

async def main():
    """메인 테스트 함수"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🧪 개발 경험 수집기 테스트                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # 기본 기능 테스트
    await test_experience_collector()
    
    # 시스템 통합 테스트
    await test_integration()
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ✅ 테스트 완료!                                            ║
║                                                                              ║
║  개발 경험 수집기가 성공적으로 작동합니다.                                    ║
║  이제 24시간 게임 개발 중 모든 가치있는 경험이 자동으로 수집되고 학습됩니다!   ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    asyncio.run(main())