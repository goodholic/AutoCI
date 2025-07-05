#!/usr/bin/env python3
"""
실시간 학습 통합기 테스트
개발 경험이 어떻게 AI 학습 데이터로 변환되는지 시연합니다.
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from modules.realtime_learning_integrator import (
    get_realtime_integrator,
    start_integration,
    add_experience,
    get_status
)
from modules.development_experience_collector import get_experience_collector
from core.continuous_learning_system import ContinuousLearningSystem

async def test_realtime_integration():
    """실시간 통합 테스트"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           🔄 실시간 학습 통합기 테스트                       ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # 시스템 초기화
    print("1️⃣ 시스템 초기화 중...")
    
    # 학습 시스템 초기화
    learning_system = ContinuousLearningSystem()
    print("   ✅ 연속 학습 시스템 준비")
    
    # 경험 수집기 초기화
    experience_collector = get_experience_collector()
    print("   ✅ 경험 수집기 준비")
    
    # 통합기 시작
    integrator = await start_integration(
        continuous_learning_system=learning_system,
        experience_collector=experience_collector
    )
    print("   ✅ 실시간 통합기 시작\n")
    
    # 테스트 경험 데이터 추가
    print("2️⃣ 테스트 경험 데이터 추가 중...")
    
    # 1. 오류 해결 경험
    await add_experience('error_solution', {
        'error': {
            'type': 'NullReferenceException',
            'description': 'Godot 노드가 null 참조 예외 발생',
            'context': 'Player 스크립트에서 _Ready() 메서드 실행 중'
        },
        'solution': {
            'steps': '1. 노드 경로 확인\n2. GetNode() 호출 전 null 체크\n3. 노드 존재 확인',
            'strategy': 'defensive_programming'
        },
        'code': '''
public override void _Ready()
{
    // 해결 전
    var healthBar = GetNode<ProgressBar>("UI/HealthBar");
    healthBar.Value = 100; // NullReferenceException!
    
    // 해결 후
    var healthBar = GetNode<ProgressBar>("UI/HealthBar");
    if (healthBar != null)
    {
        healthBar.Value = 100;
    }
    else
    {
        GD.PrintErr("HealthBar 노드를 찾을 수 없습니다!");
    }
}
''',
        'success': True,
        'attempts': 2,
        'effectiveness': 0.95
    })
    print("   ✅ 오류 해결 경험 추가")
    
    # 2. 게임 메카닉 경험
    await add_experience('game_mechanic', {
        'name': '대시 시스템',
        'description': '플레이어가 빠르게 이동하는 대시 메카닉',
        'code_snippet': '''
public class DashSystem : Node
{
    [Export] private float dashSpeed = 500f;
    [Export] private float dashDuration = 0.2f;
    private bool isDashing = false;
    
    public async void PerformDash(Vector2 direction)
    {
        if (isDashing) return;
        
        isDashing = true;
        var player = GetParent<CharacterBody2D>();
        
        // 대시 실행
        var dashVelocity = direction.Normalized() * dashSpeed;
        player.Velocity = dashVelocity;
        
        // 대시 지속 시간
        await ToSignal(GetTree().CreateTimer(dashDuration), "timeout");
        
        isDashing = false;
    }
}
''',
        'performance': {
            'fps_impact': 'minimal',
            'memory_usage': '< 1MB'
        },
        'complexity': 25,
        'effectiveness': 0.9
    })
    print("   ✅ 게임 메카닉 경험 추가")
    
    # 3. 성능 최적화 경험
    await add_experience('performance_opt', {
        'type': 'draw_calls',
        'before': {
            'fps': 45,
            'draw_calls': 150,
            'vertices': 50000
        },
        'after': {
            'fps': 60,
            'draw_calls': 50,
            'vertices': 30000
        },
        'method': 'Texture Atlas와 Batching 활용',
        'code_changes': '''
// 이전: 개별 스프라이트
foreach (var enemy in enemies)
{
    enemy.Texture = GD.Load<Texture2D>("res://enemies/enemy.png");
}

// 이후: 텍스처 아틀라스
var atlas = GD.Load<Texture2D>("res://enemies/enemy_atlas.png");
foreach (var enemy in enemies)
{
    enemy.Texture = atlas;
    enemy.RegionEnabled = true;
    enemy.RegionRect = new Rect2(x, y, 32, 32);
}
''',
        'improvement': 33.3
    })
    print("   ✅ 성능 최적화 경험 추가")
    
    # 4. AI 발견 경험
    await add_experience('ai_discovery', {
        'discovery_type': 'pathfinding_optimization',
        'description': 'A* 알고리즘에 점프 포인트 서치 기법 결합',
        'code': '''
public class ImprovedPathfinding : NavigationAgent2D
{
    // AI가 발견한 최적화: 대각선 이동 시 점프 포인트 활용
    private List<Vector2> FindJumpPoints(Vector2 start, Vector2 end)
    {
        var jumpPoints = new List<Vector2>();
        // 혁신적인 점프 포인트 탐색 로직
        return jumpPoints;
    }
}
''',
        'context': '대규모 맵에서 경로 탐색 성능 개선',
        'creativity_score': 8,
        'effectiveness': 0.85
    })
    print("   ✅ AI 발견 경험 추가\n")
    
    # 처리 대기
    print("3️⃣ 경험 데이터 처리 중...")
    await asyncio.sleep(3)
    
    # 상태 확인
    status = get_status()
    print(f"\n4️⃣ 통합 상태:")
    print(f"   - 변환된 경험: {status['stats']['total_experiences_converted']}")
    print(f"   - 생성된 Q&A: {status['stats']['qa_pairs_generated']}")
    print(f"   - 지식 업데이트: {status['stats']['knowledge_updates']}")
    print(f"   - 큐 크기: {status['queue_size']}")
    
    # 특화 데이터셋 생성
    print("\n5️⃣ 특화 학습 데이터셋 생성 중...")
    dataset = await integrator.create_specialized_training_dataset('error')
    print(f"   ✅ 오류 해결 데이터셋 생성: {dataset['statistics']['total_pairs']}개 Q&A")
    
    dataset = await integrator.create_specialized_training_dataset('game')
    print(f"   ✅ 게임 메카닉 데이터셋 생성: {dataset['statistics']['total_pairs']}개 Q&A")
    
    # 지식 베이스 확인
    print("\n6️⃣ 지식 베이스 업데이트 확인:")
    kb = learning_system.knowledge_base
    print(f"   - C# 패턴: {len(kb['csharp_patterns'])}개")
    print(f"   - 한글 번역: {len(kb['korean_translations'])}개")
    print(f"   - Godot 통합: {len(kb['godot_integrations'])}개")
    print(f"   - 공통 오류: {len(kb['common_errors'])}개")
    print(f"   - 모범 사례: {len(kb['best_practices'])}개")
    
    # 학습 시스템에 추가된 Q&A 확인
    print("\n7️⃣ 학습 시스템 Q&A 확인:")
    recent_qa_dir = Path("continuous_learning/answers") / datetime.now().strftime("%Y%m%d")
    if recent_qa_dir.exists():
        qa_files = list(recent_qa_dir.glob("*.json"))
        print(f"   ✅ 오늘 생성된 Q&A 파일: {len(qa_files)}개")
    
    # 보고서 생성
    print("\n8️⃣ 통합 보고서 생성 중...")
    await integrator._generate_integration_report()
    print("   ✅ 보고서 생성 완료")
    
    # 통합 중지
    print("\n9️⃣ 통합 중지 중...")
    await integrator.stop_realtime_processing()
    print("   ✅ 실시간 통합 중지")
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    ✅ 테스트 완료!                           ║
║                                                              ║
║  개발 경험이 성공적으로 AI 학습 데이터로 변환되었습니다.    ║
║  이제 AI는 실제 개발 경험을 바탕으로 더 나은 답변을        ║
║  제공할 수 있습니다.                                        ║
╚══════════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    from datetime import datetime
    
    print(f"실행 시간: {datetime.now()}")
    asyncio.run(test_realtime_integration())