#!/usr/bin/env python3
"""
공유 지식 베이스 통합 테스트
autoci fix, learn, create 간의 지식 공유 검증
"""

import asyncio
import logging
from modules.shared_knowledge_base import get_shared_knowledge_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_shared_knowledge():
    """공유 지식 베이스 테스트"""
    shared_kb = get_shared_knowledge_base()
    
    print("🧪 AutoCI 공유 지식 베이스 테스트")
    print("=" * 50)
    
    # 1. 초기 상태 확인
    stats = shared_kb.get_knowledge_stats()
    print(f"\n📊 초기 지식 베이스 상태:")
    print(f"   • 총 검색 결과: {stats['total_searches']}개")
    print(f"   • 캐시된 검색: {stats['cached_searches']}개")
    print(f"   • 저장된 솔루션: {stats['total_solutions']}개")
    print(f"   • 베스트 프랙티스: {stats['total_practices']}개")
    
    # 2. 검색 결과 저장 테스트 (autoci fix가 사용)
    print(f"\n🔍 검색 결과 저장 테스트...")
    test_search = {
        "keyword": "Godot C# 플레이어 이동",
        "results": [
            {"source": "Godot Docs", "content": "CharacterBody2D 사용법"},
            {"source": "GitHub", "content": "Input.GetVector() 예제"}
        ],
        "summary": "Godot에서 C#으로 플레이어 이동 구현 방법"
    }
    await shared_kb.save_search_result(test_search["keyword"], test_search)
    print("✅ 검색 결과 저장 완료")
    
    # 3. 캐시된 검색 확인 (autoci learn/create가 사용)
    print(f"\n📚 캐시된 검색 결과 확인...")
    cached = await shared_kb.get_cached_search("Godot C# 플레이어 이동")
    if cached:
        print(f"✅ 캐시 발견! 키워드: {cached.get('keyword')}")
        print(f"   요약: {cached.get('summary')}")
    else:
        print("❌ 캐시를 찾을 수 없음")
    
    # 4. 솔루션 저장 테스트
    print(f"\n🔧 오류 솔루션 저장 테스트...")
    await shared_kb.save_solution(
        error_type="NullReferenceException",
        error_message="Object reference not set to an instance",
        solution="null 체크 추가: if (myObject != null)",
        success=True
    )
    print("✅ 솔루션 저장 완료")
    
    # 5. 솔루션 검색
    print(f"\n🔍 저장된 솔루션 검색...")
    solution = await shared_kb.get_solution_for_error(
        "NullReferenceException",
        "Object reference not set to an instance"
    )
    if solution:
        print(f"✅ 솔루션 발견: {solution.get('solution')}")
    else:
        print("❌ 솔루션을 찾을 수 없음")
    
    # 6. 베스트 프랙티스 저장
    print(f"\n⭐ 베스트 프랙티스 저장 테스트...")
    await shared_kb.save_best_practice(
        topic="Godot C# 최적화",
        practice={
            "title": "오브젝트 풀링",
            "description": "자주 생성/삭제되는 오브젝트는 풀링 사용",
            "code_example": "ObjectPool<Bullet> bulletPool = new ObjectPool<Bullet>();",
            "performance_gain": "30-50% 성능 향상"
        }
    )
    print("✅ 베스트 프랙티스 저장 완료")
    
    # 7. 베스트 프랙티스 조회
    print(f"\n📖 베스트 프랙티스 조회...")
    practices = await shared_kb.get_best_practices("Godot C# 최적화")
    if practices:
        print(f"✅ {len(practices)}개의 베스트 프랙티스 발견")
        for p in practices[:2]:
            print(f"   • {p.get('title', 'N/A')}")
    
    # 8. 최종 통계
    final_stats = shared_kb.get_knowledge_stats()
    print(f"\n📊 최종 지식 베이스 상태:")
    print(f"   • 총 검색 결과: {final_stats['total_searches']}개 (+{final_stats['total_searches'] - stats['total_searches']})")
    print(f"   • 캐시된 검색: {final_stats['cached_searches']}개 (+{final_stats['cached_searches'] - stats['cached_searches']})")
    print(f"   • 저장된 솔루션: {final_stats['total_solutions']}개 (+{final_stats['total_solutions'] - stats['total_solutions']})")
    print(f"   • 베스트 프랙티스: {final_stats['total_practices']}개 (+{final_stats['total_practices'] - stats['total_practices']})")
    
    print("\n✅ 공유 지식 베이스 테스트 완료!")
    print("이제 autoci fix가 수집한 정보를 autoci learn과 create가 활용할 수 있습니다.")

if __name__ == "__main__":
    asyncio.run(test_shared_knowledge())