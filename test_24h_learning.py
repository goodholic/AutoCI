#!/usr/bin/env python3
"""
24시간 학습 시스템 테스트
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

async def test_learning_system():
    """학습 시스템 테스트"""
    print("🧪 24시간 학습 시스템 테스트")
    print("=" * 60)
    
    try:
        from modules.csharp_24h_user_learning import CSharp24HUserLearning
        
        # 학습 시스템 초기화
        learning = CSharp24HUserLearning()
        
        # 현재 상태 확인
        print("\n📊 현재 학습 상태:")
        status = learning.get_learning_status()
        print(f"  전체 주제: {status['total_topics']}개")
        print(f"  완료된 주제: {status['completed_topics']}개")
        print(f"  남은 주제: {status['remaining_topics']}개")
        print(f"  완료율: {status['completion_rate']:.1f}%")
        print(f"  총 학습 시간: {status['total_learning_time']:.1f}시간")
        print(f"  현재 수준: {status['current_level']}")
        
        if status['next_topics']:
            print(f"\n📝 다음 학습 예정 주제:")
            for topic in status['next_topics']:
                print(f"  - {topic}")
        
        # 짧은 테스트 세션 실행
        print("\n🧪 테스트 세션 실행 (1개 주제만)")
        if status['next_topics']:
            test_topic = status['next_topics'][0]
            print(f"  테스트 주제: {test_topic}")
            
            # 빠른 복습 모드로 테스트
            await learning.quick_topic_review(test_topic)
            
            print("\n✅ 테스트 완료!")
        else:
            print("  ⚠️ 모든 주제가 완료되어 테스트할 주제가 없습니다.")
        
        print("\n💡 전체 실행 방법:")
        print("  - 남은 주제 학습: autoci learn 24h")
        print("  - 전체 주제 학습: autoci learn all")
        print("  - 대화형 메뉴: autoci learn")
        
    except ImportError as e:
        print(f"❌ 모듈 임포트 실패: {e}")
        print("필요한 모듈이 설치되었는지 확인하세요.")
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_learning_system())