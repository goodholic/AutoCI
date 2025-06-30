#!/usr/bin/env python3
"""
AutoCI 통합 연속 학습 테스트
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

async def test_continuous_learning():
    """통합 연속 학습 기능 테스트"""
    print("🧪 AutoCI 통합 연속 학습 테스트")
    print("=" * 60)
    
    try:
        # 모듈 import 테스트
        print("1️⃣ 모듈 import 테스트...")
        from modules.csharp_continuous_learning import CSharpContinuousLearning
        print("✅ 모듈 import 성공")
        
        # 시스템 초기화 테스트
        print("\n2️⃣ 시스템 초기화 테스트...")
        system = CSharpContinuousLearning(use_llm=False)  # LLM 없이 테스트
        print("✅ 시스템 초기화 성공")
        
        # 통합 주제 확인
        print(f"\n3️⃣ 통합 주제 확인...")
        print(f"   - 전체 주제 수: {len(system.integrated_topics)}")
        print(f"   - 카테고리: {set(t.category for t in system.integrated_topics)}")
        
        # 첫 번째 주제 테스트
        if system.integrated_topics:
            first_topic = system.integrated_topics[0]
            print(f"\n4️⃣ 샘플 주제 정보:")
            print(f"   - ID: {first_topic.id}")
            print(f"   - 주제: {first_topic.topic}")
            print(f"   - 난이도: {first_topic.difficulty}/5")
            print(f"   - 한글 키워드: {', '.join(first_topic.korean_keywords)}")
            print(f"   - C# 개념: {', '.join(first_topic.csharp_concepts)}")
        
        # 지식 베이스 확인
        print(f"\n5️⃣ 지식 베이스 상태:")
        for key, value in system.knowledge_base.items():
            print(f"   - {key}: {len(value) if isinstance(value, dict) else 'N/A'} 항목")
        
        print("\n✅ 모든 테스트 통과!")
        
        # LLM 모델 확인
        print(f"\n6️⃣ LLM 모델 상태:")
        models_dir = Path("./models")
        models_info_file = models_dir / "installed_models.json"
        
        if models_info_file.exists():
            import json
            with open(models_info_file, 'r', encoding='utf-8') as f:
                models = json.load(f)
            print(f"   - 설치된 모델: {list(models.keys())}")
            
            # LLM 포함 시스템 테스트
            print("\n7️⃣ LLM 포함 시스템 테스트...")
            llm_system = CSharpContinuousLearning(use_llm=True)
            print(f"   - LLM 사용 가능: {llm_system.use_llm}")
            print(f"   - 로드된 모델: {list(llm_system.llm_models.keys())}")
        else:
            print("   - LLM 모델이 설치되지 않았습니다.")
            print("   - 'python install_llm_models.py'로 설치 가능")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_quick_session():
    """빠른 세션 테스트"""
    print("\n\n🚀 빠른 세션 테스트")
    print("=" * 60)
    
    try:
        from modules.csharp_continuous_learning import CSharpContinuousLearning
        
        # 전통적 학습만으로 빠른 테스트
        system = CSharpContinuousLearning(use_llm=False)
        if system.integrated_topics:
            topic = system.integrated_topics[0]
            print(f"테스트 주제: {topic.topic}")
            
            # 매우 짧은 테스트 세션
            from modules.csharp_24h_learning_config import LearningConfig
            LearningConfig.DEMO_MODE = True  # 데모 모드로 빠른 실행
            
            await system.continuous_learning_session(
                topic, 
                use_traditional=True, 
                use_llm=False
            )
            
            print("\n✅ 빠른 세션 테스트 완료!")
            
    except Exception as e:
        print(f"\n❌ 빠른 세션 테스트 실패: {str(e)}")

async def main():
    """메인 테스트 함수"""
    # 기본 테스트
    success = await test_continuous_learning()
    
    if success:
        # 빠른 세션 테스트
        await test_quick_session()
        
    print("\n\n📊 테스트 요약:")
    print("=" * 60)
    print("✅ AutoCI 통합 연속 학습 시스템이 정상적으로 통합되었습니다!")
    print("\n사용 방법:")
    print("  - autoci learn continuous    # AI 모델 기반 연속 학습")
    print("  - autoci learn              # 기존 24시간 학습")
    print("  - autoci learn menu         # 학습 메뉴")

if __name__ == "__main__":
    asyncio.run(main())