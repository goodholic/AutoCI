#!/usr/bin/env python3
"""
AutoCI Talk 기능 테스트
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

async def test_talk():
    """Talk 기능 테스트"""
    try:
        from modules.korean_conversation import KoreanConversationSystem
        from modules.self_evolution_system import get_evolution_system
        
        print("🧠 학습한 지식 기반 대화 시스템 테스트")
        print("=" * 60)
        
        # 시스템 초기화
        conversation = KoreanConversationSystem()
        evolution = get_evolution_system()
        
        # 지식 베이스 확인
        if conversation.knowledge_base:
            print(f"✅ 지식 베이스 로드됨: {len(conversation.knowledge_base)} 카테고리")
            for category in list(conversation.knowledge_base.keys())[:5]:
                print(f"   - {category}")
        else:
            print("⚠️ 지식 베이스가 비어있습니다")
        
        print("\n📚 테스트 질문들:")
        test_questions = [
            "Panda3D 최적화 방법 알려줘",
            "pytorch 텐서 기초 설명해줘",
            "게임에 AI 추가해줘",
            "물리엔진이 뭐야?"
        ]
        
        for question in test_questions:
            print(f"\n👤 질문: {question}")
            response = await conversation.process_user_input(question, evolution)
            print(f"🤖 답변: {response[:200]}...")
            
            # 지식 베이스 검색 결과 확인
            entities = await conversation._extract_entities(question)
            results = conversation._search_knowledge_base(question, entities)
            if results:
                print(f"   📚 관련 지식 {len(results)}개 발견")
        
        print("\n✅ 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_talk())