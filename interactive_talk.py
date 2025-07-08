#!/usr/bin/env python3
"""
대화형 Talk 테스트
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

async def interactive_talk():
    """대화형 Talk 테스트"""
    try:
        from modules.korean_conversation import KoreanConversationSystem
        from modules.self_evolution_system import get_evolution_system
        
        print("\n💬 AutoCI 대화 모드")
        print("🧠 학습한 지식을 바탕으로 질문에 답변하고 게임을 개선합니다")
        print("=" * 60)
        print("종료하려면 '종료', 'exit', 'quit'를 입력하세요.")
        print("=" * 60)
        
        # 시스템 초기화
        conversation = KoreanConversationSystem()
        evolution = get_evolution_system()
        
        # 지식 베이스 상태 표시
        if conversation.knowledge_base:
            total_items = sum(
                len(items) if isinstance(items, list) else 
                sum(len(subitems) for subitems in items.values() if isinstance(subitems, list))
                for items in conversation.knowledge_base.values()
            )
            print(f"\n✅ 지식 베이스: {len(conversation.knowledge_base)} 카테고리, 총 {total_items}개 지식")
        
        # PyTorch 튜터 상태
        if conversation.pytorch_tutor:
            print("✅ PyTorch 학습 도우미 활성화")
        
        print("\n💡 사용 가능한 명령어:")
        print("  - 일반 질문: 'Panda3D 최적화 방법 알려줘'")
        print("  - PyTorch 학습: 'pytorch 텐서 기초 설명해줘'")
        print("  - 게임 AI: '게임에 AI 추가해줘'")
        print("  - 지식 검색: '물리엔진이 뭐야?'")
        
        while True:
            try:
                user_input = input("\n👤 당신: ")
                
                if user_input.lower() in ['종료', 'exit', 'quit']:
                    print("\n👋 대화를 종료합니다. 감사합니다!")
                    break
                
                if not user_input.strip():
                    continue
                
                # 응답 생성
                print("\n🤖 AutoCI: ", end="", flush=True)
                response = await conversation.process_user_input(user_input, evolution)
                
                # 마크다운 렌더링 (간단한 버전)
                response = response.replace("**", "")
                print(response)
                
                # 지식 베이스 검색 결과 표시
                entities = await conversation._extract_entities(user_input)
                results = conversation._search_knowledge_base(user_input, entities)
                if results:
                    print(f"\n   📚 관련 지식 {len(results)}개를 활용했습니다.")
                
            except KeyboardInterrupt:
                print("\n\n대화가 중단되었습니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ 초기화 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(interactive_talk())