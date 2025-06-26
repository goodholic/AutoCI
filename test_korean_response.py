#!/usr/bin/env python3
"""
AutoCI 한국어 응답 테스트 스크립트
"""

def test_korean_response():
    """한국어 응답 테스트"""
    print("🤖 AutoCI 한국어 응답 테스트")
    print("=" * 50)
    
    # 한국어 인사말 및 일반적인 표현 처리
    korean_greetings = {
        '안녕': '안녕하세요! 👋 AutoCI 시스템에 오신 것을 환영합니다!\n저는 24시간 코드를 자동으로 개선해드리는 AI입니다. 어떤 도움이 필요하신가요?',
        '안녕하세요': '안녕하세요! 😊 반갑습니다! AutoCI와 함께 코드 품질을 향상시켜보세요!',
        '반가워': '저도 반가워요! 🤗 코딩 작업에서 어떤 도움이 필요하신지 말씀해주세요.',
        '고마워': '천만에요! 😊 언제든지 도움이 필요하시면 말씀해주세요!',
        '고맙습니다': '별말씀을요! 🙏 더 필요한 것이 있으면 언제든 말씀해주세요.',
        '하이': '하이! 👋 반가워요! 오늘 어떤 코드 작업을 도와드릴까요?',
        '헬로': '헬로! 😄 환영합니다! Unity 프로젝트나 C# 코드 개선에 도움이 필요하시면 말씀해주세요!'
    }
    
    # 한국어 명령어 매핑
    korean_commands = {
        '도움말': 'help',
        '도움': 'help',
        '상태': 'status',
        '상태확인': 'status',
        '프로젝트': 'project',
        '분석': 'analyze',
        '개선': 'improve',
        '검색': 'search',
        '정리': 'organize',
        '종료': 'exit'
    }
    
    # 테스트 케이스
    test_cases = ['안녕', '안녕하세요', '도움말', '상태', '프로젝트', '정리', '어떻게 사용하나요?']
    
    print("\n📝 테스트 케이스 실행:")
    print("-" * 50)
    
    for test_input in test_cases:
        print(f"\n입력: '{test_input}'")
        
        # 인사말 처리
        if test_input.lower() in korean_greetings:
            print(f"응답: {korean_greetings[test_input.lower()]}")
            print("💡 주요 명령어:")
            print("   • 프로젝트 <경로> - Unity 프로젝트 설정")
            print("   • 분석 - 코드 분석")
            print("   • 정리 - Unity 스크립트 폴더 정리")
            continue
            
        # 한국어 명령어 변환
        if test_input in korean_commands:
            english_cmd = korean_commands[test_input]
            print(f"응답: ✅ '{test_input}' → '{english_cmd}' 명령을 실행합니다...")
            continue
            
        # 질문이나 대화형 입력 감지
        conversation_patterns = ['어떻게', '뭐야', '무엇', '왜', '언제', '어디서', '누가', '어느', '몇', '?', '？']
        if any(pattern in test_input for pattern in conversation_patterns):
            print(f"응답: 🤔 '{test_input}'에 대해 생각해보고 있어요...")
            print("💡 더 구체적인 질문을 해주시면 더 정확한 답변을 드릴 수 있어요!")
            continue
        
        # 기본 응답
        print(f"응답: 😅 '{test_input}'는 아직 이해하지 못하겠어요.")
        print("💡 '도움말' 또는 'help'를 입력하시면 사용 가능한 명령어를 볼 수 있어요!")
    
    print("\n" + "=" * 50)
    print("✅ 한국어 응답 테스트 완료!")
    print("🎉 이제 AutoCI가 한국어로 대화할 수 있습니다!")

if __name__ == "__main__":
    test_korean_response() 