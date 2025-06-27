#!/usr/bin/env python3
"""
한국어 AI 시스템 테스트 스크립트
ChatGPT 수준의 자연스러운 한국어 대화 테스트
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from advanced_korean_ai import AdvancedKoreanAI
import time


def test_conversation_scenarios():
    """다양한 대화 시나리오 테스트"""
    print("🤖 AutoCI ChatGPT 수준 한국어 AI 테스트")
    print("=" * 70)
    
    ai = AdvancedKoreanAI()
    
    # 테스트 시나리오
    scenarios = [
        {
            'name': '🌟 인사 및 소개',
            'conversations': [
                "안녕하세요! 처음 뵙겠습니다.",
                "Unity 개발에 도움을 받고 싶어서 왔어요.",
                "저는 Unity 초보자인데 괜찮을까요?"
            ]
        },
        {
            'name': '🐛 에러 해결 요청',
            'conversations': [
                "PlayerController.cs에서 NullReferenceException이 계속 나요 ㅠㅠ",
                "transform.position을 접근할 때 에러가 나는 것 같아요",
                "GameObject가 null인지 체크하는 방법이 뭐에요?"
            ]
        },
        {
            'name': '📚 기술적 질문',
            'conversations': [
                "코루틴이랑 async/await 중에 뭐가 더 좋아?",
                "Unity에서 Object Pool 패턴 구현하는 방법 알려줘",
                "SOLID 원칙이 뭔지 쉽게 설명해줄 수 있어?"
            ]
        },
        {
            'name': '🎯 프로젝트 관리',
            'conversations': [
                "Scripts 폴더가 너무 복잡해졌어요. 어떻게 정리하면 좋을까요?",
                "UI 스크립트들을 어떤 폴더에 넣는게 좋을까요?",
                "게임 매니저 싱글톤 패턴으로 만들어도 될까요?"
            ]
        },
        {
            'name': '😊 감정 표현 및 격려',
            'conversations': [
                "와!! 드디어 버그 해결했어요!! 진짜 감사합니다!!! 😄",
                "하... 계속 에러만 나고 너무 힘들어요... 😢",
                "오늘 하루종일 코딩했는데 진전이 없네요 ㅠㅠ"
            ]
        },
        {
            'name': '🎮 Unity 특화 질문',
            'conversations': [
                "Raycast로 마우스 클릭 감지하는 방법이 뭐야?",
                "Animator Controller에서 파라미터 설정하는 방법 알려줘",
                "Physics.OverlapSphere 사용법이 궁금해"
            ]
        }
    ]
    
    # 각 시나리오 실행
    for scenario in scenarios:
        print(f"\n\n{'='*70}")
        print(f"테스트 시나리오: {scenario['name']}")
        print(f"{'='*70}\n")
        
        for user_input in scenario['conversations']:
            print(f"👤 사용자: {user_input}")
            
            # AI 분석 및 응답
            analysis = ai.analyze_input(user_input)
            response = ai.generate_response(analysis)
            
            # 분석 정보 표시
            print(f"📊 [분석 정보]")
            print(f"   - 의도: {analysis['intent']}")
            print(f"   - 주제: {analysis['topic']}")
            print(f"   - 감정: {analysis['emotion']}")
            print(f"   - 격식: {analysis['formality']}")
            print(f"   - 복잡도: {analysis['complexity']}")
            if analysis['keywords']:
                print(f"   - 키워드: {', '.join(analysis['keywords'][:5])}")
            
            print(f"\n🤖 AutoCI: {response}\n")
            print("-" * 70)
            
            time.sleep(0.5)  # 읽기 쉽도록 짧은 대기
    
    # 대화 요약
    print(f"\n\n{'='*70}")
    print("📊 전체 대화 요약")
    print(f"{'='*70}")
    print(ai.get_conversation_summary())
    
    # 학습 효과 테스트
    print(f"\n\n{'='*70}")
    print("🧠 학습 효과 테스트")
    print(f"{'='*70}")
    
    # 피드백 학습
    ai.learn_from_feedback("정말 도움이 됐어요!", True)
    ai.learn_from_feedback("설명이 부족해요", False)
    
    print("✅ 피드백 학습 완료")
    
    # 반복 질문 테스트 (학습된 패턴 활용)
    print("\n📝 학습된 패턴으로 응답 개선 테스트:")
    repeat_question = "GameObject가 null인지 체크하는 방법이 뭐에요?"
    print(f"👤 사용자: {repeat_question}")
    
    analysis = ai.analyze_input(repeat_question)
    response = ai.generate_response(analysis)
    print(f"🤖 AutoCI: {response}")


def test_edge_cases():
    """엣지 케이스 테스트"""
    print(f"\n\n{'='*70}")
    print("🔧 엣지 케이스 테스트")
    print(f"{'='*70}\n")
    
    ai = AdvancedKoreanAI()
    
    edge_cases = [
        "ㅋㅋㅋㅋㅋ",
        "...",
        "???",
        "안뇽하세요 ㅎㅎ 유니티 초보에욬ㅋㅋ",
        "Transform.position.x += Time.deltaTime * speed;",
        "에러: NullReferenceException: Object reference not set to an instance of an object",
        "😊😊😊",
        "야 이거 왜 안돼",
        "선생님 도와주세요 제발요 ㅠㅠㅠㅠ",
        "unity singleton pattern implementation in c# with thread safety"
    ]
    
    for test_input in edge_cases:
        print(f"👤 입력: {test_input}")
        
        try:
            analysis = ai.analyze_input(test_input)
            response = ai.generate_response(analysis)
            print(f"🤖 응답: {response}")
            print(f"📊 분석: 의도={analysis['intent']}, 감정={analysis['emotion']}")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        print("-" * 50)


def test_context_understanding():
    """문맥 이해 테스트"""
    print(f"\n\n{'='*70}")
    print("🧠 문맥 이해 능력 테스트")
    print(f"{'='*70}\n")
    
    ai = AdvancedKoreanAI()
    
    # 연속된 대화로 문맥 테스트
    context_conversation = [
        "Unity에서 플레이어 이동을 구현하고 싶어요",
        "그거를 코루틴으로 하면 어떨까요?",  # '그거' = 플레이어 이동
        "아니면 Update에서 하는게 나을까요?",
        "성능상으로는 어떤게 더 좋아요?",
        "그럼 그렇게 구현해볼게요. 감사합니다!"
    ]
    
    for user_input in context_conversation:
        print(f"👤 사용자: {user_input}")
        
        analysis = ai.analyze_input(user_input)
        response = ai.generate_response(analysis)
        
        # 문맥 의존성 체크
        if analysis.get('context_needed'):
            print(f"   [문맥 참조 필요: ✓]")
        
        print(f"🤖 AutoCI: {response}\n")
        time.sleep(0.3)


def test_formality_adaptation():
    """격식 적응 테스트"""
    print(f"\n\n{'='*70}")
    print("🎭 격식 수준 적응 테스트")
    print(f"{'='*70}\n")
    
    ai = AdvancedKoreanAI()
    
    formality_tests = [
        ("안녕하십니까. Unity 개발에 대해 여쭤봐도 되겠습니까?", "formal"),
        ("안녕하세요! Unity 질문 좀 해도 될까요?", "polite"),
        ("야 Unity 이거 어떻게 하는거야?", "casual"),
        ("유니티에서 코루틴 사용법 좀 알려주실 수 있나요?", "polite"),
        ("GameObject 찾는 방법 알려줘", "casual")
    ]
    
    for user_input, expected_formality in formality_tests:
        print(f"👤 사용자: {user_input}")
        
        analysis = ai.analyze_input(user_input)
        response = ai.generate_response(analysis)
        
        print(f"   예상 격식: {expected_formality} → 감지된 격식: {analysis['formality']}")
        print(f"🤖 AutoCI: {response}\n")


def main():
    """메인 테스트 실행"""
    print("\n" + "🚀 " * 20)
    print("AutoCI ChatGPT 수준 한국어 AI 시스템 종합 테스트")
    print("🚀 " * 20 + "\n")
    
    # 기본 대화 시나리오
    test_conversation_scenarios()
    
    # 엣지 케이스
    test_edge_cases()
    
    # 문맥 이해
    test_context_understanding()
    
    # 격식 적응
    test_formality_adaptation()
    
    print("\n" + "✅ " * 20)
    print("모든 테스트 완료!")
    print("✅ " * 20 + "\n")


if __name__ == "__main__":
    main()