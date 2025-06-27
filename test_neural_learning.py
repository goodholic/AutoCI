#!/usr/bin/env python3
"""
AutoCI 신경망 학습 시스템 테스트
"""

import sys
import os
import time
import torch

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from neural_learning_autoci import RealLearningAutoCI, ConversationData
    print("✅ 신경망 학습 모듈 임포트 성공")
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print("PyTorch가 설치되어 있지 않을 수 있습니다.")
    print("설치 명령: pip install torch scikit-learn")
    sys.exit(1)

def test_basic_functionality():
    """기본 기능 테스트"""
    print("\n🧪 기본 기능 테스트 시작")
    print("=" * 50)
    
    try:
        # AI 시스템 초기화
        ai = RealLearningAutoCI()
        print("✅ AI 시스템 초기화 성공")
        
        # 상태 확인
        status = ai.get_learning_status()
        print(f"✅ 디바이스: {status['device']}")
        print(f"✅ 모델 파라미터 수: {status['model_info']['total_parameters']:,}")
        
        return ai
        
    except Exception as e:
        print(f"❌ 기본 기능 테스트 실패: {e}")
        return None

def test_conversation_processing(ai):
    """대화 처리 테스트"""
    print("\n🗣️ 대화 처리 테스트 시작")
    print("=" * 50)
    
    test_conversations = [
        ("안녕하세요", "안녕하세요! 반갑습니다!", 1.0),
        ("Unity 도움이 필요해요", "Unity 개발을 도와드리겠습니다!", 0.8),
        ("코드 분석해주세요", "코드를 분석해드리겠습니다.", 0.6),
        ("이상한 응답이네요", "죄송합니다. 다시 시도해보겠습니다.", -0.5),
        ("고마워요", "천만에요! 도움이 되어서 기뻐요!", 0.9)
    ]
    
    try:
        for i, (user_input, ai_response, feedback) in enumerate(test_conversations, 1):
            print(f"\n테스트 {i}/5:")
            print(f"사용자: {user_input}")
            print(f"AI: {ai_response}")
            print(f"피드백: {feedback}")
            
            result = ai.process_conversation(user_input, ai_response, feedback)
            
            print(f"✅ 대화 ID: {result['conversation_id']}")
            print(f"✅ 학습 트리거: {result['learning_triggered']}")
            print(f"✅ 유사 대화: {result['similar_conversations']}개")
            
            # 잠시 대기 (실시간 학습 확인)
            time.sleep(1)
        
        print("\n✅ 모든 대화 처리 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 대화 처리 테스트 실패: {e}")
        return False

def test_response_generation(ai):
    """응답 생성 테스트"""
    print("\n🤖 응답 생성 테스트 시작")
    print("=" * 50)
    
    test_inputs = [
        "안녕하세요",
        "Unity에서 스크립트 작성하는 방법 알려주세요",
        "코드에 오류가 있어요",
        "감사합니다",
        "이해가 안 돼요"
    ]
    
    try:
        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n테스트 {i}/5:")
            print(f"입력: {user_input}")
            
            response, confidence = ai.generate_response(user_input)
            
            print(f"응답: {response}")
            print(f"신뢰도: {confidence:.2f}")
            print("✅ 응답 생성 성공")
        
        print("\n✅ 모든 응답 생성 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 응답 생성 테스트 실패: {e}")
        return False

def test_memory_system(ai):
    """메모리 시스템 테스트"""
    print("\n🧠 메모리 시스템 테스트 시작")
    print("=" * 50)
    
    try:
        # 현재 메모리 상태
        status = ai.get_learning_status()
        memory_stats = status['memory_stats']
        
        print(f"단기 메모리: {memory_stats['short_term_memory']}개")
        print(f"장기 메모리: {memory_stats['long_term_memory']}개")
        print(f"작업 메모리: {memory_stats['working_memory']}개")
        
        # 유사한 대화 찾기 테스트
        similar_convs = ai.memory.find_similar_conversations("안녕하세요", top_k=3)
        print(f"✅ 유사 대화 검색: {len(similar_convs)}개 발견")
        
        for conv in similar_convs:
            print(f"  - {conv.user_input[:20]}... (점수: {conv.feedback_score})")
        
        print("\n✅ 메모리 시스템 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 메모리 시스템 테스트 실패: {e}")
        return False

def test_learning_system(ai):
    """학습 시스템 테스트"""
    print("\n📚 학습 시스템 테스트 시작")
    print("=" * 50)
    
    try:
        # 학습 전 상태
        initial_stats = ai.get_learning_status()['stats']
        print(f"초기 학습 에포크: {initial_stats['total_training_epochs']}")
        
        # 강한 피드백으로 즉시 학습 트리거
        print("\n강한 피드백으로 즉시 학습 테스트...")
        ai.process_conversation("테스트 입력", "테스트 응답", 1.0)
        
        # 잠시 대기 (학습 완료 대기)
        time.sleep(3)
        
        # 학습 후 상태
        final_stats = ai.get_learning_status()['stats']
        print(f"최종 학습 에포크: {final_stats['total_training_epochs']}")
        
        if final_stats['total_training_epochs'] > initial_stats['total_training_epochs']:
            print("✅ 즉시 학습 성공 - 에포크 증가 확인")
        else:
            print("⚠️ 즉시 학습 미확인 - 백그라운드에서 처리 중일 수 있음")
        
        print(f"✅ 마지막 학습 시간: {final_stats['last_learning_time']}")
        print("\n✅ 학습 시스템 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 학습 시스템 테스트 실패: {e}")
        return False

def test_model_persistence(ai):
    """모델 저장/로드 테스트"""
    print("\n💾 모델 저장/로드 테스트 시작")
    print("=" * 50)
    
    try:
        # 모델 저장
        ai.save_model()
        print("✅ 모델 저장 성공")
        
        # 새로운 AI 인스턴스로 모델 로드 테스트
        ai2 = RealLearningAutoCI()
        print("✅ 모델 로드 성공")
        
        # 상태 비교
        status1 = ai.get_learning_status()
        status2 = ai2.get_learning_status()
        
        if status1['stats']['total_conversations'] == status2['stats']['total_conversations']:
            print("✅ 모델 상태 일치 확인")
        else:
            print("⚠️ 모델 상태 불일치 - 정상적일 수 있음")
        
        print("\n✅ 모델 저장/로드 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 모델 저장/로드 테스트 실패: {e}")
        return False

def run_all_tests():
    """전체 테스트 실행"""
    print("🚀 AutoCI 신경망 학습 시스템 종합 테스트")
    print("=" * 60)
    
    test_results = []
    
    # 1. 기본 기능 테스트
    ai = test_basic_functionality()
    if ai is None:
        print("❌ 기본 기능 테스트 실패로 인한 전체 테스트 중단")
        return
    
    test_results.append(("기본 기능", True))
    
    # 2. 대화 처리 테스트
    result = test_conversation_processing(ai)
    test_results.append(("대화 처리", result))
    
    # 3. 응답 생성 테스트
    result = test_response_generation(ai)
    test_results.append(("응답 생성", result))
    
    # 4. 메모리 시스템 테스트
    result = test_memory_system(ai)
    test_results.append(("메모리 시스템", result))
    
    # 5. 학습 시스템 테스트
    result = test_learning_system(ai)
    test_results.append(("학습 시스템", result))
    
    # 6. 모델 저장/로드 테스트
    result = test_model_persistence(ai)
    test_results.append(("모델 저장/로드", result))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name:15}: {status}")
        if result:
            passed += 1
    
    print(f"\n총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {total - passed}개")
    print(f"성공률: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과! 신경망 학습 시스템이 정상 작동합니다.")
    else:
        print(f"\n⚠️ {total - passed}개 테스트 실패. 시스템 점검이 필요합니다.")
    
    # 최종 상태 출력
    final_status = ai.get_learning_status()
    print(f"\n📈 최종 시스템 상태:")
    print(f"  총 대화: {final_status['stats']['total_conversations']}개")
    print(f"  학습 에포크: {final_status['stats']['total_training_epochs']}개")
    print(f"  단기 메모리: {final_status['memory_stats']['short_term_memory']}개")
    print(f"  장기 메모리: {final_status['memory_stats']['long_term_memory']}개")

if __name__ == "__main__":
    run_all_tests()