#!/usr/bin/env python3
"""
AutoCI 연속 학습 시스템 테스트 스크립트
"""

import sys
import time
import json
from pathlib import Path

def test_simple_learning():
    """간단한 학습 시스템 테스트"""
    print("🧪 AutoCI 연속 학습 시스템 테스트")
    print("=" * 50)
    
    try:
        # 간단한 학습 시스템 테스트
        from autoci_simple_continuous_learning import SimpleContinuousLearningAI
        
        # AI 초기화
        learning_ai = SimpleContinuousLearningAI()
        
        print("✅ SimpleContinuousLearningAI 초기화 성공")
        
        # 잠깐 학습 시작
        print("🚀 5초간 테스트 학습 시작...")
        learning_ai.start_continuous_learning()
        
        # 5초 대기
        for i in range(5):
            time.sleep(1)
            print(f"   {5-i}초 남음...")
        
        # 상태 확인
        status = learning_ai.get_learning_status()
        print("\n📊 학습 상태:")
        print(f"🔄 활성: {'✅' if status['learning_active'] else '❌'}")
        print(f"🧠 세션: {status['learning_stats']['sessions_completed']}")
        print(f"📚 지식량: {status['knowledge_base_size']['total']}")
        
        # 학습 중지
        learning_ai.stop_continuous_learning()
        print("✅ 테스트 완료!")
        
        return True
        
    except ImportError as e:
        print(f"❌ 모듈 임포트 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

def test_background_manager():
    """백그라운드 관리자 테스트"""
    print("\n🧪 백그라운드 관리자 테스트")
    print("=" * 30)
    
    try:
        from start_continuous_learning import AutoCIBackgroundLearning
        
        manager = AutoCIBackgroundLearning()
        print("✅ AutoCIBackgroundLearning 초기화 성공")
        
        # 상태 확인만
        print("📊 현재 상태:")
        is_running = manager.is_running()
        print(f"🔄 실행 중: {'✅' if is_running else '❌'}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 모듈 임포트 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

def test_file_existence():
    """필요한 파일들 존재 확인"""
    print("\n🧪 파일 존재 확인")
    print("=" * 20)
    
    files = [
        "autoci_continuous_learning.py",
        "autoci_simple_continuous_learning.py", 
        "start_continuous_learning.py",
        "README_continuous_learning.md"
    ]
    
    all_exist = True
    for file in files:
        exists = Path(file).exists()
        status = "✅" if exists else "❌"
        print(f"{status} {file}")
        if not exists:
            all_exist = False
    
    return all_exist

def main():
    """메인 테스트 함수"""
    print("🚀 AutoCI 24시간 연속 학습 시스템 테스트")
    print("=" * 60)
    
    results = []
    
    # 1. 파일 존재 확인
    print("\n1️⃣ 파일 존재 확인")
    results.append(test_file_existence())
    
    # 2. 백그라운드 관리자 테스트
    print("\n2️⃣ 백그라운드 관리자 테스트")
    results.append(test_background_manager())
    
    # 3. 간단한 학습 시스템 테스트
    print("\n3️⃣ 간단한 학습 시스템 테스트")
    results.append(test_simple_learning())
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 테스트 결과 요약")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ 통과: {passed}/{total}")
    print(f"❌ 실패: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과!")
        print("💡 이제 다음 명령어로 실제 시스템을 사용할 수 있습니다:")
        print("   python autoci_simple_continuous_learning.py")
        print("   python start_continuous_learning.py help")
    else:
        print("\n⚠️  일부 테스트 실패")
        print("💡 실패한 기능은 의존성 설치 후 다시 시도해주세요.")
    
    print("\n🔗 사용법:")
    print("   ./autoci learn start     # 백그라운드 학습 시작")
    print("   ./autoci learn status    # 학습 상태 확인")
    print("   ./autoci learn simple    # 간단한 대화형 버전")
    print("   ./autoci learn help      # 자세한 도움말")

if __name__ == "__main__":
    main() 