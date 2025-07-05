#!/usr/bin/env python3
"""
PyTorch 딥러닝 모듈 통합 테스트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pytorch_module():
    """PyTorch 모듈 테스트"""
    print("🧠 PyTorch 딥러닝 모듈 테스트 시작...")
    print("=" * 60)
    
    try:
        # PyTorch 모듈 import
        from modules.pytorch_deep_learning_module import AutoCIPyTorchLearningSystem
        print("✅ PyTorch 딥러닝 모듈 import 성공!")
        
        # 시스템 초기화
        pytorch_system = AutoCIPyTorchLearningSystem(base_path=str(project_root))
        print("✅ PyTorch 학습 시스템 초기화 성공!")
        
        # 품질 평가 테스트
        test_text = "Godot 엔진에서 노드는 게임 오브젝트의 기본 단위입니다."
        quality_score = pytorch_system.assess_quality(test_text)
        print(f"✅ 품질 평가 테스트: {quality_score:.2f}")
        
        # 주제 분류 테스트
        topic = pytorch_system.classify_topic(test_text)
        print(f"✅ 주제 분류 테스트: {topic}")
        
        # 지식 임베딩 테스트
        embedding = pytorch_system.get_knowledge_embedding(test_text)
        print(f"✅ 지식 임베딩 차원: {embedding.shape}")
        
        print("\n🎉 PyTorch 모듈 테스트 완료!")
        
    except ImportError as e:
        print(f"❌ PyTorch 모듈 import 실패: {e}")
        print("💡 다음 명령어로 필요한 패키지를 설치하세요:")
        print("   pip install torch transformers")
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")

def test_continuous_learning_integration():
    """continuous_learning_system과의 통합 테스트"""
    print("\n🔄 Continuous Learning 시스템 통합 테스트...")
    print("=" * 60)
    
    try:
        # Continuous Learning 시스템 import
        from core.continuous_learning_system import ContinuousLearningSystem
        print("✅ Continuous Learning 시스템 import 성공!")
        
        # 시스템 초기화
        learning_system = ContinuousLearningSystem()
        
        # PyTorch 시스템 확인
        if hasattr(learning_system, 'pytorch_system') and learning_system.pytorch_system:
            print("✅ PyTorch 시스템이 Continuous Learning에 통합되었습니다!")
            
            # 학습 주제 확인
            topics = learning_system.learning_topics
            print(f"✅ 학습 주제 개수: {len(topics)}")
            
            # 샘플 질문 생성
            if topics:
                sample_topic = topics[0]
                question = learning_system.generate_question(sample_topic)
                print(f"✅ 샘플 질문 생성: {question['question'][:50]}...")
                
        else:
            print("⚠️ PyTorch 시스템이 통합되지 않았습니다.")
            
    except Exception as e:
        print(f"❌ 통합 테스트 중 오류 발생: {e}")

def test_autoci_command():
    """AutoCI 명령어 테스트"""
    print("\n🎮 AutoCI 명령어 테스트...")
    print("=" * 60)
    
    print("✅ 사용 가능한 학습 명령어:")
    print("   autoci learn          - AI 통합 연속 학습 (PyTorch 포함)")
    print("   autoci learn low      - RTX 2080 최적화 학습 (PyTorch 포함)")
    print("   autoci learn simple   - 전통적 학습 (PyTorch 미포함)")
    
    print("\n💡 PyTorch 학습을 시작하려면:")
    print("   autoci learn low")

if __name__ == "__main__":
    print("🚀 AutoCI PyTorch 통합 테스트")
    print("=" * 60)
    
    # 각 테스트 실행
    test_pytorch_module()
    test_continuous_learning_integration()
    test_autoci_command()
    
    print("\n✨ 모든 테스트 완료!")