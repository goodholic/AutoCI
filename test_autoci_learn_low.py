#!/usr/bin/env python3
"""
autoci learn low 명령어 테스트 스크립트
현재 설치된 모델로 작동하는지 확인
"""

import sys
import json
import asyncio
from pathlib import Path

# continuous_learning_system 임포트
try:
    from continuous_learning_system import ContinuousLearningSystem
    print("✅ continuous_learning_system 모듈 로드 성공")
except ImportError as e:
    print(f"❌ 모듈 로드 실패: {e}")
    sys.exit(1)

async def test_autoci_learn_low():
    """autoci learn low 시뮬레이션 테스트"""
    print("🧪 autoci learn low 테스트 시작")
    print("=" * 50)
    
    try:
        # ContinuousLearningSystem 초기화 (RTX 2080 설정)
        system = ContinuousLearningSystem(
            models_dir="./models",
            learning_dir="./continuous_learning", 
            max_memory_gb=24.0  # RTX 2080 + 32GB RAM 최적화
        )
        
        print("🔍 모델 정보 로드 중...")
        system.load_model_info()
        
        # DeepSeek-coder 상태 우선 확인
        deepseek_status = system.available_models.get("deepseek-coder-7b", {}).get('status', 'not_found')
        if deepseek_status == 'installed':
            print("🔥 DeepSeek-coder-v2 6.7B: 설치됨 ✅")
            print("   → 5가지 핵심 주제 최적화 학습 가능!")
        else:
            print("⚠️  DeepSeek-coder-v2 6.7B: 미설치 ❌")
            print("   → python download_deepseek_coder.py로 설치하세요")
        
        print("\n📋 전체 모델 목록:")
        for model_name, info in system.available_models.items():
            status = info.get('status', 'unknown')
            rtx_opt = info.get('rtx_2080_optimized', False)
            
            # DeepSeek-coder 강조
            if model_name == "deepseek-coder-7b":
                prefix = "🔥 [5가지 핵심 주제 특화]"
            elif rtx_opt:
                prefix = "🎯 [RTX 2080 최적화]"
            else:
                prefix = "⚠️"
            
            print(f"  - {prefix} {model_name}: {status}")
        
        # 설치된 모델 찾기
        installed_models = [
            name for name, info in system.available_models.items() 
            if info.get('status') == 'installed'
        ]
        
        if not installed_models:
            print("❌ 설치된 모델이 없습니다!")
            print("💡 먼저 다음 명령어를 실행하세요:")
            print("   python install_llm_models_rtx2080.py")
            return False
        
        print(f"✅ 설치된 모델: {', '.join(installed_models)}")
        
        # 첫 번째 설치된 모델로 테스트
        test_model = installed_models[0]
        print(f"🧪 {test_model} 모델로 테스트...")
        
        # 모델 로드 테스트
        if system.load_model(test_model):
            print(f"✅ {test_model} 모델 로드 성공!")
        else:
            print(f"❌ {test_model} 모델 로드 실패")
            return False
        
        # 간단한 질문 생성 및 답변 테스트
        print("💭 테스트 질문 생성 중...")
        
        test_question = {
            "id": "test_001",
            "topic": "C# 기초",
            "question": "C#에서 async/await를 간단히 설명해주세요.",
            "language": "korean",
            "type": "explain",
            "difficulty": 2
        }
        
        print(f"❓ 질문: {test_question['question']}")
        
        # 모델에게 질문
        answer = await system.ask_model(test_model, test_question)
        
        if answer and answer.get('success', False):
            response = answer.get('response', 'No response')
            print("✅ 답변 생성 성공!")
            print("📝 답변 미리보기:")
            print("-" * 40)
            print(response[:300] + "..." if len(response) > 300 else response)
            print("-" * 40)
        else:
            print("❌ 답변 생성 실패")
            print(f"오류: {answer.get('error', 'Unknown error')}")
            return False
        
        print("\n🎉 autoci learn low 기본 기능 테스트 완료!")
        print("✅ 현재 시스템은 정상 작동합니다.")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_deepseek_availability():
    """DeepSeek-coder 모델 사용 가능성 확인"""
    print("\n🔍 DeepSeek-coder 사용 가능성 확인...")
    
    models_file = Path("models/installed_models.json")
    if not models_file.exists():
        print("❌ models/installed_models.json 파일이 없습니다.")
        return False
    
    try:
        with open(models_file, 'r', encoding='utf-8') as f:
            models = json.load(f)
        
        deepseek_info = models.get('deepseek-coder-7b', {})
        status = deepseek_info.get('status', 'unknown')
        
        print(f"📋 DeepSeek-coder 상태: {status}")
        
        if status == 'installed':
            print("✅ DeepSeek-coder가 이미 설치되어 있습니다!")
            return True
        elif status == 'not_downloaded':
            print("⏳ DeepSeek-coder가 설정되어 있지만 다운로드되지 않았습니다.")
            print("\n💡 다운로드 방법:")
            print("1. 수동 다운로드 (권장):")
            print("   python download_deepseek_coder.py")
            print("\n2. 가상환경에서 직접:")
            print("   source autoci_env/bin/activate  # Linux/WSL")
            print("   autoci_env\\Scripts\\activate     # Windows")
            print("   pip install transformers torch")
            print("   python download_deepseek_coder.py")
            return False
        else:
            print("❓ DeepSeek-coder 상태를 확인할 수 없습니다.")
            return False
            
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🎯 AutoCI Learn Low 상태 확인")
    print("=" * 50)
    
    # 1. 기본 기능 테스트
    print("1️⃣ 기본 기능 테스트")
    success = asyncio.run(test_autoci_learn_low())
    
    if success:
        print("\n✅ autoci learn low는 현재 설치된 모델로 정상 작동합니다!")
    else:
        print("\n❌ autoci learn low에 문제가 있습니다.")
    
    # 2. DeepSeek-coder 확인
    print("\n2️⃣ DeepSeek-coder 확인")
    deepseek_ready = check_deepseek_availability()
    
    # 3. 권장사항 출력
    print("\n📋 권장사항:")
    if success and deepseek_ready:
        print("🎉 모든 준비가 완료되었습니다!")
        print("👉 autoci learn low 명령어를 실행하세요.")
    elif success and not deepseek_ready:
        print("✅ 기본 기능은 작동합니다.")
        print("💡 DeepSeek-coder를 설치하면 더 나은 코딩 답변을 받을 수 있습니다.")
        print("👉 python download_deepseek_coder.py")
    else:
        print("❌ 먼저 모델을 설치해야 합니다.")
        print("👉 python install_llm_models_rtx2080.py")
    
    return success

if __name__ == "__main__":
    main() 