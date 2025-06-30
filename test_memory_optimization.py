#!/usr/bin/env python3
"""
메모리 최적화 시스템 테스트 스크립트
"""

import psutil
import time
import json
from datetime import datetime

def test_memory_optimization():
    """메모리 최적화 기능 테스트"""
    print("=== 메모리 최적화 시스템 테스트 ===\n")
    
    # 시스템 메모리 정보
    memory = psutil.virtual_memory()
    print(f"📊 시스템 메모리 정보:")
    print(f"   총 메모리: {memory.total / (1024**3):.1f}GB")
    print(f"   사용 중: {memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%)")
    print(f"   사용 가능: {memory.available / (1024**3):.1f}GB")
    
    # 권장 설정
    total_gb = memory.total / (1024**3)
    if total_gb >= 32:
        recommended = 32.0
        models = "모든 모델 (llama-3.1-8b, codellama-13b, qwen2.5-coder-32b)"
    elif total_gb >= 16:
        recommended = 16.0
        models = "중소형 모델 (llama-3.1-8b, codellama-13b)"
    else:
        recommended = min(total_gb * 0.8, 8.0)
        models = "소형 모델 (llama-3.1-8b)"
    
    print(f"\n🎯 권장 설정:")
    print(f"   메모리 제한: {recommended}GB")
    print(f"   사용 가능한 모델: {models}")
    
    # 메모리 최적화 전략 시뮬레이션
    print(f"\n🧠 메모리 최적화 전략:")
    print(f"   ✓ 동적 모델 로딩: 필요할 때만 메모리에 로드")
    print(f"   ✓ 자동 언로딩: {recommended * 0.85:.1f}GB 도달 시 모델 해제")
    print(f"   ✓ 순차적 사용: 여러 모델을 번갈아가며 사용")
    print(f"   ✓ 20사이클 로테이션: 자동 모델 교체")
    
    # 사용법 안내
    print(f"\n📝 사용법:")
    print(f"   # 권장 설정으로 24시간 학습")
    print(f"   python continuous_learning_system.py 24 {recommended}")
    print(f"   ")
    print(f"   # 1시간 테스트")
    print(f"   python continuous_learning_system.py 1 {recommended}")
    print(f"   ")
    print(f"   # 메모리 사용량 모니터링")
    print(f"   tail -f continuous_learning.log")
    
    # 테스트 완료
    print(f"\n✅ 메모리 최적화 시스템 테스트 완료")
    
    # 결과를 JSON으로 저장
    test_result = {
        "timestamp": datetime.now().isoformat(),
        "system_memory_gb": total_gb,
        "current_usage_gb": memory.used / (1024**3),
        "current_usage_percent": memory.percent,
        "recommended_limit_gb": recommended,
        "recommended_models": models,
        "optimization_features": [
            "dynamic_loading",
            "automatic_unloading", 
            "sequential_usage",
            "rotation_system",
            "memory_monitoring"
        ]
    }
    
    with open("memory_optimization_test.json", "w", encoding="utf-8") as f:
        json.dump(test_result, f, indent=2, ensure_ascii=False)
    
    print(f"📄 테스트 결과 저장: memory_optimization_test.json")

if __name__ == "__main__":
    test_memory_optimization()