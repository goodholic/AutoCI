#!/usr/bin/env python3
"""
autoci learn 테스트 스크립트
"""

import sys
import os

# PYTHONPATH에 현재 디렉토리 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔍 모듈 임포트 테스트...")

# 1. 정보 수집기 테스트
try:
    from modules.intelligent_information_gatherer import IntelligentInformationGatherer
    print("✅ 정보 수집기 임포트 성공")
except Exception as e:
    print(f"❌ 정보 수집기 임포트 실패: {e}")

# 2. AI 모델 컨트롤러 테스트
try:
    from modules.ai_model_controller import AIModelController
    print("✅ AI 모델 컨트롤러 임포트 성공")
except Exception as e:
    print(f"❌ AI 모델 컨트롤러 임포트 실패: {e}")

# 3. PyTorch 모듈 테스트
try:
    from modules.pytorch_deep_learning_module import AutoCIPyTorchLearningSystem
    print("✅ PyTorch 딥러닝 모듈 임포트 성공")
except Exception as e:
    print(f"❌ PyTorch 딥러닝 모듈 임포트 실패: {e}")

# 4. 모델 경로 확인
print("\n📁 모델 경로 확인:")
model_paths = [
    "./models/deepseek-coder-7b/model",
    "./models/llama-3.1-8b/model_files",
    "./models/codellama-13b"
]

for path in model_paths:
    if os.path.exists(path):
        print(f"✅ {path} 존재")
    else:
        print(f"❌ {path} 없음")

# 5. 필수 패키지 확인
print("\n📦 필수 패키지 확인:")
packages = ["torch", "transformers", "accelerate", "psutil", "numpy"]

for package in packages:
    try:
        __import__(package)
        print(f"✅ {package} 설치됨")
    except ImportError:
        print(f"❌ {package} 설치 필요")

print("\n테스트 완료!")