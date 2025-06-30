#!/usr/bin/env python3
"""
데모용 모델 설정 스크립트
네트워크 문제를 우회하고 시뮬레이션 모드 제공
"""

import os
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_setup():
    """데모용 설정 생성"""
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    # 설치된 모델 정보 (시뮬레이션)
    installed_models = {
        "llama-3.1-8b": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "path": str(models_dir / "llama-3.1-8b"),
            "quantization": "8bit",
            "features": ["general", "korean", "csharp"],
            "installed_at": "2025-06-30T04:00:00",
            "size_gb": 16,
            "status": "demo_mode"
        },
        "codellama-13b": {
            "model_id": "codellama/CodeLlama-13b-Instruct-hf",
            "path": str(models_dir / "codellama-13b"),
            "quantization": "8bit",
            "features": ["code", "csharp", "godot"],
            "installed_at": "2025-06-30T04:00:00",
            "size_gb": 26,
            "status": "demo_mode"
        },
        "qwen2.5-coder-32b": {
            "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "path": str(models_dir / "qwen2.5-coder-32b"),
            "quantization": "4bit",
            "features": ["code", "korean", "csharp", "advanced"],
            "installed_at": "2025-06-30T04:00:00",
            "size_gb": 64,
            "status": "demo_mode"
        }
    }
    
    # 모델 정보 저장
    with open(models_dir / "installed_models.json", "w", encoding='utf-8') as f:
        json.dump(installed_models, f, indent=2, ensure_ascii=False)
        
    logger.info("✓ 데모 모델 설정 생성 완료")
    
    # 각 모델 디렉토리 생성
    for model_name, info in installed_models.items():
        model_path = Path(info["path"])
        model_path.mkdir(exist_ok=True)
        
        # 더미 토크나이저 설정
        tokenizer_path = model_path / "tokenizer"
        tokenizer_path.mkdir(exist_ok=True)
        
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizer",
            "model_max_length": 2048,
            "padding_side": "left",
            "demo_mode": True
        }
        
        with open(tokenizer_path / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)
            
        logger.info(f"  - {model_name} 디렉토리 생성")
        
    # 연속 학습 시스템 설정
    learning_config = {
        "mode": "demo",
        "models_available": list(installed_models.keys()),
        "learning_topics": {
            "csharp_basics": ["변수", "타입", "메서드", "클래스", "상속"],
            "korean_terms": ["변수", "함수", "클래스", "객체", "상속"],
            "godot_integration": ["노드", "시그널", "씬", "스크립트", "물리"],
        },
        "demo_responses": {
            "explain": "이것은 {topic}에 대한 설명입니다. 실제 모델이 설치되면 더 자세한 답변을 제공합니다.",
            "example": "```csharp\n// {topic} 예제 코드\npublic class Example {{\n    // 실제 구현은 모델 설치 후\n}}\n```",
            "translate": "{keyword}는 한글로 '{korean}'이며, C#에서 중요한 개념입니다.",
        }
    }
    
    with open(models_dir / "learning_config.json", "w", encoding='utf-8') as f:
        json.dump(learning_config, f, indent=2, ensure_ascii=False)
        
    logger.info("✓ 학습 시스템 설정 완료")
    
    # 사용 안내
    logger.info("\n=== 데모 설정 완료 ===")
    logger.info("\n실제 모델 다운로드는 네트워크 문제로 인해 실패했지만,")
    logger.info("데모 모드로 시스템을 테스트할 수 있습니다.")
    logger.info("\n사용 방법:")
    logger.info("1. 데모 학습 시스템 실행:")
    logger.info("   python continuous_learning_demo.py")
    logger.info("\n2. 실제 모델 설치 (네트워크 안정 시):")
    logger.info("   python install_llm_models_robust.py")
    logger.info("\n현재 설정:")
    logger.info(f"- 모델 디렉토리: {models_dir}")
    logger.info(f"- 데모 모델: {list(installed_models.keys())}")
    
    # README 업데이트 제안
    logger.info("\n참고: README.md에 네트워크 문제 해결 방법을 추가했습니다.")

if __name__ == "__main__":
    create_demo_setup()