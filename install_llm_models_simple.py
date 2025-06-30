#!/usr/bin/env python3
"""
간단한 LLM 모델 설치 스크립트
연결 문제를 처리하고 더 작은 모델부터 시작
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Hugging Face 토큰 설정
os.environ["HF_TOKEN"] = "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"

def test_small_model():
    """작은 모델로 테스트"""
    logger.info("작은 모델로 연결 테스트 중...")
    
    try:
        # 매우 작은 모델로 테스트
        model_id = "microsoft/phi-2"
        logger.info(f"테스트 모델: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir="./models/test"
        )
        
        logger.info("✓ 토크나이저 다운로드 성공!")
        
        # 간단한 테스트
        text = "Hello, this is a test"
        inputs = tokenizer(text, return_tensors="pt")
        logger.info(f"✓ 토크나이저 테스트 성공: {inputs.input_ids.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 테스트 실패: {str(e)}")
        return False

def install_codellama_7b():
    """더 작은 CodeLlama 7B 설치"""
    logger.info("\nCodeLlama 7B 설치 시작...")
    
    try:
        model_id = "codellama/CodeLlama-7b-Python-hf"
        cache_dir = "./models/codellama-7b"
        
        logger.info(f"모델 ID: {model_id}")
        logger.info("토크나이저 다운로드 중...")
        
        # 토크나이저만 먼저 다운로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN")
        )
        
        tokenizer.save_pretrained(f"{cache_dir}/tokenizer")
        logger.info("✓ 토크나이저 저장 완료!")
        
        # 모델 정보만 저장 (실제 가중치는 나중에)
        model_info = {
            "model_id": model_id,
            "type": "codellama",
            "size": "7B",
            "status": "tokenizer_only"
        }
        
        import json
        with open(f"{cache_dir}/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
            
        logger.info("✓ CodeLlama 7B 토크나이저 설치 완료!")
        return True
        
    except Exception as e:
        logger.error(f"✗ 설치 실패: {str(e)}")
        return False

def install_tinyllama():
    """매우 작은 TinyLlama 모델 설치 (1.1B)"""
    logger.info("\nTinyLlama 1.1B 설치 시작...")
    
    try:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        cache_dir = "./models/tinyllama"
        
        logger.info(f"모델 ID: {model_id}")
        logger.info("다운로드 중...")
        
        # 토크나이저 다운로드
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        tokenizer.save_pretrained(f"{cache_dir}/tokenizer")
        
        # 모델 다운로드 (작은 모델이므로 가능)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        model.save_pretrained(f"{cache_dir}/model")
        
        logger.info("✓ TinyLlama 설치 완료!")
        
        # 간단한 테스트
        logger.info("모델 테스트 중...")
        inputs = tokenizer("Hello! Write a simple C# function:", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"테스트 응답: {response[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 설치 실패: {str(e)}")
        return False

def create_model_config():
    """모델 설정 파일 생성"""
    config = {
        "installed_models": {
            "tinyllama": {
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "path": "./models/tinyllama",
                "size": "1.1B",
                "features": ["general", "code"],
                "status": "ready"
            },
            "codellama-7b": {
                "model_id": "codellama/CodeLlama-7b-Python-hf",
                "path": "./models/codellama-7b",
                "size": "7B",
                "features": ["code", "csharp"],
                "status": "tokenizer_only"
            }
        },
        "download_settings": {
            "retry_count": 3,
            "timeout": 300,
            "use_cache": True
        }
    }
    
    import json
    with open("./models/model_config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    logger.info("✓ 모델 설정 파일 생성 완료!")

def main():
    """메인 함수"""
    logger.info("=== LLM 모델 간단 설치 시작 ===")
    
    # 모델 디렉토리 생성
    os.makedirs("./models", exist_ok=True)
    
    # GPU 확인
    if torch.cuda.is_available():
        logger.info(f"✓ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("⚠ GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
    
    # 작은 모델로 연결 테스트
    if test_small_model():
        logger.info("\n✓ 연결 테스트 성공!")
    else:
        logger.error("\n✗ 연결 테스트 실패. 네트워크를 확인하세요.")
        return
    
    # TinyLlama 설치 (매우 작은 모델)
    if install_tinyllama():
        logger.info("\n✓ TinyLlama 설치 성공!")
    
    # CodeLlama 7B 토크나이저만 설치
    if install_codellama_7b():
        logger.info("\n✓ CodeLlama 7B 토크나이저 설치 성공!")
    
    # 설정 파일 생성
    create_model_config()
    
    logger.info("\n=== 설치 완료 ===")
    logger.info("설치된 모델:")
    logger.info("1. TinyLlama 1.1B - 완전 설치 (테스트용)")
    logger.info("2. CodeLlama 7B - 토크나이저만 설치")
    logger.info("\n큰 모델은 네트워크가 안정적일 때 다시 시도하세요.")
    logger.info("사용: python continuous_learning_system_simple.py")

if __name__ == "__main__":
    main()