#!/usr/bin/env python3
"""
강력한 LLM 모델 설치 스크립트
재시도 로직과 부분 다운로드 지원
"""

import os
import sys
import json
import time
import torch
from pathlib import Path
from typing import Dict, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import logging
from huggingface_hub import snapshot_download, HfApi

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hugging Face 토큰
HF_TOKEN = "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"
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# 모델 설정
MODEL_CONFIGS = {
    "llama-3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "quantization": "8bit",
        "memory_gb": 16,
        "download_strategy": "snapshot"
    },
    "codellama-13b": {
        "model_id": "codellama/CodeLlama-13b-Instruct-hf",
        "quantization": "8bit", 
        "memory_gb": 26,
        "download_strategy": "files"
    },
    "qwen2.5-coder-32b": {
        "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "quantization": "4bit",
        "memory_gb": 64,
        "download_strategy": "snapshot"
    }
}

class RobustModelInstaller:
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.api = HfApi()
        
    def check_gpu_memory(self) -> float:
        """GPU 메모리 확인"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0
        
    def download_with_retry(self, repo_id: str, local_dir: str, max_retries: int = 3):
        """재시도 로직이 있는 다운로드"""
        for attempt in range(max_retries):
            try:
                logger.info(f"다운로드 시도 {attempt + 1}/{max_retries}")
                
                # snapshot_download 사용
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    token=HF_TOKEN,
                    local_dir_use_symlinks=False,
                    resume_download=True,  # 중단된 다운로드 재개
                    max_workers=2  # 동시 다운로드 수 제한
                )
                
                logger.info("✓ 다운로드 성공!")
                return True
                
            except Exception as e:
                logger.error(f"다운로드 실패 (시도 {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)
                    logger.info(f"{wait_time}초 대기 후 재시도...")
                    time.sleep(wait_time)
                    
        return False
        
    def install_model_safe(self, model_name: str) -> bool:
        """안전한 모델 설치"""
        if model_name not in MODEL_CONFIGS:
            logger.error(f"알 수 없는 모델: {model_name}")
            return False
            
        config = MODEL_CONFIGS[model_name]
        model_id = config["model_id"]
        model_path = self.models_dir / model_name
        model_path.mkdir(exist_ok=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{model_name} 설치 시작")
        logger.info(f"모델 ID: {model_id}")
        logger.info(f"필요 메모리: {config['memory_gb']} GB")
        logger.info(f"양자화: {config['quantization']}")
        
        # GPU 메모리 확인
        gpu_memory = self.check_gpu_memory()
        logger.info(f"사용 가능한 GPU 메모리: {gpu_memory:.1f} GB")
        
        try:
            # 1단계: 토크나이저 다운로드
            logger.info("\n1단계: 토크나이저 다운로드")
            tokenizer_path = model_path / "tokenizer"
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=HF_TOKEN,
                cache_dir=str(model_path / "cache"),
                trust_remote_code=True
            )
            tokenizer.save_pretrained(str(tokenizer_path))
            logger.info("✓ 토크나이저 저장 완료")
            
            # 2단계: 모델 파일 다운로드
            logger.info("\n2단계: 모델 파일 다운로드")
            model_cache_dir = model_path / "model_files"
            
            if config["download_strategy"] == "snapshot":
                success = self.download_with_retry(
                    repo_id=model_id,
                    local_dir=str(model_cache_dir)
                )
                
                if not success:
                    logger.error("모델 다운로드 실패")
                    return False
                    
            # 3단계: 설치 정보 저장
            install_info = {
                "model_id": model_id,
                "model_name": model_name,
                "quantization": config["quantization"],
                "status": "downloaded",
                "tokenizer_path": str(tokenizer_path),
                "model_path": str(model_cache_dir),
                "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "memory_required": config["memory_gb"]
            }
            
            with open(model_path / "install_info.json", "w") as f:
                json.dump(install_info, f, indent=2)
                
            logger.info(f"\n✓ {model_name} 설치 완료!")
            return True
            
        except Exception as e:
            logger.error(f"\n✗ 설치 중 오류 발생: {str(e)}")
            return False
            
    def test_model_loading(self, model_name: str) -> bool:
        """모델 로딩 테스트"""
        model_path = self.models_dir / model_name
        info_path = model_path / "install_info.json"
        
        if not info_path.exists():
            logger.error(f"{model_name}이 설치되지 않았습니다.")
            return False
            
        with open(info_path, "r") as f:
            info = json.load(f)
            
        logger.info(f"\n{model_name} 로딩 테스트...")
        
        try:
            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(info["tokenizer_path"])
            
            # 간단한 토크나이저 테스트
            test_text = "Hello, this is a test for C# programming."
            tokens = tokenizer(test_text, return_tensors="pt")
            logger.info(f"✓ 토크나이저 테스트 성공: {tokens.input_ids.shape}")
            
            # 양자화 설정
            if info["quantization"] == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            elif info["quantization"] == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
            else:
                quantization_config = None
                
            logger.info("모델 로딩은 메모리가 충분할 때 시도하세요.")
            logger.info(f"필요 메모리: {info['memory_required']} GB")
            
            return True
            
        except Exception as e:
            logger.error(f"테스트 실패: {str(e)}")
            return False

def main():
    """메인 함수"""
    installer = RobustModelInstaller()
    
    logger.info("=== 강력한 LLM 모델 설치 시작 ===")
    logger.info(f"Hugging Face 토큰: {HF_TOKEN[:10]}...")
    
    # 시스템 정보
    logger.info("\n시스템 정보:")
    logger.info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {installer.check_gpu_memory():.1f} GB")
        
    # 설치할 모델 선택
    if len(sys.argv) > 1:
        # 특정 모델만 설치
        model_name = sys.argv[1]
        if model_name in MODEL_CONFIGS:
            success = installer.install_model_safe(model_name)
            if success:
                installer.test_model_loading(model_name)
        else:
            logger.error(f"알 수 없는 모델: {model_name}")
            logger.info(f"사용 가능한 모델: {list(MODEL_CONFIGS.keys())}")
    else:
        # 모든 모델 설치 시도
        logger.info("\n모든 모델을 순차적으로 설치합니다.")
        logger.info("네트워크 상태에 따라 시간이 오래 걸릴 수 있습니다.")
        
        for model_name in MODEL_CONFIGS:
            success = installer.install_model_safe(model_name)
            if success:
                installer.test_model_loading(model_name)
            else:
                logger.warning(f"{model_name} 설치 실패. 다음 모델로 진행합니다.")
                
    logger.info("\n=== 설치 프로세스 완료 ===")
    
    # 설치 요약
    installed_models = []
    for model_name in MODEL_CONFIGS:
        info_path = installer.models_dir / model_name / "install_info.json"
        if info_path.exists():
            installed_models.append(model_name)
            
    logger.info(f"\n설치된 모델: {installed_models}")
    logger.info("\n다음 단계:")
    logger.info("1. 설치된 모델 확인: ls -la models/")
    logger.info("2. 연속 학습 실행: python continuous_learning_system.py")

if __name__ == "__main__":
    main()