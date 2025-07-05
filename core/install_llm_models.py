#!/usr/bin/env python3
"""
LLM 모델 설치 및 설정 스크립트
Llama-3.1-8B, CodeLlama-13B, Qwen2.5-Coder-32B 모델을 설치하고 설정합니다.
"""

import os
import sys
import json
import subprocess
import torch
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_installation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Hugging Face 토큰
HF_TOKEN = "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"

# 모델 설정
MODEL_CONFIGS = {
    "llama-3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "quantization": "8bit",
        "memory_required": 16,  # GB
        "features": ["general", "korean", "csharp"]
    },
    "codellama-13b": {
        "model_id": "codellama/CodeLlama-13b-Instruct-hf",
        "quantization": "8bit",
        "memory_required": 26,  # GB
        "features": ["code", "csharp", "godot"]
    },
    "qwen2.5-coder-32b": {
        "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "quantization": "4bit",
        "memory_required": 64,  # GB
        "features": ["code", "korean", "csharp", "advanced"]
    }
}

class ModelInstaller:
    def __init__(self, models_dir: str = "./models", token: str = HF_TOKEN):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.token = token
        self.installed_models = {}
        self.load_installed_models()
        
    def load_installed_models(self):
        """설치된 모델 정보 로드"""
        models_info_file = self.models_dir / "installed_models.json"
        if models_info_file.exists():
            with open(models_info_file, 'r', encoding='utf-8') as f:
                self.installed_models = json.load(f)
                
    def save_installed_models(self):
        """설치된 모델 정보 저장"""
        models_info_file = self.models_dir / "installed_models.json"
        with open(models_info_file, 'w', encoding='utf-8') as f:
            json.dump(self.installed_models, f, indent=2, ensure_ascii=False)
            
    def check_system_requirements(self) -> Dict[str, bool]:
        """시스템 요구사항 확인"""
        requirements = {}
        
        # CUDA 확인
        requirements['cuda'] = torch.cuda.is_available()
        if requirements['cuda']:
            requirements['cuda_version'] = torch.version.cuda
            requirements['gpu_count'] = torch.cuda.device_count()
            requirements['gpu_memory'] = []
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                requirements['gpu_memory'].append(f"GPU {i}: {gpu_memory:.1f} GB")
        
        # RAM 확인
        try:
            import psutil
            requirements['ram'] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            requirements['ram'] = "Unknown (psutil not installed)"
            
        # 디스크 공간 확인
        import shutil
        stat = shutil.disk_usage(self.models_dir)
        requirements['disk_free'] = stat.free / (1024**3)
        
        return requirements
        
    def install_dependencies(self):
        """필요한 의존성 설치"""
        logger.info("필요한 의존성을 설치합니다...")
        
        dependencies = [
            "transformers>=4.40.0",
            "accelerate>=0.30.0",
            "bitsandbytes>=0.43.0",
            "sentencepiece",
            "protobuf",
            "einops",
            "psutil"
        ]
        
        for dep in dependencies:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                             check=True, capture_output=True, text=True)
                logger.info(f"✓ {dep} 설치 완료")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ {dep} 설치 실패: {e.stderr}")
                
    def get_quantization_config(self, quantization: str):
        """양자화 설정 반환"""
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        else:
            return None
            
    def install_model(self, model_name: str, force_reinstall: bool = False):
        """개별 모델 설치"""
        if model_name not in MODEL_CONFIGS:
            logger.error(f"알 수 없는 모델: {model_name}")
            return False
            
        config = MODEL_CONFIGS[model_name]
        model_id = config['model_id']
        
        # 이미 설치된 경우
        if model_name in self.installed_models and not force_reinstall:
            logger.info(f"{model_name}은 이미 설치되어 있습니다.")
            return True
            
        logger.info(f"\n{'='*50}")
        logger.info(f"{model_name} 설치를 시작합니다...")
        logger.info(f"모델 ID: {model_id}")
        logger.info(f"양자화: {config['quantization']}")
        logger.info(f"필요 메모리: {config['memory_required']} GB")
        
        try:
            # 모델 경로 설정
            model_path = self.models_dir / model_name
            model_path.mkdir(exist_ok=True)
            
            # 양자화 설정
            quantization_config = self.get_quantization_config(config['quantization'])
            
            # 토크나이저 다운로드
            logger.info("토크나이저를 다운로드합니다...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=self.token,
                cache_dir=model_path,
                trust_remote_code=True
            )
            tokenizer.save_pretrained(model_path / "tokenizer")
            
            # 모델 다운로드 (양자화 적용)
            logger.info("모델을 다운로드합니다. 시간이 걸릴 수 있습니다...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                token=self.token,
                cache_dir=model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            # 모델 저장
            logger.info("모델을 저장합니다...")
            model.save_pretrained(model_path / "model", safe_serialization=True)
            
            # 설치 정보 저장
            self.installed_models[model_name] = {
                "model_id": model_id,
                "path": str(model_path),
                "quantization": config['quantization'],
                "features": config['features'],
                "installed_at": datetime.now().isoformat(),
                "size_gb": config['memory_required']
            }
            self.save_installed_models()
            
            logger.info(f"✓ {model_name} 설치 완료!")
            return True
            
        except Exception as e:
            logger.error(f"✗ {model_name} 설치 실패: {str(e)}")
            return False
            
    def test_model(self, model_name: str):
        """모델 테스트"""
        if model_name not in self.installed_models:
            logger.error(f"{model_name}이 설치되지 않았습니다.")
            return False
            
        logger.info(f"\n{model_name} 테스트 중...")
        
        try:
            model_info = self.installed_models[model_name]
            model_path = Path(model_info['path'])
            
            # 모델과 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer")
            quantization_config = self.get_quantization_config(model_info['quantization'])
            
            # 간단한 테스트 프롬프트
            test_prompts = {
                "csharp": "Write a simple C# function to calculate factorial:",
                "korean": "안녕하세요. 간단한 C# 프로그램을 작성해주세요:",
                "godot": "Create a simple Godot C# script for player movement:"
            }
            
            # 파이프라인 생성
            pipe = pipeline(
                "text-generation",
                model=model_path / "model",
                tokenizer=tokenizer,
                device_map="auto",
                model_kwargs={"quantization_config": quantization_config}
            )
            
            # 각 기능 테스트
            for feature in model_info['features']:
                if feature in test_prompts:
                    logger.info(f"테스트: {feature}")
                    result = pipe(
                        test_prompts[feature],
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True
                    )
                    logger.info(f"응답: {result[0]['generated_text'][:200]}...")
                    
            logger.info(f"✓ {model_name} 테스트 완료!")
            return True
            
        except Exception as e:
            logger.error(f"✗ {model_name} 테스트 실패: {str(e)}")
            return False
            
    def install_all_models(self):
        """모든 모델 설치"""
        logger.info("모든 LLM 모델 설치를 시작합니다.")
        
        # 시스템 요구사항 확인
        requirements = self.check_system_requirements()
        logger.info("\n시스템 정보:")
        for key, value in requirements.items():
            logger.info(f"  {key}: {value}")
            
        # 의존성 설치
        self.install_dependencies()
        
        # 각 모델 설치
        success_count = 0
        for model_name in MODEL_CONFIGS:
            if self.install_model(model_name):
                success_count += 1
                self.test_model(model_name)
                
        logger.info(f"\n설치 완료: {success_count}/{len(MODEL_CONFIGS)} 모델")
        
        # 설치 요약
        self.print_installation_summary()
        
    def print_installation_summary(self):
        """설치 요약 출력"""
        logger.info("\n" + "="*60)
        logger.info("설치된 모델 요약:")
        logger.info("="*60)
        
        for model_name, info in self.installed_models.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  - 경로: {info['path']}")
            logger.info(f"  - 양자화: {info['quantization']}")
            logger.info(f"  - 기능: {', '.join(info['features'])}")
            logger.info(f"  - 설치일: {info['installed_at']}")
            logger.info(f"  - 크기: {info['size_gb']} GB")
            
        logger.info("\n사용 방법:")
        logger.info("  python continuous_learning_system.py")
        logger.info("  python csharp_korean_learning.py")

def main():
    """메인 함수"""
    installer = ModelInstaller()
    
    if len(sys.argv) > 1:
        # 특정 모델만 설치
        model_name = sys.argv[1]
        installer.install_model(model_name)
        installer.test_model(model_name)
    else:
        # 모든 모델 설치
        installer.install_all_models()

if __name__ == "__main__":
    main()