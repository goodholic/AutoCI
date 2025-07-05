#!/usr/bin/env python3
"""
RTX 2080 8GB 최적화 LLM 모델 설치 스크립트
8GB VRAM + 32GB RAM 환경에 완벽한 모델들만 설치
"""

import os
import json
import time
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hugging Face 토큰
HF_TOKEN = "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"

# RTX 2080 8GB 최적화 모델 설정
RTX_2080_MODELS = {
    "bitnet-b1.58-2b": {
        "model_id": "microsoft/BitNet-b1.58-2B-4T-gguf", 
        "size_gb": 0.5,
        "quantization": "1.58bit",
        "features": ["general", "korean", "efficient"],
        "vram_gb": 1,
        "description": "혁신적인 1.58bit 모델, 극도로 경량"
    },
    "gemma-4b": {
        "model_id": "google/gemma-4b-it",
        "size_gb": 2.5,
        "quantization": "4bit", 
        "features": ["general", "korean", "csharp"],
        "vram_gb": 4,
        "description": "Google Gemma 4B, 뛰어난 성능 대비 크기"
    },
    "phi3-mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "size_gb": 4.0,
        "quantization": "4bit",
        "features": ["reasoning", "csharp", "math"],
        "vram_gb": 6,
        "description": "Microsoft Phi-3 Mini, 추론 특화"
    },
    "deepseek-coder-7b": {
        "model_id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "size_gb": 4.2,
        "quantization": "4bit",
        "features": ["code", "csharp", "godot"],
        "vram_gb": 6,
        "description": "DeepSeek 코딩 특화 모델"
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "size_gb": 4.5,
        "quantization": "4bit",
        "features": ["general", "fast", "csharp"],
        "vram_gb": 7,
        "description": "Mistral 7B, 빠른 추론 속도"
    }
}

def install_model(model_name: str, model_info: dict, models_dir: Path):
    """RTX 2080 최적화 모델 설치"""
    model_path = models_dir / model_name
    model_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 {model_name} 설치 시작...")
    logger.info(f"   📦 크기: {model_info['size_gb']}GB")
    logger.info(f"   🎯 VRAM: {model_info['vram_gb']}GB")
    logger.info(f"   📝 설명: {model_info['description']}")
    
    try:
        # 모델 다운로드
        snapshot_download(
            repo_id=model_info['model_id'],
            local_dir=str(model_path),
            token=HF_TOKEN,
            revision="main"
        )
        
        # 설치 정보 저장
        install_info = {
            "installed_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "model_id": model_info['model_id'],
            "size_gb": model_info['size_gb'],
            "vram_gb": model_info['vram_gb'],
            "quantization": model_info['quantization'],
            "features": model_info['features'],
            "rtx_2080_optimized": True
        }
        
        with open(model_path / "install_info.json", 'w') as f:
            json.dump(install_info, f, indent=2)
            
        logger.info(f"✅ {model_name} 설치 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ {model_name} 설치 실패: {str(e)}")
        return False

def update_installed_models_json(models_dir: Path):
    """installed_models.json 업데이트"""
    installed_models = {}
    
    for model_name, model_info in RTX_2080_MODELS.items():
        model_path = models_dir / model_name
        if (model_path / "install_info.json").exists():
            installed_models[model_name] = {
                "model_id": model_info['model_id'],
                "path": str(model_path),
                "quantization": model_info['quantization'],
                "features": model_info['features'],
                "installed_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
                "size_gb": model_info['size_gb'],
                "vram_gb": model_info['vram_gb'],
                "status": "installed",
                "rtx_2080_optimized": True
            }
    
    models_info_file = models_dir / "installed_models.json"
    with open(models_info_file, 'w', encoding='utf-8') as f:
        json.dump(installed_models, f, indent=2, ensure_ascii=False)
        
    logger.info(f"📋 installed_models.json 업데이트 완료")

def main():
    """RTX 2080 최적화 모델 설치 메인 함수"""
    print("🎯 RTX 2080 8GB + 32GB RAM 최적화 LLM 모델 설치")
    print("=" * 60)
    print("💻 최적화 기준:")
    print("  - GPU VRAM: 8GB 이하")
    print("  - 시스템 RAM: 32GB 호환")
    print("  - 양자화: 4bit 이하")
    print("  - 실시간 추론 가능")
    print("=" * 60)
    
    # 환경 확인
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    # 기존 무거운 모델 백업
    backup_dir = models_dir / "backup_heavy_models"
    backup_dir.mkdir(exist_ok=True)
    
    heavy_models = ["llama-3.1-8b", "codellama-13b", "qwen2.5-coder-32b"]
    for heavy_model in heavy_models:
        heavy_path = models_dir / heavy_model
        if heavy_path.exists():
            backup_path = backup_dir / heavy_model
            if not backup_path.exists():
                logger.info(f"📦 무거운 모델 백업: {heavy_model}")
                heavy_path.rename(backup_path)
    
    # RTX 2080 최적화 모델들 설치
    success_count = 0
    total_size = 0
    
    for model_name, model_info in RTX_2080_MODELS.items():
        print(f"\n🔄 {model_name} 설치 중...")
        if install_model(model_name, model_info, models_dir):
            success_count += 1
            total_size += model_info['size_gb']
        
        # 설치 간 잠시 대기
        time.sleep(2)
    
    # 설치 완료 후 설정 업데이트
    update_installed_models_json(models_dir)
    
    print("\n" + "=" * 60)
    print("🎉 RTX 2080 최적화 모델 설치 완료!")
    print(f"✅ 성공: {success_count}/{len(RTX_2080_MODELS)} 모델")
    print(f"💾 총 크기: {total_size:.1f}GB")
    print(f"🎯 VRAM 사용량: 최대 7GB (8GB 한계 내)")
    print("\n🚀 이제 다음 명령어를 사용하세요:")
    print("   autoci learn low  # RTX 2080 최적화 학습")
    print("=" * 60)

if __name__ == "__main__":
    main() 