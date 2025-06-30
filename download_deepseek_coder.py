#!/usr/bin/env python3
"""
DeepSeek-coder-v2 6.7B 모델 다운로드 및 설치 스크립트
RTX 2080 8GB + 32GB RAM 최적화
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def setup_environment():
    """가상환경 설정 확인"""
    venv_path = Path("autoci_env")
    if not venv_path.exists():
        print("❌ 가상환경이 없습니다. 먼저 가상환경을 생성하세요:")
        print("python -m venv autoci_env")
        return False
    
    # Windows에서는 Scripts, Linux에서는 bin
    activate_script = venv_path / "Scripts" / "activate" if os.name == 'nt' else venv_path / "bin" / "activate"
    if not activate_script.exists():
        print("❌ 가상환경 활성화 스크립트를 찾을 수 없습니다.")
        return False
    
    print("✅ 가상환경 확인 완료")
    return True

def check_system_requirements():
    """시스템 요구사항 확인"""
    print("🔍 시스템 요구사항 확인 중...")
    
    # GPU 확인
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU: {gpu_name} ({vram:.1f}GB VRAM)")
        
        if vram < 6:
            print("⚠️ 경고: VRAM이 6GB 미만입니다. 양자화가 필요할 수 있습니다.")
    else:
        print("⚠️ CUDA GPU를 찾을 수 없습니다. CPU 모드로 실행됩니다.")
    
    # RAM 확인 (대략적)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"✅ 시스템 RAM: {ram_gb:.1f}GB")
        
        if ram_gb < 16:
            print("⚠️ 경고: 시스템 RAM이 16GB 미만입니다.")
    except ImportError:
        print("ℹ️ psutil이 없어 RAM 확인을 건너뜁니다.")
    
    # 디스크 공간 확인
    free_space = os.statvfs('.').f_bavail * os.statvfs('.').f_frsize / (1024**3)
    print(f"✅ 사용 가능한 디스크 공간: {free_space:.1f}GB")
    
    if free_space < 15:
        print("❌ 오류: 디스크 공간이 부족합니다 (15GB 이상 필요)")
        return False
    
    return True

def download_deepseek_coder():
    """DeepSeek-coder 6.7B 모델 다운로드"""
    print("🚀 DeepSeek-coder-v2 6.7B 다운로드 시작...")
    
    model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
    model_dir = Path("models") / "deepseek-coder-7b"
    
    # 모델 디렉토리 생성
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("📥 토크나이저 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=model_dir / "cache"
        )
        
        print("💾 토크나이저 저장 중...")
        tokenizer.save_pretrained(model_dir / "tokenizer")
        
        print("📥 모델 다운로드 중... (시간이 오래 걸릴 수 있습니다)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            cache_dir=model_dir / "cache"
        )
        
        print("💾 모델 저장 중...")
        model.save_pretrained(model_dir / "model")
        
        # 설치 정보 저장
        install_info = {
            "model_id": model_id,
            "model_name": "deepseek-coder-7b",
            "version": "6.7b-instruct",
            "download_date": str(Path().cwd()),
            "size_gb": 14.2,
            "vram_requirement_gb": 6,
            "optimized_for": "RTX 2080 8GB",
            "quantization": "bfloat16",
            "status": "installed"
        }
        
        with open(model_dir / "install_info.json", "w", encoding="utf-8") as f:
            json.dump(install_info, f, indent=2, ensure_ascii=False)
        
        print("✅ DeepSeek-coder-v2 6.7B 다운로드 완료!")
        print(f"📁 설치 위치: {model_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return False

def update_model_config():
    """모델 설정 업데이트"""
    print("⚙️ 모델 설정 업데이트 중...")
    
    config_file = Path("models") / "installed_models.json"
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # deepseek-coder-7b 설정 업데이트
        if "deepseek-coder-7b" in config["models"]:
            config["models"]["deepseek-coder-7b"].update({
                "status": "installed",
                "model_path": "models/deepseek-coder-7b/model",
                "tokenizer_path": "models/deepseek-coder-7b/tokenizer",
                "last_updated": "2025-01-03",
                "rtx_2080_optimized": True,
                "priority_score": 10,  # 최고 우선순위
                "quantization_support": ["4bit", "8bit", "bfloat16"],
                "recommended_quantization": "bfloat16"
            })
        
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("✅ 모델 설정 업데이트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 설정 업데이트 실패: {e}")
        return False

def test_model():
    """모델 테스트"""
    print("🧪 모델 테스트 중...")
    
    try:
        model_dir = Path("models") / "deepseek-coder-7b"
        
        print("📥 토크나이저 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir / "tokenizer",
            trust_remote_code=True
        )
        
        print("📥 모델 로드 중...")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir / "model",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # 간단한 테스트
        test_prompt = "# C#에서 async/await를 사용한 예제를 작성해주세요"
        
        messages = [
            {"role": "user", "content": test_prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(model.device)
        
        print("💭 테스트 생성 중...")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        print("✅ 모델 테스트 성공!")
        print("📝 테스트 응답:")
        print("-" * 50)
        print(response[:300] + "..." if len(response) > 300 else response)
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🎯 DeepSeek-coder-v2 6.7B 설치 프로그램")
    print("=" * 50)
    
    # 환경 확인
    if not setup_environment():
        return False
    
    if not check_system_requirements():
        return False
    
    # 사용자 확인
    print("\n📋 설치 정보:")
    print("- 모델: deepseek-ai/deepseek-coder-6.7b-instruct")
    print("- 크기: ~14GB")
    print("- VRAM 요구사항: 6GB")
    print("- 최적화 대상: RTX 2080 8GB")
    
    confirm = input("\n계속 진행하시겠습니까? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 설치가 취소되었습니다.")
        return False
    
    # 다운로드 및 설정
    if not download_deepseek_coder():
        return False
    
    if not update_model_config():
        return False
    
    # 테스트
    if not test_model():
        print("⚠️ 모델 테스트에 실패했지만 설치는 완료되었습니다.")
    
    print("\n🎉 DeepSeek-coder-v2 6.7B 설치 완료!")
    print("이제 다음 명령어로 사용할 수 있습니다:")
    print("👉 autoci learn low")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 