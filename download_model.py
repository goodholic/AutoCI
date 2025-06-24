#!/usr/bin/env python3
"""
Code Llama 7B-Instruct 모델 다운로드 스크립트
Hugging Face에서 모델을 다운로드하고 설정
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import shutil
import torch
from huggingface_hub import snapshot_download, login
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Code Llama 모델 다운로더"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.model_name = "codellama/CodeLlama-7b-Instruct-hf"
        self.model_dir = self.base_dir / "CodeLlama-7b-Instruct-hf"
        
    def check_requirements(self):
        """시스템 요구사항 확인"""
        logger.info("🔍 시스템 요구사항 확인 중...")
        
        # Python 버전 확인
        if sys.version_info < (3, 8):
            logger.error("❌ Python 3.8 이상이 필요합니다")
            return False
        
        # 디스크 공간 확인
        stat = shutil.disk_usage(".")
        free_gb = stat.free / (1024**3)
        
        if free_gb < 20:
            logger.warning(f"⚠️  디스크 여유 공간이 {free_gb:.1f}GB입니다. 최소 20GB 이상 권장")
            response = input("계속하시겠습니까? (y/n): ")
            if response.lower() != 'y':
                return False
        
        # PyTorch 확인
        try:
            import torch
            logger.info(f"✅ PyTorch {torch.__version__} 확인")
            
            # CUDA 확인
            if torch.cuda.is_available():
                logger.info(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"   GPU VRAM: {vram:.1f}GB")
            else:
                logger.warning("⚠️  GPU를 사용할 수 없습니다. CPU로 실행됩니다 (느림)")
                
        except ImportError:
            logger.error("❌ PyTorch가 설치되지 않았습니다")
            return False
        
        return True
    
    def check_existing_model(self):
        """기존 모델 확인"""
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            logger.info(f"✅ 모델이 이미 존재합니다: {self.model_dir}")
            
            # 모델 무결성 확인
            required_files = [
                "config.json",
                "tokenizer_config.json",
                "pytorch_model.bin.index.json"
            ]
            
            missing_files = []
            for file in required_files:
                if not (self.model_dir / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                logger.warning(f"⚠️  일부 파일이 누락되었습니다: {missing_files}")
                response = input("다시 다운로드하시겠습니까? (y/n): ")
                if response.lower() == 'y':
                    shutil.rmtree(self.model_dir)
                    return False
            
            return True
        return False
    
    def download_model(self):
        """모델 다운로드"""
        logger.info(f"📥 {self.model_name} 다운로드 시작...")
        logger.info("   크기: 약 13GB (시간이 걸릴 수 있습니다)")
        
        try:
            # Hugging Face 로그인 (선택사항)
            token = os.getenv("HUGGINGFACE_TOKEN")
            if token:
                login(token=token)
                logger.info("✅ Hugging Face 토큰으로 로그인됨")
            
            # 모델 다운로드
            logger.info("📦 모델 파일 다운로드 중...")
            snapshot_download(
                repo_id=self.model_name,
                local_dir=str(self.model_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                cache_dir=None  # 캐시 사용 안 함
            )
            
            logger.info("✅ 모델 다운로드 완료!")
            
            # 모델 크기 확인
            total_size = sum(f.stat().st_size for f in self.model_dir.rglob("*") if f.is_file())
            size_gb = total_size / (1024**3)
            logger.info(f"   총 크기: {size_gb:.2f}GB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 다운로드 실패: {str(e)}")
            logger.info("\n💡 해결 방법:")
            logger.info("1. 인터넷 연결을 확인하세요")
            logger.info("2. Hugging Face가 차단되지 않았는지 확인하세요")
            logger.info("3. 디스크 공간이 충분한지 확인하세요")
            logger.info("4. 다음 명령으로 수동 다운로드 시도:")
            logger.info(f"   git lfs clone https://huggingface.co/{self.model_name}")
            return False
    
    def verify_model(self):
        """모델 검증"""
        logger.info("🔍 모델 검증 중...")
        
        try:
            # 토크나이저 로드 테스트
            logger.info("   토크나이저 로드 테스트...")
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            
            # 간단한 토큰화 테스트
            test_text = "Hello, World!"
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)
            logger.info(f"   ✅ 토크나이저 테스트 성공: '{test_text}' -> {len(tokens)} tokens")
            
            # 설정 파일 확인
            config_path = self.model_dir / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"   ✅ 모델 타입: {config.get('model_type', 'unknown')}")
            logger.info(f"   ✅ 히든 크기: {config.get('hidden_size', 'unknown')}")
            
            # 모델 파일 확인
            model_files = list(self.model_dir.glob("pytorch_model*.bin"))
            if model_files:
                logger.info(f"   ✅ 모델 파일 확인: {len(model_files)}개 샤드")
            else:
                logger.warning("   ⚠️  모델 파일을 찾을 수 없습니다")
                return False
            
            logger.info("✅ 모델 검증 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 모델 검증 실패: {str(e)}")
            return False
    
    def create_model_info(self):
        """모델 정보 파일 생성"""
        info = {
            "model_name": self.model_name,
            "model_path": str(self.model_dir),
            "download_date": str(Path.ctime(self.model_dir)),
            "verified": True,
            "requirements": {
                "min_ram_gb": 16,
                "recommended_ram_gb": 32,
                "gpu_vram_gb": 8
            },
            "usage": {
                "load_model": f"model = AutoModelForCausalLM.from_pretrained('{self.model_dir}')",
                "load_tokenizer": f"tokenizer = AutoTokenizer.from_pretrained('{self.model_dir}')"
            }
        }
        
        info_path = self.model_dir / "model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📝 모델 정보 저장: {info_path}")
    
    def create_quick_test_script(self):
        """빠른 테스트 스크립트 생성"""
        test_script = '''#!/usr/bin/env python3
"""Code Llama 빠른 테스트"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("🤖 Code Llama 7B-Instruct 테스트")

# 모델과 토크나이저 로드
print("📥 모델 로드 중... (시간이 걸릴 수 있습니다)")
model_path = "./CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# 테스트 프롬프트
prompt = """### Instruction:
Write a simple C# function that calculates the factorial of a number.

### Response:
"""

print("💭 코드 생성 중...")
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    max_length=200,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\\n✅ 생성된 코드:")
print(response)
'''
        
        test_path = self.base_dir / "test_model.py"
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_script)
        
        os.chmod(test_path, 0o755)
        logger.info(f"🧪 테스트 스크립트 생성: {test_path}")
        logger.info("   실행: python test_model.py")
    
    def run(self, check_only=False):
        """메인 실행"""
        logger.info("🚀 Code Llama 7B-Instruct 모델 다운로더")
        logger.info("="*50)
        
        # 시스템 요구사항 확인
        if not self.check_requirements():
            return False
        
        # 기존 모델 확인
        if self.check_existing_model():
            if check_only:
                logger.info("✅ 모델이 이미 설치되어 있습니다")
                return True
            
            response = input("\n모델이 이미 존재합니다. 다시 다운로드하시겠습니까? (y/n): ")
            if response.lower() != 'y':
                logger.info("✅ 기존 모델을 사용합니다")
                return True
        
        if check_only:
            logger.info("❌ 모델이 설치되지 않았습니다")
            return False
        
        # 모델 다운로드
        if not self.download_model():
            return False
        
        # 모델 검증
        if not self.verify_model():
            logger.warning("⚠️  모델 검증에 실패했지만 계속 진행합니다")
        
        # 모델 정보 생성
        self.create_model_info()
        
        # 테스트 스크립트 생성
        self.create_quick_test_script()
        
        logger.info("\n" + "="*50)
        logger.info("🎉 모델 설치가 완료되었습니다!")
        logger.info("="*50)
        logger.info("\n다음 단계:")
        logger.info("1. 모델 테스트: python test_model.py")
        logger.info("2. 전체 시스템 시작: python start_all.py")
        
        return True

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Code Llama 7B-Instruct 모델 다운로더")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="모델 존재 여부만 확인"
    )
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    success = downloader.run(check_only=args.check_only)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()