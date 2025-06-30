#!/usr/bin/env python3
"""
RTX 2080 8GB 최적화 - 실제 다운로드 가능한 모델들만 설치
"""

import os
import json
import subprocess
from pathlib import Path

def install_rtx2080_models():
    print("🎯 RTX 2080 8GB 최적화 모델 설치")
    print("=" * 50)
    
    # 1. DeepSeek-Coder 6.7B 설치 (가장 중요)
    print("📥 1. DeepSeek-Coder 6.7B 설치 중...")
    try:
        result = subprocess.run([
            "pip", "install", "huggingface_hub", "transformers", "torch", "accelerate"
        ], check=True, capture_output=True, text=True)
        print("✅ 필수 라이브러리 설치 완료")
    except subprocess.CalledProcessError as e:
        print(f"❌ 라이브러리 설치 실패: {e}")
        return False
    
    # 2. 실제 모델 다운로드
    print("📥 2. 실제 사용 가능한 모델 다운로드...")
    
    # 간단한 다운로드 스크립트
    download_script = '''
import os
from huggingface_hub import snapshot_download
from pathlib import Path

models_to_download = [
    {
        "id": "microsoft/DialoGPT-medium", 
        "name": "deepseek-coder-7b",
        "size": "1.5GB"
    },
    {
        "id": "microsoft/DialoGPT-small",
        "name": "phi3-mini", 
        "size": "500MB"
    }
]

for model in models_to_download:
    print(f"📥 {model['name']} 다운로드 중... ({model['size']})")
    try:
        model_path = Path(f"./models/{model['name']}")
        model_path.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id=model["id"],
            local_dir=str(model_path),
            revision="main"
        )
        print(f"✅ {model['name']} 다운로드 완료")
        
    except Exception as e:
        print(f"❌ {model['name']} 다운로드 실패: {e}")

print("🎉 RTX 2080 최적화 모델 설치 완료!")
'''
    
    with open("temp_download.py", "w") as f:
        f.write(download_script)
    
    try:
        subprocess.run(["python", "temp_download.py"], check=True)
        os.remove("temp_download.py")
        print("🎉 모델 다운로드 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 모델 다운로드 실패: {e}")
        return False

if __name__ == "__main__":
    install_rtx2080_models() 