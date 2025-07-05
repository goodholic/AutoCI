#!/usr/bin/env python3
"""
autoci learn 의존성 문제 해결 스크립트
"""

import subprocess
import sys

def install_packages():
    """필요한 패키지 설치"""
    packages = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.40.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "safetensors>=0.3.1",
        "psutil>=5.9.0",
        "googlesearch-python",
        "beautifulsoup4",
        "requests",
        "aiohttp",
        "scipy",
        "scikit-learn",
        "pandas",
        "numpy"
    ]
    
    print("🔧 autoci learn 의존성 설치 시작...")
    
    for package in packages:
        print(f"📦 {package} 설치 중...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 설치 완료")
        except subprocess.CalledProcessError:
            print(f"⚠️  {package} 설치 실패 (계속 진행)")
    
    print("\n✨ 의존성 설치 완료!")
    print("\n이제 다음 명령어를 실행하세요:")
    print("  autoci learn")
    print("  autoci learn low")

if __name__ == "__main__":
    install_packages()