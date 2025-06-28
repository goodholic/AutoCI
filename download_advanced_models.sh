#!/bin/bash

# AutoCI v3.0 - 고급 AI 모델 다운로드 스크립트
# 대형 모델들을 단계별로 다운로드합니다 (총 150GB+)

echo "🚀 AutoCI v3.0 - 고급 AI 모델 다운로드"
echo "======================================="
echo ""
echo "⚠️  주의사항:"
echo "  • 총 다운로드 크기: ~150GB"
echo "  • 예상 소요 시간: 1-3시간 (인터넷 속도에 따라)"
echo "  • 32GB RAM 권장"
echo "  • 200GB 여유 공간 필요"
echo ""

# 가상환경 확인
if [ -z "$VIRTUAL_ENV" ] && [ -d "autoci_env" ]; then
    echo "🔄 가상환경 활성화 중..."
    source autoci_env/bin/activate
fi

# Python 환경 확인
if ! command -v python &> /dev/null; then
    echo "❌ Python이 설치되어 있지 않습니다."
    exit 1
fi

# 필요한 패키지 설치
echo "📦 필요한 패키지 설치 중..."
python -m pip install huggingface-hub transformers torch --quiet

# 모델 저장 디렉토리 생성
mkdir -p models
cd models

# 사용자에게 어떤 모델을 다운로드할지 선택하게 함
echo ""
echo "다운로드할 모델을 선택하세요:"
echo "1. Llama 3.1 70B (4-bit 양자화) - ~35GB"
echo "2. Qwen2.5 72B (4-bit 양자화) - ~36GB"  
echo "3. DeepSeek V2.5 (4-bit 양자화) - ~50GB"
echo "4. 모든 모델 다운로드 (권장) - ~150GB"
echo "5. 기본 모델만 (Code Llama 7B) - ~13GB"
echo ""
read -p "선택하세요 (1-5): " choice

case $choice in
    1)
        echo "🦙 Llama 3.1 70B 다운로드 중..."
        download_llama=true
        ;;
    2)
        echo "🤖 Qwen2.5 72B 다운로드 중..."
        download_qwen=true
        ;;
    3)
        echo "🧠 DeepSeek V2.5 다운로드 중..."
        download_deepseek=true
        ;;
    4)
        echo "📦 모든 고급 모델 다운로드 중..."
        download_llama=true
        download_qwen=true
        download_deepseek=true
        ;;
    5)
        echo "📦 기본 모델 다운로드 중..."
        download_basic=true
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "다운로드를 시작합니다..."
echo ""

# Python 다운로드 스크립트 생성
cat > download_models.py << 'EOF'
#!/usr/bin/env python3
"""
AutoCI 고급 모델 다운로드 스크립트
"""

import os
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model(repo_id, local_dir, description):
    """모델을 다운로드합니다"""
    print(f"📥 {description} 다운로드 중...")
    print(f"   저장소: {repo_id}")
    print(f"   저장 위치: {local_dir}")
    
    try:
        # 모델 다운로드
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            cache_dir="./cache"
        )
        print(f"✅ {description} 다운로드 완료!")
        return True
    except Exception as e:
        print(f"❌ {description} 다운로드 실패: {str(e)}")
        return False

def main():
    """메인 다운로드 함수"""
    import sys
    
    models_to_download = []
    
    # 명령행 인수로 어떤 모델을 다운로드할지 결정
    if len(sys.argv) > 1:
        if "llama" in sys.argv:
            models_to_download.append({
                "repo_id": "meta-llama/Llama-3.1-70B-Instruct",
                "local_dir": "./Llama-3.1-70B-Instruct",
                "description": "Llama 3.1 70B Instruct"
            })
        
        if "qwen" in sys.argv:
            models_to_download.append({
                "repo_id": "Qwen/Qwen2.5-72B-Instruct",
                "local_dir": "./Qwen2.5-72B-Instruct", 
                "description": "Qwen2.5 72B Instruct"
            })
        
        if "deepseek" in sys.argv:
            models_to_download.append({
                "repo_id": "deepseek-ai/DeepSeek-V2.5",
                "local_dir": "./DeepSeek-V2.5",
                "description": "DeepSeek V2.5"
            })
            
        if "basic" in sys.argv:
            models_to_download.append({
                "repo_id": "codellama/CodeLlama-7b-Instruct-hf",
                "local_dir": "../CodeLlama-7b-Instruct-hf",
                "description": "Code Llama 7B Instruct"
            })
    
    if not models_to_download:
        print("❌ 다운로드할 모델이 선택되지 않았습니다.")
        return
    
    print(f"📋 총 {len(models_to_download)}개 모델을 다운로드합니다.")
    print("")
    
    success_count = 0
    for i, model in enumerate(models_to_download, 1):
        print(f"[{i}/{len(models_to_download)}] {model['description']}")
        if download_model(model["repo_id"], model["local_dir"], model["description"]):
            success_count += 1
        print("")
    
    print(f"🎉 다운로드 완료: {success_count}/{len(models_to_download)} 성공")
    
    # 모델 정보 파일 생성
    create_model_info_file(models_to_download, success_count)

def create_model_info_file(downloaded_models, success_count):
    """다운로드된 모델 정보 파일 생성"""
    info_content = f"""# AutoCI 다운로드된 모델 정보

## 다운로드 완료: {success_count}/{len(downloaded_models)}

### 사용 가능한 모델들:
"""
    
    for model in downloaded_models:
        if os.path.exists(model["local_dir"]):
            info_content += f"- ✅ {model['description']} - {model['local_dir']}\n"
        else:
            info_content += f"- ❌ {model['description']} - 다운로드 실패\n"
    
    info_content += f"""
### 사용법:
```python
# 모델 로드 예시
from transformers import AutoModelForCausalLM, AutoTokenizer

# Llama 3.1 70B (예시)
tokenizer = AutoTokenizer.from_pretrained("./Llama-3.1-70B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "./Llama-3.1-70B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True  # 메모리 절약
)
```

### 메모리 사용량:
- 4-bit 양자화 사용 시 모델당 ~25-30GB RAM
- 32GB RAM에서 1개 모델 동시 실행 권장
- GPU 메모리: 8GB+ 권장

### 성능 최적화:
- `load_in_4bit=True` 사용
- `device_map="auto"` 사용
- 필요시 CPU 오프로딩 활용
"""
    
    with open("../downloaded_models_info.md", "w", encoding="utf-8") as f:
        f.write(info_content)
    
    print("📄 모델 정보 파일 생성: downloaded_models_info.md")

if __name__ == "__main__":
    main()
EOF

# 선택에 따라 Python 스크립트 실행
args=""
if [ "$download_llama" = true ]; then
    args="$args llama"
fi
if [ "$download_qwen" = true ]; then
    args="$args qwen"
fi
if [ "$download_deepseek" = true ]; then
    args="$args deepseek"
fi
if [ "$download_basic" = true ]; then
    args="$args basic"
fi

# Python 스크립트 실행
python download_models.py $args

# 정리
rm -f download_models.py

echo ""
echo "🎉 모델 다운로드 작업이 완료되었습니다!"
echo ""
echo "📋 다운로드된 모델 확인:"
if [ -d "Llama-3.1-70B-Instruct" ]; then
    echo "  ✅ Llama 3.1 70B"
fi
if [ -d "Qwen2.5-72B-Instruct" ]; then
    echo "  ✅ Qwen2.5 72B"
fi
if [ -d "DeepSeek-V2.5" ]; then
    echo "  ✅ DeepSeek V2.5"
fi
if [ -d "../CodeLlama-7b-Instruct-hf" ]; then
    echo "  ✅ Code Llama 7B"
fi

echo ""
echo "📖 자세한 정보: downloaded_models_info.md 파일 확인"
echo "🚀 AutoCI 시작: python start_autoci_agent.py --advanced-models"
echo "" 