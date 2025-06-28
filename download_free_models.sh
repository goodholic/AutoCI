#!/bin/bash

# AutoCI v3.0 - 무료 고성능 AI 모델 다운로드
# 인증 없이 사용 가능한 고성능 모델들

echo "🚀 AutoCI v3.0 - 무료 고성능 모델 다운로드"
echo "============================================"
echo ""
echo "✅ 장점: 인증 불필요, 즉시 사용 가능"
echo "📊 성능: 상용 모델 수준의 고품질"
echo "💾 크기: 7B-13B (적당한 메모리 사용)"
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
echo "📦 필요한 패키지 확인 중..."
python -m pip install huggingface-hub transformers torch --quiet

# 모델 저장 디렉토리 생성
mkdir -p models
cd models

echo ""
echo "사용 가능한 무료 고성능 모델:"
echo "1. Code Llama 7B (기본) - 코딩 전문, 빠른 속도"
echo "2. Code Llama 13B - 더 높은 성능, 중간 속도"
echo "3. Mistral 7B Instruct - 범용 고성능, 한국어 지원"
echo "4. OpenCodeInterpreter 6.7B - 코딩 전문, 최신 모델"
echo "5. 모든 추천 모델 다운로드 (권장)"
echo ""
read -p "선택하세요 (1-5): " choice

case $choice in
    1)
        echo "📦 Code Llama 7B 다운로드..."
        download_codelllama7b=true
        ;;
    2)
        echo "📦 Code Llama 13B 다운로드..."
        download_codelllama13b=true
        ;;
    3)
        echo "📦 Mistral 7B 다운로드..."
        download_mistral=true
        ;;
    4)
        echo "📦 OpenCodeInterpreter 다운로드..."
        download_opencodeinterpreter=true
        ;;
    5)
        echo "📦 모든 추천 모델 다운로드..."
        download_codelllama7b=true
        download_codelllama13b=true
        download_mistral=true
        download_opencodeinterpreter=true
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
cat > download_free_models.py << 'EOF'
#!/usr/bin/env python3
"""
AutoCI 무료 고성능 모델 다운로드 스크립트
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
        if "codelllama7b" in sys.argv:
            models_to_download.append({
                "repo_id": "codellama/CodeLlama-7b-Instruct-hf",
                "local_dir": "../CodeLlama-7b-Instruct-hf",
                "description": "Code Llama 7B Instruct"
            })
        
        if "codelllama13b" in sys.argv:
            models_to_download.append({
                "repo_id": "codellama/CodeLlama-13b-Instruct-hf",
                "local_dir": "./CodeLlama-13b-Instruct-hf",
                "description": "Code Llama 13B Instruct"
            })
        
        if "mistral" in sys.argv:
            models_to_download.append({
                "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
                "local_dir": "./Mistral-7B-Instruct-v0.3",
                "description": "Mistral 7B Instruct v0.3"
            })
            
        if "opencodeinterpreter" in sys.argv:
            models_to_download.append({
                "repo_id": "m-a-p/OpenCodeInterpreter-DS-6.7B",
                "local_dir": "./OpenCodeInterpreter-DS-6.7B",
                "description": "OpenCodeInterpreter DS 6.7B"
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
    info_content = f"""# AutoCI 무료 고성능 모델 정보

## 다운로드 완료: {success_count}/{len(downloaded_models)}

### 사용 가능한 모델들:
"""
    
    for model in downloaded_models:
        if os.path.exists(model["local_dir"]):
            info_content += f"- ✅ {model['description']} - {model['local_dir']}\n"
        else:
            info_content += f"- ❌ {model['description']} - 다운로드 실패\n"
    
    info_content += f"""

### 모델 특징:
- **Code Llama 7B**: 빠른 속도, 코딩 전문, 16GB RAM 권장
- **Code Llama 13B**: 높은 성능, 코딩 전문, 24GB RAM 권장  
- **Mistral 7B**: 범용 고성능, 한국어 지원, 16GB RAM 권장
- **OpenCodeInterpreter**: 최신 코딩 모델, 창의적 코딩, 16GB RAM 권장

### 사용법:
```python
# 모델 로드 예시
from transformers import AutoModelForCausalLM, AutoTokenizer

# Code Llama 7B (예시)
tokenizer = AutoTokenizer.from_pretrained("./CodeLlama-7b-Instruct-hf")
model = AutoModelForCausalLM.from_pretrained(
    "./CodeLlama-7b-Instruct-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### 메모리 사용량:
- **7B 모델**: 8-16GB RAM
- **13B 모델**: 16-24GB RAM
- GPU 메모리: 6GB+ 권장 (선택사항)

### 성능 순위 (추정):
1. Code Llama 13B - 최고 코딩 성능
2. Mistral 7B - 최고 범용 성능
3. OpenCodeInterpreter - 최신 코딩 모델
4. Code Llama 7B - 빠른 속도

### AutoCI 시작:
```bash
source autoci_env/bin/activate
python start_autoci_agent.py
```
"""
    
    with open("../free_models_info.md", "w", encoding="utf-8") as f:
        f.write(info_content)
    
    print("📄 모델 정보 파일 생성: free_models_info.md")

if __name__ == "__main__":
    main()
EOF

# 선택에 따라 Python 스크립트 실행
args=""
if [ "$download_codelllama7b" = true ]; then
    args="$args codelllama7b"
fi
if [ "$download_codelllama13b" = true ]; then
    args="$args codelllama13b"
fi
if [ "$download_mistral" = true ]; then
    args="$args mistral"
fi
if [ "$download_opencodeinterpreter" = true ]; then
    args="$args opencodeinterpreter"
fi

# Python 스크립트 실행
python download_free_models.py $args

# 정리
rm -f download_free_models.py

echo ""
echo "🎉 무료 모델 다운로드 작업이 완료되었습니다!"
echo ""
echo "📋 다운로드된 모델 확인:"
if [ -d "../CodeLlama-7b-Instruct-hf" ]; then
    echo "  ✅ Code Llama 7B (기본)"
fi
if [ -d "CodeLlama-13b-Instruct-hf" ]; then
    echo "  ✅ Code Llama 13B (고성능)"
fi
if [ -d "Mistral-7B-Instruct-v0.3" ]; then
    echo "  ✅ Mistral 7B (범용 고성능)"
fi
if [ -d "OpenCodeInterpreter-DS-6.7B" ]; then
    echo "  ✅ OpenCodeInterpreter (최신 코딩)"
fi

echo ""
echo "📖 자세한 정보: free_models_info.md 파일 확인"
echo "🚀 AutoCI 시작: python start_autoci_agent.py"
echo ""
echo "💡 Tip: 이 모델들은 인증 없이 바로 사용 가능합니다!"
echo "" 