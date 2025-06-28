#!/bin/bash

# Hugging Face 인증 설정 스크립트
# Llama 3.1 70B 등 gated model 접근을 위한 인증

echo "🔐 Hugging Face 인증 설정"
echo "========================"
echo ""
echo "📝 Llama 3.1 70B, Qwen2.5 72B 등 고급 모델 사용을 위해 인증이 필요합니다."
echo ""

# 가상환경 확인
if [ -z "$VIRTUAL_ENV" ] && [ -d "autoci_env" ]; then
    echo "🔄 가상환경 활성화 중..."
    source autoci_env/bin/activate
fi

# huggingface-hub 설치
echo "📦 huggingface-hub 설치 중..."
python -m pip install huggingface-hub --upgrade --quiet

echo ""
echo "🔑 Hugging Face 인증 설정 방법:"
echo ""
echo "1단계: Hugging Face 계정 생성"
echo "   - https://huggingface.co 에서 무료 계정 생성"
echo ""
echo "2단계: 액세스 토큰 생성"
echo "   - https://huggingface.co/settings/tokens 방문"
echo "   - 'New token' 클릭"
echo "   - Name: AutoCI"
echo "   - Type: Write 선택"
echo "   - 생성된 토큰 복사"
echo ""
echo "3단계: 모델 접근 승인 요청"
echo "   - https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct 방문"
echo "   - 'Request access' 버튼 클릭"
echo "   - 몇 분~몇 시간 후 승인됨"
echo ""

echo "4단계: 터미널에서 로그인"
echo "huggingface-cli login 명령어를 실행하고 토큰을 입력하세요."
echo ""

read -p "지금 로그인하시겠습니까? (y/n): " response

if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo ""
    echo "🔐 Hugging Face 로그인 중..."
    echo ""
    echo "💡 Tip: 토큰을 붙여넣기 할 때는 Ctrl+Shift+V를 사용하세요."
    echo ""
    
    huggingface-cli login
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 로그인 성공!"
        echo ""
        echo "이제 다음 명령어로 고급 모델을 다운로드할 수 있습니다:"
        echo "./download_advanced_models.sh"
        echo ""
        
        # 로그인 상태 확인
        echo "🔍 로그인 상태 확인 중..."
        python -c "
from huggingface_hub import whoami
try:
    user_info = whoami()
    print(f'✅ 로그인됨: {user_info[\"name\"]}')
except Exception as e:
    print(f'❌ 로그인 실패: {e}')
"
    else
        echo ""
        echo "❌ 로그인 실패"
        echo "토큰을 다시 확인하고 재시도하세요."
    fi
else
    echo ""
    echo "📋 수동 로그인 방법:"
    echo "1. 터미널에서 실행: huggingface-cli login"
    echo "2. 토큰 입력"
    echo "3. 고급 모델 다운로드: ./download_advanced_models.sh"
fi

echo ""
echo "🆓 인증 없이 사용 가능한 대안:"
echo "./download_free_models.sh 를 실행하면 인증 없이도"
echo "고성능 모델들(Code Llama 7B/13B, Mistral 7B 등)을 사용할 수 있습니다."
echo ""

# 토큰 저장 위치 안내
echo "💾 토큰 저장 위치: ~/.cache/huggingface/token"
echo "🔒 보안: 토큰을 안전하게 보관하세요."
echo "" 