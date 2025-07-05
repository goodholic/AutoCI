#!/bin/bash
# AutoCI 자동 가상환경 활성화 및 실행 스크립트

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 가상환경 존재 확인
if [ ! -d "autoci_env" ]; then
    echo "🔧 가상환경이 없습니다. 생성 중..."
    python3 -m venv autoci_env
    
    # pip 업그레이드
    ./autoci_env/bin/pip install --upgrade pip
    
    # 기본 패키지 설치
    echo "📦 필수 패키지 설치 중..."
    ./autoci_env/bin/pip install -r requirements_minimal.txt 2>/dev/null || true
fi

# 가상환경 활성화하여 Python 스크립트 실행
./autoci_env/bin/python autoci "$@"