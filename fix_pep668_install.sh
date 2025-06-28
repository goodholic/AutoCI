#!/bin/bash

# AutoCI PEP 668 Problem Fix for Ubuntu 24.04
# 이 스크립트는 "externally-managed-environment" 에러를 해결합니다

echo "🔧 AutoCI PEP 668 문제 해결 스크립트"
echo "===================================="
echo ""

# 현재 디렉토리 확인
echo "현재 위치: $(pwd)"
echo ""

# Python3와 venv가 설치되어 있는지 확인
echo "📦 Python 설치 확인..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3가 설치되어 있지 않습니다. 설치 중..."
    sudo apt update
    sudo apt install -y python3 python3-venv python3-pip python3-full
fi

# 기존 가상환경이 있다면 제거
if [ -d "autoci_env" ]; then
    echo "🗑️ 기존 가상환경 제거 중..."
    rm -rf autoci_env
fi

# 새로운 가상환경 생성
echo "🐍 새 가상환경 생성 중..."
python3 -m venv autoci_env

# 가상환경 활성화
echo "✅ 가상환경 활성화 중..."
source autoci_env/bin/activate

# 가상환경이 제대로 활성화되었는지 확인
echo "🔍 가상환경 확인:"
echo "  Python 경로: $(which python)"
echo "  Python 버전: $(python --version)"
echo "  가상환경 경로: $VIRTUAL_ENV"
echo ""

# pip 업그레이드
echo "⬆️ pip 업그레이드 중..."
python -m pip install --upgrade pip

# 기본 패키지 설치
echo "📦 기본 패키지 설치 중..."
python -m pip install wheel setuptools

# requirements.txt가 있다면 설치
if [ -f "requirements.txt" ]; then
    echo "📋 requirements.txt에서 패키지 설치 중..."
    python -m pip install -r requirements.txt
else
    echo "📦 AutoCI 기본 패키지 설치 중..."
    python -m pip install \
        torch \
        transformers \
        accelerate \
        sentencepiece \
        fastapi \
        uvicorn \
        psutil \
        rich \
        pyyaml
fi

echo ""
echo "✅ 설치 완료!"
echo ""
echo "🚀 가상환경 사용법:"
echo "  1. 활성화: source autoci_env/bin/activate"
echo "  2. 비활성화: deactivate"
echo ""
echo "💡 앞으로 AutoCI를 실행할 때는 반드시 가상환경을 활성화하세요:"
echo "   source autoci_env/bin/activate"
echo "   python start_autoci_agent.py"
echo "" 