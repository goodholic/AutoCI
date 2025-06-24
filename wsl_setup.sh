#!/bin/bash
#
# WSL 환경을 위한 AutoCI 설정 스크립트
# 이 스크립트는 WSL에서 AutoCI를 실행하기 위한 환경을 설정합니다.
#

echo "🔧 WSL AutoCI 환경 설정 시작..."

# WSL 환경 확인
if grep -qi microsoft /proc/version; then
    echo "✅ WSL 환경이 확인되었습니다."
else
    echo "⚠️  WSL 환경이 아닙니다. 일반 Linux로 진행합니다."
fi

# 필요한 패키지 설치
echo ""
echo "📦 필요한 시스템 패키지 설치 중..."

# apt 업데이트
sudo apt update

# Python 3.8+ 확인 및 설치
if ! command -v python3 &> /dev/null; then
    echo "Python3 설치 중..."
    sudo apt install -y python3 python3-pip python3-venv
else
    echo "✅ Python3가 이미 설치되어 있습니다."
fi

# .NET SDK 설치 확인
if ! command -v dotnet &> /dev/null; then
    echo ".NET SDK 설치 중..."
    # Microsoft 패키지 저장소 추가
    wget https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
    sudo dpkg -i packages-microsoft-prod.deb
    rm packages-microsoft-prod.deb
    
    # .NET SDK 설치
    sudo apt update
    sudo apt install -y dotnet-sdk-8.0
else
    echo "✅ .NET SDK가 이미 설치되어 있습니다."
fi

# Git 설치 확인
if ! command -v git &> /dev/null; then
    echo "Git 설치 중..."
    sudo apt install -y git
else
    echo "✅ Git이 이미 설치되어 있습니다."
fi

# 기타 필요한 도구 설치
echo ""
echo "🔧 추가 도구 설치 중..."
sudo apt install -y curl wget build-essential

# Python 가상환경 생성
echo ""
echo "🐍 Python 가상환경 설정 중..."

# 기존 가상환경이 있다면 활성화, 없다면 생성
if [ -d "llm_venv" ]; then
    echo "✅ 기존 가상환경을 사용합니다."
    source llm_venv/bin/activate
else
    echo "새 가상환경 생성 중..."
    python3 -m venv llm_venv
    source llm_venv/bin/activate
fi

# pip 업그레이드
pip install --upgrade pip

# WSL 특화 설정
echo ""
echo "🌐 WSL 네트워크 설정 중..."

# WSL IP 주소 가져오기
WSL_IP=$(hostname -I | awk '{print $1}')
echo "WSL IP 주소: $WSL_IP"

# 환경 변수 설정
export ASPNETCORE_URLS="http://0.0.0.0:5049;http://0.0.0.0:7100"
export ASPNETCORE_ENVIRONMENT="Development"

# 방화벽 안내
echo ""
echo "🔥 Windows 방화벽 설정 안내:"
echo "Windows에서 WSL 서비스에 접근하려면 다음 PowerShell 명령을 관리자 권한으로 실행하세요:"
echo ""
echo "New-NetFirewallRule -DisplayName 'WSL Port 8000' -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow"
echo "New-NetFirewallRule -DisplayName 'WSL Port 8080' -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow"
echo "New-NetFirewallRule -DisplayName 'WSL Port 5049' -Direction Inbound -LocalPort 5049 -Protocol TCP -Action Allow"
echo "New-NetFirewallRule -DisplayName 'WSL Port 7100' -Direction Inbound -LocalPort 7100 -Protocol TCP -Action Allow"

# 실행 권한 부여
echo ""
echo "🔑 실행 권한 설정 중..."
chmod +x wsl_start_all.py
chmod +x start_expert_learning.py
chmod +x download_model.py

# 완료 메시지
echo ""
echo "✅ WSL 환경 설정이 완료되었습니다!"
echo ""
echo "🚀 AutoCI를 시작하려면 다음 명령을 실행하세요:"
echo "   python3 wsl_start_all.py"
echo ""
echo "📌 또는 기존 명령도 사용 가능합니다:"
echo "   python3 start_all.py"
echo ""
echo "💡 Windows에서 접속하려면:"
echo "   http://$WSL_IP:7100"
echo ""