#!/bin/bash
# Godot 관련 명령어 설치 스크립트

echo "🔧 Godot 명령어 설치 중..."

# 현재 디렉토리
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 실행 권한 설정
chmod +x "$SCRIPT_DIR/build-godot"
chmod +x "$SCRIPT_DIR/check-godot"

# build-godot 설치
if [ -f /usr/local/bin/build-godot ]; then
    echo "기존 build-godot 명령어 제거 중..."
    sudo rm /usr/local/bin/build-godot
fi

echo "build-godot 명령어 설치 중..."
sudo ln -s "$SCRIPT_DIR/build-godot" /usr/local/bin/build-godot

# check-godot 설치
if [ -f /usr/local/bin/check-godot ]; then
    echo "기존 check-godot 명령어 제거 중..."
    sudo rm /usr/local/bin/check-godot
fi

echo "check-godot 명령어 설치 중..."
sudo ln -s "$SCRIPT_DIR/check-godot" /usr/local/bin/check-godot

echo "✅ 설치 완료!"
echo ""
echo "사용 가능한 명령어:"
echo "  build-godot - AI Godot 빌드"
echo "  check-godot - AI Godot 상태 확인"
echo ""
echo "WSL 터미널 어디서나 사용할 수 있습니다."