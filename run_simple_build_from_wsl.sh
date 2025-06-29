#!/bin/bash
# WSL에서 Windows 배치 파일 실행

echo "🚀 WSL에서 간단한 AI Godot 설정 시작..."
echo "=" * 50

# 현재 디렉토리 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "현재 디렉토리: $SCRIPT_DIR"

# Windows 경로로 변환
WIN_PATH=$(wslpath -w "$SCRIPT_DIR/RUN_SIMPLE_BUILD.bat")
echo "Windows 경로: $WIN_PATH"

# cmd.exe를 통해 배치 파일 실행
echo ""
echo "Windows에서 빌드 스크립트 실행 중..."
cmd.exe /c "$WIN_PATH"

echo ""
echo "✅ 완료! 이제 'autoci' 명령어를 사용할 수 있습니다."