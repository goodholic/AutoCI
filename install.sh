#!/bin/bash
# AutoCI 글로벌 설치 스크립트

echo "🚀 AutoCI 글로벌 설치 시작..."

# 현재 디렉토리 저장
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 심볼릭 링크 생성 디렉토리
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    # Linux/Mac
    LINK_DIR="/usr/local/bin"
    SCRIPT_NAME="autoci.sh"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows Git Bash
    LINK_DIR="/usr/bin"
    SCRIPT_NAME="autoci.sh"
else
    echo "❌ 지원하지 않는 운영체제입니다."
    exit 1
fi

# 권한 확인
if [ ! -w "$LINK_DIR" ]; then
    echo "⚠️  관리자 권한이 필요합니다."
    echo "다음 명령어를 실행하세요:"
    echo "  sudo $0"
    exit 1
fi

# 기존 링크 제거
if [ -L "$LINK_DIR/autoci" ]; then
    rm "$LINK_DIR/autoci"
    echo "✅ 기존 링크 제거됨"
fi

# 새 심볼릭 링크 생성
ln -s "$INSTALL_DIR/$SCRIPT_NAME" "$LINK_DIR/autoci"

# 실행 권한 설정
chmod +x "$INSTALL_DIR/$SCRIPT_NAME"
chmod +x "$INSTALL_DIR/autoci"

echo "✅ AutoCI가 성공적으로 설치되었습니다!"
echo ""
echo "이제 어디서나 다음 명령어를 사용할 수 있습니다:"
echo "  autoci                    # 대화형 모드"
echo "  autoci learn              # AI 학습"
echo "  autoci learn low          # 메모리 최적화 학습"
echo "  autoci fix                # AI 엔진 업데이트"
echo "  autoci monitor            # 모니터링 대시보드"
echo "  autoci chat               # 한글 대화 모드"
echo ""
echo "💡 팁: 가상환경은 자동으로 활성화됩니다!"