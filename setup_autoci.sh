#!/bin/bash
# AutoCI 설정 스크립트

echo "🚀 AutoCI 설정 중..."

# 현재 디렉토리
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 실행 권한 부여
chmod +x "$SCRIPT_DIR/autoci"
chmod +x "$SCRIPT_DIR"/*.py

# PATH에 추가 (이미 없는 경우)
if ! echo $PATH | grep -q "$SCRIPT_DIR"; then
    echo "export PATH=\"$SCRIPT_DIR:\$PATH\"" >> ~/.bashrc
    echo "✅ PATH에 AutoCI 추가됨"
fi

# 별칭 추가 (선택사항)
if ! grep -q "alias autoci=" ~/.bashrc; then
    echo "alias autoci='$SCRIPT_DIR/autoci'" >> ~/.bashrc
    echo "✅ autoci 별칭 추가됨"
fi

echo ""
echo "✅ 설정 완료!"
echo ""
echo "다음 명령을 실행하여 설정을 적용하세요:"
echo "  source ~/.bashrc"
echo ""
echo "그 후 터미널 어디서나 'autoci'를 입력하면 시작됩니다!"
echo ""
echo "사용 예시:"
echo "  autoci                    # 대화형 모드"
echo "  autoci enhance start      # 24시간 시스템"
echo "  autoci dual start         # Dual Phase 시스템"
echo "  autoci help               # 도움말"