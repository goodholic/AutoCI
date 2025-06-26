#!/bin/bash

echo "🔧 AutoCI 명령어 설정"
echo "====================="

# 현재 디렉토리 경로
AUTOCI_DIR="$(pwd)"
echo "📁 AutoCI 디렉토리: $AUTOCI_DIR"

# autoci 파일에 실행 권한 부여
chmod +x "$AUTOCI_DIR/autoci"
echo "✅ autoci 실행 권한 부여 완료"

# WSL 사용자의 bashrc 파일 경로
BASHRC_FILE="$HOME/.bashrc"

# PATH에 AutoCI 디렉토리 추가 (중복 방지)
if ! grep -q "export PATH.*$AUTOCI_DIR" "$BASHRC_FILE" 2>/dev/null; then
    echo "" >> "$BASHRC_FILE"
    echo "# AutoCI 명령어 PATH 추가 (자동 생성)" >> "$BASHRC_FILE"
    echo "export PATH=\"\$PATH:$AUTOCI_DIR\"" >> "$BASHRC_FILE"
    echo "✅ ~/.bashrc에 PATH 추가 완료"
else
    echo "ℹ️  이미 PATH에 등록되어 있습니다"
fi

# 현재 세션에서도 PATH 추가
export PATH="$PATH:$AUTOCI_DIR"

echo ""
echo "🎉 설정 완료!"
echo "===================="
echo ""
echo "🚀 사용 방법:"
echo "1. 새 터미널 열기 또는 'source ~/.bashrc' 실행"
echo "2. 어디서든 'autoci' 명령어 사용 가능"
echo ""
echo "📋 주요 명령어:"
echo "  autoci                    # ChatGPT 수준 한국어 AI 대화 모드"
echo "  autoci korean             # 한국어 AI 대화 모드"
echo "  autoci k                  # 한국어 모드 (단축키)"
echo "  autoci 한국어             # 한글 명령어"
echo "  autoci help               # 전체 도움말"
echo "  autoci terminal           # 기존 터미널 모드"
echo ""
echo "💬 테스트 예시:"
echo "  autoci"
echo "  > 안녕하세요! Unity 게임 개발 도와주세요"
echo "  > 너랑 대화할 수 있어?"
echo "  > PlayerController 스크립트 만들어줘"

echo ""
echo "⚠️  참고: 새 터미널에서 또는 'source ~/.bashrc' 후 사용하세요"

# 테스트 실행
echo ""
echo "🧪 현재 세션 테스트:"
if ./autoci help > /dev/null 2>&1; then
    echo "✅ autoci 명령어 테스트 성공"
    echo ""
    echo "📋 업그레이드된 기능 미리보기:"
    ./autoci help | grep -A 5 "기본 명령어"
else
    echo "❌ autoci 명령어 테스트 실패"
fi 