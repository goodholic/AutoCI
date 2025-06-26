#!/bin/bash

echo "🚀 WSL에서 AutoCI ChatGPT 수준 한국어 AI 한 번에 설치"
echo "=================================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 현재 디렉토리 확인
AUTOCI_DIR="$(pwd)"
echo -e "${BLUE}📁 설치 디렉토리: $AUTOCI_DIR${NC}"

# 필수 파일들 확인
echo -e "${YELLOW}🔍 필수 파일 확인 중...${NC}"
REQUIRED_FILES=("autoci" "autoci_interactive.py")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ $file${NC}"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo -e "${RED}⚠️  누락된 파일이 있습니다. AutoCI 디렉토리에서 실행하세요.${NC}"
    exit 1
fi

# 실행 권한 부여
echo -e "${YELLOW}🔧 실행 권한 설정 중...${NC}"
chmod +x "$AUTOCI_DIR/autoci"
chmod +x "$AUTOCI_DIR/install_dependencies_wsl.sh"
echo -e "${GREEN}✅ 실행 권한 부여 완료${NC}"

# 의존성 설치
echo -e "${YELLOW}📦 Python 의존성 설치 중...${NC}"
bash "$AUTOCI_DIR/install_dependencies_wsl.sh"

# PATH 설정
echo -e "${YELLOW}🛠️  PATH 환경변수 설정 중...${NC}"
BASHRC_FILE="$HOME/.bashrc"

# 기존 AutoCI PATH 제거 (업그레이드 대비)
if grep -q "# AutoCI 명령어 PATH" "$BASHRC_FILE" 2>/dev/null; then
    echo -e "${YELLOW}🔄 기존 AutoCI PATH 설정 업데이트 중...${NC}"
    sed -i '/# AutoCI 명령어 PATH/,+1d' "$BASHRC_FILE"
fi

# 새로운 PATH 추가
echo "" >> "$BASHRC_FILE"
echo "# AutoCI 명령어 PATH 추가 (ChatGPT 수준 한국어 AI)" >> "$BASHRC_FILE"
echo "export PATH=\"\$PATH:$AUTOCI_DIR\"" >> "$BASHRC_FILE"

# 현재 세션에서도 PATH 적용
export PATH="$PATH:$AUTOCI_DIR"

echo -e "${GREEN}✅ PATH 설정 완료${NC}"

# alias 설정 (한국어 명령어 지원)
echo -e "${YELLOW}🌏 한국어 명령어 alias 설정 중...${NC}"

# 기존 alias 제거
sed -i '/# AutoCI 한국어 alias/,+10d' "$BASHRC_FILE" 2>/dev/null

# 새로운 alias 추가
cat >> "$BASHRC_FILE" << 'EOF'

# AutoCI 한국어 alias (ChatGPT 수준 AI)
alias 오토시아이='autoci'
alias 한국어ai='autoci korean'
alias ai대화='autoci korean'
alias 코드생성='autoci create'
alias 코드수정='autoci modify'
alias 코드개선='autoci improve'
alias 버그수정='autoci fix'
alias 도움말='autoci help'
EOF

echo -e "${GREEN}✅ 한국어 alias 설정 완료${NC}"

# 테스트 실행
echo -e "${CYAN}🧪 설치 테스트 중...${NC}"
if ./autoci help > /dev/null 2>&1; then
    echo -e "${GREEN}✅ AutoCI 명령어 테스트 성공${NC}"
else
    echo -e "${RED}❌ AutoCI 명령어 테스트 실패${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 설치 완료!${NC}"
echo -e "${CYAN}=================================================${NC}"
echo ""
echo -e "${BLUE}🚀 사용 방법:${NC}"
echo -e "${YELLOW}1. 새 터미널 열기 또는 다음 명령어 실행:${NC}"
echo -e "   ${CYAN}source ~/.bashrc${NC}"
echo ""
echo -e "${YELLOW}2. 어디서든 다음 명령어로 사용:${NC}"
echo -e "   ${GREEN}autoci${NC}               # ChatGPT 수준 한국어 AI"
echo -e "   ${GREEN}autoci korean${NC}        # 한국어 대화 모드"
echo -e "   ${GREEN}autoci k${NC}             # 한국어 모드 (단축키)"
echo -e "   ${GREEN}autoci 한국어${NC}        # 한글 명령어"
echo -e "   ${GREEN}오토시아이${NC}            # 한국어 alias"
echo -e "   ${GREEN}한국어ai${NC}             # 한국어 AI alias"
echo -e "   ${GREEN}ai대화${NC}               # AI 대화 alias"
echo ""
echo -e "${YELLOW}3. 빠른 코드 작업:${NC}"
echo -e "   ${GREEN}autoci c PlayerController${NC}    # 코드 생성"
echo -e "   ${GREEN}코드생성 PlayerController${NC}   # 한국어로"
echo -e "   ${GREEN}autoci m script.cs${NC}           # 코드 수정"
echo -e "   ${GREEN}코드수정 script.cs${NC}           # 한국어로"
echo ""
echo -e "${YELLOW}💬 대화 예시:${NC}"
echo -e "   ${CYAN}autoci${NC}"
echo -e "   > ${GREEN}안녕하세요! Unity 게임 개발 도와주세요${NC}"
echo -e "   > ${GREEN}너랑 대화할 수 있어?${NC}"
echo -e "   > ${GREEN}PlayerController 스크립트 만들어줘${NC}"
echo -e "   > ${GREEN}이 코드 어떻게 개선할 수 있을까요?${NC}"
echo ""
echo -e "${BLUE}📋 전체 명령어: ${CYAN}autoci help${NC}"
echo ""
echo -e "${YELLOW}⚡ 빠른 시작:${NC}"
echo -e "   ${CYAN}source ~/.bashrc && autoci${NC}" 