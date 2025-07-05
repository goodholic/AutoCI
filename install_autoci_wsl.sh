#!/bin/bash
# AutoCI WSL 전역 명령어 설치 스크립트

echo "🚀 AutoCI WSL 설치 시작..."
echo "================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 현재 디렉토리 저장
AUTOCI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo -e "${GREEN}📁 AutoCI 디렉토리: $AUTOCI_DIR${NC}"

# WSL 확인
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo -e "${YELLOW}⚠️  WSL 환경이 아닌 것 같습니다. 계속하시겠습니까? (y/n)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "설치를 취소합니다."
        exit 0
    fi
fi

# Python 확인
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3가 설치되어 있지 않습니다.${NC}"
    echo "다음 명령어로 설치해주세요:"
    echo "  sudo apt update && sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✅ Python $PYTHON_VERSION 감지됨${NC}"

# 가상환경 확인 및 생성
VENV_DIR="$AUTOCI_DIR/autoci_env"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}📦 가상환경 생성 중...${NC}"
    python3 -m venv "$VENV_DIR"
    
    # pip 업그레이드
    "$VENV_DIR/bin/pip" install --upgrade pip
    
    # requirements.txt가 있으면 패키지 설치
    if [ -f "$AUTOCI_DIR/requirements.txt" ]; then
        echo -e "${YELLOW}📚 필요한 패키지 설치 중...${NC}"
        "$VENV_DIR/bin/pip" install -r "$AUTOCI_DIR/requirements.txt"
    fi
else
    echo -e "${GREEN}✅ 가상환경이 이미 존재합니다${NC}"
fi

# 실행 권한 부여
chmod +x "$AUTOCI_DIR/core/autoci_wsl_launcher.py"
chmod +x "$AUTOCI_DIR/core/autoci.py"
chmod +x "$AUTOCI_DIR/core/autoci_command.py"

# /usr/local/bin/autoci 스크립트 생성
AUTOCI_SCRIPT="/usr/local/bin/autoci"
TEMP_SCRIPT="/tmp/autoci_launcher.sh"

cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
# AutoCI WSL Launcher

# 프로젝트 디렉토리로 이동
cd "$AUTOCI_DIR"

# 가상환경의 Python으로 launcher 실행
exec "$VENV_DIR/bin/python" "$AUTOCI_DIR/core/autoci_wsl_launcher.py" "\$@"
EOF

# 스크립트 설치
echo -e "${YELLOW}🔧 전역 명령어 설치 중... (sudo 권한 필요)${NC}"
sudo cp "$TEMP_SCRIPT" "$AUTOCI_SCRIPT"
sudo chmod +x "$AUTOCI_SCRIPT"

# 임시 파일 삭제
rm -f "$TEMP_SCRIPT"

# PATH 확인
if ! echo "$PATH" | grep -q "/usr/local/bin"; then
    echo -e "${YELLOW}⚠️  /usr/local/bin이 PATH에 없습니다.${NC}"
    echo "다음을 ~/.bashrc에 추가해주세요:"
    echo '  export PATH="/usr/local/bin:$PATH"'
fi

# 설치 완료
echo ""
echo -e "${GREEN}✅ AutoCI 설치 완료!${NC}"
echo "================================"
echo ""
echo "사용 방법:"
echo -e "  ${GREEN}autoci${NC}                    # AutoCI 터미널 시작"
echo -e "  ${GREEN}autoci learn${NC}             # AI 학습 모드"
echo -e "  ${GREEN}autoci chat${NC}              # 한글 대화 모드"
echo -e "  ${GREEN}autoci monitor${NC}           # 실시간 모니터링"
echo -e "  ${GREEN}autoci --help${NC}            # 도움말"
echo ""
echo -e "${YELLOW}💡 팁: 새 터미널을 열거나 'source ~/.bashrc'를 실행하세요.${NC}"

# 바로 실행 테스트
echo ""
echo -e "${YELLOW}🧪 설치 테스트 중...${NC}"
if $AUTOCI_SCRIPT --version 2>/dev/null | grep -q "AutoCI"; then
    echo -e "${GREEN}✅ AutoCI가 정상적으로 작동합니다!${NC}"
else
    # 버전 정보가 없으면 도움말로 테스트
    if $AUTOCI_SCRIPT --help 2>/dev/null | grep -q "AutoCI"; then
        echo -e "${GREEN}✅ AutoCI가 정상적으로 설치되었습니다!${NC}"
    else
        echo -e "${YELLOW}⚠️  AutoCI 실행 테스트를 건너뜁니다.${NC}"
    fi
fi