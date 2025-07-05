#!/bin/bash

# AutoCI 전역 명령어 설치 스크립트
echo "🚀 AutoCI 전역 명령어 설치 시작"
echo "========================================"

# AutoCI 경로 설정
AUTOCI_DIR="/mnt/d/AutoCI/AutoCI"
BIN_DIR="/usr/local/bin"

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 권한 확인
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}⚠️  sudo 권한이 필요합니다. 다시 실행해주세요:${NC}"
    echo "sudo ./install_global_autoci.sh"
    exit 1
fi

# 1. autoci 명령어 생성
echo -e "${GREEN}📦 autoci 명령어 생성 중...${NC}"
cat > "$BIN_DIR/autoci" << 'EOF'
#!/bin/bash
# AutoCI 전역 실행 스크립트

AUTOCI_DIR="/mnt/d/AutoCI/AutoCI"
cd "$AUTOCI_DIR" || exit 1

# admin/autoci Python 스크립트 직접 실행
# admin/autoci는 가상환경을 자동으로 활성화합니다
exec python3 "$AUTOCI_DIR/admin/autoci" "$@"
EOF

# 2. 실행 권한 부여
chmod +x "$BIN_DIR/autoci"
echo -e "${GREEN}✅ autoci 명령어 설치 완료${NC}"

# 3. 환경 변수 설정
echo -e "${GREEN}🔧 환경 변수 설정 중...${NC}"

# ~/.bashrc에 추가
BASHRC_FILE="$HOME/.bashrc"
if ! grep -q "AUTOCI_DIR" "$BASHRC_FILE"; then
    echo "" >> "$BASHRC_FILE"
    echo "# AutoCI Environment Variables" >> "$BASHRC_FILE"
    echo "export AUTOCI_DIR=\"$AUTOCI_DIR\"" >> "$BASHRC_FILE"
    echo "export PATH=\"\$PATH:$AUTOCI_DIR\"" >> "$BASHRC_FILE"
fi

# 4. Python 의존성 설치
echo -e "${GREEN}📦 Python 의존성 설치 중...${NC}"
cd "$AUTOCI_DIR"

# 가상환경이 없으면 생성
if [ ! -d "autoci_env" ]; then
    python3 -m venv autoci_env
fi

# 가상환경 활성화 및 의존성 설치
source autoci_env/bin/activate
pip install --upgrade pip
pip install psutil aiofiles

# 5. 권한 설정
echo -e "${GREEN}🔐 권한 설정 중...${NC}"
chmod +x "$AUTOCI_DIR/core/autoci.py"
chmod +x "$AUTOCI_DIR/admin/autoci"

# 6. 디렉토리 생성
echo -e "${GREEN}📁 필요한 디렉토리 생성 중...${NC}"
mkdir -p "$AUTOCI_DIR/admin"
mkdir -p "$AUTOCI_DIR/logs"
mkdir -p "$AUTOCI_DIR/game_projects"
mkdir -p "$AUTOCI_DIR/python_24h_learning"
mkdir -p "$AUTOCI_DIR/panda3d_development"
mkdir -p "$AUTOCI_DIR/panda3d_ai_improved"

# 7. 설치 완료 메시지
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ AutoCI 전역 명령어 설치 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "사용 가능한 명령어:"
echo "  autoci              # 24시간 자동 개발 시작"
echo "  autoci --help       # 도움말"
echo "  autoci --status     # 시스템 상태"
echo "  autoci learn        # AI 모델 기반 연속 학습"
echo "  autoci learn low    # 메모리 최적화 연속 학습"
echo "  autoci fix          # 학습을 토대로 AI의 게임 제작 능력 업데이트"
echo "  autoci chat         # 한글 대화 모드"
echo "  autoci --production # 24시간 개발 모드"
echo ""
echo "⚠️  새 터미널을 열거나 다음 명령어를 실행하세요:"
echo "source ~/.bashrc"
echo ""