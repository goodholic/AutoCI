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

# learn 명령어 처리
if [ "$1" = "learn" ]; then
    shift  # 'learn' 제거
    # learn 하위 명령어 처리
    if [ "$1" = "24h" ] || [ "$1" = "marathon" ]; then
        exec python3 "$AUTOCI_DIR/autoci.py" --learn-24h
    elif [ "$1" = "all" ]; then
        exec python3 "$AUTOCI_DIR/autoci.py" --learn-all
    else
        # 기본 learn 메뉴
        exec python3 "$AUTOCI_DIR/autoci.py" --learn "$@"
    fi
else
    # 기본 autoci 실행
    exec python3 "$AUTOCI_DIR/autoci.py" "$@"
fi
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
chmod +x "$AUTOCI_DIR/autoci.py"
chmod +x "$AUTOCI_DIR/autoci"

# 6. 디렉토리 생성
echo -e "${GREEN}📁 필요한 디렉토리 생성 중...${NC}"
mkdir -p "$AUTOCI_DIR/admin"
mkdir -p "$AUTOCI_DIR/logs"
mkdir -p "$AUTOCI_DIR/game_projects"
mkdir -p "$AUTOCI_DIR/csharp_24h_learning"
mkdir -p "$AUTOCI_DIR/godot_development"

# 7. 설치 완료 메시지
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ AutoCI 전역 명령어 설치 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "사용 가능한 명령어:"
echo "  autoci              # 기본 실행"
echo "  autoci --help       # 도움말"
echo "  autoci --status     # 시스템 상태"
echo "  autoci learn        # C# 학습 메뉴"
echo "  autoci learn 24h    # 24시간 학습 마라톤"
echo "  autoci learn all    # 전체 주제 학습"
echo "  autoci --godot      # Godot AI 데모"
echo "  autoci --production # 24시간 개발 모드"
echo ""
echo "⚠️  새 터미널을 열거나 다음 명령어를 실행하세요:"
echo "source ~/.bashrc"
echo ""