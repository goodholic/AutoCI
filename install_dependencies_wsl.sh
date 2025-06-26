#!/bin/bash

echo "🔧 AutoCI WSL 의존성 설치"
echo "========================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 현재 디렉토리
AUTOCI_DIR="$(pwd)"
echo -e "${BLUE}📁 AutoCI 디렉토리: $AUTOCI_DIR${NC}"

# 가상환경 찾기
VENV_PATH=""
if [ -d "llm_venv_wsl" ]; then
    VENV_PATH="llm_venv_wsl"
elif [ -d "llm_venv" ]; then
    VENV_PATH="llm_venv"
elif [ -d "venv" ]; then
    VENV_PATH="venv"
fi

if [ -z "$VENV_PATH" ]; then
    echo -e "${YELLOW}⚠️  가상환경을 찾을 수 없습니다. 새로 만듭니다.${NC}"
    python3 -m venv llm_venv_wsl
    VENV_PATH="llm_venv_wsl"
fi

echo -e "${GREEN}✅ 가상환경: $VENV_PATH${NC}"

# 가상환경 활성화
source "$VENV_PATH/bin/activate"

echo -e "${YELLOW}📦 필수 패키지 설치 중...${NC}"

# 필수 패키지 목록
PACKAGES=(
    "rich>=12.0.0"
    "colorama>=0.4.0" 
    "psutil>=5.9.0"
    "requests>=2.28.0"
)

# 패키지 설치
for package in "${PACKAGES[@]}"; do
    echo -e "${BLUE}Installing: $package${NC}"
    pip install "$package"
done

echo ""
echo -e "${GREEN}🎉 의존성 설치 완료!${NC}"

# 설치된 패키지 확인
echo -e "${YELLOW}📋 설치된 패키지:${NC}"
pip list | grep -E "(rich|colorama|psutil|requests)"

echo ""
echo -e "${BLUE}✅ 이제 autoci를 실행할 수 있습니다!${NC}"
echo -e "${CYAN}   ./autoci${NC}" 