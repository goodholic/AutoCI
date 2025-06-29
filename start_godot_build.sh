#!/bin/bash
#
# AI Godot 빌드 시작 스크립트
# AutoCI를 위한 완전 제어 가능한 Godot 빌드
#

echo "🚀 AI Godot 빌드 준비 중..."

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 현재 디렉토리 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}        AI 완전 제어 Godot 빌드 시스템${NC}"
echo -e "${BLUE}===================================================${NC}"
echo ""

# 필수 도구 확인
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1이(가) 설치되지 않았습니다.${NC}"
        echo -e "${YELLOW}   설치 방법: $2${NC}"
        return 1
    else
        echo -e "${GREEN}✓ $1 확인됨${NC}"
        return 0
    fi
}

echo "🔍 필수 도구 확인 중..."
TOOLS_OK=1

check_tool python3 "sudo apt install python3" || TOOLS_OK=0
check_tool pip3 "sudo apt install python3-pip" || TOOLS_OK=0
check_tool git "sudo apt install git" || TOOLS_OK=0

# SCons 확인
if ! python3 -m pip show scons &> /dev/null; then
    echo -e "${YELLOW}📦 SCons 설치 중...${NC}"
    pip3 install scons
fi

# Windows 빌드 도구 확인
echo ""
echo -e "${YELLOW}⚠️  Windows 빌드 도구 확인${NC}"
echo "   Visual Studio Build Tools 또는 MinGW가 필요합니다."
echo "   설치되어 있지 않다면:"
echo "   1. https://visualstudio.microsoft.com/downloads/"
echo "   2. 'Build Tools for Visual Studio' 다운로드"
echo ""

# 빌드 옵션 선택
echo -e "${BLUE}빌드 옵션 선택:${NC}"
echo "1. 전체 빌드 (AI 기능 모두 포함) - 권장"
echo "2. 빠른 빌드 (기본 AI 기능만)"
echo "3. 디버그 빌드 (개발용)"
echo ""
read -p "선택 (1-3) [기본: 1]: " BUILD_OPTION

BUILD_OPTION=${BUILD_OPTION:-1}

# 빌드 시작
echo ""
echo -e "${GREEN}🔨 빌드를 시작합니다...${NC}"
echo ""

# Python 스크립트 실행
if [ "$BUILD_OPTION" == "2" ]; then
    python3 build_ai_godot.py --quick
elif [ "$BUILD_OPTION" == "3" ]; then
    python3 build_ai_godot.py --debug
else
    python3 build_ai_godot.py
fi

# 빌드 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}===================================================${NC}"
    echo -e "${GREEN}🎉 빌드 성공!${NC}"
    echo -e "${GREEN}===================================================${NC}"
    echo ""
    echo "다음 단계:"
    echo "1. AutoCI 실행: autoci"
    echo "2. AI Godot이 자동으로 감지됩니다"
else
    echo ""
    echo -e "${RED}===================================================${NC}"
    echo -e "${RED}❌ 빌드 실패${NC}"
    echo -e "${RED}===================================================${NC}"
    echo ""
    echo "문제 해결:"
    echo "1. godot_build.log 파일 확인"
    echo "2. Visual Studio Build Tools 설치 확인"
    echo "3. GitHub Issues에 문제 보고"
fi