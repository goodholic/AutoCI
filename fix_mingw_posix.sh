#!/bin/bash
"""
MinGW posix threads 설정 자동 수정 스크립트
AI 수정된 Godot Windows 빌드를 위한 환경 설정
"""

# 색상 코드
GREEN='\033[92m'
YELLOW='\033[93m'
RED='\033[91m'
BLUE='\033[94m'
RESET='\033[0m'

echo -e "${BLUE}🔧 MinGW posix threads 설정 중...${RESET}"

# 1. 현재 설정 확인
echo -e "${YELLOW}📋 현재 MinGW 설정 확인:${RESET}"
update-alternatives --display x86_64-w64-mingw32-g++ 2>/dev/null || echo "  설정되지 않음"

# 2. posix 버전으로 자동 설정
echo -e "${BLUE}🔄 posix threads 버전으로 자동 변경 중...${RESET}"

# gcc
sudo update-alternatives --install /usr/bin/x86_64-w64-mingw32-gcc x86_64-w64-mingw32-gcc /usr/bin/x86_64-w64-mingw32-gcc-posix 60
sudo update-alternatives --install /usr/bin/x86_64-w64-mingw32-gcc x86_64-w64-mingw32-gcc /usr/bin/x86_64-w64-mingw32-gcc-win32 50

# g++
sudo update-alternatives --install /usr/bin/x86_64-w64-mingw32-g++ x86_64-w64-mingw32-g++ /usr/bin/x86_64-w64-mingw32-g++-posix 60
sudo update-alternatives --install /usr/bin/x86_64-w64-mingw32-g++ x86_64-w64-mingw32-g++ /usr/bin/x86_64-w64-mingw32-g++-win32 50

# 자동으로 posix 선택
sudo update-alternatives --set x86_64-w64-mingw32-gcc /usr/bin/x86_64-w64-mingw32-gcc-posix
sudo update-alternatives --set x86_64-w64-mingw32-g++ /usr/bin/x86_64-w64-mingw32-g++-posix

# 3. 설정 확인
echo -e "${GREEN}✅ 설정 완료! 확인 중...${RESET}"
x86_64-w64-mingw32-g++ --version | head -1

# 4. posix threads 지원 테스트
echo -e "${BLUE}🧪 posix threads 지원 테스트...${RESET}"
echo '#include <thread>' > /tmp/test_mingw.cpp
echo 'int main(){ std::thread t; return 0; }' >> /tmp/test_mingw.cpp

if x86_64-w64-mingw32-g++ -std=c++11 /tmp/test_mingw.cpp -o /tmp/test_mingw.exe 2>/dev/null; then
    echo -e "${GREEN}✅ posix threads 지원 확인됨${RESET}"
    rm -f /tmp/test_mingw.cpp /tmp/test_mingw.exe
    echo -e "\n${GREEN}🎉 MinGW 설정 완료! AI Godot Windows 빌드 준비됨${RESET}"
    echo -e "${YELLOW}다음 명령으로 빌드를 진행하세요:${RESET}"
    echo -e "  ${BLUE}build-godot${RESET}"
else
    echo -e "${RED}❌ posix threads 지원 실패${RESET}"
    echo -e "${YELLOW}수동 설정이 필요할 수 있습니다:${RESET}"
    echo -e "  sudo update-alternatives --config x86_64-w64-mingw32-g++"
fi 