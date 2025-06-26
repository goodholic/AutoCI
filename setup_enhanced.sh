#!/bin/bash
# AutoCI Enhanced Setup Script

echo "🚀 AutoCI Enhanced Setup"
echo "========================"

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Python 환경 확인
echo -e "${YELLOW}1. Python 환경 확인...${NC}"
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓ Python3 설치됨: $(python3 --version)${NC}"
else
    echo "❌ Python3가 설치되어 있지 않습니다."
    exit 1
fi

# 2. 필요한 디렉토리 생성
echo -e "${YELLOW}2. 디렉토리 구조 생성...${NC}"
mkdir -p expert_learning_data
mkdir -p learning_results
mkdir -p logs
mkdir -p models
mkdir -p rag_cache
echo -e "${GREEN}✓ 디렉토리 생성 완료${NC}"

# 3. Python 패키지 설치
echo -e "${YELLOW}3. Python 패키지 설치...${NC}"
pip3 install -r requirements_enhanced.txt
echo -e "${GREEN}✓ 패키지 설치 완료${NC}"

# 4. 실행 권한 설정
echo -e "${YELLOW}4. 실행 권한 설정...${NC}"
chmod +x autoci
chmod +x autoci_terminal.py
chmod +x enhanced_rag_system_v2.py
chmod +x advanced_indexer.py
chmod +x dual_phase_system.py
echo -e "${GREEN}✓ 실행 권한 설정 완료${NC}"

# 5. 심볼릭 링크 생성 (선택사항)
echo -e "${YELLOW}5. 전역 명령어 설정...${NC}"
if [ -w /usr/local/bin ]; then
    ln -sf "$(pwd)/autoci" /usr/local/bin/autoci
    echo -e "${GREEN}✓ 'autoci' 명령어를 어디서나 사용할 수 있습니다.${NC}"
else
    echo "⚠️  /usr/local/bin에 쓰기 권한이 없습니다."
    echo "   현재 디렉토리에서 ./autoci 로 실행하세요."
fi

# 6. 초기 설정 파일 생성
echo -e "${YELLOW}6. 설정 파일 생성...${NC}"
cat > autoci_config.json << EOF
{
    "rag_port": 8001,
    "api_port": 8002,
    "llm_port": 8000,
    "auto_switch_model": true,
    "training_batch_size": 32,
    "training_epochs": 3,
    "max_concurrent_tasks": 5,
    "task_check_interval": 30
}
EOF
echo -e "${GREEN}✓ 설정 파일 생성 완료${NC}"

# 7. 테스트
echo -e "${YELLOW}7. 시스템 테스트...${NC}"
python3 -c "import flask, requests, numpy, sklearn; print('✓ 모든 패키지 정상 로드')"

echo ""
echo -e "${GREEN}🎉 AutoCI Enhanced 설정 완료!${NC}"
echo ""
echo "사용 방법:"
echo "  1. 대화형 모드: autoci terminal"
echo "  2. 빠른 실행: autoci create PlayerController 클래스"
echo "  3. 전체 시작: autoci start"
echo "  4. 도움말: autoci help"
echo ""
echo "첫 실행 추천:"
echo "  autoci data index    # 기존 데이터 인덱싱"
echo "  autoci dual start    # Dual Phase System 시작"