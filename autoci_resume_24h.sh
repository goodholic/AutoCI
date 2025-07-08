#!/bin/bash
# AutoCI Resume 24시간 실행 스크립트

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 스크립트 디렉토리
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$SCRIPT_DIR/logs/24h_session"
mkdir -p "$LOG_DIR"

# 타임스탬프
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/autoci_resume_${TIMESTAMP}.log"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           AutoCI Resume 24시간 안정화 실행 스크립트              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# WSL 환경 확인
if grep -qi microsoft /proc/version; then
    echo -e "${GREEN}✓ WSL 환경 감지됨${NC}"
    IS_WSL=true
else
    echo -e "${YELLOW}⚠ 일반 Linux 환경${NC}"
    IS_WSL=false
fi

# Python 환경 확인
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python3를 찾을 수 없습니다${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python3 확인됨: $(python3 --version)${NC}"

# 프로젝트 경로 확인
if [ -z "$1" ]; then
    # Godot 프로젝트 자동 검색
    echo -e "${YELLOW}프로젝트 경로가 지정되지 않았습니다. 자동 검색 중...${NC}"
    
    # 여러 경로에서 Godot 프로젝트 검색
    SEARCH_PATHS=(
        "/home/$USER/Documents/Godot/Projects"
        "~/Documents/Godot/Projects"
        "//wsl.localhost/Ubuntu/home/$USER/Documents/Godot/Projects"
        "/mnt/c/Users/$USER/Documents/Godot/Projects"
    )
    
    PROJECT_PATH=""
    for path in "${SEARCH_PATHS[@]}"; do
        expanded_path=$(eval echo "$path")
        if [ -d "$expanded_path" ]; then
            # project.godot 파일이 있는 디렉토리 찾기
            PROJECT_PATH=$(find "$expanded_path" -name "project.godot" -type f 2>/dev/null | head -n 1 | xargs dirname)
            if [ -n "$PROJECT_PATH" ]; then
                break
            fi
        fi
    done
    
    if [ -z "$PROJECT_PATH" ]; then
        echo -e "${RED}✗ Godot 프로젝트를 찾을 수 없습니다${NC}"
        echo "사용법: $0 <프로젝트_경로>"
        exit 1
    fi
else
    PROJECT_PATH="$1"
fi

if [ ! -d "$PROJECT_PATH" ]; then
    echo -e "${RED}✗ 프로젝트 경로를 찾을 수 없습니다: $PROJECT_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 프로젝트 경로: $PROJECT_PATH${NC}"

# 이미 실행 중인 프로세스 확인
if pgrep -f "autoci_daemon.py" > /dev/null; then
    echo -e "${YELLOW}⚠ AutoCI 데몬이 이미 실행 중입니다${NC}"
    read -p "기존 프로세스를 종료하고 새로 시작하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "autoci_daemon.py"
        sleep 2
    else
        exit 0
    fi
fi

# 실행 방법 선택
echo ""
echo "실행 방법을 선택하세요:"
echo "1) 포그라운드 실행 (현재 터미널에서 실행)"
echo "2) 백그라운드 실행 (nohup 사용)"
echo "3) Screen 세션에서 실행 (권장)"
echo "4) Tmux 세션에서 실행"
echo "5) 데몬 모드로 실행 (자동 재시작 포함)"
read -p "선택 [1-5]: " choice

case $choice in
    1)
        echo -e "${BLUE}포그라운드에서 실행합니다...${NC}"
        cd "$PROJECT_PATH"
        python3 "$SCRIPT_DIR/autoci" resume 2>&1 | tee "$LOG_FILE"
        ;;
        
    2)
        echo -e "${BLUE}백그라운드에서 실행합니다...${NC}"
        cd "$PROJECT_PATH"
        nohup python3 "$SCRIPT_DIR/autoci" resume > "$LOG_FILE" 2>&1 &
        PID=$!
        echo -e "${GREEN}✓ PID: $PID${NC}"
        echo -e "${GREEN}✓ 로그 파일: $LOG_FILE${NC}"
        echo "로그 확인: tail -f $LOG_FILE"
        ;;
        
    3)
        if ! command -v screen &> /dev/null; then
            echo -e "${YELLOW}Screen이 설치되어 있지 않습니다. 설치 중...${NC}"
            sudo apt-get update && sudo apt-get install -y screen
        fi
        
        SESSION_NAME="autoci_resume_24h"
        echo -e "${BLUE}Screen 세션에서 실행합니다...${NC}"
        
        # 기존 세션 종료
        screen -S "$SESSION_NAME" -X quit 2>/dev/null
        
        # 새 세션 시작
        screen -dmS "$SESSION_NAME" bash -c "cd '$PROJECT_PATH' && python3 '$SCRIPT_DIR/autoci' resume 2>&1 | tee '$LOG_FILE'"
        
        echo -e "${GREEN}✓ Screen 세션 시작됨: $SESSION_NAME${NC}"
        echo "세션 접속: screen -r $SESSION_NAME"
        echo "세션 분리: Ctrl+A, D"
        echo "로그 확인: tail -f $LOG_FILE"
        ;;
        
    4)
        if ! command -v tmux &> /dev/null; then
            echo -e "${YELLOW}Tmux가 설치되어 있지 않습니다. 설치 중...${NC}"
            sudo apt-get update && sudo apt-get install -y tmux
        fi
        
        SESSION_NAME="autoci_resume_24h"
        echo -e "${BLUE}Tmux 세션에서 실행합니다...${NC}"
        
        # 기존 세션 종료
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null
        
        # 새 세션 시작
        tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_PATH' && python3 '$SCRIPT_DIR/autoci' resume 2>&1 | tee '$LOG_FILE'"
        
        echo -e "${GREEN}✓ Tmux 세션 시작됨: $SESSION_NAME${NC}"
        echo "세션 접속: tmux attach -t $SESSION_NAME"
        echo "세션 분리: Ctrl+B, D"
        echo "로그 확인: tail -f $LOG_FILE"
        ;;
        
    5)
        echo -e "${BLUE}데몬 모드로 실행합니다 (자동 재시작 지원)...${NC}"
        
        # 데몬 로그 파일
        DAEMON_LOG="$LOG_DIR/daemon_${TIMESTAMP}.log"
        
        # Screen에서 데몬 실행
        if command -v screen &> /dev/null; then
            SESSION_NAME="autoci_daemon"
            screen -dmS "$SESSION_NAME" bash -c "python3 '$SCRIPT_DIR/autoci_daemon.py' '$PROJECT_PATH' 2>&1 | tee '$DAEMON_LOG'"
            echo -e "${GREEN}✓ 데몬이 Screen 세션에서 시작됨${NC}"
            echo "세션 접속: screen -r $SESSION_NAME"
        else
            nohup python3 "$SCRIPT_DIR/autoci_daemon.py" "$PROJECT_PATH" > "$DAEMON_LOG" 2>&1 &
            PID=$!
            echo -e "${GREEN}✓ 데몬 PID: $PID${NC}"
        fi
        
        echo -e "${GREEN}✓ 데몬 로그: $DAEMON_LOG${NC}"
        echo "상태 확인: cat $SCRIPT_DIR/logs/daemon_status.json | jq ."
        ;;
        
    *)
        echo -e "${RED}잘못된 선택입니다${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ AutoCI Resume 24시간 실행이 시작되었습니다!${NC}"
echo ""
echo "유용한 명령어:"
echo "- 프로세스 확인: ps aux | grep autoci"
echo "- 메모리 사용량: free -h"
echo "- GPU 사용량: nvidia-smi (NVIDIA GPU가 있는 경우)"
echo "- 로그 실시간 확인: tail -f $LOG_FILE"
echo ""

# 모니터링 옵션
read -p "실시간 모니터링을 시작하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 모니터링 화면 분할 (가능한 경우)
    if command -v tmux &> /dev/null; then
        tmux new-session \; \
            send-keys "tail -f $LOG_FILE" C-m \; \
            split-window -h \; \
            send-keys "watch -n 5 'ps aux | grep autoci | grep -v grep'" C-m \; \
            split-window -v \; \
            send-keys "watch -n 10 free -h" C-m \; \
            select-pane -t 0
    else
        echo "간단한 모니터링을 시작합니다..."
        tail -f "$LOG_FILE"
    fi
fi