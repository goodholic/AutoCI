#!/bin/bash
# AI 진행 상태 모니터링 시작 스크립트

echo "🤖 AI 학습 모니터링 시작 옵션을 선택하세요:"
echo ""
echo "1) 일반 실행 (터미널 종료 시 같이 종료됨)"
echo "2) 백그라운드 실행 (터미널 종료 후에도 계속 실행)"
echo "3) screen 세션으로 실행 (분리 가능)"
echo "4) 현재 실행 중인 모니터링 확인"
echo ""
read -p "선택하세요 (1-4): " choice

case $choice in
    1)
        echo "🚀 일반 모드로 모니터링을 시작합니다..."
        python3 ai_progress_monitor.py
        ;;
    2)
        echo "🔄 백그라운드 모드로 모니터링을 시작합니다..."
        nohup python3 ai_progress_monitor.py > monitor_output.log 2>&1 &
        echo "✅ 백그라운드에서 실행 중입니다."
        echo "📄 로그 확인: tail -f monitor_output.log"
        echo "🛑 종료 방법: pkill -f ai_progress_monitor.py"
        ;;
    3)
        # screen이 설치되어 있는지 확인
        if command -v screen &> /dev/null; then
            echo "🖥️  screen 세션 'ai_monitor'로 시작합니다..."
            screen -dmS ai_monitor python3 ai_progress_monitor.py
            echo "✅ screen 세션이 시작되었습니다."
            echo "🔗 연결 방법: screen -r ai_monitor"
            echo "🔌 분리 방법: Ctrl+A, D"
            echo "🛑 종료 방법: screen -S ai_monitor -X quit"
        else
            echo "❌ screen이 설치되지 않았습니다."
            echo "설치 후 다시 시도하세요: sudo apt install screen"
        fi
        ;;
    4)
        echo "🔍 실행 중인 모니터링 프로세스 확인..."
        
        # 일반 프로세스 확인
        pids=$(pgrep -f "ai_progress_monitor.py")
        if [ ! -z "$pids" ]; then
            echo "📊 실행 중인 모니터링 프로세스:"
            ps aux | grep ai_progress_monitor.py | grep -v grep
        else
            echo "❌ 실행 중인 모니터링 프로세스가 없습니다."
        fi
        
        # screen 세션 확인
        if command -v screen &> /dev/null; then
            echo ""
            echo "🖥️  실행 중인 screen 세션:"
            screen -ls | grep ai_monitor || echo "❌ ai_monitor screen 세션이 없습니다."
        fi
        
        # 로그 파일 확인
        if [ -f "monitor_output.log" ]; then
            echo ""
            echo "📄 최근 모니터링 로그 (마지막 5줄):"
            tail -5 monitor_output.log
        fi
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        ;;
esac 