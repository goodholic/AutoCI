#!/bin/bash

echo "AutoCI 시작 스크립트"
echo "===================="

# Python 서버 시작
echo "1. Python 서버 시작 중..."
cd MyAIWebApp/Models
source llm_venv/bin/activate 2>/dev/null || llm_venv\\Scripts\\activate
python simple_server.py &
PYTHON_PID=$!
echo "Python 서버 PID: $PYTHON_PID"

# 잠시 대기
sleep 3

# 백엔드 시작
echo "2. 백엔드 API 시작 중..."
cd ../Backend
dotnet run &
BACKEND_PID=$!
echo "백엔드 PID: $BACKEND_PID"

# 잠시 대기
sleep 5

# 프론트엔드 시작
echo "3. 프론트엔드 시작 중..."
cd ../Frontend
dotnet run &
FRONTEND_PID=$!
echo "프론트엔드 PID: $FRONTEND_PID"

echo ""
echo "모든 서비스가 시작되었습니다!"
echo "================================"
echo "Python API: http://localhost:8000"
echo "Backend API: http://localhost:5049"
echo "Frontend: http://localhost:5100"
echo ""
echo "종료하려면 Ctrl+C를 누르세요."

# 종료 신호 처리
trap "echo '종료 중...'; kill $PYTHON_PID $BACKEND_PID $FRONTEND_PID; exit" INT TERM

# 프로세스 대기
wait