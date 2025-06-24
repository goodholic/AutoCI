@echo off
echo AutoCI 시작 스크립트
echo ====================

echo 1. Python 서버 시작 중...
cd MyAIWebApp\Models
start /B cmd /c "llm_venv\Scripts\activate && python simple_server.py"

timeout /t 3 /nobreak > nul

echo 2. 백엔드 API 시작 중...
cd ..\Backend
start /B cmd /c "dotnet run"

timeout /t 5 /nobreak > nul

echo 3. 프론트엔드 시작 중...
cd ..\Frontend
start /B cmd /c "dotnet run"

echo.
echo 모든 서비스가 시작되었습니다!
echo ================================
echo Python API: http://localhost:8000
echo Backend API: http://localhost:5049
echo Frontend: http://localhost:5100
echo.
echo 종료하려면 이 창을 닫으세요.

pause