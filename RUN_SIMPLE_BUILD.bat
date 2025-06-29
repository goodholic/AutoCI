@echo off
REM 간단한 AI Godot 설정 실행
echo ===================================================
echo     간단한 AI Godot 설정
echo ===================================================
echo.
echo 이 스크립트는 다음 작업을 수행합니다:
echo 1. Godot 4.3 다운로드
echo 2. AI 설정 파일 생성
echo 3. AutoCI와 연동 설정
echo.
pause

cd /d "%~dp0"

REM Python 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [오류] Python이 설치되지 않았습니다.
    echo        https://python.org 에서 다운로드하세요.
    pause
    exit /b 1
)

echo.
echo [시작] 설정을 시작합니다...
echo.

REM 스크립트 실행
python simple_godot_build.py

echo.
pause