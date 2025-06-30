@echo off
REM AI Godot 빌드 실행 - Windows에서 직접 실행
echo ===================================================
echo         AI Godot 빌드 시작
echo ===================================================
echo.

cd /d "%~dp0"

REM Python 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [오류] Python이 설치되지 않았습니다.
    echo        https://python.org 에서 다운로드하세요.
    pause
    exit /b 1
)

echo [확인] Python 설치 확인됨
echo.

REM 빌드 스크립트 실행
echo Windows 빌드를 시작합니다...
python build_ai_godot_windows.py

if %errorlevel% equ 0 (
    echo.
    echo ===================================================
    echo        빌드 성공!
    echo ===================================================
    echo.
    echo AI Godot이 성공적으로 빌드되었습니다.
    echo 이제 autoci 명령어로 사용할 수 있습니다.
) else (
    echo.
    echo ===================================================
    echo        빌드 실패
    echo ===================================================
    echo.
    echo godot_build.log 파일을 확인하세요.
)

pause