@echo off
REM AI Godot 빌드 시작 스크립트 (Windows)
REM AutoCI를 위한 완전 제어 가능한 Godot 빌드

echo ===================================================
echo         AI 완전 제어 Godot 빌드 시스템
echo ===================================================
echo.

REM Python 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [오류] Python이 설치되지 않았습니다.
    echo        https://python.org 에서 다운로드하세요.
    pause
    exit /b 1
)

REM Git 확인
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [오류] Git이 설치되지 않았습니다.
    echo        https://git-scm.com 에서 다운로드하세요.
    pause
    exit /b 1
)

echo [확인] 필수 도구 확인 완료
echo.

REM SCons 설치 확인
pip show scons >nul 2>&1
if %errorlevel% neq 0 (
    echo [설치] SCons 설치 중...
    pip install scons
)

echo.
echo 빌드 옵션:
echo 1. 전체 빌드 (AI 기능 모두 포함) - 권장
echo 2. 빠른 빌드 (기본 AI 기능만)
echo 3. 디버그 빌드 (개발용)
echo.
set /p BUILD_OPTION="선택 (1-3) [기본: 1]: "
if "%BUILD_OPTION%"=="" set BUILD_OPTION=1

echo.
echo [시작] 빌드를 시작합니다...
echo.

REM Python 스크립트 실행
if "%BUILD_OPTION%"=="2" (
    python build_ai_godot.py --quick
) else if "%BUILD_OPTION%"=="3" (
    python build_ai_godot.py --debug
) else (
    python build_ai_godot.py
)

if %errorlevel% equ 0 (
    echo.
    echo ===================================================
    echo             빌드 성공!
    echo ===================================================
    echo.
    echo 다음 단계:
    echo 1. WSL에서 autoci 실행
    echo 2. AI Godot이 자동으로 감지됩니다
) else (
    echo.
    echo ===================================================
    echo             빌드 실패
    echo ===================================================
    echo.
    echo 문제 해결:
    echo 1. godot_build.log 파일 확인
    echo 2. Visual Studio Build Tools 설치 확인
)

echo.
pause