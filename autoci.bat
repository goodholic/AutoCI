@echo off
REM AutoCI 자동 가상환경 활성화 및 실행 스크립트 (Windows)

REM 스크립트 디렉토리로 이동
cd /d "%~dp0"

REM 가상환경 존재 확인
if not exist "autoci_env" (
    echo 🔧 가상환경이 없습니다. 생성 중...
    python -m venv autoci_env
    
    REM pip 업그레이드
    autoci_env\Scripts\pip.exe install --upgrade pip
    
    REM 기본 패키지 설치
    echo 📦 필수 패키지 설치 중...
    autoci_env\Scripts\pip.exe install -r requirements_minimal.txt 2>NUL
)

REM 가상환경 활성화하여 Python 스크립트 실행
autoci_env\Scripts\python.exe autoci %*