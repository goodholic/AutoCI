@echo off
:: AutoCI 전역 명령어 설치 스크립트 (Windows)

echo.
echo ===================================================
echo         AutoCI 전역 명령어 설치 (Windows)
echo ===================================================
echo.

:: 현재 디렉토리 저장
set CURRENT_DIR=%~dp0
set CURRENT_DIR=%CURRENT_DIR:~0,-1%

:: Python 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [오류] Python이 설치되어 있지 않거나 PATH에 없습니다.
    echo Python 3.8 이상을 설치해주세요.
    pause
    exit /b 1
)

:: autoci.bat 파일 생성
echo @echo off > "%CURRENT_DIR%\autoci.bat"
echo python "%CURRENT_DIR%\autoci_launcher.py" %%* >> "%CURRENT_DIR%\autoci.bat"

:: autoci.cmd 파일도 생성 (호환성)
copy "%CURRENT_DIR%\autoci.bat" "%CURRENT_DIR%\autoci.cmd" >nul

echo.
echo [1/3] 실행 파일 생성 완료
echo.

:: PATH 확인
echo %PATH% | find /i "%CURRENT_DIR%" >nul
if %errorlevel% equ 0 (
    echo [2/3] 이미 PATH에 등록되어 있습니다.
) else (
    echo [2/3] PATH에 추가가 필요합니다.
    echo.
    echo 다음 중 하나를 선택하세요:
    echo.
    echo A. 자동으로 사용자 PATH에 추가 (권장)
    echo B. 수동으로 PATH에 추가
    echo C. 건너뛰기
    echo.
    
    choice /c ABC /n /m "선택 (A/B/C): "
    
    if errorlevel 3 (
        echo PATH 추가를 건너뜁니다.
    ) else if errorlevel 2 (
        echo.
        echo 수동 설정 방법:
        echo 1. Windows + X 키를 누르고 "시스템" 선택
        echo 2. "고급 시스템 설정" 클릭
        echo 3. "환경 변수" 클릭
        echo 4. 사용자 변수에서 "Path" 선택 후 "편집"
        echo 5. "새로 만들기" 클릭 후 다음 경로 추가:
        echo    %CURRENT_DIR%
        echo 6. 모든 창에서 "확인" 클릭
        echo.
    ) else (
        :: PowerShell을 사용하여 사용자 PATH에 추가
        powershell -Command "[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ';%CURRENT_DIR%', 'User')"
        echo PATH에 추가되었습니다. 새 명령 프롬프트에서 적용됩니다.
    )
)

echo.
echo [3/3] 설치 완료!
echo.
echo ===================================================
echo              사용 가능한 명령어
echo ===================================================
echo.
echo   autoci              - 대화형 모드로 AutoCI 시작
echo   autoci create       - 새 게임 자동 생성 (24시간)
echo   autoci learn        - AI 모델 기반 연속 학습
echo   autoci learn low    - 메모리 최적화 연속 학습
echo   autoci fix          - 학습 기반 엔진 능력 업데이트
echo   autoci monitor      - 실시간 개발 모니터링
echo   autoci demo         - 5분 빠른 데모
echo.
echo 예시:
echo   autoci create --name MyGame --type platformer
echo   autoci learn
echo   autoci monitor --port 5001
echo.
echo ===================================================
echo.
echo 참고: 새 명령 프롬프트를 열어서 명령어를 사용하세요.
echo.
pause