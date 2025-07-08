@echo off
echo === AutoCI 필수 패키지 설치 ===
echo.
echo sklearn 및 기타 누락된 패키지를 설치합니다...
echo.

REM sklearn 설치
echo [1/4] Installing scikit-learn...
py -m pip install scikit-learn

REM keyboard 설치 (GUI 자동화용)
echo.
echo [2/4] Installing keyboard...
py -m pip install keyboard

REM 기타 필수 패키지
echo.
echo [3/4] Installing other required packages...
py -m pip install numpy pillow aiohttp aiofiles psutil pyyaml python-dotenv

REM opencv 및 screeninfo
echo.
echo [4/4] Installing GUI automation packages...
py -m pip install opencv-python screeninfo pynput

echo.
echo === 설치 완료! ===
echo.
echo 이제 create와 fix 명령을 사용할 수 있습니다:
echo   - .\create-now.cmd
echo   - .\fix-now.cmd
echo.
pause