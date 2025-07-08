@echo off
echo === AutoCI 전체 필수 패키지 설치 ===
echo.

REM 기본 패키지
echo [1/10] Installing basic packages...
py -m pip install numpy pillow

REM 웹 관련
echo.
echo [2/10] Installing web packages...
py -m pip install aiohttp aiofiles requests

REM 시스템 모니터링
echo.
echo [3/10] Installing system monitoring...
py -m pip install psutil GPUtil

REM GUI 자동화
echo.
echo [4/10] Installing GUI automation...
py -m pip install keyboard mouse pyautogui screeninfo pynput

REM 머신러닝
echo.
echo [5/10] Installing ML packages...
py -m pip install scikit-learn torch torchvision transformers

REM 설정 관리
echo.
echo [6/10] Installing config packages...
py -m pip install pyyaml python-dotenv

REM 이미지 처리
echo.
echo [7/10] Installing image processing...
py -m pip install opencv-python

REM 기타 유틸리티
echo.
echo [8/10] Installing utilities...
py -m pip install colorama rich tqdm

REM 게임 개발 관련
echo.
echo [9/10] Installing game dev packages...
py -m pip install panda3d pygame

REM Flask 웹 서버
echo.
echo [10/10] Installing web server...
py -m pip install flask flask-socketio

echo.
echo === 모든 패키지 설치 완료! ===
echo.
pause