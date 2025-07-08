@echo off
echo === AutoCI 최소 필수 패키지 설치 ===
echo.

echo Installing missing packages...
py -m pip install GPUtil mouse

echo.
echo Done! Now try running create-now.cmd again.
pause