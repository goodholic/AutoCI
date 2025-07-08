@echo off
echo =====================================================
echo AutoCI Create/Fix Import Error Fix
echo =====================================================
echo.
echo This will install the minimal required packages for
echo the 'create' and 'fix' commands to work properly.
echo.
echo Installing required packages...
echo.

py -m pip install numpy pillow torch transformers flask flask-socketio aiohttp aiofiles psutil pyyaml python-dotenv screeninfo pynput opencv-python colorama rich tqdm

echo.
echo =====================================================
echo Installation complete!
echo.
echo You can now run:
echo   py autoci create
echo   py autoci fix
echo.
echo Or use the Windows UI:
echo   py autoci-windows-ui.py
echo =====================================================
pause