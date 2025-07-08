@echo off
echo ========================================
echo AutoCI Windows Setup
echo ========================================
echo.

cd /d "%~dp0"

REM Check if py launcher is available
where py >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo Creating virtual environment...
if not exist "autoci_env" (
    py -m venv autoci_env
    echo Virtual environment created!
) else (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...
call autoci_env\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing requirements...
if exist "requirements.txt" (
    python -m pip install -r requirements.txt
) else (
    echo Installing basic requirements...
    python -m pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    python -m pip install asyncio pathlib dataclasses psutil
    python -m pip install flask flask-cors
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To use AutoCI:
echo 1. Activate virtual environment: autoci_env\Scripts\activate.bat
echo 2. Run commands:
echo    python autoci learn
echo    python autoci create
echo    python autoci fix
echo.
pause