@echo off
REM AutoCI Windows Installation Script
REM Adds AutoCI to Windows PATH for global access

setlocal enabledelayedexpansion

echo ===================================
echo AutoCI Windows Installation
echo ===================================
echo.

REM Get the directory where this script is located
set "INSTALL_DIR=%~dp0"
if "%INSTALL_DIR:~-1%"=="\\" set "INSTALL_DIR=%INSTALL_DIR:~0,-1%"

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This script requires administrator privileges to modify PATH.
    echo Please run as administrator or manually add to PATH:
    echo %INSTALL_DIR%
    echo.
    pause
    exit /b 1
)

REM Add to system PATH
echo Adding AutoCI to system PATH...
setx /M PATH "%PATH%;%INSTALL_DIR%" >nul 2>&1

if %errorlevel% equ 0 (
    echo Successfully added to system PATH!
) else (
    echo Failed to add to system PATH. Trying user PATH...
    setx PATH "%PATH%;%INSTALL_DIR%" >nul 2>&1
    if !errorlevel! equ 0 (
        echo Successfully added to user PATH!
    ) else (
        echo Failed to modify PATH. Please add manually.
    )
)

echo.
echo Creating virtual environment...
if not exist "%INSTALL_DIR%\autoci_env" (
    python -m venv "%INSTALL_DIR%\autoci_env"
    echo Virtual environment created!
) else (
    echo Virtual environment already exists.
)

echo.
echo Installing requirements...
if exist "%INSTALL_DIR%\autoci_env\Scripts\python.exe" (
    "%INSTALL_DIR%\autoci_env\Scripts\python.exe" -m pip install --upgrade pip
    "%INSTALL_DIR%\autoci_env\Scripts\python.exe" -m pip install -r "%INSTALL_DIR%\requirements.txt"
) else (
    echo Warning: Could not find virtual environment Python.
    echo Please install requirements manually.
)

echo.
echo ===================================
echo Installation Complete!
echo ===================================
echo.
echo You can now use AutoCI commands from any directory:
echo   autoci learn
echo   autoci create [game_type]
echo   autoci fix
echo   autoci chat
echo   autoci sessions
echo   autoci resume
echo.
echo Note: You may need to restart your command prompt for PATH changes to take effect.
echo.
pause