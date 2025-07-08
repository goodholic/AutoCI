@echo off
REM AutoCI with automatic virtual environment activation

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

REM Check if virtual environment exists
if exist "%SCRIPT_DIR%\autoci_env\Scripts\python.exe" (
    REM Use virtual environment Python directly
    "%SCRIPT_DIR%\autoci_env\Scripts\python.exe" "%SCRIPT_DIR%\autoci" %*
) else (
    echo Virtual environment not found!
    echo Please run setup-windows.bat first.
    pause
    exit /b 1
)

endlocal