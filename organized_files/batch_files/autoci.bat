@echo off
REM AutoCI - Windows Batch Script
REM Cross-platform AI Game Development System

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Change to script directory
cd /d "%SCRIPT_DIR%"

REM Check if virtual environment exists
if exist "%SCRIPT_DIR%\autoci_env\Scripts\python.exe" (
    REM Use virtual environment Python
    set "PYTHON_EXE=%SCRIPT_DIR%\autoci_env\Scripts\python.exe"
    echo Using virtual environment Python
) else (
    REM Try different Python commands
    echo Virtual environment not found, checking system Python...
    
    REM Try python command
    python --version >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
        echo Using system Python
    ) else (
        REM Try python3 command
        python3 --version >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_EXE=python3"
            echo Using system Python3
        ) else (
            REM Try py launcher
            py --version >nul 2>&1
            if not errorlevel 1 (
                set "PYTHON_EXE=py"
                echo Using Python Launcher
            ) else (
                echo Error: Python not found!
                echo Please install Python from https://www.python.org/
                echo Make sure to check "Add Python to PATH" during installation
                pause
                exit /b 1
            )
        )
    )
)

REM Check if Python is available
"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or later from https://www.python.org/
    pause
    exit /b 1
)

REM Display command being executed
echo Running: autoci %*

REM Set Python path for imports
set PYTHONPATH=%SCRIPT_DIR%;%SCRIPT_DIR%\core_system;%SCRIPT_DIR%\modules
set AUTOCI_SKIP_VENV_CHECK=1

REM Run the main autoci script with all arguments
REM Try autoci.py first (Windows), then autoci (Unix-style)
if exist "%SCRIPT_DIR%\autoci.py" (
    "%PYTHON_EXE%" "%SCRIPT_DIR%\autoci.py" %*
) else (
    "%PYTHON_EXE%" "%SCRIPT_DIR%\autoci" %*
)

REM Check error level
if errorlevel 1 (
    echo.
    echo Error occurred while running AutoCI
    pause
)

endlocal