@echo off
echo Checking Python installations...
echo ================================
echo.

echo Checking system Python:
python --version 2>nul
if %errorlevel% neq 0 (
    echo [X] Python not found in PATH
) else (
    echo [OK] Found system Python
    where python
)

echo.
echo Checking python3:
python3 --version 2>nul
if %errorlevel% neq 0 (
    echo [X] python3 not found
) else (
    echo [OK] Found python3
    where python3
)

echo.
echo Checking py launcher:
py --version 2>nul
if %errorlevel% neq 0 (
    echo [X] py launcher not found
) else (
    echo [OK] Found py launcher
    where py
)

echo.
echo Checking virtual environment:
if exist "%~dp0autoci_env\Scripts\python.exe" (
    echo [OK] Virtual environment found
    "%~dp0autoci_env\Scripts\python.exe" --version
) else (
    echo [X] Virtual environment not found
)

echo.
echo ================================
echo.
echo If Python is not found, please:
echo 1. Download Python from https://www.python.org/downloads/
echo 2. During installation, CHECK "Add Python to PATH"
echo 3. Restart this terminal after installation
echo.
pause