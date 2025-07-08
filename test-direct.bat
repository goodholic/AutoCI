@echo off
echo Testing direct Python execution...
echo.

REM Test 1: Check if autoci.py exists
if exist "autoci.py" (
    echo [OK] autoci.py found
) else (
    echo [ERROR] autoci.py not found
)

REM Test 2: Run autoci.py with full python command
echo.
echo Running: python autoci.py learn
python autoci.py learn

REM Test 3: Check error level
echo.
echo Exit code: %errorlevel%

REM Test 4: Try with python -u (unbuffered)
echo.
echo Running: python -u autoci.py learn
python -u autoci.py learn

REM Test 5: Check Python imports
echo.
echo Checking if Python can import required modules...
python -c "import sys; print('Python version:', sys.version)"
python -c "import os; print('OS module OK')"
python -c "import asyncio; print('Asyncio module OK')"

pause