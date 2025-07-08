@echo off
echo === Python Diagnosis ===
echo.

echo 1. Checking where python command points to:
where python
echo.

echo 2. Checking python3 command:
where python3 2>nul
if %errorlevel% neq 0 (
    echo python3 not found
)
echo.

echo 3. Checking py launcher:
where py 2>nul
if %errorlevel% neq 0 (
    echo py launcher not found
)
echo.

echo 4. Testing direct python execution:
python -c "print('Hello from Python')"
echo.

echo 5. Checking if this is Microsoft Store Python:
python -c "import sys; print('Executable:', sys.executable)"
python -c "import sys; print('Version:', sys.version)"
echo.

echo 6. Testing script execution:
echo print('Test script executed') > test_script.py
python test_script.py
del test_script.py
echo.

echo 7. Testing with full path:
for /f "tokens=*" %%i in ('where python') do set PYTHON_PATH=%%i
echo Found Python at: %PYTHON_PATH%
"%PYTHON_PATH%" -c "print('Full path execution works')"
echo.

pause