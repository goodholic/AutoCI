@echo off
echo Finding Python installation...
echo ================================
echo.

REM Check common Python locations
set PYTHON_FOUND=0

REM Check if python is in PATH
where python >nul 2>&1
if %errorlevel% equ 0 (
    echo Found Python in PATH:
    where python
    for /f "tokens=*" %%i in ('where python') do set PYTHON_PATH=%%i
    set PYTHON_FOUND=1
)

REM Check py launcher
where py >nul 2>&1
if %errorlevel% equ 0 (
    echo Found Python Launcher:
    where py
    set PY_FOUND=1
)

REM Check common installation directories
if exist "C:\Python312\python.exe" (
    echo Found Python at C:\Python312
    set PYTHON_PATH=C:\Python312\python.exe
    set PYTHON_FOUND=1
)

if exist "C:\Python311\python.exe" (
    echo Found Python at C:\Python311
    set PYTHON_PATH=C:\Python311\python.exe
    set PYTHON_FOUND=1
)

if exist "C:\Python310\python.exe" (
    echo Found Python at C:\Python310
    set PYTHON_PATH=C:\Python310\python.exe
    set PYTHON_FOUND=1
)

if exist "C:\Python39\python.exe" (
    echo Found Python at C:\Python39
    set PYTHON_PATH=C:\Python39\python.exe
    set PYTHON_FOUND=1
)

if exist "C:\Python38\python.exe" (
    echo Found Python at C:\Python38
    set PYTHON_PATH=C:\Python38\python.exe
    set PYTHON_FOUND=1
)

REM Check Program Files
if exist "%ProgramFiles%\Python312\python.exe" (
    echo Found Python in Program Files
    set PYTHON_PATH=%ProgramFiles%\Python312\python.exe
    set PYTHON_FOUND=1
)

if exist "%ProgramFiles%\Python311\python.exe" (
    echo Found Python in Program Files
    set PYTHON_PATH=%ProgramFiles%\Python311\python.exe
    set PYTHON_FOUND=1
)

if exist "%ProgramFiles%\Python310\python.exe" (
    echo Found Python in Program Files
    set PYTHON_PATH=%ProgramFiles%\Python310\python.exe
    set PYTHON_FOUND=1
)

REM Check user local installation
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    echo Found Python in LocalAppData Python312
    set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    set PYTHON_FOUND=1
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    echo Found Python in LocalAppData Python311
    set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
    set PYTHON_FOUND=1
)

if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    echo Found Python in LocalAppData Python310
    set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python310\python.exe
    set PYTHON_FOUND=1
)

if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" (
    echo Found Python in LocalAppData Python39
    set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python39\python.exe
    set PYTHON_FOUND=1
)

if exist "%LOCALAPPDATA%\Programs\Python\Python38\python.exe" (
    echo Found Python in LocalAppData Python38
    set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python38\python.exe
    set PYTHON_FOUND=1
)

echo.
if %PYTHON_FOUND% equ 1 (
    echo Python found at: %PYTHON_PATH%
    echo.
    echo Creating fixed autoci launcher...
    
    REM Create new autoci launcher with hardcoded Python path
    echo @echo off > autoci-fixed.bat
    echo REM AutoCI with fixed Python path >> autoci-fixed.bat
    echo set "SCRIPT_DIR=%%~dp0" >> autoci-fixed.bat
    echo if "%%SCRIPT_DIR:~-1%%"=="\\" set "SCRIPT_DIR=%%SCRIPT_DIR:~0,-1%%" >> autoci-fixed.bat
    echo cd /d "%%SCRIPT_DIR%%" >> autoci-fixed.bat
    echo "%PYTHON_PATH%" "%%SCRIPT_DIR%%\autoci" %%* >> autoci-fixed.bat
    
    echo.
    echo Fixed launcher created: autoci-fixed.bat
    echo.
    echo You can now use:
    echo   .\autoci-fixed.bat learn
    echo   .\autoci-fixed.bat create platformer
    echo   .\autoci-fixed.bat fix
) else (
    echo Python not found in common locations!
    echo Please tell me where Python is installed.
)

echo.
pause