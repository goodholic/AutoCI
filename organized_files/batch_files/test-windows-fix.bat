@echo off
echo === Windows AutoCI Test ===
echo.

cd /d "%~dp0"

echo [1] Testing Python...
py --version
if errorlevel 1 (
    echo Python not found with 'py' command
    python --version
)

echo.
echo [2] Testing autoci.bat...
call autoci.bat --help

echo.
echo [3] Testing learn command...
call autoci.bat learn
if errorlevel 1 echo Learn command failed

echo.
echo [4] Testing create command directly...
set PYTHONPATH=%~dp0;%~dp0\core_system;%~dp0\modules
set AUTOCI_SKIP_VENV_CHECK=1
py autoci.py create
if errorlevel 1 (
    echo Create with autoci.py failed
    echo Trying autoci file...
    py autoci create
)

echo.
echo [5] Testing fix command directly...
py autoci.py fix
if errorlevel 1 (
    echo Fix with autoci.py failed
    echo Trying autoci file...
    py autoci fix
)

echo.
pause