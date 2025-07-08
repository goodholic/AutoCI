@echo off
echo === AutoCI Debug Info ===
echo.

echo Current Directory: %CD%
echo Script Directory: %~dp0
echo.

echo Checking Python:
python --version
echo.

echo Checking if autoci file exists:
if exist "%~dp0autoci" (
    echo [OK] autoci file found
) else (
    echo [ERROR] autoci file NOT found!
)
echo.

echo Trying to run autoci directly:
python "%~dp0autoci" --help
echo.

echo Trying with learn command:
python "%~dp0autoci" learn
echo.

pause