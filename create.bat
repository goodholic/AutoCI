@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0;%~dp0\core_system;%~dp0\modules
set AUTOCI_SKIP_VENV_CHECK=1

echo === AutoCI Create ===
echo.

REM Find Python
where py >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON=py"
) else (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON=python"
    ) else (
        echo Error: Python not found!
        pause
        exit /b 1
    )
)

REM Try main autoci first
echo Running create command...
if exist "autoci.py" (
    "%PYTHON%" autoci.py create %*
) else (
    "%PYTHON%" autoci create %*
)

REM If failed, try stub version
if errorlevel 1 (
    if exist "create-stub.py" (
        echo.
        echo Using fallback create script...
        "%PYTHON%" create-stub.py %*
    )
)

endlocal