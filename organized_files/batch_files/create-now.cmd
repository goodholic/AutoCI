@echo off
cd /d "%~dp0"
set PYTHONPATH=%cd%;%cd%\core_system;%cd%\modules
set AUTOCI_SKIP_VENV_CHECK=1
echo Starting AutoCI Create...
py "%~dp0autoci.py" create %*
if errorlevel 1 (
    echo.
    echo Trying alternative method...
    py "%~dp0autoci" create %*
    if errorlevel 1 (
        echo.
        echo Using fallback stub version...
        py "%~dp0create-stub.py" %*
    )
)