@echo off
cd /d "%~dp0"
set PYTHONPATH=%cd%;%cd%\core_system;%cd%\modules
set AUTOCI_SKIP_VENV_CHECK=1
echo Starting AutoCI Fix...
py "%~dp0autoci.py" fix %*
if errorlevel 1 (
    echo.
    echo Trying alternative method...
    py "%~dp0autoci" fix %*
    if errorlevel 1 (
        echo.
        echo Using fallback stub version...
        py "%~dp0fix-stub.py" %*
    )
)