@echo off
REM AutoCI for Windows - Microsoft Store Python version

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Change to script directory
cd /d "%SCRIPT_DIR%"

REM Use Microsoft Store Python
set "PYTHON_EXE=python"

REM Run the main autoci script with all arguments
echo Running AutoCI with Microsoft Store Python...
%PYTHON_EXE% "%SCRIPT_DIR%\autoci" %*

endlocal