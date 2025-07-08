@echo off
REM AutoCI Learn - Windows Batch Script

setlocal

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Call main autoci with learn command
call "%SCRIPT_DIR%\autoci.bat" learn %*

endlocal