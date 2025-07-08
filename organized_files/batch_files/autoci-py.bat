@echo off
REM AutoCI using py launcher (works around Microsoft Store Python issues)

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

REM Check for py launcher first
where py >nul 2>&1
if %errorlevel% equ 0 (
    echo Using Python via py launcher...
    py "%SCRIPT_DIR%\autoci" %*
) else (
    REM Try python3
    where python3 >nul 2>&1
    if !errorlevel! equ 0 (
        echo Using python3...
        python3 "%SCRIPT_DIR%\autoci" %*
    ) else (
        REM Last resort - try to find real Python
        set FOUND=0
        
        REM Check common Python locations
        if exist "C:\Python312\python.exe" (
            "C:\Python312\python.exe" "%SCRIPT_DIR%\autoci" %*
            set FOUND=1
        ) else if exist "C:\Python311\python.exe" (
            "C:\Python311\python.exe" "%SCRIPT_DIR%\autoci" %*
            set FOUND=1
        ) else if exist "C:\Python310\python.exe" (
            "C:\Python310\python.exe" "%SCRIPT_DIR%\autoci" %*
            set FOUND=1
        ) else if exist "C:\Python39\python.exe" (
            "C:\Python39\python.exe" "%SCRIPT_DIR%\autoci" %*
            set FOUND=1
        ) else if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
            "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%SCRIPT_DIR%\autoci" %*
            set FOUND=1
        ) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
            "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" "%SCRIPT_DIR%\autoci" %*
            set FOUND=1
        ) else if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
            "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" "%SCRIPT_DIR%\autoci" %*
            set FOUND=1
        )
        
        if !FOUND! equ 0 (
            echo Error: Could not find Python installation
            echo Please install Python from https://www.python.org/
            echo Or disable Windows App Execution Aliases for Python
            pause
            exit /b 1
        )
    )
)

endlocal