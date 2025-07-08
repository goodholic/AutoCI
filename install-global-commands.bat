@echo off
REM Install AutoCI as global commands on Windows
REM This creates wrapper scripts in a directory that's added to PATH

setlocal enabledelayedexpansion

echo ========================================
echo AutoCI Global Commands Installer
echo ========================================
echo.

REM Get current directory
set "AUTOCI_DIR=%~dp0"
if "%AUTOCI_DIR:~-1%"=="\\" set "AUTOCI_DIR=%AUTOCI_DIR:~0,-1%"

REM Create bin directory for global commands
set "BIN_DIR=%AUTOCI_DIR%\bin"
if not exist "%BIN_DIR%" (
    mkdir "%BIN_DIR%"
    echo Created bin directory: %BIN_DIR%
)

REM Create wrapper batch files that work globally
echo Creating global command wrappers...

REM autoci command
echo @echo off > "%BIN_DIR%\autoci.bat"
echo call "%AUTOCI_DIR%\autoci.bat" %%* >> "%BIN_DIR%\autoci.bat"

REM autoci subcommands for convenience
echo @echo off > "%BIN_DIR%\autoci-learn.bat"
echo call "%AUTOCI_DIR%\autoci.bat" learn %%* >> "%BIN_DIR%\autoci-learn.bat"

echo @echo off > "%BIN_DIR%\autoci-create.bat"
echo call "%AUTOCI_DIR%\autoci.bat" create %%* >> "%BIN_DIR%\autoci-create.bat"

echo @echo off > "%BIN_DIR%\autoci-fix.bat"
echo call "%AUTOCI_DIR%\autoci.bat" fix %%* >> "%BIN_DIR%\autoci-fix.bat"

echo @echo off > "%BIN_DIR%\autoci-chat.bat"
echo call "%AUTOCI_DIR%\autoci.bat" chat %%* >> "%BIN_DIR%\autoci-chat.bat"

echo @echo off > "%BIN_DIR%\autoci-resume.bat"
echo call "%AUTOCI_DIR%\autoci.bat" resume %%* >> "%BIN_DIR%\autoci-resume.bat"

echo @echo off > "%BIN_DIR%\autoci-sessions.bat"
echo call "%AUTOCI_DIR%\autoci.bat" sessions %%* >> "%BIN_DIR%\autoci-sessions.bat"

echo Command wrappers created!
echo.

REM Check if bin directory is already in PATH
echo %PATH% | find /i "%BIN_DIR%" >nul
if %errorlevel% equ 0 (
    echo %BIN_DIR% is already in PATH
) else (
    echo Adding %BIN_DIR% to PATH...
    
    REM Try to add to system PATH (requires admin)
    setx /M PATH "%PATH%;%BIN_DIR%" >nul 2>&1
    if !errorlevel! equ 0 (
        echo Successfully added to system PATH!
    ) else (
        echo Could not modify system PATH (admin required)
        echo Trying user PATH instead...
        
        REM Add to user PATH
        for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%b"
        setx PATH "!USER_PATH!;%BIN_DIR%" >nul 2>&1
        
        if !errorlevel! equ 0 (
            echo Successfully added to user PATH!
        ) else (
            echo Failed to modify PATH automatically.
            echo.
            echo Please add this directory to your PATH manually:
            echo %BIN_DIR%
        )
    )
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo IMPORTANT: Close and reopen your terminal for PATH changes to take effect!
echo.
echo After reopening, you can use these commands from anywhere:
echo   autoci learn
echo   autoci create platformer
echo   autoci fix
echo   autoci chat
echo   autoci resume
echo   autoci sessions
echo.
echo Or use the short versions:
echo   autoci-learn
echo   autoci-create platformer
echo   autoci-fix
echo.
pause