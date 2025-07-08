@echo off
echo === AutoCI Create (Windows Safe Mode) ===
cd /d "%~dp0"
py create-windows.py %*
pause