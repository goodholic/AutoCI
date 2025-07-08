@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
py "%~dp0autoci-windows-ui-fixed.py"