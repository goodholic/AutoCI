@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set AUTOCI_SKIP_VENV_CHECK=1
py "%~dp0autoci" fix %*