@echo off
REM Fix Korean encoding for Windows
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo UTF-8 encoding set for this session.
echo You can now use AutoCI commands.