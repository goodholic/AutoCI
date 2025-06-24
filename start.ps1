# AutoCI 시작 스크립트 (PowerShell)

Write-Host "AutoCI 시작 스크립트" -ForegroundColor Green
Write-Host "====================" -ForegroundColor Green

# Python 서버 시작
Write-Host "`n1. Python 서버 시작 중..." -ForegroundColor Yellow
$pythonPath = Join-Path $PSScriptRoot "MyAIWebApp\Models"
Start-Process -FilePath "cmd.exe" -ArgumentList "/k cd /d `"$pythonPath`" && llm_venv\Scripts\activate && python simple_server.py" -WorkingDirectory $pythonPath

Start-Sleep -Seconds 3

# 백엔드 시작
Write-Host "2. 백엔드 API 시작 중..." -ForegroundColor Yellow
$backendPath = Join-Path $PSScriptRoot "MyAIWebApp\Backend"
Start-Process -FilePath "cmd.exe" -ArgumentList "/k cd /d `"$backendPath`" && dotnet run" -WorkingDirectory $backendPath

Start-Sleep -Seconds 5

# 프론트엔드 시작
Write-Host "3. 프론트엔드 시작 중..." -ForegroundColor Yellow
$frontendPath = Join-Path $PSScriptRoot "MyAIWebApp\Frontend"
Start-Process -FilePath "cmd.exe" -ArgumentList "/k cd /d `"$frontendPath`" && dotnet run" -WorkingDirectory $frontendPath

Write-Host "`n모든 서비스가 시작되었습니다!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host "Python API: " -NoNewline; Write-Host "http://localhost:8000" -ForegroundColor Cyan
Write-Host "Backend API: " -NoNewline; Write-Host "http://localhost:5049" -ForegroundColor Cyan
Write-Host "Frontend: " -NoNewline; Write-Host "http://localhost:5100" -ForegroundColor Cyan
Write-Host "`n브라우저에서 http://localhost:5100 으로 접속하세요." -ForegroundColor Yellow
Write-Host "`n모든 창을 닫으려면 각 cmd 창을 개별적으로 닫으세요." -ForegroundColor White

# 5초 후 브라우저 자동 실행
Start-Sleep -Seconds 5
Start-Process "http://localhost:5100"