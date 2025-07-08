# AutoCI PowerShell Activation Script

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $scriptDir "autoci_env"

if (Test-Path (Join-Path $venvPath "Scripts\Activate.ps1")) {
    & "$venvPath\Scripts\Activate.ps1"
    Write-Host "Virtual environment activated!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Now you can use:" -ForegroundColor Yellow
    Write-Host "  python autoci learn" -ForegroundColor Cyan
    Write-Host "  python autoci create" -ForegroundColor Cyan
    Write-Host "  python autoci fix" -ForegroundColor Cyan
} else {
    Write-Host "Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: .\setup-windows.bat" -ForegroundColor Yellow
}