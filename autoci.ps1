# AutoCI - Windows PowerShell Script
# Cross-platform AI Game Development System

# Get the directory where this script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if virtual environment exists
$venvPath = Join-Path $scriptDir "autoci_env\Scripts\python.exe"
if (Test-Path $venvPath) {
    # Use virtual environment Python
    $pythonExe = $venvPath
    Write-Host "Using virtual environment Python" -ForegroundColor Green
} else {
    # Use system Python
    $pythonExe = "python"
    Write-Host "Using system Python" -ForegroundColor Yellow
}

# Check if Python is available
try {
    $pythonVersion = & $pythonExe --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or later from https://www.python.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Display command being executed
Write-Host "Running: autoci $args" -ForegroundColor Cyan

# Run the main autoci script with all arguments
$autociScript = Join-Path $scriptDir "autoci"
try {
    & $pythonExe $autociScript $args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nError occurred while running AutoCI" -ForegroundColor Red
        Read-Host "Press Enter to continue"
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Read-Host "Press Enter to continue"
}