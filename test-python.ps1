# Test Python execution

Write-Host "=== Python Test ===" -ForegroundColor Yellow

# Test 1: Simple Python command
Write-Host "`nTest 1: Simple Python version check" -ForegroundColor Cyan
python --version

# Test 2: Run Python with inline code
Write-Host "`nTest 2: Inline Python code" -ForegroundColor Cyan
python -c "print('Hello from Python')"

# Test 3: Check Python executable location
Write-Host "`nTest 3: Python location" -ForegroundColor Cyan
Get-Command python | Format-List

# Test 4: Try to import sys and print path
Write-Host "`nTest 4: Python sys.path" -ForegroundColor Cyan
python -c "import sys; print('\n'.join(sys.path[:3]))"

# Test 5: Check if autoci file exists and try to run it
Write-Host "`nTest 5: AutoCI file check" -ForegroundColor Cyan
$autoциPath = Join-Path $PSScriptRoot "autoci"
if (Test-Path $autoциPath) {
    Write-Host "[OK] autoci file exists at: $autoциPath" -ForegroundColor Green
    
    # Try to read first few lines
    Write-Host "`nFirst 5 lines of autoci file:" -ForegroundColor Yellow
    Get-Content $autoциPath -Head 5
    
    # Try to execute
    Write-Host "`nTrying to execute autoci..." -ForegroundColor Yellow
    & python $autoциPath --help
} else {
    Write-Host "[ERROR] autoci file NOT found at: $autoциPath" -ForegroundColor Red
}

# Test 6: Try different execution methods
Write-Host "`nTest 6: Different execution methods" -ForegroundColor Cyan
Write-Host "Method 1: python autoci learn"
Start-Process -FilePath "python" -ArgumentList "autoci", "learn" -NoNewWindow -Wait

Write-Host "`nMethod 2: python.exe with full path"
Start-Process -FilePath "python.exe" -ArgumentList "$autoциPath", "learn" -NoNewWindow -Wait