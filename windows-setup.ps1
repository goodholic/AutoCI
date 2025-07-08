# AutoCI Windows Setup - Simple Solution
# This creates .cmd files in C:\Windows\System32 for global access

param(
    [switch]$Force
)

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

if (-not $isAdmin) {
    Write-Host "This script needs to run as Administrator to install global commands." -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    
    if (-not $Force) {
        Write-Host "`nAlternatively, we'll create local shortcuts instead..." -ForegroundColor Green
        Start-Sleep -Seconds 2
    }
}

$autoциDir = $PSScriptRoot

# Create simple wrapper scripts
$learnScript = @"
@echo off
cd /d "$autoциDir"
py autoci learn %*
"@

$createScript = @"
@echo off
cd /d "$autoциDir"
py autoci create %*
"@

$fixScript = @"
@echo off
cd /d "$autoциDir"
py autoci fix %*
"@

if ($isAdmin) {
    # Install globally in System32
    Write-Host "Installing global commands..." -ForegroundColor Green
    
    $learnScript | Out-File -FilePath "C:\Windows\System32\learn.cmd" -Encoding ASCII
    $createScript | Out-File -FilePath "C:\Windows\System32\create.cmd" -Encoding ASCII
    $fixScript | Out-File -FilePath "C:\Windows\System32\fix.cmd" -Encoding ASCII
    
    Write-Host "Global commands installed!" -ForegroundColor Green
    Write-Host "You can now use 'learn', 'create', and 'fix' from anywhere!" -ForegroundColor Yellow
} else {
    # Create local batch files
    Write-Host "Creating local command files..." -ForegroundColor Green
    
    $learnScript | Out-File -FilePath "$autoциDir\learn.cmd" -Encoding ASCII
    $createScript | Out-File -FilePath "$autoциDir\create.cmd" -Encoding ASCII
    $fixScript | Out-File -FilePath "$autoциDir\fix.cmd" -Encoding ASCII
    
    # Also create PowerShell-friendly versions
    @"
@echo off
py "$autoциDir\autoci" learn %*
"@ | Out-File -FilePath "$autoциDir\_learn.cmd" -Encoding ASCII

    @"
@echo off
py "$autoциDir\autoci" create %*
"@ | Out-File -FilePath "$autoциDir\_create.cmd" -Encoding ASCII

    @"
@echo off
py "$autoциDir\autoci" fix %*
"@ | Out-File -FilePath "$autoциDir\_fix.cmd" -Encoding ASCII
    
    Write-Host "Local commands created!" -ForegroundColor Green
    Write-Host "`nUse these commands:" -ForegroundColor Yellow
    Write-Host "  .\_learn" -ForegroundColor Cyan
    Write-Host "  .\_create" -ForegroundColor Cyan
    Write-Host "  .\_fix" -ForegroundColor Cyan
}

Write-Host "`nSetup complete!" -ForegroundColor Green