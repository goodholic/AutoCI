# AutoCI Windows PowerShell Setup
# Simple and direct approach

Write-Host "Setting up AutoCI for Windows PowerShell..." -ForegroundColor Green

# Get current directory
$autociDir = $PSScriptRoot

# Create profile if doesn't exist
if (!(Test-Path $PROFILE)) {
    New-Item -Type File -Path $PROFILE -Force | Out-Null
}

# Functions to add
$newFunctions = @"

# === AutoCI Windows Commands ===
function learn {
    Push-Location "$autociDir"
    python autoci_windows.py learn `$args
    Pop-Location
}

function create {
    Push-Location "$autoциDir"
    python autoci_windows.py create `$args
    Pop-Location
}

function fix {
    Push-Location "$autoциDir"
    python autoci_windows.py fix
    Pop-Location
}

function autoci {
    Push-Location "$autoциDir"
    python autoci_windows.py `$args
    Pop-Location
}

Write-Host "AutoCI ready! Commands: learn, create, fix" -ForegroundColor Green
"@

# Read current profile
$currentProfile = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue

# Remove old AutoCI section if exists
$currentProfile = $currentProfile -replace '# === AutoCI Windows Commands ===[\s\S]*?# === End AutoCI ===', ''

# Add new functions
$currentProfile += "`n$newFunctions`n"

# Save profile
$currentProfile | Set-Content $PROFILE

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Run this to activate: . `$PROFILE" -ForegroundColor Yellow