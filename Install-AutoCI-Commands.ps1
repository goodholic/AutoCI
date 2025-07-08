# AutoCI PowerShell Commands Installer
# This script adds AutoCI commands to your PowerShell profile

Write-Host "Installing AutoCI commands for PowerShell..." -ForegroundColor Green

# Get the AutoCI directory
$autoциDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Create PowerShell profile if it doesn't exist
if (!(Test-Path $PROFILE)) {
    New-Item -Type File -Path $PROFILE -Force | Out-Null
    Write-Host "Created PowerShell profile at: $PROFILE" -ForegroundColor Yellow
}

# Define the AutoCI functions
$functionsToAdd = @"

# ========== AutoCI Commands ==========
# AutoCI directory
`$global:AUTOCI_DIR = "$autoциDir"

# Main autoci function
function autoci {
    param([Parameter(ValueFromRemainingArguments=`$true)]`$args)
    & python "`$global:AUTOCI_DIR\autoci.py" @args
}

# Specific command functions for convenience
function autoci-learn {
    param([Parameter(ValueFromRemainingArguments=`$true)]`$args)
    & python "`$global:AUTOCI_DIR\autoci.py" learn @args
}

function autoci-create {
    param([Parameter(ValueFromRemainingArguments=`$true)]`$args)
    & python "`$global:AUTOCI_DIR\autoci.py" create @args
}

function autoci-fix {
    & python "`$global:AUTOCI_DIR\autoci.py" fix
}

# Simplified 3 core commands
function learn {
    param([Parameter(ValueFromRemainingArguments=`$true)]`$args)
    & python "`$global:AUTOCI_DIR\autoci.py" learn @args
}

function create {
    param([Parameter(ValueFromRemainingArguments=`$true)]`$args)
    & python "`$global:AUTOCI_DIR\autoci.py" create @args
}

function fix {
    & python "`$global:AUTOCI_DIR\autoci.py" fix
}

function autoci-chat {
    & python "`$global:AUTOCI_DIR\autoci.py" chat
}

function autoci-resume {
    & python "`$global:AUTOCI_DIR\autoci.py" resume
}

function autoci-sessions {
    & python "`$global:AUTOCI_DIR\autoci.py" sessions
}

# Aliases for even shorter commands
Set-Alias -Name "aci" -Value autoci
Set-Alias -Name "aci-learn" -Value autoci-learn
Set-Alias -Name "aci-create" -Value autoci-create
Set-Alias -Name "aci-fix" -Value autoci-fix

# Help function
function autoci-help {
    Write-Host ""
    Write-Host "AutoCI Commands:" -ForegroundColor Yellow
    Write-Host "  autoci learn              - Start AI learning"
    Write-Host "  autoci create             - Create/resume game (interactive)"
    Write-Host "  autoci fix                - Fix engine based on learning"
    Write-Host ""
    Write-Host "Even shorter (3 core commands):" -ForegroundColor Green
    Write-Host "  learn                     - Start AI learning"
    Write-Host "  create                    - Create/resume game"
    Write-Host "  fix                       - Fix engine"
    Write-Host "  autoci chat               - Korean chat mode"
    Write-Host "  autoci resume             - Resume paused game"
    Write-Host "  autoci sessions           - Show all game sessions"
    Write-Host ""
    Write-Host "Shortcuts:" -ForegroundColor Green
    Write-Host "  aci learn                 - Same as 'autoci learn'"
    Write-Host "  aci create platformer     - Same as 'autoci create platformer'"
    Write-Host "  aci fix                   - Same as 'autoci fix'"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  learn"
    Write-Host "  create                    (will ask for game type)"
    Write-Host "  create platformer         (direct selection)"
    Write-Host "  fix"
    Write-Host ""
}

Write-Host "AutoCI commands loaded! Type 'autoci-help' for usage." -ForegroundColor Green
# ========== End AutoCI Commands ==========

"@

# Read current profile
$currentProfile = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue

# Check if AutoCI commands already exist
if ($currentProfile -match "AutoCI Commands") {
    Write-Host "Removing old AutoCI commands..." -ForegroundColor Yellow
    # Remove old AutoCI section
    $currentProfile = $currentProfile -replace '# ========== AutoCI Commands ==========[\s\S]*?# ========== End AutoCI Commands ==========\r?\n?', ''
}

# Add new commands to profile
$newProfile = $currentProfile + "`n" + $functionsToAdd
$newProfile | Set-Content $PROFILE -Encoding UTF8

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To use the new commands, either:" -ForegroundColor Yellow
Write-Host "1. Close and reopen PowerShell" -ForegroundColor Cyan
Write-Host "2. Or run: . `$PROFILE" -ForegroundColor Cyan
Write-Host ""
Write-Host "Then you can use:" -ForegroundColor Green
Write-Host "  autoci learn" -ForegroundColor White
Write-Host "  autoci create platformer" -ForegroundColor White
Write-Host "  autoci fix" -ForegroundColor White
Write-Host ""
Write-Host "Or use shortcuts:" -ForegroundColor Green
Write-Host "  aci learn" -ForegroundColor White
Write-Host "  aci create platformer" -ForegroundColor White
Write-Host "  aci fix" -ForegroundColor White