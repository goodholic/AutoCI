# AutoCI PowerShell Setup Script
# This script creates PowerShell functions for easy AutoCI access

Write-Host "Setting up AutoCI for PowerShell..." -ForegroundColor Green

# Get the AutoCI directory
$autoциDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Create PowerShell profile if it doesn't exist
if (!(Test-Path $PROFILE)) {
    New-Item -Type File -Path $PROFILE -Force | Out-Null
}

# Add AutoCI functions to PowerShell profile
$profileContent = @"

# AutoCI Functions
function autoci {
    & "$autoциDir\autoci.ps1" `$args
}

function autoci-learn {
    & "$autoциDir\autoci.ps1" learn `$args
}

function autoci-create {
    & "$autoциDir\autoci.ps1" create `$args
}

function autoci-fix {
    & "$autoциDir\autoci.ps1" fix
}

function autoci-chat {
    & "$autoциDir\autoci.ps1" chat
}

function autoci-resume {
    & "$autoциDir\autoci.ps1" resume
}

function autoci-sessions {
    & "$autoциDir\autoci.ps1" sessions
}

Write-Host "AutoCI commands loaded! Type 'autoci-help' for usage." -ForegroundColor Cyan

function autoci-help {
    Write-Host "`nAutoCI Commands:" -ForegroundColor Yellow
    Write-Host "  autoci learn        - Start AI learning"
    Write-Host "  autoci-learn        - Start AI learning (alias)"
    Write-Host "  autoci create TYPE  - Create/resume game (platformer, racing, rpg, puzzle)"
    Write-Host "  autoci-create TYPE  - Create/resume game (alias)"
    Write-Host "  autoci fix          - Fix engine based on learning"
    Write-Host "  autoci-fix          - Fix engine (alias)"
    Write-Host "  autoci chat         - Korean chat mode"
    Write-Host "  autoci resume       - Resume paused game"
    Write-Host "  autoci sessions     - Show all game sessions"
    Write-Host "`nExamples:" -ForegroundColor Green
    Write-Host "  autoci-learn"
    Write-Host "  autoci-create platformer"
    Write-Host "  autoci-fix`n"
}
"@

# Check if AutoCI functions already exist in profile
$currentProfile = Get-Content $PROFILE -ErrorAction SilentlyContinue
if ($currentProfile -notcontains "# AutoCI Functions") {
    Add-Content -Path $PROFILE -Value $profileContent
    Write-Host "AutoCI functions added to PowerShell profile!" -ForegroundColor Green
} else {
    Write-Host "AutoCI functions already exist in profile. Updating..." -ForegroundColor Yellow
    # Remove old AutoCI section and add new one
    $newProfile = $currentProfile | Where-Object { $_ -notmatch "# AutoCI Functions" -and $_ -notmatch "function autoci" }
    $newProfile | Set-Content $PROFILE
    Add-Content -Path $PROFILE -Value $profileContent
}

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "Please run the following command to reload your profile:" -ForegroundColor Yellow
Write-Host "  . `$PROFILE" -ForegroundColor Cyan
Write-Host "`nOr restart PowerShell to use the new commands." -ForegroundColor Yellow
Write-Host "`nAfter reloading, you can use:" -ForegroundColor Green
Write-Host "  autoci-learn" -ForegroundColor Cyan
Write-Host "  autoci-create platformer" -ForegroundColor Cyan
Write-Host "  autoci-fix" -ForegroundColor Cyan