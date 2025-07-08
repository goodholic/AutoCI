# AutoCI Global Commands for PowerShell
Write-Host "Installing AutoCI global commands..." -ForegroundColor Green

$autoциDir = $PSScriptRoot

# Create profile if doesn't exist
if (!(Test-Path $PROFILE)) {
    New-Item -Type File -Path $PROFILE -Force | Out-Null
}

# Define functions
$functions = @"

# AutoCI Global Commands
`$global:AUTOCI_DIR = "$autoциDir"

function autoci {
    param([Parameter(ValueFromRemainingArguments=`$true)]`$args)
    `$venvPython = Join-Path `$global:AUTOCI_DIR "autoci_env\Scripts\python.exe"
    if (Test-Path `$venvPython) {
        & `$venvPython (Join-Path `$global:AUTOCI_DIR "autoci") `$args
    } else {
        & py (Join-Path `$global:AUTOCI_DIR "autoci") `$args
    }
}

function autoci-learn {
    autoci learn `$args
}

function autoci-create {
    autoci create `$args
}

function autoci-fix {
    autoci fix
}

# Short aliases
Set-Alias -Name "learn" -Value autoci-learn -Option AllScope
Set-Alias -Name "create" -Value autoci-create -Option AllScope
Set-Alias -Name "fix" -Value autoci-fix -Option AllScope

Write-Host "AutoCI commands ready: learn, create, fix" -ForegroundColor Green
"@

# Remove old AutoCI section
$profile = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue
$profile = $profile -replace '# AutoCI Global Commands[\s\S]*?# End AutoCI', ''

# Add new functions
$profile += "`n$functions`n# End AutoCI`n"
$profile | Set-Content $PROFILE

Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "Run: . `$PROFILE" -ForegroundColor Yellow