param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [string]$Config = 'configs/confirmatory_restart_synthetic_v2.json',
    [string]$Output = 'runs/confirmatory_restart_boot_v2',
    [string]$Protocol = '',
    [string]$Python = '.venv/Scripts/python.exe'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $Workspace
$env:PYTHONPATH = Join-Path $Workspace 'src'
$pythonPath = Join-Path $Workspace $Python
$driverArguments = @('--config', $Config, '--output', $Output)
if ($Protocol) { $driverArguments += @('--protocol', $Protocol) }
$outputPath = Join-Path $Workspace $Output
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$bootTime = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')
$env:PAPER1_BOOT_ID = $bootTime
@{ event = 'startup_wrapper'; boot_utc = $bootTime; started_utc = [DateTime]::UtcNow.ToString('o'); synthetic = (-not $Protocol) } |
    ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $outputPath 'startup.jsonl')
while ($true) {
    $ErrorActionPreference = 'Continue'
    & $pythonPath (Join-Path $Workspace 'scripts/run_confirmatory.py') @driverArguments `
        >> (Join-Path $outputPath 'console.log') 2>&1
    $driverExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($driverExit -eq 0) {
        if ($Protocol) { exit 0 }
        $ErrorActionPreference = 'Continue'
        & powershell.exe -NoProfile -NonInteractive -File (Join-Path $Workspace 'scripts/check_confirmatory_boot.ps1') -Output $Output -ObserveAutomaticLogon `
            >> (Join-Path $outputPath 'boot-check-console.log') 2>&1
        $bootCheckExit = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        @{ event = 'boot_check'; exit_code = $bootCheckExit; utc = [DateTime]::UtcNow.ToString('o') } |
            ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $outputPath 'startup.jsonl')
        exit 0
    }
    @{ event = 'driver_retry'; exit_code = $driverExit; utc = [DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $outputPath 'startup.jsonl')
    if (Test-Path -LiteralPath (Join-Path $outputPath 'STOP')) { exit $driverExit }
    Start-Sleep -Seconds 20
}
