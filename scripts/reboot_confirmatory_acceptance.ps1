param([int]$DelaySeconds = 45)
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$runPath = Join-Path $workspacePath 'runs/confirmatory_restart_boot_v2'
Start-Sleep -Seconds $DelaySeconds
if (Test-Path -LiteralPath (Join-Path $runPath 'complete.json')) {
    throw 'Synthetic job already complete; reboot would not test mid-job recovery.'
}
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'startup-synthetic/resume.pt'))) {
    throw 'No durable synthetic checkpoint; reboot acceptance is not ready.'
}
@{ event = 'authorized_reboot_requested'; synthetic = $true; utc = [DateTime]::UtcNow.ToString('o') } |
    ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $runPath 'startup.jsonl')
& shutdown.exe /r /t 0
if ($LASTEXITCODE -ne 0) { throw 'Windows rejected the restart request.' }
