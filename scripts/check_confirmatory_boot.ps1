param([string]$Output = 'runs/confirmatory_restart_boot_v2', [switch]$ObserveAutomaticLogon)
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspacePath
$outputPath = Join-Path $workspacePath $Output
$bootTime = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
$throughTime = Get-Date
$logonTimes = @()
try {
    $events = @(Get-WinEvent -FilterHashtable @{ LogName = 'System'; ProviderName = 'Microsoft-Windows-Winlogon'; Id = 7001; StartTime = $bootTime; EndTime = $throughTime } -ErrorAction Stop)
    $logonTimes = @($events | ForEach-Object { ([DateTimeOffset]$_.TimeCreated).ToUnixTimeMilliseconds() / 1000.0 })
} catch {
    if ($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*') { throw }
}
$provider = Get-WinEvent -ListProvider Microsoft-Windows-Winlogon -ErrorAction Stop
if (-not ($provider.Events | Where-Object Id -eq 7001)) { throw 'Winlogon event schema unavailable' }
$audit = @{ query_succeeded = $true; source = 'System/Microsoft-Windows-Winlogon/7001';
    boot_unix = ([DateTimeOffset]$bootTime).ToUnixTimeMilliseconds() / 1000.0;
    queried_through_unix = ([DateTimeOffset]$throughTime).ToUnixTimeMilliseconds() / 1000.0;
    interactive_logon_unix = $logonTimes }
$auditPath = Join-Path $outputPath 'logon-audit.json'
$audit | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $auditPath -Encoding UTF8
$env:PYTHONPATH = Join-Path $workspacePath 'src'
$checkerArguments = @((Join-Path $workspacePath 'scripts/check_confirmatory_boot.py'), '--output', $outputPath)
if ($ObserveAutomaticLogon) { $checkerArguments += '--observe-automatic-logon' }
& (Join-Path $workspacePath '.venv/Scripts/python.exe') @checkerArguments
exit $LASTEXITCODE
