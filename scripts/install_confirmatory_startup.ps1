param([switch]$PrepareOnly, [switch]$Passwordless, [switch]$RegisterOnly)
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$foundationPath = Join-Path $workspacePath 'reports/foundation'
$taskXmlPath = Join-Path $foundationPath $(if ($Passwordless) { 'confirmatory-startup.s4u.requested.xml' } else { 'confirmatory-startup.requested.xml' })
$exportPath = Join-Path $foundationPath 'confirmatory-startup.exported.xml'
$taskUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$escapeXml = { param($value) [Security.SecurityElement]::Escape($value) }
$escapedUser = & $escapeXml $taskUser
$escapedWorkspace = & $escapeXml $workspacePath
$escapedScript = & $escapeXml (Join-Path $workspacePath 'scripts/confirmatory_startup.ps1')
$logonType = if ($Passwordless) { 'S4U' } else { 'Password' }
$taskXml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Paper1 synthetic unattended restart acceptance. Empirical dispatch remains gated.</Description></RegistrationInfo>
  <Triggers><BootTrigger><Enabled>true</Enabled><Delay>PT20S</Delay></BootTrigger></Triggers>
  <Principals><Principal id="Author"><UserId>$escapedUser</UserId><LogonType>$logonType</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>false</AllowHardTerminate><StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings><StopOnIdleEnd>false</StopOnIdleEnd><RestartOnIdle>false</RestartOnIdle></IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand><Enabled>true</Enabled><Hidden>true</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle><WakeToRun>true</WakeToRun><ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority><RestartOnFailure><Interval>PT1M</Interval><Count>999</Count></RestartOnFailure>
  </Settings>
  <Actions Context="Author"><Exec>
    <Command>powershell.exe</Command>
    <Arguments>-NoProfile -NonInteractive -WindowStyle Hidden -File &quot;$escapedScript&quot;</Arguments>
    <WorkingDirectory>$escapedWorkspace</WorkingDirectory>
  </Exec></Actions>
</Task>
"@
[IO.File]::WriteAllText($taskXmlPath, $taskXml, [Text.Encoding]::Unicode)
$logonArguments = if ($Passwordless) { '/NP' } else { '/RP *' }
$userArguments = if ($Passwordless) { '' } else { ' /RU "' + $taskUser + '"' }
$commandRecord = 'schtasks.exe /Create /TN "Paper1-Confirmatory" /XML "' + $taskXmlPath + '"' + $userArguments + ' ' + $logonArguments
$commandFilename = if ($Passwordless) { 'confirmatory-startup.s4u.command.txt' } else { 'confirmatory-startup-command.txt' }
[IO.File]::WriteAllText((Join-Path $foundationPath $commandFilename), $commandRecord + [Environment]::NewLine)
if ($PrepareOnly) { Write-Output 'Prepared XML only; task not installed.'; exit 0 }
$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $Passwordless -and -not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this installer in an elevated local PowerShell. Enter the Windows account password only in the schtasks prompt; never in chat or a file.'
}
if ($Passwordless) {
    & schtasks.exe /Create /TN 'Paper1-Confirmatory' /XML $taskXmlPath /NP
} else {
    & schtasks.exe /Create /TN 'Paper1-Confirmatory' /XML $taskXmlPath /RU $taskUser /RP '*'
}
if ($LASTEXITCODE -ne 0) { throw 'Task registration failed; no reboot is authorized by this installer.' }
$exportedXml = & schtasks.exe /Query /TN 'Paper1-Confirmatory' /XML
if ($LASTEXITCODE -ne 0) { throw 'Task export failed.' }
[IO.File]::WriteAllText($exportPath, ($exportedXml -join [Environment]::NewLine), [Text.Encoding]::Unicode)
if ($RegisterOnly) { Write-Output 'Task installed and XML exported. Power settings were not reapplied.'; exit 0 }
& powercfg.exe /change standby-timeout-ac 0
if ($LASTEXITCODE -ne 0) { throw 'AC sleep setting failed' }
& powercfg.exe /change standby-timeout-dc 0
if ($LASTEXITCODE -ne 0) { throw 'DC sleep setting failed' }
& powercfg.exe /hibernate off
if ($LASTEXITCODE -ne 0) { throw 'Hibernation setting failed' }
& powercfg.exe /setacvalueindex SCHEME_CURRENT SUB_BUTTONS PBUTTONACTION 0
if ($LASTEXITCODE -ne 0) { throw 'AC power-button setting failed' }
& powercfg.exe /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS PBUTTONACTION 0
if ($LASTEXITCODE -ne 0) { throw 'DC power-button setting failed' }
& powercfg.exe /setactive SCHEME_CURRENT
& powercfg.exe /query SCHEME_CURRENT SUB_SLEEP | Set-Content -LiteralPath (Join-Path $foundationPath 'confirmatory-power-after-install.txt')
& powercfg.exe /qh SCHEME_CURRENT SUB_BUTTONS | Add-Content -LiteralPath (Join-Path $foundationPath 'confirmatory-power-after-install.txt')
& powercfg.exe /a | Add-Content -LiteralPath (Join-Path $foundationPath 'confirmatory-power-after-install.txt')
Write-Output 'Task installed and XML exported. Reboot acceptance still needs to be run and checked.'
