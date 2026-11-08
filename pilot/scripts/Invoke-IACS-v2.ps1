[CmdletBinding()]
param(
    [ValidateSet('collect','prepare','validate','statistics','report','test','decide')][string]$Step = 'collect',
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Config = 'configs/iacs_v2_development.json',
    [switch]$Resume,
    [switch]$RetryFailed,
    [string]$OutputPath,
    [string]$StatisticsPath,
    [string]$ImagePath,
    [string]$PolicyPath,
    [string]$PythonPath
)
$ErrorActionPreference = 'Stop'
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
if (-not $PythonPath) { $PythonPath = Join-Path $resolvedRoot '.venv-compression\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw 'Audited Python environment not found.' }
$configPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $resolvedRoot $Config }
$settings = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$logRoot = Join-Path $settings.run_root 'logs'
[void](New-Item -ItemType Directory -Path $logRoot -Force)
$logPath = Join-Path $logRoot ('powershell_' + $Step + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.log')
$arguments = @('-E','-m','iacs_v2',$Step,'--config',$configPath)
if ($Step -eq 'test') { $arguments = @('-E','-m','pytest','tests','tests_iacs_v2','-q') }
if ($Resume) { $arguments += '--resume' }
if ($RetryFailed) { $arguments += '--retry-failed' }
if ($OutputPath) { $arguments += @('--output',$OutputPath) }
if ($StatisticsPath) { $arguments += @('--statistics',$StatisticsPath) }
if ($ImagePath) { $arguments += @('--image',$ImagePath) }
if ($PolicyPath) { $arguments += @('--policy',$PolicyPath) }
Start-Transcript -LiteralPath $logPath | Out-Null
Push-Location -LiteralPath $resolvedRoot
try {
    & $PythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw ('IACS_v2 failed with exit code ' + $LASTEXITCODE) }
}
finally {
    Pop-Location
    Stop-Transcript | Out-Null
}
