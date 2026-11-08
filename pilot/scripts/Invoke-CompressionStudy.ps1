[CmdletBinding()]
param(
    [ValidateSet('inventory','run','export','dashboard','test')][string]$Command = 'inventory',
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Config = 'configs/local.json',
    [string]$RunPath,
    [string]$OutputPath,
    [string[]]$ExtraArguments = @()
)
$ErrorActionPreference = 'Stop'
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$pythonPath = Join-Path $resolvedRoot '.venv-compression\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'The isolated research environment is missing: .venv-compression\Scripts\python.exe'
}
$logDirectory = Join-Path $resolvedRoot 'research_output\logs'
[void](New-Item -ItemType Directory -Path $logDirectory -Force)
$logPath = Join-Path $logDirectory ('powershell_' + $Command + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.log')
$arguments = @('-E', '-m', 'compression_study', $Command)
switch ($Command) {
    'inventory' {
        $arguments += @('--config', $Config)
        if (-not $OutputPath) { $OutputPath = 'research_output/inventory_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') }
        $arguments += @('--output', $OutputPath)
    }
    'run' { $arguments += @('--config', $Config) }
    'export' {
        if (-not $RunPath) { throw 'Provide -RunPath for an existing completed run.' }
        if (-not $OutputPath) { $OutputPath = Join-Path $RunPath 'export' }
        $arguments += @('--run', $RunPath, '--output', $OutputPath)
    }
    'dashboard' {
        if (-not $RunPath) { throw 'Provide -RunPath for an existing run.' }
        if (-not $OutputPath) { $OutputPath = Join-Path $RunPath 'dashboard.html' }
        $arguments += @('--run', $RunPath, '--output', $OutputPath)
    }
    'test' { $arguments = @('-E', '-m', 'pytest', 'tests', '-q') }
}
$arguments += $ExtraArguments
Start-Transcript -LiteralPath $logPath | Out-Null
Push-Location -LiteralPath $resolvedRoot
try {
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw ('Research command failed with exit code ' + $LASTEXITCODE) }
    Write-Output ('Completed. Local transcript: ' + $logPath)
}
finally {
    Pop-Location
    Stop-Transcript | Out-Null
}
