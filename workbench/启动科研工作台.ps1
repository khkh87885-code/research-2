$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw '请先按 USER_MANUAL.md 创建运行环境。' }
& $taskPython -I (Join-Path $taskRoot 'run.py') serve --port 8765
