[CmdletBinding()]
param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$pythonPath = Join-Path $resolvedRoot '.venv-compression\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw 'Missing isolated research Python environment.' }
$logDirectory = Join-Path $resolvedRoot 'research_output\logs'
[void](New-Item -ItemType Directory -Path $logDirectory -Force)
$logPath = Join-Path $logDirectory ('environment_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.log')
$probe = @'
import importlib, json, platform, sys
result = {"python": sys.version.split()[0], "platform": platform.platform(), "modules": {}}
required = ["numpy", "scipy", "pandas", "PIL", "torch", "torchvision", "matplotlib", "sklearn", "openpyxl", "pytest"]
failed = False
for name in required:
    try:
        module = importlib.import_module(name)
        result["modules"][name] = {"available": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as error:
        failed = True
        result["modules"][name] = {"available": False, "error_type": type(error).__name__}
try:
    from PIL import features
    result["codecs"] = {name: features.check(name) for name in ["jpg", "zlib", "webp", "jpg_2000"]}
    import torch
    result["cuda"] = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda}
    if torch.cuda.is_available():
        result["cuda"].update(gpu=torch.cuda.get_device_name(0), total_bytes=torch.cuda.get_device_properties(0).total_memory)
except Exception as error:
    failed = True
    result["probe_error_type"] = type(error).__name__
print(json.dumps(result, ensure_ascii=True, indent=2))
raise SystemExit(1 if failed else 0)
'@
Start-Transcript -LiteralPath $logPath | Out-Null
try {
    $probe | & $pythonPath -E -
    if ($LASTEXITCODE -ne 0) { throw 'One or more required imports failed; see the local environment transcript.' }
    Write-Output ('Environment checks passed. Local transcript: ' + $logPath)
}
finally { Stop-Transcript | Out-Null }
