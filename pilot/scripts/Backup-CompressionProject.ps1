[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$BackupDirectory
)
$ErrorActionPreference = 'Stop'
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path.TrimEnd('\')
$approvedBase = [IO.Path]::GetFullPath((Join-Path $resolvedRoot 'upgrade_backups')).TrimEnd('\')
if (-not $BackupDirectory) { $BackupDirectory = Join-Path $approvedBase ('code_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')) }
$backupTarget = [IO.Path]::GetFullPath($BackupDirectory).TrimEnd('\')
if (-not $backupTarget.StartsWith($approvedBase + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Backup destination must be a new child directory under this project upgrade_backups.'
}
if (Test-Path -LiteralPath $backupTarget) { throw 'Backup destination already exists; refusing to overwrite.' }
# Only selected code and configuration files are copied. No images, weights or result trees.
$files = @(Get-ChildItem -LiteralPath $resolvedRoot -File | Where-Object { $_.Extension -in @('.py','.md','.json') })
$configDirectory = Join-Path $resolvedRoot 'configs'
if (Test-Path -LiteralPath $configDirectory -PathType Container) {
    $configItem = Get-Item -LiteralPath $configDirectory
    if (($configItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Linked configuration directories are not backed up.' }
    $linkedDirectories = @(Get-ChildItem -LiteralPath $configDirectory -Recurse -Directory | Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 })
    if ($linkedDirectories.Count -gt 0) { throw 'Linked configuration subdirectories are not backed up.' }
    $files += @(Get-ChildItem -LiteralPath $configDirectory -Recurse -File | Where-Object { $_.Extension -in @('.json','.yaml','.yml','.toml','.ini') })
}
[void](New-Item -ItemType Directory -Path $backupTarget)
$manifest = @()
foreach ($file in $files) {
    if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Linked files are not included in the backup.' }
    $sourcePath = [IO.Path]::GetFullPath($file.FullName)
    if (-not $sourcePath.StartsWith($resolvedRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Source escapes the project directory.' }
    $relative = $sourcePath.Substring($resolvedRoot.Length + 1)
    $destination = Join-Path $backupTarget $relative
    [void](New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force)
    Copy-Item -LiteralPath $sourcePath -Destination $destination -ErrorAction Stop
    $sourceHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash
    $copiedHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
    if ($sourceHash -ne $copiedHash) { throw ('Backup hash verification failed: ' + $relative) }
    $manifest += [pscustomobject]@{ relative_path = $relative; bytes = $file.Length; sha256 = $sourceHash; verified = $true }
}
$manifestPath = Join-Path $backupTarget 'BACKUP_MANIFEST.json'
ConvertTo-Json -InputObject @($manifest) -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Output ('Verified ' + $manifest.Count + ' code/config files. Backup: ' + $backupTarget)
