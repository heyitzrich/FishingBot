param(
    [switch]$Clean,
    [switch]$Installer,
    [string]$AppVersion = "1.0.0"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Get-IsccPath {
    $candidates = @(
        "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    return $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

Write-Host "FishingBot EXE build starting..."

python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

if ($Clean) {
    Write-Host "Cleaning previous build artifacts..."
    Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force installer\output -ErrorAction SilentlyContinue
}

Write-Host "Running PyInstaller..."
python -m PyInstaller --clean FishingBot.spec

$exePath = Join-Path $RepoRoot "dist\FishingBot\FishingBot.exe"
if (-not (Test-Path $exePath)) {
    throw "Build succeeded but expected executable was not found at $exePath"
}

Write-Host ""
Write-Host "Build complete."
Write-Host "Executable folder: dist\\FishingBot"
Write-Host "Executable file:   dist\\FishingBot\\FishingBot.exe"

if (-not $Installer) {
    return
}

Write-Host ""
Write-Host "Building installer..."

$iscc = Get-IsccPath
if (-not $iscc) {
    throw "Inno Setup 6 (ISCC.exe) not found. Install Inno Setup and rerun with -Installer."
}

$issPath = Join-Path $RepoRoot "installer\FishingBot.iss"
if (-not (Test-Path $issPath)) {
    throw "Installer script not found: $issPath"
}

New-Item -ItemType Directory -Path (Join-Path $RepoRoot "installer\output") -Force | Out-Null
& $iscc "/DMyAppVersion=$AppVersion" "/DSourceDir=$RepoRoot" $issPath
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup failed with exit code $LASTEXITCODE"
}

$installerFile = Get-ChildItem (Join-Path $RepoRoot "installer\output") -Filter "FishingBot-Setup-*.exe" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

Write-Host "Installer output: installer\\output"
if ($installerFile) {
    Write-Host "Installer file:   $($installerFile.FullName)"
}
