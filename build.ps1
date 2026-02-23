param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

Write-Host "FishingBot EXE build starting..."

python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

if ($Clean) {
    Write-Host "Cleaning previous build artifacts..."
    Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
}

Write-Host "Running PyInstaller..."
python -m PyInstaller --clean FishingBot.spec

Write-Host ""
Write-Host "Build complete."
Write-Host "Executable folder: dist\\FishingBot"
Write-Host "Executable file:   dist\\FishingBot\\FishingBot.exe"
