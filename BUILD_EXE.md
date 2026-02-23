# Build FishingBot EXE (Windows)

## Prerequisites
- Python 3.11+ on Windows
- WoW target machine is Windows
- Optional for installer: Inno Setup 6 (for `ISCC.exe`)

## Build
From the project root:

```powershell
.\build.ps1 -Clean
```

Or double-click:
- `build.cmd` (runs `build.ps1` with any args you pass)

Output:
- `dist\FishingBot\FishingBot.exe`

## Build installer (recommended for end users)
From the project root:

```powershell
.\build.ps1 -Clean -Installer -AppVersion 1.0.0
```

Double-click option:
```bat
build.cmd -Clean -Installer -AppVersion 1.0.0
```

Output:
- `installer\output\FishingBot-Setup-1.0.0.exe`

## Runtime behavior
- On first run, the app creates user data at:
  - `%APPDATA%\FishingBot\config.yaml`
  - `%APPDATA%\FishingBot\logs\`
  - `%APPDATA%\FishingBot\templates\` (seeded from bundled templates)
- If no flags are provided in the EXE, it defaults to GUI mode.
- To force legacy CLI mode in EXE:

```powershell
FishingBot.exe --cli
```

## Release checklist
1. Run `.\build.ps1 -Clean -Installer -AppVersion <version>`
2. Smoke test `dist\FishingBot\FishingBot.exe --gui`
3. Smoke test installed app from `installer\output\FishingBot-Setup-<version>.exe`
4. Publish the installer `.exe` (zip the portable `dist\FishingBot\` only if needed)
5. (Recommended) Code-sign the executable and installer before distribution
