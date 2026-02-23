# Build FishingBot EXE (Windows)

## Prerequisites
- Python 3.11+ on Windows
- WoW target machine is Windows

## Build
From the project root:

```powershell
.\build.ps1 -Clean
```

Output:
- `dist\FishingBot\FishingBot.exe`

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
1. Run `.\build.ps1 -Clean`
2. Smoke test `dist\FishingBot\FishingBot.exe --gui`
3. Zip `dist\FishingBot\` and publish
4. (Recommended) Code-sign the executable before distribution
