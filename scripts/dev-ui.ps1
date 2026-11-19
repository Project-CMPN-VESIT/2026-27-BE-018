# Live-reload UI development on http://localhost:5173 (API must be running via run.ps1 on :8000).
$root = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $root "frontend")
if (-not (Test-Path node_modules)) { npm ci }
npm run dev
Pop-Location
