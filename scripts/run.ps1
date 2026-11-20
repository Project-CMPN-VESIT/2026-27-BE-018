# Start GeoLitterNet Watch (API + web UI) on http://localhost:8000
# Usage:  .\scripts\run.ps1  [-Port 8000] [-NoBrowser] [-Lan]
param([int]$Port = 8000, [switch]$NoBrowser, [switch]$Lan)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$vpy = Join-Path $backend ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { throw "Run .\scripts\setup.ps1 first." }
if (-not (Test-Path (Join-Path $root "frontend\dist\index.html"))) { Write-Warning "frontend\dist missing - the API will run without the web UI (build it with: cd frontend; npm ci; npm run build)." }
$bind = if ($Lan) { "0.0.0.0" } else { "127.0.0.1" }
if (-not $NoBrowser) { Start-Job -ScriptBlock { param($p) Start-Sleep 6; Start-Process "http://localhost:$p" } -ArgumentList $Port | Out-Null }
Push-Location $backend
try { & $vpy -m app --host $bind --port $Port }
finally { Pop-Location }
