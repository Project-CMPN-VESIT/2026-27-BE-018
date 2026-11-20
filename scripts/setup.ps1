# One-time setup on Windows (no Docker): creates backend\.venv and installs every dependency (CPU PyTorch).
# Usage: double-click Setup.bat, or from PowerShell in the project folder:  .\scripts\setup.ps1
# Everything is also written to setup.log in the project folder (useful for troubleshooting).
$ErrorActionPreference = "Continue"   # native tools (pip) write warnings to stderr; failures are checked via exit codes
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$log = Join-Path $root "setup.log"
Start-Transcript -Path $log -Force | Out-Null

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Run([string]$exe, [string[]]$argv) {
  Write-Host "> $exe $($argv -join ' ')" -ForegroundColor DarkGray
  & $exe @argv 2>&1 | ForEach-Object { "$_" }
  if ($LASTEXITCODE -ne 0) { throw "command failed (exit $LASTEXITCODE): $exe $($argv -join ' ')" }
}

try {
  Step "System"
  Write-Host "Windows: $([Environment]::OSVersion.VersionString)  PowerShell: $($PSVersionTable.PSVersion)"
  Write-Host "Project: $root"

  Step "Finding Python 3.10 - 3.13"
  $PyExe = $null; $PyArgs = @()
  foreach ($v in @("3.12", "3.11", "3.13", "3.10")) {
    try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $PyExe = "py"; $PyArgs = @("-$v"); break } } catch {}
  }
  if (-not $PyExe) {
    try { & python -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13)" 2>$null; if ($LASTEXITCODE -eq 0) { $PyExe = "python" } } catch {}
  }
  try { Write-Host "Installed via py launcher:"; & py -0p 2>&1 | ForEach-Object { "  $_" } } catch { Write-Host "  (py launcher not found)" }
  if (-not $PyExe) { throw "Python 3.10-3.13 not found. Install Python 3.12 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run Setup.bat again." }
  & $PyExe @PyArgs -c "import sys; print('Using', sys.executable, sys.version)"

  Step "Virtual environment (backend\.venv)"
  $venv = Join-Path $backend ".venv"
  $vpy = Join-Path $venv "Scripts\python.exe"
  if (-not (Test-Path $vpy)) { Run $PyExe ($PyArgs + @("-m", "venv", $venv)) } else { Write-Host "exists, reusing" }
  Run $vpy @("-m", "pip", "install", "--upgrade", "pip", "wheel", "--disable-pip-version-check")

  Step "PyTorch (CPU build, ~200 MB)"
  Run $vpy @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu", "--disable-pip-version-check")

  Step "Backend requirements"
  Run $vpy @("-m", "pip", "install", "-r", (Join-Path $backend "requirements.txt"), "--disable-pip-version-check")

  Step "Configuration"
  if (-not (Test-Path (Join-Path $root ".env"))) { Copy-Item (Join-Path $root ".env.example") (Join-Path $root ".env"); Write-Host "created .env from .env.example" }
  else { Write-Host ".env exists, kept" }

  Step "Model check"
  Push-Location $backend
  & $vpy -c "from app.config import get_settings as g; from app.ml.engine import ModelEngine as M; s=g(); e=M(s.model_path); e.load(); print('model OK:', e.n_params, 'parameters')" 2>&1 | ForEach-Object { "$_" }
  $mc = $LASTEXITCODE
  Pop-Location

  Step "Result"
  if ($mc -ne 0) { throw "the model could not be loaded - see above" }
  Write-Host "SETUP COMPLETE. Start the app with Start.bat (or .\scripts\run.ps1) and open http://localhost:8000" -ForegroundColor Green
}
catch {
  Write-Host "`nSETUP FAILED: $($_.Exception.Message)" -ForegroundColor Red
  Write-Host "Full log: $log"
  Stop-Transcript | Out-Null
  exit 1
}
Stop-Transcript | Out-Null
