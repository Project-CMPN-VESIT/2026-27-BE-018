# Run the backend test-suite (model parity, providers, masks, API end-to-end).
$root = Split-Path -Parent $PSScriptRoot
$vpy = Join-Path $root "backend\.venv\Scripts\python.exe"
Push-Location (Join-Path $root "backend")
& $vpy -m pytest -q tests
Pop-Location
