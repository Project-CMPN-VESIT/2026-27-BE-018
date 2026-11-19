# Diagnostics. Add -Live to download a real Sentinel-2 scene and run the model end to end.
param([switch]$Live, [string]$Provider = "")
$root = Split-Path -Parent $PSScriptRoot
$vpy = Join-Path $root "backend\.venv\Scripts\python.exe"
Push-Location (Join-Path $root "backend")
$a = @("-m", "app.selfcheck"); if ($Live) { $a += "--live" }; if ($Provider) { $a += @("--provider", $Provider) }
& $vpy @a
Pop-Location
