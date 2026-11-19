# Rebuild the web UI (only needed after changing frontend\src). Requires Node.js 18+.
$root = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $root "frontend")
npm ci; npm run build
Pop-Location
