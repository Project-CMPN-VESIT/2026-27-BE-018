#!/usr/bin/env bash
# Linux/macOS setup (no Docker)
set -euo pipefail
cd "$(dirname "$0")/../backend"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip wheel
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-dev.txt
[ -f ../.env ] || cp ../.env.example ../.env
.venv/bin/python -m app.selfcheck || true
echo "Start with: scripts/run.sh"
