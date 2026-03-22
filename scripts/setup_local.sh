#!/usr/bin/env bash
# scripts/setup_local.sh
# Run this once to set up local development environment.
# Usage: bash scripts/setup_local.sh

set -e

echo ""
echo "╔══════════════════════════════════╗"
echo "║   T-GIB Local Setup              ║"
echo "╚══════════════════════════════════╝"
echo ""

# ── 1. Python virtualenv ──────────────────────────────────────────────────
echo "→ Creating Python virtualenv..."
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r backend/requirements.txt
echo "  ✓ Python dependencies installed"

# ── 2. Check Docker ───────────────────────────────────────────────────────
if ! command -v docker &> /dev/null; then
  echo "  ⚠  Docker not found. Install from https://docs.docker.com/get-docker/"
else
  echo "  ✓ Docker found: $(docker --version)"
fi

echo ""
echo "Setup complete. To start:"
echo ""
echo "  Option A — Docker (recommended):"
echo "    docker compose up --build"
echo "    → Backend:  http://localhost:8000/docs"
echo "    → Frontend: http://localhost:3000"
echo ""
echo "  Option B — Python only (no Docker):"
echo "    source .venv/bin/activate"
echo "    cd backend && uvicorn app.main:app --reload --port 8000"
echo "    # Then open frontend/public/index.html in your browser"
echo ""
echo "  Run tests:"
echo "    source .venv/bin/activate"
echo "    pytest backend/tests/ -v"
echo ""
