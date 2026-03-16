#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_VENV="${BRAWL_AI_BACKEND_VENV:-$ROOT_DIR/.venv-runtime}"
PYTHON_BIN="${BRAWL_AI_RUNTIME_PYTHON:-$BACKEND_VENV/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "backend runtime python not found at $PYTHON_BIN"
  exit 1
fi

exec "$PYTHON_BIN" -m gunicorn \
  -w "${BRAWL_AI_GUNICORN_WORKERS:-4}" \
  -k uvicorn.workers.UvicornWorker \
  web_server:app \
  --bind 0.0.0.0:7001
