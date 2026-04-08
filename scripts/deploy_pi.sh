#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
BACKEND_VENV="${BACKEND_VENV:-$ROOT_DIR/.venv-runtime}"

cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for deployment"
  exit 1
fi

if [[ ! -f backend/src/config.ini ]]; then
  echo "backend/src/config.ini is missing"
  exit 1
fi

if [[ ! -f frontend/frontend_config.ini ]]; then
  echo "frontend/frontend_config.ini is missing"
  exit 1
fi

pushd backend/src >/dev/null
rm -rf "$BACKEND_VENV"
uv venv "$BACKEND_VENV" --python "$PYTHON_BIN"
uv pip install --python "$BACKEND_VENV/bin/python" -r requirements.runtime.txt
"$BACKEND_VENV/bin/python" - <<'PY'
import web_server
print("backend import ok", hasattr(web_server, "app"))
PY
popd >/dev/null

pushd frontend >/dev/null
npm ci
rm -rf .next
npm run build
popd >/dev/null

pm2 delete frontend >/dev/null 2>&1 || true
pkill -f "/home/oleg/brawl-ai/frontend/node_modules/.bin/next start -p 3003" >/dev/null 2>&1 || true
pm2 startOrReload ecosystem.config.js --only brawl-backend --update-env
pm2 start ecosystem.config.js --only frontend --update-env
pm2 save
