#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SPEC="${PYTHON_SPEC:-${PYTHON_BIN:-3.12}}"
BACKEND_VENV="${BACKEND_VENV:-$ROOT_DIR/.venv-runtime}"

cd "$ROOT_DIR"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user uv
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
if [[ "$PYTHON_SPEC" == */* ]] || command -v "$PYTHON_SPEC" >/dev/null 2>&1; then
  PYTHON_FOR_VENV="$PYTHON_SPEC"
else
  uv python install "$PYTHON_SPEC"
  PYTHON_FOR_VENV="$PYTHON_SPEC"
fi
uv venv "$BACKEND_VENV" --python "$PYTHON_FOR_VENV"
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
