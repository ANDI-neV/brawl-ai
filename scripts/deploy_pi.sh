#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/oleg/.pyenv/versions/3.10.6/bin/python}"
PIP_BIN="${PIP_BIN:-/home/oleg/.pyenv/versions/3.10.6/bin/pip}"

cd "$ROOT_DIR"

if [[ ! -f backend/src/config.ini ]]; then
  echo "backend/src/config.ini is missing"
  exit 1
fi

if [[ ! -f frontend/frontend_config.ini ]]; then
  echo "frontend/frontend_config.ini is missing"
  exit 1
fi

pushd backend/src >/dev/null
"$PIP_BIN" install -r requirements.txt
"$PYTHON_BIN" - <<'PY'
import web_server
print("backend import ok", hasattr(web_server, "app"))
PY
popd >/dev/null

pushd frontend >/dev/null
npm install
rm -rf .next
npm run build
popd >/dev/null

pm2 startOrReload ecosystem.config.js --update-env
pm2 save
