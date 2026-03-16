#!/bin/bash
set -euo pipefail

exec /home/oleg/.pyenv/versions/3.10.6/bin/gunicorn \
  -w "${BRAWL_AI_GUNICORN_WORKERS:-4}" \
  -k uvicorn.workers.UvicornWorker \
  web_server:app \
  --bind 0.0.0.0:7001
