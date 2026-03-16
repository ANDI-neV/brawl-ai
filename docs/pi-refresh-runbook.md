# BrawlAI Refresh Runbook

## Scope
- Train on the local workstation with the RTX 3090.
- Treat the database as disposable training state.
- Use the Raspberry Pi only as the deployment target for the backend/frontend runtime.

## Prerequisites
- Copy the ignored runtime configs into the checkout:
  - `backend/src/config.ini`
  - `frontend/frontend_config.ini`
- Create the local Python environment with `uv`:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r backend/src/requirements.txt
```

- Verify CUDA is available:

```bash
uv run --python /home/oleg/repos/brawl-ai/.venv/bin/python python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

## Fresh Data Collection
Do not train on the stale database contents. Reset the tables first.

```bash
uv run --python /home/oleg/repos/brawl-ai/.venv/bin/python python - <<'PY'
from db import Database
db = Database()
db.reset()
db.close()
PY
```

Then run a bounded fresh crawl from recent ranked battlelogs. The codebase currently supports `ranked` and `soloRanked` and ignores `tournament`.

Recommended refresh window:
- cutoff date: `01.03.2026`
- seed pool: `feeding.DevBrawlManager.best_players`
- crawl size: a few hundred battlelogs at most

After the crawl, confirm the DB contains only fresh data and includes current brawlers.

## Metadata and Model Refresh
Regenerate metadata and caches from the fresh DB snapshot:

```bash
cd backend/src
uv run --python /home/oleg/repos/brawl-ai/.venv/bin/python python scraper.py
```

Train and export locally on the 3090:

```bash
cd backend/src
PYTHONUNBUFFERED=1 uv run --python /home/oleg/repos/brawl-ai/.venv/bin/python \
  python ai.py --model-name model_latest.pth --epochs 10 --skip-metadata-refresh
```

Expected outputs:
- `backend/src/out/brawlers/brawlers.json`
- `backend/src/out/brawlers/stripped_brawlers.json`
- `backend/src/out/brawlers/brawler_supercell_id_mapping.json`
- `backend/src/out/brawlers/brawler_pickrates.json`
- `backend/src/out/brawlers/brawler_winrates.json`
- `backend/src/out/brawlers/map_data.json`
- `backend/src/out/models/map_id_mapping.json`
- `backend/src/out/models/model_latest.pth`
- `backend/src/out/models/model.onnx`

## Deployment to the Pi
1. Sync the branch to the Pi.
2. Copy the refreshed `backend/src/out/` artifacts to the Pi checkout.
3. Run the checked-in deploy script on the Pi. It creates a `uv`-managed runtime virtualenv under `/home/oleg/brawl-ai/.venv-runtime` and installs only `backend/src/requirements.runtime.txt`, so the Pi does not pull in local training dependencies like `torch`.

```bash
./scripts/deploy_pi.sh
```

4. Verify:

```bash
curl http://127.0.0.1:7001/healthz
curl http://127.0.0.1:7001/readiness
pm2 ls
```

## Rollback
- Pi backups live under `/home/oleg/backups/brawl-ai/<timestamp>`.
- Restore:
  - `backend-config.ini`
  - `frontend.env`
  - `frontend_config.ini`
  - `backend-out.tgz`
  - `pm2-jlist.json` or `pm2-dump.pm2`

If deployment fails, restore the backed-up `out/` directory and PM2 state before restarting the services.

## Notes
- The local workstation is the only supported training host going forward.
- The Pi can still be used to serve the refreshed ONNX model, but not to train it.
- `backend/schema.sql` is the source of truth for recreating the database schema from scratch.
