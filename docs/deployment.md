# Deployment

This is a portfolio project — it isn't hosted anywhere paid, and that's a
deliberate choice, not an oversight. This document is the deployment
**story**: how the pieces are built to be deployed, and exactly what you'd
do to actually run them somewhere. Nothing here has been exercised against
a real cloud provider; the Dockerfile hasn't been `docker build`-tested in
this environment (no local Docker) — treat it as a well-formed starting
point, not a verified artifact, and run a real build before trusting it.

## The shape of the deployment

Three independently deployable pieces:

1. **Backend** (`backend/`) — a FastAPI app serving predictions from
   *persisted* model artifacts. It does not train on request, and it does
   not fetch market data on request — both of those are offline batch
   steps (see "Refreshing the data and model" below).
2. **Frontend** (`frontend/`) — a static Vite/React build. Once built, it's
   just HTML/CSS/JS files; any static host works.
3. **Data** (`data/`) — gitignored, machine-generated (raw/processed
   parquet, feature/label parquet, trained model `.joblib` artifacts). The
   backend reads this at request time; it is never baked into the backend
   image, so refreshing it doesn't require a rebuild.

## Why data is a mounted volume, not part of the image

`backend/app/data/paths.py` resolves where `data/` lives via the
`MARKET_DATA_ROOT` environment variable (falling back to a path computed
relative to the source file, which only works in the local dev checkout
layout — see the comment in that file). This split exists because:

- The full pipeline (ingest → clean → build features → label → tune →
  train → persist) takes real time and needs network access to Yahoo
  Finance — not something you want happening inside a container build.
- Model artifacts should be refreshable (new market data, a re-tuned
  model) without rebuilding and redeploying the backend image.

So the deployment pattern is: **build the backend image from code only**,
then run it with `data/` mounted as a volume that's populated separately.

## Running the full pipeline (produces `data/`)

From `backend/`, with dependencies installed (`pip install -e ".[dev]"`):

```bash
python -m app.data.ingest      # fetch raw OHLCV for the pilot universe (needs network)
python -m app.data.clean       # raw -> processed
python -m app.features.build   # processed -> features (technical + cross-asset)
python -m app.features.labels  # features -> labels (k=1.5 volatility-scaled)
python -m app.models.persist   # tunes + trains + saves data/models/lightgbm_{5,20}d.joblib
```

This is what "refreshing the model" means in this project — there is no
online/incremental retraining. Re-run this sequence (ingestion picks up
however much new history Yahoo has), redeploy the resulting `data/`
directory to wherever the backend reads its volume from, and restart the
backend process so `ModelRegistry`'s `lru_cache` singleton reloads the new
artifacts.

## Backend

**Locally / without Docker**, this is just:

```bash
cd backend
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Containerized** (`backend/Dockerfile`):

```bash
cd backend
docker build -t market-intel-backend .
docker run -p 8000:8000 \
  -v /absolute/path/to/data:/data \
  -e MARKET_CORS_ORIGINS=https://your-frontend-domain.example \
  market-intel-backend
```

Any container host that takes a Dockerfile and a volume works: Render,
Railway, Fly.io, a plain VM with `docker run`. The two things every one of
them needs configured:

- A **persistent volume** mounted at `/data`, pre-populated by the
  pipeline above (or populated by a one-off job/init container that runs
  it before the service starts).
- `MARKET_CORS_ORIGINS` set to the frontend's real deployed origin — the
  default only allows the local Vite dev server, and a mismatch here fails
  silently from the frontend's perspective (the browser blocks the
  response; the backend logs show nothing wrong).

## Frontend

Static build, deployable anywhere that serves static files (Vercel,
Netlify, Cloudflare Pages, S3+CloudFront, or the backend container itself
behind a reverse proxy):

```bash
cd frontend
npm ci
VITE_API_BASE_URL=https://your-backend-domain.example npm run build
```

`VITE_API_BASE_URL` is read at *build* time (Vite inlines `import.meta.env`
values into the bundle) — it must be set before `npm run build`, not as a
runtime environment variable on the static host. The build output lands in
`frontend/dist/`.

## CI

`.github/workflows/ci.yml` runs on every push/PR: the full backend test
suite (128+ tests, all against synthetic/mocked data — no network, no
trained model artifacts required, by design) and a frontend type-check +
build. Neither job deploys anything; this project has no CD step, since
there's nowhere it's actually hosted. Wiring a real CD step (build image,
push to a registry, deploy) would be the natural next addition once an
actual hosting target exists.

## What's deliberately not here

- **No secrets management** — nothing in this project needs an API key
  (yfinance is unauthenticated) or a database credential, so there's
  nothing to demonstrate here. A real deployment with secrets would use
  the host's standard mechanism (GitHub Actions secrets, Render/Fly env
  vars) rather than anything project-specific.
- **No autoscaling / load balancing story** — this serves single-digit
  requests per dashboard load against 23 instruments; it was never
  designed for meaningful traffic, and pretending otherwise would be
  over-engineering for what this project actually is.
- **No automated data refresh (cron)** — re-running the pipeline is a
  manual step here. A real system would schedule it (see
  `anthropic-skills:schedule` if using Claude Code's own scheduler, or a
  standard cron/Airflow/GitHub Actions scheduled workflow) — noted as a
  natural extension, not built, since there's no live deployment for it to
  refresh.
