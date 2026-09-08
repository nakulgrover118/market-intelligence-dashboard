# Market Intelligence Dashboard — Frontend

React + TypeScript + Vite frontend for the [backend API](../backend), consuming
`/instruments`, `/predictions/latest`, and `/predictions/{ticker}` to show
calibrated move-probabilities and their SHAP explanations. See
[../docs/roadmap.md](../docs/roadmap.md) for the full project methodology.

## Running locally

Requires the backend running first (`cd ../backend && uvicorn app.main:app`,
plus persisted models — see the backend README/`app/models/persist.py`).

```bash
npm install
npm run dev
```

Opens on `http://localhost:5173` by default, which the backend's CORS config
already allows. Override the API base URL with a `VITE_API_BASE_URL` env var
if the backend isn't on `http://127.0.0.1:8000`.

## Structure

```
src/
  api/          fetch client + TypeScript types mirroring the backend's Pydantic schemas
  components/   ProbabilityBar (magnitude indicator), ShapChart (diverging contribution chart)
  pages/        Dashboard (instrument list), InstrumentDetail (probability + explanation)
  theme.css     Colors — the dataviz skill's validated reference palette (light/dark)
```

## Design notes

- Probability is a magnitude → sequential blue ramp. SHAP contribution is a
  polarity (pushes probability up vs down) → the diverging blue/red pair.
  These are different color jobs on purpose, not a stylistic choice.
- The probability bar's visual scale caps at 30%, not 100%: predicted
  probabilities empirically cluster between ~3% and ~20% (see the reliability
  diagrams in `../docs/roadmap.md`), so a 0-100% scale would make every bar
  look nearly empty.
