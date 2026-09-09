# Market Intelligence Dashboard — Frontend

Part of [Market & Commodities Intelligence Dashboard](https://github.com/nakulgrover118/market-intelligence-dashboard)
([top-level README](../README.md)) — React + TypeScript + Vite frontend for
the [backend API](../backend), consuming
`/instruments`, `/predictions/latest`, and `/predictions/{ticker}` to show
two independent, calibrated move-probabilities per instrument — up and
down — with their SHAP explanations. See
[../docs/roadmap.md](../docs/roadmap.md) for the full project methodology.

**Live**: [market-intelligence-dashboard-two-lilac.vercel.app](https://market-intelligence-dashboard-two-lilac.vercel.app) (backend is free-tier Render, so it can take ~50s to wake up if idle — see [`../docs/deployment.md`](../docs/deployment.md))

## Running locally

Requires the backend running first (`cd ../backend && uvicorn app.main:app`,
plus persisted models for both directions — see the backend README/
`app/models/persist.py`).

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
  components/   DirectionBar (diverging up/down probability bar), ShapChart (diverging contribution chart)
  pages/        Dashboard (instrument list), InstrumentDetail (both probabilities + both explanations)
  theme.css     Colors — the dataviz skill's validated reference palette (light/dark)
```

## Design notes

- **Why two probabilities, not one**: the backend serves independent "up" and
  "down" models (see `docs/roadmap.md`) — a single upside-only number can't
  honestly be read as bullish/bearish, since a low value could mean "calm" just
  as easily as "expect a drop." Both are shown together, always, never reduced
  to "whichever is bigger."
- **Direction is a polarity → diverging color, not a magnitude → sequential
  ramp.** `DirectionBar` (up vs down) and `ShapChart` (pushes probability up vs
  down) both use the same diverging blue/red pair for this reason — it's a
  color *job*, not a stylistic choice.
- Both probabilities can be elevated **simultaneously**: a calm market (low
  current volatility) lowers the bar for a big move in *either* direction
  (see `docs/roadmap.md`'s Phase 7/8/12 findings) — confirmed live, not just
  in theory: `HDFCBANK.NS` at the 5-day horizon shows up=16.6% and
  down=12.9% *at once*, both driven by the same low `realized_vol_20d`
  reading. `DirectionBar`'s derived "lean" label is based on the *spread*
  between up and down, not on which number is larger, precisely because of
  this.
- The bar's visual scale caps at 30%, not 100%: predicted probabilities
  empirically cluster between ~3% and ~20% (see the reliability diagrams in
  `../docs/roadmap.md`), so a 0-100% scale would make every bar look nearly
  empty.
