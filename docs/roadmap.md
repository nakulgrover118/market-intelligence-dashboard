# Roadmap

Each phase is built, tested, and confirmed before moving to the next.

- [x] **Phase 0 — Scoping & scaffold**: target definition, repo skeleton, FastAPI health check.
- [x] **Phase 1 — Data acquisition & storage**: historical OHLCV for the pilot universe, clean ingestion pipeline, local storage (Parquet), no lookahead in the pipeline itself.
- [ ] **Phase 2 — Feature engineering**: technical indicators, volatility/regime features, cross-asset features.
- [ ] **Phase 3 — Labeling**: fixed-horizon threshold labels first; triple-barrier labeling as an upgrade.
- [ ] **Phase 4 — Baseline models**: logistic regression / GARCH baselines to set a floor.
- [ ] **Phase 5 — ML models**: gradient boosting (XGBoost/LightGBM) with walk-forward cross-validation.
- [ ] **Phase 6 — Calibration**: Platt scaling / isotonic regression on top of raw model scores.
- [ ] **Phase 7 — Explainability**: SHAP attributions per prediction.
- [ ] **Phase 8 — Backtesting & evaluation**: Brier score, log-loss, reliability diagrams; no accuracy-only claims.
- [ ] **Phase 9 — AI narrative layer** (optional): LLM-generated synthesis over quant outputs.
- [ ] **Phase 10 — Dashboard UI**: React frontend consuming the FastAPI backend.
- [ ] **Phase 11 — Tests, CI, deployment docs**.

## Pilot universe (v1) — Indian market

Finalized in Phase 1 (`backend/app/data/universe.py` is the source of truth):

- 23 NSE-listed instruments, fetched via `yfinance`.
- 22 large-cap stocks across IT, Banking, Energy, FMCG, Auto, Pharma, Infrastructure, Metals, Telecom.
- Indices: `^NSEI` (Nifty 50), `^NSEBANK` (Nifty Bank).
- Commodities (ETF proxies — MCX futures are not available via yfinance): `GOLDBEES.NS`, `SILVERBEES.NS`.
- Switched from US to Indian market: same data-source coverage and rigor, more differentiated on a CV, and NSE-native ETF proxies for gold/silver work just as well as the US GLD/SLV pattern would have.

## Key methodology decisions log

- **Target**: `P(forward N-day return >= +theta%)`, N in {5, 20}, calibrated probability output. Fixed-horizon threshold labels now; triple-barrier (López de Prado) as a later upgrade.
- **Validation**: walk-forward / expanding-window CV only. No shuffled k-fold on time series. Must use the NSE trading calendar derived from the data itself, not a generic/US calendar.
- **Stack**: FastAPI + React (chosen over Streamlit/Dash for stronger engineering signal). Python dependency management via plain venv + pyproject.toml (no uv/poetry — kept simple, no extra installs required).
- **Ingestion**: raw layer is immutable (`backend/data/raw/*.parquet`, one file per ticker, gitignored). Validation flags issues (duplicate dates, non-monotonic index, non-positive prices, missing columns) without silently fixing them — cleaning happens in a later, explicit stage.

## Data quality findings from Phase 1 (real, not hypothetical)

- **`TATAMOTORS.NS` returned HTTP 404.** Root cause: Tata Motors demerged on 2025-10-01 into `TMPV.NS` (passenger vehicles + JLR) and `TMLCV.NS` (commercial vehicles); the old ticker stopped resolving. Swapped the universe to `TMPV.NS`. Its pre-demerger history reflects the old combined entity, not passenger-vehicles-only — a structural break to keep in mind if this stock's features look inconsistent pre/post 2025-10-01.
- **`GOLDBEES.NS` has `Open = 0.0` for its first ~247 trading days (2009-2010)**, likely thin trading shortly after listing. Caught by the ingestion validator, not fixed at this layer — to be handled explicitly when we build the cleaning step (most likely: truncate the series to start once `Open` becomes reliably non-zero).
