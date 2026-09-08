# Roadmap

Each phase is built, tested, and confirmed before moving to the next.

- [x] **Phase 0 — Scoping & scaffold**: target definition, repo skeleton, FastAPI health check.
- [ ] **Phase 1 — Data acquisition & storage**: historical OHLCV for the pilot universe, clean ingestion pipeline, local storage (Parquet), no lookahead in the pipeline itself.
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

## Pilot universe (v1)

To be finalized in Phase 1, but the working plan:

- ~8-10 large-cap US stocks across a couple of sectors
- S&P 500 and Nasdaq Composite (or a representative ETF proxy, e.g. SPY/QQQ)
- Gold and Silver (futures tickers GC=F/SI=F, or ETF proxies GLD/SLV)

## Key methodology decisions log

- **Target**: `P(forward N-day return >= +theta%)`, N in {5, 20}, calibrated probability output. Fixed-horizon threshold labels now; triple-barrier (López de Prado) as a later upgrade.
- **Validation**: walk-forward / expanding-window CV only. No shuffled k-fold on time series.
- **Stack**: FastAPI + React (chosen over Streamlit/Dash for stronger engineering signal). Python dependency management via plain venv + pyproject.toml (no uv/poetry — kept simple, no extra installs required).
