# Roadmap

Each phase is built, tested, and confirmed before moving to the next.

- [x] **Phase 0 — Scoping & scaffold**: target definition, repo skeleton, FastAPI health check.
- [x] **Phase 1 — Data acquisition & storage**: historical OHLCV for the pilot universe, clean ingestion pipeline, local storage (Parquet), no lookahead in the pipeline itself.
- [x] **Phase 2a/2b — Cleaning & core feature engineering**: raw->processed cleaning (leading bad-price/zero-volume truncation), manual trailing-only technical indicators with no-lookahead tests.
- [x] **Phase 2c — Cross-asset features**: rolling beta/correlation vs Nifty 50, excess return vs Nifty, gold/silver returns, gold/silver ratio.
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
- **Cleaning**: only truncates *leading* unusable rows (bad core price or zero volume); a bad row appearing mid-series is flagged as a warning, never silently dropped, since dropping from the middle of a time series creates a gap that a `.shift()`-based feature would silently paper over. "Core price" columns are chosen based on what features actually consume (`open` is excluded) — a column-level decision informed by looking at what broke in real data, not a generic "clean everything" rule.
- **Feature engineering**: technical indicators implemented manually in pandas (not a library like pandas-ta or TA-Lib), specifically so every function's trailing-only windowing can be verified rather than trusted. Every indicator has a "no lookahead" test asserting its value at day t is unchanged when more history is appended after t — this is the actual leakage-safety guarantee, not just a spot-checked formula.
- **Cross-asset alignment**: any two instruments are joined via an *inner join* on date before computing a rolling stat (beta, correlation, excess return) — never forward-filled — since a fabricated "flat" day for a missing date would quietly bias the estimate. The result is reindexed back onto each instrument's own calendar, correctly reintroducing NaN wherever the reference series has no data that day.
- **Reference-feature skipping**: an instrument never gets a cross-asset feature computed against itself (e.g. `^NSEI` has no beta-vs-Nifty column, `GOLDBEES.NS` has no gold-return column) — those would be degenerate (beta≈1, correlation≈1) or exactly redundant with the instrument's own single-asset features.

## Data quality findings from Phase 1 (real, not hypothetical)

- **`TATAMOTORS.NS` returned HTTP 404.** Root cause: Tata Motors demerged on 2025-10-01 into `TMPV.NS` (passenger vehicles + JLR) and `TMLCV.NS` (commercial vehicles); the old ticker stopped resolving. Swapped the universe to `TMPV.NS`. Its pre-demerger history reflects the old combined entity, not passenger-vehicles-only — a structural break to keep in mind if this stock's features look inconsistent pre/post 2025-10-01.
- **`GOLDBEES.NS` has `Open = 0.0` for its first ~247 trading days (2009-2010)**, likely thin trading shortly after listing. Resolved in Phase 2 cleaning by *not* treating `Open` as a core column: no feature we compute uses raw Open, so these rows are left as-is rather than truncated (truncating would have destroyed real, valid Close/Volume data for no benefit). GOLDBEES also has ~353 mid-series rows with a bad price or zero volume scattered through its history (mostly early, thin-trading years) — flagged as a warning, not dropped, per the mid-series-vs-leading distinction below.
- **`NESTLEIND.NS` has a completely flat price and zero trading volume for its first ~5 years (2005-01-03 to 2010-01-07, 1238 rows).** This is vendor placeholder data, not a real market condition — no real instrument trades at an unchanged price with zero volume for five years. Generalized the cleaning truncation rule from "leading non-positive price" to "leading non-positive core price OR zero volume" to catch this class of issue; NESTLEIND's usable history now correctly starts 2010-01-08, and its feature warm-up rate went from 77% to 95% (in line with peers).
- **The NESTLEIND fix above initially broke `^NSEI` and `^NSEBANK`.** Both indices legitimately have `Volume == 0` for years at a time (`^NSEI`: its entire 2007-2013 history) — Yahoo doesn't populate real trade volume for an index, only for the securities that make it up, while the index *price level* moves completely normally over that period (verified: non-zero std, realistic 2007-2013 values including the financial crisis). The generalized zero-volume-means-fake rule, written with stocks/ETFs in mind, wrongly truncated 5+ years of valid Nifty history. Caught by an unexpectedly low cross-asset feature warm-up rate on Phase 2c features that depend on the Nifty series, not by inspection. Fixed by scoping the zero-volume check to `AssetClass.STOCK`/`COMMODITY` only — indices are checked on price validity alone.
- **Cross-asset feature coverage**: any feature derived from `SILVERBEES.NS` (`silver_return_*`, `gold_silver_ratio`) is only defined from its 2022 listing onward — about 4 years out of the ~20-year stock histories. A naive "drop any row with a NaN feature" policy at the modeling stage would therefore throw away ~80% of the dataset; Phase 4/5 must decide per-experiment whether to include silver-derived features (accepting a recent-period-only model) or exclude them (keeping full history). Excluding silver-derived columns, most stocks are warmed up on 80%+ of their history; the two indices sit lower (64-72%) purely from stacking warm-up costs (own SMA-200 window + gold's 2009 start + the 60-day beta/correlation window), not a further bug.
