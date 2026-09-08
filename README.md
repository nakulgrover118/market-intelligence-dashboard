# Market & Commodities Intelligence Dashboard

A probabilistic forecasting system for equities, major indices, gold, and silver.
Rather than emitting BUY/SELL signals, the system estimates **calibrated
probabilities** of future price movements over fixed horizons, using a
combination of statistical baselines, machine learning, and an explainability
layer.

## Status

Early scaffold. See `docs/roadmap.md` for the phased build plan.

## Core methodology (summary)

- **Target**: for each asset and horizon N (5 and 20 trading days), predict
  `P(forward return >= +theta%)` as a calibrated probability, not a point
  forecast or a binary signal.
- **Validation**: walk-forward / expanding-window cross-validation only.
  Random k-fold CV is invalid for time series and is not used anywhere in
  this project.
- **Calibration**: raw classifier scores are calibrated (Platt/isotonic)
  before being reported as probabilities.
- **Explainability**: SHAP-based feature attribution accompanies every
  prediction shown in the UI.

## Project layout

```
backend/    FastAPI service: data pipeline, features, models, backtesting, API
frontend/   React dashboard (added in a later phase)
notebooks/  Exploratory research notebooks
docs/       Design notes and the phased roadmap
```

## Running the backend (dev)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
pytest
```
