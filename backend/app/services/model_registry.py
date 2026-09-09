"""Loads persisted model artifacts (see app.models.persist) and serves
predictions/explanations against the latest available feature data — no
retraining per request. One TreeExplainer per (direction, horizon) is
built once at load time and reused across requests, since building it is
the relatively expensive part; predicting from it per row is cheap.

Two directions, not one: "up" and "down" are independent models sharing
the same features — see docs/roadmap.md for why a single upside-only
probability can't honestly support a bullish/bearish read.
"""

from functools import lru_cache

import joblib
import pandas as pd
import shap

from app.data.paths import FEATURES_DATA_DIR, MODELS_DATA_DIR, ticker_filename
from app.data.universe import UNIVERSE, Instrument
from app.models.explain import explain_single_prediction, normalize_shap_output
from app.models.persist import artifact_filename

_INSTRUMENT_BY_TICKER: dict[str, Instrument] = {i.ticker: i for i in UNIVERSE}


class ModelNotAvailableError(RuntimeError):
    """Raised when a persisted model artifact hasn't been trained yet —
    run `python -m app.models.persist` first."""


class ModelRegistry:
    def __init__(self, horizons: tuple[int, ...] = (5, 20), directions: tuple[str, ...] = ("up", "down")):
        self._artifacts: dict[tuple[str, int], dict] = {}
        self._explainers: dict[tuple[str, int], shap.TreeExplainer] = {}
        for direction in directions:
            for horizon in horizons:
                path = MODELS_DATA_DIR / artifact_filename(direction, horizon)
                if not path.exists():
                    continue
                artifact = joblib.load(path)
                key = (direction, horizon)
                self._artifacts[key] = artifact
                self._explainers[key] = shap.TreeExplainer(artifact["pipeline"].named_steps["model"])

    def available_models(self) -> list[tuple[str, int]]:
        return sorted(self._artifacts.keys())

    def _latest_feature_row(self, ticker: str, numeric_columns: list[str]) -> pd.Series | None:
        path = FEATURES_DATA_DIR / ticker_filename(ticker)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        valid = df.dropna(subset=numeric_columns)
        if valid.empty:
            return None
        return valid.iloc[-1]

    def predict_and_explain(
        self, direction: str, horizon: int, ticker: str, include_explanation: bool = True, top_n: int = 10
    ) -> dict | None:
        key = (direction, horizon)
        if key not in self._artifacts:
            raise ModelNotAvailableError(
                f"No persisted model for direction={direction!r} horizon={horizon}d — "
                "run `python -m app.models.persist` first."
            )
        artifact = self._artifacts[key]
        numeric_columns = artifact["numeric_columns"]
        categorical_columns = artifact["categorical_columns"]

        row = self._latest_feature_row(ticker, numeric_columns)
        if row is None:
            return None

        instrument = _INSTRUMENT_BY_TICKER.get(ticker)
        sector = (instrument.sector or instrument.asset_class.value) if instrument else "Unknown"
        X = pd.DataFrame([row[numeric_columns]])
        X["sector"] = sector
        X = X[numeric_columns + categorical_columns]

        pipeline = artifact["pipeline"]
        probability = float(pipeline.predict_proba(X)[0, 1])

        result = {
            "ticker": ticker,
            "horizon": horizon,
            "direction": direction,
            "probability": probability,
            "as_of_date": row.name.isoformat(),
        }

        if include_explanation:
            preprocessor = pipeline.named_steps["preprocess"]
            X_transformed = preprocessor.transform(X)
            feature_names = list(preprocessor.get_feature_names_out())
            shap_values = normalize_shap_output(self._explainers[key].shap_values(X_transformed))
            explanation = explain_single_prediction(shap_values, feature_names, 0, X_transformed)
            result["top_features"] = explanation.head(top_n).to_dict(orient="records")

        return result

    def predict_both_directions(
        self, horizon: int, ticker: str, include_explanation: bool = True, top_n: int = 10
    ) -> dict | None:
        """Convenience wrapper returning `{"up": {...}, "down": {...}}` in
        one call — the shape a dashboard entry actually needs, since
        showing a bullish/bearish lean requires both directions together,
        not two separate round trips. None if either direction has no
        feature data for this ticker (a model simply not being trained
        yet is a different case — that still raises ModelNotAvailableError,
        same as predict_and_explain, since it's a setup problem, not a
        per-ticker data gap)."""
        up = self.predict_and_explain("up", horizon, ticker, include_explanation, top_n)
        down = self.predict_and_explain("down", horizon, ticker, include_explanation, top_n)
        if up is None or down is None:
            return None
        return {"up": up, "down": down}


@lru_cache
def get_registry() -> ModelRegistry:
    """Singleton so model artifacts and TreeExplainers are loaded once per
    process, not once per request."""
    return ModelRegistry()
