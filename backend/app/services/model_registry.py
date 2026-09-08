"""Loads persisted model artifacts (see app.models.persist) and serves
predictions/explanations against the latest available feature data — no
retraining per request. One TreeExplainer per horizon is built once at
load time and reused across requests, since building it is the relatively
expensive part; predicting from it per row is cheap.
"""

from functools import lru_cache

import pandas as pd
import shap

from app.data.paths import FEATURES_DATA_DIR, MODELS_DATA_DIR, ticker_filename
from app.data.universe import UNIVERSE, Instrument
from app.models.explain import explain_single_prediction, normalize_shap_output

_INSTRUMENT_BY_TICKER: dict[str, Instrument] = {i.ticker: i for i in UNIVERSE}


class ModelNotAvailableError(RuntimeError):
    """Raised when a persisted model artifact hasn't been trained yet —
    run `python -m app.models.persist` first."""


class ModelRegistry:
    def __init__(self, horizons: tuple[int, ...] = (5, 20)):
        self._artifacts: dict[int, dict] = {}
        self._explainers: dict[int, shap.TreeExplainer] = {}
        for horizon in horizons:
            path = MODELS_DATA_DIR / f"lightgbm_{horizon}d.joblib"
            if not path.exists():
                continue
            import joblib

            artifact = joblib.load(path)
            self._artifacts[horizon] = artifact
            self._explainers[horizon] = shap.TreeExplainer(artifact["pipeline"].named_steps["model"])

    def available_horizons(self) -> list[int]:
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
        self, horizon: int, ticker: str, include_explanation: bool = True, top_n: int = 10
    ) -> dict | None:
        if horizon not in self._artifacts:
            raise ModelNotAvailableError(
                f"No persisted model for horizon={horizon}d — run `python -m app.models.persist` first."
            )
        artifact = self._artifacts[horizon]
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
            "probability": probability,
            "as_of_date": row.name.isoformat(),
        }

        if include_explanation:
            preprocessor = pipeline.named_steps["preprocess"]
            X_transformed = preprocessor.transform(X)
            feature_names = list(preprocessor.get_feature_names_out())
            shap_values = normalize_shap_output(self._explainers[horizon].shap_values(X_transformed))
            explanation = explain_single_prediction(shap_values, feature_names, 0, X_transformed)
            result["top_features"] = explanation.head(top_n).to_dict(orient="records")

        return result


@lru_cache
def get_registry() -> ModelRegistry:
    """Singleton so model artifacts and TreeExplainers are loaded once per
    process, not once per request."""
    return ModelRegistry()
