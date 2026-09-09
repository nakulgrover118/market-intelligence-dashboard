from pydantic import BaseModel


class InstrumentOut(BaseModel):
    ticker: str
    name: str
    sector: str | None
    asset_class: str


class FeatureContribution(BaseModel):
    feature: str
    feature_value: float
    shap_value: float


class PredictionOut(BaseModel):
    ticker: str
    name: str
    sector: str | None
    horizon: int
    up_probability: float
    down_probability: float
    as_of_date: str


class PredictionDetailOut(PredictionOut):
    up_top_features: list[FeatureContribution]
    down_top_features: list[FeatureContribution]
