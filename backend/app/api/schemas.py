from pydantic import BaseModel


class InstrumentOut(BaseModel):
    ticker: str
    name: str
    sector: str | None
    asset_class: str


class PredictionOut(BaseModel):
    ticker: str
    name: str
    sector: str | None
    horizon: int
    probability: float
    as_of_date: str


class FeatureContribution(BaseModel):
    feature: str
    feature_value: float
    shap_value: float


class PredictionDetailOut(PredictionOut):
    top_features: list[FeatureContribution]
