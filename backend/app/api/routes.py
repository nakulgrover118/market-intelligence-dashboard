from fastapi import APIRouter, HTTPException

from app.api.schemas import InstrumentOut, PredictionDetailOut, PredictionOut
from app.data.universe import UNIVERSE
from app.services.model_registry import ModelNotAvailableError, get_registry

router = APIRouter()

VALID_HORIZONS = (5, 20)


def _validate_horizon(horizon: int) -> None:
    # Not Literal[5, 20]: FastAPI/Pydantic v2 doesn't coerce a query
    # string ("5") into an int Literal the way it does for a plain `int`
    # annotation, so Literal here would 422 on every legitimate request.
    if horizon not in VALID_HORIZONS:
        raise HTTPException(status_code=422, detail=f"horizon must be one of {VALID_HORIZONS}")


@router.get("/instruments", response_model=list[InstrumentOut])
def list_instruments() -> list[InstrumentOut]:
    return [
        InstrumentOut(ticker=i.ticker, name=i.name, sector=i.sector, asset_class=i.asset_class.value)
        for i in UNIVERSE
    ]


@router.get("/predictions/latest", response_model=list[PredictionOut])
def latest_predictions(horizon: int = 5) -> list[PredictionOut]:
    _validate_horizon(horizon)
    registry = get_registry()
    try:
        results = []
        for instrument in UNIVERSE:
            prediction = registry.predict_and_explain(horizon, instrument.ticker, include_explanation=False)
            if prediction is None:
                continue
            results.append(
                PredictionOut(
                    ticker=instrument.ticker,
                    name=instrument.name,
                    sector=instrument.sector,
                    horizon=horizon,
                    probability=prediction["probability"],
                    as_of_date=prediction["as_of_date"],
                )
            )
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    results.sort(key=lambda r: r.probability, reverse=True)
    return results


@router.get("/predictions/{ticker}", response_model=PredictionDetailOut)
def prediction_detail(ticker: str, horizon: int = 5) -> PredictionDetailOut:
    _validate_horizon(horizon)
    instrument = next((i for i in UNIVERSE if i.ticker == ticker), None)
    if instrument is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")

    registry = get_registry()
    try:
        result = registry.predict_and_explain(horizon, ticker, include_explanation=True, top_n=10)
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if result is None:
        raise HTTPException(status_code=404, detail=f"No feature data available yet for {ticker}")

    return PredictionDetailOut(
        ticker=ticker,
        name=instrument.name,
        sector=instrument.sector,
        horizon=horizon,
        probability=result["probability"],
        as_of_date=result["as_of_date"],
        top_features=result["top_features"],
    )
