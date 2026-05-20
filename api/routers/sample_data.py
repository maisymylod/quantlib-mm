"""Sample data endpoint exposing synthetic price/return series."""

from fastapi import APIRouter, HTTPException

from ..sample_data import available_tickers, get_prices, get_returns
from ..schemas import SampleDataResponse

router = APIRouter()


@router.get("/sample-data/tickers", response_model=list[str])
def list_tickers() -> list[str]:
    return available_tickers()


@router.get("/sample-data/{ticker}", response_model=SampleDataResponse)
def fetch(ticker: str) -> SampleDataResponse:
    if ticker not in available_tickers():
        raise HTTPException(
            status_code=404,
            detail=f"Unknown ticker {ticker!r}. Available: {available_tickers()}",
        )
    return SampleDataResponse(
        ticker=ticker,
        prices=get_prices(ticker).tolist(),
        returns=get_returns(ticker).tolist(),
    )
