"""Time series analysis endpoint."""

from fastapi import APIRouter, HTTPException

from quantlib_mm import ReturnsAnalyzer

from ..sample_data import available_tickers, get_prices
from ..schemas import TimeSeriesRequest, TimeSeriesResponse

router = APIRouter()


@router.post("/timeseries/analyze", response_model=TimeSeriesResponse)
def analyze(req: TimeSeriesRequest) -> TimeSeriesResponse:
    if req.ticker not in available_tickers():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ticker {req.ticker!r}. Available: {available_tickers()}",
        )

    prices = get_prices(req.ticker)
    ra = ReturnsAnalyzer(prices)

    jb_stat, jb_p = ra.jarque_bera_test()
    return TimeSeriesResponse(
        log_returns=ra.log_returns().tolist(),
        rolling_vol=ra.rolling_volatility(window=req.rolling_window).tolist(),
        ewma_vol=ra.ewma_volatility(span=req.ewma_span).tolist(),
        skewness=ra.skewness(),
        kurtosis=ra.kurtosis(),
        jarque_bera_stat=jb_stat,
        jarque_bera_pvalue=jb_p,
        autocorrelation=ra.autocorrelation(lags=req.acf_lags).tolist(),
    )
