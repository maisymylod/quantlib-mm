"""Risk metrics endpoint."""

import numpy as np
from fastapi import APIRouter, HTTPException

from quantlib_mm import RiskMetrics

from ..sample_data import available_tickers, get_returns
from ..schemas import RiskRequest, RiskResponse

router = APIRouter()


@router.post("/risk/metrics", response_model=RiskResponse)
def metrics(req: RiskRequest) -> RiskResponse:
    if req.ticker not in available_tickers():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ticker {req.ticker!r}. Available: {available_tickers()}",
        )

    returns = get_returns(req.ticker)
    rm = RiskMetrics(returns)

    # Histogram for the frontend
    counts, edges = np.histogram(returns, bins=40)
    histogram = {
        "counts": counts.tolist(),
        "edges": edges.tolist(),
    }

    wealth = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth / running_max - 1.0).tolist()

    return RiskResponse(
        var_historical=rm.var_historical(req.confidence),
        var_parametric=rm.var_parametric(req.confidence),
        cvar=rm.cvar(req.confidence),
        max_drawdown=rm.max_drawdown(),
        sortino=rm.sortino_ratio(req.risk_free_rate),
        calmar=rm.calmar_ratio(req.risk_free_rate),
        histogram=histogram,
        drawdown_series=drawdown,
        cumulative_wealth=wealth.tolist(),
    )
