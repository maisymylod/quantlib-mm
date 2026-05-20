"""Portfolio optimization endpoint: efficient frontier."""

import numpy as np
from fastapi import APIRouter, HTTPException

from quantlib_mm import Portfolio

from ..sample_data import get_returns_matrix, available_tickers
from ..schemas import (
    EfficientFrontierRequest,
    EfficientFrontierResponse,
    FrontierPoint,
)

router = APIRouter()


@router.post("/portfolio/efficient-frontier", response_model=EfficientFrontierResponse)
def efficient_frontier(req: EfficientFrontierRequest) -> EfficientFrontierResponse:
    tickers = req.tickers
    valid = available_tickers()
    bad = [t for t in tickers if t not in valid]
    if bad:
        raise HTTPException(status_code=400, detail=f"Unknown tickers: {bad}. Available: {valid}")

    returns = get_returns_matrix(tickers)  # (T, n)
    # Annualise: daily log returns
    expected_returns = returns.mean(axis=0) * 252
    cov_matrix = np.cov(returns, rowvar=False, ddof=1) * 252

    port = Portfolio(expected_returns=expected_returns, cov_matrix=cov_matrix)
    risks, rets = port.efficient_frontier(n_points=req.n_points)
    frontier = [FrontierPoint(risk=float(r), ret=float(e)) for r, e in zip(risks, rets)]

    mv_w = port.min_variance_portfolio()
    ms_w = port.max_sharpe_portfolio(risk_free_rate=req.risk_free_rate)

    def _summary(weights):
        ret, vol, sharpe = port.portfolio_performance(weights)
        return {
            "weights": {t: float(w) for t, w in zip(tickers, weights)},
            "return": float(ret),
            "volatility": float(vol),
            "sharpe": float(sharpe),
        }

    return EfficientFrontierResponse(
        frontier=frontier,
        min_variance=_summary(mv_w),
        max_sharpe=_summary(ms_w),
        expected_returns={t: float(r) for t, r in zip(tickers, expected_returns)},
        annualised_vols={
            t: float(np.sqrt(cov_matrix[i, i])) for i, t in enumerate(tickers)
        },
    )
