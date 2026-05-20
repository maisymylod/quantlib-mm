"""Correlation analysis endpoint."""

from fastapi import APIRouter, HTTPException

from quantlib_mm import CorrelationAnalyzer

from ..sample_data import available_tickers, get_returns_matrix
from ..schemas import CorrelationRequest, CorrelationResponse

router = APIRouter()


@router.post("/correlation/analyze", response_model=CorrelationResponse)
def analyze(req: CorrelationRequest) -> CorrelationResponse:
    tickers = req.tickers
    valid = available_tickers()
    bad = [t for t in tickers if t not in valid]
    if bad:
        raise HTTPException(status_code=400, detail=f"Unknown tickers: {bad}. Available: {valid}")

    returns = get_returns_matrix(tickers)
    ca = CorrelationAnalyzer(returns)

    eigvals, eigvecs, evr = ca.pca_decomposition()

    return CorrelationResponse(
        tickers=tickers,
        correlation_matrix=ca.correlation_matrix().tolist(),
        covariance_matrix=ca.covariance_matrix().tolist(),
        shrunk_covariance=ca.shrink_covariance(req.shrinkage).tolist(),
        explained_variance_ratio=evr.tolist(),
        pc1_loadings=eigvecs[:, 0].tolist(),
        pc2_loadings=eigvecs[:, 1].tolist(),
    )
