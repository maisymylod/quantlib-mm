"""Pydantic request/response models for the quantlib-mm API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


class MonteCarloRequest(BaseModel):
    s0: float = Field(gt=0, default=100.0)
    mu: float = Field(default=0.08)
    sigma: float = Field(ge=0, default=0.2)
    T: float = Field(gt=0, default=1.0)
    n_steps: int = Field(ge=1, le=2520, default=252)
    n_paths: int = Field(ge=1, le=5000, default=200)
    antithetic: bool = False
    seed: Optional[int] = 42


class MonteCarloResponse(BaseModel):
    times: list[float]
    sample_paths: list[list[float]]
    mean_path: list[float]
    p5_path: list[float]
    p95_path: list[float]
    terminal_mean: float
    terminal_std: float
    terminal_min: float
    terminal_max: float
    percentiles: dict[int, float]
    max_drawdown_mean: float


# ---------------------------------------------------------------------------
# Black-Scholes + Greeks
# ---------------------------------------------------------------------------


class BlackScholesRequest(BaseModel):
    S: float = Field(gt=0, default=100.0)
    K: float = Field(gt=0, default=105.0)
    T: float = Field(gt=0, default=1.0)
    r: float = Field(default=0.05)
    sigma: float = Field(gt=0, default=0.2)


class GreeksPayload(BaseModel):
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class BlackScholesResponse(BaseModel):
    call_price: float
    put_price: float
    call_greeks: GreeksPayload
    put_greeks: GreeksPayload
    parity_holds: bool


# ---------------------------------------------------------------------------
# Binomial Tree
# ---------------------------------------------------------------------------


class BinomialRequest(BaseModel):
    S: float = Field(gt=0, default=100.0)
    K: float = Field(gt=0, default=100.0)
    T: float = Field(gt=0, default=1.0)
    r: float = Field(default=0.05)
    sigma: float = Field(gt=0, default=0.2)
    n_steps: int = Field(ge=1, le=20, default=8)
    option_type: Literal["call", "put"] = "call"
    style: Literal["european", "american"] = "european"


class BinomialNode(BaseModel):
    step: int
    index: int
    spot: float
    value: float


class BinomialResponse(BaseModel):
    price: float
    u: float
    d: float
    p: float
    nodes: list[BinomialNode]


# ---------------------------------------------------------------------------
# Monte Carlo option pricing
# ---------------------------------------------------------------------------


class MCPricingRequest(BaseModel):
    S: float = Field(gt=0, default=100.0)
    K: float = Field(gt=0, default=100.0)
    T: float = Field(gt=0, default=1.0)
    r: float = Field(default=0.05)
    sigma: float = Field(gt=0, default=0.2)
    n_paths: int = Field(ge=1000, le=200_000, default=50_000)
    n_steps: int = Field(ge=10, le=1000, default=252)
    option_type: Literal["call", "put", "asian-arithmetic", "asian-geometric"] = "call"


class MCPricingResponse(BaseModel):
    price: float
    std_error: float
    ci_lower: float
    ci_upper: float


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


class EfficientFrontierRequest(BaseModel):
    tickers: list[str] = Field(min_length=2, max_length=8)
    n_points: int = Field(ge=10, le=100, default=40)
    risk_free_rate: float = 0.02


class FrontierPoint(BaseModel):
    risk: float
    ret: float


class TickerWeight(BaseModel):
    ticker: str
    weight: float


class EfficientFrontierResponse(BaseModel):
    frontier: list[FrontierPoint]
    min_variance: dict
    max_sharpe: dict
    expected_returns: dict[str, float]
    annualised_vols: dict[str, float]


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


class RiskRequest(BaseModel):
    ticker: str = "AAPL"
    confidence: float = Field(gt=0, lt=1, default=0.95)
    risk_free_rate: float = 0.02


class RiskResponse(BaseModel):
    var_historical: float
    var_parametric: float
    cvar: float
    max_drawdown: float
    sortino: float
    calmar: float
    histogram: dict
    drawdown_series: list[float]
    cumulative_wealth: list[float]


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


class TimeSeriesRequest(BaseModel):
    ticker: str = "AAPL"
    rolling_window: int = Field(ge=5, le=120, default=21)
    ewma_span: int = Field(ge=5, le=120, default=21)
    acf_lags: int = Field(ge=1, le=40, default=20)


class TimeSeriesResponse(BaseModel):
    log_returns: list[float]
    rolling_vol: list[float]
    ewma_vol: list[float]
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    autocorrelation: list[float]


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


class CorrelationRequest(BaseModel):
    tickers: list[str] = Field(min_length=2, max_length=8)
    shrinkage: float = Field(ge=0.0, le=1.0, default=0.1)


class CorrelationResponse(BaseModel):
    tickers: list[str]
    correlation_matrix: list[list[float]]
    covariance_matrix: list[list[float]]
    shrunk_covariance: list[list[float]]
    explained_variance_ratio: list[float]
    pc1_loadings: list[float]
    pc2_loadings: list[float]


# ---------------------------------------------------------------------------
# Yield curve
# ---------------------------------------------------------------------------


class YieldCurveRequest(BaseModel):
    """Yield curve request.

    The library's bootstrap implementation assumes annual-coupon par bonds
    with consecutive integer maturities (1, 2, 3, ..., N), so the default
    values reflect that contract.
    """

    maturities: list[float] = Field(default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    par_rates: list[float] = Field(
        default=[0.043, 0.044, 0.0445, 0.0455, 0.046, 0.0465, 0.047, 0.0475, 0.048, 0.0485]
    )


class YieldCurveResponse(BaseModel):
    maturities: list[float]
    zero_rates: list[float]
    discount_factors: list[float]
    forward_rates: list[float]
    forward_starts: list[float]
    forward_ends: list[float]


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


class SampleDataResponse(BaseModel):
    ticker: str
    prices: list[float]
    returns: list[float]
