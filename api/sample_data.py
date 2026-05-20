"""Generate deterministic synthetic price + return series for showcase tickers.

Uses the library's own GeometricBrownianMotion so the demo data is internally
consistent with the rest of the showcase.
"""

from __future__ import annotations

import numpy as np

from quantlib_mm import GeometricBrownianMotion

# (ticker, mu, sigma, seed) -- one fixed seed per ticker so the series is stable
TICKER_PARAMS: dict[str, tuple[float, float, int]] = {
    "AAPL": (0.18, 0.27, 11),
    "MSFT": (0.16, 0.24, 22),
    "GOOG": (0.14, 0.26, 33),
    "AMZN": (0.20, 0.32, 44),
    "TSLA": (0.25, 0.55, 55),
    "NVDA": (0.32, 0.42, 66),
}

# Two years of trading days
N_STEPS = 504
T = 2.0
S0 = 100.0


def _generate_prices(ticker: str) -> np.ndarray:
    if ticker not in TICKER_PARAMS:
        raise KeyError(ticker)
    mu, sigma, seed = TICKER_PARAMS[ticker]
    gbm = GeometricBrownianMotion(
        s0=S0, mu=mu, sigma=sigma, T=T, n_steps=N_STEPS, n_paths=1, seed=seed
    )
    paths = gbm.simulate()
    return paths[0]  # shape: (N_STEPS + 1,)


def get_prices(ticker: str) -> np.ndarray:
    """Return synthetic prices for *ticker* (length N_STEPS + 1)."""
    return _generate_prices(ticker)


def get_returns(ticker: str) -> np.ndarray:
    """Return synthetic log returns for *ticker* (length N_STEPS)."""
    prices = get_prices(ticker)
    return np.log(prices[1:] / prices[:-1])


def get_returns_matrix(tickers: list[str]) -> np.ndarray:
    """Return a (N_STEPS, n_tickers) returns matrix."""
    cols = [get_returns(t) for t in tickers]
    return np.column_stack(cols)


def available_tickers() -> list[str]:
    return list(TICKER_PARAMS.keys())
