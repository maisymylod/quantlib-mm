"""Mathematical utilities for quantitative finance."""

import numpy as np
from scipy.stats import norm


def standard_normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return norm.cdf(x)


def standard_normal_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return norm.pdf(x)


def annualize_returns(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualize a series of periodic returns."""
    total_return = np.prod(1 + returns) - 1
    n_periods = len(returns)
    return (1 + total_return) ** (periods_per_year / n_periods) - 1


def annualize_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualize volatility from periodic returns."""
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)
