"""Risk metrics for portfolio and return-series analysis.

Provides historical VaR, parametric (Gaussian) VaR, Conditional VaR
(Expected Shortfall), maximum drawdown, Sortino ratio, and Calmar ratio.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class RiskMetrics:
    """Compute common risk metrics from an array of periodic returns.

    Parameters
    ----------
    returns : array_like
        1-D array of simple (arithmetic) returns, e.g. daily log-returns or
        simple percentage returns expressed as decimals.
    """

    def __init__(self, returns: np.ndarray | list[float]) -> None:
        self.returns = np.asarray(returns, dtype=np.float64)
        if self.returns.ndim != 1 or len(self.returns) == 0:
            raise ValueError("returns must be a non-empty 1-D array")

    # ------------------------------------------------------------------
    # Value at Risk
    # ------------------------------------------------------------------

    def var_historical(self, confidence: float = 0.95) -> float:
        """Historical Value at Risk.

        Returns the portfolio loss (as a negative number) at the given
        confidence level using the empirical quantile of the return series.
        """
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        return float(np.percentile(self.returns, (1 - confidence) * 100))

    def var_parametric(self, confidence: float = 0.95) -> float:
        """Parametric (Gaussian) Value at Risk.

        Assumes returns are normally distributed and computes VaR from the
        fitted mean and standard deviation.
        """
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        mu = np.mean(self.returns)
        sigma = np.std(self.returns, ddof=1)
        return float(stats.norm.ppf(1 - confidence, loc=mu, scale=sigma))

    # ------------------------------------------------------------------
    # Conditional VaR (Expected Shortfall)
    # ------------------------------------------------------------------

    def cvar(self, confidence: float = 0.95) -> float:
        """Conditional Value at Risk (Expected Shortfall).

        Mean of all returns that fall at or below the historical VaR
        threshold.
        """
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        var = self.var_historical(confidence)
        tail = self.returns[self.returns <= var]
        if len(tail) == 0:
            return var  # edge-case: no observations in the tail
        return float(np.mean(tail))

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def max_drawdown(self) -> float:
        """Maximum drawdown from peak.

        Returns the largest peak-to-trough decline (as a negative number or
        zero) computed on the cumulative wealth index (1 + r_1)(1 + r_2)...
        """
        wealth = np.cumprod(1 + self.returns)
        running_max = np.maximum.accumulate(wealth)
        drawdowns = wealth / running_max - 1.0
        return float(np.min(drawdowns))

    # ------------------------------------------------------------------
    # Risk-adjusted return ratios
    # ------------------------------------------------------------------

    def sortino_ratio(
        self, risk_free_rate: float = 0.0, periods: int = 252
    ) -> float:
        """Sortino ratio — excess return per unit of downside deviation.

        Parameters
        ----------
        risk_free_rate : float
            Annualised risk-free rate (decimal).
        periods : int
            Number of return observations per year (252 for daily).
        """
        rf_per_period = risk_free_rate / periods
        excess = self.returns - rf_per_period
        downside = excess[excess < 0]
        if len(downside) == 0:
            return np.inf
        downside_std = np.sqrt(np.mean(downside ** 2))
        annualised_excess = np.mean(excess) * periods
        annualised_downside_std = downside_std * np.sqrt(periods)
        return float(annualised_excess / annualised_downside_std)

    def calmar_ratio(
        self, risk_free_rate: float = 0.0, periods: int = 252
    ) -> float:
        """Calmar ratio — annualised return divided by maximum drawdown.

        Parameters
        ----------
        risk_free_rate : float
            Annualised risk-free rate (decimal).
        periods : int
            Number of return observations per year (252 for daily).
        """
        rf_per_period = risk_free_rate / periods
        annualised_return = np.mean(self.returns - rf_per_period) * periods
        mdd = self.max_drawdown()
        if mdd == 0:
            return np.inf
        return float(annualised_return / abs(mdd))
