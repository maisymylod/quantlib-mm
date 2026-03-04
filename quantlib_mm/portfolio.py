"""
Mean-variance portfolio optimization (Markowitz framework).

Uses numpy and scipy.optimize.minimize to compute minimum variance portfolios,
maximum Sharpe ratio portfolios, and efficient frontiers.
"""

import numpy as np
from scipy.optimize import minimize


class Portfolio:
    """Markowitz mean-variance portfolio optimizer.

    Parameters
    ----------
    expected_returns : array-like
        Expected returns for each asset (length n).
    cov_matrix : array-like
        Covariance matrix of asset returns (n x n).
    """

    def __init__(self, expected_returns, cov_matrix):
        self.expected_returns = np.asarray(expected_returns, dtype=float)
        self.cov_matrix = np.asarray(cov_matrix, dtype=float)
        self.n_assets = len(self.expected_returns)

        if self.cov_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                f"Covariance matrix shape {self.cov_matrix.shape} is inconsistent "
                f"with {self.n_assets} assets."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _portfolio_return(self, weights):
        """Expected portfolio return."""
        return float(weights @ self.expected_returns)

    def _portfolio_volatility(self, weights):
        """Portfolio standard deviation (volatility)."""
        return float(np.sqrt(weights @ self.cov_matrix @ weights))

    def _neg_sharpe(self, weights, risk_free_rate):
        """Negative Sharpe ratio (for minimization)."""
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        if vol == 0:
            return 0.0
        return -(ret - risk_free_rate) / vol

    @property
    def _weight_bounds(self):
        """Bounds: each weight in [0, 1] (long-only)."""
        return tuple((0.0, 1.0) for _ in range(self.n_assets))

    @property
    def _sum_to_one_constraint(self):
        """Equality constraint: weights sum to 1."""
        return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def _initial_weights(self):
        """Equal-weight starting point."""
        return np.full(self.n_assets, 1.0 / self.n_assets)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def min_variance_portfolio(self):
        """Compute the minimum variance portfolio.

        Returns
        -------
        weights : np.ndarray
            Optimal asset weights (sum to 1, long-only).
        """
        result = minimize(
            self._portfolio_volatility,
            self._initial_weights(),
            method="SLSQP",
            bounds=self._weight_bounds,
            constraints=self._sum_to_one_constraint,
        )
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        return result.x

    def max_sharpe_portfolio(self, risk_free_rate=0.0):
        """Compute the maximum Sharpe ratio portfolio.

        Parameters
        ----------
        risk_free_rate : float
            The risk-free rate used in the Sharpe ratio calculation.

        Returns
        -------
        weights : np.ndarray
            Optimal asset weights (sum to 1, long-only).
        """
        result = minimize(
            self._neg_sharpe,
            self._initial_weights(),
            args=(risk_free_rate,),
            method="SLSQP",
            bounds=self._weight_bounds,
            constraints=self._sum_to_one_constraint,
        )
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        return result.x

    def efficient_frontier(self, n_points=50):
        """Compute the efficient frontier.

        Traces the set of optimal portfolios from the minimum-variance
        portfolio return up to the maximum single-asset return.

        Parameters
        ----------
        n_points : int
            Number of points along the frontier.

        Returns
        -------
        risks : np.ndarray
            Portfolio volatilities along the frontier.
        returns : np.ndarray
            Portfolio expected returns along the frontier.
        """
        # Determine the return range
        min_var_weights = self.min_variance_portfolio()
        min_ret = self._portfolio_return(min_var_weights)
        max_ret = float(np.max(self.expected_returns))

        target_returns = np.linspace(min_ret, max_ret, n_points)
        risks = np.empty(n_points)
        returns = np.empty(n_points)

        for i, target in enumerate(target_returns):
            constraints = [
                self._sum_to_one_constraint,
                {
                    "type": "eq",
                    "fun": lambda w, t=target: self._portfolio_return(w) - t,
                },
            ]
            result = minimize(
                self._portfolio_volatility,
                self._initial_weights(),
                method="SLSQP",
                bounds=self._weight_bounds,
                constraints=constraints,
            )
            risks[i] = self._portfolio_volatility(result.x)
            returns[i] = self._portfolio_return(result.x)

        return risks, returns

    def portfolio_performance(self, weights):
        """Evaluate portfolio performance for given weights.

        Parameters
        ----------
        weights : array-like
            Asset weights.

        Returns
        -------
        expected_return : float
        volatility : float
        sharpe_ratio : float
            Sharpe ratio assuming risk-free rate of 0.
        """
        weights = np.asarray(weights, dtype=float)
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        sharpe = ret / vol if vol > 0 else 0.0
        return ret, vol, sharpe
