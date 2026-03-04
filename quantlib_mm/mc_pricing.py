"""Monte Carlo option pricing engine.

Provides risk-neutral Monte Carlo simulation for European, Asian, and barrier
options using Geometric Brownian Motion (GBM) path generation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


class MonteCarloOptionPricer:
    """Monte Carlo pricer for vanilla and exotic options.

    Parameters
    ----------
    S : float
        Current underlying spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate (annualised, continuous compounding).
    sigma : float
        Volatility of the underlying (annualised).
    n_paths : int, optional
        Number of simulated paths (default 100 000).
    n_steps : int, optional
        Number of time steps per path (default 252, ~trading days).
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_paths: int = 100_000,
        n_steps: int = 252,
    ) -> None:
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps

        # Populated after each pricing call
        self._last_price: float | None = None
        self._last_std_error: float | None = None

    # ------------------------------------------------------------------
    # Path generation
    # ------------------------------------------------------------------

    def _simulate_paths(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Simulate GBM paths under the risk-neutral measure.

        Returns
        -------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Simulated price paths including the initial price at index 0.
        """
        if rng is None:
            rng = np.random.default_rng()

        dt = self.T / self.n_steps
        nudt = (self.r - 0.5 * self.sigma ** 2) * dt
        sigdt = self.sigma * np.sqrt(dt)

        # Generate standard‐normal increments
        Z = rng.standard_normal((self.n_paths, self.n_steps))

        # Log‐returns then cumulative sum to get log‐prices
        log_increments = nudt + sigdt * Z
        log_paths = np.zeros((self.n_paths, self.n_steps + 1))
        log_paths[:, 0] = np.log(self.S)
        log_paths[:, 1:] = np.cumsum(log_increments, axis=1) + np.log(self.S)

        return np.exp(log_paths)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_results(self, payoffs: np.ndarray) -> float:
        """Discount mean payoff, compute standard error, and store results."""
        discount = np.exp(-self.r * self.T)
        discounted = discount * payoffs
        price = float(np.mean(discounted))
        std_error = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))

        self._last_price = price
        self._last_std_error = std_error
        return price

    # ------------------------------------------------------------------
    # Pricing methods
    # ------------------------------------------------------------------

    def price_european_call(self, rng: np.random.Generator | None = None) -> float:
        """Price a European call option via Monte Carlo.

        Returns
        -------
        float
            Discounted expected payoff max(S_T - K, 0).
        """
        paths = self._simulate_paths(rng)
        ST = paths[:, -1]
        payoffs = np.maximum(ST - self.K, 0.0)
        return self._store_results(payoffs)

    def price_european_put(self, rng: np.random.Generator | None = None) -> float:
        """Price a European put option via Monte Carlo.

        Returns
        -------
        float
            Discounted expected payoff max(K - S_T, 0).
        """
        paths = self._simulate_paths(rng)
        ST = paths[:, -1]
        payoffs = np.maximum(self.K - ST, 0.0)
        return self._store_results(payoffs)

    def price_asian_call(
        self,
        averaging: str = "arithmetic",
        rng: np.random.Generator | None = None,
    ) -> float:
        """Price an Asian call option.

        Parameters
        ----------
        averaging : str
            ``'arithmetic'`` or ``'geometric'``.

        Returns
        -------
        float
            Discounted expected payoff max(A - K, 0) where A is the
            path‐average price (excluding the initial observation).
        """
        if averaging not in ("arithmetic", "geometric"):
            raise ValueError(
                f"averaging must be 'arithmetic' or 'geometric', got {averaging!r}"
            )

        paths = self._simulate_paths(rng)
        # Average over monitoring dates (exclude t=0 observation)
        if averaging == "arithmetic":
            A = np.mean(paths[:, 1:], axis=1)
        else:
            A = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))

        payoffs = np.maximum(A - self.K, 0.0)
        return self._store_results(payoffs)

    def price_barrier_call(
        self,
        barrier: float,
        barrier_type: str = "up-and-out",
        rng: np.random.Generator | None = None,
    ) -> float:
        """Price a barrier call option.

        Parameters
        ----------
        barrier : float
            Barrier level.
        barrier_type : str
            One of ``'up-and-out'``, ``'up-and-in'``,
            ``'down-and-out'``, ``'down-and-in'``.

        Returns
        -------
        float
            Discounted expected payoff of the barrier option.
        """
        valid_types = ("up-and-out", "up-and-in", "down-and-out", "down-and-in")
        if barrier_type not in valid_types:
            raise ValueError(
                f"barrier_type must be one of {valid_types}, got {barrier_type!r}"
            )

        paths = self._simulate_paths(rng)
        ST = paths[:, -1]
        vanilla_payoffs = np.maximum(ST - self.K, 0.0)

        if barrier_type.startswith("up"):
            hit = np.any(paths >= barrier, axis=1)
        else:
            hit = np.any(paths <= barrier, axis=1)

        if barrier_type.endswith("out"):
            payoffs = np.where(hit, 0.0, vanilla_payoffs)
        else:  # "in"
            payoffs = np.where(hit, vanilla_payoffs, 0.0)

        return self._store_results(payoffs)

    # ------------------------------------------------------------------
    # Confidence interval
    # ------------------------------------------------------------------

    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Return a confidence interval for the last computed price.

        Parameters
        ----------
        confidence : float
            Confidence level (e.g. 0.95 for 95 %).

        Returns
        -------
        (lower, upper) : tuple[float, float]
        """
        if self._last_price is None or self._last_std_error is None:
            raise RuntimeError(
                "No price has been computed yet. Call a pricing method first."
            )

        z = sp_stats.norm.ppf(0.5 + confidence / 2.0)
        half_width = z * self._last_std_error
        return (self._last_price - half_width, self._last_price + half_width)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def last_price(self) -> float | None:
        """Most recently computed price."""
        return self._last_price

    @property
    def last_std_error(self) -> float | None:
        """Standard error of the most recently computed price."""
        return self._last_std_error
