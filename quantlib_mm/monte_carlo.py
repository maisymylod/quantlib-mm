"""
Monte Carlo simulation engine for quantitative finance.

Implements Geometric Brownian Motion (GBM) path generation using the
stochastic differential equation:

    dS = μ S dt + σ S dW

where:
    S  = asset price
    μ  = drift (annualized expected return)
    σ  = volatility (annualized)
    dW = Wiener process increment ~ N(0, dt)

Supports antithetic variates for variance reduction and provides
comprehensive path-level statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PathStatistics:
    """Summary statistics computed across a set of simulated price paths.

    All terminal-value statistics are computed from the final prices (last
    column) of the path matrix.  Path-level statistics (e.g. max drawdown)
    are computed across the full trajectory.

    Attributes
    ----------
    mean : float
        Mean of terminal prices.
    std : float
        Standard deviation of terminal prices.
    percentiles : dict[int, float]
        Mapping from percentile rank to terminal price value.
    min : float
        Minimum terminal price observed.
    max : float
        Maximum terminal price observed.
    mean_path : np.ndarray
        Element-wise mean across all paths (shape: ``(n_steps + 1,)``).
    max_drawdown_mean : float
        Mean of per-path maximum drawdown fractions.
    """

    mean: float
    std: float
    percentiles: dict[int, float]
    min: float
    max: float
    mean_path: np.ndarray
    max_drawdown_mean: float


class GeometricBrownianMotion:
    """Monte Carlo simulator for Geometric Brownian Motion.

    Parameters
    ----------
    s0 : float
        Initial asset price (must be positive).
    mu : float
        Annualized drift / expected return.
    sigma : float
        Annualized volatility (must be non-negative).
    T : float
        Time horizon in years (must be positive).
    n_steps : int
        Number of discrete time steps per path (must be >= 1).
    n_paths : int
        Number of Monte Carlo paths to simulate (must be >= 1).
    antithetic : bool, optional
        If ``True``, use antithetic variates variance reduction.  The
        effective number of paths will remain ``n_paths``; half are
        generated from standard normal draws and the other half from
        their negation.  ``n_paths`` must be even when this is enabled.
    seed : int or None, optional
        Random seed for reproducibility.

    Raises
    ------
    ValueError
        If any parameter constraint is violated.
    """

    def __init__(
        self,
        s0: float,
        mu: float,
        sigma: float,
        T: float,
        n_steps: int,
        n_paths: int,
        antithetic: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if s0 <= 0:
            raise ValueError(f"Initial price s0 must be positive, got {s0}")
        if sigma < 0:
            raise ValueError(f"Volatility sigma must be non-negative, got {sigma}")
        if T <= 0:
            raise ValueError(f"Time horizon T must be positive, got {T}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")
        if antithetic and n_paths % 2 != 0:
            raise ValueError(
                f"n_paths must be even when antithetic=True, got {n_paths}"
            )

        self.s0 = float(s0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.T = float(T)
        self.n_steps = int(n_steps)
        self.n_paths = int(n_paths)
        self.antithetic = antithetic
        self.seed = seed

        self.dt = self.T / self.n_steps
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def simulate(self) -> np.ndarray:
        """Generate Monte Carlo price paths via the exact GBM solution.

        Uses the log-normal exact discretisation:

            S(t+dt) = S(t) * exp((μ - σ²/2) dt + σ √dt Z)

        where Z ~ N(0, 1).  This is exact (no discretisation error) for GBM.

        Returns
        -------
        paths : np.ndarray, shape ``(n_paths, n_steps + 1)``
            Simulated price paths.  Column 0 is the initial price ``s0``
            replicated across all paths.
        """
        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)

        if self.antithetic:
            half = self.n_paths // 2
            z_half = self._rng.standard_normal((half, self.n_steps))
            z = np.concatenate([z_half, -z_half], axis=0)
        else:
            z = self._rng.standard_normal((self.n_paths, self.n_steps))

        # Log-increments and cumulative sum to build log-price paths
        log_increments = drift + diffusion * z  # (n_paths, n_steps)
        log_paths = np.cumsum(log_increments, axis=1)

        # Prepend zero so that column 0 corresponds to log(S0/S0) = 0
        log_paths = np.concatenate(
            [np.zeros((self.n_paths, 1)), log_paths], axis=1
        )

        paths = self.s0 * np.exp(log_paths)
        return paths

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_statistics(
        self,
        paths: np.ndarray,
        percentile_ranks: Optional[list[int]] = None,
    ) -> PathStatistics:
        """Compute summary statistics from a set of simulated price paths.

        Parameters
        ----------
        paths : np.ndarray, shape ``(n_paths, n_steps + 1)``
            Path matrix as returned by :meth:`simulate`.
        percentile_ranks : list of int, optional
            Percentile ranks to compute.  Defaults to
            ``[5, 25, 50, 75, 95]``.

        Returns
        -------
        PathStatistics
            Frozen dataclass of summary statistics.
        """
        if percentile_ranks is None:
            percentile_ranks = [5, 25, 50, 75, 95]

        terminal = paths[:, -1]

        # Per-path maximum drawdown: max peak-to-trough decline as fraction
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (running_max - paths) / running_max
        max_drawdowns = np.max(drawdowns, axis=1)

        percentiles = {
            int(p): float(np.percentile(terminal, p)) for p in percentile_ranks
        }

        return PathStatistics(
            mean=float(np.mean(terminal)),
            std=float(np.std(terminal, ddof=1)),
            percentiles=percentiles,
            min=float(np.min(terminal)),
            max=float(np.max(terminal)),
            mean_path=np.mean(paths, axis=0),
            max_drawdown_mean=float(np.mean(max_drawdowns)),
        )
