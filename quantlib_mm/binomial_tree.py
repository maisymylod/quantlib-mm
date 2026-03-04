"""
Binomial tree option pricing using the Cox-Ross-Rubinstein (CRR) parameterization.

Supports European and American options (calls and puts) with backward induction
and early exercise handling.
"""

import numpy as np


class BinomialTree:
    """
    Cox-Ross-Rubinstein binomial tree pricer for vanilla options.

    Parameters
    ----------
    S : float
        Current underlying spot price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    sigma : float
        Volatility of the underlying (annualized).
    n_steps : int
        Number of time steps in the tree.
    option_type : str
        'call' or 'put'.
    style : str
        'european' or 'american'.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_steps: int,
        option_type: str = "call",
        style: str = "european",
    ):
        if S <= 0:
            raise ValueError("Spot price S must be positive.")
        if K <= 0:
            raise ValueError("Strike price K must be positive.")
        if T <= 0:
            raise ValueError("Time to expiration T must be positive.")
        if sigma <= 0:
            raise ValueError("Volatility sigma must be positive.")
        if n_steps < 1:
            raise ValueError("Number of steps n_steps must be at least 1.")
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        if style not in ("european", "american"):
            raise ValueError("style must be 'european' or 'american'.")

        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.n_steps = int(n_steps)
        self.option_type = option_type
        self.style = style

        # CRR parameters
        self.dt = self.T / self.n_steps
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)

        # Cache
        self._price_tree = None

    def _payoff(self, spot: np.ndarray) -> np.ndarray:
        """Compute the intrinsic option payoff for an array of spot prices."""
        if self.option_type == "call":
            return np.maximum(spot - self.K, 0.0)
        else:
            return np.maximum(self.K - spot, 0.0)

    def _build_tree(self) -> list[np.ndarray]:
        """
        Build the full option-value tree via backward induction.

        Returns a list of length (n_steps + 1), where element i is a numpy
        array of size (i + 1) holding option values at time step i.
        """
        n = self.n_steps
        disc = np.exp(-self.r * self.dt)

        # Terminal asset prices at step n: S * u^(n-2j) for j = 0..n
        # Node j at step i has asset price S * u^(i - 2*j)
        terminal_spots = self.S * self.u ** (n - 2 * np.arange(n + 1, dtype=float))
        terminal_values = self._payoff(terminal_spots)

        # We store every layer for get_tree()
        tree = [None] * (n + 1)
        tree[n] = terminal_values.copy()

        # Backward induction
        values = terminal_values
        for i in range(n - 1, -1, -1):
            # Discounted expected value under risk-neutral measure
            values = disc * (self.p * values[:-1] + (1.0 - self.p) * values[1:])

            # Early exercise for American options
            if self.style == "american":
                spots_i = self.S * self.u ** (i - 2 * np.arange(i + 1, dtype=float))
                intrinsic = self._payoff(spots_i)
                values = np.maximum(values, intrinsic)

            tree[i] = values.copy()

        self._price_tree = tree
        return tree

    def price(self) -> float:
        """
        Compute the option price.

        Returns
        -------
        float
            The fair value of the option at time 0.
        """
        tree = self._build_tree()
        return float(tree[0][0])

    def get_tree(self) -> list[np.ndarray]:
        """
        Return the full option-value tree.

        Each element ``tree[i]`` is a numpy array of size ``(i + 1)`` holding
        the option values at time step *i*  (i = 0 … n_steps).

        Returns
        -------
        list[np.ndarray]
        """
        if self._price_tree is None:
            self._build_tree()
        return self._price_tree
