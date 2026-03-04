"""
Black-Scholes option pricing model for European options.

Implements closed-form pricing, put-call parity verification,
and implied volatility calculation via Newton-Raphson iteration.
"""

import numpy as np
from scipy.stats import norm


class BlackScholes:
    """European option pricing using the Black-Scholes closed-form solution.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years (must be > 0).
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Annualised volatility of the underlying asset (must be > 0).
    """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        if T <= 0:
            raise ValueError("Time to expiry T must be positive.")
        if sigma <= 0:
            raise ValueError("Volatility sigma must be positive.")
        if S <= 0:
            raise ValueError("Spot price S must be positive.")
        if K <= 0:
            raise ValueError("Strike price K must be positive.")

        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _d1(self) -> float:
        """Compute d1 of the Black-Scholes formula."""
        return (
            np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))

    def _d2(self) -> float:
        """Compute d2 = d1 - sigma * sqrt(T)."""
        return self._d1() - self.sigma * np.sqrt(self.T)

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def call_price(self) -> float:
        """Return the Black-Scholes price of a European call option.

        C = S * N(d1) - K * exp(-r*T) * N(d2)
        """
        d1 = self._d1()
        d2 = self._d2()
        return float(
            self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        )

    def put_price(self) -> float:
        """Return the Black-Scholes price of a European put option.

        P = K * exp(-r*T) * N(-d2) - S * N(-d1)
        """
        d1 = self._d1()
        d2 = self._d2()
        return float(
            self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        )

    # ------------------------------------------------------------------
    # Put-call parity
    # ------------------------------------------------------------------

    def put_call_parity(self, atol: float = 1e-8) -> bool:
        """Verify put-call parity: C - P == S - K * exp(-r*T).

        Parameters
        ----------
        atol : float
            Absolute tolerance for the parity check.

        Returns
        -------
        bool
            True if parity holds within the specified tolerance.
        """
        lhs = self.call_price() - self.put_price()
        rhs = self.S - self.K * np.exp(-self.r * self.T)
        return bool(np.isclose(lhs, rhs, atol=atol))

    # ------------------------------------------------------------------
    # Implied volatility
    # ------------------------------------------------------------------

    @staticmethod
    def implied_volatility(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        tol: float = 1e-8,
        max_iter: int = 200,
        initial_guess: float = 0.2,
    ) -> float:
        """Compute implied volatility using Newton-Raphson iteration.

        The method solves for sigma such that BS(sigma) == market_price by
        iterating:  sigma_{n+1} = sigma_n - (BS(sigma_n) - market_price) / vega

        Parameters
        ----------
        market_price : float
            Observed market price of the option.
        S : float
            Spot price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Risk-free rate.
        option_type : str
            ``"call"`` or ``"put"``.
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum number of Newton-Raphson iterations.
        initial_guess : float
            Starting value for sigma.

        Returns
        -------
        float
            The implied volatility.

        Raises
        ------
        ValueError
            If the algorithm does not converge.
        """
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        sigma = initial_guess

        for i in range(max_iter):
            bs = BlackScholes(S, K, T, r, sigma)

            if option_type == "call":
                price = bs.call_price()
            else:
                price = bs.put_price()

            diff = price - market_price

            # Vega: dC/d(sigma) = S * sqrt(T) * phi(d1)
            # Vega is the same for calls and puts.
            d1 = bs._d1()
            vega = S * np.sqrt(T) * norm.pdf(d1)

            if vega < 1e-12:
                raise ValueError(
                    f"Vega near zero at iteration {i}; Newton-Raphson cannot proceed."
                )

            sigma = sigma - diff / vega

            if sigma <= 0:
                # Reset to a small positive value to stay in the valid domain.
                sigma = 1e-4

            if abs(diff) < tol:
                return float(sigma)

        raise ValueError(
            f"Implied volatility did not converge after {max_iter} iterations "
            f"(last sigma={sigma:.6f}, residual={diff:.2e})."
        )
