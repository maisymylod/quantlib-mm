"""
Analytical Black-Scholes Greeks for European options.

Implements first- and second-order sensitivities using the closed-form
Black-Scholes formulas with standard normal PDF/CDF via scipy.
"""

import numpy as np
from scipy.stats import norm


class Greeks:
    """Compute Black-Scholes Greeks for a European option.

    Parameters
    ----------
    S : float
        Current spot price of the underlying.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualised, continuous compounding).
    sigma : float
        Volatility of the underlying (annualised).
    option_type : str
        ``'call'`` or ``'put'``.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> None:
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.option_type = option_type

        # Pre-compute d1 and d2
        self._d1 = (
            np.log(self.S / self.K)
            + (self.r + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))
        self._d2 = self._d1 - self.sigma * np.sqrt(self.T)

    # ------------------------------------------------------------------
    # First-order Greeks
    # ------------------------------------------------------------------

    def delta(self) -> float:
        """dV/dS — rate of change of option value with respect to spot."""
        if self.option_type == "call":
            return float(norm.cdf(self._d1))
        return float(norm.cdf(self._d1) - 1.0)

    def gamma(self) -> float:
        """d²V/dS² — second derivative of option value w.r.t. spot.

        Gamma is identical for calls and puts.
        """
        return float(
            norm.pdf(self._d1) / (self.S * self.sigma * np.sqrt(self.T))
        )

    def theta(self) -> float:
        """dV/dt — time decay expressed **per calendar day**.

        The annualised theta is divided by 365 so the result represents
        the expected daily change in option value, all else being equal.
        """
        S, K, T, r, sigma = self.S, self.K, self.T, self.r, self.sigma
        d1, d2 = self._d1, self._d2

        first_term = -(S * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))

        if self.option_type == "call":
            annual_theta = first_term - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            annual_theta = first_term + r * K * np.exp(-r * T) * norm.cdf(-d2)

        return float(annual_theta / 365.0)

    def vega(self) -> float:
        """dV/dσ — sensitivity to volatility per 1 percentage-point move.

        The raw vega (dV per unit change in σ) is divided by 100 so the
        result represents the value change for a 0.01 (1 %) increase in
        implied volatility.  Vega is identical for calls and puts.
        """
        raw_vega = self.S * norm.pdf(self._d1) * np.sqrt(self.T)
        return float(raw_vega / 100.0)

    def rho(self) -> float:
        """dV/dr — sensitivity to interest rate per 1 percentage-point move.

        Divided by 100 so the result represents the value change for a
        0.01 (1 %) increase in the risk-free rate.
        """
        K, T, r = self.K, self.T, self.r
        d2 = self._d2

        if self.option_type == "call":
            raw_rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            raw_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        return float(raw_rho / 100.0)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dict with all five Greeks."""
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "theta": self.theta(),
            "vega": self.vega(),
            "rho": self.rho(),
        }
