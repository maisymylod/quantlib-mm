"""Yield curve construction, interpolation, and analytics."""

import numpy as np
from scipy.interpolate import interp1d


class YieldCurve:
    """A yield curve built from discrete maturity/rate pairs.

    Parameters
    ----------
    maturities : array-like
        Maturities in years (must be positive and strictly increasing).
    rates : array-like
        Continuously-compounded zero rates corresponding to each maturity.
    """

    def __init__(self, maturities, rates):
        self.maturities = np.asarray(maturities, dtype=float)
        self.rates = np.asarray(rates, dtype=float)

        if self.maturities.shape != self.rates.shape:
            raise ValueError("maturities and rates must have the same length")
        if len(self.maturities) < 2:
            raise ValueError("at least two maturity/rate pairs are required")

        self._interp = interp1d(
            self.maturities,
            self.rates,
            kind="linear",
            fill_value="extrapolate",
        )

    # ------------------------------------------------------------------
    # Core analytics
    # ------------------------------------------------------------------

    def interpolate(self, t):
        """Return the linearly-interpolated zero rate at maturity *t*.

        Parameters
        ----------
        t : float or array-like
            Target maturity (years).

        Returns
        -------
        float or ndarray
            Interpolated continuously-compounded zero rate.
        """
        return float(self._interp(t)) if np.isscalar(t) else np.asarray(self._interp(t))

    def discount_factor(self, t):
        """Return the discount factor for maturity *t*.

        Uses the continuously-compounded zero rate:  DF(t) = exp(-r(t) * t).

        Parameters
        ----------
        t : float
            Maturity in years.

        Returns
        -------
        float
            Discount factor.
        """
        r = self.interpolate(t)
        return float(np.exp(-r * t))

    def forward_rate(self, t1, t2):
        """Return the implied forward rate between *t1* and *t2*.

        The forward rate f(t1, t2) satisfies:

            DF(t2) = DF(t1) * exp(-f * (t2 - t1))

        Parameters
        ----------
        t1 : float
            Start maturity (years).
        t2 : float
            End maturity (years).  Must be > t1.

        Returns
        -------
        float
            Continuously-compounded forward rate.
        """
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")

        r1 = self.interpolate(t1)
        r2 = self.interpolate(t2)
        return float((r2 * t2 - r1 * t1) / (t2 - t1))

    # ------------------------------------------------------------------
    # Curve transformations
    # ------------------------------------------------------------------

    def shift(self, basis_points):
        """Return a new :class:`YieldCurve` shifted by *basis_points*.

        Parameters
        ----------
        basis_points : float
            Parallel shift in basis points (1 bp = 0.0001).

        Returns
        -------
        YieldCurve
            New curve with every rate shifted.
        """
        delta = basis_points / 10_000.0
        return YieldCurve(self.maturities.copy(), self.rates + delta)

    # ------------------------------------------------------------------
    # Bootstrapping
    # ------------------------------------------------------------------

    @classmethod
    def bootstrap(cls, par_rates, maturities):
        """Bootstrap zero-coupon rates from par bond yields.

        Assumes annual coupon bonds with face value 1, where each par
        rate is the coupon rate that makes the bond price equal to par.

        Parameters
        ----------
        par_rates : array-like
            Par coupon rates (as decimals, e.g. 0.05 for 5 %).
        maturities : array-like
            Maturities in whole years corresponding to each par rate.
            Must be sorted and start at 1 with increments of 1.

        Returns
        -------
        YieldCurve
            Curve constructed from the bootstrapped zero rates.
        """
        par_rates = np.asarray(par_rates, dtype=float)
        maturities = np.asarray(maturities, dtype=float)

        if len(par_rates) != len(maturities):
            raise ValueError("par_rates and maturities must have the same length")

        zero_rates = np.empty_like(par_rates)
        discount_factors = np.empty_like(par_rates)

        for i, (c, T) in enumerate(zip(par_rates, maturities)):
            # Sum of discounted coupons for earlier maturities
            coupon_pv = sum(c * discount_factors[j] for j in range(i))

            # 1 = coupon_pv + (1 + c) * DF(T)
            df_T = (1.0 - coupon_pv) / (1.0 + c)
            discount_factors[i] = df_T

            # Convert discount factor to continuously-compounded zero rate
            zero_rates[i] = -np.log(df_T) / T

        return cls(maturities, zero_rates)
