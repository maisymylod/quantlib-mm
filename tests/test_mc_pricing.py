"""Tests for Monte Carlo option pricing engine."""

import numpy as np
import pytest
from scipy.stats import norm

from quantlib_mm.mc_pricing import MonteCarloOptionPricer


# ---------------------------------------------------------------------------
# Black-Scholes closed-form helpers
# ---------------------------------------------------------------------------

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
SEED = 42


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMonteCarloOptionPricer:
    """Suite of Monte Carlo pricing tests."""

    def test_european_call_converges_to_bs(self):
        """European call MC estimate should fall within its own 95% CI
        around the Black-Scholes analytical price."""
        rng = np.random.default_rng(SEED)
        pricer = MonteCarloOptionPricer(S, K, T, r, sigma, n_paths=200_000)
        mc_call = pricer.price_european_call(rng=rng)
        bs = bs_call(S, K, T, r, sigma)

        lo, hi = pricer.confidence_interval(0.95)
        # The BS price should be inside the MC confidence interval
        assert lo <= bs <= hi, (
            f"BS call {bs:.4f} not in 95% CI [{lo:.4f}, {hi:.4f}]"
        )
        # And the MC estimate should be close in absolute terms
        assert abs(mc_call - bs) < 1.0, (
            f"MC call {mc_call:.4f} too far from BS {bs:.4f}"
        )

    def test_put_call_parity(self):
        """Put-call parity: C - P ~ S - K*exp(-rT)."""
        rng_c = np.random.default_rng(SEED)
        rng_p = np.random.default_rng(SEED)
        pricer = MonteCarloOptionPricer(S, K, T, r, sigma, n_paths=200_000)

        mc_call = pricer.price_european_call(rng=rng_c)
        mc_put = pricer.price_european_put(rng=rng_p)

        parity_lhs = mc_call - mc_put
        parity_rhs = S - K * np.exp(-r * T)

        assert abs(parity_lhs - parity_rhs) < 0.5, (
            f"Put-call parity violated: C-P={parity_lhs:.4f}, "
            f"S-Ke^{{-rT}}={parity_rhs:.4f}"
        )

    def test_asian_call_leq_european_call(self):
        """An arithmetic-average Asian call should be worth no more than
        the corresponding European call (by Jensen's inequality / lower
        variance of average)."""
        rng_eu = np.random.default_rng(SEED)
        rng_as = np.random.default_rng(SEED)
        pricer = MonteCarloOptionPricer(S, K, T, r, sigma, n_paths=200_000)

        eu_call = pricer.price_european_call(rng=rng_eu)
        asian_call = pricer.price_asian_call(averaging="arithmetic", rng=rng_as)

        assert asian_call <= eu_call + 0.5, (
            f"Asian call {asian_call:.4f} should be <= European call {eu_call:.4f}"
        )

    def test_barrier_call_leq_vanilla(self):
        """An up-and-out barrier call (barrier above spot) should be worth
        no more than the vanilla European call."""
        rng_eu = np.random.default_rng(SEED)
        rng_bar = np.random.default_rng(SEED)
        barrier = 130.0  # above spot
        pricer = MonteCarloOptionPricer(S, K, T, r, sigma, n_paths=200_000)

        eu_call = pricer.price_european_call(rng=rng_eu)
        bar_call = pricer.price_barrier_call(barrier, barrier_type="up-and-out", rng=rng_bar)

        assert bar_call <= eu_call + 1e-8, (
            f"Barrier call {bar_call:.4f} should be <= vanilla {eu_call:.4f}"
        )

    def test_confidence_interval_contains_true_price(self):
        """The 99% confidence interval of a European call MC estimate should
        contain the Black-Scholes true price."""
        rng = np.random.default_rng(SEED)
        pricer = MonteCarloOptionPricer(S, K, T, r, sigma, n_paths=200_000)
        pricer.price_european_call(rng=rng)

        lo, hi = pricer.confidence_interval(0.99)
        bs = bs_call(S, K, T, r, sigma)

        assert lo <= bs <= hi, (
            f"BS price {bs:.4f} not in 99% CI [{lo:.4f}, {hi:.4f}]"
        )
