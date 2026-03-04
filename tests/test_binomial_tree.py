"""
Tests for the CRR binomial tree option pricer.
"""

import numpy as np
import pytest
from scipy.stats import norm

from quantlib_mm.binomial_tree import BinomialTree


# ---------------------------------------------------------------------------
# Helper: Black-Scholes closed-form for European options
# ---------------------------------------------------------------------------

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Closed-form Black-Scholes price for European vanilla options."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# 1. European call convergence to Black-Scholes
# ---------------------------------------------------------------------------

class TestEuropeanCallConvergence:
    """The CRR tree price should converge to Black-Scholes as n_steps grows."""

    def test_convergence(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        bs_price = black_scholes(S, K, T, r, sigma, "call")

        # With a large number of steps the error should be very small
        tree = BinomialTree(S, K, T, r, sigma, n_steps=2000, option_type="call", style="european")
        tree_price = tree.price()

        assert tree_price == pytest.approx(bs_price, abs=0.05), (
            f"CRR tree price {tree_price:.6f} did not converge to BS price {bs_price:.6f}"
        )

    def test_convergence_put(self):
        S, K, T, r, sigma = 110.0, 100.0, 0.5, 0.03, 0.25
        bs_price = black_scholes(S, K, T, r, sigma, "put")

        tree = BinomialTree(S, K, T, r, sigma, n_steps=2000, option_type="put", style="european")
        tree_price = tree.price()

        assert tree_price == pytest.approx(bs_price, abs=0.05)


# ---------------------------------------------------------------------------
# 2. American put >= European put
# ---------------------------------------------------------------------------

class TestAmericanVsEuropean:
    """An American option should always be worth at least as much as its European counterpart."""

    @pytest.mark.parametrize(
        "S, K, T, r, sigma",
        [
            (100, 100, 1.0, 0.05, 0.2),
            (80, 100, 0.5, 0.08, 0.3),
            (120, 100, 2.0, 0.02, 0.15),
        ],
    )
    def test_american_put_geq_european_put(self, S, K, T, r, sigma):
        n = 500
        eu = BinomialTree(S, K, T, r, sigma, n, option_type="put", style="european").price()
        am = BinomialTree(S, K, T, r, sigma, n, option_type="put", style="american").price()
        assert am >= eu - 1e-12, f"American put {am:.6f} < European put {eu:.6f}"

    @pytest.mark.parametrize(
        "S, K, T, r, sigma",
        [
            (100, 100, 1.0, 0.05, 0.2),
            (90, 100, 0.5, 0.08, 0.3),
        ],
    )
    def test_american_call_geq_european_call(self, S, K, T, r, sigma):
        n = 500
        eu = BinomialTree(S, K, T, r, sigma, n, option_type="call", style="european").price()
        am = BinomialTree(S, K, T, r, sigma, n, option_type="call", style="american").price()
        assert am >= eu - 1e-12


# ---------------------------------------------------------------------------
# 3. Edge cases / input validation
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Verify that invalid inputs raise appropriate errors."""

    def test_negative_spot(self):
        with pytest.raises(ValueError, match="Spot price"):
            BinomialTree(-1, 100, 1, 0.05, 0.2, 100)

    def test_zero_strike(self):
        with pytest.raises(ValueError, match="Strike price"):
            BinomialTree(100, 0, 1, 0.05, 0.2, 100)

    def test_negative_time(self):
        with pytest.raises(ValueError, match="Time to expiration"):
            BinomialTree(100, 100, -1, 0.05, 0.2, 100)

    def test_negative_vol(self):
        with pytest.raises(ValueError, match="Volatility"):
            BinomialTree(100, 100, 1, 0.05, -0.2, 100)

    def test_zero_steps(self):
        with pytest.raises(ValueError, match="n_steps"):
            BinomialTree(100, 100, 1, 0.05, 0.2, 0)

    def test_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            BinomialTree(100, 100, 1, 0.05, 0.2, 100, option_type="straddle")

    def test_invalid_style(self):
        with pytest.raises(ValueError, match="style"):
            BinomialTree(100, 100, 1, 0.05, 0.2, 100, style="bermudan")

    def test_single_step(self):
        """A 1-step tree should still return a valid price."""
        bt = BinomialTree(100, 100, 1, 0.05, 0.2, 1)
        p = bt.price()
        assert np.isfinite(p) and p >= 0


# ---------------------------------------------------------------------------
# 4. Tree shape validation
# ---------------------------------------------------------------------------

class TestTreeShape:
    """The price tree returned by get_tree() must have the correct structure."""

    def test_tree_length(self):
        n = 10
        bt = BinomialTree(100, 100, 1, 0.05, 0.2, n)
        tree = bt.get_tree()
        assert len(tree) == n + 1

    def test_layer_sizes(self):
        n = 15
        bt = BinomialTree(100, 100, 1, 0.05, 0.2, n)
        tree = bt.get_tree()
        for i, layer in enumerate(tree):
            assert len(layer) == i + 1, f"Layer {i} has size {len(layer)}, expected {i + 1}"

    def test_root_equals_price(self):
        bt = BinomialTree(100, 100, 1, 0.05, 0.2, 50)
        price = bt.price()
        tree = bt.get_tree()
        assert tree[0][0] == pytest.approx(price)

    def test_terminal_layer_is_payoff(self):
        """Terminal layer should equal the intrinsic payoff."""
        S, K, T, r, sigma, n = 100, 105, 1.0, 0.05, 0.2, 20
        bt = BinomialTree(S, K, T, r, sigma, n, option_type="put", style="european")
        tree = bt.get_tree()
        terminal = tree[n]
        spots = S * bt.u ** (n - 2 * np.arange(n + 1, dtype=float))
        expected = np.maximum(K - spots, 0.0)
        np.testing.assert_allclose(terminal, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Put-call parity for European options
# ---------------------------------------------------------------------------

class TestPutCallParity:
    """
    For European options, put-call parity must hold:
        C - P = S - K * exp(-rT)
    """

    @pytest.mark.parametrize(
        "S, K, T, r, sigma",
        [
            (100, 100, 1.0, 0.05, 0.2),
            (90, 110, 0.25, 0.08, 0.35),
            (150, 100, 2.0, 0.01, 0.10),
        ],
    )
    def test_put_call_parity(self, S, K, T, r, sigma):
        n = 1000
        call = BinomialTree(S, K, T, r, sigma, n, option_type="call", style="european").price()
        put = BinomialTree(S, K, T, r, sigma, n, option_type="put", style="european").price()

        lhs = call - put
        rhs = S - K * np.exp(-r * T)

        assert lhs == pytest.approx(rhs, abs=0.05), (
            f"Put-call parity violated: C-P={lhs:.6f}, S-Ke^{{-rT}}={rhs:.6f}"
        )
