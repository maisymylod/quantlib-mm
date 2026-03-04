"""Tests for quantlib_mm.greeks — Black-Scholes analytical Greeks."""

import pytest
from quantlib_mm.greeks import Greeks


# Common parameters for a vanilla European option
S = 100.0
K = 100.0
T = 0.5
r = 0.05
sigma = 0.2


class TestCallDelta:
    """Call delta must lie in (0, 1)."""

    @pytest.mark.parametrize("strike", [80, 100, 120])
    def test_call_delta_between_0_and_1(self, strike):
        g = Greeks(S, strike, T, r, sigma, option_type="call")
        d = g.delta()
        assert 0.0 < d < 1.0, f"Call delta {d} not in (0, 1)"


class TestPutDelta:
    """Put delta must lie in (-1, 0)."""

    @pytest.mark.parametrize("strike", [80, 100, 120])
    def test_put_delta_between_neg1_and_0(self, strike):
        g = Greeks(S, strike, T, r, sigma, option_type="put")
        d = g.delta()
        assert -1.0 < d < 0.0, f"Put delta {d} not in (-1, 0)"


class TestGammaPositive:
    """Gamma is always positive for both calls and puts."""

    @pytest.mark.parametrize("opt_type", ["call", "put"])
    def test_gamma_positive(self, opt_type):
        g = Greeks(S, K, T, r, sigma, option_type=opt_type)
        assert g.gamma() > 0.0


class TestATMVegaHighest:
    """ATM vega should be higher than OTM and ITM vega."""

    def test_atm_vega_is_highest(self):
        strikes = [80, 90, 100, 110, 120]
        vegas = {
            k: Greeks(S, k, T, r, sigma, option_type="call").vega()
            for k in strikes
        }
        atm_vega = vegas[100]
        for strike, v in vegas.items():
            if strike != 100:
                assert atm_vega >= v, (
                    f"ATM vega ({atm_vega:.6f}) should be >= "
                    f"vega at K={strike} ({v:.6f})"
                )


class TestPutCallDeltaRelationship:
    """For European options: delta_call - delta_put ≈ exp(0) = 1
    (approximately, adjusting for discounting is not needed for the
    non-dividend case where delta_call - delta_put = 1 exactly via
    N(d1) - (N(d1) - 1) = 1)."""

    def test_delta_call_minus_delta_put_approx_1(self):
        call_g = Greeks(S, K, T, r, sigma, option_type="call")
        put_g = Greeks(S, K, T, r, sigma, option_type="put")
        diff = call_g.delta() - put_g.delta()
        assert abs(diff - 1.0) < 1e-10, (
            f"delta_call - delta_put = {diff}, expected ≈ 1.0"
        )
