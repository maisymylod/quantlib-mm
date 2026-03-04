"""Tests for quantlib_mm.yield_curve module."""

import numpy as np
import pytest

from quantlib_mm.yield_curve import YieldCurve


class TestInterpolation:
    """Interpolation between known points."""

    def test_interpolate_between_known_points(self):
        maturities = [1.0, 2.0, 5.0, 10.0]
        rates = [0.02, 0.025, 0.03, 0.035]
        curve = YieldCurve(maturities, rates)

        # At exact knots the interpolated rate must match
        assert curve.interpolate(1.0) == pytest.approx(0.02)
        assert curve.interpolate(10.0) == pytest.approx(0.035)

        # Mid-point between 1y (2 %) and 2y (2.5 %) should be 2.25 %
        assert curve.interpolate(1.5) == pytest.approx(0.0225)

        # Between 2y and 5y: linear from 2.5 % to 3 % over 3 years
        # At 3y: 0.025 + (1/3)*(0.03 - 0.025) = 0.02667
        assert curve.interpolate(3.0) == pytest.approx(
            0.025 + (1.0 / 3.0) * 0.005, rel=1e-9
        )


class TestDiscountFactor:
    """Discount factor correctness."""

    def test_discount_factor_matches_formula(self):
        maturities = [1.0, 2.0, 5.0, 10.0]
        rates = [0.03, 0.035, 0.04, 0.045]
        curve = YieldCurve(maturities, rates)

        for t, r in zip(maturities, rates):
            expected_df = np.exp(-r * t)
            assert curve.discount_factor(t) == pytest.approx(expected_df, rel=1e-12)

        # Interpolated point
        t_mid = 3.0
        r_mid = curve.interpolate(t_mid)
        assert curve.discount_factor(t_mid) == pytest.approx(
            np.exp(-r_mid * t_mid), rel=1e-12
        )


class TestForwardRate:
    """Forward rate consistency."""

    def test_forward_rate_consistent_with_discount_factors(self):
        maturities = [1.0, 2.0, 5.0, 10.0]
        rates = [0.02, 0.025, 0.03, 0.035]
        curve = YieldCurve(maturities, rates)

        t1, t2 = 2.0, 5.0
        f = curve.forward_rate(t1, t2)

        # DF(t2) should equal DF(t1) * exp(-f * (t2 - t1))
        df1 = curve.discount_factor(t1)
        df2 = curve.discount_factor(t2)
        assert df2 == pytest.approx(df1 * np.exp(-f * (t2 - t1)), rel=1e-12)

    def test_forward_rate_raises_on_invalid_range(self):
        curve = YieldCurve([1.0, 2.0], [0.03, 0.04])
        with pytest.raises(ValueError):
            curve.forward_rate(5.0, 2.0)


class TestBootstrap:
    """Bootstrap round-trip."""

    def test_bootstrap_round_trip(self):
        # Start with known zero rates and compute par rates, then
        # bootstrap back and verify we recover the original zeros.
        original_zeros = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
        maturities = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Compute discount factors from the original zero rates
        dfs = np.exp(-original_zeros * maturities)

        # Derive par rates: c = (1 - DF(T)) / sum(DF(1..T))
        par_rates = np.empty_like(original_zeros)
        for i in range(len(maturities)):
            annuity = np.sum(dfs[: i + 1])
            par_rates[i] = (1.0 - dfs[i]) / annuity

        # Bootstrap back
        bootstrapped = YieldCurve.bootstrap(par_rates, maturities)

        np.testing.assert_allclose(bootstrapped.rates, original_zeros, atol=1e-12)


class TestShift:
    """Parallel shift."""

    def test_shift_moves_all_rates(self):
        maturities = [1.0, 2.0, 5.0, 10.0]
        rates = [0.02, 0.025, 0.03, 0.035]
        curve = YieldCurve(maturities, rates)

        shifted = curve.shift(50)  # +50 bp

        for t, r in zip(maturities, rates):
            assert shifted.interpolate(t) == pytest.approx(r + 0.005, rel=1e-12)

        # Original curve should be unchanged
        for t, r in zip(maturities, rates):
            assert curve.interpolate(t) == pytest.approx(r, rel=1e-12)
