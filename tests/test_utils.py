"""Tests for quantlib_mm.utils."""

import numpy as np
import pytest
from scipy.stats import norm

from quantlib_mm.utils import (
    annualize_returns,
    annualize_volatility,
    standard_normal_cdf,
    standard_normal_pdf,
)


class TestStandardNormalCDF:
    def test_zero(self):
        assert standard_normal_cdf(0.0) == pytest.approx(0.5, abs=1e-12)

    def test_symmetry(self):
        for x in (-2.5, -1.0, -0.3, 0.7, 1.96, 3.0):
            assert standard_normal_cdf(x) + standard_normal_cdf(-x) == pytest.approx(
                1.0, abs=1e-12
            )

    def test_matches_scipy(self):
        xs = np.linspace(-4.0, 4.0, 9)
        for x in xs:
            assert standard_normal_cdf(x) == pytest.approx(norm.cdf(x), abs=1e-12)


class TestStandardNormalPDF:
    def test_zero_is_peak(self):
        assert standard_normal_pdf(0.0) == pytest.approx(1.0 / np.sqrt(2 * np.pi), abs=1e-12)

    def test_symmetry(self):
        for x in (-2.0, -0.5, 0.5, 2.0, 3.5):
            assert standard_normal_pdf(x) == pytest.approx(
                standard_normal_pdf(-x), abs=1e-12
            )

    def test_matches_scipy(self):
        xs = np.linspace(-4.0, 4.0, 9)
        for x in xs:
            assert standard_normal_pdf(x) == pytest.approx(norm.pdf(x), abs=1e-12)


class TestAnnualizeReturns:
    def test_zero_returns(self):
        returns = np.zeros(252)
        assert annualize_returns(returns) == pytest.approx(0.0, abs=1e-12)

    def test_constant_daily_return(self):
        daily = 0.0004  # ~10% annualised
        returns = np.full(252, daily)
        expected = (1 + daily) ** 252 - 1
        assert annualize_returns(returns) == pytest.approx(expected, rel=1e-10)

    def test_custom_periods(self):
        monthly = np.full(12, 0.01)
        expected = (1.01) ** 12 - 1
        assert annualize_returns(monthly, periods_per_year=12) == pytest.approx(
            expected, rel=1e-10
        )


class TestAnnualizeVolatility:
    def test_zero_returns(self):
        returns = np.zeros(252)
        assert annualize_volatility(returns) == pytest.approx(0.0, abs=1e-12)

    def test_known_std(self):
        rng = np.random.default_rng(0)
        daily = rng.normal(loc=0.0, scale=0.01, size=10_000)
        expected = np.std(daily, ddof=1) * np.sqrt(252)
        assert annualize_volatility(daily) == pytest.approx(expected, rel=1e-10)

    def test_custom_periods(self):
        weekly = np.array([0.01, -0.02, 0.015, -0.005, 0.03, -0.01, 0.02, 0.0])
        expected = np.std(weekly, ddof=1) * np.sqrt(52)
        assert annualize_volatility(weekly, periods_per_year=52) == pytest.approx(
            expected, rel=1e-12
        )
