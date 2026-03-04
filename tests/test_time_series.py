"""Tests for quantlib_mm.time_series."""

import numpy as np
import pytest
from quantlib_mm.time_series import ReturnsAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gbm_prices(n=1000, mu=0.0, sigma=0.2, s0=100.0, seed=42):
    """Generate a geometric Brownian motion sample path."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    log_ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    prices = s0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))
    return prices


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLogVsSimpleReturns:
    """Log and simple returns should satisfy r_simple = exp(r_log) - 1."""

    def test_relationship(self):
        prices = _make_gbm_prices(500)
        ra = ReturnsAnalyzer(prices)
        log_ret = ra.log_returns()
        simple_ret = ra.simple_returns()
        np.testing.assert_allclose(simple_ret, np.exp(log_ret) - 1, rtol=1e-9)


class TestRollingVolWindowSize:
    """Rolling volatility output length must equal len(returns) - window + 1."""

    def test_output_length(self):
        prices = _make_gbm_prices(300)
        ra = ReturnsAnalyzer(prices)
        for window in [5, 10, 21, 50]:
            vol = ra.rolling_volatility(window=window)
            expected_len = len(ra.log_returns()) - window + 1
            assert len(vol) == expected_len, (
                f"window={window}: expected length {expected_len}, got {len(vol)}"
            )


class TestSkewnessKurtosisNormal:
    """For data drawn from a normal distribution, skewness ~ 0 and excess kurtosis ~ 0."""

    def test_near_zero(self):
        rng = np.random.default_rng(99)
        # Build synthetic prices from perfectly normal log returns
        log_ret = rng.standard_normal(10_000) * 0.01
        prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))
        ra = ReturnsAnalyzer(prices)
        assert abs(ra.skewness()) < 0.1, f"skewness too large: {ra.skewness()}"
        assert abs(ra.kurtosis()) < 0.1, f"kurtosis too large: {ra.kurtosis()}"


class TestJarqueBeraOnNormal:
    """Jarque-Bera should NOT reject normality for Gaussian returns (p > 0.05)."""

    def test_high_p_value(self):
        rng = np.random.default_rng(77)
        log_ret = rng.standard_normal(5000) * 0.01
        prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))
        ra = ReturnsAnalyzer(prices)
        stat, p_value = ra.jarque_bera_test()
        assert p_value > 0.05, f"p_value={p_value} (stat={stat}); normality rejected"


class TestAutocorrelationWhiteNoise:
    """Autocorrelation of white-noise returns should be close to zero at all lags."""

    def test_near_zero(self):
        rng = np.random.default_rng(123)
        log_ret = rng.standard_normal(5000) * 0.01
        prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))
        ra = ReturnsAnalyzer(prices)
        acorr = ra.autocorrelation(lags=10)
        assert acorr.shape == (10,)
        # Each autocorrelation should be small for white noise
        assert np.all(np.abs(acorr) < 0.05), (
            f"autocorrelations not near zero: {acorr}"
        )
