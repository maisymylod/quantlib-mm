"""Tests for quantlib_mm.risk — RiskMetrics class."""

import numpy as np
import pytest
from scipy import stats as sp_stats

from quantlib_mm.risk import RiskMetrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def normal_returns():
    """Large sample of normally-distributed returns (seed for reproducibility)."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0005, scale=0.01, size=10_000)


@pytest.fixture()
def negative_drift_returns():
    """Returns with a clear negative drift to ensure losses."""
    rng = np.random.default_rng(99)
    return rng.normal(loc=-0.002, scale=0.015, size=5_000)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVaRIsNegative:
    """VaR at 95 % confidence on a loss-bearing series should be negative."""

    def test_historical_var_negative(self, negative_drift_returns):
        rm = RiskMetrics(negative_drift_returns)
        var = rm.var_historical(confidence=0.95)
        assert var < 0, f"Expected negative VaR, got {var}"

    def test_parametric_var_negative(self, negative_drift_returns):
        rm = RiskMetrics(negative_drift_returns)
        var = rm.var_parametric(confidence=0.95)
        assert var < 0, f"Expected negative parametric VaR, got {var}"


class TestCVarGreaterThanVaR:
    """CVaR (Expected Shortfall) must be >= VaR in magnitude of loss."""

    def test_cvar_exceeds_var_magnitude(self, negative_drift_returns):
        rm = RiskMetrics(negative_drift_returns)
        var = rm.var_historical(confidence=0.95)
        cvar = rm.cvar(confidence=0.95)
        # Both are negative; CVaR should be more negative (worse) than VaR.
        assert cvar <= var, (
            f"CVaR ({cvar}) should be <= VaR ({var}) "
            "(i.e. larger loss in magnitude)"
        )


class TestParametricVsHistoricalOnNormalData:
    """On truly normal data the two VaR estimates should be close."""

    def test_parametric_close_to_historical(self, normal_returns):
        rm = RiskMetrics(normal_returns)
        h_var = rm.var_historical(confidence=0.95)
        p_var = rm.var_parametric(confidence=0.95)
        # With 10 000 normal samples the two should agree within 15 %
        assert abs(h_var - p_var) / abs(h_var) < 0.15, (
            f"Historical VaR={h_var:.6f}, Parametric VaR={p_var:.6f} "
            "differ by more than 15 %"
        )


class TestMaxDrawdownBounds:
    """Max drawdown must lie in [-1, 0]."""

    def test_drawdown_in_range(self, normal_returns):
        rm = RiskMetrics(normal_returns)
        mdd = rm.max_drawdown()
        assert -1.0 <= mdd <= 0.0, f"Max drawdown {mdd} outside [-1, 0]"

    def test_all_positive_returns(self):
        """If every return is positive there is no drawdown."""
        rm = RiskMetrics(np.array([0.01, 0.02, 0.005, 0.03]))
        assert rm.max_drawdown() == 0.0


class TestSortinoRatio:
    """Sortino ratio sanity checks."""

    def test_positive_for_positive_drift(self, normal_returns):
        rm = RiskMetrics(normal_returns)
        sortino = rm.sortino_ratio(risk_free_rate=0.0, periods=252)
        assert sortino > 0, f"Sortino ratio should be positive, got {sortino}"

    def test_no_downside_returns_inf(self):
        """All returns above risk-free ⟹ Sortino = +inf."""
        rm = RiskMetrics(np.array([0.01, 0.02, 0.03]))
        assert rm.sortino_ratio() == np.inf
