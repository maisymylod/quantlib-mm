"""Tests for the mean-variance portfolio optimizer."""

import numpy as np
import pytest

from quantlib_mm.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_asset_portfolio():
    """A simple 3-asset universe with known parameters."""
    expected_returns = [0.10, 0.15, 0.20]
    cov_matrix = [
        [0.005, 0.002, 0.001],
        [0.002, 0.010, 0.003],
        [0.001, 0.003, 0.020],
    ]
    return Portfolio(expected_returns, cov_matrix)


@pytest.fixture
def two_asset_portfolio():
    """Edge-case: 2 correlated assets."""
    expected_returns = [0.08, 0.12]
    cov_matrix = [
        [0.004, 0.001],
        [0.001, 0.009],
    ]
    return Portfolio(expected_returns, cov_matrix)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMinVarianceWeightsSumToOne:
    """Min-variance portfolio weights must sum to 1."""

    def test_weights_sum_to_one(self, three_asset_portfolio):
        weights = three_asset_portfolio.min_variance_portfolio()
        assert weights.shape == (3,)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)

    def test_weights_non_negative(self, three_asset_portfolio):
        weights = three_asset_portfolio.min_variance_portfolio()
        assert np.all(weights >= -1e-8)


class TestMaxSharpeBeatMinVariance:
    """Max Sharpe portfolio should have a higher Sharpe ratio than min var."""

    def test_sharpe_ratio_comparison(self, three_asset_portfolio):
        pf = three_asset_portfolio
        min_var_w = pf.min_variance_portfolio()
        max_sh_w = pf.max_sharpe_portfolio(risk_free_rate=0.02)

        _, _, sharpe_min = pf.portfolio_performance(min_var_w)
        _, _, sharpe_max = pf.portfolio_performance(max_sh_w)

        # Max-Sharpe should be at least as good (numerically may be equal
        # in degenerate cases, so use >=).
        assert sharpe_max >= sharpe_min - 1e-6


class TestEfficientFrontierMonotonic:
    """Returns along the efficient frontier should be non-decreasing."""

    def test_returns_non_decreasing(self, three_asset_portfolio):
        risks, returns = three_asset_portfolio.efficient_frontier(n_points=20)
        assert len(risks) == 20
        assert len(returns) == 20
        # Returns should be monotonically non-decreasing
        for i in range(1, len(returns)):
            assert returns[i] >= returns[i - 1] - 1e-8


class TestPortfolioPerformance:
    """portfolio_performance should return correct values."""

    def test_equal_weight(self, three_asset_portfolio):
        pf = three_asset_portfolio
        w = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        ret, vol, sharpe = pf.portfolio_performance(w)

        expected_ret = np.dot(w, pf.expected_returns)
        expected_vol = np.sqrt(w @ pf.cov_matrix @ w)
        expected_sharpe = expected_ret / expected_vol

        assert np.isclose(ret, expected_ret, atol=1e-10)
        assert np.isclose(vol, expected_vol, atol=1e-10)
        assert np.isclose(sharpe, expected_sharpe, atol=1e-10)


class TestTwoAssetEdgeCase:
    """Edge case with only 2 assets."""

    def test_min_variance_two_assets(self, two_asset_portfolio):
        weights = two_asset_portfolio.min_variance_portfolio()
        assert weights.shape == (2,)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
        assert np.all(weights >= -1e-8)

        # With lower variance, asset 0 should get more weight
        assert weights[0] > weights[1]
