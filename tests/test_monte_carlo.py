"""Tests for the Monte Carlo simulation engine (GeometricBrownianMotion)."""

import numpy as np
import pytest

from quantlib_mm.monte_carlo import GeometricBrownianMotion, PathStatistics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_gbm():
    """A standard GBM instance with fixed seed for reproducible tests."""
    return GeometricBrownianMotion(
        s0=100.0,
        mu=0.05,
        sigma=0.2,
        T=1.0,
        n_steps=252,
        n_paths=10_000,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Test 1 – Basic simulation shape and initial condition
# ---------------------------------------------------------------------------

class TestBasicSimulation:
    def test_paths_shape(self, default_gbm):
        """simulate() must return (n_paths, n_steps + 1) array."""
        paths = default_gbm.simulate()
        assert paths.shape == (10_000, 253)

    def test_initial_price(self, default_gbm):
        """Every path must start at s0."""
        paths = default_gbm.simulate()
        np.testing.assert_allclose(paths[:, 0], 100.0)

    def test_all_positive(self, default_gbm):
        """GBM paths are strictly positive by construction."""
        paths = default_gbm.simulate()
        assert np.all(paths > 0)


# ---------------------------------------------------------------------------
# Test 2 – Statistical properties of GBM (mean and variance of terminal S)
# ---------------------------------------------------------------------------

class TestGBMStatisticalProperties:
    def test_terminal_mean(self):
        """E[S(T)] = S0 * exp(μT).  With enough paths this should be close."""
        gbm = GeometricBrownianMotion(
            s0=100.0, mu=0.08, sigma=0.3, T=1.0,
            n_steps=252, n_paths=50_000, seed=123,
        )
        paths = gbm.simulate()
        expected_mean = 100.0 * np.exp(0.08 * 1.0)
        sample_mean = np.mean(paths[:, -1])
        # Allow 1.5 % relative tolerance for 50k paths
        np.testing.assert_allclose(sample_mean, expected_mean, rtol=0.015)

    def test_terminal_variance(self):
        """Var[S(T)] = S0^2 exp(2μT)(exp(σ²T) - 1)."""
        gbm = GeometricBrownianMotion(
            s0=100.0, mu=0.05, sigma=0.2, T=1.0,
            n_steps=252, n_paths=100_000, seed=999,
        )
        paths = gbm.simulate()
        expected_var = (100.0 ** 2) * np.exp(2 * 0.05 * 1.0) * (
            np.exp(0.2 ** 2 * 1.0) - 1
        )
        sample_var = np.var(paths[:, -1], ddof=1)
        # Variance estimator is noisier; allow 5 % relative tolerance
        np.testing.assert_allclose(sample_var, expected_var, rtol=0.05)


# ---------------------------------------------------------------------------
# Test 3 – Antithetic variates
# ---------------------------------------------------------------------------

class TestAntitheticVariates:
    def test_antithetic_reduces_variance(self):
        """Antithetic variates should yield a lower variance of the sample
        mean estimator compared to plain MC with the same path count."""
        n_trials = 30
        means_plain = []
        means_anti = []

        for i in range(n_trials):
            gbm_plain = GeometricBrownianMotion(
                s0=100, mu=0.05, sigma=0.2, T=1.0,
                n_steps=50, n_paths=2000, seed=i,
            )
            gbm_anti = GeometricBrownianMotion(
                s0=100, mu=0.05, sigma=0.2, T=1.0,
                n_steps=50, n_paths=2000, antithetic=True, seed=i,
            )
            means_plain.append(np.mean(gbm_plain.simulate()[:, -1]))
            means_anti.append(np.mean(gbm_anti.simulate()[:, -1]))

        var_plain = np.var(means_plain)
        var_anti = np.var(means_anti)
        # Antithetic variance should be meaningfully smaller
        assert var_anti < var_plain, (
            f"Antithetic variance {var_anti:.4f} not less than "
            f"plain variance {var_plain:.4f}"
        )

    def test_antithetic_odd_paths_raises(self):
        """n_paths must be even when antithetic=True."""
        with pytest.raises(ValueError, match="even"):
            GeometricBrownianMotion(
                s0=100, mu=0.05, sigma=0.2, T=1.0,
                n_steps=10, n_paths=101, antithetic=True,
            )

    def test_antithetic_shape(self):
        """Antithetic simulation should still return the correct shape."""
        gbm = GeometricBrownianMotion(
            s0=50, mu=0.1, sigma=0.25, T=0.5,
            n_steps=100, n_paths=500, antithetic=True, seed=7,
        )
        paths = gbm.simulate()
        assert paths.shape == (500, 101)


# ---------------------------------------------------------------------------
# Test 4 – PathStatistics computation
# ---------------------------------------------------------------------------

class TestPathStatistics:
    def test_statistics_fields(self, default_gbm):
        """compute_statistics must return a PathStatistics with all fields."""
        paths = default_gbm.simulate()
        stats = default_gbm.compute_statistics(paths)
        assert isinstance(stats, PathStatistics)
        assert isinstance(stats.mean, float)
        assert isinstance(stats.std, float)
        assert isinstance(stats.min, float)
        assert isinstance(stats.max, float)
        assert isinstance(stats.mean_path, np.ndarray)
        assert stats.mean_path.shape == (253,)
        assert set(stats.percentiles.keys()) == {5, 25, 50, 75, 95}

    def test_percentile_ordering(self, default_gbm):
        """Percentile values must be monotonically non-decreasing."""
        paths = default_gbm.simulate()
        stats = default_gbm.compute_statistics(paths)
        vals = [stats.percentiles[p] for p in sorted(stats.percentiles)]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_custom_percentiles(self, default_gbm):
        """User-supplied percentile ranks should be respected."""
        paths = default_gbm.simulate()
        stats = default_gbm.compute_statistics(paths, percentile_ranks=[10, 90])
        assert set(stats.percentiles.keys()) == {10, 90}

    def test_max_drawdown_nonnegative(self, default_gbm):
        """Mean maximum drawdown must be in [0, 1]."""
        paths = default_gbm.simulate()
        stats = default_gbm.compute_statistics(paths)
        assert 0.0 <= stats.max_drawdown_mean <= 1.0


# ---------------------------------------------------------------------------
# Test 5 – Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    @pytest.mark.parametrize("field,value,match", [
        ("s0", -1, "positive"),
        ("s0", 0, "positive"),
        ("sigma", -0.1, "non-negative"),
        ("T", 0, "positive"),
        ("n_steps", 0, ">= 1"),
        ("n_paths", 0, ">= 1"),
    ])
    def test_invalid_params_raise(self, field, value, match):
        kwargs = dict(s0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=10, n_paths=100)
        kwargs[field] = value
        with pytest.raises(ValueError, match=match):
            GeometricBrownianMotion(**kwargs)
