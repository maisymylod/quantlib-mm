"""Tests for quantlib_mm.correlation module."""

import numpy as np
import pytest

from quantlib_mm.correlation import CorrelationAnalyzer


@pytest.fixture
def analyzer():
    """Create a CorrelationAnalyzer with reproducible random returns."""
    rng = np.random.default_rng(42)
    returns = rng.standard_normal((500, 4))
    return CorrelationAnalyzer(returns)


def test_correlation_diagonal_is_ones(analyzer):
    """Diagonal of the Pearson correlation matrix must be 1."""
    corr = analyzer.correlation_matrix()
    np.testing.assert_allclose(np.diag(corr), np.ones(analyzer.n_assets), atol=1e-12)


def test_covariance_symmetry(analyzer):
    """Sample covariance matrix must be symmetric."""
    cov = analyzer.covariance_matrix()
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


def test_pca_explained_variance_sums_to_one(analyzer):
    """Explained-variance ratios from PCA must sum to 1."""
    _, _, evr = analyzer.pca_decomposition()
    np.testing.assert_allclose(evr.sum(), 1.0, atol=1e-10)


def test_shrinkage_moves_toward_identity(analyzer):
    """Increasing shrinkage should bring the covariance closer to a
    scaled identity matrix."""
    cov_low = analyzer.shrink_covariance(shrinkage=0.1)
    cov_high = analyzer.shrink_covariance(shrinkage=0.9)

    mu = np.trace(analyzer.covariance_matrix()) / analyzer.n_assets
    target = mu * np.eye(analyzer.n_assets)

    dist_low = np.linalg.norm(cov_low - target, "fro")
    dist_high = np.linalg.norm(cov_high - target, "fro")

    assert dist_high < dist_low, (
        "Higher shrinkage should produce a matrix closer to the diagonal target"
    )


def test_positive_definite_check(analyzer):
    """A well-conditioned random covariance should be positive definite."""
    assert analyzer.is_positive_definite() is True
