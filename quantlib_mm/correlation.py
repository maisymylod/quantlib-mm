"""Correlation and covariance matrix tools with PCA and shrinkage.

Provides utilities for computing correlation matrices, covariance matrices,
exponentially weighted moving average (EWMA) covariance, Ledoit-Wolf style
shrinkage, PCA decomposition, and positive-definiteness projection.
"""

import numpy as np
from scipy import linalg


class CorrelationAnalyzer:
    """Analyze correlations and covariances for a panel of asset returns.

    Parameters
    ----------
    returns : array-like, shape (n_observations, n_assets)
        2-D array where each row is an observation (time step) and each
        column is an asset.
    """

    def __init__(self, returns):
        self.returns = np.asarray(returns, dtype=float)
        if self.returns.ndim != 2:
            raise ValueError("returns must be a 2-D array")
        self.n_obs, self.n_assets = self.returns.shape

    # ------------------------------------------------------------------
    # Core matrices
    # ------------------------------------------------------------------

    def correlation_matrix(self):
        """Pearson correlation matrix.

        Returns
        -------
        corr : ndarray, shape (n_assets, n_assets)
        """
        return np.corrcoef(self.returns, rowvar=False)

    def covariance_matrix(self):
        """Sample covariance matrix (unbiased, ddof=1).

        Returns
        -------
        cov : ndarray, shape (n_assets, n_assets)
        """
        return np.cov(self.returns, rowvar=False, ddof=1)

    # ------------------------------------------------------------------
    # EWMA covariance
    # ------------------------------------------------------------------

    def ewma_covariance(self, span=60):
        """Exponentially weighted moving average covariance matrix.

        Parameters
        ----------
        span : int
            Decay span; the decay factor alpha = 2 / (span + 1).

        Returns
        -------
        cov : ndarray, shape (n_assets, n_assets)
        """
        alpha = 2.0 / (span + 1)
        n_obs, n_assets = self.returns.shape
        mean = self.returns.mean(axis=0)
        demeaned = self.returns - mean

        # Build weighted covariance iteratively (most recent obs last)
        cov = np.zeros((n_assets, n_assets))
        for t in range(n_obs):
            cov = (1 - alpha) * cov + alpha * np.outer(demeaned[t], demeaned[t])
        return cov

    # ------------------------------------------------------------------
    # Shrinkage
    # ------------------------------------------------------------------

    def shrink_covariance(self, shrinkage=0.1):
        """Ledoit-Wolf style shrinkage toward the diagonal (scaled identity).

        The shrinkage target is ``mu * I`` where ``mu`` is the average
        variance (trace of S / p).

        Parameters
        ----------
        shrinkage : float
            Shrinkage intensity in [0, 1].  0 = sample covariance,
            1 = diagonal target.

        Returns
        -------
        cov_shrunk : ndarray, shape (n_assets, n_assets)
        """
        S = self.covariance_matrix()
        mu = np.trace(S) / self.n_assets
        target = mu * np.eye(self.n_assets)
        return (1 - shrinkage) * S + shrinkage * target

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    def pca_decomposition(self, n_components=None):
        """PCA on the correlation matrix.

        Parameters
        ----------
        n_components : int or None
            Number of components to retain.  ``None`` keeps all.

        Returns
        -------
        eigenvalues : ndarray, shape (n_components,)
        eigenvectors : ndarray, shape (n_assets, n_components)
            Column ``j`` is the eigenvector for eigenvalue ``j``.
        explained_variance_ratio : ndarray, shape (n_components,)
        """
        corr = self.correlation_matrix()
        # eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = linalg.eigh(corr)

        # Reverse to descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total = eigenvalues.sum()
        explained_variance_ratio = eigenvalues / total

        if n_components is not None:
            eigenvalues = eigenvalues[:n_components]
            eigenvectors = eigenvectors[:, :n_components]
            explained_variance_ratio = explained_variance_ratio[:n_components]

        return eigenvalues, eigenvectors, explained_variance_ratio

    # ------------------------------------------------------------------
    # Positive-definiteness utilities
    # ------------------------------------------------------------------

    def is_positive_definite(self):
        """Check whether the sample covariance matrix is positive definite.

        Returns
        -------
        bool
        """
        cov = self.covariance_matrix()
        try:
            linalg.cholesky(cov, lower=True)
            return True
        except linalg.LinAlgError:
            return False

    def nearest_positive_definite(self):
        """Project the sample covariance to the nearest positive-definite matrix.

        Uses Higham's (2002) alternating-projections algorithm.

        Returns
        -------
        X : ndarray, shape (n_assets, n_assets)
        """
        A = self.covariance_matrix()
        return self._nearest_pd(A)

    @staticmethod
    def _nearest_pd(A, max_iter=100, tol=1e-10):
        """Higham's alternating-projections algorithm for nearest PD matrix."""
        n = A.shape[0]
        Y = A.copy()
        delta_S = np.zeros_like(A)

        for _ in range(max_iter):
            R = Y - delta_S

            # Project onto PSD cone
            eigvals, eigvecs = linalg.eigh(R)
            eigvals = np.maximum(eigvals, 0)
            X = eigvecs @ np.diag(eigvals) @ eigvecs.T

            delta_S = X - R
            Y = X.copy()

            # Check convergence
            if linalg.norm(Y - A, 'fro') < tol:
                break

        # Ensure symmetry
        X = (X + X.T) / 2

        # Nudge eigenvalues to be strictly positive
        eigvals = linalg.eigvalsh(X)
        if eigvals.min() < tol:
            spacing = np.spacing(linalg.norm(X))
            X += np.eye(n) * max(spacing, tol) * (1 + abs(eigvals.min()))

        return X
