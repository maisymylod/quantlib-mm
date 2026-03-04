"""Time series analysis for financial returns."""

import numpy as np
from scipy import stats


class ReturnsAnalyzer:
    """Analyzer for price series: returns, volatility, and distribution metrics.

    Parameters
    ----------
    prices : array-like
        One-dimensional array of asset prices ordered chronologically.
    """

    def __init__(self, prices):
        self.prices = np.asarray(prices, dtype=float)
        if self.prices.ndim != 1 or len(self.prices) < 2:
            raise ValueError("prices must be a 1-D array with at least two elements")

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def log_returns(self) -> np.ndarray:
        """Compute log returns: ln(S_t / S_{t-1})."""
        return np.log(self.prices[1:] / self.prices[:-1])

    def simple_returns(self) -> np.ndarray:
        """Compute simple returns: (S_t - S_{t-1}) / S_{t-1}."""
        return np.diff(self.prices) / self.prices[:-1]

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    def rolling_volatility(self, window: int = 21) -> np.ndarray:
        """Rolling annualized volatility of log returns.

        Parameters
        ----------
        window : int
            Number of observations in each rolling window.

        Returns
        -------
        numpy.ndarray
            Array of length ``len(log_returns) - window + 1``.
        """
        log_ret = self.log_returns()
        if window > len(log_ret):
            raise ValueError(
                f"window ({window}) exceeds the number of log returns ({len(log_ret)})"
            )
        # Compute rolling standard deviation using stride tricks for efficiency
        shape = (len(log_ret) - window + 1, window)
        strides = (log_ret.strides[0], log_ret.strides[0])
        windows = np.lib.stride_tricks.as_strided(log_ret, shape=shape, strides=strides)
        rolling_std = np.std(windows, axis=1, ddof=1)
        return rolling_std * np.sqrt(252)

    def ewma_volatility(self, span: int = 21) -> np.ndarray:
        """Exponentially weighted moving average volatility (annualized).

        Uses the decay factor ``lambda = 1 - 2/(span+1)`` applied to squared
        log returns, then annualises by multiplying by sqrt(252).

        Parameters
        ----------
        span : int
            Span for the exponential weighting (controls half-life).

        Returns
        -------
        numpy.ndarray
            Array with the same length as log returns.
        """
        log_ret = self.log_returns()
        alpha = 2.0 / (span + 1)
        sq = log_ret ** 2
        ewma_var = np.empty_like(sq)
        ewma_var[0] = sq[0]
        for i in range(1, len(sq)):
            ewma_var[i] = alpha * sq[i] + (1 - alpha) * ewma_var[i - 1]
        return np.sqrt(ewma_var) * np.sqrt(252)

    # ------------------------------------------------------------------
    # Distribution shape
    # ------------------------------------------------------------------

    def skewness(self) -> float:
        """Sample skewness of log returns (Fisher definition)."""
        return float(stats.skew(self.log_returns(), bias=False))

    def kurtosis(self) -> float:
        """Excess kurtosis of log returns (Fisher definition)."""
        return float(stats.kurtosis(self.log_returns(), bias=False))

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def jarque_bera_test(self) -> tuple:
        """Jarque-Bera test for normality of log returns.

        Returns
        -------
        tuple
            ``(statistic, p_value)``
        """
        jb_stat, p_value = stats.jarque_bera(self.log_returns())
        return (float(jb_stat), float(p_value))

    # ------------------------------------------------------------------
    # Autocorrelation
    # ------------------------------------------------------------------

    def autocorrelation(self, lags: int = 10) -> np.ndarray:
        """Sample autocorrelation of log returns for lags 1..lags.

        Parameters
        ----------
        lags : int
            Maximum lag to compute.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(lags,)`` with autocorrelations at lags 1 to *lags*.
        """
        r = self.log_returns()
        n = len(r)
        mean = np.mean(r)
        var = np.var(r, ddof=0)
        acorr = np.empty(lags)
        for k in range(1, lags + 1):
            cov = np.sum((r[k:] - mean) * (r[:-k] - mean)) / n
            acorr[k - 1] = cov / var
        return acorr
