"""quantlib-mm: Quantitative finance library built on pure mathematics."""

from quantlib_mm.binomial_tree import BinomialTree
from quantlib_mm.black_scholes import BlackScholes
from quantlib_mm.correlation import CorrelationAnalyzer
from quantlib_mm.greeks import Greeks
from quantlib_mm.mc_pricing import MonteCarloOptionPricer
from quantlib_mm.monte_carlo import GeometricBrownianMotion, PathStatistics
from quantlib_mm.portfolio import Portfolio
from quantlib_mm.risk import RiskMetrics
from quantlib_mm.time_series import ReturnsAnalyzer
from quantlib_mm.yield_curve import YieldCurve

__version__ = "0.1.0"

__all__ = [
    "BinomialTree",
    "BlackScholes",
    "CorrelationAnalyzer",
    "GeometricBrownianMotion",
    "Greeks",
    "MonteCarloOptionPricer",
    "PathStatistics",
    "Portfolio",
    "ReturnsAnalyzer",
    "RiskMetrics",
    "YieldCurve",
    "__version__",
]
