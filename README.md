# quantlib-mm

A Python library for quantitative finance, covering Monte Carlo simulations, option pricing, portfolio optimization, and risk analysis.

Built with a pure mathematics foundation: stochastic calculus, probability theory, and numerical methods.

## Features

- **Monte Carlo Engine**: Geometric Brownian Motion, variance reduction, path generation
- **Option Pricing**: Black-Scholes, binomial trees, Monte Carlo pricing for European & American options
- **Portfolio Optimization**: Mean-variance optimization, efficient frontier, Sharpe ratio maximization
- **Risk Metrics**: Value at Risk (VaR), Conditional VaR, Greeks computation
- **Yield Curves**: Bootstrap construction, interpolation, discount factor calculation
- **Time Series**: Returns analysis, volatility estimation, correlation matrices

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from quantlib_mm import GeometricBrownianMotion, BlackScholes, Greeks

# Simulate asset price paths
gbm = GeometricBrownianMotion(
    s0=100, mu=0.08, sigma=0.2, T=1.0, n_steps=252, n_paths=10000, seed=42
)
paths = gbm.simulate()
stats = gbm.compute_statistics(paths)
print(f"Terminal price mean: ${stats.mean:.2f}")

# Price a European call option
bs = BlackScholes(S=100, K=105, T=1.0, r=0.05, sigma=0.2)
print(f"Call price: ${bs.call_price():.2f}")

# Compute Greeks
g = Greeks(S=100, K=105, T=1.0, r=0.05, sigma=0.2, option_type="call")
print(f"Delta: {g.delta():.4f}")
```

## Project Structure

```
quantlib_mm/
├── __init__.py
├── monte_carlo.py      # GBM path simulation
├── black_scholes.py    # European option closed-form pricing
├── binomial_tree.py    # American/European binomial-tree pricing
├── mc_pricing.py       # Monte Carlo option pricing (European, Asian, barrier)
├── greeks.py           # Analytical Black-Scholes Greeks
├── portfolio.py        # Mean-variance optimization, efficient frontier
├── risk.py             # VaR, CVaR, drawdown, Sortino, Calmar
├── yield_curve.py      # Bootstrap, interpolation, forward rates
├── time_series.py      # Returns analysis, volatility, ACF, JB test
├── correlation.py      # Correlation, covariance, PCA, shrinkage
└── utils.py            # Math utilities
tests/
├── test_monte_carlo.py
├── test_black_scholes.py
├── test_binomial_tree.py
├── test_mc_pricing.py
├── test_greeks.py
├── test_portfolio.py
├── test_risk.py
├── test_yield_curve.py
├── test_time_series.py
└── test_correlation.py
```

## License

MIT
