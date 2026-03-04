# quantlib-mm

A Python library for quantitative finance — Monte Carlo simulations, option pricing, portfolio optimization, and risk analysis.

Built with a pure mathematics foundation: stochastic calculus, probability theory, and numerical methods.

## Features

- **Monte Carlo Engine** — Geometric Brownian Motion, variance reduction, path generation
- **Option Pricing** — Black-Scholes, binomial trees, Monte Carlo pricing for European & American options
- **Portfolio Optimization** — Mean-variance optimization, efficient frontier, Sharpe ratio maximization
- **Risk Metrics** — Value at Risk (VaR), Conditional VaR, Greeks computation
- **Yield Curves** — Bootstrap construction, interpolation, discount factor calculation
- **Time Series** — Returns analysis, volatility estimation, correlation matrices

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from quantlib_mm.monte_carlo import GeometricBrownianMotion
from quantlib_mm.options import BlackScholes

# Simulate asset price paths
gbm = GeometricBrownianMotion(S0=100, mu=0.08, sigma=0.2)
paths = gbm.simulate(n_paths=10000, n_steps=252, T=1.0)

# Price a European call option
bs = BlackScholes(S=100, K=105, T=1.0, r=0.05, sigma=0.2)
print(f"Call price: ${bs.call_price():.2f}")
print(f"Delta: {bs.delta():.4f}")
```

## Project Structure

```
quantlib_mm/
├── __init__.py
├── monte_carlo.py      # Monte Carlo simulation engine
├── options.py           # Option pricing models
├── portfolio.py         # Portfolio optimization
├── risk.py              # Risk metrics (VaR, CVaR, Greeks)
├── yield_curve.py       # Yield curve construction
├── time_series.py       # Returns & volatility analysis
└── utils.py             # Math utilities
tests/
├── test_monte_carlo.py
├── test_options.py
├── test_portfolio.py
└── ...
```

## License

MIT
