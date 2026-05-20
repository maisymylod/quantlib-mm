# quantlib-mm api

FastAPI REST surface that exposes every module of the `quantlib_mm` Python
library to the showcase frontend.

## Local dev

From the repo root (so the package can be installed editable):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8765
```

OpenAPI docs auto-serve at <http://127.0.0.1:8765/docs>.

## Environment

| Variable          | Default                                            | Purpose                            |
| ----------------- | -------------------------------------------------- | ---------------------------------- |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000`      | Comma-separated CORS allow-list    |

## Endpoints

All POST except sample data; all return JSON.

- `POST /api/montecarlo/simulate`     — GBM paths + statistics
- `POST /api/options/black-scholes`   — call/put price + 5 Greeks
- `POST /api/options/binomial`        — CRR tree (european/american)
- `POST /api/options/monte-carlo`     — MC pricing (european, asian)
- `POST /api/portfolio/efficient-frontier` — Markowitz frontier
- `POST /api/risk/metrics`            — VaR, CVaR, drawdown, Sortino, Calmar
- `POST /api/timeseries/analyze`      — log returns, rolling vol, JB, ACF
- `POST /api/correlation/analyze`     — correlation, cov, PCA
- `POST /api/yield-curve/build`       — bootstrap zero rates + forwards
- `GET  /api/sample-data/tickers`     — list of supported sample tickers
- `GET  /api/sample-data/{ticker}`    — synthetic price + return series
- `GET  /healthz`                     — liveness probe
